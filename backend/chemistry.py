"""Live compound / assay retrieval. No inferred binding is labelled measured."""
import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import httpx
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

CACHE = Path(__file__).parent / "data" / "cache"
CACHE_VERSION = 4


def aggregate_similar_targets(similar_molecules, activity_payloads):
    """Aggregate activity assays of similar compounds per target (inference, not measurement).

    similar_molecules: [{molecule_chembl_id, similarity}]
    activity_payloads: {molecule_chembl_id: [activity rows]}
    Returns a per-target evidence summary preserving similarity and supporting-compound count.
    """
    by_target = {}
    for mol in similar_molecules:
        sim = float(mol.get("similarity") or 0)
        for act in activity_payloads.get(mol.get("molecule_chembl_id"), []):
            name = act.get("target_pref_name") or act.get("target_name")
            if not name:
                continue
            t = by_target.setdefault(name, {"target_pref_name": name, "n_molecules": 0,
                                            "max_similarity": 0., "molecules": set(),
                                            "best_value": None, "best_type": None, "best_units": None})
            t["molecules"].add(mol["molecule_chembl_id"])
            t["max_similarity"] = max(t["max_similarity"], sim)
            value = act.get("standard_value")
            units = act.get("standard_units")
            if value is not None and (t["best_value"] is None or value < t["best_value"]):
                t["best_value"], t["best_type"], t["best_units"] = value, act.get("standard_type"), units
    out = []
    for t in by_target.values():
        t["n_molecules"] = len(t["molecules"])
        t["molecules"] = sorted(t["molecules"])
        out.append(t)
    return sorted(out, key=lambda t: (-t["max_similarity"], t["best_value"] or 1e12))


async def predict_targets_by_similarity(client, smiles):
    """Infer potential binding targets via ChEMBL structural similarity (>=70% Tanimoto).

    The result is assay evidence from similar compounds, not a measurement of this compound.
    """
    cb = "https://www.ebi.ac.uk/chembl/api/data"
    payload = await fetch_json(client, cb + "/similarity/" + smiles + "/70", {"limit": 15})
    molecules = payload.get("molecules", [])
    if not molecules:
        return []
    fetches = [fetch_json(client, cb + "/activity.json",
                          {"molecule_chembl_id": m["molecule_chembl_id"],
                           "standard_type__in": "Kd,Ki,IC50,EC50", "limit": 10})
               for m in molecules]
    results = await asyncio.gather(*fetches, return_exceptions=True)
    activity_payloads = {}
    for m, r in zip(molecules, results):
        if isinstance(r, dict):
            activity_payloads[m["molecule_chembl_id"]] = r.get("activities", [])
    return aggregate_similar_targets(molecules, activity_payloads)


def exact_binding_records(payload, inchikey):
    """A similarity hit is not identity. Retain full InChIKey matches only.

    This endpoint omits assay units/conditions; retain its raw values without
    guessing nM or using them as an in-vivo occupancy constant.
    """
    rows = payload.get("getLindsByUniprotResponse", {}).get("bdb.affinities", [])
    if isinstance(rows, dict): rows = [rows]
    records = []
    for row in rows:
        mol = Chem.MolFromSmiles(row.get("bdb.smiles", ""))
        if mol is None or Chem.MolToInchiKey(mol) != inchikey: continue
        affinity = str(row.get("bdb.affinity", "")).strip()
        match = re.fullmatch(r"([<>=~]*)\s*([\d.eE+\-]+)", affinity)
        records.append({"source":"BindingDB", "monomer_id":row.get("bdb.monomerid"),
                        "standard_type":row.get("bdb.affinity_type"),
                        "standard_relation":match[1] or "=" if match else "",
                        "standard_value":match[2] if match else affinity,
                        "standard_units":"(units not provided by API)",
                        "target_pref_name":row.get("bdb.target"),
                        "target_organism":row.get("bdb.species"),
                        "identity_match":"full_inchikey", "raw_record":row})
    return records


async def opentargets_mechanisms(client, name):
    """Query a drug's mechanism of action and targets from the Open Targets GraphQL API.

    Supplements peptides/biologics lacking ChEMBL mechanism records. Results
    follow the mechanism-record shape and are tagged with their source.
    """
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    safe = name.replace('"', '')
    search = ('{ search(queryString:"%s",entityNames:["drug"],'
              'page:{index:0,size:3}){hits{id name entity}}}' % safe)
    resp = await client.post(url, json={"query": search})
    resp.raise_for_status()
    hits = (resp.json().get("data") or {}).get("search", {}).get("hits", [])
    mechanisms = []
    for hit in hits:
        detail = ('{ drug(chemblId:"%s"){ mechanismsOfAction{ rows{'
                  ' mechanismOfAction actionType targets{ approvedName'
                  ' approvedSymbol } } } } }' % hit["id"])
        r = await client.post(url, json={"query": detail})
        r.raise_for_status()
        drug = (r.json().get("data") or {}).get("drug") or {}
        for row in (drug.get("mechanismsOfAction") or {}).get("rows", []):
            mechanisms.append({
                "mechanism_of_action": row.get("mechanismOfAction"),
                "action_type": row.get("actionType"),
                "target_pref_name": "; ".join(
                    t.get("approvedName") for t in row.get("targets", []) if t.get("approvedName")),
                "direct_interaction": True,
                "source": "opentargets",
                "drug_id": hit["id"]})
        if mechanisms:
            break
    return mechanisms


async def fetch_json(client, url, params=None):
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def resolve_compound(name: str):
    cache_path = CACHE / (hashlib.sha256(name.casefold().encode()).hexdigest()+".json")
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            age = (datetime.now(timezone.utc)-datetime.fromisoformat(cached["retrieved_at"])).total_seconds()
            if cached.get("cache_version") == CACHE_VERSION and age < (3600 if cached.get("warnings") else 604800):
                cached["cache_hit"] = True
                return cached
        except (ValueError, KeyError): pass
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    result = {"query": name, "status": "unresolved", "cache_version":CACHE_VERSION, "mechanisms": [], "activities": [], "sources": [], "warnings": []}
    async with httpx.AsyncClient(timeout=18, follow_redirects=False) as client:
        try:
            # Encode a single name path segment, including slashes and query syntax.
            from urllib.parse import quote
            encoded = quote(name, safe="")
            payload = await fetch_json(client, f"{base}/compound/name/{encoded}/property/MolecularFormula,MolecularWeight,IsomericSMILES,InChIKey/JSON")
            props = payload["PropertyTable"]["Properties"]
            if not props:
                result["warnings"].append("Substance not found in PubChem. Please specify the exact active ingredient.")
                return result
            if len(props) != 1:
                # Peptides/biologics often have several records of identical composition.
                # If all compositions match, the first CID is used; otherwise it stays ambiguous.
                formulas = {p.get("MolecularFormula") for p in props}
                if len(formulas) == 1:
                    result["warnings"].append(f"PubChem returned {len(props)} records of identical composition; the first was used.")
                else:
                    result["warnings"].append("Multiple substance candidates exist. Please specify the exact active ingredient.")
                    return result
            p = props[0]; smiles = p.get("SMILES") or p.get("IsomericSMILES")
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                result["warnings"].append("Could not validate the molecular structure with RDKit.")
                return result
            if Chem.MolToInchiKey(mol) != p["InChIKey"]:
                result["warnings"].append("Molecular identification held: full InChIKey does not match the PubChem structure.")
                return result
            drawer = rdMolDraw2D.MolDraw2DSVG(320, 180)
            drawer.drawOptions().clearBackground = False
            drawer.DrawMolecule(mol); drawer.FinishDrawing()
            result.update(status="resolved", cid=p["CID"], smiles=Chem.MolToSmiles(mol),
                          formula=p["MolecularFormula"], molecular_weight=float(p["MolecularWeight"]),
                          inchikey=p["InChIKey"], rdkit_molecular_weight=Descriptors.MolWt(mol),
                          logp=Descriptors.MolLogP(mol), tpsa=rdMolDescriptors.CalcTPSA(mol),
                          structure_svg=drawer.GetDrawingText(),
                          retrieved_at=datetime.now(timezone.utc).isoformat())
            result["sources"].append({"id":f"pubchem:{p['CID']}", "title":"PubChem and molecular identification", "url":f"https://pubchem.ncbi.nlm.nih.gov/compound/{p['CID']}"})
        except (httpx.HTTPError, KeyError, ValueError):
            result["warnings"].append("PubChem lookup failed. Substance information was not fabricated.")
            return result
        try:
            cb = "https://www.ebi.ac.uk/chembl/api/data"
            found = await fetch_json(client, cb+"/molecule.json", {"molecule_structures__standard_inchi_key":result["inchikey"], "limit":3})
            molecules = found.get("molecules", [])
            if molecules:
                cid = molecules[0]["molecule_chembl_id"]; result["chembl_id"] = cid
                result["preferred_name"] = molecules[0].get("pref_name") or name
                a, b = await asyncio.gather(
                    fetch_json(client, cb+"/mechanism.json", {"molecule_chembl_id":cid,"limit":12}),
                    fetch_json(client, cb+"/activity.json", {"molecule_chembl_id":cid,"standard_type__in":"Kd,Ki,IC50,EC50","limit":20}),
                    return_exceptions=True,
                )
                if isinstance(a, dict):
                    result["mechanisms"] = [{k: m.get(k) for k in ("mechanism_of_action","action_type","target_chembl_id","direct_interaction","mechanism_refs")} for m in a.get("mechanisms", [])]
                else: result["warnings"].append("ChEMBL mechanism-of-action lookup did not complete.")
                if isinstance(b, dict):
                    result["activities"] = [{k: m.get(k) for k in ("activity_id","standard_type","standard_relation","standard_value","standard_units","target_pref_name","target_organism","target_chembl_id","assay_chembl_id","assay_type","document_chembl_id","data_validity_comment")} for m in b.get("activities", [])]
                    result["activity_total"] = b.get("page_meta", {}).get("total_count", len(result["activities"]))
                else: result["warnings"].append("ChEMBL activity-assay lookup did not complete.")
                result["sources"].append({"id":f"chembl:{cid}","title":"ChEMBL and mechanism of action / activity assays","url":f"https://www.ebi.ac.uk/chembl/explore/compound/{cid}"})
        except (httpx.HTTPError, KeyError, ValueError):
            result["warnings"].append("ChEMBL lookup failed; this does not mean there are no targets.")
        if not result["activities"]:
            try:
                url = "https://bindingdb.org/rest/getTargetByCompound"
                params = {"smiles":result["smiles"],"cutoff":1.0,"response":"application/json"}
                payload = await fetch_json(client, url, params)
                result["activities"] = exact_binding_records(payload, result["inchikey"])[:20]
                if result["activities"]:
                    result["sources"].append({"id":"bindingdb:"+result["inchikey"],
                        "title":"BindingDB - target assays of identical structure", "url":str(httpx.URL(url,params=params))})
                    result["warnings"].append("BindingDB shows only records whose full InChIKey matches. Values from this API lacking units/assay conditions are not used in occupancy calculations.")
            except (httpx.HTTPError, KeyError, ValueError, TypeError):
                result["warnings"].append("The BindingDB fallback lookup did not complete.")
        if not result["mechanisms"]:
            try:
                # For drugs without ChEMBL mechanism records (peptides, biologics),
                # curated mechanisms from Open Targets supplement the lookup.
                ot = await opentargets_mechanisms(client, name)
                if ot:
                    result["mechanisms"] = ot
                    result["sources"].append({"id": "opentargets:" + (ot[0].get("drug_id") or name),
                                              "title": "Open Targets - mechanism of action",
                                              "url": "https://platform.opentargets.org/"})
            except (httpx.HTTPError, KeyError, ValueError, TypeError):
                pass
        try:
            # Infer potential targets from SMILES similarity. This is assay evidence
            # from similar compounds, not this compound, wired at a lower confidence tier.
            predicted = await predict_targets_by_similarity(client, result["smiles"])
            if predicted:
                result["predicted_targets"] = predicted[:20]
                result["sources"].append({"id": f"chembl_similarity:{result['inchikey']}",
                                          "title": "ChEMBL - target inference from structurally similar compounds",
                                          "url": "https://www.ebi.ac.uk/chembl/"})
                result["warnings"].append("Structure-similarity target inference is an estimate, not a direct measurement of this compound.")
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            pass
        try:
            # Ligand-based virtual screening: compares against panels of known active
            # ligands via 3D pharmacophore shape plus fingerprint similarity.
            from .vscreen import virtual_screen
            vs_hits = await virtual_screen(client, fetch_json, result["smiles"])
            if vs_hits:
                result["vs_targets"] = vs_hits
                result["sources"].append({"id": f"vscreen:{result['inchikey']}",
                                          "title": "Virtual screening - active-ligand panel comparison",
                                          "url": "https://www.ebi.ac.uk/chembl/"})
                result["warnings"].append("Virtual-screening targets are computational inference, not docking or direct experiment.")
        except Exception:
            pass
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
