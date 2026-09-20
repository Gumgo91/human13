"""Ligand-based virtual screening against a curated human target panel.

For each panel target we fetch known actives (pChEMBL ≥ 6) from ChEMBL once,
generate one 3D conformer per ligand, and persist the USRCAT pharmacophore-shape
descriptors plus Morgan fingerprints under data/cache/vscreen. A query molecule
is then embedded once and scored against every cached descriptor:

    score = 0.5 * USRCAT similarity + 0.5 * ECFP4 Tanimoto   (per ligand)
    target score = max over the target's ligand ensemble

This is ligand-based virtual screening: it infers plausible binding targets from
3D pharmacophore shape and substructure similarity to known actives. It is
inference, not measurement, and is wired downstream at a lower confidence than
direct experimental records. No receptor structure or docking is claimed.
"""
import asyncio
import hashlib
import json
import math
from pathlib import Path

CACHE = Path(__file__).parent / "data" / "cache" / "vscreen"

# Human SINGLE-PROTEIN targets that the anchor map can act on. Names are matched
# against ChEMBL pref_name; the first human hit is used to fetch actives.
PANEL_TARGETS = [
    "Cyclooxygenase-1", "Cyclooxygenase-2", "Beta-1 adrenergic receptor",
    "Beta-2 adrenergic receptor", "Alpha-1a adrenergic receptor",
    "Dopamine D2 receptor", "Dopamine transporter", "Serotonin transporter",
    "Norepinephrine transporter", "Serotonin 2a (5-HT2a) receptor",
    "Serotonin 1a (5-HT1a) receptor", "Mu opioid receptor",
    "Gamma-aminobutyric acid receptor alpha-1", "Glutamate receptor ionotropic, NMDA",
    "HMG-CoA reductase", "Angiotensin-converting enzyme", "Renin",
    "Angiotensin II receptor type 1", "Endothelin receptor ET-A",
    "Voltage-gated potassium channel subunit Kv11.1", "Thrombin",
    "Coagulation factor X", "Platelet glycoprotein IIb/IIIa",
    "Dipeptidyl peptidase IV", "Sodium/glucose cotransporter 2",
    "Glucagon-like peptide 1 receptor", "Insulin receptor",
    "Androgen receptor", "Estrogen receptor alpha", "Glucocorticoid receptor",
    "Mineralocorticoid receptor", "Progesterone receptor",
    "Vasopressin V2 receptor", "Oxytocin receptor",
    "Histamine H1 receptor", "Histamine H2 receptor",
    "Adenosine receptor A2a", "Adenosine receptor A1",
    "Muscarinic acetylcholine receptor M1", "Muscarinic acetylcholine receptor M2",
    "Nicotinic acetylcholine receptor alpha4/beta2",
    "Phosphodiesterase 5A", "Phosphodiesterase 4B",
    "Monoamine oxidase A", "Monoamine oxidase B", "Catechol O-methyltransferase",
    "Acetylcholinesterase", "Carbonic anhydrase II", "Xanthine dehydrogenase",
    "5-lipoxygenase", "Tyrosine-protein kinase JAK1",
    "Tyrosine-protein kinase JAK2", "Tyrosine-protein kinase BTK",
    "Epidermal growth factor receptor erbB1", "Vascular endothelial growth factor receptor 2",
    "Serine/threonine-protein kinase mTOR", "Aromatase",
    "Steroid 5-alpha-reductase 1", "Vitamin D receptor",
    "Peroxisome proliferator-activated receptor gamma", "Sphingosine 1-phosphate receptor Edg-1",
    "C-C chemokine receptor type 5", "Cannabinoid receptor 1",
    "Transient receptor potential cation channel subfamily V member 1",
    "Voltage-gated sodium channel", "Prostaglandin E synthase",
    "Influenza A neuraminidase", "HIV-1 protease",
]


def _panel_path(target_name):
    key = hashlib.sha256(target_name.casefold().encode()).hexdigest()[:16]
    return CACHE / f"{key}.json"


def _descriptors(smiles):
    """One MMFF conformer -> USRCAT distribution + ECFP4 on-bits, or None."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    try:
        usrcat = list(rdMolDescriptors.GetUSRCAT(mol))
    except Exception:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return {"usrcat": usrcat, "fp": sorted(fp.GetOnBits()), "smiles": Chem.MolToSmiles(mol)}


def _score_query(query_desc, panel_desc):
    """0-1 combined score: USRCAT pharmacophore shape + ECFP4 Tanimoto.

    USR similarity is the standard 1/(1+Manhattan) score — identical to
    RDKit's GetUSRScore on conformer descriptors.
    """
    from rdkit import DataStructs
    d = sum(abs(a-b) for a, b in zip(query_desc["usrcat"], panel_desc["usrcat"]))
    shape = 1. / (1. + d)
    qfp = DataStructs.ExplicitBitVect(2048); pfp = DataStructs.ExplicitBitVect(2048)
    for b in query_desc["fp"]: qfp.SetBit(b)
    for b in panel_desc["fp"]: pfp.SetBit(b)
    tanimoto = DataStructs.FingerprintSimilarity(qfp, pfp)
    return 0.5 * shape + 0.5 * tanimoto


async def _fetch_panel(client, fetch_json, target_name, limit=20):
    """Build (or load) the descriptor panel for one ChEMBL target name."""
    path = _panel_path(target_name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    cb = "https://www.ebi.ac.uk/chembl/api/data"
    found = await fetch_json(client, cb + "/target.json",
                             {"pref_name__icontains": target_name,
                              "target_type": "SINGLE PROTEIN",
                              "organism": "Homo sapiens", "limit": 3})
    targets = found.get("targets", [])
    if not targets:
        return None
    tid = targets[0]["target_chembl_id"]
    acts = await fetch_json(client, cb + "/activity.json",
                            {"target_chembl_id": tid, "pchembl_value__gte": 6,
                             "standard_type__in": "Ki,Kd,IC50,EC50", "limit": limit * 3})
    smiles, seen = [], set()
    for a in acts.get("activities", []):
        s = (a.get("molecule_structures") or {}).get("canonical_smiles") or a.get("canonical_smiles")
        if s and s not in seen:
            seen.add(s); smiles.append(s)
        if len(smiles) >= limit:
            break
    # Conformer generation is CPU-bound; run off the event loop.
    descs = await asyncio.gather(*[asyncio.to_thread(_descriptors, s) for s in smiles])
    descs = [d for d in descs if d]
    if not descs:
        return None
    panel = {"target": target_name, "chembl_id": tid, "ligands": descs}
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = _panel_path(target_name).with_suffix(".tmp")
    tmp.write_text(json.dumps(panel), encoding="utf-8")
    tmp.replace(path)
    return panel


async def virtual_screen(client, fetch_json, smiles, top=12, min_score=0.45):
    """Screen one molecule against the panel; return ranked target hits."""
    qdesc = await asyncio.to_thread(_descriptors, smiles)
    if qdesc is None:
        return []
    panels = await asyncio.gather(
        *[_fetch_panel(client, fetch_json, name) for name in PANEL_TARGETS],
        return_exceptions=True)
    hits = []
    for name, panel in zip(PANEL_TARGETS, panels):
        if not isinstance(panel, dict):
            continue
        scores = await asyncio.gather(
            *[asyncio.to_thread(_score_query, qdesc, lig) for lig in panel["ligands"]],
            return_exceptions=True)
        scores = [s for s in scores if isinstance(s, float)]
        if not scores:
            continue
        best = max(scores)
        if best >= min_score:
            hits.append({"target_pref_name": panel["target"],
                         "chembl_target_id": panel["chembl_id"],
                         "screen_score": round(best, 3),
                         "n_ligands": len(scores),
                         "method": "virtual_screening"})
    hits.sort(key=lambda h: -h["screen_score"])
    return hits[:top]
