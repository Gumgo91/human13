from rdkit import Chem
from backend.chemistry import exact_binding_records
from backend.contracts import Interpretation, Intervention, RunSettings
from backend.simulation import simulate


def test_similarity_hits_do_not_become_exact_binding_evidence():
    identity=Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))
    payload={"getLindsByUniprotResponse":{"bdb.affinities":[
        {"bdb.smiles":"OCC","bdb.target":"target A","bdb.affinity_type":"Ki","bdb.affinity":" >100","bdb.species":"Human"},
        {"bdb.smiles":"CCCO","bdb.target":"target B","bdb.affinity_type":"Kd","bdb.affinity":"1"},
    ]}}
    records=exact_binding_records(payload,identity)
    assert len(records)==1
    assert records[0]["standard_relation"]==">"
    assert records[0]["standard_units"]=="(units not provided by API)"


def test_experimental_evidence_never_creates_a_biomarker_prediction():
    event=Interpretation(title="evidence",interventions=[Intervention(kind="chemical",label="ethanol",entity="ethanol",quantity=10,unit="mg",route="oral")])
    compound={"0":{"query":"ethanol","status":"resolved","inchikey":"LFQSCWFLJHTTHZ-UHFFFAOYSA-N","sources":[],"activities":[{"target_pref_name":"test target","standard_type":"IC50","standard_value":"10"}]}}
    r=simulate(event,RunSettings(allow_assumptions=False),compound)
    assert any(n["id"]=="target:0:0" and n["method"]=="knowledge" for n in r["nodes"])
    assert all(m["values"] is None for m in r["metrics"])
    assert all(e["source"] in {n["id"] for n in r["nodes"]} and e["target"] in {n["id"] for n in r["nodes"]} for e in r["edges"])
