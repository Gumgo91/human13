"""Unit tests for molecular-target-keyed affinity mapping (drug_targets).

Fixtures are synthetic and live only here — no research data. They exercise
the audit fix: assays pool by molecular target, never by destination module.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backend.drug_targets as dt  # noqa: E402
from backend.contracts import Intervention  # noqa: E402


def act(name, type_="IC50", value=100.0, units="nM", rel="=", org="Homo sapiens", aid=1):
    return {"activity_id": aid, "standard_type": type_,
            "standard_value": str(value), "standard_units": units,
            "standard_relation": rel, "target_pref_name": name,
            "target_organism": org, "assay_chembl_id": "CHEMBL_T",
            "document_chembl_id": "DOC_T"}


def mech(text, action="INHIBITOR", tcid=None):
    return {"mechanism_of_action": text, "action_type": action,
            "target_chembl_id": tcid, "direct_interaction": 1}


def events():
    return [Intervention(kind="chemical", label="took x", entity="x",
                         quantity=100, unit="mg", route="oral", duration_min=30)]


def anchors_for(mechs, acts):
    c = {"mechanisms": mechs, "activities": acts}
    return dt.drug_anchors(events(), {"0": c})


def test_isoforms_do_not_share_affinity():
    """COX-1 and COX-2 both reach 'prostaglandin'; a COX-2 mechanism must get
    the COX-2 potency, not the pooled minimum across both isoforms.
    v4: the COX-1 assay is an independent contribution, kept with its own
    affinity — every anchor's affinity must match ITS OWN target_key."""
    acts = [act("Prostaglandin G/H synthase 1", value=0.7, org="Mus musculus"),
            act("Prostaglandin G/H synthase 2", value=120.0)]
    out = anchors_for([mech("Cyclooxygenase-2 inhibitor")], acts)
    assert out, "mechanism produced no anchors"
    expected = {"ptgs2": 120e-9, "ptgs1": 0.7e-9}
    assert {a["target_key"] for a in out} == {"ptgs2", "ptgs1"}
    for a in out:
        assert abs(a["affinity_m"] - expected[a["target_key"]]) < 1e-12, (
            a["id"], a["target_key"], a["affinity_m"])


def test_family_assay_never_fills_member():
    """A family-level record ('Cyclooxygenase') cannot stand in for COX-1."""
    acts = [act("Prostaglandin G/H synthase (cyclooxygenase)", value=220, org="Mus musculus")]
    out = anchors_for([mech("Cyclooxygenase-1 inhibitor")], acts)
    ptgs1 = [a for a in out if a["target_key"] == "ptgs1"]
    assert ptgs1
    for a in ptgs1:
        assert a["affinity_m"] is None
        assert a["affinity_note"] == "no_same_target_assay"


def test_family_target_may_pool_members_flagged():
    """A family-level mechanism with only member assays gets a flagged pool —
    or, where member assays wire the same destination under the same profile
    note, the family claim is represented by its member anchors (duplicate-
    claim merge keeps the deeper-level key); siblings stay independent."""
    acts = [act("Alpha-1A adrenergic receptor", value=5.0),
            act("Alpha-1B adrenergic receptor", value=9.0)]
    out = anchors_for([mech("Adrenergic receptor alpha-1 antagonist",
                            action="ANTAGONIST")], acts)
    assert out
    keys = {a["target_key"] for a in out}
    if "adra1_family" in keys:
        fam = [a for a in out if a["target_key"] == "adra1_family"][0]
        assert abs(fam["affinity_m"] - 5e-9) < 1e-12
        assert "family" in fam["affinity_note"]
    else:
        # family claim merged into member-level anchors of the same note
        assert keys & {"adra1a", "adra1b"}
        for a in out:
            assert "same_target" in a["affinity_note"]


def test_member_keeps_only_member_assays():
    acts = [act("Alpha-1A adrenergic receptor", value=5.0),
            act("Adrenergic receptor alpha-1", value=1.0)]
    # member mechanism must NOT see the stronger family assay
    out = anchors_for([mech("Alpha-1A adrenergic receptor antagonist",
                            action="ANTAGONIST")], acts)
    member = [a for a in out if a["target_key"] == "adra1a"]
    assert member
    for a in member:
        assert abs(a["affinity_m"] - 5e-9) < 1e-12
        assert "same_target" in a["affinity_note"]


def test_missing_target_exposes_gap():
    acts = [act("Acetylcholinesterase", value=10.0)]
    out = anchors_for([mech("Mu opioid receptor agonist", action="AGONIST")], acts)
    mech_anchors = [a for a in out if a["target_key"] == "oprm"]
    assert mech_anchors, "mechanism produced no oprm-keyed anchors"
    for a in mech_anchors:
        assert a["affinity_m"] is None
        assert a["affinity_note"] == "no_same_target_assay"
    # the acetylcholinesterase assay correctly anchors its OWN target
    ache_anchors = [a for a in out if a["target_key"] == "ache"]
    for a in ache_anchors:
        assert abs(a["affinity_m"] - 10e-9) < 1e-12


def test_inequality_is_flagged_bound_not_point():
    acts = [act("Prostaglandin G/H synthase 2", value=100000.0, rel="<"),
            act("Prostaglandin G/H synthase 2", value=14000.0, rel="=")]
    out = anchors_for([mech("Cyclooxygenase-2 inhibitor")], acts)
    assert abs(out[0]["affinity_m"] - 14000e-9) < 1e-8
    bounds_only = anchors_for(
        [mech("Cyclooxygenase-2 inhibitor")],
        [act("Prostaglandin G/H synthase 2", value=50000.0, rel=">")])
    assert abs(bounds_only[0]["affinity_m"] - 50000e-9) < 1e-7
    assert "bound" in bounds_only[0]["affinity_note"]


def test_kind_priority_over_global_min():
    """An IC50 value is not merged with a Ki: equilibrium constants win."""
    acts = [act("Prostaglandin G/H synthase 2", type_="IC50", value=39.0),
            act("Prostaglandin G/H synthase 2", type_="Ki", value=200.0)]
    out = anchors_for([mech("Cyclooxygenase-2 inhibitor")], acts)
    assert abs(out[0]["affinity_m"] - 200e-9) < 1e-12


def test_ec50_on_family_flips_member_assays():
    """Functional EC50 on the alpha-2 family overrides IC50 'assumed
    inhibition' on the alpha-2A member (guanfacine regression)."""
    acts = [act("Alpha-2A adrenergic receptor", type_="IC50", value=50.0),
            act("Adrenergic receptor alpha-2", type_="EC50", value=400.0)]
    out = anchors_for([], acts)
    assert out, "assay anchors missing"
    assert all("agonism" in a["action"] for a in out), [a["action"] for a in out]


def test_organism_named_target_never_supplies_affinity():
    acts = [act("Mus musculus", value=1.0, org="Mus musculus")]
    out = anchors_for([mech("Cyclooxygenase-2 inhibitor")], acts)
    assert out[0]["affinity_m"] is None
    assert out[0]["affinity_note"] == "no_same_target_assay"


def test_assay_groups_keep_isoform_affinities_separate():
    """Assay-derived anchors: PTGS1 and PTGS2 groups carry their own K."""
    acts = [act("Prostaglandin G/H synthase 1", value=2000.0),
            act("Prostaglandin G/H synthase 2", value=40.0)]
    out = anchors_for([], acts)
    by_key = {}
    for a in out:
        by_key.setdefault(a["target_key"], a["affinity_m"])
    assert abs(by_key.get("ptgs1") - 2000e-9) < 1e-9
    assert abs(by_key.get("ptgs2") - 40e-9) < 1e-12


def test_evidence_preserves_provenance():
    acts = [act("Prostaglandin G/H synthase 2", value=120.0, aid=42)]
    out = anchors_for([mech("Cyclooxygenase-2 inhibitor", tcid="CHEMBL230")], acts)
    ev = out[0]["affinity_evidence"][0]
    assert ev["activity_id"] == 42 and ev["assay_chembl_id"] == "CHEMBL_T"
    assert ev["standard_value"] == "120.0" and abs(ev["value_m"] - 120e-9) < 1e-12
    assert out[0]["target_chembl_id"] == "CHEMBL230"


def test_canon_synonyms():
    assert dt._canon("Cyclooxygenase-2 inhibitor") == "ptgs2"
    assert dt._canon("Prostaglandin G/H synthase 2") == "ptgs2"
    assert dt._canon("Cyclooxygenase") == "ptgs_family"
    assert dt._canon("Prostaglandin G/H synthase (cyclooxygenase)") == "ptgs_family"
    assert dt._canon("Adrenergic receptor alpha-2 agonist") == "adra2_family"
    assert dt._canon("Alpha-2A adrenergic receptor") == "adra2a"
    assert dt._canon("D2-like dopamine receptor antagonist") == "drd2"
    assert dt._canon("HMG-CoA reductase inhibitor") == "hmgcr"
    assert dt._canon("3-hydroxy-3-methylglutaryl-coenzyme A reductase") == "hmgcr"
    assert dt._canon("Vasopressin V2 receptor") == "avpr2"
    assert dt._canon("Heat-stable enterotoxin receptor") == "gucy2c"
    assert dt._canon("Carbonic anhydrase II") == "ca2"
    assert dt._canon("Carbonic anhydrase 2") == "ca2"
    assert dt._canon("Estrogen receptor") == "esr_family"
    assert dt._canon("Estrogen receptor alpha") == "esr1"
    assert dt._canon("Prothrombin") != dt._canon("Thrombin")
    assert (dt._canon("Mus musculus") or "").startswith("nonprotein:")
    assert (dt._canon("Unchecked") or "").startswith("nonprotein:")
