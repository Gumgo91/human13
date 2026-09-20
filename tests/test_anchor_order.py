"""Anchor dedup must be target-aware and order-independent (v4 P0-1).

An anchor is identified by (event_index, canonical target_key, kind, ident,
sign). Different molecular targets wiring the same downstream module are
independent contributions — dropping one because the module is 'occupied'
makes the result depend on assay order. Same-target duplicates across
mechanism/assay/profile evidence must not double-count."""
from types import SimpleNamespace

from backend.drug_targets import drug_anchors


def _assay(name, value, stype="IC50", rel="=", org="Homo sapiens", tid=None):
    d = dict(standard_type=stype, standard_value=str(value),
             standard_units="nM", standard_relation=rel,
             target_pref_name=name, target_organism=org)
    if tid:
        d["target_chembl_id"] = tid
    return d


def _event():
    return SimpleNamespace(kind="chemical", entity="__diagnostic__",
                           label="__diagnostic__")


def _run(acts, mechs=None):
    ev = _event()
    return drug_anchors([ev], {"0": {"mechanisms": mechs or [],
                                     "activities": acts}})


def _sig(out):
    """Order-free anchor signature: (key, kind, id, sign, affinity)."""
    return sorted((a.get("target_key"), a["kind"], a["id"], a["sign"],
                   round(a["affinity_m"] or 0, 12)) for a in out)


PTGS1 = "Prostaglandin G/H synthase 1"
PTGS2 = "Prostaglandin G/H synthase 2"


def test_assay_order_does_not_change_anchors():
    """The v3 diagnostic: swapping assay order flipped the platelet anchor
    from ptgs1/-1/2000nM to ptgs2/+1/40nM. The multiset must be identical."""
    a1 = _assay(PTGS1, 2000)
    a2 = _assay(PTGS2, 40)
    assert _sig(_run([a1, a2])) == _sig(_run([a2, a1]))


def test_both_targets_reach_shared_module():
    """ptgs1 and ptgs2 both wire 'platelet' — neither is dropped."""
    out = _run([_assay(PTGS1, 2000), _assay(PTGS2, 40)])
    platelet = {(a.get("target_key"), a["sign"]) for a in out
                if a["kind"] == "module" and a["id"] == "platelet"}
    assert ("ptgs1", -1) in platelet and ("ptgs2", 1) in platelet


def test_duplicate_records_do_not_double_count():
    a1 = _assay(PTGS1, 2000)
    assert _sig(_run([a1])) == _sig(_run([a1, dict(a1)]))


def test_mechanism_order_does_not_change_anchors():
    m1 = dict(mechanism_of_action="Cyclooxygenase-1 inhibitor",
              action_type="INHIBITOR", target_pref_name=PTGS1)
    m2 = dict(mechanism_of_action="Cyclooxygenase-2 inhibitor",
              action_type="INHIBITOR", target_pref_name=PTGS2)
    acts = [_assay(PTGS1, 2000), _assay(PTGS2, 40)]
    assert _sig(_run(acts, [m1, m2])) == _sig(_run(acts, [m2, m1]))


def test_mixed_evidence_order_permutation():
    """mechanism+assay+profile-free mix: any input order -> same multiset."""
    import itertools
    acts = [_assay(PTGS1, 2000), _assay(PTGS2, 40)]
    mechs = [dict(mechanism_of_action="Cyclooxygenase-1 inhibitor",
                  action_type="INHIBITOR", target_pref_name=PTGS1)]
    ref = _sig(_run(acts, mechs))
    for perm in itertools.permutations(acts):
        for mp in itertools.permutations(mechs):
            assert _sig(_run(list(perm), list(mp))) == ref


def test_synonym_targets_share_canonical_key():
    a = _run([_assay("Adrenoceptor beta 2", 10)])
    b = _run([_assay("Beta-2 adrenergic receptor", 10)])
    assert _sig(a) == _sig(b)


def test_member_does_not_inherit_sibling_or_family_potency():
    """ptgs2 member target cannot receive the family-level Cyclooxygenase
    assay nor ptgs1's value."""
    out = _run([_assay("Cyclooxygenase", 500), _assay(PTGS2, 40)])
    ptgs2 = [a for a in out if a.get("target_key") == "ptgs2"]
    assert all(abs(a["affinity_m"] - 40e-9) < 1e-15 for a in ptgs2)


def test_family_target_pools_member_assays():
    """A family-level claim covers its members: ptgs_family anchor may use
    the ptgs2 member assay when no family assay exists."""
    out = _run([_assay("Cyclooxygenase", 900), _assay(PTGS2, 40)],
               [dict(mechanism_of_action="Cyclooxygenase inhibitor",
                     action_type="INHIBITOR",
                     target_pref_name="Cyclooxygenase")])
    fam = [a for a in out if a.get("target_key") == "ptgs_family"]
    assert fam and all(a["affinity_m"] for a in fam)


def test_bound_only_potency_is_flagged_not_exact():
    """'<'-only assays may provide a bound, flagged distinct from '=' points."""
    out = _run([_assay(PTGS2, 40, rel="<")])
    ptgs2 = [a for a in out if a.get("target_key") == "ptgs2"]
    assert ptgs2 and all("bound" in (a.get("affinity_note") or "")
                         for a in ptgs2)


def test_declared_mechanism_sign_governs_related_assay_direction():
    """An agonist mechanism on a family target governs binding assays on
    member targets: morphine's kappa/delta IC50s are agonist binding, not
    antagonism. Mechanism direction > EC50 evidence > assumed inhibition."""
    mech = dict(mechanism_of_action="Mu opioid receptor agonist",
                action_type="AGONIST",
                target_pref_name="opioid receptor mu 1")
    acts = [_assay("Mu-type opioid receptor", 1),
            _assay("Kappa-type opioid receptor", 24),
            _assay("Delta-type opioid receptor", 140)]
    out = _run(acts, [mech])
    pain = [a for a in out if a["kind"] == "module" and a["id"] == "pain"]
    assert pain and all(a["sign"] == -1 for a in pain)


def test_chembl_id_is_used_when_name_does_not_canonicalize():
    """When the cache carries target_chembl_id, an ID shared with a
    mechanism record maps the assay to that canonical target even if the
    assay's name string alone would not resolve."""
    mech = dict(mechanism_of_action="Cyclooxygenase-2 inhibitor",
                action_type="INHIBITOR", target_pref_name=PTGS2,
                target_chembl_id="CHEMBL230")
    act = _assay("PTGS-2 protein", 40, tid="CHEMBL230")
    out = _run([act], [mech])
    ptgs2 = [a for a in out if a.get("target_key") == "ptgs2"]
    assert ptgs2 and all(abs(a["affinity_m"] - 40e-9) < 1e-15
                         for a in ptgs2)


def test_unnamed_class_mechanism_does_not_set_assay_direction():
    """A mechanism record with no resolved target name (class-level MOA text
    like 'Adrenergic receptor agonist') must not override the assay-default
    direction at specific subtypes. Subtype engagement varies within a class;
    only a NAMED target declares direction for related assays (epinephrine's
    alpha-2 binding then defaults to inhibition -> +1 on the sympathetic
    autoreceptor brake, matching the v3 convention)."""
    mech = dict(mechanism_of_action="Adrenergic receptor agonist",
                action_type="AGONIST", target_pref_name=None)
    acts = [_assay("Adrenergic receptor alpha-2", 8)]
    out = _run(acts, [mech])
    symp = [a for a in out if a["id"] == "sympathetic"]
    assert symp and all(a["sign"] == +1 for a in symp)


def test_named_family_mechanism_still_overrides_member_assays():
    """A mechanism naming a resolved target (even at family level) still
    governs assays on its members — the morphine fix."""
    mech = dict(mechanism_of_action="Mu opioid receptor agonist",
                action_type="AGONIST",
                target_pref_name="opioid receptor mu 1")
    acts = [_assay("Kappa-type opioid receptor", 24)]
    out = _run(acts, [mech])
    pain = [a for a in out if a["id"] == "pain"]
    assert pain and all(a["sign"] == -1 for a in pain)
