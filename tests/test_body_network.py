import copy
import numpy as np
import pytest
from backend.body_library import body_catalog, MODULES, BODY_PORTS, SYSTEMS
from backend.body_network import route_body_inputs
from backend.contracts import Interpretation, Intervention, RunSettings, Expansion, Hypothesis, ExecutionProposal
from backend.simulation import simulate, quantity
from backend.nutrients import normalize_nutrients
from test_equation_engine import relay, event_plan


def event(label, kind="other", **kwargs):
    return Interpretation(title=label,interventions=[Intervention(kind=kind,label=label,**kwargs)])


def test_all_systems_are_executable_zero_input_equilibrium_and_catalog_not_mutated():
    catalog=copy.deepcopy(body_catalog())
    r=simulate(Interpretation(title="rest",interventions=[]),RunSettings(horizon_min=60),{})
    assert r["body"]["system_count"]==len(SYSTEMS) and r["body"]["state_count"]==len(MODULES)
    assert len([s for s in r["state_specs"] if s["owner"].startswith("body_network:")])==len(MODULES)
    assert all(max(abs(v) for v in r["traces"]["body:"+m["id"]])<1e-10 for m in MODULES)
    assert len(r["metrics"])==len(MODULES)+7 and len({m["id"] for m in r["metrics"]})==len(MODULES)+7
    assert set(BODY_PORTS).issubset(r["model_context"]["ports"])
    assert all(s["changed"]==0 for s in r["body"]["systems"])
    assert body_catalog()==catalog
    ids={n["id"] for n in r["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in r["edges"])


def test_exercise_spreads_to_respiratory_muscle_renal_and_circulatory_outputs():
    plan=event("Exercise",kind="activity",start_min=10.5,duration_min=20,intensity_met=6)
    r=simulate(plan,RunSettings(horizon_min=80),{})
    t=np.array(r["time"])
    for key in ("ventilation","lactate","ampk","gfr","il6"):
        y=np.array(r["traces"]["body:"+key])
        assert np.max(np.abs(y[t<10.5]))<1e-8
        assert np.max(np.abs(y[t>10.5]))>.001
    assert r["traces"]["energy"][-1]==pytest.approx(6*3.5*70/200*20,rel=1e-5)
    assert np.asarray(r["traces"]["body:cardiac_output"])==pytest.approx(np.asarray(r["traces"]["hr"])*np.asarray(r["traces"]["sv"])/1000)
    assert all(b["passed"] for b in r["validation"]["balances"])


def test_injury_infection_couples_immune_coagulation_liver_and_temperature():
    plan=event("infected wound",duration_min=60)
    plan.interventions[0].body_channels=["injury","infection"]
    r=simulate(plan,RunSettings(horizon_min=120),{})
    for key in ("nfkb","tnf","il6","tcell","platelet","fibrin","crp","cortisol"):
        assert max(r["traces"]["body:"+key])>.001,key
    assert max(r["traces"]["temperature"])>37.05
    assert r["event_coverage"][0]["native"]
    assert all(abs(v)<=3+1e-5 for key,values in r["traces"].items() if key.startswith("body:") and not key.startswith("body:rate:") and key!="body:cardiac_output" for v in values)


def test_explicit_nutrient_masses_preserved_without_inventing_mixed_food_composition():
    for entity,channel in [("protein","protein"),("fat","fat"),("sodium chloride","salt"),("fiber","fiber")]:
        plan=normalize_nutrients(event(entity,kind="nutrition",entity=entity,quantity=20,unit="g",route="oral",body_channels=[channel]))
        assert plan.interventions[0].quantity==20
        r=simulate(plan,RunSettings(horizon_min=30),{})
        assert r["body"]["input_profiles"] and r["event_coverage"][0]["native"]
        assert not any(s.startswith("meal:") for s in r["traces"])
    bread=normalize_nutrients(event("빵",kind="nutrition",entity="bread",quantity=50,unit="g"))
    assert bread.interventions[0].quantity is None


def test_chemical_exposure_propagates_into_detox_network_without_llm():
    # Drug plasma exposure propagates through detox/hepatic-load paths even without LLM target decisions.
    plan=event("약",kind="chemical",entity="test compound",quantity=200,unit="mg",route="oral",start_min=5)
    r=simulate(plan,RunSettings(horizon_min=120),{})
    for key in ("cyp450","hepatic_load","glutathione","bilirubin","emetic_drive"):
        assert max(np.abs(r["traces"]["body:"+key]))>.001,key
    # Extends further downstream: hepatic load -> albumin, drug exposure -> dopamine reward signal.
    assert max(np.abs(r["traces"]["body:albumin"]))>0
    assert max(np.abs(r["traces"]["body:dopamine_reward"]))>0
    assert not any(s.startswith("model:") for s in r["traces"])


def test_one_anchor_fans_out_across_many_systems_deterministically():
    plan=event("sauna 30 min",duration_min=30)
    plan.interventions[0].body_channels=["heat"]
    r=simulate(plan,RunSettings(horizon_min=90),{})
    assert "heat" in {p["channel"] for p in r["body"]["input_profiles"]}
    responding={s["id"] for s in r["body"]["systems"] if s["changed"]>0}
    assert len(responding)>=8
    for key in ("thermosensor","sweat","skin_bloodflow","orthostatic_stress","sympathetic","hrv","contractility","preload"):
        assert max(np.abs(r["traces"]["body:"+key]))>.001,key
    # A sauna must not create edema - correct silence is also correct propagation.
    assert max(np.abs(r["traces"]["body:edema"]))<0.01
    # Deterministic re-run of the same plan - the LLM is not involved in downstream propagation.
    again=simulate(plan,RunSettings(horizon_min=90),{})
    assert r["traces"]==again["traces"] and r["plan_hash"]==again["plan_hash"]


def test_hemorrhage_vaccination_and_uv_reach_distal_declared_paths():
    bleed_plan=event("blood donation 400 mL",quantity=400,unit="mL",duration_min=30)
    bleed_plan.interventions[0].body_channels=["hemorrhage"]
    bleed=simulate(bleed_plan,RunSettings(horizon_min=120),{})
    assert "hemorrhage" in {p["channel"] for p in bleed["body"]["input_profiles"]}
    for key in ("epo","erythropoiesis","iron_store","plasma_vol","renin","preload"):
        assert max(np.abs(bleed["traces"]["body:"+key]))>.001,key
    vaccine_plan=event("vaccination",duration_min=60)
    vaccine_plan.interventions[0].body_channels=["vaccination"]
    vaccine=simulate(vaccine_plan,RunSettings(horizon_min=120),{})
    for key in ("dendritic","tcell","bcell","nfkb","lymphocyte_traffic"):
        assert max(np.abs(vaccine["traces"]["body:"+key]))>.001,key
    sun_plan=event("sunlight exposure",duration_min=60)
    sun_plan.interventions[0].body_channels=["uv"]
    sun=simulate(sun_plan,RunSettings(horizon_min=120),{})
    for key in ("vitamin_d_synthesis","calcitriol","uv_damage","skin_damage"):
        assert max(np.abs(sun["traces"]["body:"+key]))>.001,key


def test_generic_meal_and_probiotic_channels_feed_gut_network():
    meal_plan=event("ramen meal",duration_min=40)
    meal_plan.interventions[0].body_channels=["meal","salt"]
    r=simulate(meal_plan,RunSettings(horizon_min=120),{})
    channels={p["channel"] for p in r["body"]["input_profiles"]}
    assert {"meal","salt"}<=channels
    for key in ("gastric_acid","satiety","splanchnic_flow","ghrelin","motility","sodium_load","osmotic_drive"):
        assert max(np.abs(r["traces"]["body:"+key]))>.001,key
    yogurt_plan=event("yogurt",duration_min=40)
    yogurt_plan.interventions[0].body_channels=["probiotic","calcium"]
    yogurt=simulate(yogurt_plan,RunSettings(horizon_min=120),{})
    for key in ("probiotic_drive","scfa","calcium_signal","barrier_damage"):
        assert max(np.abs(yogurt["traces"]["body:"+key]))>.001,key


def test_resolved_drug_target_wires_exposure_into_body_without_llm():
    # ChEMBL-style target records -> deterministic anchors wire plasma exposure into body nodes.
    plan=event("이부프로펜 400mg",kind="chemical",entity="ibuprofen",quantity=400,unit="mg",route="oral",start_min=5)
    compounds={"0":{"status":"resolved","preferred_name":"IBUPROFEN","cid":3672,"query":"ibuprofen",
        "mechanisms":[{"mechanism_of_action":"Cyclooxygenase inhibitor","action_type":"INHIBITOR"}],
        "activities":[{"standard_type":"IC50","target_pref_name":"Cyclooxygenase-2"}],
        "sources":[{"id":"chembl:x","title":"ChEMBL","url":"https://example.org"}]}}
    r=simulate(plan,RunSettings(horizon_min=120),compounds)
    anchors=r["body"]["drug_anchors"]
    assert {a["id"] for a in anchors}>={"prostaglandin","platelet"}
    # Prostaglandin/pyrogen anchors are inhibitory. Platelet carries both arms:
    # COX-1 blockade is antiplatelet, endothelial COX-2/PGI2 blockade is
    # prothrombotic — mixed anchor signs are expected biology.
    assert all(a["sign"]<0 for a in anchors if a["id"] in {"prostaglandin","pyrogen"})
    # Plasma exposure actually suppresses the target node (no LLM, no channels).
    assert min(r["traces"]["body:prostaglandin"])<-.001
    assert min(r["traces"]["body:platelet"])<-.001
    assert any(e["kind"]=="pharmacology" and e["target"]=="body:prostaglandin" for e in r["edges"])
    # A drug without evidence creates no target wiring.
    bare=simulate(event("약",kind="chemical",entity="test compound",quantity=200,unit="mg",route="oral"),RunSettings(horizon_min=60),{})
    assert bare["body"]["drug_anchors"]==[]
    # The inhaled route must also reach plasma.
    inh=simulate(event("흡입 약물",kind="chemical",entity="test drug",quantity=10,unit="mg",route="inhaled",start_min=5),RunSettings(horizon_min=120),{})
    assert "drug0:lung" in inh["traces"] and max(inh["traces"]["drug0:plasma"])>0
    assert max(np.abs(inh["traces"]["body:cyp450"]))>.001
    # Injection without a dose: normalized exposure drives the target anchors and counts as modeled.
    inj=event("주사제",kind="chemical",entity="glp agonist",route="other")
    rj=simulate(inj,RunSettings(horizon_min=240),{"0":{"status":"resolved","query":"x",
        "mechanisms":[{"mechanism_of_action":"Glucagon-like peptide 1 receptor agonist","action_type":"AGONIST"}],"activities":[],"sources":[]}})
    assert rj["event_coverage"][0]["status"]=="modeled"
    assert max(rj["traces"]["body:satiety"])>.001 and min(rj["traces"]["body:appetite"])<-.001


def test_llm_can_drive_any_native_body_port_and_feedback_remains_when_llm_disabled():
    model=relay()
    model.model.couplings[0].port="body_platelet"
    plan=event_plan(5,20)
    r=simulate(plan,RunSettings(horizon_min=70),{},model)
    off=simulate(plan,RunSettings(horizon_min=70,enable_llm_coupling=False),{},model)
    assert max(r["traces"]["body:platelet"])>.01
    assert max(r["traces"]["body:fibrin"])>.001
    assert max(np.abs(off["traces"]["body:platelet"]))<1e-10
    assert off["body"]["status"]=="active"
    assert any(e["source"]=="model:remote_response" and e["target"]=="body:platelet" for e in r["edges"])
    no_body=simulate(plan,RunSettings(horizon_min=70,enable_body_network=False),{},model)
    assert no_body["generated"]["status"]=="compiled"
    assert no_body["generated"]["disabled_body_couplings"][0]["port"]=="body_platelet"
    assert "model:remote_response" in no_body["traces"] and "body:platelet" not in no_body["traces"]


def test_native_body_reuse_is_exact_and_legacy_port_displays_additive_signal():
    plan=event("스트레스",duration_min=20)
    h=Hypothesis(event_index=0,organ="nervous system",cell_type="sympathetic neurons",target="sympathetic drive",mechanism="reuse",direction="increase",limitation="test",execution=ExecutionProposal(mode="reuse",module_id="body_sympathetic"))
    base=simulate(plan,RunSettings(horizon_min=40),{})
    reused=simulate(plan,RunSettings(horizon_min=40),{},Expansion(summary="reuse",hypotheses=[h]))
    assert base["traces"]==reused["traces"]
    h.execution=ExecutionProposal(mode="surrogate",effect_port="body_platelet",gain=.2,gain_low=0,gain_high=.4)
    r=simulate(event_plan(5,20),RunSettings(horizon_min=40),{},Expansion(summary="legacy",hypotheses=[h]))
    assert r["traces"]["hypothesis:0:multiplier"][0]==0
    assert max(r["traces"]["body:platelet"])>0


def test_llm_adjudicated_channels_replace_text_parsing():
    # An anchor chosen by the LLM executes as-is even when the label matches no keyword.
    plan=event("불가사의한 저녁",duration_min=30)
    plan.interventions[0].body_channels=["infection","stress"]
    r=simulate(plan,RunSettings(horizon_min=90),{})
    assert {p["channel"] for p in r["body"]["input_profiles"]}=={"infection","stress"}
    assert max(r["traces"]["body:il6"])>.001
    # [] is a "no anchor" decision - it is never resurrected by a regex fallback.
    quiet=event("스트레스를 받았지만 아무 일도",duration_min=20)
    quiet.interventions[0].body_channels=[]
    r2=simulate(quiet,RunSettings(horizon_min=60),{})
    assert r2["body"]["input_profiles"]==[]
    assert max(np.abs(r2["traces"]["body:cortisol"]))<1e-10
    # Names not on the LLM list are discarded.
    odd=event("Exercise",kind="activity",duration_min=20,intensity_met=6)
    odd.interventions[0].body_channels=["made_up_channel"]
    assert route_body_inputs(odd.interventions,quantity)[0]==[]


def test_unknown_chemical_and_negated_events_do_not_invent_specific_body_effects():
    r=simulate(event("mystery drug",kind="chemical",entity="unregistered",quantity=10,unit="mg",route="oral"),RunSettings(horizon_min=30),{})
    assert r["body"]["input_profiles"]==[]
    assert max(np.abs(r["traces"]["body:platelet"]))<1e-10
    # Negation is the interpreter's job: a negated stimulus emits no channels,
    # and no keyword fallback may resurrect one.
    negated=event("no stress at all",duration_min=20)
    negated.interventions[0].body_channels=[]
    profiles,_=route_body_inputs(negated.interventions,quantity)
    assert profiles==[]
    deprived=event("couldn't sleep",duration_min=20)
    deprived.interventions[0].body_channels=["sleep_deprivation"]
    profiles,_=route_body_inputs(deprived.interventions,quantity)
    assert {p["channel"] for p in profiles}=={"sleep_deprivation"}
    strict=simulate(event("infected wound",duration_min=30),RunSettings(allow_assumptions=False),{})
    assert strict["body"]["status"]=="disabled"
    assert not any(k.startswith("body:") for k in strict["traces"])


def test_turning_off_whole_body_is_a_real_structural_change_and_preserves_existing_modules():
    plan=event("Exercise",kind="activity",duration_min=20,intensity_met=6)
    on=simulate(plan,RunSettings(horizon_min=60),{})
    off=simulate(plan,RunSettings(horizon_min=60,enable_body_network=False),{})
    assert off["body"]["status"]=="disabled" and not any(k.startswith("body:") for k in off["traces"])
    assert len(off["metrics"])==6
    assert np.max(np.abs(np.array(on["traces"]["hr"])-np.array(off["traces"]["hr"])))>.01
    assert off["traces"]["energy"][-1]==pytest.approx(on["traces"]["energy"][-1],rel=1e-5)


def test_affinity_occupancy_and_selectivity_weighting():
    """Measured affinities wire through the occupancy model; inferred targets at a lower confidence tier."""
    comp={"0":{"entity":"ibuprofen","query":"ibuprofen","molecular_weight":206.28,
        "mechanisms":[{"mechanism_of_action":"Cyclooxygenase inhibitor","action_type":"INHIBITOR"}],
        "activities":[{"standard_type":"Ki","standard_value":300,"standard_units":"nM","target_pref_name":"Cyclooxygenase-1"},
                      {"standard_type":"IC50","standard_value":7000,"standard_units":"nM","target_pref_name":"Cyclooxygenase-2"}]}}
    def run(mg):
        plan=event("ibu",kind="chemical",entity="ibuprofen",quantity=mg,unit="mg",route="oral")
        return simulate(plan,RunSettings(horizon_min=240),comp)
    hi,lo=run(400),run(40)
    anchors={a["id"]:a for a in hi["body"]["drug_anchors"]}
    assert anchors["prostaglandin"]["affinity_m"]==pytest.approx(3e-7)
    # Weaker targets get a lower selectivity weight (Kmin/K).
    assert anchors["nfkb"]["weight"]<anchors["prostaglandin"]["weight"]
    # Occupancy is dose-dependent but saturates: a 10x dose difference is not a 10x response.
    d_hi=abs(min(hi["traces"]["body:prostaglandin"]));d_lo=abs(min(lo["traces"]["body:prostaglandin"]))
    assert d_hi>d_lo>0 and d_hi<3*d_lo


def test_virtual_screen_panel_scoring_is_deterministic():
    from backend.vscreen import _descriptors,_score_query
    a=_descriptors("CC(=O)Oc1ccccc1C(=O)O");assert a is not None
    assert _score_query(a,a)==pytest.approx(1.)
    other=_descriptors("CN1CCC[C@H]1c1cccnc1")
    assert _score_query(a,other)<0.4


def test_claimed_outcome_backtraces_to_event_path():
    """A claim like "drug then weight loss" is reconstructed as an outcome-node -> event path."""
    comp={"0":{"entity":"tirzepatide","query":"tirzepatide",
        "mechanisms":[{"mechanism_of_action":"Glucagon-like peptide 1 receptor agonist","action_type":"AGONIST"}],
        "activities":[]}}
    plan=event("mounjaro",kind="chemical",entity="tirzepatide",route="other")
    plan.interventions[0].outcome_nodes=["lipolysis"]
    res=simulate(plan,RunSettings(horizon_min=240),comp)
    paths=res["body"]["outcome_paths"]
    assert paths and paths[0]["found"]
    ids=[e["id"] for e in paths[0]["chain"]]
    assert ids[0].startswith("rx_") and ids[-1]=="body:lipolysis"
    # No path is created for an outcome that was not claimed.
    plan2=event("mounjaro",kind="chemical",entity="tirzepatide",route="other")
    assert simulate(plan2,RunSettings(horizon_min=60),comp)["body"]["outcome_paths"]==[]
