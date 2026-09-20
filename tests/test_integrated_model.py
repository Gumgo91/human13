import numpy as np
import pytest
from backend.contracts import Interpretation,Intervention,RunSettings,Expansion,Hypothesis,ExecutionProposal
from backend.simulation import simulate
from backend.nutrients import normalize_nutrients


def meal(species="sucrose",grams=50):
    return Interpretation(title="meal",interventions=[Intervention(kind="nutrition",label=species,entity=species,quantity=grams,unit="g",route="oral")])


def hypothesis(execution,event_index=0,target="SGLT1"):
    return Hypothesis(event_index=event_index,organ="small intestine",cell_type="Intestinal epithelial cells",target=target,mechanism="Synthetic integration test",direction="increase",limitation="Test gain, not measured",execution=execution)


def test_sucrose_entity_is_canonicalized_without_recovering_amount_from_text():
    # The interpreter emits the alias; normalization canonicalizes it. Amounts
    # are never recovered from text - that is the interpreter's job.
    legacy=Interpretation(title="Sugar",interventions=[Intervention(kind="nutrition",label="sugar intake",entity="Sugar")])
    recovered=normalize_nutrients(legacy,"50g의 Sugar을 먹었어")
    assert recovered.interventions[0].entity=="sucrose"
    assert recovered.interventions[0].quantity is None
    bread=Interpretation(title="bread",interventions=[Intervention(kind="nutrition",entity="bread",label="빵",quantity=50,unit="g")])
    assert normalize_nutrients(bread,"빵 50g 먹음").interventions[0].quantity is None


def test_sucrose_uses_both_sugar_paths_and_conserves_carbon():
    r=simulate(meal(),RunSettings(),{})
    assert sum(m["values"] is not None for m in r["metrics"] if not m["id"].startswith("body:"))==6
    assert r["body"]["status"]=="active"
    assert all(b["passed"] for b in r["validation"]["balances"])
    assert r["traces"]["meal:delivered:glucose"][-1]>0
    assert r["traces"]["meal:delivered:fructose"][-1]>0
    assert max(r["traces"]["flux:sglt1"])>0
    assert max(r["traces"]["flux:glut5"])>0
    assert max(r["traces"]["flux:glut2"])>0
    assert max(r["traces"]["insulin"])>r["traces"]["insulin"][0]
    ids={n["id"] for n in r["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in r["edges"])
    assert r["traces"]["hydrolysis_water"][-1]==pytest.approx(50000/342.2965,rel=.001)


def test_sglt1_loss_of_function_changes_downstream_glucose_and_preserves_fructose_path():
    normal=simulate(meal(),RunSettings(),{})
    blocked=simulate(meal(),RunSettings(sglt1_scale=0),{})
    assert max(blocked["traces"]["flux:sglt1"])==0
    assert blocked["traces"]["meal:delivered:glucose"][-1]==0
    assert blocked["traces"]["meal:delivered:fructose"][-1]>0
    assert max(blocked["traces"]["glucose"])<max(normal["traces"]["glucose"])-20


def test_glut2_or_sucrase_block_prevents_sucrose_glucose_appearance():
    for setting in [RunSettings(glut2_scale=0),RunSettings(sucrase_scale=0)]:
        r=simulate(meal(),setting,{})
        assert max(r["traces"]["flux:appearance"])==0
        assert all(b["passed"] for b in r["validation"]["balances"])


def test_reused_llm_path_does_not_double_count_known_transport():
    base=simulate(meal(),RunSettings(),{})
    expansion=Expansion(summary="reuse",hypotheses=[hypothesis(ExecutionProposal(mode="reuse",module_id="sglt1"))])
    linked=simulate(meal(),RunSettings(),{},expansion)
    assert base["traces"]==linked["traces"]
    assert linked["coupling"][0]["status"]=="reused"
    node=next(n for n in linked["nodes"] if n["id"]=="transport:sglt1")
    assert node["method"]=="hybrid" and node["linked_hypotheses"]


@pytest.mark.parametrize("driver",["event_exposure","glucose_deviation"])
def test_llm_surrogate_is_causal_and_changes_actual_solver_state(driver):
    plan=meal("glucose",10)
    plan.interventions.append(Intervention(kind="chemical",label="compound",entity="test",quantity=100,unit="mg",route="oral",start_min=40))
    expansion=Expansion(summary="new coupling",hypotheses=[hypothesis(ExecutionProposal(mode="surrogate",driver=driver,effect_port="heart_rate_drive",gain=.2,gain_low=.1,gain_high=.3,tau_min=10),1,"new receptor response")])
    linked=simulate(plan,RunSettings(horizon_min=120),{},expansion)
    off=simulate(plan,RunSettings(horizon_min=120,enable_llm_coupling=False),{},expansion)
    assert linked["coupling"][0]["status"]=="compiled"
    times=np.array(linked["time"])
    a=np.array(linked["traces"]["hr"]);b=np.array(off["traces"]["hr"])
    assert np.max(np.abs(a[times<40]-b[times<40]))<1e-5
    assert max(a)-max(b)>1
    assert all(bal["passed"] for bal in linked["validation"]["balances"])


def test_unsupported_ports_stay_explicit_and_cannot_write_arbitrary_state():
    proposal=ExecutionProposal(mode="surrogate",effect_port="arbitrary_glucose_assignment",gain=.2,gain_low=0,gain_high=.3)
    r=simulate(meal(),RunSettings(),{},Expansion(summary="invalid",hypotheses=[hypothesis(proposal)]))
    assert r["coupling"][0]["status"]=="unresolved"
    assert not any(key.startswith("hypothesis:") for key in r["traces"])


def test_drug_effect_needs_exposure_coupling_even_when_systemic_model_exists():
    plan=meal("glucose",10)
    plan.interventions.append(Intervention(kind="chemical",label="compound",entity="test",quantity=100,unit="mg",route="oral"))
    p=ExecutionProposal(mode="reuse",module_id="cardiovascular",effect_port="heart_rate_drive",gain=.2,gain_low=.1,gain_high=.3)
    r=simulate(plan,RunSettings(horizon_min=60),{},Expansion(summary="drug",hypotheses=[hypothesis(p,1,"drug receptor")]))
    assert r["coupling"][0]["status"]=="compiled"
    assert r["coupling"][0]["driver"]=="event_exposure"
    assert max(r["traces"]["hr"])>76


def test_mixed_sugar_doses_at_noninteger_times_conserve_stoichiometry():
    plan=meal("lactose",10)
    plan.interventions.extend([Intervention(kind="nutrition",label="starch",entity="starch",quantity=20,unit="g",start_min=19.25),Intervention(kind="nutrition",label="fructose",entity="fructose",quantity=5,unit="g",start_min=60)])
    r=simulate(plan,RunSettings(horizon_min=60),{})
    assert all(b["passed"] for b in r["validation"]["balances"])
    assert max(r["traces"]["meal:delivered:galactose"])>0
    assert r["traces"]["meal:mouth:fructose"][-1]==pytest.approx(5000/180.156)
