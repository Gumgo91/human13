import math
import numpy as np
import pytest
from backend.contracts import Expansion, Interpretation, Intervention, RunSettings
from backend.equation_contracts import EquationModel, ModelBinding
from backend.simulation import simulate
from backend.expressions import ModelCompileError, compile_expression, variable, output_function


def state(id,unit="dimensionless",initial=0,lower=0,upper=10,compartment=None):
    return dict(id=id,label=id,entity=id,compartment=compartment or id,level="cell",unit=unit,initial=initial,lower=lower,upper=upper,description="Declared test quantity")


def parameter(id,value,unit="dimensionless",low=0,high=100):
    return dict(id=id,label=id,value=value,unit=unit,low=low,high=high,rationale="Test parameter, not an empirical physiological claim")


def process(id,rate,changes,event_indices=None):
    return dict(id=id,label=id,rate=rate,changes=[dict(state=k,coefficient=v) for k,v in changes.items()],event_indices=event_indices or [0],description="Declared process")


def expansion(**data):return Expansion(summary="Generated equations",hypotheses=[],model=EquationModel.model_validate(data))


def relay():
    return expansion(states=[state("arbitrary_sensor"),state("remote_response")],
        parameters=[parameter("tau",5,"min",1,30),parameter("strength",1,high=2),parameter("heart_gain",.1,low=-.5,high=.5)],
        processes=[process("sense","(strength*event_0-arbitrary_sensor)/tau",{"arbitrary_sensor":1}),process("relay","(arbitrary_sensor-remote_response)/tau",{"remote_response":1})],
        observations=[dict(id="novel_output",label="arbitrary downstream relative index",expression="remote_response",unit="dimensionless",description="Relative test output")],
        couplings=[dict(port="heart_rate_drive",expression="heart_gain*remote_response",description="Declared native connection")])


def event_plan(start=10,duration=20):
    return Interpretation(title="unregistered domain",interventions=[Intervention(kind="other",label="unregistered exposure",entity="unregistered exposure",start_min=start,duration_min=duration)])


def test_arbitrary_new_entities_execute_and_reach_native_outputs_with_causal_recovery():
    e=relay();r=simulate(event_plan(),RunSettings(horizon_min=100),{},e)
    t=np.asarray(r["time"]);x=np.asarray(r["traces"]["model:remote_response"])
    assert r["coverage"]["events_modeled"]==1 and r["generated"]["states"]==2
    assert np.max(np.abs(x[t<10]))==0 and max(x)>.85
    assert x[-1]<.0001
    hr=np.asarray(r["traces"]["hr"])
    assert np.max(np.abs(hr[t<10]-72))<1e-8 and hr.max()>77
    assert any(m["id"]=="model:obs:novel_output" and m["values"] for m in r["metrics"])
    ids={n["id"] for n in r["nodes"]}
    assert all(edge["source"] in ids and edge["target"] in ids for edge in r["edges"])
    repeated=simulate(event_plan(),RunSettings(horizon_min=100),{},e)
    assert r["traces"]==repeated["traces"] and r["plan_hash"]==repeated["plan_hash"]


def test_generated_parameters_change_entire_shared_solution_and_can_be_disabled():
    baseline=simulate(event_plan(),RunSettings(horizon_min=80),{},relay())
    changed=simulate(event_plan(),RunSettings(horizon_min=80,model_parameters={"strength":.5}),{},relay())
    assert max(changed["traces"]["model:remote_response"])==pytest.approx(max(baseline["traces"]["model:remote_response"])*.5,rel=1e-5)
    assert max(changed["traces"]["hr"])<max(baseline["traces"]["hr"])
    off=simulate(event_plan(),RunSettings(enable_llm_coupling=False),{},relay())
    assert off["generated"]["status"]=="disabled" and not any(k.startswith("model:") for k in off["traces"])
    with pytest.raises(ModelCompileError,match="outside the declared range"):
        simulate(event_plan(),RunSettings(model_parameters={"strength":99}),{},relay())


def test_multi_event_opposing_signals_share_one_state_without_pre_event_effect():
    model=relay();model.model.parameters.append(type(model.model.parameters[0])(**parameter("inhibit",.5)))
    model.model.processes[0].rate="(strength*event_0-inhibit*event_1-arbitrary_sensor)/tau"
    model.model.processes[0].event_indices=[0,1]
    for s in model.model.states:s.lower=-10
    plan=event_plan(10,60);plan.interventions.append(Intervention(kind="other",label="opposite",start_min=40,duration_min=30))
    r=simulate(plan,RunSettings(horizon_min=100),{},model)
    one=simulate(event_plan(10,60),RunSettings(horizon_min=100),{},relay())
    t=np.array(r["time"])
    # Different solver event grids: compare exact common times before event 2.
    for i,minute in enumerate(r["time"]):
        if minute<40 and minute in one["time"]:
            j=one["time"].index(minute)
            assert r["traces"]["model:remote_response"][i]==pytest.approx(one["traces"]["model:remote_response"][j],abs=1e-6)
    assert r["coverage"]["events_modeled"]==2
    assert len([s for s in r["state_specs"] if s["key"]=="model:arbitrary_sensor"])==1


def test_general_reaction_conserves_amount_across_units_and_final_time_impulse():
    model=expansion(states=[state("reservoir","mg",upper=1000),state("receiver","g",upper=10)],parameters=[parameter("k",.1,"1/min")],
      processes=[process("transport","k*reservoir",{"reservoir":-1,"receiver":1},[0,1])],
      impulses=[dict(event_index=0,state="reservoir",amount="dose_0"),dict(event_index=1,state="reservoir",amount="dose_1")],
      observations=[dict(id="received",label="transferred amount",expression="receiver",unit="mg",description="Amount in receiver")],
      invariants=[dict(label="mass",weights=[dict(state="reservoir",coefficient=1),dict(state="receiver",coefficient=1)])])
    plan=Interpretation(title="reaction",interventions=[Intervention(kind="other",label="dose",quantity=200,unit="mg",start_min=10.5),Intervention(kind="other",label="last dose",quantity=100,unit="mg",start_min=60)])
    r=simulate(plan,RunSettings(horizon_min=60),{},model)
    assert r["traces"]["model:reservoir"][-1]==pytest.approx(200*math.exp(-.1*49.5)+100,abs=1e-5)
    assert r["traces"]["model:obs:received"][-1]==pytest.approx(200*(1-math.exp(-.1*49.5)),abs=1e-5)
    assert all(l["passed"] for l in r["validation"]["balances"])
    assert r["validation"]["balances"][0]["max_error"]<1e-10
    model.model.processes[0].changes[1].coefficient=2
    with pytest.raises(ModelCompileError,match="conservation law"):
        simulate(plan,RunSettings(horizon_min=60),{},model)


@pytest.mark.parametrize("formula",["__import__('os').system('whoami')","arbitrary_sensor.__class__","arbitrary_sensor[0]","[x for x in range(5)]","2**99"])
def test_expression_boundary_does_not_execute_code(formula):
    with pytest.raises(ModelCompileError):compile_expression(formula,{"arbitrary_sensor":variable("dimensionless",lambda t,y:0)})


def test_compiler_rejects_units_orphan_paths_unknown_names_and_trajectory_violation():
    model=relay();model.model.parameters[0].unit="mg"
    with pytest.raises(ModelCompileError,match="Unit mismatch"):simulate(event_plan(),RunSettings(),{},model)
    model=relay();model.model.observations=[];model.model.couplings=[]
    result=simulate(event_plan(),RunSettings(),{},model)
    assert result["generated"]["auto_observed"]==["remote_response"]
    assert result["generated"]["observations"][0]["expression"]=="remote_response"
    model=relay();model.model.processes[0].rate="-arbitrary_sensor/tau"
    with pytest.raises(ModelCompileError,match="actual equation drive"):simulate(event_plan(),RunSettings(),{},model)
    model=relay();model.model.processes[0].rate="invented*event_0"
    with pytest.raises(ModelCompileError,match="Undefined formula variable"):simulate(event_plan(),RunSettings(),{},model)
    model=relay();model.model.states[0].upper=.1
    with pytest.raises(ModelCompileError,match="state range"):simulate(event_plan(),RunSettings(),{},model)


def test_native_to_generated_to_native_feedback_is_inside_same_ode():
    model=relay()
    # Feedback deviations can cross below zero during autonomic recovery.
    for s in model.model.states:s.lower=-10
    model.model.bindings=[ModelBinding(id="native_hr",source="hr")]
    # Revalidate edited data to keep the typed schema at the boundary.
    model=Expansion.model_validate(model.model_dump())
    model.model.parameters.extend([type(model.model.parameters[0])(**parameter("hr_base",72,"1/min",1,100)),type(model.model.parameters[0])(**parameter("feedback",.1))])
    model.model.processes[0].rate="(strength*event_0-feedback*(native_hr/hr_base-1)-arbitrary_sensor)/tau"
    r=simulate(event_plan(),RunSettings(horizon_min=80),{},model)
    assert max(r["traces"]["hr"])>72
    assert any(e["source"]=="cardiovascular" and e["target"]=="model:arbitrary_sensor" for e in r["edges"])


def test_unit_conversion_in_arithmetic_and_offset_native_temperatures():
    symbols={"a":variable("mg",lambda t,y:1000),"b":variable("g",lambda t,y:2)}
    fn=output_function(compile_expression("a+b",symbols),"mg")
    assert fn(0,[])==pytest.approx(3000)
    symbols={"t":variable("degC",lambda t,y:37),"base":variable("kelvin",lambda t,y:310.15)}
    assert output_function(compile_expression("t-base",symbols),"kelvin")(0,[])==pytest.approx(0)


def test_initial_relative_level_cannot_shift_native_physiology_before_event():
    model=relay();model.model.states[1].initial=1
    with pytest.raises(ModelCompileError,match="must be 0 at the initial state"):
        simulate(event_plan(40,10),RunSettings(),{},model)
