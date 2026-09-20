import numpy as np
import pytest
from backend.contracts import Interpretation,Intervention,RunSettings
from backend.kernel import ModularSystem
from backend.reference import ToppModel
from backend.simulation import simulate,quantity


def test_transfer_conservation_and_fractional_events():
    engine=ModularSystem()
    engine.state("a","source","mg",0,"source")
    engine.state("b","target","g",0,"target")
    engine.transfer("ab","a","b",.1)
    engine.impulse(7.25,"a",200)
    engine.impulse(40,"a",100)
    t,y,_=engine.integrate(40)
    assert np.max(np.abs(y[0]+1000*y[1]-np.where(t>=7.25,200,0)-np.where(t>=40,100,0)))<1e-7
    assert y[0,-1]>=100


def test_invalid_ownership_and_units():
    e=ModularSystem();e.state("a","a","mg",0,"one")
    with pytest.raises(ValueError):e.state("a","a","mg",0,"two")
    e.state("b","b","L",0,"two")
    with pytest.raises(Exception):e.transfer("invalid","a","b",1)
    assert quantity(2,"g","mg")==2000
    assert quantity(2,"L","mg") is None


def test_reference_equations_against_cvode_and_analytical_solution():
    report=ToppModel().reproduce()
    assert report["passed"]
    assert report["source_initial_conditions"]=={"G":600.,"I":0.,"beta":0.}
    assert report["perturbed_initial_max_difference"]<1e-4


def test_combined_inputs_share_state_and_conserve_mass():
    events=Interpretation(title="mixed",interventions=[
        Intervention(kind="chemical",label="test compound",entity="test",quantity=200,unit="mg",route="oral"),
        Intervention(kind="chemical",label="second dose",entity="test",quantity=50,unit="mg",route="oral",start_min=30.5),
        Intervention(kind="nutrition",label="carbs",entity="carbohydrate",quantity=20,unit="g",route="oral",start_min=10),
        Intervention(kind="activity",label="exercise",duration_min=30,intensity_met=6,start_min=60),
        Intervention(kind="hydration",label="water",quantity=300,unit="mL",route="oral",start_min=20),
    ])
    result=simulate(events,RunSettings(horizon_min=120),{})
    assert all(b["passed"] for b in result["validation"]["balances"])
    assert sum(m["values"] is not None for m in result["metrics"] if not m["id"].startswith("body:"))==6
    assert result["body"]["status"]=="active"
    assert "drug1:plasma" not in result["traces"] # repeated substance shares states
    assert result["traces"]["energy"][-1]==pytest.approx(6*3.5*70/200*30,rel=1e-5)
    assert max(result["traces"]["glucose"])>100
    replay=simulate(events,RunSettings(horizon_min=120),{})
    assert result["plan_hash"]==replay["plan_hash"]
    assert result["traces"]==replay["traces"]


def test_unknown_and_strict_inputs_do_not_fabricate_biomarkers():
    events=Interpretation(title="unknown",interventions=[Intervention(kind="other",label="잠을 설침"),Intervention(kind="chemical",entity="unknown",label="unknown",quantity=200,unit="mg",route="oral")])
    r=simulate(events,RunSettings(allow_assumptions=False),{})
    assert all(m["values"] is None for m in r["metrics"])
    assert len(r["unmodeled"])==2
    assert not any(key.startswith("drug") for key in r["traces"])


def test_zero_input_equilibrium_and_activity_end():
    empty=simulate(Interpretation(title="baseline",interventions=[]),RunSettings(),{})
    assert np.max(np.abs(np.array(empty["traces"]["glucose"])-100))<1e-8
    exercise=Interpretation(title="activity",interventions=[Intervention(kind="activity",label="work",duration_min=30,intensity_met=4,start_min=10.5)])
    r=simulate(exercise,RunSettings(horizon_min=120),{})
    t=np.array(r["time"]);energy=np.array(r["traces"]["energy"])
    assert np.max(energy[t<10.5])==0
    assert np.max(energy[t>=40.5])-np.min(energy[t>=40.5])<1e-6
