import asyncio
import pytest
from fastapi.testclient import TestClient
from backend import llm, app as api, model_library
from backend.contracts import Interpretation, Intervention, Expansion
from backend.equation_contracts import ModelSynthesis, EquationModel, ModelBinding
from backend.coupling import MODULES as COUPLING_MODULES
from backend.body_library import MODULES
from backend.nutrients import normalize_nutrients
from test_model_generation_api import events
from test_equation_engine import relay
from backend.simulation import simulate
from backend.contracts import RunSettings
from backend.expressions import ModelCompileError


def plan():
    return Interpretation(title="stress", interventions=[Intervention(kind="other", label="stress", duration_min=10, body_channels=["stress"])])


@pytest.mark.parametrize("synthesis", [
    ModelSynthesis(summary="model synthesis failed", model=EquationModel()),
    ModelSynthesis(summary="could not construct the model", model=None, status="failed"),
])
def test_empty_or_self_reported_failed_generation_is_rejected(monkeypatch, synthesis):
    async def research(_): return []
    async def call(*args): return synthesis, {}
    monkeypatch.setattr(llm, "research_events", research)
    monkeypatch.setattr(llm, "structured_call", call)
    with pytest.raises(RuntimeError): asyncio.run(llm.expand(plan(), {}))


def test_native_only_generation_can_reuse_many_body_processes_and_roundtrip(monkeypatch):
    ids=[key for key in COUPLING_MODULES if key.startswith("body_")][:20]
    context={"states":[],"ports":{},"processes":[{"id":COUPLING_MODULES[key],"label":key,"event_indices":[0]} for key in ids]}
    async def research(_): return []
    async def call(*args): return ModelSynthesis(summary="reuses existing whole-body connections.", model=None, reuse=ids), {}
    monkeypatch.setattr(llm,"research_events",research)
    monkeypatch.setattr(llm,"structured_call",call)
    result,_=asyncio.run(llm.expand(plan(),{},context))
    assert len(result.hypotheses)==20 and result.model is None
    assert Expansion.model_validate_json(result.model_dump_json())==result


def test_dietary_salt_is_not_a_drug_but_intravenous_salt_stays_chemical():
    for route,expected in [("oral","nutrition"),("iv","chemical")]:
        p=Interpretation(title="salt",interventions=[Intervention(kind="chemical",entity="sodium chloride",label="table salt 3g",quantity=3,unit="g",route=route)])
        normalized=normalize_nutrients(p).interventions[0]
        assert normalized.kind==expected and normalized.quantity==3


def test_native_success_does_not_launch_an_unnecessary_mechanism_review(monkeypatch,tmp_path):
    async def interpret(_): return plan(), {}
    async def expand(*args): raise AssertionError("Native pathways already execute")
    monkeypatch.setattr(api.llm,"interpret",interpret)
    monkeypatch.setattr(api.llm,"expand",expand)
    monkeypatch.setattr(api,"RUNS",tmp_path/"runs")
    monkeypatch.setattr(model_library,"LIBRARY",tmp_path/"models")
    with TestClient(api.app) as client:
        result=dict(events(client.post("/api/simulate",json={"text":"stress","settings":{"horizon_min":30}})))["result"]
        assert result["generation"]["native_complete"]
        assert result["generation"]["complete"]
        assert result["generation"]["planning_status"]=="not_needed"
        assert result["body"]["state_count"]==len(MODULES)


def test_independent_formula_errors_are_returned_together_for_repair():
    model=relay()
    model.model.parameters[1].unit="mg"
    model.model.parameters[2].unit="min"
    with pytest.raises(ModelCompileError) as error:
        simulate(plan(),RunSettings(horizon_min=30),{},model)
    assert "sense:" in str(error.value) and "heart_rate_drive:" in str(error.value)


def test_body_dependent_extension_is_explicitly_disabled_when_required_system_is_off():
    model=relay()
    model.model.bindings=[ModelBinding(id="native_cortisol",source="body:cortisol")]
    model.model.processes[0].rate="(strength*event_0+native_cortisol-arbitrary_sensor)/tau"
    on=simulate(plan(),RunSettings(horizon_min=30),{},model)
    assert on["generated"]["status"]=="compiled"
    off=simulate(plan(),RunSettings(horizon_min=30,enable_body_network=False),{},model)
    assert off["generated"]["status"]=="disabled" and "body:cortisol" in off["generated"]["reason"]
    assert not any(k.startswith("model:") for k in off["traces"])
    assert model.model.bindings[0].source=="body:cortisol"


def test_declared_inhibition_cannot_silently_send_a_positive_signal():
    model=relay()
    model.model.couplings[0].signal_direction="decrease"
    with pytest.raises(ModelCompileError,match="signal_direction"):
        simulate(plan(),RunSettings(horizon_min=30),{},model)
    model.model.couplings[0].expression="-heart_gain*remote_response"
    result=simulate(plan(),RunSettings(horizon_min=30),{},model)
    key=result["generated"]["couplings"][0]["trace_key"]
    assert min(result["traces"][key])<0 and max(result["traces"][key])==0
