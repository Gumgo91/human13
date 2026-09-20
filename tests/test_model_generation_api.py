import json
from fastapi.testclient import TestClient
from backend import app as api
from backend import model_library
from test_equation_engine import relay, event_plan


def events(response):
    return [(block.splitlines()[0][7:],json.loads(block.splitlines()[1][6:])) for block in response.text.strip().split("\n\n")]


def test_model_compile_feedback_repair_exact_cache_replay_and_explicit_rebuild(monkeypatch,tmp_path):
    calls=[]
    async def interpret(text):return event_plan(0,10),{"elapsed_seconds":0}
    async def expand(plan,compounds,context,repair=None):
        calls.append(repair)
        assert "states" in context and "ports" in context
        model=relay()
        if len(calls)==1:model.model.parameters[0].unit="mg"
        if repair:assert "Unit mismatch" in repair["error"]
        return model,{"elapsed_seconds":0}
    monkeypatch.setattr(api.llm,"interpret",interpret)
    monkeypatch.setattr(api.llm,"expand",expand)
    monkeypatch.setattr(api,"RUNS",tmp_path/"runs")
    monkeypatch.setattr(model_library,"LIBRARY",tmp_path/"models")
    body={"text":"new physiological domain","settings":{"horizon_min":60}}
    with TestClient(api.app) as client:
        first=events(client.post("/api/simulate",json=body))
        assert [k for k,v in first].index("preview")<[k for k,v in first].index("result")
        result=dict(first)["result"]
        assert result["generation"]["complete"] and result["generation"]["repaired"]==1
        assert result["generated"]["status"]=="compiled" and len(calls)==2
        again=dict(events(client.post("/api/simulate",json=body)))["result"]
        assert again["generation"]["cached"] and len(calls)==2
        assert result["traces"]==again["traces"] and result["plan_hash"]==again["plan_hash"]
        replay=dict(events(client.post("/api/simulate",json={**body,"replay_id":result["run_id"]})))["result"]
        assert result["traces"]==replay["traces"] and len(calls)==2
        rebuilt=dict(events(client.post("/api/simulate",json={**body,"rebuild_model":True})))["result"]
        assert not rebuilt["generation"]["cached"] and len(calls)==3
        assert len(client.get("/api/model-library").json()["models"])==1


def test_rejected_models_are_not_cached_or_presented_as_completed(monkeypatch,tmp_path):
    async def interpret(text):return event_plan(0,10),{}
    async def expand(*args):
        model=relay();model.model.processes[0].rate="missing*event_0"
        return model,{}
    monkeypatch.setattr(api.llm,"interpret",interpret)
    monkeypatch.setattr(api.llm,"expand",expand)
    monkeypatch.setattr(api,"RUNS",tmp_path/"runs")
    monkeypatch.setattr(model_library,"LIBRARY",tmp_path/"models")
    with TestClient(api.app) as client:
        result=dict(events(client.post("/api/simulate",json={"text":"novel event","settings":{"horizon_min":60}})))["result"]
        assert not result["generation"]["complete"]
        assert result["generated"]["status"]=="absent"
        assert len(result["generation"]["rejected_attempts"])==3
        assert not client.get("/api/model-library").json()["models"]
        assert not any(key.startswith("model:") for key in result["traces"])
