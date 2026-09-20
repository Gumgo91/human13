import asyncio
import json
import time
import httpx
import pytest
from backend import progress, llm, localization, app as api, model_library
from backend.contracts import Interpretation, Intervention, Expansion, RunSettings
from backend.equation_contracts import ModelSynthesis
from fastapi.testclient import TestClient


def test_heartbeats_preserve_phase_and_preview_order_while_work_waits(monkeypatch):
    monkeypatch.setattr(progress,"HEARTBEAT_SECONDS",.01)
    async def work(emit):
        await emit("preview",{"ready":True})
        await progress.report("Reviewing additional mechanisms.","generation")
        await asyncio.sleep(.045)
        await emit("result",{"ready":True})
    async def collect():return [row async for row in progress.stream_work(work)]
    rows=asyncio.run(collect())
    heartbeats=[data for kind,data in rows if kind=="progress" and data["heartbeat"]]
    assert heartbeats and all(h["phase"]=="generation" for h in heartbeats)
    assert all(h["message"]=="Reviewing additional mechanisms." for h in heartbeats)
    kinds=[kind for kind,_ in rows]
    assert kinds.count("preview")==1 and kinds.count("result")==1
    assert kinds.index("preview")<kinds.index("result")


def test_external_deadline_is_wall_clock_even_if_transport_keeps_waiting(monkeypatch):
    client_class=httpx.AsyncClient
    async def handler(request):
        await asyncio.sleep(1)
        return httpx.Response(200,json={})
    transport=httpx.MockTransport(handler)
    monkeypatch.setattr(llm.httpx,"AsyncClient",lambda **kwargs:client_class(transport=transport,**kwargs))
    monkeypatch.setenv("OPENROUTER_API_KEY","test-placeholder")
    monkeypatch.setattr(llm,"TEXT_DEADLINE_SECONDS",.02)
    start=time.monotonic()
    with pytest.raises(llm.ProviderDeadlineError):asyncio.run(llm.structured_call("test",{},Interpretation))
    assert time.monotonic()-start<.5


def test_existing_nonchemical_paths_use_bounded_reasoning_but_new_paths_keep_review():
    data={"interventions":{"interventions":[{"kind":"nutrition"}]},"available_model":{"processes":[{"event_indices":[0]}]}}
    assert llm.reasoning_options(data,ModelSynthesis)["effort"]=="low"
    data["available_model"]["processes"]=[]
    assert llm.reasoning_options(data,ModelSynthesis)["effort"]=="medium"
    data["repair"]={"error":"bad equation"}
    assert llm.reasoning_options(data,ModelSynthesis)["effort"]=="low"


def test_translation_cannot_delay_initial_preview_or_hold_final_results(monkeypatch):
    calls=[]
    async def slow(strings):calls.append(True);await asyncio.sleep(1)
    monkeypatch.setattr(localization,"translate_strings",slow)
    monkeypatch.setattr(localization,"PRESENTATION_DEADLINE_SECONDS",.02)
    monkeypatch.setattr(localization,"_cache",{})
    run={"nodes":[{"label":"미번역", "state_key":"x"}],"traces":{"x":[0,1]}}
    initial=asyncio.run(localization.localize_run(run,allow_network=False))
    assert not calls and initial["traces"]==run["traces"]
    final=asyncio.run(localization.localize_run(run))
    assert final["presentation_warning"] and final["traces"]==run["traces"]


def test_failed_review_keeps_preview_results_and_reports_output_limit(monkeypatch,tmp_path):
    async def interpret(text):return Interpretation(title="Novel exposure",interventions=[Intervention(kind="other",label="Novel exposure",entity="novel exposure")]),{}
    async def expand(*args):raise llm.OutputLimitError("output limit")
    monkeypatch.setattr(api.llm,"interpret",interpret);monkeypatch.setattr(api.llm,"expand",expand)
    monkeypatch.setattr(api,"RUNS",tmp_path/"runs");monkeypatch.setattr(model_library,"LIBRARY",tmp_path/"models")
    with TestClient(api.app) as client:response=client.post("/api/simulate",json={"text":"novel exposure","settings":{"horizon_min":30}})
    events={block.splitlines()[0][7:]:json.loads(block.splitlines()[1][6:]) for block in response.text.strip().split("\n\n")}
    assert events["result"]["generation"]["failure_reason"]=="output_limit"
    assert events["result"]["generation"]["update"]=={"added_states":0,"changed_observables":[]}
    assert events["preview"]["traces"]==events["result"]["traces"]


def test_native_input_never_generates_even_when_rebuild_is_requested(monkeypatch,tmp_path):
    calls=[]
    async def interpret(text):return Interpretation(title="Candy",interventions=[Intervention(kind="nutrition",label="One candy",entity="candy",route="oral")]),{}
    async def expand(*args):calls.append(True);return Expansion(summary="Existing pathways suffice",hypotheses=[],model=None),{}
    monkeypatch.setattr(api.llm,"interpret",interpret);monkeypatch.setattr(api.llm,"expand",expand)
    monkeypatch.setattr(api,"RUNS",tmp_path/"runs");monkeypatch.setattr(model_library,"LIBRARY",tmp_path/"models")
    with TestClient(api.app) as client:
        for i in range(2):
            response=client.post("/api/simulate",json={"text":"ate 1 candy","rebuild_model":bool(i),"settings":{"horizon_min":30}})
            result=next(json.loads(block.splitlines()[1][6:]) for block in response.text.strip().split("\n\n") if block.startswith("event: result"))
            assert result["generation"]["complete"]
            assert result["generation"]["planning_status"]=="not_needed"
            assert not result["generation"]["cached"]
    assert not calls


def test_only_missing_paths_and_chemical_effects_need_model_construction():
    plan=Interpretation(title="Candy",interventions=[Intervention(kind="nutrition",label="Candy",entity="candy")])
    covered={"event_coverage":[{"event_index":0,"status":"modeled"}]}
    assert api.generation_reason(plan,covered,RunSettings())=="existing_models"
    plan.interventions.append(Intervention(kind="other",label="Novel stimulus"))
    assert api.generation_reason(plan,covered,RunSettings())=="missing_pathways"
    plan.interventions=[Intervention(kind="chemical",label="Aspirin",entity="aspirin")]
    assert api.generation_reason(plan,covered,RunSettings())=="chemical_effects"
    assert api.generation_reason(plan,covered,RunSettings(enable_llm_coupling=False))=="disabled"


def test_native_result_replays_without_false_generation_failure(monkeypatch,tmp_path):
    async def interpret(text):return Interpretation(title="Candy",interventions=[Intervention(kind="nutrition",label="One candy",entity="candy",route="oral")]),{}
    async def expand(*args):raise AssertionError("No model generation is needed")
    monkeypatch.setattr(api.llm,"interpret",interpret);monkeypatch.setattr(api.llm,"expand",expand)
    monkeypatch.setattr(api,"RUNS",tmp_path/"runs")
    def result(response):return next(json.loads(block.splitlines()[1][6:]) for block in response.text.strip().split("\n\n") if block.startswith("event: result"))
    with TestClient(api.app) as client:
        first=result(client.post("/api/simulate",json={"text":"ate 1 candy","settings":{"horizon_min":30}}))
        again=result(client.post("/api/simulate",json={"text":"ate 1 candy","settings":{"horizon_min":30},"replay_id":first["run_id"]}))
    assert again["generation"]["planning_status"]=="not_needed"
    assert again["generation"]["complete"]
    assert first["traces"]==again["traces"]
