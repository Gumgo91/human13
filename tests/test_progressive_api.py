import json
from fastapi.testclient import TestClient
from backend import app as api
from backend.contracts import Interpretation,Intervention,Expansion
from backend.body_library import MODULES


def test_covered_input_finishes_after_one_calculation_without_llm_review(monkeypatch,tmp_path):
    async def interpret(text):
        return Interpretation(title="Sugar",interventions=[Intervention(kind="nutrition",label="Sugar",entity="sucrose",quantity=50,unit="g",route="oral")]),{}
    calls=[]
    actual_simulate=api.simulate
    def simulate(*args):calls.append(True);return actual_simulate(*args)
    async def expand(*args):raise AssertionError("A covered input must not trigger generation")
    monkeypatch.setattr(api,"simulate",simulate)
    monkeypatch.setattr(api.llm,"interpret",interpret)
    monkeypatch.setattr(api.llm,"expand",expand)
    monkeypatch.setattr(api,"RUNS",tmp_path)
    with TestClient(api.app) as client:
        response=client.post("/api/simulate",json={"text":"50g의 Sugar을 먹었어","settings":{"horizon_min":60}})
    events={};order=[]
    for block in response.text.strip().split("\n\n"):
        kind=block.splitlines()[0][7:];data=json.loads(block.splitlines()[1][6:]);events[kind]=data;order.append(kind)
    assert "preview" not in order
    assert len(calls)==1
    assert events["result"]["generation"]["planning_status"]=="not_needed"
    assert events["result"]["generation"]["complete"]
    assert events["result"]["coverage"]["quantified"]>=6
    assert events["result"]["body"]["state_count"]==len(MODULES)
    assert not events["result"].get("is_preview")
    assert events["result"]["original_text"]=="50g의 Sugar을 먹었어"
    assert len(list(tmp_path.glob("*.json")))==1
