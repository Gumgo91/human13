"""An explicit live smoke run (uses OpenRouter credits); never run by pytest."""
import json
import time
from pathlib import Path
import httpx

BASE="http://127.0.0.1:8000"
TEXT="카페인 100mg을 먹었어. 탄수화물 40g도 먹었고, 30분 뒤에 6 MET 강도로 30분 운동했어. 물 300mL도 마셨어."


def run(body):
    result=None
    with httpx.stream("POST",BASE+"/api/simulate",json=body,timeout=240) as response:
        response.raise_for_status()
        kind=""
        for line in response.iter_lines():
            if line.startswith("event: "):kind=line[7:]
            if line.startswith("data: "):
                data=json.loads(line[6:])
                if kind=="progress":print(data["message"],flush=True)
                if kind=="error":raise AssertionError(data)
                if kind=="result":result=data
    assert result is not None
    return result


if __name__=="__main__":
    started=time.monotonic()
    first=run({"text":TEXT,"settings":{"horizon_min":120}})
    assert first["llm"]["used"] and first["llm"]["calls"]==2,first["warnings"]
    assert len(first["interpretation"]["interventions"])==4
    assert first["compounds"]["0"]["cid"]==2519
    assert first["compounds"]["0"]["activities"]
    assert any(n["method"]=="knowledge" and n["id"].startswith("target:") for n in first["nodes"])
    assert first["coverage"]["quantified"]==6
    assert all(b["passed"] for b in first["validation"]["balances"])
    second=run({"text":TEXT,"settings":{"horizon_min":120},"replay_id":first["run_id"]})
    assert second["llm"]["replayed"]
    assert first["traces"]==second["traces"]
    assert first["plan_hash"]==second["plan_hash"]
    strict=run({"text":TEXT,"settings":{"horizon_min":120,"allow_assumptions":False},"replay_id":first["run_id"]})
    assert strict["coverage"]["quantified"]==1
    assert next(m for m in strict["metrics"] if m["id"]=="glucose")["values"] is None
    assert httpx.get(BASE+"/api/runs/"+first["run_id"]).json()["plan_hash"]==first["plan_hash"]
    denied=httpx.post(BASE+"/api/simulate",headers={"Origin":"https://example.com"},json={"text":"test"})
    assert denied.status_code==403
    report={"passed":True,"run_id":first["run_id"],"model":first["llm"]["model"],"llm_calls":first["llm"]["calls"],"nodes":len(first["nodes"]),"coverage":first["coverage"],"binding_records":len(first["compounds"]["0"]["activities"]),"balances":first["validation"]["balances"],"replay_exact":True,"strict_numeric_outputs":strict["coverage"]["quantified"],"elapsed_seconds":round(time.monotonic()-started,2)}
    out=Path(__file__).resolve().parents[1]/"backend/data/smoke-report.json"
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
