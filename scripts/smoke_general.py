"""Explicit live test of novel domains. This calls the configured OpenRouter model."""
import json
import time
from pathlib import Path
import httpx


def run(body):
    result=None
    with httpx.stream("POST","http://127.0.0.1:8000/api/simulate",json=body,timeout=600) as response:
        response.raise_for_status();kind=""
        for line in response.iter_lines():
            if line.startswith("event: "):kind=line[7:]
            if line.startswith("data: "):
                data=json.loads(line[6:])
                if kind=="progress":print(data["message"],flush=True)
                if kind=="error":raise AssertionError(data)
                if kind=="result":result=data
    assert result
    return result


if __name__=="__main__":
    started=time.monotonic()
    text="밤에 1000 lux의 밝은 빛에 60분 노출되었고, 20분 뒤부터 15분 동안 심한 정신적 스트레스를 받았어."
    first=run({"text":text,"settings":{"horizon_min":180}})
    assert first["generated"]["status"]=="compiled",first["warnings"]
    assert first["coverage"]["events_modeled"]==first["coverage"]["events_total"]==2,first["event_coverage"]
    assert first["generated"]["states"]>=3
    assert len(first["generated"]["observations"])>=2
    assert first["generation"]["complete"]
    second=run({"text":text,"settings":{"horizon_min":180},"replay_id":first["run_id"]})
    assert first["traces"]==second["traces"] and first["plan_hash"]==second["plan_hash"]
    report={"passed":True,"run_id":first["run_id"],"model":first["llm"]["model"],"coverage":first["coverage"],"generation":first["generation"],"states":[s["label"] for s in first["nodes"] if s.get("generated")],"observations":[o["label"] for o in first["generated"]["observations"]],"parameters":len(first["generated"]["parameters"]),"exact_replay":True,"elapsed_seconds":round(time.monotonic()-started,2)}
    out=Path(__file__).resolve().parents[1]/"backend/data/general-smoke.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
