"""Live multi-system scenario, including an actual LLM-to-body extension."""
import json
from pathlib import Path
import time
from smoke_general import run

text = "카페인 100mg을 먹었어. 20분 뒤부터 6 MET로 30분 운동했어. 단백질 25g과 소금 3g을 먹었고, 밤에 1000 lux 조명에 60분 노출됐어."
start=time.monotonic()
result=run({"text":text,"settings":{"horizon_min":180},"rebuild_model":True})
assert result["body"]["state_count"]==79
assert result["body"]["system_count"]==15
assert result["generation"]["complete"], result["event_coverage"]
assert result["generated"]["status"]=="compiled", result["warnings"]
assert len(result["metrics"])>=35
assert any(c["port"].startswith("body_") for c in result["generated"].get("couplings",[])), result["generated"].get("couplings")
assert all(b["passed"] for b in result["validation"]["balances"])
again=run({"text":text,"settings":result["plan"]["settings"],"replay_id":result["run_id"]})
assert again["traces"]==result["traces"] and again["plan_hash"]==result["plan_hash"]
summary={"run_id":result["run_id"],"systems":result["body"]["system_count"],"body_states":79,
         "generated_states":result["generated"]["states"],"generated_connections":result["generated"].get("couplings"),
         "metrics":len(result["metrics"]),"nodes":len(result["nodes"]),"events":result["event_coverage"],
         "changed_systems":sum(s["changed"]>0 for s in result["body"]["systems"]),
         "exact_replay":True,"elapsed_seconds":time.monotonic()-start,"generation":result["generation"]}
out=Path(__file__).resolve().parents[1]/"backend/data/body-smoke.json"
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
