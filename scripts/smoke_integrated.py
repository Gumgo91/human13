"""Live v0.2 regression. Uses the configured OpenRouter model."""
import json
from pathlib import Path
import httpx
from scripts.smoke_api import run


if __name__=="__main__":
    text="50g의 설탕을 먹었어"
    normal=run({"text":text})
    assert normal["llm"]["calls"]==2,normal["warnings"]
    event=normal["interpretation"]["interventions"][0]
    assert event["entity"]=="sucrose" and event["quantity"]==50
    assert normal["coverage"]["quantified"]==6
    assert normal["coverage"]["integrated"]>=1,normal["coupling"]
    assert all(b["passed"] for b in normal["validation"]["balances"])
    blocked=run({"text":text,"replay_id":normal["run_id"],"settings":{"sglt1_scale":0}})
    assert max(blocked["traces"]["flux:sglt1"])==0
    assert max(blocked["traces"]["glucose"])<max(normal["traces"]["glucose"])-20
    repeat=run({"text":text,"replay_id":normal["run_id"]})
    assert repeat["traces"]==normal["traces"]
    assert repeat["plan_hash"]==normal["plan_hash"]
    report={"passed":True,"run_id":normal["run_id"],"input":text,"nodes":len(normal["nodes"]),"coverage":normal["coverage"],"coupling":normal["coupling"],"peak_glucose_normal":max(normal["traces"]["glucose"]),"peak_glucose_sglt1_off":max(blocked["traces"]["glucose"]),"llm_seconds":normal["llm"]["elapsed_seconds"],"replay_exact":True}
    Path("backend/data/integrated-smoke.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
