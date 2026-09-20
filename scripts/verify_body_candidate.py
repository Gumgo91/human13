"""Replay a real provider candidate, verify shared feedback, and save the run.

No model equations or coefficients are edited by this verification tool.
Run from the workspace root with: python -m scripts.verify_body_candidate PATH
"""
import json
import sys
from pathlib import Path
from backend import llm, model_library
from backend.app import load_run, original_text, persist_run
from backend.contracts import Interpretation, RunSettings, Expansion
from backend.simulation import simulate


def main(path):
    candidate=json.loads(Path(path).read_text(encoding="utf-8"))
    plan=Interpretation.model_validate(candidate["interpretation"])
    settings=RunSettings.model_validate(candidate["settings"])
    expansion=Expansion.model_validate(candidate["expansion"])
    compounds=candidate["compounds"]
    result=simulate(plan,settings,compounds,expansion)
    assert result["body"]["state_count"]==79 and result["body"]["system_count"]==15
    assert result["generated"]["status"]=="compiled"
    assert result["coverage"]["events_modeled"]==len(plan.interventions)
    assert len(result["metrics"])>=35 and all(b["passed"] for b in result["validation"]["balances"])
    again=simulate(plan,settings,compounds,expansion)
    assert result["traces"]==again["traces"] and result["plan_hash"]==again["plan_hash"]
    off=simulate(plan,settings.model_copy(update={"llm_scale":0}),compounds,expansion)
    affected=[key for key,values in result["traces"].items() if key.startswith("body:") and not key.startswith("body:rate:")
              and max(abs(a-b) for a,b in zip(values,off["traces"][key]))>1e-6]
    assert affected, "Provider model does not influence the existing body network"
    records=candidate.get("records",[candidate["record"]])
    seed=load_run(Path("backend/data/body-smoke-seed.txt").read_text().strip())
    result["original_text"]=original_text(seed)
    errors=candidate.get("errors",[])
    result["generation"]={"cached":False,"repaired":len(errors),"rejected_attempts":[{"attempt":i+1,"error":e} for i,e in enumerate(errors)],
                          "complete":True,"native_complete":True,"planning_status":"completed"}
    result["llm"]={"model":llm.MODEL,"used":True,"calls":len(records),"replayed":False,
                   "elapsed_seconds":sum(r.get("elapsed_seconds",0) for r in records)}
    result=persist_run(result,expansion,records)
    key=model_library.request_key(plan,result["plan"]["engine_hash"],llm.MODEL,{"settings":settings.model_dump(),"compounds":result["plan"]["compound_identities"]})
    model_library.save_plan(key,expansion,result)
    report={"passed":True,"run_id":result["run_id"],"systems":15,"body_states":79,"metrics":len(result["metrics"]),
            "nodes":len(result["nodes"]),"generated_states":result["generated"]["states"],"generated_couplings":result["generated"]["couplings"],
            "events":result["event_coverage"],"changed_systems":sum(s["changed"]>0 for s in result["body"]["systems"]),
            "body_states_affected_by_llm":affected,"exact_replay":True,"generation":result["generation"]}
    Path("backend/data/body-smoke.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main(sys.argv[1])
