"""Revalidate a saved generated plan, repairing through the same real provider.

Explicitly invoked developer smoke tool; no hand-coded physiological patches.
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from backend.app import load_run, original_text, persist_run
from backend import llm, model_library
from backend.contracts import Interpretation,RunSettings,Expansion
from backend.simulation import simulate
from backend.equation_engine import catalog_from_result


async def main(seed_id):
    started=time.monotonic();saved=load_run(seed_id)
    plan=Interpretation.model_validate(saved["result"]["interpretation"])
    settings=RunSettings.model_validate(saved["result"]["plan"]["settings"])
    compounds=saved["result"]["compounds"];expansion=Expansion.model_validate(saved["expansion"])
    preview=simulate(plan,settings,compounds);records=saved.get("llm_records",[]);repairs=[]
    for attempt in range(4):
        try:
            result=simulate(plan,settings,compounds,expansion)
            assert result["generated"]["status"]=="compiled"
            assert result["coverage"]["events_modeled"]==len(plan.interventions)
            break
        except Exception as error:
            print("Compiler:",str(error),flush=True)
            if attempt==3:raise
            repairs.append(str(error))
            expansion,record=await llm.expand(plan,compounds,catalog_from_result(preview),{"expansion":expansion.model_dump(),"error":str(error)})
            records.append(record)
    repeat=simulate(plan,settings,compounds,expansion)
    assert result["traces"]==repeat["traces"] and result["plan_hash"]==repeat["plan_hash"]
    result["original_text"]=original_text(saved)
    result["generation"]={"cached":False,"repaired":len(repairs),"rejected_attempts":[{"attempt":i+1,"error":e} for i,e in enumerate(repairs)],"complete":True}
    result["llm"]={"model":llm.MODEL,"used":True,"calls":len(records),"replayed":False,"elapsed_seconds":sum(r.get("elapsed_seconds",0) for r in records)}
    result=persist_run(result,expansion,records)
    key=model_library.request_key(plan,result["plan"]["engine_hash"],llm.MODEL,{"settings":settings.model_dump(),"compounds":result["plan"]["compound_identities"]})
    model_library.save_plan(key,expansion,result)
    report={"passed":True,"run_id":result["run_id"],"coverage":result["coverage"],"generation":result["generation"],"observations":[o["label"] for o in result["generated"]["observations"]],"states":[n["label"] for n in result["nodes"] if n.get("generated")],"native_couplings":result["generated"]["couplings"],"exact_replay":True,"elapsed_seconds":round(time.monotonic()-started,2)}
    Path("backend/data/general-smoke.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":asyncio.run(main(sys.argv[1]))
