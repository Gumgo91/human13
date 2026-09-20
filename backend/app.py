import asyncio
import json
import os
import re
import uuid
import copy
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse
from .contracts import SimulationRequest, Interpretation, Expansion
from . import llm
from .chemistry import resolve_compound
from .simulation import simulate
from .reference import ToppModel
from .nutrients import normalize_nutrients
from .coupling import model_catalog
from .equation_engine import catalog_from_result
from .expressions import ModelCompileError
from . import model_library
from .body_library import body_catalog, SYSTEMS
from .credibility import build_study_report, GUIDANCE
from .analysis import AnalysisRequest, analyze_snapshot, save_analysis
from . import analysis as analysis_store
from .localization import localize_run
from .progress import stream_work
import numpy as np

ROOT=Path(__file__).parent
RUNS=ROOT/"data"/"runs"
ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]
app=FastAPI(title="Human13 local simulation engine",version="0.5.0")
app.add_middleware(CORSMiddleware,allow_origins=ORIGINS,allow_methods=["GET","POST"],allow_headers=["Content-Type"])
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["localhost","127.0.0.1","testserver"])
# LSODA and some native scientific libraries are not re-entrant.
simulation_lock=asyncio.Lock()


async def run_numerics(*args):
    # Cancellation must not release the LSODA lock while its thread is running.
    async with simulation_lock:
        worker = asyncio.create_task(asyncio.to_thread(simulate, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await asyncio.gather(worker, return_exceptions=True)
            raise


def failure_reason(error):
    if isinstance(error,llm.OutputLimitError):return "output_limit"
    if isinstance(error,llm.ProviderDeadlineError):return "timeout"
    return "model_error"


def model_update(preview,result):
    if not preview:return None
    changed=[]
    before={m["id"]:m for m in preview["metrics"]}
    for metric in result["metrics"]:
        previous=before.get(metric["id"])
        if not previous or previous["values"] is None or metric["values"] is None:continue
        baseline=np.interp(result["time"],preview["time"],previous["values"])
        if not np.allclose(baseline,metric["values"],rtol=1e-6,atol=1e-5):changed.append(metric["id"])
    return {"added_states":result["generated"]["states"]-preview["generated"]["states"],
            "changed_observables":changed}


def generation_reason(interpretation, result, settings):
    """Only construct missing execution paths; never review finished native ones.

    Chemical transport alone does not execute a compound's target effects.
    Those remain an essential model-construction step, not an optional review.
    """
    if not settings.allow_assumptions or not settings.enable_llm_coupling:
        return "disabled"
    if not interpretation.interventions:
        return "empty_input"
    covered={e["event_index"] for e in result["event_coverage"] if e["status"]=="modeled"}
    if any(i not in covered for i in range(len(interpretation.interventions))):
        return "missing_pathways"
    if any(e.kind=="chemical" for e in interpretation.interventions):
        return "chemical_effects"
    return "existing_models"


@app.middleware("http")
async def origin_check(request: Request,call_next):
    if request.method=="POST" and request.headers.get("origin") not in [None,*ORIGINS]:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail":"Origin not allowed"},status_code=403)
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"status":"ready","version":"0.5.0","llm_configured":bool(os.getenv("OPENROUTER_API_KEY")),"model":llm.MODEL,"engines":["Dimension-checked equation compiler","SciPy LSODA/Radau","libRoadRunner CVODE","RDKit"],"local_only":True}


@app.get("/api/registry")
def registry():
    return {"modules":[
        {"id":"equation_model","name":"Physiology model auto-composed from input","engine":"Declared states / reactions / feedback / observations -> shared ODE","status":"general equation compiler","scope":"Generates new states, compartments, reactions, and observations without code changes. Runs after unit, causal-wiring, state-range, and conservation checks."},
        {"id":"topp2000","name":"Glucose / insulin / beta cells","engine":"CellML -> SciPy / SBML -> RoadRunner","status":"reproduces source-file numbers","scope":"Equilibrium from PMR coefficients; distinct from paper/clinical reproduction.","source":ToppModel().sha256},
        {"id":"chemical_transport","name":"Substance transport / distribution / elimination","engine":"Conservative compartment ODE","status":"exploratory assumption","scope":"Uncalibrated transport template applied uniformly across substances"},
        {"id":"nutrition","name":"Per-species digestion / reaction stoichiometry","engine":"Sucrose / lactose / maltose / starch -> monosaccharides","status":"mechanism-based reduction","scope":"Per-sugar hydrolysis, water participation, and hexose-equivalent conservation"},
        {"id":"enterocyte","name":"SGLT1 / GLUT2 / GLUT5 / Na-K pump","engine":"Saturable / reversible carrier transport","status":"mechanism-based reduction","scope":"Physically moves lumen, intracellular, and portal-vein states. Vmax and volumes are uncalibrated."},
        {"id":"endocrine","name":"Incretin / insulin / GLUT4","engine":"Shared-state endocrine / uptake feedback","status":"integrated equations","scope":"Feedback across transport -> gut endocrine -> beta cells -> peripheral uptake"},
        {"id":"renal","name":"Filtration / SGLT2/SGLT1 / urinary excretion","engine":"Filtration / capped reabsorption / net loss","status":"mechanism-based reduction","scope":"Only net excretion after reabsorption is subtracted from shared blood glucose"},
        {"id":"energy","name":"Activity / energy demand","engine":"MET conversion + time integral","status":"definitional conversion","scope":"Total energy estimated from explicit/assumed MET and body mass"},
        {"id":"whole_body_proxy","name":"Circulation / thermal / fluid coupling","engine":"Coupled exploratory ODE","status":"exploratory assumption","scope":"Process couplings not calibrated per person or per experiment"},
        {"id":"mechanism","name":"Existing model + LLM runtime coupling","engine":"Typed reuse / lag-response surrogate compiler","status":"integrated execution","scope":"Reuses existing processes or couples approximations with coefficients/time constants into the same ODE"},
    ]+[{"id":"body_"+id,"name":label,"engine":"Shared-state reduced ODE network","status":"Whole-body mechanism reduction, uncalibrated","scope":scope} for id,label,scope in SYSTEMS]}


@app.get("/api/body-library")
async def body_library(language:str|None=None):
    catalog=body_catalog()
    if language=="en":
        from .localization import localize_mapping
        return await localize_mapping(catalog)
    return catalog


@app.get("/api/reference")
async def reference():
    path=ROOT/"models"/"reproduction.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    async with simulation_lock:return await asyncio.to_thread(ToppModel().reproduce)


def load_run(run_id):
    if not re.fullmatch(r"[a-f0-9]{16}",run_id):raise HTTPException(404,"Run record not found.")
    path=RUNS/(run_id+".json")
    if not path.exists():raise HTTPException(404,"Run record not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def original_text(snapshot):
    if snapshot["result"].get("original_text"):return snapshot["result"]["original_text"]
    for record in snapshot.get("llm_records",[]):
        if record.get("input",{}).get("text"):return record["input"]["text"]
    return snapshot["result"]["interpretation"]["title"]


@app.get("/api/runs/{run_id}")
async def get_run(run_id:str, language:str|None=None):
    saved=load_run(run_id)
    result=saved["result"]
    response={**result,"original_text":original_text(saved),"study_report":build_study_report(result)}
    if response.get("generation",{}).get("planning_status")=="failed" and not response["generation"].get("failure_reason"):
        reason="output_limit" if any("length limit" in w for w in result.get("warnings",[])) else "model_error"
        response["generation"]={**response["generation"],"failure_reason":reason}
    return await localize_run(response) if language=="en" else response


@app.get("/api/guidance")
def guidance():
    return {"references":GUIDANCE,"interpretation":"Methodology reference material. Not an FDA qualification decision."}


@app.post("/api/runs/{run_id}/analysis")
async def analyze_run(run_id:str,body:AnalysisRequest):
    snapshot=load_run(run_id)
    try:
        async with simulation_lock:
            report=await asyncio.to_thread(analyze_snapshot,snapshot,body)
        return save_analysis(report)
    except (ValueError,OverflowError,ZeroDivisionError) as error:
        raise HTTPException(422,str(error)[:400]) from error


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id:str):
    if not re.fullmatch(r"[a-f0-9]{16}",analysis_id):raise HTTPException(404,"Analysis record not found.")
    path=analysis_store.ANALYSES/(analysis_id+".json")
    if not path.exists():raise HTTPException(404,"Analysis record not found.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/execution-catalog")
def execution_catalog():return model_catalog()


@app.get("/api/model-library")
def library():return {"models":model_library.list_plans()}


def sse(event,data):return f"event: {event}\ndata: {json.dumps(data,ensure_ascii=False,allow_nan=False)}\n\n"


def persist_run(result,expansion,records):
    run_id=uuid.uuid4().hex[:16];result["run_id"]=run_id
    result["study_report"]=build_study_report(result)
    snapshot={"result":result,"expansion":expansion.model_dump() if expansion else None,"llm_records":records}
    RUNS.mkdir(parents=True,exist_ok=True)
    path=RUNS/(run_id+".json");temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False),encoding="utf-8");temporary.replace(path)
    return result


@app.post("/api/simulate")
async def simulation(body:SimulationRequest,request:Request):
    if not body.text.strip():raise HTTPException(422,"Please provide an event.")
    replay=load_run(body.replay_id) if body.replay_id else None
    async def work(emit):
        planning_failure=None
        reason="replay" if replay else "empty_input"
        records=[];notes=[];compounds={};expansion=None;preview=None;cache_key=None;cache_hit=False;rejected=[]
        try:
            await emit("progress",{"stage":0,"message":"Interpreting the event, amount, and timing.","fraction":.08})
            if replay:
                interpretation=normalize_nutrients(Interpretation.model_validate(replay["result"]["interpretation"]),original_text(replay))
                compounds=replay["result"]["compounds"]
                expansion=Expansion.model_validate(replay["expansion"]) if replay.get("expansion") else None
                records=replay.get("llm_records",[])
                notes.append("Reused stored events, molecular data, and the LLM plan.")
            elif body.interpretation:
                interpretation=body.interpretation
                notes.append("Structured events were provided directly.")
            else:
                # No fallback: interpretation is the LLM's job. A failed
                # interpretation fails the run rather than fabricating anchors.
                try:
                    interpretation,record=await llm.interpret(body.text);records.append(record)
                except Exception as error:
                    raise ValueError("Automated interpretation is required and the language-model request failed.") from error
            interpretation=normalize_nutrients(interpretation,original_text(replay) if replay else body.text)
            if await request.is_disconnected():return
            await emit("interpretation",interpretation.model_dump())
            await emit("progress",{"stage":1,"message":"Retrieving molecular structures and target evidence.","fraction":.28})
            if not replay:
                chemical=[(i,e.entity) for i,e in enumerate(interpretation.interventions) if e.kind=="chemical" and e.entity]
                results=await asyncio.gather(*[resolve_compound(name) for _,name in chemical],return_exceptions=True)
                for (i,name),r in zip(chemical,results):
                    compounds[str(i)]=r if isinstance(r,dict) else {"query":name,"status":"unresolved","warnings":["external molecule lookup failed"],"sources":[],"mechanisms":[],"activities":[]}
            if await request.is_disconnected():return
            await emit("progress",{"stage":2,"message":"Calculating the existing physiological model.","fraction":.5})
            if not replay and interpretation.interventions:
                preview=await run_numerics(interpretation,body.settings,compounds,None)
                preview.update(run_id=uuid.uuid4().hex[:16],is_preview=True,original_text=body.text,
                               llm={"model":llm.MODEL,"used":bool(records),"calls":len(records),"replayed":False,"elapsed_seconds":sum(r.get("elapsed_seconds",0) for r in records)})
                reason=generation_reason(interpretation,preview,body.settings)
                preview["summary"]="Initial equations are available while missing execution pathways are constructed."
                if reason in {"missing_pathways","chemical_effects"}:
                    await emit("preview",await localize_run(preview,allow_network=False) if body.language=="en" else preview)
            if not replay and reason in {"missing_pathways","chemical_effects"}:
                try:
                    cache_key=model_library.request_key(interpretation,preview["plan"]["engine_hash"],llm.MODEL,{"settings":body.settings.model_dump(),"compounds":preview["plan"]["compound_identities"]})
                    cached=None if body.rebuild_model else model_library.get_plan(cache_key)
                    if cached:
                        expansion,_=cached;cache_hit=True;notes.append("Reused the local model plan for identical events and engine.")
                    else:
                        await emit("progress",{"stage":2,"message":"Constructing missing physiological pathways.","fraction":.57})
                        expansion,record=await llm.expand(interpretation,compounds,catalog_from_result(preview));records.append(record)
                except Exception as error:
                    planning_failure=failure_reason(error)
                    notes.append(str(error) if isinstance(error,RuntimeError) else "Could not validate the LLM pathway response; running numeric modules only.")
            if await request.is_disconnected():return
            if expansion is not None or preview is None:
                await emit("progress",{"stage":3,"message":"Integrating the model and checking conservation.","fraction":.78})
            for attempt in range(3):
                try:
                    if preview is not None and expansion is None:
                        result=copy.deepcopy(preview)
                        result.pop("is_preview",None)
                    else:
                        result=await run_numerics(interpretation,body.settings,compounds,expansion)
                    if expansion and body.settings.allow_assumptions and body.settings.enable_llm_coupling and result["generated"]["status"]!="disabled":
                        uncovered=[e["label"] for e in result["event_coverage"] if e["status"]=="uncovered"]
                        if uncovered:raise ModelCompileError("Physiological events lost their execution path: "+", ".join(uncovered))
                    break
                except (ModelCompileError,ZeroDivisionError,OverflowError) as error:
                    rejected.append({"attempt":attempt+1,"error":str(error)[:500]})
                    if replay:raise
                    if not expansion or attempt>=2:
                        notes.append("The generated model did not pass checks; showing baseline-model results: "+str(error)[:250])
                        expansion=None
                        result=await run_numerics(interpretation,body.settings,compounds,None)
                        break
                    await emit("progress",{"stage":3,"message":f"Checking model corrections · attempt {attempt+1} of 2.","fraction":.8})
                    try:
                        expansion,record=await llm.expand(interpretation,compounds,catalog_from_result(preview),{"expansion":expansion.model_dump(),"error":str(error)[:1800]});records.append(record)
                    except Exception as error:
                        planning_failure=failure_reason(error)
                        notes.append("Automatic repair of generated equations failed; events not connected to baseline equations are shown.")
                        expansion=None
            native_complete=bool(result["event_coverage"]) and all(e["status"]=="modeled" for e in result["event_coverage"])
            status="failed"
            if expansion is not None:status="completed"
            elif reason in {"existing_models","empty_input"}:status="not_needed"
            elif reason=="disabled":status="disabled"
            elif replay and replay["result"].get("generation",{}).get("planning_status")=="not_needed":status="not_needed"
            if status=="not_needed":result["summary"]="Calculated with the existing physiological models."
            result["generation"]={"cached":cache_hit,"repaired":len(rejected),"rejected_attempts":rejected,"complete":native_complete and (expansion is not None or status=="not_needed"),
                                  "native_complete":native_complete,"planning_status":status,"reason":reason,
                                  "failure_reason":planning_failure if expansion is None else None,"update":model_update(preview,result)}
            if expansion and cache_key and native_complete and (result["generated"]["status"]=="compiled" or expansion.model is None):model_library.save_plan(cache_key,expansion,result)
            result["original_text"]=original_text(replay) if replay else body.text
            result["warnings"].extend(notes)
            for c in compounds.values():result["warnings"].extend(c.get("warnings",[]))
            result["llm"]={"model":llm.MODEL,"used":bool(records),"calls":len(records),"replayed":bool(replay),"elapsed_seconds":sum(r.get("elapsed_seconds",0) for r in records)}
            persist_run(result,expansion,records)
            await emit("progress",{"stage":4,"message":"Finalizing the response.","fraction":1})
            await emit("result",await localize_run(result) if body.language=="en" else result)
        except Exception as error:
            # No headers, credentials or raw provider payloads are returned.
            message=str(error)[:300] if isinstance(error,ValueError) else "The run could not be completed. Check the conditions and try again."
            await emit("error",{"message":message})
    async def stream():
        async for kind,data in stream_work(work):
            yield sse(kind,data)
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
