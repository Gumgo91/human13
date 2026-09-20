"use client";
import {useCallback,useEffect,useMemo,useRef,useState} from "react";
import {ArrowDownRight,LoaderCircle,RotateCcw,Square} from "lucide-react";
import {Button} from "@/components/ui/button";
import {Textarea} from "@/components/ui/textarea";
import {API,Run,Progress,defaultSettings,requestSimulation} from "@/lib/simulation";
import {buildEventGraph,nextReveal,reconcileReveal} from "@/lib/event-graph";
import {EventFlow} from "./event-flow";
import {ResearchNodeDetail} from "./research-node-detail";
import {progressMessage,reviewMessage} from "@/lib/run-status";
import {SystemResponse} from "./system-response";
import "./research.css";

export default function Workspace(){
  const [text,setText]=useState("");const [run,setRun]=useState<Run|null>(null);
  const [busy,setBusy]=useState(false),[loading,setLoading]=useState(false),[error,setError]=useState("");
  const [progress,setProgress]=useState<Progress|null>(null),[shown,setShown]=useState<string[]>([]);
  const [selected,setSelected]=useState<string|null>(null);
  const [elapsed,setElapsed]=useState(0);const started=useRef(0);
  const abort=useRef<AbortController|null>(null),loadAbort=useRef<AbortController|null>(null);
  const graph=useMemo(()=>run?buildEventGraph(run):null,[run]);
  const revealed=useMemo(()=>graph?reconcileReveal(graph,shown):[],[graph,shown]);
  useEffect(()=>{const id=new URLSearchParams(window.location.search).get("run");
    if(!id||!/^[a-f0-9]{16}$/.test(id))return;
    const controller=new AbortController();loadAbort.current=controller;setLoading(true);
    fetch(API+"/api/runs/"+id+"?language=en",{signal:controller.signal}).then(r=>{if(!r.ok)throw Error("Could not load the saved result.");return r.json() as Promise<Run>})
      .then(r=>{if(controller.signal.aborted)return;setRun(r);setText(r.original_text??r.title);setShown([]);}).catch(e=>{if(e.name!=="AbortError")setError(e.message);}).finally(()=>setLoading(false));
    return()=>controller.abort();},[]);
  useEffect(()=>()=>abort.current?.abort(),[]);
  useEffect(()=>{if(!busy)return;const timer=setInterval(()=>setElapsed(Math.floor((Date.now()-started.current)/1000)),1000);return()=>clearInterval(timer);},[busy]);
  useEffect(()=>{if(selected&&graph&&!graph.nodes.some(n=>n.id===selected))setSelected(null);},[graph,selected]);
  useEffect(()=>{if(!graph||selected||revealed.length>=graph.nodes.length)return;
    const reduced=window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const timer=setTimeout(()=>setShown(current=>reduced?graph.nodes.map(n=>n.id):nextReveal(graph,current)),reduced?0:280);return()=>clearTimeout(timer);},[graph,revealed,selected]);
  const execute=useCallback(async(value:string)=>{
    if(!value.trim())throw Error("Enter an event.");
    loadAbort.current?.abort();abort.current?.abort();const controller=new AbortController();abort.current=controller;
    started.current=Date.now();setElapsed(0);setBusy(true);setError("");setRun(null);setShown([]);setSelected(null);setProgress({stage:0,message:"Interpreting the event.",fraction:0});
    try{const result=await requestSimulation(value,{...defaultSettings,model_parameters:{}},controller.signal,p=>{if(!controller.signal.aborted)setProgress(p);},()=>{},undefined,p=>{if(!controller.signal.aborted)setRun(p);});
      controller.signal.throwIfAborted();setRun(result);
      const url=new URL(window.location.href);url.searchParams.delete("view");url.searchParams.set("run",result.run_id);window.history.replaceState(null,"",url);return result;
    }catch(e){if((e as Error).name!=="AbortError")setError((e as Error).message||"Could not complete the simulation.");throw e;}
    finally{if(abort.current===controller){setBusy(false);setProgress(null);}}
  },[]);
  const executeRef=useRef(execute);executeRef.current=execute;
  useEffect(()=>{const context=(document as Document&{modelContext?:{registerTool:(tool:unknown,options:{signal:AbortSignal})=>unknown}}).modelContext;
    if(!context?.registerTool)return;const lifecycle=new AbortController();
    try{Promise.resolve(context.registerTool({name:"simulate_physiology_events",title:"Simulate physiological events",description:"Simulate events and display connected responses. Includes external LLM calls.",inputSchema:{type:"object",properties:{text:{type:"string",minLength:1,maxLength:2000}},required:["text"],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:true},execute:async(input:unknown)=>{
      if(!input||typeof input!=="object"||!("text" in input)||typeof input.text!=="string"||!input.text.trim()||input.text.length>2000)throw Error("Enter an event description.");
      setText(input.text);const result=await executeRef.current(input.text);if(!result)throw Error("The run was cancelled.");return {run_id:result.run_id,nodes:buildEventGraph(result).nodes.length};
    }},{signal:lifecycle.signal})).catch(()=>{});}catch{}return()=>lifecycle.abort();},[]);
  return <main className="research-page">
    <header className="research-header"><div className="research-eyebrow">PHYSIOLOGY · SYSTEMS MODELING</div><h1>Human<span>13</span></h1>
      <p className="research-description" lang="en">A research workspace for tracing how physiological events propagate from molecules and cells to whole-body responses.</p>
      <div className="research-author" lang="en"><span>AUTHOR</span><strong>Hyunseung Kong</strong><i/><span>GC Biopharma</span></div>
    </header>
    <section className="event-section" aria-labelledby="event-heading"><div className="event-label"><h2 id="event-heading">Event</h2><span>EVENT INPUT</span></div>
      <form className="event-composer" onSubmit={e=>{e.preventDefault();if(!busy)void execute(text).catch(()=>{});}}>
        <Textarea id="event-input" aria-label="Event" value={text} onChange={e=>setText(e.target.value)} maxLength={2000} placeholder="Describe what happened, how much, and when."
          onKeyDown={e=>{if((e.ctrlKey||e.metaKey)&&e.key==="Enter"&&!busy){e.preventDefault();void execute(text).catch(()=>{});}}}/>
        <div className="composer-bottom">{busy?<Button type="button" variant="outline" onClick={()=>abort.current?.abort()}><Square size={14}/>Stop</Button>:<Button type="submit" disabled={!text.trim()||loading}>Trace response<ArrowDownRight size={17}/></Button>}</div>
      </form>
      {error&&<p className="research-error" role="alert">{error}</p>}
      {(busy||loading)&&<p className="research-progress" role="status"><LoaderCircle className="spin" size={16}/>{loading?"Loading the saved result.":progressMessage(progress)}{busy&&<span className="progress-elapsed">{elapsed}s</span>}</p>}
    </section>
    {graph&&run&&<section className="event-results" aria-labelledby="flow-heading">
      {!!run.reference_scenarios?.length&&<div className="reference-notice" role="note"><strong>Reference scenario · actual amount unknown</strong><p>{run.reference_scenarios.map(r=>`${r.quantity} ${r.unit} ${r.entity}`).join(" + ")} is used to explore the pathway. Food composition is assumed. These curves do not predict the serving you consumed.</p></div>}
      {run.presentation_warning&&<p className="flow-note">{run.presentation_warning}</p>}
      {!busy&&reviewMessage(run)&&<p className="review-outcome" role="status">{reviewMessage(run)}</p>}
      <SystemResponse run={run} graph={graph} onSelect={setSelected}/>
      <div className="flow-heading"><div><span className="research-eyebrow">MECHANISTIC DETAIL</span><h2 id="flow-heading">Response pathway</h2></div><Button variant="ghost" onClick={()=>{setSelected(null);setShown([]);}} disabled={!graph.nodes.length}><RotateCcw size={14}/>Replay reveal</Button></div>
      <EventFlow graph={graph} shown={revealed} selected={selected} onSelect={setSelected}/>
      {run.is_preview&&<p className="flow-note">{busy?"Initial results are available while missing pathways are constructed.":"Initial model results. Some pathways could not be completed."}</p>}
      {!!run.event_coverage?.some(e=>e.status==="uncovered")&&<p className="flow-note">Inputs without an executable pathway: {run.event_coverage.filter(e=>e.status==="uncovered").map(e=>e.label).join(" · ")}</p>}
      {!graph.nodes.length&&!busy&&<p className="empty-response">No changing numerical pathway is available for this input. This does not establish an absence of physiological effects.</p>}
      {graph.nodes.length>0&&<div className="flow-foot"><span>{revealed.length} / {graph.nodes.length} nodes</span>{revealed.length<graph.nodes.length&&<button onClick={()=>setShown(graph.nodes.map(n=>n.id))}>Reveal all</button>}</div>}
      {graph.nodes.length>0&&<p className="flow-explanation">Connected pathways with changing model values. Primary links flow downward; additional connections and feedback appear in node details.</p>}
    </section>}
    {run&&graph&&<ResearchNodeDetail run={run} graph={graph} node={graph.nodes.find(n=>n.id===selected)??null} onClose={()=>setSelected(null)} onSelect={setSelected}/>}
    <footer className="research-footer"><span>Human13</span><p>Research use · Model-based exploration</p></footer>
  </main>;
}
