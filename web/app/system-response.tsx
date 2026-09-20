"use client";
import {ArrowUpRight} from "lucide-react";
import type {Run} from "@/lib/simulation";
import {fmt} from "@/lib/simulation";
import {changeLabel,systemChanges,type EventGraph,type SystemChange} from "@/lib/event-graph";

export function SystemResponse({run,graph,onSelect}:{run:Run;graph:EventGraph;onSelect:(id:string)=>void}){
  const changes=systemChanges(run,graph);
  const reading=(item:SystemChange,compact=false)=><button key={item.id} className={compact?"response-extra":"response-reading"} onClick={()=>onSelect(item.nodeId)}>
    <span>{item.label}<ArrowUpRight size={14}/></span><strong>{changeLabel(item.change.delta)} <small>{["dimensionless","1"].includes(item.unit)?"relative index":item.unit}</small></strong><p>Peak change at {fmt(item.time)} min</p>
  </button>;
  return <section className="system-response" aria-labelledby="response-heading">
    <div className="response-header"><div><span className="research-eyebrow">SYSTEMIC SUMMARY</span><h2 id="response-heading">Whole-body response</h2></div>{run.is_preview&&<span className="response-status">Initial model</span>}</div>
    {changes.length?<><p className="response-caption">{run.reference_scenarios?.length?"In this reference scenario, ":"In this model, "}{changes.length} systemic {changes.length===1?"observable changes":"observables change"}. Values show the largest departure from the initial state over {fmt(run.time.at(-1),0)} minutes.</p>
      <div className="response-readings">{changes.slice(0,6).map(item=>reading(item))}</div>
      {changes.length>6&&<details className="response-more"><summary>{changes.length-6} more changing observables</summary><div>{changes.slice(6).map(item=>reading(item,true))}</div></details>}
      <p className="response-caption response-scope">Model changes are not measured outcomes or differences from a no-event control.</p>
    </>:<p className="response-caption">No connected whole-body observable changes in the current model. This does not establish that the event has no effect.</p>}
  </section>;
}
