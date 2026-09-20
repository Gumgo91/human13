"use client";
import {useId,useLayoutEffect,useMemo,useRef,useState} from "react";
import {ArrowUpRight} from "lucide-react";
import type {EventGraph,FlowNode} from "@/lib/event-graph";
import {changeLabel} from "@/lib/event-graph";
import {fmt} from "@/lib/simulation";

export function EventFlow({graph,shown,selected,onSelect}:{graph:EventGraph;shown:string[];selected:string|null;onSelect:(id:string)=>void}){
  const root=useRef<HTMLDivElement>(null),cards=useRef(new Map<string,HTMLButtonElement>());
  const [hovered,setHovered]=useState<string|null>(null);
  const marker=useId().replace(/:/g,"");
  const [lines,setLines]=useState<{width:number;height:number;paths:{key:string;d:string;active:boolean}[]}>({width:0,height:0,paths:[]});
  const visible=useMemo(()=>new Set(shown),[shown]);
  const layers=useMemo(()=>{const groups=new Map<number,FlowNode[]>();for(const n of graph.nodes)if(visible.has(n.id))groups.set(n.depth,[...(groups.get(n.depth)??[]),n]);return [...groups].sort((a,b)=>a[0]-b[0]);},[graph,visible]);
  useLayoutEffect(()=>{const container=root.current;if(!container)return;
    let frame=0;
    const measure=()=>{const box=container.getBoundingClientRect();const paths=graph.tree.flatMap(e=>{
      const source=cards.current.get(e.source),target=cards.current.get(e.target);if(!source||!target)return [];
      const a=source.getBoundingClientRect(),b=target.getBoundingClientRect();
      const x1=a.left+a.width/2-box.left,y1=a.bottom-box.top,x2=b.left+b.width/2-box.left,y2=b.top-box.top;
      const sourceLayer=source.parentElement!.getBoundingClientRect(),targetLayer=target.parentElement!.getBoundingClientRect();
      const unobstructed=a.bottom>=sourceLayer.bottom-2&&b.top<=targetLayer.top+2;
      const gutter=x1<box.width/2?5:box.width-5,mid=(y1+y2)/2;
      const d=unobstructed?`M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2-4}`:
        `M${x1},${y1} L${x1},${y1+6} Q${x1},${y1+10} ${x1+(gutter<x1?-6:6)},${y1+10} L${gutter},${y1+10} L${gutter},${y2-10} L${x2},${y2-10} L${x2},${y2-4}`;
      const focus=selected??hovered;
      return [{key:e.source+"/"+e.target,d,active:focus===e.source||focus===e.target}];
    });setLines({width:box.width,height:box.height,paths});};
    const schedule=()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(measure);};
    const observer=new ResizeObserver(schedule);observer.observe(container);cards.current.forEach(card=>observer.observe(card));schedule();
    return()=>{observer.disconnect();cancelAnimationFrame(frame);};
  },[graph,shown,selected,hovered]);
  return <div className="pathway-canvas" ref={root}>
    <svg className="pathway-lines" width={lines.width} height={lines.height} aria-hidden="true"><defs><marker id={marker} markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M1 1 L5 3.5 L1 6" fill="none" stroke="#b0bfd2" strokeWidth="1.2"/></marker></defs>{lines.paths.map(p=><path key={p.key} d={p.d} fill="none" stroke={p.active?"#5e81b4":"#c4d0df"} strokeWidth={p.active?1.7:1.2} markerEnd={`url(#${marker})`}/>)}</svg>
    {layers.map(([depth,nodes])=><div className="pathway-layer" key={depth} data-size={Math.min(3,graph.nodes.filter(n=>n.depth===depth).length)}>
      {nodes.map(n=><button key={n.id} type="button" ref={el=>{if(el)cards.current.set(n.id,el);else cards.current.delete(n.id);}} onClick={()=>onSelect(n.id)}
        onMouseEnter={()=>setHovered(n.id)} onMouseLeave={()=>setHovered(null)} onFocus={()=>setHovered(n.id)} onBlur={()=>setHovered(null)}
        className={`pathway-node ${n.stage===0?"input-node":""} ${n.observation?"observation-node":""} ${n.generated?"generated-node":""} ${selected===n.id?"selected":""}`} aria-label={`${n.label} details`}>
        <span className="node-eyebrow">{n.stage===0?"INPUT":n.generated?"MODEL EXTENSION":n.group}<ArrowUpRight size={14}/></span>
        <strong>{n.label}</strong>
        {n.reference_scenario&&<span className="reference-badge">Reference: {n.reference_scenario.quantity} {n.reference_scenario.unit} {n.reference_scenario.entity}</span>}<span className="node-bottom">{n.change?<span className="node-delta">Δ {changeLabel(n.change.delta)} <small>{n.unit==="dimensionless"?"relative deviation":n.unit}</small></span>:<span>{n.intervention?.start_min?`${fmt(n.intervention.start_min)} min after start`:"Start"}</span>}{n.observation&&<span className="observation-label">Observable</span>}</span>
      </button>)}
    </div>)}
  </div>;
}
