"use client";
import {ArrowDownLeft,ArrowUpRight,ExternalLink} from "lucide-react";
import {Sheet,SheetContent,SheetHeader,SheetTitle,SheetDescription} from "@/components/ui/sheet";
import {Accordion,AccordionItem,AccordionTrigger,AccordionContent} from "@/components/ui/accordion";
import {ChartContainer} from "@/components/ui/chart";
import {CartesianGrid,LineChart,Line,XAxis,YAxis,Tooltip} from "recharts";
import type {Run} from "@/lib/simulation";
import {fmt,methodLabels} from "@/lib/simulation";
import type {EventGraph,FlowNode} from "@/lib/event-graph";
import {changeLabel} from "@/lib/event-graph";

export function ResearchNodeDetail({run,graph,node,onClose,onSelect}:{run:Run;graph:EventGraph;node:FlowNode|null;onClose:()=>void;onSelect:(id:string)=>void}){
  const values=node?.state_key?run.traces[node.state_key]:null;
  const chart=values?run.time.map((time,i)=>({time,value:values[i]})):[];
  const linked=node?graph.edges.filter(e=>e.source===node.id||e.target===node.id):[];
  const chemical=node?.compound??(node?.id.startsWith("event:")?run.compounds[node.id.slice(6)]:Object.values(run.compounds).find(c=>c.query.toLowerCase()===node?.group.toLowerCase()));
  const unit=node?.unit==="dimensionless"?"relative deviation":node?.unit;
  return <Sheet open={!!node} onOpenChange={open=>{if(!open)onClose();}}><SheetContent className="research-detail" aria-describedby="node-description">
    {node&&<><SheetHeader><span className="research-eyebrow">{node.group} · {methodLabels[node.method]}</span><SheetTitle>{node.label}</SheetTitle><SheetDescription id="node-description">{node.description}</SheetDescription></SheetHeader>
    <div className="research-detail-body">{!!run.reference_scenarios?.length&&<p className="reference-notice">Conditional reference scenario. The nutrient amount is unknown; these curves do not predict the consumed serving.</p>}
      {node.change&&<><div className="detail-reading"><span>Peak change from the initial value</span><strong>{changeLabel(node.change.delta)} <small>{unit}</small></strong><p>{fmt(run.time[node.change.peakIndex])} min · model value {fmt(node.change.peak,4)} {unit}</p></div>
      <ChartContainer config={{value:{label:node.label,color:"#315da5"}}} className="research-chart"><LineChart data={chart} margin={{left:0,right:16,top:10,bottom:0}}><CartesianGrid stroke="#e8edf3" vertical={false}/><XAxis dataKey="time" tickFormatter={v=>`${fmt(Number(v),0)}m`} minTickGap={35} axisLine={false} tickLine={false}/><YAxis width={55} domain={["auto","auto"]} tickFormatter={v=>fmt(Number(v),2)} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#fff",border:"1px solid #dbe2eb",borderRadius:6,fontSize:13}} labelFormatter={v=>`${fmt(Number(v))} min`} formatter={v=>[`${fmt(Number(v),4)} ${unit}`,node.label]}/><Line type="linear" dataKey="value" stroke="#315da5" strokeWidth={2} dot={false} isAnimationActive={false}/></LineChart></ChartContainer><p className="detail-caption">Δ describes change over time, not a difference from a no-event control.</p></>}
      {node.cell_types.length>0&&<section className="detail-section"><h3>Related cells</h3><p>{node.cell_types.join(" · ")}</p></section>}
      {linked.length>0&&<section className="detail-section"><h3>Connections & feedback</h3><div className="detail-links">{linked.map((e,i)=>{const outward=e.source===node.id;const other=graph.nodes.find(n=>n.id===(outward?e.target:e.source));if(!other)return null;return <button key={e.source+e.target+i} onClick={()=>onSelect(other.id)}>{outward?<ArrowUpRight size={15}/>:<ArrowDownLeft size={15}/>}<span><b>{other.label}</b><small>{e.label}</small></span></button>;})}</div></section>}
      <Accordion type="multiple" className="detail-accordion">
        {node.equations.length>0&&<AccordionItem value="equations"><AccordionTrigger>Equations & parameters</AccordionTrigger><AccordionContent>{node.equations.map((e,i)=><code className="research-equation" key={i}>{e}</code>)}<dl className="detail-parameters">{node.parameters.map((p,i)=><div key={i}><dt>{p.name}</dt><dd>{fmt(p.value,5)} {p.unit}</dd></div>)}</dl></AccordionContent></AccordionItem>}
        {chemical&&<AccordionItem value="molecule"><AccordionTrigger>Molecular identity</AccordionTrigger><AccordionContent>{chemical.structure_svg&&<img className="research-molecule" src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(chemical.structure_svg)}`} alt={`${chemical.query} molecular structure`}/>}<dl className="detail-parameters"><div><dt>Formula</dt><dd>{chemical.formula??"Not available"}</dd></div><div><dt>Molecular weight</dt><dd>{fmt(chemical.molecular_weight,3)} g/mol</dd></div></dl>{chemical.smiles&&<code className="research-equation">{chemical.smiles}</code>}{chemical.mechanisms.map((m,i)=><p key={i}>{m.mechanism_of_action}</p>)}</AccordionContent></AccordionItem>}
        <AccordionItem value="evidence"><AccordionTrigger>Evidence & scope</AccordionTrigger><AccordionContent><div className="research-sources">{[...node.sources,...(chemical?.sources??[])].filter((s,i,a)=>a.findIndex(x=>x.url===s.url)===i&&/^https?:\/\//i.test(s.url)).map(s=><a key={s.url} href={s.url} target="_blank" rel="noreferrer">{s.title}<ExternalLink size={13}/></a>)}</div>{!node.sources.length&&!chemical?.sources?.length&&<p>No experimental evidence is directly linked.</p>}<ul>{node.limitations.map((l,i)=><li key={i}>{l}</li>)}</ul><p>Executable equations do not establish empirical validity. Uncalibrated relative indices are not concentrations or binding fractions.</p></AccordionContent></AccordionItem>
      </Accordion>
    </div></>}
  </SheetContent></Sheet>;
}
