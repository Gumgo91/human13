import type {GraphEdge, GraphNode, Run} from "./simulation";

export type NodeChange={initial:number;peak:number;delta:number;peakIndex:number;changed:boolean};
export type FlowNode=GraphNode&{depth:number;parent:string|null;change:NodeChange|null;observation:boolean};
export type EventGraph={nodes:FlowNode[];edges:GraphEdge[];tree:GraphEdge[];hidden:number};
export type SystemChange={id:string;nodeId:string;label:string;unit:string;change:NodeChange;time:number};

export function systemChanges(run:Run,graph:EventGraph):SystemChange[]{
  const changes:SystemChange[]=[];
  const seen=new Set<string>();
  for(const metric of run.metrics){
    const output=run.nodes.find(n=>n.id===`output:${metric.id}`||n.metric_id===metric.id);
    const node=graph.nodes.find(n=>n.observation&&n.state_key&&n.state_key===output?.state_key);
    const change=traceChange(metric.values??undefined);
    if(!node||!change?.changed||seen.has(node.state_key!))continue;
    seen.add(node.state_key!);
    changes.push({id:metric.id,nodeId:node.id,label:metric.label,unit:metric.unit,change,time:run.time[change.peakIndex]});
  }
  return changes;
}

export function reconcileReveal(graph:EventGraph,shown:string[]):string[]{
  const requested=new Set(shown),kept=new Set<string>();
  for(const n of graph.nodes)if(requested.has(n.id)&&(!n.parent||kept.has(n.parent)))kept.add(n.id);
  return [...kept];
}

export function nextReveal(graph:EventGraph,shown:string[]):string[]{
  const valid=reconcileReveal(graph,shown),seen=new Set(valid);
  const next=graph.nodes.find(n=>!seen.has(n.id)&&(!n.parent||seen.has(n.parent)));
  return next?[...valid,next.id]:valid;
}

export function changeLabel(value:number):string {
  if(value===0)return "0";
  const number=Math.abs(value)<.001?value.toExponential(1):value.toLocaleString("en-US",{maximumFractionDigits:3});
  return value>0?"+"+number:number;
}

export function traceChange(values:number[]|undefined):NodeChange|null {
  if(!values?.length||values.some(v=>!Number.isFinite(v)))return null;
  const initial=values[0];let peakIndex=0;
  for(let i=1;i<values.length;i++)if(Math.abs(values[i]-initial)>Math.abs(values[peakIndex]-initial))peakIndex=i;
  const peak=values[peakIndex],delta=peak-initial;
  return {initial,peak,delta,peakIndex,changed:Math.abs(delta)>1e-5+1e-6*Math.max(Math.abs(initial),Math.abs(peak))};
}

/** A view of executing paths, never a new causal model. Baseline-only states,
 * disconnected changes and knowledge-only edges disappear. Stateless connectors
 * survive only on a path to a changing calculated state. */
export function buildEventGraph(run:Run):EventGraph {
  const changes=new Map(run.nodes.map(n=>[n.id,traceChange(n.state_key?run.traces[n.state_key]:undefined)]));
  const roots=new Set(run.nodes.filter(n=>n.stage===0).map(n=>n.id));
  const eligible=new Map(run.nodes.filter(n=>roots.has(n.id)||(!['unknown','knowledge','llm_hypothesis'].includes(n.method)&&(!n.state_key||changes.get(n.id)?.changed))).map(n=>[n.id,n]));
  // A projected observation of the exact same state is a badge on its process,
  // not another duplicate card. Preserve graph identity through this remapping.
  const aliases=new Map<string,string>(),observed=new Set<string>();
  for(const n of eligible.values())if(n.stage===4){
    const process=[...eligible.values()].find(p=>p.stage!==4&&p.state_key&&p.state_key===n.state_key);
    const id=process?.id??n.id;observed.add(id);if(process)aliases.set(n.id,id);
  }
  const edgeKeys=new Set<string>();
  const edges=run.edges.filter(e=>!['knowledge','hypothesis'].includes(e.kind))
    .map(e=>({...e,source:aliases.get(e.source)??e.source,target:aliases.get(e.target)??e.target}))
    .filter(e=>{const key=e.source+"|"+e.target+"|"+e.kind;if(e.source===e.target||!eligible.has(e.source)||!eligible.has(e.target)||edgeKeys.has(key))return false;edgeKeys.add(key);return true;});
  for(const id of aliases.keys())eligible.delete(id);
  const forward=new Map<string,GraphEdge[]>(),backward=new Map<string,string[]>();
  for(const e of edges){forward.set(e.source,[...(forward.get(e.source)??[]),e]);backward.set(e.target,[...(backward.get(e.target)??[]),e.source]);}
  const depth=new Map<string,number>(),parent=new Map<string,GraphEdge>();
  const queue=[...roots].filter(id=>eligible.has(id));queue.forEach(id=>depth.set(id,0));
  for(let i=0;i<queue.length;i++)for(const e of forward.get(queue[i])??[]){
    if(depth.has(e.target))continue;depth.set(e.target,depth.get(queue[i])!+1);parent.set(e.target,e);queue.push(e.target);
  }
  const useful=new Set<string>(),pending=queue.filter(id=>changes.get(id)?.changed);
  while(pending.length){const id=pending.pop()!;if(useful.has(id))continue;useful.add(id);pending.push(...(backward.get(id)??[]).filter(id=>depth.has(id)));}
  const nodes=queue.filter(id=>useful.has(id)).map(id=>({...eligible.get(id)!,depth:depth.get(id)!,parent:parent.get(id)?.source??null,change:changes.get(id)??null,observation:observed.has(id)}));
  const kept=new Set(nodes.map(n=>n.id));
  return {nodes,edges:edges.filter(e=>kept.has(e.source)&&kept.has(e.target)),tree:nodes.flatMap(n=>parent.has(n.id)?[parent.get(n.id)!]:[]),hidden:run.nodes.length-nodes.length};
}
