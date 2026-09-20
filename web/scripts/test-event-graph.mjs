import test from "node:test";
import assert from "node:assert/strict";
import {buildEventGraph,traceChange,nextReveal,reconcileReveal,systemChanges} from "../lib/event-graph.ts";

const node=(id,state_key=null,stage=2,extra={})=>({id,label:id,state_key,stage,method:"assumption",...extra});
const edge=(source,target,kind="regulation")=>({source,target,kind,label:""});
const root=node("event:0",null,0);

test("only input-connected changing paths survive; duplicate observations merge",()=>{
  const graph=buildEventGraph({nodes:[root,node("active","x"),node("output:x","x",4),node("rest","rest"),node("island","island"),node("lookup",null,1,{method:"knowledge"})],
    traces:{x:[0,1,0],rest:[100,100,100],island:[0,2,1]},edges:[edge("event:0","active"),edge("event:0","rest"),edge("active","output:x"),edge("event:0","lookup","knowledge")]});
  assert.deepEqual(graph.nodes.map(n=>n.id),["event:0","active"]);
  assert.equal(graph.nodes[1].observation,true);
  assert.equal(graph.tree.length,1);
});

test("stateless connectors need a changing descendant, and unknown paths stay hidden",()=>{
  const graph=buildEventGraph({nodes:[root,node("connector"),node("dead"),node("response","x"),node("unknown","u",2,{method:"unknown"})],
    traces:{x:[0,1],u:[0,2]},edges:[edge("event:0","connector"),edge("connector","response"),edge("event:0","dead"),edge("event:0","unknown")]});
  assert.deepEqual(graph.nodes.map(n=>n.id),["event:0","connector","response"]);
});

test("cycles remain in details while the primary graph stays directed downward",()=>{
  const input={nodes:[root,node("a","a"),node("b","b")],traces:{a:[0,1],b:[0,2]},
    edges:[edge("event:0","a"),edge("a","b"),edge("b","a","feedback"),edge("a","a")]};
  const graph=buildEventGraph(input);assert.equal(graph.edges.length,3);assert.equal(graph.tree.length,2);
  for(const e of graph.tree)assert.ok(graph.nodes.find(n=>n.id===e.source).depth<graph.nodes.find(n=>n.id===e.target).depth);
  let shown=[];while(shown.length<graph.nodes.length){shown=nextReveal(graph,shown);const seen=new Set();for(const id of shown){const n=graph.nodes.find(n=>n.id===id);assert.ok(!n.parent||seen.has(n.parent));seen.add(id);}}
});

test("baseline flow, solver noise and invalid numbers do not count as response",()=>{
  assert.equal(traceChange([120,120,120]).changed,false);
  assert.equal(traceChange([100,100.00000001]).changed,false);
  assert.equal(traceChange([0,NaN]),null);
  assert.equal(traceChange([0,1,0]).changed,true);
  assert.equal(traceChange([10,0]).delta,-10);
});

test("final model adds new nodes progressively without restarting the old path",()=>{
  const graph=buildEventGraph({nodes:[root,node("a","a"),node("b","b")],traces:{a:[0,1],b:[0,2]},edges:[edge("event:0","a"),edge("a","b")]});
  assert.deepEqual(reconcileReveal(graph,["event:0","a"]),["event:0","a"]);
  assert.deepEqual(nextReveal(graph,["event:0","a"]),["event:0","a","b"]);
  assert.deepEqual(reconcileReveal(graph,["b"]),[]);
});

test("independent events never gain a fabricated edge between them",()=>{
  const graph=buildEventGraph({nodes:[root,node("event:1",null,0),node("a","a"),node("b","b")],traces:{a:[0,1],b:[0,2]},edges:[edge("event:0","a"),edge("event:1","b")]});
  assert.deepEqual(graph.tree.map(e=>[e.source,e.target]),[["event:0","a"],["event:1","b"]]);
});

test("systemic summary includes only changing connected observables, not internal states",()=>{
  const run={nodes:[root,node("g","glucose"),node("output:glucose","glucose",4),node("static","hr"),node("output:hr","hr",4),node("island","island"),node("output:island","island",4),node("enzyme","enzyme")],
    traces:{glucose:[100,110,100],hr:[60,60,60],island:[0,2,0],enzyme:[0,5,0]},time:[0,30,60],
    edges:[edge("event:0","g"),edge("g","output:glucose"),edge("event:0","static"),edge("static","output:hr"),edge("island","output:island"),edge("event:0","enzyme")],
    metrics:[{id:"glucose",label:"Blood glucose",unit:"mg/dL",values:[100,110,100]},{id:"hr",label:"Heart rate",unit:"bpm",values:[60,60,60]},{id:"island",label:"Unconnected",unit:"1",values:[0,2,0]}]};
  const summary=systemChanges(run,buildEventGraph(run));
  assert.equal(summary.length,1);assert.equal(summary[0].nodeId,"g");assert.equal(summary[0].change.delta,10);assert.equal(summary[0].time,30);
});
