"""Compile arbitrary declared physiological ODE graphs into the shared solver.

Domain names are data. Native physiology remains owned by its source modules;
extensions read those states and modulate their exposed processes. Generated
states may interact, branch, feed back, react, and define additional outputs.
"""
import hashlib
import json
import math
import numpy as np
from .expressions import ModelCompileError, compile_expression, variable, output_function, unit_info
from .presentation import make_node
from .coupling import PORTS
from .equation_contracts import ModelObservation


def catalog_from_result(result):
    original=result["model_context"]
    # Avoid repeating 79 verbose system descriptions and rates in each LLM call.
    # All states and control ports remain available; no biological scope is removed.
    context={"states":[{k:s[k] for k in ("key","label","unit","initial")} for s in original["states"]],
             "fluxes":[s for s in original["fluxes"] if not s["key"].startswith("body:rate:")],
             "processes":[{k:p[k] for k in ("id","label","event_indices")} for p in original["processes"]],
             "ports":{k:{f:v[f] for f in ("node","label","mode") if f in v} for k,v in original["ports"].items()}}
    return {**context,"reference_scenarios":original.get("reference_scenarios",[]),"language":{
        "expressions":"Arithmetic + - * / **, min/max/exp/log/tanh/abs. NO Python code. All parameters have units. time and minute have time units; event_N is dimensionless exposure; dose_N has original dose units when known.",
        "states":"Any new entity, compartment and scale; one shared state per quantity. Native state ownership cannot be overwritten. Use dimensionless deviation from 0 or relative level from 1 unless an absolute quantity can be justified.",
        "processes":"Each declared rate contributes coefficient*rate to each changes.state. Rate units must match state/min. One relaxation: (gain*event_0-x)/tau with tau in min. Mass action: k*a*b; saturable transport: vmax*a/(km+a); feedback: (gain*x/(half+x)-y)/tau. Declare every parameter with units and ranges.",
        "bindings":"Read ONLY native source keys listed here. States/parameters/event_N/dose_N are already formula variables; do not bind them.",
        "observations":"Declare any meaningful additional output with expression and unit, connected to generated states. A relative activity index must say relative index. Existing systemic outputs already exist.",
        "couplings":"expression is a dimensionless log multiplier on a native port: default ports use parameter *= exp(clamp(sum(expressions),-1.2,1.2)); ports with mode=additive_target add the bounded dimensionless signal to the native relative-state target. New-state to new-state interactions use rate expressions and have NO fixed port list.",
        "validation":"All states need a process or impulse, every event modeled in this extension must reach an observation or native coupling; no disconnected decoration. Each process lists causal event_indices; no process runs before its earliest event.",
        "impulses":"Optional initial material insertion at an event; amount uses dose_N or constant parameters and must match state units. Do not add a native drug dose a second time.",
        "invariants":"Optional weighted conservation laws. Declared laws are checked structurally AND over the trajectory including impulses.",
    }}


def model_fingerprint(model):
    return hashlib.sha256(json.dumps(model.model_dump(),sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def compile_model(sys,bus,events,model,nodes,edges,exposures,settings,sources=()):
    result={"status":"absent","states":0,"processes":0,"observations":[],"parameters":[],"events":[],"checks":[],"hash":None}
    if not model:return result
    result["hash"]=model_fingerprint(model)
    if not model.states:
        if model.processes or model.observations or model.couplings:raise ModelCompileError("Generated processes were declared without states.")
        return result
    if not bus.enabled():
        result.update(status="disabled",reason="Exploratory assumptions or LLM numeric coupling is disabled.")
        return result
    evaluation={"baseline":False}
    symbols={"minute":variable("min",lambda t,y:1),"time":variable("min",lambda t,y:t,["time"])}
    node_for={};event_starts={};indices={};params={};prepared=[];compiled_obs=[];compiled_couplings=[];impulses=[]
    state_by={s.id:s for s in model.states}
    if len(state_by)!=len(model.states):raise ModelCompileError("Duplicate generated-state ID")
    semantic={}
    for s in model.states:
        identity=(s.entity.casefold().strip(),s.compartment.casefold().strip())
        if identity in semantic:raise ModelCompileError(f"Duplicate state of the same quantity and compartment: {semantic[identity]} / {s.id} ({s.entity}, {s.compartment}). Sum per-event contributions into one shared state.")
        semantic[identity]=s.id
        if not s.lower<=s.initial<=s.upper or s.lower==s.upper:raise ModelCompileError(f"Initial value / state range error: {s.id}")
        # Construct indices first, but do not mutate the solver until validation succeeds.
        index=len(sys.states)+len(indices);indices[s.id]=index
        if s.id in symbols:raise ModelCompileError(f"Reserved name: {s.id}")
        symbols[s.id]=variable(s.unit,lambda t,y,i=index:float(y[i]),[s.id]);node_for[s.id]="model:"+s.id
    def add_symbol(key,value):
        if key in symbols:raise ModelCompileError(f"Duplicate formula name: {key}")
        symbols[key]=value
    for i,event in enumerate(events):
        duration=event.duration_min or 30.
        event_starts[i]=event.start_min
        raw,source=exposures.get(i,(lambda t,y,e=event,d=duration:float(e.start_min<=t<e.start_min+d),f"event:{i}"))
        add_symbol(f"event_{i}",variable("dimensionless",lambda t,y,fn=raw,e=event:fn(t,y) if t>=e.start_min and not evaluation["baseline"] else 0,[f"event:{i}"]))
        node_for[f"event:{i}"]=source
        if event.quantity is not None and event.unit:
            try:add_symbol(f"dose_{i}",variable(event.unit,lambda t,y,v=event.quantity:v))
            except ModelCompileError:pass
    for p in model.parameters:
        if not p.low<=p.value<=p.high:raise ModelCompileError(f"Parameter range error: {p.id}")
        value=settings.model_parameters.get(p.id,p.value)
        if not math.isfinite(value) or not p.low<=value<=p.high:raise ModelCompileError(f"{p.id}: override value is outside the declared range.")
        add_symbol(p.id,variable(p.unit,lambda t,y,v=value:v))
        params[p.id]={**p.model_dump(),"name":p.label,"value":value,"status":"user_input" if p.id in settings.model_parameters else "llm_estimate"}
    unknown_overrides=set(settings.model_parameters)-set(params)
    if unknown_overrides:raise ModelCompileError("Override parameter not in this model: "+", ".join(sorted(unknown_overrides)))
    for b in model.bindings:
        if b.source in sys.indices:unit=sys.states[sys.indices[b.source]].unit
        elif b.source in sys.observables:unit=sys.observables[b.source][0]
        else:raise ModelCompileError(f"Unknown existing state/observation: {b.source}. bindings are read-only over native outputs. Do not put generated states/parameters in bindings; define needed gains in parameters.")
        add_symbol(b.id,variable(unit,lambda t,y,k=b.source:sys.value(k,t,y),["native:"+b.source]))
        node_for["native:"+b.source]=next((n["id"] for n in nodes if n.get("state_key")==b.source),None)
    # Report independent formula errors together so one provider repair can fix
    # all unit mistakes; no equation is accepted or rewritten by this preflight.
    formula_errors=[]
    formulas=[(p.id,p.rate,[(state_by[c.state].unit,True) for c in p.changes if c.state in state_by]) for p in model.processes]
    formulas += [(o.id,o.expression,[(o.unit,False)]) for o in model.observations]
    formulas += [(c.port,c.expression,[("dimensionless",False)]) for c in model.couplings]
    for label,formula,expected in formulas:
        try:
            checked=compile_expression(formula,symbols)
            for unit,is_rate in expected:output_function(checked,unit,rate=is_rate)
        except ModelCompileError as error:formula_errors.append(label+": "+str(error))
    if formula_errors:raise ModelCompileError("Formula check failed:\n"+"\n".join(formula_errors[:8]))
    seen=set();assigned=set();adjacency={};causes={}
    def arc(a,b):adjacency.setdefault(a,set()).add(b)
    # Connect actual native event paths to bound shared states. A source model
    # merely existing is insufficient: there must be a path from that event.
    forward={}
    for edge in edges:
        if edge["kind"] not in {"knowledge","hypothesis"}:forward.setdefault(edge["source"],set()).add(edge["target"])
    for i in event_starts:
        pending=[f"event:{i}"];reachable=set()
        while pending:
            key=pending.pop()
            if key in reachable:continue
            reachable.add(key);pending.extend(forward.get(key,()))
        for binding in model.bindings:
            native="native:"+binding.source
            if node_for.get(native) in reachable:arc(f"event:{i}",native)
    for p in model.processes:
        if p.id in seen:raise ModelCompileError("Duplicate process ID: "+p.id)
        seen.add(p.id)
        if any(i not in event_starts for i in p.event_indices):raise ModelCompileError("Process references a nonexistent event: "+p.id)
        expr=compile_expression(p.rate,symbols);changes=[]
        if len({c.state for c in p.changes})!=len(p.changes):raise ModelCompileError("Duplicate state coefficient inside one reaction")
        for c in p.changes:
            if c.state not in indices or c.coefficient==0:raise ModelCompileError("A reaction may only contribute a nonzero coefficient to newly declared states.")
            fn=output_function(expr,state_by[c.state].unit,rate=True)
            changes.append((indices[c.state],c.coefficient,fn));assigned.add(c.state)
            for dep in expr.dependencies:arc(dep,c.state)
        # Causality includes declared start gates, but a gate alone is not
        # evidence of a physiologically driven output. Track actual references.
        prepared.append((p,expr,changes,min(event_starts[i] for i in p.event_indices)))
        causes[p.id]=set(p.event_indices)
    initial=np.r_[[s.initial for s in sys.states],[s.initial for s in model.states]]
    for impulse in model.impulses:
        if impulse.event_index not in event_starts or impulse.state not in indices:raise ModelCompileError("Invalid event impulse")
        expr=compile_expression(impulse.amount,symbols)
        if expr.dependencies:raise ModelCompileError("Event impulse amounts may only use constants or explicitly stated doses.")
        amount=output_function(expr,state_by[impulse.state].unit)(0,initial)
        if not math.isfinite(amount) or amount<0:raise ModelCompileError("Invalid impulse amount")
        impulses.append((event_starts[impulse.event_index],"model:"+impulse.state,amount))
        assigned.add(impulse.state);arc(f"event:{impulse.event_index}",impulse.state)
    if set(indices)-assigned:raise ModelCompileError("State without a computing process: "+", ".join(set(indices)-assigned))
    endpoints=set();obs_ids=set()
    for o in model.observations:
        if o.id in obs_ids:raise ModelCompileError("Duplicate observation ID")
        obs_ids.add(o.id);expr=compile_expression(o.expression,symbols)
        if not expr.dependencies.intersection(indices):raise ModelCompileError("A generated observation must actually read a generated state.")
        fn=output_function(expr,o.unit);compiled_obs.append((o,expr,fn));endpoints.update(expr.dependencies)
    for c in model.couplings:
        if c.port not in bus.available or PORTS[c.port]["node"] not in {n["id"] for n in nodes}:raise ModelCompileError("Coupling point absent from this run: "+c.port)
        expr=compile_expression(c.expression,symbols)
        if not expr.dependencies.intersection(indices):raise ModelCompileError("An existing-process coupling must actually read a generated state.")
        fn=output_function(expr,"dimensionless");compiled_couplings.append((c,expr,fn));endpoints.update(expr.dependencies)
        if abs(fn(0,initial))>1e-10:raise ModelCompileError("The log multiplier passed to an existing process must be 0 at the initial state: "+c.port+". Subtract the initial value so a relative level delivers a change amount.")
    def reaches(start):
        pending=[start];visited=set()
        while pending:
            key=pending.pop()
            if key in endpoints:return True
            if key in visited:continue
            visited.add(key);pending.extend(adjacency.get(key,()))
        return False
    # An otherwise executable terminal physiological quantity is itself a valid
    # observation. Automatically project terminal states instead of discarding
    # an entire model because the planner omitted its output card. This changes
    # presentation/observation only; it never invents a physiological connection.
    auto_observed=[]
    while True:
        dead=[s for s in indices if not reaches(s)]
        if not dead:break
        terminal=next((s for s in dead if not (adjacency.get(s,set())&set(dead)-{s})),dead[-1])
        s=state_by[terminal];id="auto_"+s.id[:42]
        suffix=0
        while id in obs_ids:
            suffix+=1;id="auto_"+s.id[:35]+"_"+str(suffix)
        obs_ids.add(id)
        label=s.label+(" - relative index" if not unit_info(s.unit)[0].dimensionality else "")
        o=ModelObservation(id=id,label=label[:100],expression=s.id,unit=s.unit,description="Automatically observed terminal physiological state. "+s.description[:330])
        expr=compile_expression(s.id,symbols);compiled_obs.append((o,expr,output_function(expr,s.unit)))
        endpoints.add(s.id);auto_observed.append(s.id)
    connected_events=[i for i in event_starts if reaches(f"event:{i}")]
    used_events=set().union(*(set(p.event_indices) for p in model.processes),*(set([i.event_index]) for i in model.impulses))
    if used_events-set(connected_events):raise ModelCompileError("A declared event never reaches an observation through actual equation drive: "+str(sorted(used_events-set(connected_events))))
    # Conservation is checked before execution; a missing sink cannot pass by
    # happening to have zero initial substrate.
    invariants=[]
    for law in model.invariants:
        weights={w.state:w.coefficient for w in law.weights}
        if len(weights)!=len(law.weights) or any(k not in indices for k in weights):raise ModelCompileError("Invalid conservation-ledger state")
        first=state_by[next(iter(weights))];base,_,offset=unit_info(first.unit)
        if offset:raise ModelCompileError("Offset units cannot be used in a conservation ledger.")
        scales={}
        for key,weight in weights.items():
            unit,scale,shift=unit_info(state_by[key].unit)
            if unit.dimensionality!=base.dimensionality or shift:raise ModelCompileError("Conservation-ledger unit mismatch")
            scales[key]=scale*weight
        for p,expr,_,_ in prepared:
            total=sum(c.coefficient*scales.get(c.state,0)/unit_info(state_by[c.state].unit)[1] for c in p.changes)
            if abs(total)>1e-9:raise ModelCompileError(f"Reaction violating the conservation law: {law.label} / {p.id}")
        invariants.append({"label":law.label,"scales":scales,"unit":str(base)})
    # Probe equations at initial state. Singular/overflowing proposals never
    # enter the shared ODE. Real trajectory bounds are checked after solving.
    try:
        evaluation["baseline"]=True
        resting=np.zeros(len(initial))
        for p,expr,changes,start in prepared:
            for index,coefficient,fn in changes:resting[index]+=coefficient*fn(0,initial)
        drifting=[s.id for s in model.states if not unit_info(s.unit)[0].dimensionality and abs(resting[indices[s.id]])>1e-8]
        if drifting:raise ModelCompileError("A relative state drifts from its initial baseline with zero stimulus: "+", ".join(drifting)+". Fix the zero-stimulus equilibrium. Relative changes use initial=0 with recovery target 0; a relative level with initial=1 must include 1 in its target.")
        evaluation["baseline"]=False
        for p,expr,_,start in prepared:
            if not math.isfinite(expr.fn(start,initial)):raise ValueError()
        for _,_,fn in compiled_obs:
            if not math.isfinite(fn(0,initial)):raise ValueError()
    except ModelCompileError:raise
    except (ValueError,OverflowError,ZeroDivisionError,TypeError) as error:raise ModelCompileError("A formula is singular or non-finite at the initial state.") from error
    finally:evaluation["baseline"]=False
    # Commit fully validated model, with a single state owner per generated key.
    for s in model.states:
        sys.state("model:"+s.id,s.label,s.unit,s.initial,"equation_model",s.lower>=0)
        related=[p for p,expr,_,_ in prepared if any(c.state==s.id for c in p.changes)]
        source_ids={sid for p in related for sid in p.source_ids}
        used_params={name for name in params if any(name in p.rate.replace("("," ").replace(")"," ") or name in p.rate for p in related)}
        nodes.append(make_node("model:"+s.id,s.label,s.compartment,2 if s.level in {"molecule","system"} else 3,"llm_surrogate",s.description,state_key="model:"+s.id,unit=s.unit,process_id="model:"+s.id,
            cell_types=s.cell_types,equations=[f"d({s.id})/dt += "+" + ".join(f"{c.coefficient:g}·({p.rate})" for c in p.changes if c.state==s.id) for p in related],parameters=list(params.values()) if len(model.states)==1 else [params[k] for k in sorted(used_params)],
            sources=[source.model_dump() for source in sources if source.id in source_ids],limitations=["Automatically constructed exploratory model. Before numerical and physiological calibration; not a quantitative reproduction model from literature.",*model.limitations],generated={"level":s.level,"entity":s.entity,"lower":s.lower,"upper":s.upper,"initial":s.initial}))
    for p,expr,changes,start in prepared:
        flux="model:flux:"+p.id
        # One scalar flux is reused across all stoichiometric contributions.
        # Evaluating equivalent closures is cheap and preserves unit factors.
        def derivative(t,y,dy,changes=changes,start=start):
            if t<start:return
            for index,coefficient,fn in changes:dy[index]+=coefficient*fn(t,y)
        sys.process("model:"+p.id,derivative)
        flux_unit=state_by[p.changes[0].state].unit+"/min"
        flux_fn=output_function(expr,state_by[p.changes[0].state].unit,rate=True)
        sys.observe(flux,flux_unit,lambda t,y,fn=flux_fn,start=start:fn(t,y) if t>=start else 0.)
        for c in p.changes:
            for dep in expr.dependencies:
                source=node_for.get(dep)
                if source and source!="model:"+c.state:edges.append({"source":source,"target":"model:"+c.state,"kind":"coupling","label":p.label,"flux_key":flux,"unit":flux_unit})
        for i in p.event_indices:sys.boundaries.update([events[i].start_min,events[i].start_min+(events[i].duration_min or 30)])
    for proposal,(start,key,amount) in zip(model.impulses,impulses):
        sys.impulse(start,key,amount)
        edges.append({"source":f"event:{proposal.event_index}","target":key,"kind":"input","label":f"{amount:g} {state_by[proposal.state].unit}"})
    for o,expr,fn in compiled_obs:
        key="model:obs:"+o.id;sys.observe(key,o.unit,fn)
        label=o.label+(" - relative index" if not unit_info(o.unit)[0].dimensionality and "relative" not in o.label else "")
        result["observations"].append({**o.model_dump(),"label":label,"key":key,"baseline":float(fn(0,initial)),"sources":[node_for[d] for d in sorted(expr.dependencies) if node_for.get(d)]})
    for ci,(c,expr,fn) in enumerate(compiled_couplings):
        bus.functional_effects.setdefault(c.port,[]).append(fn);bus.compiled.append(c.port)
        coupling_key=f"model:coupling:{ci}";sys.observe(coupling_key,"dimensionless",fn)
        for dep in sorted(expr.dependencies):
            if node_for.get(dep):edges.append({"source":node_for[dep],"target":PORTS[c.port]["node"],"kind":"coupling","label":c.description[:70],"flux_key":coupling_key,"unit":"dimensionless"})
    result.update(status="compiled",states=len(model.states),processes=len(model.processes),parameters=list(params.values()),events=connected_events,
                  couplings=[dict(c.model_dump(),trace_key=f"model:coupling:{i}") for i,c in enumerate(model.couplings)],invariants=invariants,impulses=impulses,
                  checks=["formula syntax", "unit conversion", "state ownership", "event->observation reachability", "initial formula finiteness", "reaction conservation"],
                  auto_observed=auto_observed,coverage=[{"event_index":i,"status":"modeled" if i in connected_events else "native_or_uncovered"} for i in event_starts])
    return result


def validate_trajectory(report,model,times,traces):
    if report["status"]!="compiled":return []
    for c in report["couplings"]:
        values=np.asarray(traces[c["trace_key"]]);direction=c.get("signal_direction")
        if (direction=="decrease" and values.max()>1e-8) or (direction=="increase" and values.min()<-1e-8):
            raise ModelCompileError(f"{c['port']}: declared signal_direction {direction} does not match the actual delivered signal sign. Check double negatives across coefficient and expression.")
    for s in model.states:
        values=np.asarray(traces["model:"+s.id]);tol=1e-5*max(1,abs(s.lower),abs(s.upper))
        if values.min()<s.lower-tol or values.max()>s.upper+tol:raise ModelCompileError(f"{s.id}: the integrated trajectory left the declared state range. Fix the equation or range.")
    balances=[]
    for law in report.get("invariants",[]):
        actual=sum(np.asarray(traces["model:"+k])*v for k,v in law["scales"].items())
        initial=sum(next(s.initial for s in model.states if s.id==k)*v for k,v in law["scales"].items())
        expected=np.full(len(times),initial)
        for t,key,amount in report["impulses"]:
            expected+=np.asarray(times>=t,dtype=float)*amount*law["scales"].get(key.removeprefix("model:"),0)
        error=float(np.max(np.abs(actual-expected)))
        if error>1e-6*max(1,float(np.max(np.abs(expected)))):raise ModelCompileError("Conservation check failed for generated reaction: "+law["label"])
        balances.append({"name":law["label"],"max_error":error,"unit":law["unit"],"passed":True,"keys":["model:"+k for k in law["scales"]]})
    report["checks"].extend(["full-trajectory range", "numeric finiteness"])
    return balances
