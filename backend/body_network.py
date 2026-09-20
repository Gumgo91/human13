"""Execute the body library in the existing shared ODE and coupling bus.

Input interpretation (the LLM) decides only the initial anchors: non-chemical
events enter through the channels declared here, and compound events enter as
their own transport/exposure states. All propagation beyond the anchors is
performed by the deterministic node-to-node target equations declared in
body_library — the LLM does not choose downstream paths.
"""
import math
import numpy as np
from .body_library import MODULES, SYSTEMS, body_catalog
from .expressions import compile_expression, variable, output_function
from .presentation import make_node

# Channel meanings the LLM consults when choosing initial anchors. Used by
# the prompt and the wire-schema enum; names not listed here are discarded.
CHANNEL_DESCRIPTIONS = {
    "stress": "psychological stress, anxiety, conflict, deadline pressure",
    "sleep_deprivation": "lost or shortened sleep, insomnia, all-nighter",
    "sleep": "sleeping or napping",
    "light": "bright light, screens, blue light exposure",
    "injury": "wound, trauma, surgery, burn, fracture, sting, bite",
    "infection": "pathogen exposure, cold/flu, fever, sepsis-like illness",
    "allergen": "pollen, dust, atopy, allergic exposure",
    "fasting": "skipped meal, fasting, caloric abstinence",
    "protein": "protein food or supplement intake",
    "fat": "fatty food or lipid intake",
    "fiber": "vegetables, fruit, whole grains, dietary fiber",
    "salt": "salty food, sodium chloride intake",
    "potassium": "potassium-rich food",
    "calcium": "dairy or calcium intake",
    "calcium_deficit": "calcium deficiency",
    "hypoxia": "altitude, diving, low-oxygen exposure",
    "breathing": "breathing exercise, hyperventilation, deep breaths",
    "heat": "sauna, hot bath, heat wave, heat exposure",
    "cold_exposure": "cold water, winter, hypothermia, cryotherapy",
    "hemorrhage": "blood loss, bleeding, blood donation",
    "transfusion": "blood transfusion",
    "meal": "a generic meal or snack of unstated composition",
    "bigmeal": "overeating, binge, buffet",
    "nausea": "nausea, motion sickness",
    "vomiting": "vomiting",
    "diarrhea": "diarrhea, loose stool",
    "constipation": "constipation",
    "smoking": "smoking, vaping, smoke inhalation",
    "pollution": "fine dust, smog, air pollution",
    "asthma": "asthma attack, wheezing",
    "relaxation": "meditation, yoga, massage, rest, laughter",
    "cognitive": "studying, gaming, focused mental work",
    "circadian_shift": "jet lag, night shift, staying up all night",
    "darkness": "darkness, lights out, blindfold",
    "uv": "sunlight, UV, tanning, outdoor sun exposure",
    "dehydration": "dehydration, thirst, low water intake",
    "probiotic": "probiotics, fermented food, yogurt, kimchi",
    "immobility": "prolonged sitting, long flight, bed rest, cast",
    "orthostasis": "prolonged standing",
    "apnea": "breath holding, apnea, snoring",
    "vaccination": "vaccine, immunization",
    "menstrual": "menstruation, period cramps",
    "menopause": "menopause, hot flashes",
    "pregnancy": "pregnancy",
    "lactation": "breastfeeding, lactation",
    "ache": "headache, migraine, muscle/joint/back pain",
    "oxygen_therapy": "supplemental oxygen, oxygen therapy",
}


# Default stimulus duration per channel (min). Delayed/accumulating channels
# get a longer window.
LONG_CHANNELS = {"pregnancy", "menopause", "lactation", "circadian_shift", "menstrual"}
NUTRIENT_CHANNELS = {"protein", "fat", "fiber", "salt", "potassium", "calcium", "meal", "bigmeal", "probiotic", "diarrhea"}

CHANNEL_NAMES = frozenset(CHANNEL_DESCRIPTIONS)


def route_body_inputs(events, quantity):
    """Map interpreted events to declared stimulus profiles.

    Initial anchors come ONLY from `event.body_channels`, which the LLM fills
    from the fixed channel enum. There is deliberately no keyword/regex
    fallback: if no interpretation is attached, the event gets no body anchor
    rather than an invented one. Negation is likewise the interpreter's job —
    a negated stimulus should simply not appear in `body_channels`.
    """
    profiles = []; notes = []
    for i, event in enumerate(events):
        # Chemical PK/target effects must come from compound evidence + the LLM;
        # a drug name never becomes a hardcoded target occupancy assertion.
        # The drug's plasma exposure still enters the network natively via the
        # drug_exposure primitive (detox/hepatic load), without naming targets.
        if event.kind == "chemical": continue
        channels = [c for c in (event.body_channels or []) if c in CHANNEL_NAMES]
        if "sleep_deprivation" in channels and "sleep" in channels: channels.remove("sleep")
        if "hemorrhage" in channels and "transfusion" in channels: channels.remove("hemorrhage")
        if not channels: continue
        for channel in channels:
            scale = 1.; rationale = "unit stimulus; dose-response not calibrated"
            if channel == "light" and event.quantity is not None and (event.unit or "").casefold() == "lux":
                scale = min(5., event.quantity / 1000); rationale = "exploratory normalization: 1000 lux as unit stimulus"
            if channel in {"protein", "fat", "fiber", "salt", "potassium"}:
                mass = quantity(event.quantity, event.unit, "g")
                if mass is not None:
                    reference = {"protein":20., "fat":20., "fiber":10., "salt":5., "potassium":2.}[channel]
                    scale = min(5., mass / reference); rationale = f"exploratory normalization: {reference:g} g as unit stimulus; not a plasma concentration"
            if channel == "calcium":
                mg = quantity(event.quantity, event.unit, "mg")
                if mg is not None:
                    scale = min(5., mg / 500); rationale = "exploratory normalization: 500 mg as unit stimulus; not a plasma concentration"
            if channel in {"hemorrhage", "transfusion"}:
                ml = quantity(event.quantity, event.unit, "mL")
                if ml is not None:
                    scale = min(5., ml / 500); rationale = "exploratory normalization: 500 mL as unit stimulus; not a hemodynamic calculation"
            if channel == "hypoxia" and event.quantity is not None and event.unit in {"m", "meter"}:
                scale = 1-math.exp(-min(20000, event.quantity)/8434); rationale = "exponential barometric approximation of altitude; not a saturation estimate"
            duration = event.duration_min or (480 if channel in LONG_CHANNELS else 120 if channel in NUTRIENT_CHANNELS else 30)
            profiles.append({"event_index":i, "channel":channel, "scale":scale, "start_min":event.start_min,
                             "duration_min":duration, "rationale":rationale})
            notes.append(f"{event.label}: {rationale}; stimulus window {duration:g} min.")
    return profiles, notes


def attach_body(sys, bus, events, settings, met_at, quantity, exposures=None, anchors=None):
    catalog = body_catalog()
    empty = {**catalog, "status":"disabled", "nodes":[], "edges":[], "events":[], "input_profiles":[],
             "assumptions":[], "observations":[], "checks":[], "couplings":[], "state_count":0}
    if not settings.allow_assumptions or not settings.enable_body_network: return empty
    if not all(k in sys.indices for k in ("hr", "sv", "svr", "fluid", "temperature")): return empty
    profiles, notes = route_body_inputs(events, quantity)
    for profile in profiles:
        sys.boundaries.update([profile["start_min"], profile["start_min"]+profile["duration_min"]])
    def channel(name, t):
        return min(5., sum(p["scale"] for p in profiles if p["channel"]==name and p["start_min"]<=t<p["start_min"]+p["duration_min"]))
    for m in MODULES: sys.state("body:"+m["id"], m["label"], "dimensionless", 0, "body_network:"+m["system"], False)
    indices = {m["id"]:sys.indices["body:"+m["id"]] for m in MODULES}
    symbols = {id:variable("dimensionless", lambda t,y,j=j:float(y[j]), [id]) for id,j in indices.items()}
    primitive_nodes = {"work":"energy", "pressure":"cardiovascular", "cardiac_output":"cardiovascular",
                       "hydration":"fluid", "hypovolemia":"fluid", "hypervolemia":"fluid",
                       "temperature_signal":"temperature", "cold":"temperature", "heat_signal":"temperature",
                       "feeding":"liver:meal", "glucose_signal":"glucose", "insulin_signal":"insulin",
                       "hypoglycemia":"glucose", "hyperglycemia":"glucose"}
    exposures = exposures or {}
    chem_exposures = [fn for i,(fn,_node) in exposures.items() if i < len(events) and events[i].kind == "chemical"]
    def current_output(t,y): return sys.value("hr",t,y)*sys.value("sv",t,y)/1000
    def hydration(t,y): return sys.value("fluid",t,y)/(settings.body_mass_kg*20)
    primitives = {
        "work": lambda t,y:max(0,met_at(t)-1)/6,
        "pressure": lambda t,y:(current_output(t,y)*sys.value("svr",t,y)+3)/(72*70/1000*17+3)-1,
        "cardiac_output": lambda t,y:current_output(t,y)/(72*70/1000)-1,
        # Normalize fluid deviation by a circulation-like scale (body mass x
        # 20 mL), not by total body water.
        "hydration": hydration,
        "hypovolemia": lambda t,y:max(0,-hydration(t,y)),
        "hypervolemia": lambda t,y:max(0,hydration(t,y)),
        "temperature_signal": lambda t,y:sys.value("temperature",t,y)-37,
        "cold": lambda t,y:max(0,37-sys.value("temperature",t,y)),
        "heat_signal": lambda t,y:max(0,sys.value("temperature",t,y)-37),
        "glucose_signal": lambda t,y:(sys.value("glucose",t,y)-100)/100,
        "hypoglycemia": lambda t,y:max(0,(100-sys.value("glucose",t,y))/50),
        "hyperglycemia": lambda t,y:max(0,(sys.value("glucose",t,y)-100)/100),
        "insulin_signal": lambda t,y:sys.value("insulin",t,y)/9.75-1,
        "feeding": lambda t,y:(sys.value("flux:absorbed_energy",t,y)/(2+sys.value("flux:absorbed_energy",t,y)) if "flux:absorbed_energy" in sys.observables else 0)+0.2*(channel("protein",t)+channel("fat",t)+channel("meal",t)+channel("bigmeal",t)+channel("fiber",t)),
        # Max normalized (0-1) compound plasma exposure. Even without LLM
        # target calls, a drug's presence propagates to detox/hepatic/mucosa
        # pathways.
        "drug_exposure": lambda t,y:max([min(1.,max(0.,f(t,y))) for f in chem_exposures]+[0.]),
    }
    for name,fn in primitives.items(): symbols[name] = variable("dimensionless",fn,[name])
    for name in CHANNEL_NAMES: symbols[name] = variable("dimensionless",lambda t,y,name=name:channel(name,t),[name])
    # Drug-target static matching: when a looked-up target matches a declared
    # anchor, the drug's plasma-exposure signal is added to that module's
    # target equation (rx_<module> variable).
    anchors = [a for a in (anchors or []) if a.get("event_index") in exposures]
    rx_terms = {}
    for a in anchors:
        if a["kind"] == "module" and a["id"] in indices:
            rx_terms.setdefault(a["id"], []).append(a)
    PLASMA_L = 3.0  # assumed adult plasma volume (~3 L)

    def anchor_signal(a, t, y):
        """Hill occupancy C/(C+K) when affinity exists; otherwise normalized
        exposure x selectivity weight."""
        if a.get("affinity_m") and a.get("plasma_key") and a.get("mw"):
            mg = sys.value(a["plasma_key"], t, y)
            vd = a.get("vd_l") or PLASMA_L
            occ = mg / (mg + vd * 1000. * a["mw"] * a["affinity_m"]) if mg > 0 else 0.
            return a["sign"] * a["gain"] * occ
        fn = exposures[a["event_index"]][0]
        return a["sign"] * a["gain"] * a.get("weight", 1.) * min(1., max(0., fn(t, y)))

    for mid, alist in rx_terms.items():
        def rx(t, y, alist=alist):
            return min(3., max(-3., sum(anchor_signal(a, t, y) for a in alist)))
        symbols["rx_" + mid] = variable("dimensionless", rx, ["rx_" + mid])
    from .coupling import PORTS as _PORTS
    for a in anchors:
        if a["kind"] == "port" and a["id"] in bus.available:
            bus.native_effects.setdefault(a["id"], []).append(
                lambda t, y, a=a: anchor_signal(a, t, y))
    nodes=[]; edges=[]; labels={id:label for id,label,_ in SYSTEMS}; compiled=[]; dep_graph={}
    available_nodes=set(primitive_nodes.values())
    for m in MODULES:
        # Declared target equation plus any drug-target signal wired to this module.
        expr_src = m["target"] + (f"+rx_{m['id']}" if m["id"] in rx_terms else "")
        expr=compile_expression(expr_src,symbols);target=output_function(expr,"dimensionless")
        i=indices[m["id"]];port="body_"+m["id"];bus.expose(port)
        # Smooth bounded target avoids exploding positive feedback while preserving
        # signs; it is a modeling assumption with declared limits, not a safety limit.
        def derivative(t,y,dy,i=i,target=target,tau=m["tau_min"],port=port):
            signal=target(t,y)+bus.signal(port,t,y)
            dy[i]+=(3*math.tanh(signal/3)-y[i])/tau
        sys.process("body:"+m["id"], derivative)
        def flux(t,y,i=i,target=target,tau=m["tau_min"],port=port):
            return (3*math.tanh((target(t,y)+bus.signal(port,t,y))/3)-y[i])/tau
        sys.observe("body:rate:"+m["id"],"1/min",flux)
        compiled.append((m,expr,target))
        dep_graph[m["id"]]=sorted(expr.dependencies)
        equations=[f"target = {expr_src} + LLM_signal",f"d({m['id']})/dt = [3·tanh(target/3) − {m['id']}] / {m['tau_min']} min"]
        node=make_node("body:"+m["id"],m["label"],labels[m["system"]],3,"assumption",
             "Relative functional deviation from resting state 0. Inputs, other systems' states, and any LLM-added signal are integrated in the same integrator.",
             state_key="body:"+m["id"], unit="relative deviation", cell_types=m["cells"], equations=equations,
             process_id="body:"+m["id"], body_system=m["system"],
             parameters=[{"name":"relaxation_time","value":m["tau_min"],"unit":"min","status":"assumption"},
                         {"name":"target_bound","value":3,"unit":"dimensionless","status":"assumption"}],
             sources=[m["source"]] if m["source"] else [],
             limitations=["The equation's connectivity and all response gains/time constants are uncalibrated reduced-form assumptions — not actual concentrations, cell counts, or binding rates.",
                          "No original equations/parameters were ported from a reference engine, and no validation results are inherited.",
                          "Reproductive, immune, and bone time constants do not guarantee real clinical onset times."])
        nodes.append(node)
        for dep in sorted(expr.dependencies):
            sources=[]
            if dep in indices:sources=["body:"+dep]
            elif dep in CHANNEL_NAMES:sources=[f"event:{p['event_index']}" for p in profiles if p["channel"]==dep]
            elif dep.startswith("rx_"):continue
            elif dep=="drug_exposure":
                sources=[node for i,(_fn,node) in exposures.items() if i < len(events) and events[i].kind=="chemical"]
            else:
                if dep=="feeding" and "flux:absorbed_energy" not in sys.observables:
                    sources=[f"event:{p['event_index']}" for p in profiles if p["channel"] in {"fat","protein","meal","bigmeal","fiber"} and p["scale"]>0]
                else:sources=[primitive_nodes[dep]]
            for source in sources:
                if source!="body:"+m["id"]:edges.append({"source":source,"target":"body:"+m["id"],"kind":"regulation","label":dep,"flux_key":"body:rate:"+m["id"],"unit":"1/min"})
    # Graph record of drug-target anchors — from the exposure state (plasma
    # compartment) to the wired node.
    for a in anchors:
        source = exposures[a["event_index"]][1]
        target = "body:" + a["id"] if a["kind"] == "module" else _PORTS.get(a["id"], {}).get("node")
        if target:
            edges.append({"source": source, "target": target, "kind": "pharmacology",
                          "label": f"drug target: {a['target']} ({a['action'] or 'direction unknown'})"})
        ev = events[a["event_index"]]
        notes.append(f"{ev.label}: target record '{a['target']}' → {a['id']} "
                     f"({'activation' if a['sign']>0 else 'inhibition'} assumed, {a['note']}). "
                     "Static matching, not an occupancy/concentration calculation.")
    # Native process controls are distinct from LLM controls: switching off LLM
    # coupling does not accidentally switch off physiological feedback.
    connections = [
        ("heart_rate_drive","0.20*sympathetic-0.12*parasympathetic+0.08*catecholamine+0.05*thyroid+0.05*pyrogen+0.04*hypovolemia+0.05*orthostatic_stress"),
        ("vascular_resistance","0.16*sympathetic+0.12*angiotensin-0.1*tnf+0.10*endothelin-0.10*nitric_oxide+0.05*aldosterone"),
        ("hepatic_production","0.12*glycogenolysis+0.12*gluconeogenesis+0.05*cortisol+0.05*insulin_resistance+0.06*hypoglycemia"),
        ("glut4_capacity","0.08*adiponectin-0.10*tnf-0.06*insulin_resistance+0.05*ampk"),
        ("insulin_secretion","-0.08*sympathetic+0.04*hyperglycemia+0.03*satiety"),
        ("gastric_emptying","0.05*parasympathetic-0.10*sympathetic-0.06*emetic_drive-0.05*amylin-0.04*stress"),
        ("thermogenesis","0.10*thyroid+0.10*il6+0.15*shivering+0.08*browning+0.05*bigmeal"),
        ("temperature_setpoint","0.4*il6+0.5*pyrogen"),
        ("renal_filtration","0.20*gfr+0.10*renal_bloodflow+0.06*anp"),
        ("water_clearance","-0.20*aqp2+0.10*gfr+0.08*anp-0.06*angiotensin"),
    ]
    from .coupling import PORTS
    couplings=[]
    for port,formula in connections:
        if port not in bus.available:continue
        expr=compile_expression(formula,symbols);fn=output_function(expr,"dimensionless")
        bus.native_effects.setdefault(port,[]).append(fn)
        couplings.append({"port":port,"expression":formula})
        for dep in sorted(expr.dependencies):
            if dep in indices:
                edges.append({"source":"body:"+dep,"target":PORTS[port]["node"],"kind":"coupling","label":"shared-equation control"})
    # Resting zero-input state must be an equilibrium when native signals rest.
    # Unit checks above cover all declared target expressions.
    observations=[]
    observe_ids={m["id"] for m in MODULES}
    for m in MODULES:
        if m["id"] in observe_ids:observations.append({"id":"body:"+m["id"],"label":m["label"],"unit":"relative deviation","baseline":0,"system":m["system"]})
    sys.observe("body:cardiac_output","L/min",current_output)
    observations.insert(0,{"id":"body:cardiac_output","label":"Cardiac output","unit":"L/min","baseline":72*70/1000,"system":"circulation"})
    return {**catalog,"status":"active","nodes":nodes,"edges":edges,
            "events":sorted({p["event_index"] for p in profiles}), "input_profiles":profiles,
            "assumptions":notes+["Whole-body system states are an uncalibrated relative-response model. Equation constants are exploratory coefficients and are not converted to absolute hormone/electrolyte concentrations."],
            "observations":observations,"checks":["all target equations dimension-checked","single shared ODE","explicit time boundaries","relative states bounded to ±3","propagation beyond initial anchors is deterministic over the declared graph"],
            "couplings":couplings,"state_count":len(MODULES),
            "drug_anchors":[{k:a.get(k) for k in ("event_index","kind","id","sign","target","action","note","affinity_m","affinity_note","target_key","affinity_evidence","target_chembl_id","weight","basis","sources")} for a in anchors],
            "dep_graph":dep_graph,
            # Outcome nodes the user claims to have observed — summarize
            # backtraces a path to them.
            "outcome_claims":[{"event_index":i,"nodes":list(e.outcome_nodes)}
                              for i,e in enumerate(events) if getattr(e,"outcome_nodes",None)]}


def summarize_body(body,traces,nodes,edges):
    """Atlas reports both registered processes and this run's actual response."""
    if body["status"]!="active":return {k:v for k,v in body.items() if k not in {"nodes","edges","assumptions","dep_graph"}}
    for system in body["systems"]:
        for m in system["modules"]:
            values=traces["body:"+m["id"]]
            m["max_deviation"]=float(np.max(np.abs(values)))
            if m["max_deviation"]>3+1e-5:raise ValueError("body relative-state bound violated: "+m["id"])
            m["node_id"]="body:"+m["id"]
        system["changed"]=sum(m["max_deviation"]>1e-5 for m in system["modules"])
    valid={n["id"] for n in nodes}
    body["edges"]=[e for e in body["edges"] if e["source"] in valid and e["target"] in valid]
    body["pathways"]=body_pathways(body,traces)
    body["outcome_paths"]=body_outcome_paths(body,traces)
    return {k:v for k,v in body.items() if k not in {"nodes","edges","assumptions","dep_graph","outcome_claims"}}


def body_pathways(body,traces,max_depth=8,threshold=0.02):
    """Backtrace a causal chain from each responding node through its most
    strongly driven parent.

    A 'dominant path' approximation: every term in a target equation
    contributes, so a single chain is each node's dominant upstream — not the
    full interaction graph.
    """
    deps=body.get("dep_graph",{})
    if not deps:return []
    module_ids={m["id"] for m in MODULES}
    labels={m["id"]:m["label"] for m in MODULES}
    peaks={m["id"]:float(np.max(np.abs(traces["body:"+m["id"]]))) for m in MODULES}
    signs={m["id"]:1. if np.max(traces["body:"+m["id"]])>=-np.min(traces["body:"+m["id"]]) else -1. for m in MODULES}
    def entry(dep):
        if dep in module_ids:
            return {"id":"body:"+dep,"label":labels[dep],"peak":round(peaks[dep],4),"direction":"increase" if signs[dep]>0 else "decrease"}
        if dep.startswith("rx_"):
            return {"id":dep,"label":"drug target signal"}
        return {"id":dep,"label":dep}
    def main_chain(mid):
        chain=[entry(mid)];seen={mid};cur=mid
        while len(chain)<max_depth:
            parents=[d for d in deps.get(cur,[]) if d in module_ids and d not in seen and peaks[d]>1e-4]
            if not parents:
                # Attach a non-body root for context. Drug signals (rx_) outrank
                # channels — the actual driver is the drug in that case.
                roots=[d for d in deps.get(cur,[]) if d not in module_ids]
                roots.sort(key=lambda d:(not d.startswith("rx_"),d))
                if roots:chain.append(entry(roots[0]))
                break
            cur=max(parents,key=lambda d:peaks[d]);seen.add(cur);chain.append(entry(cur))
        return list(reversed(chain))
    seen_leaves=set();chains=[]
    # Build chains from the strongest responders first; skip nodes already
    # inside a longer chain.
    for mid in sorted(module_ids,key=lambda k:-peaks[k]):
        if peaks[mid]<threshold or mid in seen_leaves:continue
        chain=main_chain(mid)
        if len(chain)>=3:
            chains.append(chain)
            for e in chain:
                if e["id"].startswith("body:") and peaks.get(e["id"][5:],0)>threshold:
                    seen_leaves.add(e["id"][5:])
    chains.sort(key=lambda c:-len(c))
    return chains[:20]


def body_outcome_paths(body,traces,max_depth=8,threshold=0.02):
    """Intermediate path backtraced from a user-claimed outcome node toward
    the event.

    If the claimed outcome does not actually appear in the model, keep
    found=False — we do not invent a causal path.
    """
    deps=body.get("dep_graph",{})
    claims=body.get("outcome_claims") or []
    if not deps or not claims:return []
    module_ids={m["id"] for m in MODULES}
    labels={m["id"]:m["label"] for m in MODULES}
    peaks={m["id"]:float(np.max(np.abs(traces["body:"+m["id"]]))) for m in MODULES}
    signs={m["id"]:1. if np.max(traces["body:"+m["id"]])>=-np.min(traces["body:"+m["id"]]) else -1. for m in MODULES}
    def entry(dep):
        if dep in module_ids:
            return {"id":"body:"+dep,"label":labels[dep],"peak":round(peaks[dep],4),"direction":"increase" if signs[dep]>0 else "decrease"}
        if dep.startswith("rx_"):
            return {"id":dep,"label":"drug target signal"}
        return {"id":dep,"label":dep}
    def main_chain(mid):
        chain=[entry(mid)];seen={mid};cur=mid
        while len(chain)<max_depth:
            parents=[d for d in deps.get(cur,[]) if d in module_ids and d not in seen and peaks[d]>1e-4]
            if not parents:
                roots=[d for d in deps.get(cur,[]) if d not in module_ids]
                roots.sort(key=lambda d:(not d.startswith("rx_"),d))
                if roots:chain.append(entry(roots[0]))
                break
            cur=max(parents,key=lambda d:peaks[d]);seen.add(cur);chain.append(entry(cur))
        return list(reversed(chain))
    out=[]
    for claim in claims:
        for nid in claim["nodes"]:
            if nid not in module_ids:continue
            chain=main_chain(nid)
            responded=peaks[nid]>threshold
            out.append({"event_index":claim["event_index"],"node":"body:"+nid,"label":labels[nid],
                        "responded":responded,"peak":round(peaks[nid],4),
                        "found":responded and len(chain)>=2,"chain":chain})
    return out
