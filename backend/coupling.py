"""Compile LLM proposals onto the same ODE state and explicit model ports.

No executable text, direct biomarker assignments, or duplicate known pathway
fluxes. New mechanisms are bounded lag-response surrogates with visible gains.
"""
import math
from .presentation import make_node, METHOD_LABELS
from .body_library import BODY_PORTS

PORTS={
    "gastric_emptying":{"node":"digestion:gastric","label":"Gastric emptying rate"},
    "sucrase_capacity":{"node":"digestion:sucrase","label":"Sucrose hydrolysis capacity"},
    "sglt1_capacity":{"node":"transport:sglt1","label":"Intestinal SGLT1 transport capacity"},
    "glut2_capacity":{"node":"transport:glut2","label":"Intestinal GLUT2 transport capacity"},
    "glut5_capacity":{"node":"transport:glut5","label":"Intestinal GLUT5 transport capacity"},
    "insulin_secretion":{"node":"insulin","label":"Beta-cell insulin secretion capacity"},
    "glut4_capacity":{"node":"peripheral","label":"Peripheral GLUT4 utilization capacity"},
    "hepatic_production":{"node":"liver:production","label":"Hepatic glucose production"},
    "heart_rate_drive":{"node":"cardiovascular","label":"Heart-rate drive"},
    "vascular_resistance":{"node":"vascular","label":"Vascular resistance"},
    "thermogenesis":{"node":"temperature","label":"Heat generation"},
    "renal_reabsorption":{"node":"renal:sglt2","label":"Renal glucose reabsorption"},
    "renal_filtration":{"node":"renal:filtration","label":"Kidney Glomerulus Filtration"},
    "water_clearance":{"node":"fluid","label":"Removal of water above baseline"},
    "temperature_setpoint":{"node":"temperature","label":"Thermoregulatory setpoint change","mode":"additive_target"},
}
PORTS.update(BODY_PORTS)
MODULES={
    "sucrase":"digestion:sucrase","sglt1":"transport:sglt1","glut2":"transport:glut2","glut5":"transport:glut5",
    "nak_atpase":"transport:nak","gastric_emptying":"digestion:gastric","incretin":"incretin",
    "insulin_secretion":"insulin","glut4":"peripheral","hepatic_glucose":"liver:production",
    "renal_sglt2":"renal:sglt2","cardiovascular":"cardiovascular","thermoregulation":"temperature","fluid_balance":"fluid",
}
MODULES.update({key:value["node"] for key,value in BODY_PORTS.items()})


class CouplingBus:
    def __init__(self,sys,settings):
        self.sys=sys;self.settings=settings;self.effects={};self.functional_effects={};self.native_effects={};self.available=set();self.compiled=[]

    def expose(self,*ports):self.available.update(ports)

    def multiplier(self,port,t,y):
        total=sum(gain*y[index] for index,gain in self.effects.get(port,[]))+sum(fn(t,y) for fn in self.functional_effects.get(port,[]))
        native=sum(fn(t,y) for fn in self.native_effects.get(port,[]))
        return math.exp(max(-1.2,min(1.2,total*self.settings.llm_scale))+max(-1.2,min(1.2,native)))

    def signal(self,port,t,y):
        total=sum(gain*y[index] for index,gain in self.effects.get(port,[]))+sum(fn(t,y) for fn in self.functional_effects.get(port,[]))
        native=sum(fn(t,y) for fn in self.native_effects.get(port,[]))
        return max(-1.2,min(1.2,total*self.settings.llm_scale))+max(-1.2,min(1.2,native))

    def enabled(self):return self.settings.allow_assumptions and self.settings.enable_llm_coupling


def model_catalog():
    return {"reuse_modules":MODULES,"effect_ports":PORTS,"surrogate_law":"dz/dt=(u-z)/tau; parameter multiplier=exp(clamp(sum(gain*z)*llm_scale,-1.2,1.2))","drivers":["event_exposure","glucose_deviation","insulin_deviation","intestinal_uptake","activity_load"],"gain_range":[-.8,.8],"note":"Existing mechanisms must use reuse and must not receive an extra duplicate gain. New cross-scale hypotheses can use surrogate. All gains are uncalibrated LLM approximations, not experimental estimates."}


def prepare_expansion(expansion,events):
    if not expansion:return None
    prepared=expansion.model_copy(deep=True)
    for h in prepared.hypotheses:
        if h.event_index>=len(events):continue
        p=h.execution
        # A cardiac/insulin model's existence does not already encode a new
        # drug's effect. That effect needs an exposure-to-process connection.
        if p and p.mode=="reuse" and events[h.event_index].kind in {"chemical","other"}:
            if p.module_id in BODY_PORTS and events[h.event_index].kind!="chemical":continue
            if p.effect_port in PORTS and p.gain!=0 and p.gain_low<=p.gain<=p.gain_high:
                p.mode="surrogate";p.driver="event_exposure"
                p.rationale="No drive path for this event exists in the baseline equations; the proposed coefficients were wired as an exposure-response approximation. "+p.rationale[:290]
            else:p.mode="unresolved"
    return prepared


def compile_pathways(sys,bus,events,expansion,compounds,nodes,edges,exposures,met_at):
    report=[]
    if not expansion:return report
    lookup={n["id"]:n for n in nodes}
    aliases={"sglt1":"sglt1","glut2":"glut2","glut5":"glut5","glut4":"glut4","sucrase":"sucrase","수크라":"sucrase","Insulin":"insulin_secretion","glp-1":"incretin","glp1":"incretin"}
    for j,h in enumerate(expansion.hypotheses):
        if h.event_index>=len(events):continue
        p=h.execution;key=f"hypothesis:{j}";existing=[]
        if p and p.mode=="reuse" and p.module_id in MODULES:existing=[MODULES[p.module_id]]
        if not p and events[h.event_index].kind not in {"chemical","other"}:
            existing=list(dict.fromkeys(MODULES[value] for token,value in aliases.items() if token in h.target.casefold()))
        existing=[nid for nid in existing if nid in lookup]
        if existing:
            for nid in existing:
                n=lookup[nid];n.setdefault("linked_hypotheses",[]).append(h.model_dump())
                n["base_method"]=n.get("base_method",n["method"]);n["method"]="hybrid";n["method_label"]=METHOD_LABELS["hybrid"]
            report.append({"hypothesis":j,"status":"reused","nodes":existing,"reason":"wired to the same execution process; no duplicate flux"})
            continue
        reason="No execution process was specified for the connection."
        if p and p.mode=="surrogate":
            reason="LLM numeric coupling is disabled in settings."
            if bus.enabled() and p.effect_port in bus.available and PORTS[p.effect_port]["node"] in lookup:
                if not p.gain_low<=p.gain<=p.gain_high:
                    reason="The lower/central/upper order of the approximate coefficients is inconsistent."
                else:
                    event=events[h.event_index]
                    duration=event.duration_min or 30.0
                    sys.boundaries.update([event.start_min,event.start_min+duration])
                    driver=None;driver_node=f"event:{h.event_index}"
                    if p.driver=="event_exposure":
                        if h.event_index in exposures:driver,driver_node=exposures[h.event_index]
                        else:driver=lambda t,y,e=event,d=duration:float(e.start_min<=t<e.start_min+d)
                    elif p.driver=="glucose_deviation":
                        driver=lambda t,y:min(1,max(0,(sys.value("glucose",t,y)-100)/100));driver_node="glucose"
                    elif p.driver=="insulin_deviation":
                        driver=lambda t,y:min(1,max(0,(sys.value("insulin",t,y)-9.75)/30));driver_node="insulin"
                    elif p.driver=="intestinal_uptake" and "flux:sglt1" in sys.observables:
                        driver=lambda t,y:sys.value("flux:sglt1",t,y)/(1+sys.value("flux:sglt1",t,y));driver_node="transport:sglt1"
                    elif p.driver=="activity_load" and any(e.kind=="activity" for e in events):
                        driver=lambda t,y:min(1,max(0,(met_at(t)-1)/8));driver_node="energy"
                    if driver:
                        ungated=driver
                        driver=lambda t,y,fn=ungated,start=event.start_min:fn(t,y) if t>=start else 0.0
                        sys.state(key,h.target,"dimensionless",0,"llm_coupling")
                        index=sys.indices[key]
                        def lag(t,y,dy,index=index,tau=p.tau_min,driver=driver):dy[index]+=(driver(t,y)-y[index])/tau
                        sys.process(key,lag)
                        bus.effects.setdefault(p.effect_port,[]).append((index,p.gain))
                        effect_key=key+":multiplier"
                        additive=PORTS[p.effect_port].get("mode")=="additive_target"
                        sys.observe(effect_key,"dimensionless",lambda t,y,i=index,g=p.gain,additive=additive:max(-1.2,min(1.2,g*y[i]*bus.settings.llm_scale)) if additive else math.exp(max(-1.2,min(1.2,g*y[i]*bus.settings.llm_scale))))
                        sources=[s for s in compounds.get(str(h.event_index),{}).get("sources",[]) if s["id"] in h.source_ids]
                        target=PORTS[p.effect_port]["node"]
                        nodes.append(make_node(key,h.target,h.organ,3,"llm_surrogate",h.mechanism,state_key=effect_key,unit="relative deviation" if additive else "x",cell_types=[h.cell_type],sources=sources,process_id=key,
                            equations=["dz/dt = (u - z)/tau","signal = clamp(g*z*LLM_scale, -1.2, 1.2)" if additive else "M = exp(clamp(g*z*LLM_scale, -1.2, 1.2))",f"signal added to the target of {PORTS[p.effect_port]['label']}" if additive else f"M multiplied into {PORTS[p.effect_port]['label']}"],
                            parameters=[{"name":"gain","value":p.gain,"unit":"1","status":"llm_estimate"},{"name":"gain_low","value":p.gain_low,"unit":"1","status":"llm_estimate"},{"name":"gain_high","value":p.gain_high,"unit":"1","status":"llm_estimate"},{"name":"tau","value":p.tau_min,"unit":"min","status":"llm_estimate"}],
                            limitations=[h.limitation,"The coefficient range is an LLM exploratory proposal, not a statistical confidence interval.","It perturbs registered-process parameters; it does not estimate actual binding occupancy."],execution=p.model_dump(),linked_hypotheses=[h.model_dump()]))
                        edges.extend([{"source":driver_node,"target":key,"kind":"coupling","label":"driver state"},{"source":key,"target":target,"kind":"coupling","label":"applied to equation","flux_key":effect_key,"unit":"relative deviation" if additive else "x"}])
                        report.append({"hypothesis":j,"status":"compiled","nodes":[key,target],"driver":p.driver,"effect_port":p.effect_port,"gain":p.gain,"gain_low":p.gain_low,"gain_high":p.gain_high,"tau_min":p.tau_min})
                        bus.compiled.append(p.effect_port)
                        continue
                    reason="The selected driver state is absent from the current run graph."
            elif bus.enabled():reason="The quantitative module for that effect port is not active in this run."
        elif p and p.mode=="reuse":reason="The specified existing module is not active in this run."
        nodes.append(make_node(key,h.target,h.organ,3,"llm_hypothesis",h.mechanism,cell_types=[h.cell_type],limitations=[reason,h.limitation],execution=p.model_dump() if p else None))
        edges.append({"source":f"event:{h.event_index}","target":key,"kind":"hypothesis","label":"connection pending"})
        report.append({"hypothesis":j,"status":"unresolved","nodes":[key],"reason":reason})
    return report
