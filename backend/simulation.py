"""Reusable event -> module factories. All extension parameters are explicit.

The only imported physiology equations are the pinned Topp CellML model.
PK, meals, activity, fluid and thermal connections are exploratory, uncalibrated
templates and cannot be mistaken for a validated PBPK/whole-human predictor.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .contracts import Interpretation, RunSettings, Expansion
from .kernel import ModularSystem, UNITS
from .reference import ToppModel, SOURCE_URL
from .presentation import METHOD_LABELS, make_node
from .digestion import attach_digestion
from .coupling import CouplingBus, compile_pathways, prepare_expansion
from .drug_targets import drug_anchors
from .nutrients import normalize_nutrients, reference_nutrients
from .equation_engine import compile_model, validate_trajectory
from .coupling import PORTS
from .expressions import ModelCompileError
from .credibility import software_versions
from .body_network import attach_body, summarize_body



def quantity(value,unit,target):
    if value is None or not unit: return None
    aliases={"milligram":"mg","gram":"g","liter":"L","ml":"mL","degree":"degC","°C":"degC","kilogram":"kg"}
    try:
        result=UNITS.Quantity(value,aliases.get(unit,unit)).to(target).magnitude
        if not math.isfinite(result) or result<0:return None
        return float(result)
    except (pint_error_types()): return None


def pint_error_types():
    import pint
    return (pint.errors.PintError,ValueError,TypeError)


def simulate(interpretation: Interpretation, settings: RunSettings, compounds: dict, expansion: Expansion | None=None, observations: list | None=None):
    sys=ModularSystem();reference=ToppModel();nodes=[];edges=[];warnings=[];assumptions=[];unmodeled=[];ledgers=[];metrics=[]
    interpretation=normalize_nutrients(interpretation.model_copy(deep=True))
    input_interpretation=interpretation.model_copy(deep=True)
    interpretation,reference_scenarios=reference_nutrients(interpretation,settings.allow_assumptions)
    assumptions.extend(r["description"] for r in reference_scenarios)
    horizon=settings.horizon_min;events=interpretation.interventions
    expansion=prepare_expansion(expansion,events)
    bus=CouplingBus(sys,settings);exposures={}
    bus.expose("insulin_secretion","glut4_capacity","hepatic_production")
    source={"id":"pmr:topp2000","title":"Topp 2000 · Original CellML","url":SOURCE_URL}
    for i,event in enumerate(events):
        reference_scenario=next((r for r in reference_scenarios if r["event_index"]==i),None)
        nodes.append(make_node(f"event:{i}",event.label,"Input",0,"assumption" if reference_scenario else "derived",reference_scenario["description"] if reference_scenario else "This is a structured event from the input sentence.",intervention=input_interpretation.interventions[i].model_dump(),reference_scenario=reference_scenario))
        if event.start_min>horizon:warnings.append(f"{event.label}: start time is outside the display range.")
    if not events:unmodeled.append("No computable event was found. Please provide an explicit event and amount.")
    eq=reference.equilibrium()
    for key,label,unit,value in zip(["glucose","insulin","beta"],["Blood glucose","Insulin","Beta-cell mass"],["mg/dL","µU/mL","mg"],eq):
        sys.state(key,label,unit,value,"topp2000")
    gi,ii,bi=[sys.indices[k] for k in ("glucose","insulin","beta")]
    phys=reference.constants
    vg=settings.body_mass_kg*2
    sys.state("glut4_activity","GLUT4 activity","dimensionless",1,"peripheral_uptake")
    sys.state("hepatic_insulin_action","Hepatic insulin signaling","dimensionless",1,"hepatic_regulation")
    glut4_i=sys.indices["glut4_activity"]
    hepatic_i=sys.indices["hepatic_insulin_action"]
    def hepatic_multiplier(t,y):
        native=1/(1+.5*max(0,y[hepatic_i]-1)) if settings.allow_assumptions else 1
        return native*bus.multiplier("hepatic_production",t,y)
    def glucose_model(t,y,dy):
        original=reference.derivative(t/1440,y[[gi,ii,bi]])/1440
        dy[[gi,ii,bi]]+=original
        if settings.allow_assumptions:
            secretion=y[bi]*phys["sigma"]*y[gi]**2/(phys["alpha"]+y[gi]**2)/1440
            incretin=y[sys.indices["incretin"]] if "incretin" in sys.indices else 0
            secretion_factor=settings.insulin_secretion_scale*(1+.6*incretin)*bus.multiplier("insulin_secretion",t,y)
            dy[ii]+=secretion*(secretion_factor-1)
            original_insulin_use=phys["SI"]*y[ii]*y[gi]/1440
            coupled_insulin_use=phys["SI"]*eq[1]*y[glut4_i]*y[gi]/1440*settings.glut4_scale*bus.multiplier("glut4_capacity",t,y)
            dy[gi]+=original_insulin_use-coupled_insulin_use+phys["R0"]/1440*(hepatic_multiplier(t,y)-1)
            target=max(0,y[ii]/eq[1]+.12*(met_at(t)-1))
            dy[glut4_i]+=(target-y[glut4_i])/12
            dy[hepatic_i]+=(max(0,y[ii]/eq[1])-y[hepatic_i])/20
    sys.process("topp2000_coupled",glucose_model)
    sys.observe("flux:utilization","mg/min",lambda t,y:(phys["EG0"]*y[gi]+phys["SI"]*eq[1]*y[glut4_i]*y[gi]*settings.glut4_scale*bus.multiplier("glut4_capacity",t,y))*vg/1440)
    sys.observe("flux:hepatic_production","mg/min",lambda t,y:phys["R0"]*vg/1440*hepatic_multiplier(t,y))
    params=[{"name":k,"value":v,"unit":reference.units[k],"status":"source_model"} for k,v in reference.constants.items() if k not in ("G","I","beta")]
    nodes.extend([
      make_node("glucose","Blood glucose","Circulation",2,"source_model","Shared state: liver production, tissue utilization, and nutritional input are reflected together.",state_key="glucose",unit="mg/dL",cell_types=["Red blood cells","Vascular endothelial cells"],equations=["dG/dt = [R0·M_liver − EG0·G − SI·I_base·GLUT4·G·M_use]/1440 + Ra/Vg − J_urine/Vg"],parameters=params[:3],sources=[source],limitations=["The baseline state is an equilibrium calculated from PMR coefficients. It is not an individual's measured value.","The meal and activity connection is an uncalibrated extension outside the original model."]),
      make_node("insulin","Insulin secretion","Pancreas",3,"source_model","Insulin secretion and clearance according to glucose stimulation and beta-cell mass.",state_key="insulin",unit="µU/mL",process_id="insulin_secretion",cell_types=["Pancreatic beta cells"],equations=["dI/dt = [β·σ·G²/(α+G²) · M_secretion − k·I]/1440", "M_secretion = scale · (1+0.6H_incretin) · M_LLM"],parameters=params[3:6],sources=[source],limitations=["These are equations from the PMR model. They are not estimates of the user's secretory capacity."]),
      make_node("beta","Beta-cell population","Pancreas",3,"source_model","The mass of the cell population is represented as a single state.",state_key="beta",unit="mg",cell_types=["Pancreatic beta cells"],equations=["dβ/dt = (−d0 + r1·G − r2·G²)·β / 1440"],parameters=params[6:],sources=[source],limitations=["Individual intracellular signals are not directly calculated."]),
      make_node("peripheral","GLUT4 · Peripheral glucose utilization","Muscle and fat",3,"assumption","Insulin and activity signals are reflected together in GLUT4 activity and the glucose clearance equation.",state_key="glut4_activity",unit="×",process_id="glut4",cell_types=["Skeletal muscle cells","Adipocytes"],equations=["dGLUT4/dt = [I/I_base + 0.12·(MET−1) − GLUT4]/12", "R_use = [EG0·G + SI·I_base·GLUT4·G·M_use]/1440"],sources=[source],limitations=["This is a reduced extension that replaces the insulin-dependent utilization in Topp's original equation with delayed GLUT4 activity."]),
    ])
    nodes.append(make_node("liver:production","Liver and glucose production","Liver",3,"assumption","The inhibitory signal of insulin is connected to Topp's endogenous production term.",state_key="flux:hepatic_production",unit="mg/min",process_id="hepatic_glucose",cell_types=["Hepatocytes"],equations=["dH_liver/dt = (I/I_base − H_liver)/20", "R_EGP = R0 · Vg/1440 · M_LLM/[1+0.5·max(0,H_liver−1)]"],sources=[source],limitations=["Inhibition according to insulin signaling is approximated as a lumped state, and individual gene expression is not directly integrated."]))
    edges.append({"source":"insulin","target":"liver:production","kind":"regulation","label":"Insulin signaling and production inhibition"})
    edges.append({"source":"liver:production","target":"glucose","kind":"transport","label":"Endogenous production","flux_key":"flux:hepatic_production","unit":"mg/min"})
    edges.extend([{"source":"glucose","target":"insulin","kind":"regulation","label":"Secretion stimulation"},{"source":"insulin","target":"peripheral","kind":"regulation","label":"Utilization regulation"},{"source":"peripheral","target":"glucose","kind":"feedback","label":"Glucose utilization"},{"source":"glucose","target":"beta","kind":"regulation","label":"Growth and loss"},{"source":"beta","target":"insulin","kind":"regulation","label":"Secretory capacity"}])
    # Standardized physical activity signal drives every dependent module.
    activity=[]
    for i,e in enumerate(events):
        if e.kind!="activity":continue
        if e.duration_min is None:
            unmodeled.append(f"{e.label}: no duration given, so exercise amount was not computed.");continue
        met=e.intensity_met
        if met is None and settings.allow_assumptions:
            met=settings.activity_met;assumptions.append(f"{e.label}: {met:g} MET was used for the unspecified intensity.")
        if met is None:
            unmodeled.append(f"{e.label}: no MET available, so numeric computation was skipped.");continue
        activity.append((i,e.start_min,e.start_min+e.duration_min,met))
        sys.boundaries.update([e.start_min,e.start_min+e.duration_min])
    def met_at(t):return max([a[3] for a in activity if a[1]<=t<a[2]]+[1.0])
    if len(activity)>1:assumptions.append("Overlapping activities do not sum MET; the largest intensity is used.")
    sys.state("energy","Total energy during activity","kcal",0,"energy")
    en=sys.indices["energy"]
    def energy(t,y,dy):
        if any(a[1]<=t<a[2] for a in activity):dy[en]+=met_at(t)*3.5*settings.body_mass_kg/200
    sys.process("met_energy",energy)
    if activity:
        nodes.append(make_node("energy","Metabolic energy demand","Skeletal muscle",1,"derived","Total energy during activity converted from MET and body weight. Includes resting expenditure.",state_key="energy",unit="kcal",cell_types=["Skeletal muscle cells"],equations=["kcal/min = MET · 3.5 · body_mass_kg / 200"],parameters=[{"name":"body_mass","value":settings.body_mass_kg,"unit":"kg","status":"user_input"}],limitations=["MET does not replace individual indirect calorimetry.","Estimation based on oxygen conversion factor and energy conversion."]))
        for i,*_ in activity:edges.append({"source":f"event:{i}","target":"energy","kind":"input","label":"Intensity × time"})
    # Oral chemical cohorts share compartments for the same resolved substance.
    groups={}
    for i,e in enumerate(events):
        if e.kind!="chemical":continue
        dose=quantity(e.quantity,e.unit,"mg")
        if not e.entity:
            unmodeled.append(f"{e.label}: an active-ingredient name is required.");continue
        if dose is None or dose<=0:
            # Compound with unknown dose: assume a 0-1 normalized exposure curve rather than fabricating mass.
            # This signal drives the target anchors and detoxification path.
            if not settings.allow_assumptions:
                unmodeled.append(f"{e.label}: no dose and no validated PK parameters, so only qualitative mechanisms are shown.");continue
            key=f"exposure:{i}"
            def unit_exposure(t,y,s=e.start_min,d=e.duration_min or 480,h=settings.elimination_half_min):
                if t<s:return 0.
                if t<s+d:return 1-math.exp(-(t-s)/20.)
                return (1-math.exp(-d/20.))*math.exp(-(t-s-d)/(2*h))
            exposures[i]=(unit_exposure,key)
            nodes.append(make_node(key,f"{e.entity} Normalized exposure","Route of administration",1,"assumption","Dose not specified, so assumed 0-1 normalized exposure curve instead of mass compartment.",unit="relative exposure",cell_types=["Plasma"],compound=compounds.get(str(i),{}),limitations=["Dose, formulation, and absorption rate unknown — not an actual blood concentration curve.","Magnitude of normalized signal does not indicate actual occupancy."]))
            edges.append({"source":f"event:{i}","target":key,"kind":"input","label":"Normalized exposure"})
            assumptions.append(f"{e.label}: dose unknown - a normalized exposure over a {e.duration_min or 480:g}-min window was assumed. Not dose-based PK.")
            continue
        route=e.route
        if route=="unspecified":
            route="oral";assumptions.append(f"{e.label}: route of administration not stated; oral absorption assumed.")
        elif route=="other":
            assumptions.append(f"{e.label}: non-IV injection/parenteral dosing was assumed to be a tissue depot (subcutaneous/intramuscular approximation).")
        if not settings.allow_assumptions:
            unmodeled.append(f"{e.label}: no validated PK parameters, so only qualitative mechanisms are shown.");continue
        c=compounds.get(str(i),{})
        identity=str(c.get("cid") or e.entity.casefold())
        groups.setdefault(identity,{"compound":c,"label":e.entity,"doses":[]})["doses"].append((i,e,dose,route))
    vd_map={}
    for number,g in enumerate(groups.values()):
        prefix=f"drug{number}";c=g["compound"];keys=[]
        compartments=[("mouth","Route and formulation",1,["Oral epithelial cells"]),("esophagus","Esophageal transit",1,["Esophageal squamous epithelial cells"]),("stomach","Gastric contents",1,["Gastric mucosal epithelial cells"]),("intestine","Small intestinal absorption",1,["Intestinal epithelial cells"]),("portal","Portal vein transfer",2,["Vascular endothelial cells"]),("plasma","Systemic circulation",2,["Vascular endothelial cells"]),("tissue","Tissue distribution",3,["Cells by tissue, unresolved"]),("removed","Parent removal",3,["Hepatocytes","Renal tubular cells"])]
        inhaled=any(route=="inhaled" for _,_,_,route in g["doses"])
        if inhaled:compartments.append(("lung","lung - inhaled absorption",1,["alveolar epithelium"]))
        from .drug_pk import lookup as pk_lookup, DEFAULT_VD_L
        pk=pk_lookup(g["label"]) or pk_lookup(str(c.get("preferred_name") or c.get("query") or ""))
        vd_l=(pk["vd_l_kg"]*settings.body_mass_kg) if pk else DEFAULT_VD_L
        elim_min=pk["t_half_h"]*60 if pk else settings.elimination_half_min
        absorb_min=max(5.,(pk["t_max_h"]*60-30)/2.1) if pk and pk.get("t_max_h") else settings.absorption_half_min
        bioavail=pk["f"] if pk else 1.0
        if pk:assumptions.append(f"{g['label']}: literature PK used - F={pk['f']}, Vd={pk['vd_l_kg']}L/kg, t1/2={pk['t_half_h']}h ({pk['ref']}). Inter-individual variation not included.")
        else:assumptions.append(f"{g['label']}: no literature PK - F=1, Vd={DEFAULT_VD_L}L (plasma volume), shared elimination half-life assumed.")
        for i,e,dose,route in g["doses"]:vd_map[i]=vd_l
        par=[{"name":"Gastric emptying half-life","value":settings.gastric_half_min,"unit":"min","status":"assumption"},{"name":"Absorption half-life","value":round(absorb_min,1),"unit":"min","status":"literature" if pk else "assumption"},{"name":"Central elimination half-life","value":elim_min,"unit":"min","status":"literature" if pk else "assumption"},{"name":"Bioavailability F","value":bioavail,"unit":"","status":"literature" if pk else "assumption"},{"name":"Volume of distribution Vd","value":round(vd_l,1),"unit":"L","status":"literature" if pk else "assumption"}]
        for compartment,label,stage,cells in compartments:
            key=prefix+":"+compartment;keys.append(key);sys.state(key,label,"mg",0,prefix)
            nodes.append(make_node(key,label,g["label"],stage,"assumption","Amount of parent drug in compartment assuming first-order transport.",state_key=key,unit="mg",cell_types=cells,compound=c,parameters=par,equations=["transfer rate J = k * A_source", "dA_source/dt = -J; dA_destination/dt = +J"],sources=c.get("sources",[]),limitations=["Common transport assumption not corrected by substance. Not a validated PBPK prediction.","Removal amount is the equivalent of metabolized/excreted parent. Not metabolite mass."] if compartment=="removed" else ["Not corrected with substance-specific PK data.","This rate is not determined by molecular weight or target information alone."]))
        rates=[20.,30.,math.log(2)/settings.gastric_half_min,math.log(2)/absorb_min,.2]
        for a,b,k in zip(keys[:5],keys[1:6],rates):
            sys.transfer(a+">"+b,a,b,k);edges.append({"source":a,"target":b,"kind":"transport","label":"Primary transport","rate_coefficient":k,"source_state":a,"unit":"mg/min"})
        if bioavail<1:
            # The unabsorbed fraction goes to the elimination (fecal) compartment - preserves mass balance.
            k_int=math.log(2)/absorb_min
            sys.transfer(keys[3]+">"+keys[7],keys[3],keys[7],k_int*(1-bioavail)/bioavail)
            edges.append({"source":keys[3],"target":keys[7],"kind":"transport","label":f"unabsorbed fraction (1-F={1-bioavail:.2f})","rate_coefficient":k_int*(1-bioavail)/bioavail,"source_state":keys[3],"unit":"mg/min"})
        for a,b,k in [(keys[5],keys[6],.001),(keys[6],keys[5],.0005),(keys[5],keys[7],math.log(2)/elim_min)]:
            sys.transfer(a+">"+b,a,b,k);edges.append({"source":a,"target":b,"kind":"transport","label":"Assumed movement","rate_coefficient":k,"source_state":a,"unit":"mg/min"})
        if inhaled:
            sys.transfer(keys[8]+">"+keys[5],keys[8],keys[5],.5)
            edges.append({"source":keys[8],"target":keys[5],"kind":"transport","label":"alveolar absorption (assumed)","rate_coefficient":.5,"source_state":keys[8],"unit":"mg/min"})
        for i,e,dose,route in g["doses"]:
            destination={"oral":keys[0],"iv":keys[5],"other":keys[6]}.get(route,keys[8] if inhaled else keys[0])
            sys.impulse(e.start_min,destination,dose);edges.append({"source":f"event:{i}","target":destination,"kind":"input","label":f"{dose:g} mg"})
        for i,e,dose,route in g["doses"]:
            exposures[i]=(lambda t,y,k=keys[5],d=dose: min(1,max(0,sys.value(k,t,y)/(d*.25))),keys[5])
        ledgers.append({"name":g["label"],"keys":keys,"doses":[(e.start_min,dose) for _,e,dose,_ in g["doses"]]})
        assumptions.append(f"{g['label']}: gastric/absorption/elimination half-lives {settings.gastric_half_min:g}/{absorb_min:g}/{elim_min:g} min, tissue transfer coefficients 0.001/0.0005 min^-1, Vd={vd_l:g}L. The exploratory model structure is identical even when literature values are used.")
    digestive=attach_digestion(sys,events,settings,bus,quantity)
    nodes.extend(digestive["nodes"]);edges.extend(digestive["edges"])
    ledgers.extend(digestive["ledgers"]);assumptions.extend(digestive["assumptions"]);unmodeled.extend(digestive["unmodeled"])
    meal_events=digestive["events"];exposures.update(digestive["exposures"])
    environment=[];water=[]
    for i,e in enumerate(events):
        if e.kind=="environment":
            temp=quantity(e.quantity,e.unit,"degC")
            if temp is not None and 0<=temp<=50 and settings.allow_assumptions:
                end=e.start_min+(e.duration_min or horizon);environment.append((i,e.start_min,end,temp));sys.boundaries.update([e.start_min,end])
            else:unmodeled.append(f"{e.label}: only explicit ambient temperatures of 0-50 degC with exploratory assumptions are supported.")
        if e.kind=="hydration":
            volume=quantity(e.quantity,e.unit,"mL")
            if volume is not None and settings.allow_assumptions:water.append((i,e,volume))
            else:unmodeled.append(f"{e.label}: a water amount and water-balance assumption are required.")
    has_surrogates=bool(expansion and any(h.execution and h.execution.mode=="surrogate" for h in expansion.hypotheses))
    whole_body=settings.allow_assumptions and bool(events or settings.enable_body_network)
    if whole_body:
        bus.expose("heart_rate_drive","vascular_resistance","thermogenesis","renal_reabsorption","renal_filtration","water_clearance","temperature_setpoint")
        for k,l,u,v,nonnegative in [("hr","Heart rate","1/min",72,True),("sv","Stroke volume","mL",70,True),("svr","Systemic vascular resistance","mmHg*min/L",17,True),("temperature","Core body temperature","degC",37,False),("fluid","Body-fluid change vs baseline","mL",0,False),("water_gut","Intestinal water","mL",0,True),("water_lost","In-model water loss","mL",0,True)]:sys.state(k,l,u,v,"whole_body_proxy",nonnegative)
        ix={k:sys.indices[k] for k in ("hr","sv","svr","temperature","fluid","water_gut","water_lost")}
        def ambient(t):
            active=[e for e in environment if e[1]<=t<e[2]]
            return max(active,key=lambda e:e[1])[3] if active else 22.
        def response(t,y,dy):
            work=met_at(t)-1;temp=y[ix["temperature"]];fluid=y[ix["fluid"]]
            food_rate=sys.value("flux:absorbed_energy",t,y) if "flux:absorbed_energy" in sys.observables else 0
            feeding=food_rate/(2+food_rate)
            dy[ix["hr"]]+=((72+10*work+6*feeding)*bus.multiplier("heart_rate_drive",t,y)-y[ix["hr"]])/3
            dy[ix["sv"]]+=(70+3*work+fluid*.002-y[ix["sv"]])/4
            resistance=17*bus.multiplier("vascular_resistance",t,y)/(1+.09*work+.05*max(0,temp-37)+.08*feeding)
            dy[ix["svr"]]+=(resistance-y[ix["svr"]])/4
            heat=(work*3.5*settings.body_mass_kg/200*.8+.08*food_rate)*4184/60*bus.multiplier("thermogenesis",t,y)
            dy[ix["temperature"]]+=(heat-100*(temp-37-bus.signal("temperature_setpoint",t,y))+10*(ambient(t)-22))*60/(settings.body_mass_kg*3500)
            absorb=.035*y[ix["water_gut"]];loss=3*work+2*max(0,ambient(t)-22)+max(0,fluid)*math.log(2)/120*bus.multiplier("water_clearance",t,y)
            dy[ix["water_gut"]]-=absorb;dy[ix["fluid"]]+=absorb-loss;dy[ix["water_lost"]]+=loss

        sys.process("whole_body_response",response)
        for i,e,v in water:sys.impulse(e.start_min,"water_gut",v)
        nodes.extend([
          make_node("cardiovascular","Heart and heart rate response","Heart and blood vessels",3,"assumption","A first-order delay model that responds to activity load and fluid status.",state_key="hr",unit="bpm",cell_types=["Cardiomyocytes","Vascular smooth muscle cells"],process_id="cardiovascular",equations=["dHR/dt = [(72 + 10·(MET−1) + 6·feeding)·M_HR − HR]/3","MAP = HR · SV/1000 · SVR + 3"],parameters=[{"name":"HR_base","value":72,"unit":"bpm","status":"assumption"}],limitations=["All coefficients are exploratory assumptions. Not validated against actual blood pressure or heart rate predictions."]),
          make_node("temperature","Heat generation and dissipation","Body temperature regulation",3,"assumption","Heat capacity model linked to energy demand and ambient temperature.",state_key="temperature",unit="°C",cell_types=["Skeletal muscle cells","Sweat gland secretory cells"],process_id="thermoregulation",equations=["Q = (0.8·E_activity + 0.08·E_food) · M_heat", "T_set = 37 + bounded_pyrogen_signal; dT/dt = [Q − 100·(T−T_set) + 10·(T_ambient−22)]·60/(m·3500)"],limitations=["Heat transfer coefficients were not adjusted for body size, clothing, or humidity."]),
          make_node("fluid","Water balance","Small intestine, kidney, skin",2,"assumption","Records ingested water and model losses in shared organs.",state_key="fluid",unit="mL",cell_types=["Intestinal epithelial cells","Renal tubular cells","Sweat gland secretory cells"],equations=["dFluid/dt = absorption − sweat_proxy − excess_clearance·M_water"],limitations=["Changes relative to baseline, not total body water. Electrolyte and osmotic regulation are calculated separately as relative deviations in the whole-body connection model. Absolute concentrations are uncalibrated."]),
        ])
        nodes.append(make_node("vascular","Vascular and peripheral resistance","Heart and blood vessels",3,"assumption","Receives activity, diet, body temperature, and LLM approximation signals, which are reflected in arterial pressure calculations.",state_key="svr",unit="mmHg·min/L",process_id="vascular",cell_types=["Vascular smooth muscle cells"],equations=["dSVR/dt = [17·M_SVR/(1+0.09work+0.05ΔT+0.08feeding)−SVR]/4"]))
        edges.extend([{"source":"vascular","target":"cardiovascular","kind":"regulation","label":"Whole-body resistance"}])
        if meal_events:edges.extend([{"source":"liver:meal","target":"cardiovascular","kind":"regulation","label":"Postprandial circulatory response"},{"source":"liver:meal","target":"temperature","kind":"regulation","label":"Diet-induced thermogenesis"}])
        if activity:edges.extend([{"source":"energy","target":"cardiovascular","kind":"regulation","label":"Load approximation"},{"source":"energy","target":"temperature","kind":"regulation","label":"Heat generation"},{"source":"energy","target":"glucose","kind":"regulation","label":"Assumption of increased utilization"}])
        edges.append({"source":"fluid","target":"cardiovascular","kind":"regulation","label":"Fluid coupling assumption"})
        for i,*_ in environment:edges.append({"source":f"event:{i}","target":"temperature","kind":"input","label":"ambient temperature"})
        for i,_,_ in water:edges.append({"source":f"event:{i}","target":"fluid","kind":"input","label":"water intake"})
        assumptions.append("Heart rate, blood pressure, thermal, fluid, and activity couplings to blood glucose are uncalibrated shared-state modules; numbers are assumption-driven trajectories.")
        ledgers.append({"name":"Water ledger relative to baseline","keys":["water_gut","fluid","water_lost"],"doses":[(e.start_min,v) for _,e,v in water]})
    # Known mechanisms and hypotheses can be viewed, but never write numeric state.
    for event_index,c in compounds.items():
        records=c.get("activities",[])
        targets=list(dict.fromkeys(r.get("target_pref_name") for r in records if r.get("target_pref_name")))[:6]
        if c.get("status")=="resolved":
            nid=f"compound:{event_index}"
            nodes.append(make_node(nid,c.get("preferred_name",c["query"]),"Molecular identification",1,"knowledge","Substance confirmed by RDKit from PubChem structure.",compound=c,sources=c.get("sources",[]),limitations=["Structure identification alone does not determine pharmacokinetics or binding rates in humans."]))
            edges.append({"source":f"event:{event_index}","target":nid,"kind":"knowledge","label":"Molecular identification"})
        for j,target in enumerate(targets):
            nid=f"target:{event_index}:{j}"
            subset={**c,"activities":[r for r in records if r.get("target_pref_name")==target]}
            nodes.append(make_node(nid,target,"Experimental target",3,"knowledge","Target experiment records linked to the same substance. Check species, assay method, and conditions in the evidence tab.",compound=subset,sources=c.get("sources",[]),limitations=["In vitro activity records do not confirm binding or binding rate in humans after ingestion.","Occupancy is not calculated if units, conditions, or free concentration are not confirmed."]))
            edges.append({"source":f"compound:{event_index}","target":nid,"kind":"knowledge","label":"Experimental record"})
    if whole_body:
        # Net renal loss with explicit filtration/reabsorption observations.
        # No filtered mass is removed a second time before reabsorption.
        def renal(t,y):
            filtered=1.2*max(0,y[gi])*bus.multiplier("renal_filtration",t,y)
            tm=216*bus.multiplier("renal_reabsorption",t,y)
            sglt2=min(filtered,.9*tm);sglt1=min(max(0,filtered-sglt2),.1*tm)
            return filtered,sglt2,sglt1,max(0,filtered-sglt2-sglt1)
        sys.state("urinary_glucose","Cumulative urinary glucose","mg",0,"renal")
        def renal_process(t,y,dy):
            loss=renal(t,y)[3];dy[gi]-=loss/vg;dy[sys.indices["urinary_glucose"]]+=loss
        sys.process("renal_glucose",renal_process)
        for index,name in enumerate(["filtration","sglt2","sglt1","excretion"]):
            obs="renal:"+name;sys.observe(obs,"mg/min",lambda t,y,index=index:renal(t,y)[index])
            nodes.append(make_node(obs,{"filtration":"Renal glucose filtration","sglt2":"SGLT2 and renal reabsorption","sglt1":"SGLT1 and residual reabsorption","excretion":"Urine and glucose loss"}[name],"Kidney",3,"assumption","Calculates filtration and reabsorption, and only net excretion is subtracted from shared blood glucose.",state_key=obs,unit="mg/min",process_id=obs,cell_types=["Renal proximal tubule epithelial cells"],equations=["J_filter = GFR_base · M_filtration · G", "J_urine = max(0,J_filter − J_SGLT2 − J_SGLT1)"],parameters=[{"name":"GFR","value":1.2,"unit":"dL/min","status":"assumption"},{"name":"Tm_total","value":216,"unit":"mg/min","status":"assumption"}],limitations=["An uncalibrated model that condenses GFR, transport maximum, and SGLT2/SGLT1 contributions."]))
        edges.extend([{"source":"glucose","target":"renal:filtration","kind":"transport","label":"Filtration"},{"source":"renal:filtration","target":"renal:sglt2","kind":"transport","label":"Reabsorption"},{"source":"renal:sglt2","target":"renal:sglt1","kind":"transport","label":"Residual volume"},{"source":"renal:sglt1","target":"renal:excretion","kind":"transport","label":"Net excretion"},{"source":"renal:excretion","target":"glucose","kind":"feedback","label":"Net loss","flux_key":"renal:excretion","unit":"mg/min"}])
    anchors=drug_anchors(events,compounds)
    for a in anchors:
        # For drugs with a known dose: attach the plasma-compartment state and
        # molecular weight to anchors so body_network can build affinity-based occupancy signals.
        i=a["event_index"];c=compounds.get(str(i),{})
        node_key=exposures.get(i,(None,None))[1]
        if node_key and str(node_key).startswith("drug") and c.get("molecular_weight"):
            a["plasma_key"]=node_key;a["mw"]=float(c["molecular_weight"])
            if i in vd_map:a["vd_l"]=vd_map[i]
    body=attach_body(sys,bus,events,settings,met_at,quantity,exposures,anchors)
    disabled_body_couplings=[]
    if body["status"]!="active" and expansion and expansion.model:
        disabled_body_couplings=[c.model_dump() for c in expansion.model.couplings if c.port.startswith("body_")]
        expansion.model.couplings=[c for c in expansion.model.couplings if not c.port.startswith("body_")]
        if disabled_body_couplings:warnings.append("Whole-body systems were disabled, so LLM couplings into them were deactivated. The generated states remain as independent observations.")
    nodes.extend(body["nodes"])
    valid_nodes={n["id"] for n in nodes}
    edges.extend(e for e in body["edges"] if e["source"] in valid_nodes and e["target"] in valid_nodes)
    assumptions.extend(body["assumptions"])
    coupling=compile_pathways(sys,bus,events,expansion,compounds,nodes,edges,exposures,met_at)
    model_context={"states":[{"key":s.key,"label":s.label,"unit":s.unit,"initial":s.initial,"owner":s.owner} for s in sys.states],
                   "fluxes":[{"key":k,"unit":u} for k,(u,_) in sys.observables.items()],
                   "processes":[{"id":n["id"],"label":n["label"],"process_id":n.get("process_id")} for n in nodes if n.get("process_id")],
                   "ports":{k:PORTS[k] for k in sorted(bus.available)}}
    model_context["reference_scenarios"]=reference_scenarios
    context_edges={}
    for edge in edges:
        if edge["kind"] not in {"knowledge","hypothesis"}:context_edges.setdefault(edge["source"],set()).add(edge["target"])
    for process in model_context["processes"]:process["event_indices"]=[]
    for i in range(len(events)):
        pending=[f"event:{i}"];reachable=set()
        while pending:
            key=pending.pop()
            if key in reachable:continue
            reachable.add(key);pending.extend(context_edges.get(key,()))
        for process in model_context["processes"]:
            if process["id"] in reachable:process["event_indices"].append(i)
    missing_body_inputs=[b.source for b in expansion.model.bindings if b.source.startswith("body:") and b.source not in sys.indices and b.source not in sys.observables] if expansion and expansion.model else []
    generated=compile_model(sys,bus,events,expansion.model if expansion and not missing_body_inputs else None,nodes,edges,exposures,settings,expansion.sources if expansion else [])
    if missing_body_inputs:
        generated.update(status="disabled",reason="Whole-body systems were disabled, so generated equations lack shared states to read: "+", ".join(missing_body_inputs)+". This extension was excluded from the computation.")
        warnings.append(generated["reason"])
    generated["disabled_body_couplings"]=disabled_body_couplings
    native_events={a[0] for a in meal_events}|{a[0] for a in activity}|{a[0] for a in environment}|{a[0] for a in water}
    native_events.update(i for g in groups.values() for i,_,_,_ in g["doses"])
    # Normalized exposure of a dose-unknown drug also counts as native execution.
    native_events.update(i for i in exposures if i < len(events) and events[i].kind=="chemical")
    native_events.update(body["events"])
    connected_events=native_events|set(generated["events"])
    event_coverage=[{"event_index":i,"label":e.label,"native":i in native_events,"generated":i in generated["events"],"status":"modeled" if i in connected_events else "uncovered"} for i,e in enumerate(events)]
    for i,e in enumerate(events):
        if i not in connected_events and not any(e.label in note for note in unmodeled):unmodeled.append(f"{e.label}: no executable physiological pathway has been built yet.")
    if generated["status"]=="compiled":
        assumptions.append("Auto-generated states, equations, and coefficients are LLM exploratory assumptions not calibrated against a validated physiological model. Distinguish this from passing unit and numeric checks.")
        for i in generated["events"]:
            if events[i].duration_min is None and i not in exposures:assumptions.append(f"{events[i].label}: a 30-min exposure window was assumed for a stimulus without timing. event_{i} is a normalized stimulus, not a dose/concentration.")
        unmodeled=[note.replace("only explicit ambient temperatures of 0-50 degC with exploratory assumptions are supported.","the baseline temperature module does not apply; an auto-generated model approximates it.") if any(e.label in note for i,e in enumerate(events) if i in generated["events"]) else note for note in unmodeled]
    for observation in observations or []:
        if observation.get("mode")=="nudge":
            sys.nudge(observation["time_min"],observation["state"],observation["value"],observation.get("gain",0.15))
        else:
            sys.assimilate(observation["time_min"],observation["state"],observation["value"],observation.get("gain",0.7))
    try:times,values,validation=sys.integrate(horizon)
    except (ValueError,OverflowError,ZeroDivisionError) as error:
        if generated["status"]=="compiled":raise ModelCompileError("Generated-model numerical integration failed: "+str(error)[:200]) from error
        raise
    traces={s.key:values[i].tolist() for i,s in enumerate(sys.states)}
    try:traces.update(sys.observation_traces(times,values))
    except (ValueError,OverflowError,ZeroDivisionError) as error:
        if generated["status"]=="compiled":raise ModelCompileError("Generated observation equation numeric error: "+str(error)[:200]) from error
        raise
    balance=validate_trajectory(generated,expansion.model if expansion else None,times,traces)
    for ledger in ledgers:
        actual=np.sum([values[sys.indices[k]]*ledger.get("weights",{}).get(k,1) for k in ledger["keys"]],axis=0)
        expected=np.array([sum(d for start,d in ledger["doses"] if start<=t) for t in times])
        error=float(np.max(np.abs(actual-expected)));scale=max(1,float(expected.max()))
        passed=error<=1e-6*scale
        balance.append({"name":ledger["name"],"max_error":error,"unit":ledger.get("unit",sys.states[sys.indices[ledger["keys"][0]]].unit),"passed":passed,"keys":ledger["keys"]})
        if not passed:raise ValueError(f"Conservation check failed: {ledger['name']}")
    glycemia=bool(body["status"]=="active" or meal_events or (whole_body and activity) or any(p in {"insulin_secretion","glut4_capacity","hepatic_production","renal_reabsorption"} for p in bus.compiled))
    def metric(id,label,unit,data,baseline,method,explanation):
        nonlocal nodes,edges
        metrics.append({"id":id,"label":label,"unit":unit,"values":data,"baseline":baseline,"method":method,"explanation":explanation})
        nodes.append(make_node("output:"+id,label,"Whole-body observed values",4,method,explanation,state_key=id if data is not None else None,unit=unit,metric_id=id))
    metric("glucose","Blood glucose","mg/dL",traces["glucose"] if glycemia else None,eq[0],"assumption" if glycemia else "unknown","Exploratory results linking diet and activity" if glycemia else "Blood glucose effects of current input are not modeled. Distinguished from baseline model values.")
    metric("insulin","Insulin","µU/mL",traces["insulin"] if glycemia else None,eq[1],"assumption" if glycemia else "unknown","Results of connecting inputs to the reference model. No individual calibration.")
    metric("energy","Activity energy","kcal",traces["energy"] if (activity or meal_events) else None,0,"derived" if (activity or meal_events) else "unknown","MET-converted total energy expenditure during activity. Includes resting expenditure.")
    if whole_body:
        map_values=values[ix["hr"]]*values[ix["sv"]]/1000*values[ix["svr"]]+3
        traces["map"]=map_values.tolist()
        for id,label,unit,data,base in [("hr","Heart rate","bpm",traces["hr"],72),("map","Mean arterial pressure","mmHg",traces["map"],72*70/1000*17+3),("temperature","Core body temperature","°C",traces["temperature"],37)]:metric(id,label,unit,data,base,"assumption","Assumption-based trajectory from uncalibrated circulatory and thermal response modules.")
    else:
        for id,label,unit in [("hr","Heart rate","bpm"),("map","Mean arterial pressure","mmHg"),("temperature","Core body temperature","°C")]:metric(id,label,unit,None,None,"unknown","No quantitative model connects the current input to this observation.")
    if glycemia:edges.extend([{"source":"glucose","target":"output:glucose","kind":"observation","label":"Observation"},{"source":"insulin","target":"output:insulin","kind":"observation","label":"Observation"}])
    if activity:edges.append({"source":"energy","target":"output:energy","kind":"observation","label":"Integration"})
    if whole_body:edges.extend([{"source":"cardiovascular","target":"output:hr","kind":"observation","label":"Observation"},{"source":"cardiovascular","target":"output:map","kind":"observation","label":"HR · SV · SVR"},{"source":"temperature","target":"output:temperature","kind":"observation","label":"Observation"}])
    for observation in generated["observations"]:
        key=observation["key"]
        metric(key,observation["label"],observation["unit"],traces[key],observation["baseline"],"llm_surrogate",observation["description"]+" - exploratory result of auto-generated equations; no individual calibration.")
        for source_node in observation["sources"]:edges.append({"source":source_node,"target":"output:"+key,"kind":"observation","label":"Generation observation equation"})
    for observation in body["observations"]:
        key=observation["id"]
        metric(key,observation["label"],observation["unit"],traces[key],observation["baseline"],"assumption","Uncalibrated relative response of whole-body coupling equations." if observation["unit"]=="relative deviation" else "Cardiac output calculated as HR x SV. HR/SV coefficients are uncalibrated.")
        nodes[-1]["body_system"]=observation["system"]
        source_node="cardiovascular" if key=="body:cardiac_output" else key
        edges.append({"source":source_node,"target":"output:"+key,"kind":"observation","label":"Whole-body connection observation"})
    body_report=summarize_body(body,traces,nodes,edges)
    if not settings.allow_assumptions:warnings.append("Assumption-based numeric extension is disabled. Outputs without a validated connection remain unmodeled.")
    # Every plan includes exact source, parameters and event interpretation.
    engine_hash=hashlib.sha256(b"".join((Path(__file__).parent/name).read_bytes() for name in ["simulation.py","kernel.py","reference.py","digestion.py","coupling.py","nutrients.py","contracts.py","equation_contracts.py","equation_engine.py","expressions.py","body_library.py","body_network.py"])).hexdigest()
    plan={"schema_version":"0.5","engine_version":"0.5.0","software":software_versions(),"engine_hash":engine_hash,"generated_model_hash":generated["hash"],"coupling":coupling,"expansion":expansion.model_dump() if expansion else None,"source_hash":reference.sha256,"interpretation":input_interpretation.model_dump(),"modeled_interpretation":interpretation.model_dump(),"reference_scenarios":reference_scenarios,"compound_identities":{i:c.get("inchikey",c.get("cid",c.get("query"))) for i,c in compounds.items()},"settings":settings.model_dump(),"assumptions":assumptions,"state_owners":{s.key:s.owner for s in sys.states},"processes":[p[0] for p in sys.processes]}
    plan_hash=hashlib.sha256(json.dumps(plan,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    # Stable graph metadata does not depend on reveal stage or selected time.
    return {"title":interpretation.title,"plan_hash":plan_hash,"created_at":datetime.now(timezone.utc).isoformat(),"plan":plan,"time":times.tolist(),"traces":traces,"nodes":nodes,"edges":edges,"metrics":metrics,"compounds":compounds,"interpretation":input_interpretation.model_dump(),"reference_scenarios":reference_scenarios,"summary":expansion.summary if expansion else "Results connected using registered equations and explicit assumptions. New equation construction is not yet complete.","assumptions":assumptions,"unmodeled":unmodeled,"warnings":warnings+interpretation.notes,"validation":validation|{"balances":balance,"reference_scope":"Matches PMR source numerics. Distinct from paper or clinical reproduction."},"body":body_report,"generated":generated,"event_coverage":event_coverage,"model_context":model_context,"coupling":coupling,"coverage":{"integrated":sum(x["status"]!="unresolved" for x in coupling),"surrogates":sum(x["status"]=="compiled" for x in coupling),"unresolved":sum(x["status"]=="unresolved" for x in coupling),"quantified":sum(m["values"] is not None for m in metrics),"total":len(metrics),"hypotheses":len(expansion.hypotheses) if expansion else 0,"generated_states":generated["states"],"generated_processes":generated["processes"],"events_modeled":len(connected_events),"events_total":len(events)},"state_specs":[s.model_dump() for s in sys.states]}
