"""Stoichiometric sugar digestion -> membrane transport -> shared glycemia.

The transporter topology and Km references are literature based. Whole-organ
Vmax, volumes, endocrine gains and fructose conversion are explicit exploratory
parameters. This is a reduced model, not the full published CellML model.
"""
import math
from .nutrients import SPECIES,nutrient_identity
from .presentation import make_node

TRANSPORT_SOURCE={"id":"afshar2021","title":"Afshar 2021 · enterocyte SGLT1/GLUT2 model","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC8688934/"}
CELL_SOURCE={"id":"afshar2019","title":"Afshar 2019 · modular enterocyte model","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC6473069/"}
STOICH_SOURCE={"id":"ritschel2023","title":"Ritschel 2023 · meal model and stoichiometry","url":"https://arxiv.org/abs/2307.16444"}


def attach_digestion(sys,events,settings,bus,quantity):
    nodes=[];edges=[];notes=[];unmodeled=[];doses=[]
    for i,e in enumerate(events):
        if e.kind!="nutrition":continue
        if e.route not in {"oral","unspecified"}:
            unmodeled.append(f"{e.label}: the current nutrition module computes the oral route; other routes of administration were not converted to intestinal absorption.");continue
        species=nutrient_identity(e.entity)
        # Direct structured legacy API inputs explicitly identify carbohydrate.
        if not species and not e.entity and e.label.casefold() in {"carbs","carbohydrate"}:species="carbohydrate"
        mass=quantity(e.quantity,e.unit,"mg")
        if not species or mass is None or mass<=0 or not settings.allow_assumptions:
            unmodeled.append(f"{e.label}: digestion/absorption was skipped because species mass or transport-coefficient assumptions are missing.");continue
        doses.append((i,e,species,mass/SPECIES[species]["mw"]))
    empty={"nodes":nodes,"edges":edges,"assumptions":notes,"unmodeled":unmodeled,"ledgers":[],"events":doses,"exposures":{}}
    if not doses:return empty
    bus.expose("gastric_emptying","sglt1_capacity","glut2_capacity","glut5_capacity","sucrase_capacity")
    supplied=list(dict.fromkeys(s for _,_,s,_ in doses))
    monos=["glucose","fructose","galactose"]
    keys=[];weights={}
    def state(key,label,unit="mmol",weight=1):
        sys.state(key,label,unit,0,"intestinal_metabolism");keys.append(key);weights[key]=weight
    def add_state_node(key,label,group,stage,unit="mmol",**extra):
        nodes.append(make_node(key,label,group,stage,"assumption","A state that conserves mass and is shared with adjacent processes.",state_key=key,unit=unit,**extra))
    for s in supplied:
        info=SPECIES[s]
        for compartment,label in [("mouth","mouth"),("esophagus","esophagus"),("stomach","stomach"),("lumen","lumen")]:
            key=f"meal:{compartment}:{s}";state(key,f"{label} {info['label']}",weight=info["hexoses"])
            add_state_node(key,f"{label} - {info['label']}",f"{label} contents",1,sources=[STOICH_SOURCE],parameters=[{"name":"Molecular weight / unit mass","value":info["mw"],"unit":"mg/mmol","status":"derived"}])
        for a,b,k in [("mouth","esophagus",20),("esophagus","stomach",30)]:
            source=f"meal:{a}:{s}";target=f"meal:{b}:{s}"
            sys.transfer(source+">"+target,source,target,k)
            edges.append({"source":source,"target":target,"kind":"transport","label":"Pass through","source_state":source,"rate_coefficient":k,"unit":"mmol/min"})
    for s in monos:
        for compartment in ["lumen","cell","portal","delivered"]:
            key=f"meal:{compartment}:{s}"
            if key not in sys.indices:state(key,f"{compartment} {SPECIES[s]['label']}")
        if s not in supplied:add_state_node(f"meal:lumen:{s}","lumen - "+SPECIES[s]["label"],"Intestinal monosaccharides",1)
        add_state_node(f"meal:cell:{s}","intracellular - "+SPECIES[s]["label"],"Intestinal epithelial cells",2,cell_types=["Small intestinal absorptive epithelial cells"])
        add_state_node(f"meal:portal:{s}","Portal vein · "+SPECIES[s]["label"],"Portal vein",2)
    # Cumulative water participation is an independent reaction ledger, not
    # double counted as carbohydrate carbon or as measured whole-body fluid.
    sys.state("hydrolysis_water","Water consumed by hydrolysis","mmol",0,"intestinal_metabolism")
    sys.state("incretin","Incretin stimulation","dimensionless",0,"enteroendocrine")
    ix=sys.indices;volume_lumen=.35;volume_cell=.10;volume_portal=.30
    gastric_k=math.log(2)/settings.gastric_half_min
    absorption_scale=25/settings.absorption_half_min
    def rates(t,y):
        val=lambda comp,s:max(0,y[ix[f"meal:{comp}:{s}"]])
        r={"gastric":{},"hydrolysis":{},"apical":{},"basal":{},"clear":{}}
        for s in supplied:
            r["gastric"][s]=gastric_k*val("stomach",s)*bus.multiplier("gastric_emptying",t,y)
            if SPECIES[s]["products"]:
                k=.12 if s!="starch" else .035
                factor=settings.sucrase_scale*bus.multiplier("sucrase_capacity",t,y) if s=="sucrose" else 1
                r["hydrolysis"][s]=k*val("lumen",s)*factor
        concentrations={s:val("lumen",s)/volume_lumen for s in monos}
        sodium_factor=140**2/(140**2+25**2)
        vmax_sglt=1.2*absorption_scale*settings.sglt1_scale*bus.multiplier("sglt1_capacity",t,y)
        # Glucose and galactose compete for the same carrier capacity.
        for s in ["glucose","galactose"]:
            r["apical"][s]=vmax_sglt*concentrations[s]/(1.8+concentrations["glucose"]+concentrations["galactose"])*sodium_factor
        r["apical"]["fructose"]=.9*absorption_scale*settings.glut5_scale*bus.multiplier("glut5_capacity",t,y)*concentrations["fructose"]/(6+concentrations["fructose"])
        total_cell=sum(val("cell",s)/volume_cell for s in monos)
        total_portal=sum(val("portal",s)/volume_portal for s in monos)
        for s in monos:
            # Reversible facilitated transport shares carrier capacity. Portal
            # pools track dietary tracer material, not baseline blood glucose.
            c=val("cell",s)/volume_cell;p=val("portal",s)/volume_portal
            r["basal"][s]=2.4*absorption_scale*settings.glut2_scale*bus.multiplier("glut2_capacity",t,y)*(c-p)/(12+total_cell+total_portal)
            r["clear"][s]=.35*val("portal",s)
        return r
    vg=settings.body_mass_kg*2
    def process(t,y,dy):
        r=rates(t,y)
        for s,flux in r["gastric"].items():dy[ix[f"meal:stomach:{s}"]]-=flux;dy[ix[f"meal:lumen:{s}"]]+=flux
        for s,flux in r["hydrolysis"].items():
            dy[ix[f"meal:lumen:{s}"]]-=flux
            for product,stoich in SPECIES[s]["products"].items():dy[ix[f"meal:lumen:{product}"]]+=stoich*flux
            if s!="carbohydrate":dy[ix["hydrolysis_water"]]+=flux
        for s in monos:
            a=r["apical"][s];b=r["basal"][s];c=r["clear"][s]
            dy[ix[f"meal:lumen:{s}"]]-=a;dy[ix[f"meal:cell:{s}"]]+=a-b
            dy[ix[f"meal:portal:{s}"]]+=b-c;dy[ix[f"meal:delivered:{s}"]]+=c
        # Separate fructose and galactose hepatic conversion assumptions.
        ra=r["clear"]["glucose"]+.30*r["clear"]["fructose"]+.80*r["clear"]["galactose"]
        dy[ix["glucose"]]+=ra*180.156/vg
        stimulus=(r["apical"]["glucose"]+r["apical"]["galactose"])/(.5+r["apical"]["glucose"]+r["apical"]["galactose"])
        dy[ix["incretin"]]+=(stimulus-y[ix["incretin"]])/8
    sys.process("digestion_transport",process)
    observations={
        "gastric":lambda r:sum(r["gastric"].values()),"hydrolysis":lambda r:sum(r["hydrolysis"].values()),
        "sucrase":lambda r:r["hydrolysis"].get("sucrose",0),"sglt1":lambda r:r["apical"]["glucose"]+r["apical"]["galactose"],
        "glut5":lambda r:r["apical"]["fructose"],"glut2":lambda r:sum(r["basal"].values()),
        "nak":lambda r:2/3*(r["apical"]["glucose"]+r["apical"]["galactose"]),
        "appearance":lambda r:(r["clear"]["glucose"]+.3*r["clear"]["fructose"]+.8*r["clear"]["galactose"])*180.156,
        "absorbed_energy":lambda r:sum(r["clear"].values())*180.156/1000*4,
    }
    for name,fn in observations.items():sys.observe("flux:"+name,"mg/min" if name=="appearance" else "kcal/min" if name=="absorbed_energy" else "mmol/min",lambda t,y,fn=fn:fn(rates(t,y)))
    sys.observe("absorbed_glucose","mg",lambda t,y:180.156*y[ix["meal:delivered:glucose"]])
    sys.observe("gut_glucose","mg",lambda t,y:180.156*y[ix["meal:lumen:glucose"]])
    sys.observe("meal_energy","kcal",lambda t,y:sum(y[ix[f"meal:delivered:{s}"]] for s in monos)*180.156/1000*4)
    processes=[
        ("digestion:gastric","Gastric emptying","Stomach",1,"gastric",["Gastric smooth muscle cells"],["J = ln(2)/t½ · A_stomach · M_gastric"],[]),
        ("transport:sglt1","SGLT1 · intestinal influx","Intestinal epithelial cells",2,"sglt1",["Small intestinal absorptive epithelial cells · apical membrane"],["Jg = Vmax · Cg/(Km+Cg+Cgal) · Na²/(KNa²+Na²)","JNa = 2 · (Jg + Jgal)"],[{"name":"Km_glucose","value":1.8,"unit":"mM","status":"source_model"},{"name":"Vmax_segment","value":1.2,"unit":"mmol/min","status":"assumption"},{"name":"SGLT1 scale","value":settings.sglt1_scale,"unit":"×","status":"user_input"}]),
        ("transport:glut5","GLUT5 · fructose influx","Intestinal epithelial cells",2,"glut5",["Small intestinal absorptive epithelial cells · apical membrane"],["Jf = Vmax_f · Cf/(Km_f+Cf)"],[{"name":"Km_f","value":6,"unit":"mM","status":"assumption"},{"name":"Vmax_f","value":.9,"unit":"mmol/min","status":"assumption"}]),
        ("transport:glut2","GLUT2 · portal transport","Intestinal epithelial cells",2,"glut2",["Small intestinal absorptive epithelial cells · basolateral membrane"],["Js = Vmax · (Cs_cell−Cs_portal)/(Km+ΣC_cell+ΣC_portal)"],[{"name":"Km_reference","value":12,"unit":"mM","status":"source_model"},{"name":"Vmax_segment","value":2.4,"unit":"mmol/min","status":"assumption"},{"name":"GLUT2 scale","value":settings.glut2_scale,"unit":"×","status":"user_input"}]),
        ("transport:nak","Na⁺/K⁺-ATPase","Intestinal epithelial cells",2,"nak",["Small intestinal absorptive epithelial cells · basolateral membrane"],["J_ATP = (2/3) · J_SGLT1"],[]),
        ("liver:meal","Liver · monosaccharide processing","Liver",3,"appearance",["Hepatocytes"],["Ra = Jg + f_f→g · Jf + f_gal→g · Jgal","dG/dt += Ra · 180.156/Vg"],[{"name":"f_fructose_to_glucose","value":.3,"unit":"fraction","status":"assumption"},{"name":"f_galactose_to_glucose","value":.8,"unit":"fraction","status":"assumption"}]),
    ]
    enzymes=list(dict.fromkeys(SPECIES[s].get("enzyme") for s in supplied if SPECIES[s]["products"]))
    for enzyme in enzymes:
        matching=[s for s in supplied if SPECIES[s].get("enzyme")==enzyme]
        obs="enzyme:"+enzyme
        sys.observe(obs,"mmol/min",lambda t,y,ss=matching:sum(rates(t,y)["hydrolysis"].get(s,0) for s in ss))
        nid="digestion:"+enzyme
        label={"sucrase":"Sucrase · sucrose hydrolysis","lactase":"Lactase · lactose breakdown","maltase":"Maltase · maltose breakdown","amylase":"Amylase · starch breakdown","equivalent":"Carbohydrate · glucose equivalization"}[enzyme]
        nodes.append(make_node(nid,label,"Brush border enzymes",1,"assumption","Uses component-specific reaction stoichiometry.",state_key=obs,unit="mmol/min",process_id=nid,cell_types=["Small intestinal absorptive epithelial cells · brush border"],equations=[f"{SPECIES[s]['label']} + H₂O → "+" + ".join(f"{n} {SPECIES[p]['label']}" for p,n in SPECIES[s]["products"].items()) for s in matching],sources=[STOICH_SOURCE],limitations=["Enzyme reaction rate is an uncorrected first-order approximation.","Water participating in hydrolysis is distinguished from the original ingested sugar mass."]))
        for s in matching:
            edges.append({"source":f"meal:lumen:{s}","target":nid,"kind":"transport","label":"Breakdown","flux_key":obs,"unit":"mmol/min"})
            for p in SPECIES[s]["products"]:edges.append({"source":nid,"target":f"meal:lumen:{p}","kind":"transport","label":"Monosaccharide production","flux_key":obs,"unit":"mmol/min"})
    for nid,label,group,stage,obs,cells,equations,params in processes:
        nodes.append(make_node(nid,label,group,stage,"assumption","An execution process that changes adjacent mass or shared physiological state within the same integration step.",state_key="flux:"+obs,unit="mg/min" if obs=="appearance" else "mmol/min",process_id=nid,cell_types=cells,equations=equations,parameters=params,sources=[TRANSPORT_SOURCE,CELL_SOURCE] if nid.startswith("transport:") else [STOICH_SOURCE],limitations=["This is a reduced mechanism module, not a numerical reproduction of the full original model.","Organ-scale Vmax, volume, and conversion ratios are uncalibrated assumptions.","The Na/K pump calculates demand stoichiometry and does not integrate full membrane potential or ion concentrations."] if obs=="nak" else ["This is a reduced mechanism module, not a numerical reproduction of the full original model.","Organ-scale Vmax, volume, and conversion ratios are uncalibrated assumptions."]))
    nodes.append(make_node("incretin","GLP-1/GIP · secretion stimulation","Enteroendocrine cells",3,"assumption","The stimulus calculated from SGLT1 influx is incorporated into the beta-cell secretion equation.",state_key="incretin",unit="1",process_id="incretin",cell_types=["Intestinal L cells","Intestinal K cells"],equations=["dH/dt = [J_SGLT1/(0.5+J_SGLT1) − H]/8","insulin secretion *= (1 + 0.6H)"],limitations=["This is a pooled stimulus state that does not distinguish the actual concentrations of the two hormones."]))
    for s in supplied:
        edges.extend([{"source":f"meal:stomach:{s}","target":"digestion:gastric","kind":"transport","label":"Gastric contents"},{"source":"digestion:gastric","target":f"meal:lumen:{s}","kind":"transport","label":"Excretion"}])
    for s in monos:
        apical="transport:glut5" if s=="fructose" else "transport:sglt1"
        edges.extend([{"source":f"meal:lumen:{s}","target":apical,"kind":"transport","label":"Cellular uptake"},{"source":apical,"target":f"meal:cell:{s}","kind":"transport","label":"Influx"},{"source":f"meal:cell:{s}","target":"transport:glut2","kind":"transport","label":"Basolateral transport"},{"source":"transport:glut2","target":f"meal:portal:{s}","kind":"transport","label":"Portal vein"},{"source":f"meal:portal:{s}","target":"liver:meal","kind":"transport","label":"Hepatic influx"}])
    edges.extend([{"source":"transport:sglt1","target":"transport:nak","kind":"regulation","label":"Na⁺ gradient maintenance demand"},{"source":"transport:sglt1","target":"incretin","kind":"regulation","label":"Nutrient sensing"},{"source":"incretin","target":"insulin","kind":"regulation","label":"Secretion regulation"},{"source":"liver:meal","target":"glucose","kind":"transport","label":"Appearance in blood","flux_key":"flux:appearance","unit":"mg/min"}])
    for i,e,s,amount in doses:
        sys.impulse(e.start_min,f"meal:mouth:{s}",amount)
        edges.append({"source":f"event:{i}","target":f"meal:mouth:{s}","kind":"input","label":f"{amount:.2f} mmol {SPECIES[s]['label']}"})
    notes.extend(["SGLT1/GLUT2 topology and Km reference values follow the Afshar study, but Vmax and volumes were not recalibrated to human organ scale - a reduced model.","1 mol of sucrose/lactose/maltose hydrolyzes into 2 mol of monosaccharides. Participating water is recorded separately and digestion/transport conservation is checked in hexose-equivalent moles.","Fructose->glucose 30% and galactose->glucose 80% are lumped hepatic conversion assumptions; remaining carbon stays in amounts routed to other metabolic pathways.","The observed incretin is a dimensionless secretion stimulation, not a GLP-1/GIP concentration."])
    exposures={i:(lambda t,y: min(1,sys.value("flux:sglt1",t,y)/1.2),"transport:sglt1") for i,_,_,_ in doses}
    ledger={"name":"Hexose-equivalent ledger of ingested sugars","keys":keys,"weights":weights,"unit":"mmol (hexose equivalent)","doses":[(e.start_min,amount*SPECIES[s]["hexoses"]) for _,e,s,amount in doses]}
    return {**empty,"ledgers":[ledger],"exposures":exposures}
