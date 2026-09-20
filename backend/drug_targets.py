"""Static drug-target matching: wire looked-up experimental targets onto body-network nodes.

ChEMBL mechanism_of_action/action_type and activity-assay (IC50/Ki) target names are
matched against TARGET_ANCHORS below. On a match the drug's plasma-exposure signal is
added to the target expression of that body module (or coupling port); downstream
propagation is deterministic over the declared node graph. The LLM is not involved.

Sign rule: inhibitor/antagonist/blocker set target activity to -1; agonist/activator
to +1. anchor_sign is 'does that target's activity raise (+1) or lower (-1) this
module', so the signal applied to a module is sign(action) x anchor_sign x gain x exposure.


These values are not occupancy estimates; they are a static assumption that
'a linked target belongs to this module', flagged as an assumption on each node.
"""
import re

# ChEMBL action_type -> sign applied to target activity.
ACTION_SIGNS = {
    "inhibitor": -1, "antagonist": -1, "blocker": -1, "inverse agonist": -1,
    "negative allosteric modulator": -1, "antisense inhibitor": -1,
    "degrader": -1, "chelating agent": -1, "stabiliser": -1, "stabilizer": -1,
    "agonist": 1, "partial agonist": 1, "activator": 1, "opener": 1,
    "positive allosteric modulator": 1, "potentiator": 1, "releasing agent": 1,
    # Process-level actions used by built-in substance target profiles:
    "substrate": 1,      # feeds a process/pool (iron -> iron stores)
    "neutralizer": -1,   # chemically removes a pool (antacid -> gastric acid)
}

# (target-name regex, [(target, anchor_sign, optional relative gain)], note)
# target is a body-module id or "port:<coupling port>".
TARGET_ANCHORS = [
    # antipyresis / anti-inflammatory / analgesia — prostaglandin pathway
    (r"cyclooxygenase-?1|prostaglandin.*synthase.*\b1\b|ptgs1|cox-?1\b",
     [("prostaglandin",1),("platelet",1),("pyrogen",1)], "COX-1: platelet TXA2 and gastric-mucosal PG"),
    (r"cyclooxygenase-?2|ptgs2|cox-?2\b|prostaglandin.*synthase.*\b2\b",
     [("prostaglandin",1),("pyrogen",1),("nfkb",1),("platelet",-1,.5),("coronary_flow",1,.4),("pulmonary_vr",-1,.4)],
     "COX-2: inflammatory PG + endothelial PGI2 (antiplatelet arm weaker than platelet COX-1/TXA2); selective blockade -> prothrombotic shift"),
    (r"cyclooxygenase|prostaglandin.?endo|ptgs\b", [("prostaglandin",1),("platelet",1),("pyrogen",1)],
     "nonselective COX -> PG / platelets / fever pathway"),
    # cardiovascular / kidney / RAAS
    (r"angiotensin[- ]converting|angiotensin converting|dipeptidyl carboxypeptidase|\bace\b",
     [("angiotensin",1),("aldosterone",1)], "ACE: AngII and aldosterone production"),
    (r"angiotensin (ii )?(type.?1 |at1 )?receptor|agtr1|at1 receptor",
     [("aldosterone",1),("peripheral_perfusion",-1)], "AT1 receptor: aldosterone and vasoconstriction"),
    (r"\brenin\b|renin inhibitor", [("renin",1)], "renin"),
    (r"beta-?1 adrenergic|adrenergic.{0,12}beta-?1|adrb1|beta-?1 adrenoceptor", [("contractility",1),("hrv",-1)], "beta1: contractility / heart rate"),
    (r"beta-?2 adrenergic|adrenergic.{0,12}beta-?2|adrb2|beta-?2 adrenoceptor", [("airway",-1),("lipolysis",1)], "beta2: bronchodilation / lipolysis"),
    (r"beta-?3 adrenergic|adrenergic.{0,12}beta-?3|adrb3", [("lipolysis",1),("browning",1)], "β3: Lipolysis"),
    (r"alpha-?1[a-z]? adrenergic|adrenergic.{0,12}alpha-?1[a-z]?|adra1", [("peripheral_perfusion",-1),("preload",1,.4),("micturition",-1,.5)], "alpha1: vasoconstriction, venous tone, bladder-neck contraction"),
    (r"l-type calcium channel|calcium channel|cacna1|cacnb|dihydropyridine receptor",
     [("contractility",1),("peripheral_perfusion",-1)], "L-type Ca channel: contraction and vascular tone"),
    (r"\bherg\b|kcnh2|kv11", [("arrhythmia_risk",-1)], "hERG: channel activity maintains repolarization; blockade -> arrhythmia risk"),
    (r"\benac\b|scnn1|epithelial sodium channel", [("enac",1),("urine_output",-1)], "ENaC: Na Reabsorption"),
    (r"slc12a1|nkcc2|na-k-2cl|sodium.{0,20}potassium.{0,20}chloride.{0,20}cotransporter|na.{0,15}k.{0,15}cl.{0,15}cotransporter", [("urine_output",-1),("sodium_load",1),("k_depletion",-1,.5)], "NKCC2 (loop diuretics): Na/K/2Cl reabsorption; blockade wastes K"),
    (r"slc12a3|ncc\b|thiazide-sensitive", [("urine_output",-1),("sodium_load",1)], "NCC (thiazide target)"),
    (r"avpr2|vasopressin v2", [("aqp2",1),("adh",1)], "V2 receptor: AQP2"),
    (r"avpr1", [("peripheral_perfusion",-1)], "V1 receptor: vasoconstriction"),
    (r"slc5a2|sglt-?2|sodium-glucose.*2", [("port:renal_reabsorption",1),("urine_output",-1,.5),("sodium_load",1,.4)], "SGLT2: glucose/Na reabsorption; inhibitors cause osmotic diuresis"),
    (r"carbonic anhydrase|carbonate dehydratase", [("bicarbonate",1),("urine_output",-1)], "carbonic anhydrase"),
    # lipid / metabolism
    (r"hmg-coa reductase|hmgcr|hydroxymethylglutaryl", [("cholesterol_syn",1),("vldl",1)], "HMG-CoA reductase (statins)"),
    (r"npc1l1|niemann-pick c1-like", [("cholesterol_syn",1,.5)], "NPC1L1: intestinal cholesterol absorption (approx.)"),
    (r"pparg|peroxisome proliferator.*gamma", [("insulin_resistance",-1),("lipogenesis",1)], "PPARγ: insulin sensitivity"),
    (r"ppara|peroxisome proliferator.*alpha", [("lipolysis",1),("vldl",-1)], "PPARalpha: fatty-acid oxidation / VLDL"),
    (r"hcar2|gpr109a|niacin receptor|hydroxycarboxylic acid receptor 2",
     [("lipolysis",-1),("skin_bloodflow",-1)], "HCAR2: lipolysis suppression and flushing"),
    (r"amp-activated|prkaa|ampk\b", [("ampk",1),("gluconeogenesis",-1,.5),("insulin_resistance",-1,.4)], "AMPK (metformin): suppresses hepatic glucose production"),
    (r"mitochondrial (respiratory chain )?complex i|nadh.*dehydrogenase|m&gpd|glycerophosphate dehydrogenase|gpd2",
     [("oxidation",1),("lactate",-1),("atp_demand",-1)], "mitochondrial complex I / mGPD2 (metformin, phenformin): oxidative phosphorylation; inhibition -> lactate accumulation"),
    (r"kcnj11|abcc8|atp-sensitive potassium|sulfonylurea receptor|kir6",
     [("port:insulin_secretion",-1)], "KATP (sulfonylureas): closure triggers insulin secretion"),
    (r"insulin receptor|\binsr\b", [("muscle_glucose_uptake",1),("insulin_resistance",-1),("lipolysis",-1)], "insulin receptor: uptake, sensitivity, lipolysis suppression via AKT-PDE3B"),
    (r"glp-?1 receptor|glp1r|glucagon-like peptide 1", [("satiety",1),("appetite",-1),("motility",-1),("insulin_resistance",-1,.4)],
     "GLP-1: satiety and delayed gastric emptying"),
    (r"gipr\b|gastric inhibitory", [("satiety",1)], "GIP receptor"),
    (r"dipeptidyl peptidase-?4|\bdpp-?4\b", [("satiety",-1)], "DPP-4: incretin breakdown"),
    (r"pnlip|pancreatic lipase|lipase\b", [("lipase",1),("gi_fluid_loss",-1,.4)], "pancreatic lipase (orlistat): inhibition causes steatorrhea"),
    (r"alpha-glucosidase|mgam|glucosidase", [("intestinal_gas",-1),("amylase",-1)], "alpha-glucosidase: carbohydrate breakdown"),
    (r"xdh\b|xanthine (oxidase|dehydrogenase)", [("acid_load",1),("renal_acid",1)], "xanthine oxidase (allopurinol) -> urate approximation"),
    # neural / psychiatric
    (r"serotonin transporter|slc6a4|sert\b|5-htt", [("serotonin_tone",-1),("emetic_drive",-1,.4)], "SERT (SSRI): reuptake; blockade raises tone and causes early nausea"),
    (r"dopamine transporter|slc6a3|dat\b", [("dopamine_reward",-1),("arousal",-1,.4)], "DAT: dopamine reuptake; blockers raise reward and arousal"),
    (r"norepinephrine transporter|slc6a2|net\b", [("sympathetic",-1),("arousal",-1),("pain",1,.4)], "NET: NE reuptake; blockade raises sympathetic tone and descending analgesia"),
    (r"dopamine d2|drd2|d2 receptor|d2.{0,8}dopamine|d\(2\)", [("dopamine_reward",1),("prolactin",-1),("emetic_drive",1),("motility",-1,.5)], "D2: reward, prolactin suppression, CTZ emesis, GI motility inhibition"),
    (r"dopamine d1|drd1", [("dopamine_reward",1)], "D1 receptor"),
    (r"5-ht2a|htr2a|serotonin 2a", [("serotonin_tone",1),("mood",1)], "5-HT2A"),
    (r"5-ht1b|htr1b|5-ht1d|htr1d|serotonin 1[bd]", [("headache",-1)], "5-HT1B/D (triptans): migraine"),
    (r"5-ht3|htr3|serotonin 3", [("emetic_drive",1)], "5-HT3 (ondansetron): nausea"),
    (r"neurokinin 1|tacr1|nk1 receptor|substance p", [("emetic_drive",1)], "NK1: emetic reflex"),
    (r"calcitonin gene-related|cgrp|calcrl|ramp1", [("headache",1)], "CGRP: migraine mediator"),
    (r"opioid receptor.*mu|oprm1|mu opioid|mu-opioid|\bmu receptor",
     [("pain",-1),("ventilation",-1),("motility",-1),("gi_fluid_loss",-1,.5),("parasympathetic",1,.4)], "mu opioid: analgesia / respiratory depression / constipation / reduced secretion"),
    (r"opioid receptor|oprk1|oprd1|kappa opioid|delta opioid", [("pain",-1)], "kappa/delta opioid: analgesia"),
    (r"gaba-?a|gabra|gamma-aminobutyric.*\ba\b", [("arousal",-1),("parasympathetic",1,.4),("ventilation",-1,.3)], "GABA-A (benzodiazepines): inhibitory, mild respiratory depression"),
    (r"gaba-?b|gabbr", [("arousal",-1),("pain",-1)], "GABA-B"),
    (r"nmda|grin1|grin2|n-methyl-d-aspartate", [("neuroinflammation",1),("memory_consolidation",1),("pain",1,.6),("arousal",1,.4)], "NMDA (memantine, ketamine): excitatory transmission, nociception"),
    (r"acetylcholinesterase|ache\b|butyrylcholinesterase|bche\b",
     [("parasympathetic",-1),("cognitive_load",-1,.5)], "cholinesterase: acetylcholine breakdown"),
    (r"muscarinic|chrm[1-5]?", [("parasympathetic",1)], "muscarinic receptor"),
    (r"chrm3|m3 muscarinic", [("airway",1)], "M3 agonism -> bronchoconstriction (resistance increase)"),
    (r"nicotinic|chrna|chrnb", [("parasympathetic",1),("sympathetic",1)], "nicotinic receptor"),
    (r"adenosine receptor|adora1|adora2", [("adenosine",1),("arousal",-1),("sympathetic",-1,.4)], "adenosine receptor (caffeine antagonism)"),
    (r"orexin|hcrtr|hypocretin", [("orexin",1),("arousal",1)], "orexin receptor (suvorexant antagonism)"),
    (r"melatonin receptor|mtnr1|mt1|mt2", [("melatonin",1),("sleep_pressure",1,.6)], "melatonin receptor"),
    (r"histamine h1|hrh1|h1 receptor", [("histamine",1),("arousal",1,.5),("appetite",-1,.5)], "H1: wake-promoting; blockers sedate and increase appetite"),
    (r"histamine h2|hrh2|h2 receptor", [("gastric_acid",1)], "H2: Gastric acid secretion"),
    (r"hydrogen/potassium|h\+/k\+|atp4a|atp4b|proton pump", [("gastric_acid",1)], "proton pump (PPI)"),
    (r"sodium channel.*nav|scn9a|scn10a|scn1a|scn5a|voltage-gated sodium",
     [("pain",1),("arrhythmia_risk",1,.4)], "Na channels (local anesthetics/antiarrhythmics)"),
    (r"gaba transaminase|abat\b", [("arousal",1)], "GABA-breakdown enzyme"),
    (r"monoamine oxidase a|\bmaoa\b", [("serotonin_tone",-1),("dopamine_reward",-1)], "MAO-A: monoamine breakdown"),
    (r"monoamine oxidase b|\bmaob\b", [("dopamine_reward",-1)], "MAO-B"),
    # Immune and inflammation
    (r"tumor necrosis factor|tnf-alpha|tnfa|\btnf\b", [("tnf",1),("nfkb",1)], "TNF-alpha (anti-TNF biologic)"),
    (r"interleukin-?6|il-?6|il6r", [("il6",1),("crp",1)], "IL-6 axis"),
    (r"interleukin-?1|il-?1|il1r", [("il1",1),("pyrogen",1)], "IL-1 axis"),
    (r"interleukin-?17|il-?17", [("neutrophil",1),("skin_damage",1,.4)], "IL-17: neutrophils and skin inflammation"),
    (r"interleukin-?23|il-?23|interleukin-?12.*40|il12b", [("neutrophil",1),("nfkb",1,.5)], "IL-12/23 axis"),
    (r"interleukin-?4|il-?4|il4r|interleukin-?13|il-?13", [("ige",1),("eosinophil",1)], "IL-4/13: type 2 inflammation"),
    (r"interleukin-?5|il-?5", [("eosinophil",1)], "IL-5: Eosinophils"),
    (r"immunoglobulin e|\bige\b", [("ige",1)], "IgE (omalizumab)"),
    (r"complement c5|complement component 5|\bc5\b", [("complement",1)], "complement C5"),
    (r"complement", [("complement",1)], "complement family"),
    (r"janus kinase|\bjak[1-3]?\b|tyk2", [("nfkb",1),("tcell",1)], "JAK: cytokine signaling"),
    (r"calcineurin|ppp3|cyclophilin|fkbp12|fk506", [("tcell",1)], "calcineurin (tacrolimus/cyclosporine)"),
    (r"mechanistic target of rapamycin|\bmtor\b|frap", [("mtor",1),("tcell",1,.5)], "mTOR (sirolimus)"),
    (r"ms4a1|cd20\b", [("bcell",1)], "CD20: B cells"),
    (r"pdcd1|pd-?1\b|cd274|pd-?l1|programmed death", [("tcell",-1)], "PD-1/PD-L1: T-cell inhibitory axis (activated when blocked)"),
    (r"ctla-?4", [("tcell",-1)], "CTLA-4: T-cell inhibitory axis"),
    (r"glucocorticoid receptor|nr3c1", [("cortisol",1),("nfkb",-1,.6),("tcell",-1,.6)],
     "glucocorticoid receptor: cortisol-mimetic / immunosuppression"),
    (r"mineralocorticoid receptor|nr3c2", [("aldosterone",1),("enac",1,.7),("kaliuresis",1,.6),("urine_output",-1,.4),("sodium_load",1,.4)], "MR (spironolactone): Na retention, K wasting, water retention"),
    (r"csf3r|g-csf|granulocyte colony", [("neutrophil",1),("leukocyte",1)], "G-CSF receptor (filgrastim)"),
    (r"epor\b|erythropoietin receptor|erythropoietin\b", [("epo",1),("erythropoiesis",1)], "EPO receptor"),
    (r"mpl\b|thrombopoietin", [("thrombopoiesis",1),("platelet",1,.6)], "TPO receptor"),
    (r"interferon (alpha|beta)|ifnar|ifna\b|ifnb\b", [("interferon",1)], "interferon axis"),
    (r"dihydrofolate reductase|dhfr|folate", [("tcell",1),("leukocyte",1)], "DHFR (methotrexate): cell proliferation"),
    (r"tubulin|tuba1|tubb\b", [("neutrophil",1),("leukocyte",1)], "microtubule (colchicine / chemotherapy)"),
    (r"leukotriene|cyslt|lta4h|alox5|lipoxygenase", [("airway_inflammation",1),("mucus",1)],
     "leukotriene (montelukast): airway inflammation"),
    (r"pde4|phosphodiesterase 4", [("airway_inflammation",-1),("nfkb",-1,.5)], "PDE4: cAMP Breakdown"),
    (r"pde5|phosphodiesterase 5", [("nitric_oxide",-1),("pulmonary_vr",1)], "PDE5: cGMP breakdown raises pulmonary tone; inhibitors dilate"),
    # Hemostasis · coagulation
    (r"coagulation factor xa|\bf10\b|factor x\b", [("thrombin",1)], "factor Xa (DOACs)"),
    (r"thrombin\b|\bf2\b", [("thrombin",1)], "thrombin (dabigatran)"),
    (r"vkorc1|vitamin k epoxide", [("thrombin",1),("anticoagulant",-1)], "VKORC1 (warfarin): vitamin-K factor production; activity is pro-coagulant"),
    (r"antithrombin|serpinc1", [("thrombin",-1),("anticoagulant",1)], "antithrombin (heparin cofactor): thrombin inhibition -> anticoagulation"),
    (r"p2ry12|p2y12|adp receptor", [("platelet",1)], "P2Y12 (clopidogrel)"),
    (r"itga2b|itgb3|glycoprotein iib|gpiib", [("platelet",1)], "GP IIb/IIIa"),
    (r"plasminogen|\bplg\b|tissue plasminogen|\bplat\b|tpa\b", [("fibrinolysis",1)], "plasminogen/tPA"),
    (r"von willebrand|\bvwf\b", [("vwf",1)], "vWF"),
    # pathogen proteins — approximated as pathogen-load decrease rather than host modules
    (r"penicillin-binding|transpeptidase|mura\b|fabi\b|folp|dihydropteroate|bacterial",
     [("nfkb",1),("pyrogen",1,.5)], "bacterial-target suppression -> assumed pathogen-load decrease"),
    (r"dna gyrase|topoisomerase iv|gyra|gyrb|parce|parc\b", [("nfkb",1)], "bacterial DNA gyrase"),
    (r"50s|30s|ribosomal|rrna|16s|23s", [("nfkb",1)], "bacterial ribosome"),
    (r"reverse transcriptase|viral polymerase|neuraminidase|integrase|viral protease|proteinase|spike protein|3cl",
     [("nfkb",1),("interferon",-1,.5)], "viral-target suppression -> assumed pathogen-load decrease"),
    (r"ergosterol|cyp51|lanosterol|glucan synthase|fks1|sterol 14",
     [("nfkb",1)], "fungal target inhibition -> assumed pathogen-load reduction"),
    (r"glutamate-gated chloride|glucl",
     [("nfkb",-1),("resolution",1)],
     "GluCl activation (ivermectin): helminth paralysis -> pathogen clearance"),
    (r"nitroreductase|ferredoxin|nitroimidazole",
     [("nfkb",-1),("resolution",1)],
     "metronidazole reduced by microbial nitroreductase -> DNA damage -> anaerobe clearance"),
    # bone / calcium / vitamins
    (r"fdps|farnesyl (pyro|di)phosphate|fpps", [("resorption",1)], "FDPS (bisphosphonates): osteoclasts"),
    (r"rankl|tnfsf11|tnfrsf11a|rank\b", [("resorption",1)], "RANKL/RANK (denosumab)"),
    (r"parathyroid hormone 1 receptor|pth1r|parathyroid hormone\b|pth\b",
     [("formation",1),("pth",1)], "PTH axis (teriparatide)"),
    (r"calcitonin receptor|calcr\b", [("resorption",-1)], "calcitonin receptor: osteoclast suppression"),
    (r"vitamin d receptor|\bvdr\b", [("calcitriol",1)], "VDR"),
    (r"estrogen receptor|esr1|esr2", [("gonadal_steroid",1),("resorption",-1,.5),("gonadotropin",-1,.5)], "estrogen receptor: suppresses bone resorption and gonadotropins"),
    (r"aromatase|cyp19", [("gonadal_steroid",1)], "aromatase: estrogen synthesis"),
    (r"5-alpha-reductase|srd5a", [("gonadal_steroid",1)], "5α-reductase: DHT approximation"),
    (r"androgen receptor|\bar\b", [("gonadal_steroid",1),("epo",1,.5)], "androgen receptor: raises erythropoietin/hematocrit"),
    (r"progesterone receptor|pgr\b", [("gonadal_steroid",1),("gonadotropin",-1,.5)], "progesterone receptor: negative feedback on gonadotropins"),
    (r"gnrh receptor|gnrhr|lhrh", [("gnrh",1),("gonadotropin",1)], "GnRH axis"),
    (r"thyroid hormone receptor|thra|thrb", [("thyroid",1)], "thyroid-hormone receptor"),
    (r"thyroid peroxidase|tpo\b", [("thyroid",1)], "TPO (antithyroid agent)"),
    (r"growth hormone receptor|ghr\b|growth hormone\b", [("gh",1),("igf1",1)], "GH axis"),
    (r"oxytocin receptor|oxtr", [("oxytocin",1),("milk_production",1)], "oxytocin receptor"),
    (r"vegf|kdr\b|flk1|vascular endothelial growth", [("angiogenesis",1)], "VEGF: angiogenesis"),
    (r"cftr|cystic fibrosis", [("mucus",-1),("airway",-1,.5)], "CFTR: mucus thinning -> resistance decrease"),
    (r"endothelin receptor|ednra|ednrb", [("endothelin",1),("pulmonary_vr",1)], "endothelin receptor"),
    (r"guanylate cyclase|gucy|soluble guanylyl", [("nitric_oxide",1)], "sGC: cGMP production"),
    (r"phosphodiesterase 3|pde3", [("contractility",-1),("platelet",-1,.5)], "PDE3 (cilostazol)"),
    # added — cardiovascular / kidney
    (r"sodium/potassium|na\+/k\+|atp1a|sodium-potassium pump", [("contractility",-1),("arrhythmia_risk",-1,.4)], "Na/K ATPase (digoxin): inhibition -> Ca increase -> contractility increase; pump loss is pro-arrhythmic"),
    (r"neprilysin|\bmme\b|enkephalinase", [("anp",-1),("adh",-1,.5)], "neprilysin (sacubitril): ANP breakdown"),
    (r"pcsk9|proprotein convertase", [("vldl",-1),("cholesterol_syn",1,.4)], "PCSK9: LDLR breakdown -> LDL clearance"),
    (r"coagulation factor xi|\bf11\b", [("thrombin",1)], "factor XI"),
    (r"protein c\b|proc\b|activated protein c", [("anticoagulant",1),("fibrinolysis",1,.4)], "protein C: intrinsic anticoagulation"),
    (r"thromboxane|tbxa2r|tp receptor", [("platelet",1)], "TXA2 receptor"),
    (r"prostacyclin|ptgir|prostaglandin i2|ip receptor", [("pulmonary_vr",-1),("platelet",-1)], "prostacyclin: vasodilation and platelet inhibition"),
    (r"alpha-?2[a-z]? adrenergic|adrenergic.{0,12}alpha-?2[a-z]?|adra2", [("sympathetic",-1),("arousal",-1,.5)], "alpha2 autoreceptor (clonidine): suppressed NE release, central sedation"),
    (r"alpha2delta|cacna2d|voltage-dependent calcium.*alpha2", [("pain",1),("arousal",1,.4)], "alpha2delta (gabapentin): nociceptive transmission"),
    (r"vesicular monoamine|slc18a2|vmat2", [("dopamine_reward",1)], "VMAT2: dopamine storage/release"),
    (r"catechol-o-methyl|comt\b", [("dopamine_reward",-1)], "COMT: dopamine breakdown"),
    (r"motilin receptor|mlnr", [("motility",1)], "motilin (erythromycin): gastrointestinal motility"),
    (r"guanylate cyclase c|guanylyl cyclase c|gucy2c|enterotoxin receptor", [("motility",1),("gi_fluid_loss",1,.5)], "GC-C (linaclotide): intestinal secretion/motility"),
    # added — neural / behavioral
    (r"5-ht1a|htr1a|serotonin 1a", [("serotonin_tone",1),("mood",1)], "5-HT1A (buspirone)"),
    (r"5-ht2c|htr2c|serotonin 2c", [("appetite",-1),("satiety",1)], "5-HT2C agonism suppresses appetite (lorcaserin mechanism)"),
    (r"cannabinoid.*1|cnr1|cb1 receptor", [("appetite",1),("pain",-1),("dopamine_reward",1,.5),("arousal",-1,.4)], "CB1: appetite, analgesia, reward, sedation"),
    (r"cannabinoid.*2|cnr2|cb2 receptor", [("macrophage",1),("resolution",1,.5)], "CB2: immune modulation"),
    (r"trpv1|vanilloid|capsaicin receptor", [("pain",1)], "TRPV1 (capsaicin): nociception / heat sensation"),
    (r"ampa|gria[1-4]", [("arousal",1),("memory_consolidation",1,.5)], "AMPA receptor"),
    (r"leptin receptor|lepr\b", [("satiety",1),("appetite",-1)], "leptin receptor"),
    (r"ghrelin receptor|ghsr|growth hormone secretagogue", [("appetite",1),("ghrelin",1)], "ghrelin receptor"),
    (r"amylin|calcr.*ramp|islet amyloid", [("satiety",1),("motility",-1,.5)], "amylin receptor"),
    (r"glucagon receptor|gcgr", [("glucagon",1),("glycogenolysis",1,.6)], "glucagon receptor"),
    (r"interleukin-?2|il-?2|il2r|cd25", [("tcell",1)], "IL-2 axis"),
    (r"csf2|gm-csf|granulocyte-macrophage", [("macrophage",1),("leukocyte",1)], "GM-CSF"),
    (r"interleukin-?33|il-?33|st2\b|tslp|thymic stromal", [("eosinophil",1),("ige",1)], "IL-33/TSLP: type 2 inflammation"),
    (r"aldosterone synthase|cyp11b2", [("aldosterone",1)], "aldosterone synthase"),
    (r"hsd11b1|11-beta-hydroxysteroid|11β-hydroxysteroid", [("cortisol",1)], "11β-HSD1: cortisol reactivation"),
    (r"nitric oxide synthase|\bnos[123]?\b|inos\b", [("nitric_oxide",1)], "NOS: NO production"),
    (r"cathepsin k|ctsk", [("resorption",1)], "cathepsin K: osteoclast enzyme"),
    (r"lhcgr|lh receptor|luteinizing|fshr|follicle.stimulating", [("gonadotropin",1),("gonadal_steroid",1)], "LH/FSH receptors"),
    (r"prolactin receptor|prlr|prolactin\b", [("prolactin",1),("milk_production",1)], "prolactin axis"),
    (r"metabotropic glutamate|grm[1-8]|mglur", [("arousal",1,.5),("pain",1,.5)], "mGluR"),
    (r"sodium-iodide|slc5a5|thyroid-stimulating|tshr\b", [("thyroid",1),("tsh",1)], "thyroid axis"),
    (r"lipoprotein lipase|\blpl\b", [("lpl",1),("vldl",-1)], "LPL: lipolysis and triglyceride clearance"),
    (r"acetyle|vh.?at|slc18a3", [("parasympathetic",1)], "ACh vesicular transport"),
    (r"arginase|nos.*compet|urease", [("urea_cycle",1)], "arginase: urea cycle"),
    # Process-level targets for effects with no single protein receptor
    # (documented pharmacology, still routed through this one map):
    (r"vasopressin release|adh secretion|avp release", [("adh",1)],
     "vasopressin release (process): ethanol suppresses it -> diuresis"),
    (r"alcohol dehydrogenase|cyp2e1", [("hepatic_load",1),("cyp450",1)],
     "ethanol metabolism (substrate): hepatic processing load"),
    (r"acetaldehyde", [("hepatic_load",1),("headache",1,.6),("glutathione",-1,.4)],
     "acetaldehyde (ethanol metabolite): toxicity load, flushing/headache"),
    (r"dopamine release|dopamine secretion", [("dopamine_reward",1)],
     "dopamine release (process): releaser drugs raise synaptic dopamine"),
    (r"gastric acidity|stomach acid", [("gastric_acid",1)],
     "gastric acid pool (process): neutralizers deplete it"),
    (r"iron availability|iron stores|serum ferritin|transferrin saturation",
     [("iron_store",1),("erythropoiesis",1,.5)], "iron availability (substrate): stores and erythropoiesis"),
    (r"magnesium homeostasis|mg2\+ balance|magnesium balance", [("magnesium",1)],
     "magnesium balance (substrate)"),
    (r"phosphocreatine|creatine phosphate|creatine kinase", [("atp_demand",-1),("muscle_glycogen",1,.5)],
     "phosphocreatine shuttle (substrate): buffers ATP demand"),
    (r"gpr120|ffar4|free fatty acid receptor 4", [("resolution",1),("nfkb",-1)],
     "GPR120/FFAR4 (omega-3): anti-inflammatory signaling, Oh 2010 Cell 140:687"),
    (r"aquaporin-?3|aqp3", [("gi_fluid_loss",-1)],
     "AQP3 (senna/bisacodyl): colonic water absorption; downregulation leaves luminal water"),
    (r"purine synthesis|thiopurine|hgprt|purine nucleotide", [("tcell",1),("leukocyte",1)],
     "purine synthesis (antimetabolite, azathioprine): lymphocyte proliferation"),
    (r"impdh|inosine monophosphate dehydrogenase", [("tcell",1),("bcell",1)],
     "IMPDH (mycophenolate): lymphocyte proliferation"),
    (r"toll-like|tlr[1-9]", [("nfkb",1),("interferon",1)],
     "TLR7/9 (hydroxychloroquine): innate immune signaling"),
    (r"sv2a|synaptic vesicle protein 2", [("arousal",1)],
     "SV2A (levetiracetam): synaptic vesicle release / excitability"),
    (r"gsk3|glycogen synthase kinase", [("mood",-1)],
     "GSK3 (lithium): activity linked to mood instability; inhibition stabilizes"),
    (r"potassium channel opener|vascular katp", [("peripheral_perfusion",-1)],
     "vascular KATP opening (minoxidil): VSMC hyperpolarization -> dilation"),
    (r"alpha-?amylase|salivary amylase|pancreatic amylase", [("amylase",1)],
     "amylase (acarbose): starch digestion"),
    (r"napqi|n-acetyl.*benzoquinone", [("hepatic_load",1),("glutathione",-1)],
     "NAPQI (acetaminophen metabolite): depletes glutathione"),
]

# Built-in substance -> molecular-target profiles (entity -> [(target, action, weight?)]).
# Each pair mirrors database-level pharmacology (IUPHAR/DrugBank-style target records).
# The targets route through TARGET_ANCHORS exactly like looked-up mechanisms,
# so no per-drug physiological answer is hardcoded here.
#
# Entry classes, in decreasing mechanistic strength:
#   - molecular targets (receptors, enzymes, transporters, channels, microbial
#     proteins) — the bulk of the table
#   - database-style mechanism phrases a ChEMBL/DrugBank MoA record can return
#     ("dopamine release", "vasopressin release") — kept because they are
#     mechanism claims, not outcome claims; their notes say so
#   - substrates/metabolites ("acetaldehyde", "phosphocreatine") and
#     physicochemical neutralization ("gastric acidity") — the honest mechanism
#     for nutrients and antacids, which have no receptor pharmacology
# Every entry carries literature refs in SUBSTANCE_SOURCES below.
SUBSTANCE_TARGETS = {
    'acarbose': [('alpha-glucosidase', 'inhibitor'), ('alpha-amylase', 'inhibitor', 0.5)],
    'acetaminophen': [('cyclooxygenase-2', 'inhibitor', 0.4), ('napqi', 'substrate')],
    'acyclovir': [('viral polymerase', 'inhibitor')],
    'adalimumab': [('tumor necrosis factor', 'inhibitor')],
    'adrenaline': [('beta-1 adrenergic', 'agonist'), ('beta-2 adrenergic', 'agonist'), ('alpha-1 adrenergic', 'agonist')],
    'albuterol': [('beta-2 adrenergic', 'agonist')],
    'albuterol sulfate': [('beta-2 adrenergic', 'agonist')],
    'alcohol': [('gaba-a receptor', 'positive allosteric modulator'), ('nmda', 'antagonist'), ('dopamine release', 'releasing agent'), ('vasopressin release', 'inhibitor'), ('alcohol dehydrogenase', 'substrate')],
    'alendronate': [('fdps', 'inhibitor')],
    'allopurinol': [('xanthine oxidase', 'inhibitor')],
    'alprazolam': [('gaba-a receptor', 'positive allosteric modulator')],
    'amlodipine': [('l-type calcium channel', 'blocker')],
    'amoxicillin': [('penicillin-binding protein', 'inhibitor')],
    'amphetamine': [('dopamine transporter', 'inhibitor'), ('norepinephrine transporter', 'inhibitor'), ('vesicular monoamine', 'releasing agent')],
    'antacid': [('gastric acidity', 'neutralizer')],
    'apixaban': [('coagulation factor xa', 'inhibitor')],
    'aripiprazole': [('dopamine d2', 'partial agonist')],
    'aspirin': [('cyclooxygenase-1', 'inhibitor'), ('cyclooxygenase-2', 'inhibitor', 0.6)],
    'atenolol': [('beta-1 adrenergic', 'antagonist')],
    'atorvastatin': [('hmg-coa reductase', 'inhibitor')],
    'azathioprine': [('purine synthesis', 'inhibitor')],
    'azithromycin': [('50s ribosomal subunit', 'inhibitor')],
    'bevacizumab': [('vegf', 'inhibitor')],
    'bisacodyl': [('aquaporin-3', 'inhibitor')],
    'budesonide': [('glucocorticoid receptor', 'agonist')],
    'bupropion': [('dopamine transporter', 'inhibitor'), ('norepinephrine transporter', 'inhibitor')],
    'buspirone': [('5-ht1a', 'agonist')],
    'caffeine': [('adenosine receptor', 'antagonist'), ('phosphodiesterase 4', 'inhibitor', 0.3)],
    'cannabis': [('cannabinoid 1 receptor', 'agonist'), ('cannabinoid 2 receptor', 'agonist', 0.5)],
    'cetirizine': [('histamine h1', 'antagonist')],
    'ciprofloxacin': [('dna gyrase', 'inhibitor'), ('topoisomerase iv', 'inhibitor')],
    'citalopram': [('serotonin transporter', 'inhibitor')],
    'clonazepam': [('gaba-a receptor', 'positive allosteric modulator')],
    'clonidine': [('alpha-2 adrenergic', 'agonist')],
    'clopidogrel': [('p2y12', 'inhibitor')],
    'cocaine': [('dopamine transporter', 'inhibitor'), ('norepinephrine transporter', 'inhibitor'), ('serotonin transporter', 'inhibitor', 0.5)],
    'codeine': [('mu opioid', 'agonist', 0.6)],
    'colchicine': [('tubulin', 'inhibitor')],
    'creatine': [('phosphocreatine', 'substrate')],
    'cyclosporine': [('calcineurin', 'inhibitor')],
    'dabigatran': [('thrombin', 'inhibitor')],
    'dapagliflozin': [('sglt2', 'inhibitor')],
    'denosumab': [('rankl', 'inhibitor')],
    'desmopressin': [('vasopressin v2', 'agonist')],
    'dexamethasone': [('glucocorticoid receptor', 'agonist')],
    'diazepam': [('gaba-a receptor', 'positive allosteric modulator')],
    'digoxin': [('sodium-potassium pump', 'inhibitor')],
    'diphenhydramine': [('histamine h1', 'antagonist'), ('muscarinic', 'antagonist', 0.4)],
    'domperidone': [('dopamine d2', 'antagonist')],
    'donepezil': [('acetylcholinesterase', 'inhibitor')],
    'doxycycline': [('30s ribosomal subunit', 'inhibitor')],
    'dulaglutide': [('glp-1 receptor', 'agonist')],
    'duloxetine': [('serotonin transporter', 'inhibitor'), ('norepinephrine transporter', 'inhibitor')],
    'dupilumab': [('interleukin-4', 'inhibitor'), ('interleukin-13', 'inhibitor')],
    'dupixent': [('interleukin-4', 'inhibitor'), ('interleukin-13', 'inhibitor')],
    'dutasteride': [('5-alpha-reductase', 'inhibitor')],
    'eculizumab': [('complement c5', 'inhibitor')],
    'empagliflozin': [('sglt2', 'inhibitor')],
    'enalapril': [('angiotensin converting enzyme', 'inhibitor')],
    'epinephrine': [('beta-1 adrenergic', 'agonist'), ('beta-2 adrenergic', 'agonist'), ('alpha-1 adrenergic', 'agonist')],
    'epoetin': [('erythropoietin receptor', 'agonist')],
    'escitalopram': [('serotonin transporter', 'inhibitor')],
    'esomeprazole': [('proton pump', 'inhibitor')],
    'estradiol': [('estrogen receptor', 'agonist')],
    'etanercept': [('tumor necrosis factor', 'inhibitor')],
    'ethanol': [('gaba-a receptor', 'positive allosteric modulator'), ('nmda', 'antagonist'), ('dopamine release', 'releasing agent'), ('vasopressin release', 'inhibitor'), ('alcohol dehydrogenase', 'substrate'), ('acetaldehyde', 'substrate')],
    'ethinylestradiol': [('estrogen receptor', 'agonist')],
    'exenatide': [('glp-1 receptor', 'agonist')],
    'famotidine': [('histamine h2', 'antagonist')],
    'fentanyl': [('mu opioid', 'agonist')],
    'filgrastim': [('granulocyte colony', 'agonist')],
    'finasteride': [('5-alpha-reductase', 'inhibitor')],
    'fludrocortisone': [('mineralocorticoid receptor', 'agonist'), ('glucocorticoid receptor', 'agonist', 0.3)],
    'fluoxetine': [('serotonin transporter', 'inhibitor')],
    'fluticasone': [('glucocorticoid receptor', 'agonist')],
    'furosemide': [('nkcc2', 'inhibitor')],
    'gabapentin': [('alpha2delta', 'inhibitor')],
    'glimepiride': [('sulfonylurea receptor', 'inhibitor')],
    'glucocorticoid': [('glucocorticoid receptor', 'agonist')],
    'heparin': [('antithrombin', 'activator')],
    'humira': [('tumor necrosis factor', 'inhibitor')],
    'hydrochlorothiazide': [('thiazide-sensitive', 'inhibitor')],
    'hydrocortisone': [('glucocorticoid receptor', 'agonist'), ('mineralocorticoid receptor', 'agonist', .4)],
    'hydroxychloroquine': [('toll-like receptor', 'inhibitor')],
    'hydroxyzine': [('histamine h1', 'antagonist')],
    'ibuprofen': [('cyclooxygenase-1', 'inhibitor'), ('cyclooxygenase-2', 'inhibitor')],
    'infliximab': [('tumor necrosis factor', 'inhibitor')],
    'insulin': [('insulin receptor', 'agonist')],
    'ipratropium': [('m3 muscarinic', 'antagonist')],
    'iron': [('iron availability', 'substrate')],
    'ivermectin': [('glutamate-gated chloride channel', 'activator')],
    'ketamine': [('nmda', 'antagonist')],
    'keytruda': [('pd-1', 'inhibitor')],
    'lamotrigine': [('voltage-gated sodium channel', 'inhibitor')],
    'lasix': [('nkcc2', 'inhibitor')],
    'levetiracetam': [('sv2a', 'inhibitor')],
    'levothyroxine': [('thyroid hormone receptor', 'agonist')],
    'lidocaine': [('voltage-gated sodium channel', 'inhibitor')],
    'liraglutide': [('glp-1 receptor', 'agonist')],
    'lisinopril': [('angiotensin converting enzyme', 'inhibitor')],
    'lithium': [('gsk3', 'inhibitor')],
    'loperamide': [('mu opioid', 'agonist')],
    'loratadine': [('histamine h1', 'antagonist')],
    'lorazepam': [('gaba-a receptor', 'positive allosteric modulator')],
    'losartan': [('angiotensin ii type 1 receptor', 'antagonist')],
    'lovastatin': [('hmg-coa reductase', 'inhibitor')],
    'magnesium': [('magnesium homeostasis', 'substrate'), ('nmda', 'antagonist', 0.3)],
    'marijuana': [('cannabinoid 1 receptor', 'agonist'), ('cannabinoid 2 receptor', 'agonist', 0.5)],
    'mdma': [('serotonin transporter', 'inhibitor'), ('norepinephrine transporter', 'inhibitor'), ('dopamine transporter', 'inhibitor', 0.4)],
    'melatonin': [('melatonin receptor', 'agonist')],
    'metformin': [('amp-activated', 'activator'), ('mitochondrial complex i', 'inhibitor')],
    'methotrexate': [('dihydrofolate reductase', 'inhibitor')],
    'methylphenidate': [('dopamine transporter', 'inhibitor'), ('norepinephrine transporter', 'inhibitor')],
    'metoclopramide': [('dopamine d2', 'antagonist')],
    'metoprolol': [('beta-1 adrenergic', 'antagonist')],
    'metronidazole': [('nitroreductase', 'substrate')],
    'midazolam': [('gaba-a receptor', 'positive allosteric modulator')],
    'midodrine': [('alpha-1 adrenergic', 'agonist')],
    'minoxidil': [('potassium channel opener', 'activator')],
    'mirtazapine': [('alpha-2 adrenergic', 'antagonist'), ('histamine h1', 'antagonist'), ('5-ht2c', 'antagonist', 0.5)],
    'modafinil': [('dopamine transporter', 'inhibitor', 0.5)],
    'montelukast': [('leukotriene', 'antagonist')],
    'morphine': [('mu opioid', 'agonist')],
    'mounjaro': [('glp-1 receptor', 'agonist'), ('gastric inhibitory', 'agonist')],
    'mycophenolate': [('impdh', 'inhibitor')],
    'naloxone': [('mu opioid', 'antagonist')],
    'naltrexol': [('mu opioid', 'antagonist')],
    'naltrexone': [('mu opioid', 'antagonist')],
    'naproxen': [('cyclooxygenase-1', 'inhibitor'), ('cyclooxygenase-2', 'inhibitor')],
    'nicotine': [('nicotinic', 'agonist')],
    'nifedipine': [('l-type calcium channel', 'blocker')],
    'nivolumab': [('pd-1', 'inhibitor')],
    'norepinephrine': [('alpha-1 adrenergic', 'agonist'), ('beta-1 adrenergic', 'agonist')],
    'norethindrone': [('progesterone receptor', 'agonist')],
    'olanzapine': [('dopamine d2', 'antagonist'), ('histamine h1', 'antagonist'), ('5-ht2c', 'antagonist')],
    'omalizumab': [('immunoglobulin e', 'inhibitor')],
    'omega-3': [('gpr120', 'agonist')],
    'omeprazole': [('proton pump', 'inhibitor')],
    'ondansetron': [('5-ht3', 'antagonist')],
    'orlistat': [('pancreatic lipase', 'inhibitor')],
    'oseltamivir': [('neuraminidase', 'inhibitor')],
    'oxybutynin': [('muscarinic', 'antagonist')],
    'oxycodone': [('mu opioid', 'agonist')],
    'ozempic': [('glp-1 receptor', 'agonist')],
    'pantoprazole': [('proton pump', 'inhibitor')],
    'paracetamol': [('cyclooxygenase-2', 'inhibitor', 0.4), ('napqi', 'substrate')],
    'paroxetine': [('serotonin transporter', 'inhibitor')],
    'pembrolizumab': [('pd-1', 'inhibitor')],
    'phenylephrine': [('alpha-1 adrenergic', 'agonist')],
    'pioglitazone': [('pparg', 'agonist')],
    'prednisolone': [('glucocorticoid receptor', 'agonist'), ('mineralocorticoid receptor', 'agonist', .3)],
    'prednisone': [('glucocorticoid receptor', 'agonist'), ('mineralocorticoid receptor', 'agonist', .3)],
    'pregabalin': [('alpha2delta', 'inhibitor')],
    'progesterone': [('progesterone receptor', 'agonist'), ('gaba-a receptor', 'positive allosteric modulator', 0.4)],
    'propofol': [('gaba-a receptor', 'positive allosteric modulator')],
    'propranolol': [('beta-1 adrenergic', 'antagonist'), ('beta-2 adrenergic', 'antagonist')],
    'pseudoephedrine': [('alpha-1 adrenergic', 'agonist')],
    'quetiapine': [('dopamine d2', 'antagonist'), ('histamine h1', 'antagonist', 0.6)],
    'risedronate': [('fdps', 'inhibitor')],
    'risperidone': [('dopamine d2', 'antagonist'), ('5-ht2a', 'antagonist')],
    'rituximab': [('cd20', 'inhibitor')],
    'rivaroxaban': [('coagulation factor xa', 'inhibitor')],
    'rosuvastatin': [('hmg-coa reductase', 'inhibitor')],
    'sacubitril': [('neprilysin', 'inhibitor')],
    'salbutamol': [('beta-2 adrenergic', 'agonist')],
    'secukinumab': [('interleukin-17', 'inhibitor')],
    'semaglutide': [('glp-1 receptor', 'agonist')],
    'senna': [('aquaporin-3', 'inhibitor')],
    'sertraline': [('serotonin transporter', 'inhibitor')],
    'sildenafil': [('phosphodiesterase 5', 'inhibitor')],
    'simvastatin': [('hmg-coa reductase', 'inhibitor')],
    'sirolimus': [('mechanistic target of rapamycin', 'inhibitor')],
    'sitagliptin': [('dpp-4', 'inhibitor')],
    'spironolactone': [('mineralocorticoid receptor', 'antagonist')],
    'sumatriptan': [('5-ht1b', 'agonist'), ('5-ht1d', 'agonist')],
    'tacrolimus': [('calcineurin', 'inhibitor')],
    'tadalafil': [('phosphodiesterase 5', 'inhibitor')],
    'tamiflu': [('neuraminidase', 'inhibitor')],
    'tamsulosin': [('alpha-1 adrenergic', 'antagonist')],
    'teriparatide': [('parathyroid hormone 1 receptor', 'agonist')],
    'testosterone': [('androgen receptor', 'agonist')],
    'tetrahydrocannabinol': [('cannabinoid 1 receptor', 'agonist'), ('cannabinoid 2 receptor', 'agonist', 0.5)],
    'thc': [('cannabinoid 1 receptor', 'agonist'), ('cannabinoid 2 receptor', 'agonist', 0.5)],
    'theophylline': [('adenosine receptor', 'antagonist'), ('phosphodiesterase 4', 'inhibitor', 0.5)],
    'ticagrelor': [('p2y12', 'inhibitor')],
    'tiotropium': [('m3 muscarinic', 'antagonist')],
    'tirzepatide': [('glp-1 receptor', 'agonist'), ('gastric inhibitory', 'agonist')],
    'tobacco': [('nicotinic', 'agonist')],
    'tocilizumab': [('interleukin-6', 'inhibitor')],
    'tramadol': [('mu opioid', 'agonist', 0.5), ('serotonin transporter', 'inhibitor')],
    'trazodone': [('5-ht2a', 'antagonist'), ('serotonin transporter', 'inhibitor', 0.4), ('histamine h1', 'antagonist', 0.5)],
    'ustekinumab': [('interleukin-23', 'inhibitor')],
    'valproate': [('gaba transaminase', 'inhibitor'), ('voltage-gated sodium channel', 'inhibitor', 0.5)],
    'valsartan': [('angiotensin ii type 1 receptor', 'antagonist')],
    'varenicline': [('nicotinic', 'partial agonist')],
    'venlafaxine': [('serotonin transporter', 'inhibitor'), ('norepinephrine transporter', 'inhibitor')],
    'viagra': [('phosphodiesterase 5', 'inhibitor')],
    'vitamin d': [('vitamin d receptor', 'agonist')],
    'warfarin': [('vkorc1', 'inhibitor')],
    'wegovy': [('glp-1 receptor', 'agonist')],
    'xolair': [('immunoglobulin e', 'inhibitor')],
    'zofran': [('5-ht3', 'antagonist')],
    'zolpidem': [('gaba-a receptor', 'positive allosteric modulator')],
}

# when only activity assays exist without an action_type, these standard forms are assumed functional inhibition.

# Literature source records for each built-in substance -> target profile.
# Each entry documents where the molecular target/action claim comes from —
# FDA labels (DailyMed), IUPHAR/BPS Guide to Pharmacology target records, or
# primary literature. These are molecular-evidence citations, NOT claimed
# physiological outcomes; the body-level wiring still comes from TARGET_ANCHORS.
SUBSTANCE_SOURCES = {}


def _reg(refs, *names):
    for n in names:
        SUBSTANCE_SOURCES[n] = list(refs)


_IUPHAR = "IUPHAR/BPS Guide to Pharmacology (guidetopharmacology.org)"
_GG = "Goodman & Gilman's The Pharmacological Basis of Therapeutics, 13e"

# --- adrenergic / autonomic -------------------------------------------------
_reg([f"{_IUPHAR}: ADRB2 agonist", "FDA label (DailyMed): Ventolin/ProAir — beta2-adrenergic agonist", f"{_GG} ch.12"],
     'albuterol', 'albuterol sulfate', 'salbutamol')
_reg([f"{_IUPHAR}: ADRA1/ADRB1/ADRB2 agonist", "FDA label (DailyMed): epinephrine injection", f"{_GG} ch.12"],
     'adrenaline', 'epinephrine')
_reg([f"{_IUPHAR}: ADRA1/ADRB1 agonist", "FDA label (DailyMed): Levophed", f"{_GG} ch.12"], 'norepinephrine')
_reg([f"{_IUPHAR}: ADRB1 antagonist", "FDA label (DailyMed): Tenormin/Lopressor — beta1-selective blocker", f"{_GG} ch.12"],
     'atenolol', 'metoprolol')
_reg([f"{_IUPHAR}: ADRB1/ADRB2 antagonist", "FDA label (DailyMed): Inderal — nonselective beta blocker", f"{_GG} ch.12"], 'propranolol')
_reg([f"{_IUPHAR}: ADRA1 agonist", "FDA label (DailyMed): alpha1-adrenergic agonist", f"{_GG} ch.12"],
     'midodrine', 'phenylephrine', 'pseudoephedrine')
_reg([f"{_IUPHAR}: ADRA2 agonist", "FDA label (DailyMed): Catapres — central alpha2 agonist", f"{_GG} ch.12"], 'clonidine')
_reg([f"{_IUPHAR}: ADRA2/HRH1/HTR2C antagonist", "FDA label (DailyMed): Remeron", f"{_GG} ch.15"], 'mirtazapine')
_reg([f"{_IUPHAR}: muscarinic M3 antagonist (inhaled)", "FDA label (DailyMed): Atrovent/Spiriva", f"{_GG} ch.12"],
     'ipratropium', 'tiotropium')
_reg([f"{_IUPHAR}: muscarinic antagonist", "FDA label (DailyMed): Ditropan", f"{_GG} ch.12"], 'oxybutynin')
_reg([f"{_IUPHAR}: vascular KATP channel opener", "FDA label (DailyMed): minoxidil — arteriolar vasodilator", f"{_GG} ch.32"], 'minoxidil')

# --- cardiovascular / RAAS / renal ------------------------------------------
_reg([f"{_IUPHAR}: ACE (peptidyl-dipeptidase A) inhibitor", "FDA label (DailyMed): Vasotec/Zestril", f"{_GG} ch.26"],
     'enalapril', 'lisinopril')
_reg([f"{_IUPHAR}: AGTR1 (AT1) antagonist", "FDA label (DailyMed): Cozaar/Diovan", f"{_GG} ch.26"],
     'losartan', 'valsartan')
_reg([f"{_IUPHAR}: CaV1.2 L-type calcium channel blocker", "FDA label (DailyMed): Norvasc/Procardia", f"{_GG} ch.27"],
     'amlodipine', 'nifedipine')
_reg([f"{_IUPHAR}: SLC12A1 (NKCC2) inhibitor", "FDA label (DailyMed): Lasix — loop diuretic", f"{_GG} ch.29"],
     'furosemide', 'lasix')
_reg([f"{_IUPHAR}: SLC12A3 (NCC) inhibitor", "FDA label (DailyMed): hydrochlorothiazide", f"{_GG} ch.29"], 'hydrochlorothiazide')
_reg([f"{_IUPHAR}: mineralocorticoid receptor (NR3C2) antagonist", "FDA label (DailyMed): Aldactone", f"{_GG} ch.29"], 'spironolactone')
_reg([f"{_IUPHAR}: NR3C2/NR3C1 agonist", "FDA label (DailyMed): Florinef", f"{_GG} ch.46"], 'fludrocortisone')
_reg([f"{_IUPHAR}: Na+/K+-ATPase inhibitor", "FDA label (DailyMed): Lanoxin — positive inotrope", f"{_GG} ch.29"], 'digoxin')
_reg([f"{_IUPHAR}: neprilysin (NEP) inhibitor", "FDA label (DailyMed): Entresto (sacubitril/valsartan)", f"{_GG} ch.26"], 'sacubitril')

# --- lipid / metabolic / endocrine ------------------------------------------
_reg([f"{_IUPHAR}: HMGCR inhibitor", "FDA label (DailyMed): statin — HMG-CoA reductase inhibitor", f"{_GG} ch.31"],
     'atorvastatin', 'lovastatin', 'rosuvastatin', 'simvastatin')
_reg([f"{_IUPHAR}: SLC5A2 (SGLT2) inhibitor", "FDA label (DailyMed): Farxiga/Jardiance — incl. ketoacidosis warning", f"{_GG} ch.33"],
     'dapagliflozin', 'empagliflozin')
_reg([f"{_IUPHAR}: GLP1R agonist", "FDA label (DailyMed): incretin mimetic", f"{_GG} ch.33"],
     'dulaglutide', 'exenatide', 'liraglutide', 'semaglutide', 'ozempic', 'wegovy')
_reg([f"{_IUPHAR}: GIPR/GLP1R dual agonist", "FDA label (DailyMed): Mounjaro/Zepbound", "Jastreboff et al., NEJM 2022 (SURMOUNT-1)"],
     'tirzepatide', 'mounjaro')
_reg([f"{_IUPHAR}: DPP-4 inhibitor", "FDA label (DailyMed): Januvia", f"{_GG} ch.33"], 'sitagliptin')
_reg([f"{_IUPHAR}: KATP/SUR1 (ABCC8) closure", "FDA label (DailyMed): Amaryl — sulfonylurea", f"{_GG} ch.33"], 'glimepiride')
_reg(["FDA label (DailyMed): Glucophage — boxed warning: lactic acidosis",
      "Zhou et al., J Clin Invest 2001:108:1167 — AMPK activation",
      "Owen et al., Biochem J 2000:348:607 — mitochondrial complex I inhibition",
      "Madiraju et al., Nature 2014:510:542 — mitochondrial GPD2 inhibition",
      f"{_GG} ch.33"], 'metformin')
_reg([f"{_IUPHAR}: PPARG agonist", "FDA label (DailyMed): Actos", f"{_GG} ch.33"], 'pioglitazone')
_reg([f"{_IUPHAR}: INSR agonist", "FDA label (DailyMed): insulin products", f"{_GG} ch.33"], 'insulin')
_reg([f"{_IUPHAR}: TR (thyroid hormone receptor) agonist", "FDA label (DailyMed): Synthroid", f"{_GG} ch.41"], 'levothyroxine')
_reg([f"{_IUPHAR}: SRD5A (5-alpha-reductase) inhibitor", "FDA label (DailyMed): Proscar/Avodart", f"{_GG} ch.42"],
     'dutasteride', 'finasteride')
_reg([f"{_IUPHAR}: NR3C1 (glucocorticoid receptor) agonist", "FDA label (DailyMed): systemic/inhaled corticosteroid", f"{_GG} ch.46"],
     'budesonide', 'dexamethasone', 'fluticasone', 'glucocorticoid', 'hydrocortisone', 'prednisolone', 'prednisone')
for _n in ('hydrocortisone', 'prednisolone', 'prednisone'):
    SUBSTANCE_SOURCES[_n].append("Funder, N Engl J Med 1990;322:175 — cortisol/prednisolone mineralocorticoid activity -> Na retention, hypertension (dexamethasone negligible)")
_reg([f"{_IUPHAR}: ESR (estrogen receptor) agonist", "FDA label (DailyMed): estrogen products", f"{_GG} ch.42"],
     'estradiol', 'ethinylestradiol')
_reg([f"{_IUPHAR}: PGR (progesterone receptor) agonist", "FDA label (DailyMed): progestin products", f"{_GG} ch.42"],
     'progesterone', 'norethindrone')
_reg([f"{_IUPHAR}: AR (androgen receptor) agonist", "FDA label (DailyMed): testosterone products", f"{_GG} ch.42"], 'testosterone')
_reg([f"{_IUPHAR}: VDR (vitamin D receptor) agonist", "Institute of Medicine DRI report on calcium/vitamin D", f"{_GG} ch.44"], 'vitamin d')
_reg([f"{_IUPHAR}: PTH1R agonist", "FDA label (DailyMed): Forteo", f"{_GG} ch.44"], 'teriparatide')
_reg([f"{_IUPHAR}: EPOR agonist", "FDA label (DailyMed): Epogen/Procrit", f"{_GG} ch.41"], 'epoetin')
_reg([f"{_IUPHAR}: FDPS (farnesyl diphosphate synthase) inhibition in osteoclasts", "FDA label (DailyMed): Fosamax/Actonel", f"{_GG} ch.44"],
     'alendronate', 'risedronate')
_reg([f"{_IUPHAR}: RANKL neutralization", "FDA label (DailyMed): Prolia/Xgeva", f"{_GG} ch.44"], 'denosumab')
_reg([f"{_IUPHAR}: xanthine oxidase (XOR) inhibitor", "FDA label (DailyMed): Zyloprim", f"{_GG} ch.40"], 'allopurinol')
_reg([f"{_IUPHAR}: GPR120/FFAR4 agonist", "Oh et al., Cell 2010;140:687 — omega-3 anti-inflammatory signaling"], 'omega-3')
_reg(["Iron as substrate for heme synthesis — DRI/iron metabolism literature", f"{_GG} ch.41"], 'iron')
_reg(["Magnesium homeostasis — DRI electrolyte literature; NMDA receptor channel block", f"{_GG} ch.13"], 'magnesium')
_reg(["Phosphocreatine shuttle — creatine kinase substrate; ISSN position stand on creatine"], 'creatine')

# --- opioids / analgesia ------------------------------------------------------
_reg([f"{_IUPHAR}: OPRM1 (mu-opioid receptor) agonist", "FDA label (DailyMed): opioid agonist", f"{_GG} ch.20"],
     'codeine', 'fentanyl', 'morphine', 'oxycodone')
_reg([f"{_IUPHAR}: OPRM1 agonist + SLC6A4 inhibitor", "FDA label (DailyMed): Ultram", f"{_GG} ch.20"], 'tramadol')
_reg([f"{_IUPHAR}: peripheral OPRM1 agonist (poor CNS penetration)", "FDA label (DailyMed): Imodium", f"{_GG} ch.20"], 'loperamide')
_reg([f"{_IUPHAR}: OPRM1 antagonist", "FDA label (DailyMed): Narcan/Vivitrol", f"{_GG} ch.20"],
     'naloxone', 'naltrexone', 'naltrexol')

# --- neuro / psychiatric -----------------------------------------------------
_reg([f"{_IUPHAR}: GABAA positive allosteric modulator (benzodiazepine site)", "FDA label (DailyMed): benzodiazepine hypnotic/anxiolytic", f"{_GG} ch.19"],
     'alprazolam', 'clonazepam', 'diazepam', 'lorazepam', 'midazolam', 'zolpidem')
_reg([f"{_IUPHAR}: GABAA PAM (general anesthetic)", "FDA label (DailyMed): Diprivan", f"{_GG} ch.19"], 'propofol')
_reg([f"{_IUPHAR}: SLC6A4 (SERT) inhibitor", "FDA label (DailyMed): SSRI", f"{_GG} ch.15"],
     'citalopram', 'escitalopram', 'fluoxetine', 'paroxetine', 'sertraline')
_reg([f"{_IUPHAR}: SLC6A4/SLC6A2 inhibitor", "FDA label (DailyMed): SNRI", f"{_GG} ch.15"], 'duloxetine', 'venlafaxine')
_reg([f"{_IUPHAR}: SLC6A3/SLC6A2 inhibitor", "FDA label (DailyMed): Wellbutrin", f"{_GG} ch.15"], 'bupropion')
_reg([f"{_IUPHAR}: HTR2A antagonist + SERT inhibitor + HRH1 antagonist", "FDA label (DailyMed): trazodone", f"{_GG} ch.15"], 'trazodone')
_reg([f"{_IUPHAR}: HTR1A partial agonist", "FDA label (DailyMed): BuSpar", f"{_GG} ch.15"], 'buspirone')
_reg([f"{_IUPHAR}: SLC6A3/SLC6A2/VMAT reversal", "FDA label (DailyMed): amphetamine products", f"{_GG} ch.14"], 'amphetamine')
_reg([f"{_IUPHAR}: SLC6A3/SLC6A2 inhibitor", "FDA label (DailyMed): Ritalin", f"{_GG} ch.14"], 'methylphenidate')
_reg([f"{_IUPHAR}: SLC6A3/SLC6A2/SLC6A4 inhibitor", "Controlled substance; mechanism in DrugBank/reviews", f"{_GG} ch.14"], 'cocaine')
_reg([f"{_IUPHAR}: SLC6A4/SLC6A2/SLC6A3 reversal (releasing agent)", "Controlled substance; mechanism in reviews", f"{_GG} ch.14"], 'mdma')
_reg([f"{_IUPHAR}: SLC6A3 inhibitor (weak; wake-promoting)", "FDA label (DailyMed): Provigil", f"{_GG} ch.14"], 'modafinil')
_reg([f"{_IUPHAR}: DRD2 partial agonist", "FDA label (DailyMed): Abilify", f"{_GG} ch.16"], 'aripiprazole')
_reg([f"{_IUPHAR}: DRD2/HTR2A/HRH1/HTR2C antagonist", "FDA label (DailyMed): Zyprexa", f"{_GG} ch.16"], 'olanzapine')
_reg([f"{_IUPHAR}: DRD2/HRH1 antagonist", "FDA label (DailyMed): Seroquel", f"{_GG} ch.16"], 'quetiapine')
_reg([f"{_IUPHAR}: DRD2/HTR2A antagonist", "FDA label (DailyMed): Risperdal", f"{_GG} ch.16"], 'risperidone')
_reg([f"{_IUPHAR}: DRD2 antagonist (antiemetic/prokinetic)", "FDA label (DailyMed): Reglan/motilium", f"{_GG} ch.50"],
     'domperidone', 'metoclopramide')
_reg([f"{_IUPHAR}: GRIN (NMDA receptor) antagonist", "FDA label (DailyMed): Ketalar/esketamine", f"{_GG} ch.19"], 'ketamine')
_reg([f"{_IUPHAR}: alpha2delta (CACNA2D1) ligand", "FDA label (DailyMed): Neurontin/Lyrica", f"{_GG} ch.17"], 'gabapentin', 'pregabalin')
_reg([f"{_IUPHAR}: voltage-gated sodium channel blocker", "FDA label (DailyMed): Lamictal/lidocaine", f"{_GG} ch.17"],
     'lamotrigine', 'lidocaine')
_reg([f"{_IUPHAR}: SV2A ligand", "FDA label (DailyMed): Keppra", f"{_GG} ch.17"], 'levetiracetam')
_reg([f"{_IUPHAR}: GABA transaminase inhibitor + Na channel block", "FDA label (DailyMed): Depakote", f"{_GG} ch.17"], 'valproate')
_reg(["Klein & Melton, PNAS 1996;93:8455 — GSK3 inhibition by lithium", "FDA label (DailyMed): lithium carbonate", f"{_GG} ch.16"], 'lithium')
_reg([f"{_IUPHAR}: adenosine receptor antagonist + PDE inhibitor", f"{_GG} ch.14"], 'caffeine', 'theophylline')
_reg([f"{_IUPHAR}: nicotinic acetylcholine receptor agonist", f"{_GG} ch.13"], 'nicotine', 'tobacco')
_reg([f"{_IUPHAR}: alpha4beta2 nAChR partial agonist", "FDA label (DailyMed): Chantix", f"{_GG} ch.13"], 'varenicline')
_reg([f"{_IUPHAR}: CNR1/CNR2 (cannabinoid) agonist", "Controlled substance; mechanism in IUPHAR/reviews", f"{_GG} ch.14"],
     'cannabis', 'marijuana', 'thc', 'tetrahydrocannabinol')
_reg([f"{_IUPHAR}: MT1/MT2 (melatonin receptor) agonist", "Endogenous hormone; ramelteon label as reference", f"{_GG} ch.21"], 'melatonin')
_reg(["GABAA PAM + NMDA antagonism + mesolimbic dopamine release + ADH suppression — Lovinger et al., Alcohol Clin Exp Res reviews; ADH/acetaldehyde metabolism", f"{_GG} ch.23"],
     'ethanol', 'alcohol')

# --- allergy / GI -------------------------------------------------------------
_reg([f"{_IUPHAR}: HRH1 antagonist", "FDA label (DailyMed): second-generation antihistamine", f"{_GG} ch.49"],
     'cetirizine', 'hydroxyzine', 'loratadine')
_reg([f"{_IUPHAR}: HRH1/muscarinic antagonist", "FDA label (DailyMed): Benadryl", f"{_GG} ch.49"], 'diphenhydramine')
_reg([f"{_IUPHAR}: HRH2 antagonist", "FDA label (DailyMed): Pepcid", f"{_GG} ch.49"], 'famotidine')
_reg(["H+/K+-ATPase (gastric proton pump) irreversible inhibition", "FDA label (DailyMed): PPI", f"{_GG} ch.49"],
     'esomeprazole', 'omeprazole', 'pantoprazole')
_reg([f"{_IUPHAR}: HTR3A antagonist", "FDA label (DailyMed): Zofran", f"{_GG} ch.50"], 'ondansetron', 'zofran')
_reg(["Gastric H+ chemical neutralization (antacid pharmacology)", f"{_GG} ch.49"], 'antacid')
_reg([f"{_IUPHAR}: CysLT1 (leukotriene) receptor antagonist", "FDA label (DailyMed): Singulair", f"{_GG} ch.40"], 'montelukast')
_reg([f"{_IUPHAR}: ADRA1A antagonist (uroselective)", "FDA label (DailyMed): Flomax", f"{_GG} ch.12"], 'tamsulosin')
_reg(["Colonic AQP3 downregulation — Ikarashi et al., Biol Pharm Bull 2011 (sennoside/bisacodyl reduce aquaporin-3)", "FDA label (DailyMed): Dulcolax/Senokot", f"{_GG} ch.50"], 'bisacodyl', 'senna')
_reg([f"{_IUPHAR}: alpha-glucosidase/alpha-amylase inhibitor", "FDA label (DailyMed): Precose", f"{_GG} ch.33"], 'acarbose')
_reg(["PNLIP (pancreatic lipase) inhibitor", "FDA label (DailyMed): Xenical", f"{_GG} ch.31"], 'orlistat')

# --- pain / inflammation / coagulation ----------------------------------------
_reg([f"{_IUPHAR}: COX-1/COX-2 (PTGS1/2) inhibitor", "FDA label (DailyMed): NSAID", f"{_GG} ch.38"],
     'aspirin', 'ibuprofen', 'naproxen')
_reg(["Weak COX inhibition + NAPQI-mediated hepatotoxicity — FDA label (DailyMed): Tylenol; Hinz et al., FASEB J 2008", f"{_GG} ch.38"],
     'acetaminophen', 'paracetamol')
_reg([f"{_IUPHAR}: VKORC1 inhibitor", "FDA label (DailyMed): Coumadin", f"{_GG} ch.36"], 'warfarin')
_reg(["Antithrombin III potentiation", "FDA label (DailyMed): heparin", f"{_GG} ch.36"], 'heparin')
_reg([f"{_IUPHAR}: coagulation factor Xa inhibitor", "FDA label (DailyMed): Eliquis/Xarelto", f"{_GG} ch.36"],
     'apixaban', 'rivaroxaban')
_reg([f"{_IUPHAR}: thrombin (F2) inhibitor", "FDA label (DailyMed): Pradaxa", f"{_GG} ch.36"], 'dabigatran')
_reg([f"{_IUPHAR}: P2RY12 antagonist", "FDA label (DailyMed): Plavix/Brilinta", f"{_GG} ch.36"], 'clopidogrel', 'ticagrelor')
_reg([f"{_IUPHAR}: PDE5A inhibitor", "FDA label (DailyMed): Viagra/Cialis", f"{_GG} ch.27"],
     'sildenafil', 'tadalafil', 'viagra')
_reg([f"{_IUPHAR}: tubulin polymerization inhibitor", "FDA label (DailyMed): Colcrys", f"{_GG} ch.38"], 'colchicine')
_reg([f"{_IUPHAR}: acetylcholinesterase inhibitor", "FDA label (DailyMed): Aricept", f"{_GG} ch.12"], 'donepezil')
_reg([f"{_IUPHAR}: HTR1B/HTR1D agonist", "FDA label (DailyMed): Imitrex", f"{_GG} ch.21"], 'sumatriptan')
_reg([f"{_IUPHAR}: AVPR2 agonist", "FDA label (DailyMed): DDAVP", f"{_GG} ch.25"], 'desmopressin')

# --- antimicrobials (microbial molecular targets) ------------------------------
_reg(["Penicillin-binding proteins (transpeptidases) inhibition — FDA label (DailyMed): amoxicillin", f"{_GG} ch.57"], 'amoxicillin')
_reg(["Bacterial 50S ribosomal subunit inhibition — FDA label (DailyMed): Zithromax", f"{_GG} ch.59"], 'azithromycin')
_reg(["Bacterial DNA gyrase/topoisomerase IV inhibition — FDA label (DailyMed): Cipro", f"{_GG} ch.60"], 'ciprofloxacin')
_reg(["Bacterial 30S ribosomal subunit inhibition — FDA label (DailyMed): doxycycline", f"{_GG} ch.59"], 'doxycycline')
_reg(["Microbial nitroreductase activation -> DNA damage — FDA label (DailyMed): Flagyl", f"{_GG} ch.60"], 'metronidazole')
_reg(["Viral DNA polymerase inhibition (thymidine-kinase activated) — FDA label (DailyMed): Zovirax", f"{_GG} ch.61"], 'acyclovir')
_reg(["Influenza neuraminidase inhibition — FDA label (DailyMed): Tamiflu", f"{_GG} ch.61"], 'oseltamivir', 'tamiflu')
_reg(["Glutamate-gated chloride channel (GluCl) agonism — FDA label (DailyMed): Stromectol", f"{_GG} ch.60"], 'ivermectin')

# --- immunosuppression / biologics ---------------------------------------------
_reg(["Purine synthesis blockade via 6-MP metabolites (TPMT) — FDA label (DailyMed): Imuran", f"{_GG} ch.35"], 'azathioprine')
_reg(["Calcineurin (PP2B) inhibition via cyclophilin — FDA label (DailyMed): Neoral/Sandimmune", f"{_GG} ch.35"], 'cyclosporine')
_reg(["Calcineurin inhibition via FKBP12 — FDA label (DailyMed): Prograf", f"{_GG} ch.35"], 'tacrolimus')
_reg(["IMPDH inhibition — FDA label (DailyMed): CellCept", f"{_GG} ch.35"], 'mycophenolate')
_reg(["DHFR inhibition — FDA label (DailyMed): methotrexate", f"{_GG} ch.66"], 'methotrexate')
_reg(["FKBP12-mTORC1 inhibition — FDA label (DailyMed): Rapamune", f"{_GG} ch.35"], 'sirolimus')
_reg(["Endosomal TLR7/9 antagonism via lysosomotropism — FDA label (DailyMed): Plaquenil", f"{_GG} ch.41"], 'hydroxychloroquine')
_reg(["TNF-alpha neutralization — FDA label (DailyMed): Humira/Remicade/Enbrel", f"{_GG} ch.35"],
     'adalimumab', 'humira', 'infliximab', 'etanercept')
_reg(["VEGF-A neutralization — FDA label (DailyMed): Avastin", f"{_GG} ch.66"], 'bevacizumab')
_reg(["IL-4Ralpha blockade (IL-4/IL-13) — FDA label (DailyMed): Dupixent", f"{_GG} ch.35"], 'dupilumab', 'dupixent')
_reg(["Complement C5 blockade — FDA label (DailyMed): Soliris", f"{_GG} ch.35"], 'eculizumab')
_reg(["G-CSF receptor agonism — FDA label (DailyMed): Neupogen", f"{_GG} ch.41"], 'filgrastim')
_reg(["PD-1 blockade — FDA label (DailyMed): Keytruda/Opdivo", f"{_GG} ch.67"], 'keytruda', 'nivolumab', 'pembrolizumab')
_reg(["IgE neutralization — FDA label (DailyMed): Xolair", f"{_GG} ch.49"], 'omalizumab', 'xolair')
_reg(["CD20 B-cell depletion — FDA label (DailyMed): Rituxan", f"{_GG} ch.35"], 'rituximab')
_reg(["IL-17A neutralization — FDA label (DailyMed): Cosentyx", f"{_GG} ch.35"], 'secukinumab')
_reg(["IL-6 receptor blockade — FDA label (DailyMed): Actemra", f"{_GG} ch.35"], 'tocilizumab')
_reg(["IL-12/23 p40 neutralization — FDA label (DailyMed): Stelara", f"{_GG} ch.35"], 'ustekinumab')


INHIBITION_ASSAYS = {"IC50", "KI", "KD"}

_UNIT_TO_MOLAR = {"nm": 1e-9, "um": 1e-6, "µm": 1e-6, "μm": 1e-6,
                  "mm": 1e-3, "pm": 1e-12, "m": 1.}
_AFFINITY_TYPES = {"ki", "kd", "ic50", "ec50", "ac50"}


def parse_affinity_molar(act):
    """convert an assay standard value to molar; skipped when units are missing or only an inequality is given"""
    value = act.get("standard_value")
    units = (act.get("standard_units") or "").strip().casefold()
    kind = (act.get("standard_type") or "").strip().casefold()
    relation = (act.get("standard_relation") or "=").strip()
    if value is None or units not in _UNIT_TO_MOLAR or kind not in _AFFINITY_TYPES:
        return None
    k = float(value) * _UNIT_TO_MOLAR[units]
    # '>' means the true affinity is weaker, so the stated value is used conservatively.
    if relation not in ("=", "<", ">"):
        return None
    return k


# Assay organisms accepted as evidence for human physiology. Non-mammalian
# targets (plant, fungal, or bacterial enzymes — e.g. a soybean lipoxygenase
# assay on warfarin's ChEMBL record) must not drive human body modules;
# antimicrobial action still arrives via mechanism records/substance profiles.
MAMMALIAN_ASSAY_ORGANISMS = (
    "homo sapiens", "rattus", "mus musculus", "bos taurus", "canis",
    "oryctolagus", "cavia", "sus scrofa", "macaca", "mesocricetus",
    "meriones", "equus", "felis")


def _activity_usable(act):
    """An activity record counts as quantitative evidence only when it reports
    a measured value on a mammalian target. Records with no standard_value are
    target mentions, not potencies."""
    if act.get("standard_value") in (None, ""):
        return False
    organism = (act.get("target_organism") or "").casefold()
    return not organism or any(o in organism for o in MAMMALIAN_ASSAY_ORGANISMS)


# ---------------------------------------------------------------------------
# Molecular-target identity
#
# Activity records carry no target IDs in the cached data, so assays are
# pooled by a canonical target key derived from the target name. Two names
# map to the same protein only through the curated equivalence table below;
# otherwise the normalized name itself is the key and nothing pools across
# distinct names. Isoforms stay distinct (Alpha-1A vs Alpha-1B, COX-1 vs
# COX-2); a family-level record ("Cyclooxygenase", "Adrenergic receptor
# alpha-1") keeps a family key that never supplies member-level affinities,
# while a family-level target may aggregate member assays only as a flagged
# fallback. Destination body modules are never used to judge target identity.
# ---------------------------------------------------------------------------

# (canonical key, member-of family key or None, regex) — ordered, first hit wins.
_TARGET_EQUIV = [
    # --- eicosanoid enzymes ------------------------------------------------
    ("ptgs1", "ptgs_family", r"cyclooxygenase[\s\-_]?1\b|ptgs1|cox[\s\-_]?1\b|prostaglandin.*synthase.*\b1\b"),
    ("ptgs2", "ptgs_family", r"cyclooxygenase[\s\-_]?2\b|ptgs2|cox[\s\-_]?2\b|prostaglandin.*synthase.*\b2\b"),
    ("ptgs_family", None, r"cyclooxygenase|prostaglandin.*synthase|ptgs\b"),
    ("alox5", "alox_family", r"5[\s\-]?lipoxygenase|alox5|arachidonate 5[\s\-]?lipoxygenase"),
    ("alox12", "alox_family", r"12[\s\-]?lipoxygenase|alox12"),
    ("alox15", "alox_family", r"15[\s\-]?lipoxygenase|alox15"),
    ("alox_family", None, r"lipoxygenase|alox\b"),
    # --- adrenergic --------------------------------------------------------
    ("adra1a", "adra1_family", r"alpha[\s\-]?1a\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*alpha[\s\-]?1a\b|adra1a"),
    ("adra1b", "adra1_family", r"alpha[\s\-]?1b\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*alpha[\s\-]?1b\b|adra1b"),
    ("adra1d", "adra1_family", r"alpha[\s\-]?1d\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*alpha[\s\-]?1d\b|adra1d"),
    ("adra1_family", "adr_family", r"adrenergic receptor alpha[\s\-]?1\b|alpha[\s\-]?1 (adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*alpha[\s\-]?1\b(?!a|b|d)"),
    ("adra2a", "adra2_family", r"alpha[\s\-]?2a\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*alpha[\s\-]?2a\b|adra2a"),
    ("adra2b", "adra2_family", r"alpha[\s\-]?2b\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*alpha[\s\-]?2b\b|adra2b"),
    ("adra2c", "adra2_family", r"alpha[\s\-]?2c\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*alpha[\s\-]?2c\b|adra2c"),
    ("adra2_family", "adr_family", r"adrenergic receptor alpha[\s\-]?2\b|alpha[\s\-]?2 (adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*alpha[\s\-]?2\b(?!a|b|c)"),
    ("adrb1", "adrb_family", r"beta[\s\-]?1\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*beta[\s\-]?1\b|adrb1"),
    ("adrb2", "adrb_family", r"beta[\s\-]?2\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*beta[\s\-]?2\b|adrb2"),
    ("adrb3", "adrb_family", r"beta[\s\-]?3\b.*(adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*beta[\s\-]?3\b|adrb3"),
    ("adrb_family", "adr_family", r"adrenergic receptor beta\b|beta (adrenergic|adrenoceptor)|(adrenergic|adrenoceptor).*beta\b(?!.*[123])"),
    ("adr_family", None, r"adrenergic receptor\b|adrenoceptor\b"),
    # --- mitochondrial respiratory chain ------------------------------------
    # Complex I aliases incl. protein-complex subunit lists ("mitochondrially
    # encoded NADH:ubiquinone oxidoreductase core subunit N" x 40+ records).
    # mGPD2 / glycerophosphate dehydrogenase stays a distinct enzyme.
    ("mitochondrial_complex_i", None,
     r"mitochondrial (respiratory chain )?complex i\b|respiratory chain complex i\b|nadh[:\s]?ubiquinone oxidoreductase|nadh.*dehydrogenase|\bcomplex i\b"),
    # --- dopamine ----------------------------------------------------------
    ("drd1", "drd_family", r"dopamine d[\s\-]?1\b|d[\s\(]?1[\s\)]?dopamine|drd1"),
    ("drd2", "drd_family", r"dopamine d[\s\-]?2\b|d[\s\(]?2[\s\)]?dopamine|d2[\s\-]?like|drd2"),
    ("drd3", "drd_family", r"dopamine d[\s\-]?3\b|d[\s\(]?3[\s\)]?dopamine|drd3"),
    ("drd4", "drd_family", r"dopamine d[\s\-]?4\b|d[\s\(]?4[\s\)]?dopamine|drd4"),
    ("drd5", "drd_family", r"dopamine d[\s\-]?5\b|d[\s\(]?5[\s\)]?dopamine|drd5"),
    ("drd_family", None, r"dopamine receptor|dopamine d\b"),
    # --- muscarinic / nicotinic --------------------------------------------
    ("chrm1", "chrm_family", r"muscarinic.*receptor m[\s\-]?1\b|m1 muscarinic|chrm1"),
    ("chrm2", "chrm_family", r"muscarinic.*receptor m[\s\-]?2\b|m2 muscarinic|chrm2"),
    ("chrm3", "chrm_family", r"muscarinic.*receptor m[\s\-]?3\b|m3 muscarinic|chrm3"),
    ("chrm4", "chrm_family", r"muscarinic.*receptor m[\s\-]?4\b|m4 muscarinic|chrm4"),
    ("chrm5", "chrm_family", r"muscarinic.*receptor m[\s\-]?5\b|m5 muscarinic|chrm5"),
    ("chrm_family", None, r"muscarinic"),
    ("chrn_a4b2", "chrn_family", r"alpha4.*beta2|alpha4beta2|a4b2"),
    ("chrn_a7", "chrn_family", r"alpha[\s\-]?7\b.*nicotinic|nicotinic.*alpha[\s\-]?7"),
    ("chrn_family", None, r"nicotinic|neuronal acetylcholine receptor"),
    # --- serotonin -----------------------------------------------------------
    ("htr1a", "htr_family", r"5[\s\-]?ht[\s\-]?1a\b|5[\s\-]?hydroxytryptamine receptor 1a|serotonin.*1a\b|htr1a"),
    ("htr1b", "htr_family", r"5[\s\-]?ht[\s\-]?1b\b|serotonin.*1b\b|htr1b"),
    ("htr1d", "htr_family", r"5[\s\-]?ht[\s\-]?1d\b|serotonin.*1d\b|htr1d"),
    ("htr2a", "htr_family", r"5[\s\-]?ht[\s\-]?2a\b|serotonin 2a|serotonin.*2a\b|htr2a"),
    ("htr2c", "htr_family", r"5[\s\-]?ht[\s\-]?2c\b|serotonin 2c|serotonin.*2c\b|htr2c"),
    ("htr3a", "htr_family", r"5[\s\-]?ht[\s\-]?3a?\b|serotonin 3a?\b.*receptor|serotonin.*\(5[\s\-]?ht3\)|htr3a"),
    ("htr4", "htr_family", r"5[\s\-]?ht[\s\-]?4\b|htr4"),
    ("htr_family", None, r"serotonin.*receptor|5[\s\-]?ht.*receptor|hydroxytryptamine receptor"),
    ("slc6a4", None, r"serotonin transporter|sodium dependent serotonin transporter|sert\b|slc6a4|5[\s\-]?htt\b"),
    ("slc6a3", None, r"dopamine transporter|sodium dependent dopamine transporter|dat\b|slc6a3"),
    ("slc6a2", None, r"norepinephrine transporter|noradrenaline transporter|net\b|slc6a2"),
    # --- opioid --------------------------------------------------------------
    ("oprm", "opr_family", r"mu[\s\-]?(type\s+)?opioid|oprm|mu opioid receptor"),
    ("oprd", "opr_family", r"delta[\s\-]?(type\s+)?opioid|oprd"),
    ("oprk", "opr_family", r"kappa[\s\-]?(type\s+)?opioid|oprk"),
    ("opr_family", None, r"opioid receptor|opioid receptors"),
    # --- histamine / vasopressin / misc GPCR ---------------------------------
    ("hrh1", "hrh_family", r"histamine h[\s\-]?1\b|hrh1|h1 receptor"),
    ("hrh2", "hrh_family", r"histamine h[\s\-]?2\b|hrh2|h2 receptor"),
    ("hrh3", "hrh_family", r"histamine h[\s\-]?3\b|hrh3"),
    ("hrh4", "hrh_family", r"histamine h[\s\-]?4\b|hrh4"),
    ("hrh_family", None, r"histamine.*receptor"),
    ("avpr1a", "avpr_family", r"vasopressin v[\s\-]?1a\b|avpr1a|v1a receptor"),
    ("avpr2", "avpr_family", r"vasopressin v[\s\-]?2\b|avpr2|v2 receptor"),
    ("avpr_family", None, r"vasopressin.*receptor"),
    ("agtr1", "agtr_family", r"type[\s\-]?1.*angiotensin ii|angiotensin ii.*(type[\s\-]?1|at[\s\-]?1)|agtr1|at1 receptor"),
    ("agtr2", "agtr_family", r"type[\s\-]?2.*angiotensin ii|angiotensin ii.*(type[\s\-]?2|at[\s\-]?2)|agtr2|at2 receptor"),
    ("agtr_family", None, r"angiotensin ii receptor|angiotensin receptor"),
    ("tacr1", None, r"neurokinin 1|substance[\s\-]?p receptor|tacr1|nk1 receptor"),
    ("mlnr", None, r"motilin receptor|mlnr"),
    ("glp1r", None, r"glp[\s\-]?1 receptor|glucagon like peptide 1"),
    ("mtnr1a", "mtnr_family", r"melatonin.*(mt1|1a)|mt1 receptor|mtnr1a"),
    ("mtnr1b", "mtnr_family", r"melatonin.*(mt2|1b)|mt2 receptor|mtnr1b"),
    ("ptgir", None, r"prostacyclin receptor|ptgir|ip receptor|prostaglandin i2 receptor"),
    ("tbxa2r", None, r"thromboxane.*receptor|tbxa2r|tp receptor"),
    ("cnr1", "cnr_family", r"cannabinoid.*cb[\s\-]?1|cb1 receptor|cnr1"),
    ("cnr2", "cnr_family", r"cannabinoid.*cb[\s\-]?2|cb2 receptor|cnr2"),
    # --- nuclear receptors ---------------------------------------------------
    ("nr3c1", None, r"glucocorticoid receptor|nr3c1"),
    ("nr3c2", None, r"mineralocorticoid receptor|nr3c2"),
    ("esr1", "esr_family", r"estrogen receptor alpha|esr1|er alpha|eralpha"),
    ("esr2", "esr_family", r"estrogen receptor beta|esr2|er beta|erbeta"),
    ("esr_family", None, r"estrogen receptor"),
    ("ar", None, r"androgen receptor|\bar\b"),
    ("pgr", None, r"progesterone receptor"),
    ("ppara", "ppar_family", r"ppar.*alpha|peroxisome proliferator.*alpha|ppara"),
    ("pparg", "ppar_family", r"ppar.*gamma|peroxisome proliferator.*gamma|pparg"),
    ("ppard", "ppar_family", r"ppar.*delta|peroxisome proliferator.*delta|ppard"),
    ("ppar_family", None, r"peroxisome proliferator"),
    ("vdr", None, r"vitamin d receptor|vdr"),
    ("thr_family", None, r"thyroid hormone receptor"),
    ("rar_family", None, r"retinoic acid receptor|retinoic x receptor|rxr\b"),
    # --- enzymes / transporters / channels -----------------------------------
    ("hmgcr", None, r"hmg[\s\-]?coa reductase|3[\s\-]?hydroxy[\s\-]?3[\s\-]?methylglutaryl.*reductase|hydroxymethylglutaryl"),
    ("ache", "che_family", r"acetylcholinesterase|ache\b"),
    ("bche", "che_family", r"butyrylcholinesterase|bche|pseudocholinesterase"),
    ("pde5", "pde_family", r"phosphodiesterase 5|pde5|cgmp specific.*phosphodiesterase|cgmp specific 3 5 cyclic"),
    ("pde4", "pde_family", r"phosphodiesterase 4|pde4"),
    ("pde3", "pde_family", r"phosphodiesterase 3|pde3"),
    ("pde_family", None, r"phosphodiesterase"),
    ("dpp4", None, r"dipeptidyl peptidase (iv|4)\b|dpp[\s\-]?4|dipeptidyl peptidase iv"),
    ("ace", None, r"angiotensin[\s\-]converting enzyme|\bace\b"),
    ("ca1", "ca_family", r"carbonic anhydrase (i|1)\b(?!i|v|x)|carbonic anhydrase i\b"),
    ("ca2", "ca_family", r"carbonic anhydrase (ii|2)\b(?!i|v|x)|carbonic anhydrase ii\b"),
    ("ca4", "ca_family", r"carbonic anhydrase (iv|4)\b|carbonic anhydrase iv\b"),
    ("ca7", "ca_family", r"carbonic anhydrase (vii|7)\b|carbonic anhydrase vii\b"),
    ("ca12", "ca_family", r"carbonic anhydrase (xii|12)\b|carbonic anhydrase xii\b"),
    ("ca_family", None, r"carbonic anhydrase"),
    ("kcnh2", None, r"herg|kcnh2|inwardly rectifying.*kcnh"),
    ("cav_l", "cav_family", r"l[\s\-]?type calcium channel|cacna1|voltage gated l type|dihydropyridine receptor"),
    ("cav_family", None, r"voltage[\s\-]?gated calcium channel|calcium channel"),
    ("scn_family", None, r"sodium channel|voltage[\s\-]?gated sodium|nav\b|scn\d"),
    ("gabra_family", None, r"gaba[\s\-]?a|gamma aminobutyric acid a"),
    ("gabrb_family", None, r"gaba[\s\-]?b"),
    ("gria_family", "glu_family", r"ampa"),
    ("grik_family", "glu_family", r"kainate"),
    ("grin_family", "glu_family", r"nmda"),
    ("grm_family", "glu_family", r"metabotropic glutamate|mglur"),
    ("glu_family", None, r"glutamate receptor"),
    ("sgc", None, r"soluble guanylate cyclase|sgc\b"),
    ("gucy2c", None, r"heat stable enterotoxin receptor|guanylate cyclase c|gucy2c|gc[\s\-]?c\b"),
    ("slc12a1", None, r"sodium.*potassium.*chloride cotransporter 2|nkcc2|slc12a1|bumetanide sensitive"),
    ("slc12a3", None, r"thiazide sensitive.*cotransporter|sodium chloride cotransporter|ncc\b|slc12a3"),
    ("atp1a1", None, r"sodium.*potassium.*atpase|na k atpase|atp1a1|sodium potassium transporting"),
    ("slc5a2", None, r"sodium glucose cotransporter 2|sglt2|slc5a2"),
    ("p2ry12", None, r"p2y.*12|p2y purinoceptor 12|purinergic receptor p2y12|p2ry12"),
    ("f2a", "f2_family", r"\bthrombin\b"),
    ("f2", "f2_family", r"prothrombin|coagulation factor ii\b"),
    ("f10a", None, r"factor x[a]?\b|coagulation factor x\b|f10a"),
    ("plg", None, r"plasminogen\b(?!.*activator)|\bplasmin\b"),
    ("plat", None, r"plasminogen activator|tissue plasminogen|\btpa\b|alteplase"),
    ("vkorc1", None, r"vitamin k epoxide reductase|vkorc1"),
    ("npc1l1", None, r"niemann pick c1 like|npc1l1"),
    ("abcb1", "abc_family", r"atp dependent translocase abcb1|p[\s\-]?glycoprotein|abcb1|mdr1"),
    ("bsep", "abc_family", r"bile salt export pump|bsep|abcb11"),
    ("katp_family", None, r"atp sensitive potassium|katp|kir6|sulfonylurea receptor|abcc8|kcnj11"),
    ("adora1", "adora_family", r"adenosine receptor a[\s\-]?1\b|a1 adenosine|adora1"),
    ("adora2a", "adora_family", r"adenosine receptor a[\s\-]?2a\b|a2a adenosine|adora2a"),
    ("adora2b", "adora_family", r"adenosine receptor a[\s\-]?2b\b|a2b adenosine|adora2b"),
    ("adora3", "adora_family", r"adenosine receptor a[\s\-]?3\b|a3 adenosine|adora3"),
    ("adora_family", None, r"adenosine receptor"),
    ("lipase_gi", None, r"gastric lipase|pancreatic lipase|lipase\b"),
    ("maoa", "mao_family", r"monoamine oxidase a|mao[\s\-]?a\b|maoa"),
    ("maob", "mao_family", r"monoamine oxidase b|mao[\s\-]?b\b|maob"),
    ("comt", None, r"catechol[\s\-]?o[\s\-]?methyltransferase|comt"),
    ("xo", None, r"xanthine oxidase|xanthine dehydrogenase"),
    ("srd5a", None, r"5[\s\-]?alpha[\s\-]?reductase|srd5a|steroid 5 alpha reductase"),
    ("cyp19a1", None, r"aromatase|cyp19"),
    ("hsd11b", None, r"11[\s\-]?beta[\s\-]?hydroxysteroid dehydrogenase|hsd11b|11beta hsd"),
    ("ren", None, r"\brenin\b"),
    ("trpv1", None, r"trpv1|vanilloid receptor|capsaicin receptor"),
    ("sigmar1", None, r"sigma receptor|sigma[\s\-]?1|sigmar1"),
    ("oprl1", None, r"nociceptin|orl1|oprl1"),
    ("taar1", None, r"trace amine.*receptor|taar1"),
]

_FAMILY_OF = {canon: parent for canon, parent, _pat in _TARGET_EQUIV if parent}
_CANON_PATTERNS = [(canon, re.compile(pat, re.I)) for canon, _p, pat in _TARGET_EQUIV]

# Action words that close a mechanism phrase; the same vocabulary drives
# ACTION_SIGNS so stripping it never invents new target names.
_ACTION_SUFFIXES = sorted(set(ACTION_SIGNS) | {"blockade", "inhibition",
                        "agonism", "antagonism", "binding", "binder"},
                       key=len, reverse=True)
_TRAILING_ACTION = re.compile(
    r"(\s+(" + "|".join(re.escape(a) for a in _ACTION_SUFFIXES) + r"))+$", re.I)

# Target names that denote an organism or non-protein entity rather than a
# molecular target: they canonicalize to an isolated bucket that mechanism
# text can never hit, so they can never lend affinity to a protein anchor.
_ORGANISM_TARGET = re.compile(
    r"homo sapiens|rattus|mus musculus|oryctolagus|cavia|bos taurus|"
    r"sus scrofa|macaca|mesocricetus|meriones|equus|felis|canis|"
    r"plasmodium|trypanosoma|sars[\s\-]?cov|escherichia|staphylococc|"
    r"streptococc|mycobacter|candida|saccharomyces|soybean|zea mays|"
    r"unchecked|non[\s\-]?protein|cell\b|hepatocyte|erythrocyte|platelet\b",
    re.I)


def _canon(text):
    """Canonical molecular-target key from a target name or mechanism phrase.

    Returns None for non-protein/organism entities. Unknown proteins keep
    their normalized name as the identity (exact-name matching); only the
    curated equivalence table merges synonyms."""
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return None
    if _ORGANISM_TARGET.search(t):
        return "nonprotein:" + t
    for canon, pat in _CANON_PATTERNS:
        if pat.search(t):
            return canon
    stripped = _TRAILING_ACTION.sub("", t).strip()
    if stripped != t:
        for canon, pat in _CANON_PATTERNS:
            if pat.search(stripped):
                return canon
        return stripped or None
    return t


def _ancestors(key):
    """All family keys above `key` (member -> family -> family-of-family)."""
    out = []
    while key in _FAMILY_OF:
        key = _FAMILY_OF[key]
        out.append(key)
    return out


def _related_keys(key):
    """key, its ancestors, and all keys descending from those ancestors.

    Family-level functional evidence (e.g. an EC50 on 'Adrenergic receptor
    alpha-2') applies to its members; member-level evidence likewise pools
    upward only inside the same family tree."""
    rel = {key}
    for canon, parent in _FAMILY_OF.items():
        chain = {canon} | set(_ancestors(canon))
        if key in chain:
            rel |= chain
    rel |= set(_ancestors(key))
    return rel


def _assay_key(act, idmap=None):
    """Canonical target key for an assay record. Name canonicalization is
    primary; when the name does not resolve but the record carries a
    target_chembl_id that another record already linked to a canonical key,
    the ID supplies the identity — a database ID is authoritative over a
    name heuristic ('PTGS-2 protein' canonicalizes only to the family).
    Missing IDs are never guessed."""
    key = (idmap or {}).get(act.get("target_chembl_id"))
    return key or _canon(act.get("target_pref_name"))


def _chembl_idmap(mechanisms, activities):
    """target_chembl_id -> canonical key, from records whose NAME already
    canonicalizes. IDs are preserved and used only when an upstream record
    provides them; no ID is invented."""
    idmap = {}
    for rec in list(mechanisms or []) + list(activities or []):
        tid, key = rec.get("target_chembl_id"), _canon(rec.get("target_pref_name"))
        if tid and key:
            idmap.setdefault(tid, key)
    return idmap


def _assay_evidence(act, used, pool_level="self"):
    """Per-assay provenance retained on the anchor it informed."""
    return {"activity_id": act.get("activity_id"),
            "pool_level": pool_level,
            "assay_chembl_id": act.get("assay_chembl_id"),
            "document_chembl_id": act.get("document_chembl_id"),
            "target_pref_name": act.get("target_pref_name"),
            "target_organism": act.get("target_organism"),
            "standard_type": act.get("standard_type"),
            "standard_relation": act.get("standard_relation"),
            "standard_value": act.get("standard_value"),
            "standard_units": act.get("standard_units"),
            "value_m": parse_affinity_molar(act),
            "used_for_affinity": used}


# Kind-priority policy, declared a priori: equilibrium constants (Ki, Kd)
# are the least assay-condition-dependent potency measure, then functional
# inhibition (IC50), then functional potency (EC50/AC50). Different kinds
# are never merged into a single minimum.
_KIND_TIER = {"ki": 0, "kd": 0, "ic50": 1, "ec50": 2, "ac50": 2}


def _pick_potency(pool):
    """Choose the anchor potency inside a same-target assay pool.

    Within the best available kind, prefer human exact '=' values, then
    other mammalian exact values, then inequality bounds (flagged, never
    presented as point values). Returns (value_m, note, used_set)."""
    by_kind = {}
    for act in pool:
        kind = (act.get("standard_type") or "").casefold()
        if kind in _KIND_TIER:
            by_kind.setdefault(_KIND_TIER[kind], []).append(act)
    if not by_kind:
        return None, "no usable potency type", set()
    tier = by_kind[min(by_kind)]
    human, other, bounds = [], [], []
    for act in tier:
        k = parse_affinity_molar(act)
        if k is None:
            continue
        rel = (act.get("standard_relation") or "=").strip()
        org = (act.get("target_organism") or "").casefold()
        if rel == "=":
            (human if "homo sapiens" in org else other).append((k, act))
        else:
            bounds.append((k, act))
    if human:
        k = min(x for x, _ in human)
        return k, "human exact", {id(a) for kk, a in human if kk == k}
    if other:
        k = min(x for x, _ in other)
        orgs = sorted({(a.get("target_organism") or "?") for kk, a in other if kk == k})
        return k, "nonhuman exact: " + "/".join(orgs), {id(a) for kk, a in other if kk == k}
    if bounds:
        k = min(x for x, _ in bounds)
        rels = sorted({(a.get("standard_relation") or "?") for kk, a in bounds if kk == k})
        return k, "bound " + "/".join(rels), {id(a) for kk, a in bounds if kk == k}
    return None, "no numeric potency", set()


def _target_pool(activities, key, idmap=None):
    """Usable assays on canonical target `key`.

    A family-level target ('Cyclooxygenase', 'Adrenergic receptor alpha-2')
    pools its own assays plus member assays — the family claim covers its
    members, so member potencies are evidence about that target, flagged
    per row. A member-level target ('Alpha-1A', 'PTGS2') never receives
    family-level or sibling assays: a generic family measurement cannot be
    attributed to a specific protein."""
    usable = [a for a in (activities or []) if _activity_usable(a)]
    exact = [a for a in usable if _assay_key(a, idmap) == key]
    if key not in set(_FAMILY_OF.values()):
        return exact, ("same_target" if exact else None)
    members = [a for a in usable
               if _assay_key(a, idmap) and key in _ancestors(_assay_key(a, idmap))]
    pool = exact + members
    if pool:
        return pool, ("family_with_members" if members else "family_only")
    return [], None


def target_affinity(activities, key, idmap=None):
    """Best potency among assays on the SAME molecular target.

    Returns (affinity_m, note, evidence). affinity_m is None when no usable
    same-target assay exists — the gap is exposed, never filled by borrowing
    another target's value."""
    if not key:
        return None, "no_target_identity", []
    if key.startswith("nonprotein:"):
        return None, "non_protein_target", []
    pool, how = _target_pool(activities, key, idmap)
    if not pool:
        saw_same = any(_assay_key(a, idmap) == key for a in (activities or []))
        note = "same_target_assays_unusable" if saw_same else "no_same_target_assay"
        return None, note, []
    k, pnote, used_ids = _pick_potency(pool)
    ev = [_assay_evidence(a, id(a) in used_ids,
                          "self" if _assay_key(a, idmap) == key else "descendant")
          for a in pool]
    note = f"{how}|{pnote}"
    return k, note, ev


def _best_activity(activities, anchor_id):
    """DEPRECATED shim kept for callers: pools assays by destination module.

    New code must use target_affinity(), which pools by molecular target."""
    best = None
    for act in activities or []:
        if not _activity_usable(act):
            continue
        name = act.get("target_pref_name") or ""
        for _pat, alist, _note in _anchor_hit(name):
            if any((a[0][5:] if a[0].startswith("port:") else a[0]) == anchor_id for a in alist):
                k = parse_affinity_molar(act)
                if k is not None and (best is None or k < best):
                    best = k
    return best


def _anchor_hit(text):
    """find declared anchors whose patterns match the target-name text"""
    hits = []
    for pattern, anchors, note in TARGET_ANCHORS:
        if re.search(pattern, text, re.I):
            hits.append((pattern, anchors, note))
    return hits


def _key_anchor_hits(key, names):
    """Anchor hits for a canonical target: the canonical code itself plus
    every distinct raw name in the evidence group. Canonicalization knows
    synonyms the anchor table does not spell out (e.g. 'adrenoceptor beta 2'
    -> adrb2 -> the adrb2 anchor pattern), so identity — not one surface
    string — decides wiring. Order-free: names are sorted, hits deduped."""
    hits = _anchor_hit(key) if key and not key.startswith(("name:", "nonprotein:")) else []
    for nm in sorted({n for n in names if n}):
        hits.extend(_anchor_hit(nm))
    out, seen_p = [], set()
    for h in hits:
        if h[0] not in seen_p:
            seen_p.add(h[0])
            out.append(h)
    return out


def drug_anchors(events, compounds):
    """Convert each chemical event's lookup evidence into body-network anchors.

    Returns: [{"event_index", "kind":"module"|"port", "id", "sign", "gain",
    "target", "action", "note"}]. sign is the direction the drug applies to the
    target (action sign x anchor_sign). Targets with unknown direction are not wired.
    """
    anchors = []
    for i, e in enumerate(events):
        if e.kind != "chemical":
            continue
        c = compounds.get(str(i), {})
        acts = c.get("activities") or []
        idmap = _chembl_idmap(c.get("mechanisms"), acts)
        # Anchor identity = (canonical target_key, kind, ident, sign).
        # `wired_pair` tracks (target_key, kind, ident) already wired by a
        # higher-priority evidence class so the SAME molecular target is
        # never counted twice, while DIFFERENT targets reaching the same
        # destination remain independent contributions. Evidence priority:
        # mechanism of action > activity assay > built-in profile >
        # inferred (virtual-screening/similarity). Deterministic order:
        # every stage dedupes by key and the final list is sorted.
        seen, wired_pair = set(), set()
        mech_dirs = {}   # canonical target key -> declared action sign
        # 1) Mechanism of action: action_type provides direction.
        #    Affinity comes only from assays on the SAME molecular target —
        #    never pooled across targets that happen to share a body module.
        for mech in c.get("mechanisms", []):
            text = " ".join(str(mech.get(k) or "") for k in ("mechanism_of_action", "action_type"))
            sign = ACTION_SIGNS.get((mech.get("action_type") or "").strip().casefold())
            mkey = (_canon(mech.get("target_pref_name") or mech.get("target_name")
                          or mech.get("mechanism_of_action") or text)
                    or "name:" + (mech.get("target_pref_name") or text))
            aff, aff_note, aff_ev = target_affinity(acts, mkey, idmap)
            # Declared direction overrides the assay-default ONLY when the
            # mechanism record resolves a molecular target (target_pref_name /
            # target_name). A class-level MOA text with no named target
            # ("Adrenergic receptor agonist", target=null) does not establish
            # direction at each bound subtype — subtype engagement varies and
            # the conservative assay default must stand.
            if sign is not None and (mech.get("target_pref_name")
                                     or mech.get("target_name")):
                mech_dirs[mkey] = sign
            for _pat, alist, note in _key_anchor_hits(mkey, [text]):
                for a in alist:
                    target, anchor_sign = a[0], a[1]
                    weight = a[2] if len(a) > 2 else 1.
                    kind, ident = ("port", target[5:]) if target.startswith("port:") else ("module", target)
                    key = (mkey, kind, ident, (sign or 0) * anchor_sign)
                    if sign is None or key in seen: continue
                    seen.add(key)
                    wired_pair.add((mkey, kind, ident))
                    anchors.append({"event_index": i, "kind": kind, "id": ident,
                                    "sign": sign * anchor_sign, "gain": .8 * weight,
                                    "target": mech.get("mechanism_of_action") or text,
                                    "action": mech.get("action_type") or "",
                                    "affinity_m": aff, "target_key": mkey,
                                    "affinity_note": aff_note,
                                    "affinity_evidence": aff_ev,
                                    "target_chembl_id": mech.get("target_chembl_id"),
                                    "basis": "known", "note": note})
        # 2) Activity assays (IC50/Ki): grouped by molecular-target key, so two
        #    proteins that reach the same module keep separate affinities.
        #    A functional activation assay (EC50) on the same target — or on a
        #    family containing it — overrides the binding-displacement
        #    inhibition assumption: agonists also displace radioligand, so
        #    IC50 alone cannot decide direction (erythromycin on motilin,
        #    guanfacine on alpha-2, etc).
        agonist_keys = set()
        for act in acts:
            if (act.get("standard_type") or "").upper() != "EC50":
                continue
            if not _activity_usable(act):
                continue
            k = _assay_key(act, idmap)
            if k and not k.startswith("nonprotein:"):
                agonist_keys.add(k)
        assay_groups = {}
        for act in acts:
            if (act.get("standard_type") or "").upper() not in INHIBITION_ASSAYS:
                continue
            if not _activity_usable(act):
                continue
            k = _assay_key(act, idmap)
            if not k or k.startswith("nonprotein:"):
                continue
            assay_groups.setdefault(k, []).append(act)
        for k in sorted(assay_groups):
            group = assay_groups[k]
            aff, aff_note, aff_ev = target_affinity(acts, k, idmap)
            acts_as_agonist = bool(_related_keys(k) & agonist_keys)
            # A declared mechanism on a related target overrides the
            # assay's default inhibition assumption: agonists also show
            # binding displacement, so direction must come from the
            # mechanism record (e.g. morphine's kappa/delta assays are
            # agonist binding, not antagonism). Used only when every
            # related mechanism agrees on one sign.
            rel_dirs = {mech_dirs[kk] for kk in _related_keys(k)
                        if kk in mech_dirs}
            mech_sign = rel_dirs.pop() if len(rel_dirs) == 1 else None
            name = sorted({a.get("target_pref_name") or "" for a in group},
                          key=lambda s: -len(s))[0]
            for _pat, alist, note in _key_anchor_hits(
                    k, [a.get("target_pref_name") or "" for a in group]):
                for a in alist:
                    target, anchor_sign = a[0], a[1]
                    weight = a[2] if len(a) > 2 else 1.
                    direction = ((mech_sign * anchor_sign) if mech_sign
                                 else (anchor_sign if acts_as_agonist
                                      else -anchor_sign))
                    kind, ident = ("port", target[5:]) if target.startswith("port:") else ("module", target)
                    key = (k, kind, ident, direction)
                    if (k, kind, ident) in wired_pair:
                        continue  # mechanism already wired this same target+dest
                    if key in seen: continue
                    seen.add(key)
                    wired_pair.add((k, kind, ident))
                    anchors.append({"event_index": i, "kind": kind, "id": ident,
                                    "sign": direction, "gain": .55 * weight,
                                    "target": name,
                                    "action": ("assay -> agonism (EC50 evidence on same target/family)"
                                               if acts_as_agonist else
                                               "assay -> assumed inhibition"),
                                    "affinity_m": aff, "target_key": k,
                                    "affinity_note": aff_note,
                                    "affinity_evidence": aff_ev,
                                    "basis": "known", "note": note})
        # 3) Built-in substance->target profiles: database-level pharmacology
        # (IUPHAR/DrugBank-style records) routed through the same TARGET_ANCHORS
        # wiring as looked-up mechanisms. Deduped per module+sign so a ChEMBL
        # record naming the receptor does not shadow documented downstream effects.
        occupied = set()
        ent = (e.entity or "").casefold().strip()
        for alias, profile in SUBSTANCE_TARGETS.items():
            # Exact alias match on the interpreted entity; a padded word-boundary
            # match on the label is allowed for bare labels, but "iron" must
            # never match "spironolactone".
            label_pad = " " + e.label.casefold().strip() + " "
            if alias == ent or (not ent and " " + alias + " " in label_pad):
                for row in profile:
                    tname, action = row[0], row[1]
                    weight = row[2] if len(row) > 2 else 1.
                    sign = ACTION_SIGNS.get((action or "").casefold())
                    if sign is None: continue
                    tkey = _canon(tname) or "name:" + tname
                    aff, aff_note, aff_ev = target_affinity(acts, tkey, idmap)
                    for _pat, alist, note in _key_anchor_hits(tkey, [tname]):
                        for a in alist:
                            target, anchor_sign = a[0], a[1]
                            tg = a[2] if len(a) > 2 else 1.
                            kind = "port" if target.startswith("port:") else "module"
                            ident = target[5:] if target.startswith("port:") else target
                            asig = sign * anchor_sign
                            key = (tkey, kind, ident, asig)
                            if (tkey, kind, ident) in wired_pair or key in occupied:
                                continue  # same target+dest already wired
                            occupied.add(key)
                            wired_pair.add((tkey, kind, ident))
                            anchors.append({"event_index": i,
                                            "kind": kind,
                                            "id": ident, "sign": asig, "gain": .7 * tg * weight,
                                            "target": tname,
                                            "action": action + " (built-in target profile)",
                                            "affinity_m": aff, "target_key": tkey,
                                            "affinity_note": aff_note,
                                            "affinity_evidence": aff_ev,
                                            "basis": "known", "note": note,
                                            "sources": SUBSTANCE_SOURCES.get(alias, [])})
                break

        # 4) Inferred targets: virtual-screening / similarity results wired at a lower confidence tier.
        for source_key, conf_key, basis in (("vs_targets", "screen_score", "virtual_screening"),
                                            ("predicted_targets", "max_similarity", "virtual_screening")):
            for hit in c.get(source_key) or []:
                conf = float(hit.get(conf_key) or 0)
                name = hit.get("target_pref_name") or ""
                hkey = _canon(name) or "name:" + name
                for _pat, alist, note in _key_anchor_hits(hkey, [name]):
                    for a in alist:
                        target, anchor_sign = a[0], a[1]
                        w = a[2] if len(a) > 2 else 1.
                        kind = "port" if target.startswith("port:") else "module"
                        ident = target[5:] if target.startswith("port:") else target
                        if (hkey, kind, ident) in wired_pair:
                            continue  # same target+dest already has experimental evidence
                        key = (hkey, kind, ident, -anchor_sign)
                        if key in seen: continue
                        seen.add(key)
                        wired_pair.add((hkey, kind, ident))
                        anchors.append({"event_index": i, "kind": kind, "id": ident,
                                        "sign": -anchor_sign, "gain": .45 * conf * w,
                                        "target": name, "target_key": hkey,
                                        "action": f"{basis} inference -> assumed inhibition",
                                        "affinity_m": None, "basis": basis, "note": note})
        # 5) Selectivity weight: relative potency among targets with known affinity.
        #    w = Kmin/K_i — Hill approximation that weaker targets are less occupied at equal concentration.
        known = [a for a in anchors if a["event_index"] == i and a.get("affinity_m")]
        if len(known) > 1:
            kmin = min(a["affinity_m"] for a in known)
            for a in known:
                a["weight"] = round(max(0.05, min(1., kmin / a["affinity_m"])), 4)
    # 6) Duplicate-claim suppression: identical (destination, sign, profile
    #    note) anchors emitted under different target_keys are the SAME
    #    physiological claim only when the keys denote the same molecular
    #    entity — identical canonical key or an ancestor/descendant pair
    #    (family vs member, e.g. alpha-1 vs alpha-1A). Sibling subtypes
    #    (alpha-1A vs alpha-1B) and unrelated targets are independent
    #    contributions: blocking two receptors is not one blockade.
    #    Within a duplicate pair keep the deeper-level / better-evidenced key.
    deduped = []
    for a in anchors:
        ka = a.get("target_key") or ""
        merged = False
        for j, prev in enumerate(deduped):
            if (prev["event_index"], prev["kind"], prev["id"], prev["sign"],
                    prev.get("note")) != (a["event_index"], a["kind"], a["id"],
                                          a["sign"], a.get("note")):
                continue
            kb = prev.get("target_key") or ""
            if ka != kb and ka not in _ancestors(kb) and kb not in _ancestors(ka):
                continue
            if len(_ancestors(ka)) > len(_ancestors(kb)) or \
                    (a.get("affinity_m") is not None) > (prev.get("affinity_m") is not None):
                deduped[j] = a
            merged = True
            break
        if not merged:
            deduped.append(a)
    anchors = deduped
    # Deterministic order: the anchor multiset is a function of evidence
    # identity, never of record order.
    anchors.sort(key=lambda a: (a["event_index"], a["id"], a.get("target_key") or "",
                                a["sign"], a.get("basis") or "", a.get("target") or ""))
    return anchors
