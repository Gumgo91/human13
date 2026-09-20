"""Human13 whole-body physiology library - catalog of relative-deviation processes

with system/cell tags. Each process is a **directed transfer reaction**
carrying one signal into another. Target expressions are low-dimensional
signals equilibrating at 0 without stimulus. The kernel integrates the

shared ODE dx/dt = [3*tanh(u/3) - x]/tau. Design principle: the LLM only
decides where an input first attaches (channels, native states, ports).
The node-to-node links declared here are deterministic propagation paths

wired in code; once an initial anchor fires, the graph extends downstream
on its own. Timescale tau is a response delay, from seconds-fast neural
reflexes to days-long erythropoiesis, following real physiological time constants. These remain relative, uncalibrated exploratory priors, not measurements from human patients.
"""

# Adding a system automatically extends the whole-body summary, catalog, and atlas UI.
SYSTEMS=[
 ("neural","Autonomic and central nervous system","Baroreceptors · sympathetic/parasympathetic · nociception · arousal · emotion"),
 ("cardiovascular","Cardiovascular · circulation","Preload · contractility · endothelium · perfusion · edema"),
 ("respiratory","Respiration · gas exchange","Airway · ventilation · mucus · hypoxic drive"),
 ("endocrine","Endocrine axis","HPA · thyroid · growth · pancreatic accessory hormones"),
 ("renal","Kidney · water","Glomerular filtration · RAAS · ADH · excretion"),
 ("electrolyte","Electrolyte and acid-base","Na⁺/K⁺/Mg²⁺/phosphate · osmolality · bicarbonate"),
 ("hepatic","Hepatic metabolism","Glycogen, fat, protein synthesis · detoxification · bilirubin"),
 ("adipose","Fat and energy storage","Lipolysis · adipose-derived hormones · thermogenesis"),
 ("muscle","Muscle and cellular energy","ATP demand · AMPK/mTOR · lactate · recovery"),
 ("immune","Immune and inflammation","Innate/adaptive immunity · cytokines · resolution"),
 ("hematologic","Blood · bone marrow","Erythrocytes · iron · leukocytes · plasma volume"),
 ("hemostasis","Hemostasis · coagulation","Platelets · thrombin · fibrin · anticoagulation"),
 ("lymphatic","Lymph · spleen","Lymph return · splenic filtration · lymphocyte mobilization"),
 ("digestive","Digestion and intestinal environment","Secretion · motility · appetite · barrier · fermentation"),
 ("circadian","Circadian rhythm and sleep","Light input · SCN · melatonin · sleep"),
 ("skin","Skin, body temperature, and recovery","Temperature sensing · sweating · barrier · wound healing"),
 ("urinary","Bladder · urination","Bladder filling · micturition reflex · urinary retention"),
 ("bone","Bone and calcium","PTH/calcitriol · resorption/formation · load"),
 ("reproductive","Reproductive endocrinology","GnRH-gonadal · menstruation · lactation"),
]

# Public whole-body model documents whose system shape was borrowed (not LLM-generated).
SOURCES={
 "neural":{"title":"BioGears Nervous System Methodology","url":"https://biogearsengine.com/documentation/_nervous_methodology.html"},
 "cardiovascular":{"title":"BioGears Cardiovascular Methodology","url":"https://biogearsengine.com/documentation/_cardiovascular_methodology.html"},
 "respiratory":{"title":"Pulse Respiratory Methodology","url":"https://pulse.kitware.com/_respiratory_methodology.html"},
 "endocrine":{"title":"BioGears Endocrine Methodology","url":"https://biogearsengine.com/documentation/_endocrine_methodology.html"},
 "renal":{"title":"BioGears Renal Methodology","url":"https://biogearsengine.com/documentation/_renal_methodology.html"},
 "hepatic":{"title":"BioGears Hepatic Methodology","url":"https://biogearsengine.com/documentation/_hepatic_methodology.html"},
 "digestive":{"title":"BioGears Gastrointestinal Methodology","url":"https://biogearsengine.com/documentation/_gastrointestinal_methodology.html"},
 "hematologic":{"title":"HumMod (Hester et al., Frontiers in Physiology 2011)","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC3082131/"},
 "lymphatic":{"title":"BioGears Cardiovascular Methodology (lymph return)","url":"https://biogearsengine.com/documentation/_cardiovascular_methodology.html"},
 "urinary":{"title":"BioGears Renal Methodology","url":"https://biogearsengine.com/documentation/_renal_methodology.html"},
 "circadian":{"title":"Jewett-Forger-Kronauer circadian pacemaker (1999)","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC1690085/"},
 "muscle":{"title":"Physiome muscle energetics models","url":"https://physiomeproject.org/"},
 "immune":{"title":"BioGears Sepsis / inflammation scenario family","url":"https://biogearsengine.com/documentation/index.html"},
 "bone":{"title":"Lemaire 2004 cellular control of bone remodelling","url":"https://pubmed.ncbi.nlm.nih.gov/15013120/"},
 "hemostasis":{"title":"Chattaraj 2021 multiscale haemostasis model","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC8037693/"},
 "reproductive":{"title":"Harris-Clark 2003 GnRH pulse network","url":"https://pubmed.ncbi.nlm.nih.gov/12851311/"},
 "skin":{"title":"Pulse Environment / thermal response methodology","url":"https://pulse.kitware.com/_environment_methodology.html"},
}

# (id, system, label, cell_type, target_signal_expression, tau_min).
# Free variables of target are other module ids, input channels, and whole-body primitives.
ROWS=[
 # ── Autonomic and central nervous system ─────────────────────────────────────────────
 ("baroreceptor","neural","Carotid sinus baroreceptor input","Carotid sinus nerve endings","pressure-0.3*orthostatic_stress-0.5*hemorrhage",2),
 ("sympathetic","neural","Sympathetic drive","Sympathetic postganglionic neurons","stress+0.4*pain-0.7*baroreceptor+0.4*oxygen_deficit+0.15*thermosensor+0.3*cold_exposure+0.3*hypoglycemia-0.3*relaxation+0.2*smoking+0.2*orthostatic_stress",4),
 ("parasympathetic","neural","Parasympathetic and vagal drive","Vagus nerve · Postganglionic neurons","0.25*baroreceptor-0.4*sympathetic+0.3*sleep+0.4*relaxation",4),
 ("pain","neural","Nociceptive and pain signals","Nociceptor peripheral sensory nerves","injury+0.2*tnf+0.3*ache+0.4*menstrual_flow+0.3*prostaglandin+0.2*doms",10),
 ("cerebral_flow","neural","Cerebral blood flow regulation","Cerebral artery endothelium · Astrocytes","0.4*co2_load-0.2*sympathetic+0.2*pressure+0.2*cognitive_load-0.2*orthostatic_stress",8),
 ("orexin","neural","Orexin arousal maintenance signal","Lateral hypothalamic orexin neurons","0.4*sympathetic+0.3*histamine-0.6*sleep_pressure-0.3*melatonin+0.2*stress",10),
 ("arousal","neural","Arousal and attention level","Reticular activating system · Locus coeruleus neurons","0.5*orexin+0.4*sympathetic+0.3*cognitive+0.3*stress-0.6*sleep_pressure-0.4*melatonin",5),
 ("adenosine","neural","Adenosine sleep load","Basal forebrain · Neurons","0.5*sleep_pressure+0.3*atp_demand-0.2*sleep",90),
 ("cognitive_load","neural","Cognitive workload","Prefrontal cortex neurons","0.7*cognitive+0.3*arousal-0.3*mental_fatigue",3),
 ("mental_fatigue","neural","Central fatigue and cognitive decline","Prefrontal cortex · Anterior cingulate cortex neurons","0.4*cognitive_load+0.4*sleep_deprivation+0.3*sleep_pressure+0.2*hypoglycemia-0.3*sleep",45),
 ("headache","neural","Headache and cerebrovascular nociceptive signals","Trigeminal perivascular fibers · Meningeal sensory nerves","0.5*ache+0.3*stress+0.3*pain+0.2*hypovolemia+0.2*sleep_deprivation+0.15*light",30),
 ("mood","neural","Mood and emotional state","Limbic system · Prefrontal neurons","-0.4*stress+0.3*sleep+0.2*relaxation+0.2*feeding-0.3*pain-0.2*il6",60),
 ("memory_consolidation","neural","Memory consolidation and hippocampal replay","Hippocampal CA1 · Neocortical neurons","0.5*sleep+0.3*melatonin-0.3*stress-0.2*cortisol",180),
 ("dopamine_reward","neural","Dopamine reward signal","Ventral tegmental area · Nucleus accumbens neurons","0.3*feeding+0.3*arousal+0.2*cognitive-0.2*stress+0.3*drug_exposure",15),
 ("serotonin_tone","neural","Serotonin mood regulation","Central raphe nucleus neurons","0.2*sleep+0.2*feeding+0.2*relaxation-0.4*stress-0.2*sleep_deprivation",120),
 ("neuroinflammation","neural","Microglial neuroinflammation","Microglia · Astrocytes","0.4*tnf+0.3*il6+0.3*pain+0.2*barrier_damage-0.2*resolution",90),
 ("preoptic","neural","Preoptic thermoregulatory signal","Hypothalamic preoptic neurons","0.5*pyrogen-0.4*heat_signal+0.3*cold-0.2*melatonin",8),
 ("carb_craving","neural","Carbohydrate craving signal","Hypothalamus · Prefrontal neurons","0.4*ghrelin+0.3*hypoglycemia+0.2*stress+0.2*dopamine_reward",60),
 ("cerebral_metabolism","neural","Brain metabolic demand","Neurons · Astrocytes","0.4*cognitive_load+0.3*arousal+0.2*work",10),
 # ── Cardiovascular · circulation ──────────────────────────────────────────────
 ("preload","cardiovascular","Venous return · cardiac preload","Vena cava · Atrium","0.5*hydration+0.4*venous_return+0.4*plasma_vol-0.3*hypovolemia-0.4*hemorrhage",10),
 ("venous_return","cardiovascular","Venous return drive","Skeletal muscle pump · Venous smooth muscle cells","0.4*work+0.3*sympathetic-0.5*immobility-0.4*orthostasis",10),
 ("contractility","cardiovascular","Myocardial contractility regulation","Cardiomyocytes","0.5*sympathetic+0.4*catecholamine+0.2*thyroid-0.3*acid_load-0.2*hypovolemia",8),
 ("coronary_flow","cardiovascular","Coronary blood flow","Coronary endothelium · Smooth muscle cells","0.4*cardiac_output+0.3*work-0.3*endothelial_activation-0.2*thrombin",12),
 ("endothelin","cardiovascular","Endothelin-1 vasoconstriction signal","Vascular endothelial cells","0.5*endothelial_activation+0.3*angiotensin+0.3*hypoxia",30),
 ("nitric_oxide","cardiovascular","Nitric oxide vasodilation signal","Vascular endothelial cells","0.4*work+0.3*feeding+0.2*insulin_signal-0.4*endothelial_activation-0.3*smoking",15),
 ("orthostatic_stress","cardiovascular","Orthostatic circulatory stress","Carotid sinus baroreceptor afferent nerves","0.5*orthostasis+0.3*heat_signal+0.4*skin_bloodflow+0.3*hypovolemia+0.3*hemorrhage-0.3*sympathetic+0.3*peripheral_perfusion",8),
 ("hrv","cardiovascular","Heart rate variability · vagal regulation","Sinoatrial node · Vagus nerve","0.5*parasympathetic-0.4*sympathetic-0.3*stress+0.2*sleep",10),
 ("peripheral_perfusion","cardiovascular","Peripheral tissue perfusion","Arteriolar smooth muscle cells","0.4*cardiac_output+0.3*work-0.3*sympathetic-0.3*cold+0.2*nitric_oxide",12),
 ("edema","cardiovascular","Interstitial edema formation","Capillary endothelium · Interstitial tissue","0.4*hypervolemia+0.3*endothelial_activation-0.4*albumin+0.3*orthostasis+0.2*immobility-0.3*lymph_drainage",60),
 ("arrhythmia_risk","cardiovascular","Arrhythmia risk signal","Cardiomyocytes · Purkinje fibers","0.3*acid_load+0.3*k_depletion+0.3*catecholamine+0.2*hypovolemia",15),
 ("syncope_risk","cardiovascular","Syncope risk signal","Cerebral vasculature · Baroreceptors","0.5*orthostatic_stress+0.3*hypovolemia+0.2*arrhythmia_risk-0.3*cerebral_flow",10),
 # ── Respiration · gas exchange ────────────────────────────────────────────
 ("airway","respiratory","Airway constriction · resistance","Airway smooth muscle cells","allergen+0.2*tnf-0.2*catecholamine+0.4*asthma+0.3*histamine+0.2*smoking+0.2*pollution+0.3*airway_inflammation",8),
 ("ventilation","respiratory","Ventilation drive","Brainstem respiratory center · Diaphragm","0.7*work+0.5*co2_load+0.2*sympathetic-0.4*airway+breathing+0.4*hypoxic_drive+0.3*asthma",3),
 ("co2_load","respiratory","CO₂ load","Tissue ventilation-perfusion mismatch","0.7*work+0.15*thyroid-ventilation+0.4*apnea",6),
 ("oxygen_deficit","respiratory","Tissue oxygen deficit","Capillaries · Tissue","max(0,hypoxia+0.3*work+0.3*airway-0.25*ventilation-0.2*cardiac_output+0.4*apnea-0.5*oxygen_therapy)",5),
 ("hypoxic_drive","respiratory","Carotid body hypoxic ventilatory drive","Carotid body type I glomus cells","0.6*oxygen_deficit+0.4*hypoxia+0.3*co2_load+0.2*apnea",4),
 ("mucus","respiratory","Airway mucus secretion","Airway goblet cells · Mucous gland cells","0.5*allergen+0.4*infection+0.4*smoking+0.4*pollution+0.2*airway+0.4*parasympathetic",40),
 ("cough","respiratory","Cough reflex drive","Airway mechanosensory afferent nerves · Medullary cough center","0.5*mucus+0.4*airway+0.3*infection+0.2*asthma",10),
 ("pulmonary_vr","respiratory","Hypoxic pulmonary vasoconstriction","Pulmonary artery smooth muscle cells","0.5*oxygen_deficit+0.4*hypoxia+0.2*acid_load",15),
 ("resp_muscle_fatigue","respiratory","Respiratory muscle fatigue","Diaphragm · Intercostal muscle fibers","0.4*ventilation+0.3*hypoxic_drive+0.2*apnea-0.2*oxidation",30),
 ("airway_inflammation","respiratory","Airway inflammation","Airway epithelial cells · Macrophage","0.4*allergen+0.4*asthma+0.3*infection+0.3*pollution+0.2*ige",45),
 ("oxygen_delivery","respiratory","Tissue oxygen delivery","Red blood cells · Capillaries","0.4*cardiac_output+0.3*oxygen_carrying-0.4*oxygen_deficit+0.2*ventilation",20),
 # ── Endocrine axis ──────────────────────────────────────────────────
 ("crh","endocrine","CRH release","Hypothalamic ventromedial nucleus neurons","stress+0.3*pain+0.2*il6-0.4*cortisol+0.2*cognitive-0.2*relaxation",20),
 ("acth","endocrine","ACTH release","Anterior pituitary corticotroph cells","0.7*crh",15),
 ("cortisol","endocrine","Cortisol waveform","Adrenal cortex zona fasciculata cells","0.8*acth",40),
 ("catecholamine","endocrine","Catecholamine surge","Adrenal medulla chromaffin cells","0.6*sympathetic+0.2*work+0.2*hypoglycemia+0.2*apnea",6),
 ("tsh","endocrine","TSH release","Pituitary thyrotroph cells","-0.4*thyroid",90),
 ("thyroid","endocrine","Thyroid hormone axis","Thyroid follicular cells","0.6*tsh-0.3*cortisol",180),
 ("gh","endocrine","Growth hormone pulsatility","Pituitary somatotroph cells","0.4*deep_sleep+0.2*sleep+0.2*work+0.2*fasting-0.3*igf1",60),
 ("igf1","endocrine","IGF-1 axis","Hepatocytes · Chondrocytes","0.5*gh+0.2*protein",240),
 ("prolactin","endocrine","Prolactin secretion","Anterior pituitary lactotroph cells","0.4*stress+0.3*sleep+0.6*lactation+0.4*pregnancy",45),
 ("oxytocin","endocrine","Oxytocin secretion","Hypothalamic paraventricular nucleus · Posterior pituitary","0.4*relaxation+0.3*sleep+0.3*lactation+0.2*mood",30),
 ("dhea","endocrine","DHEA adrenal androgen","Adrenal cortex zona reticularis cells","0.3*acth+0.2*stress",120),
 ("insulin_resistance","endocrine","Insulin resistance signaling","Skeletal muscle cells · Adipocyte receptor downstream","0.4*tnf+0.3*cortisol+0.3*lipolysis+0.2*pregnancy+0.2*fasting-0.3*adiponectin-0.2*work",240),
 ("calcitonin","endocrine","Calcitonin secretion","Thyroid parafollicular cells","0.4*calcium_signal-0.3*pth",60),
 ("amylin","endocrine","Amylin secretion","Pancreatic beta cells","0.4*insulin_signal+0.3*feeding+0.2*meal",30),
 ("beta_cell_stress","endocrine","Beta-cell mitochondrial load","Pancreatic beta cells","0.4*hyperglycemia+0.3*insulin_signal+0.2*systemic_inflammation-0.2*oxidation",240),
 # ── Kidney · water ────────────────────────────────────────────────
 ("gfr","renal","Glomerular filtration rate","Glomerular endothelium · Podocytes","0.3*pressure+0.2*hydration+0.25*angiotensin+0.2*renal_bloodflow+0.1*anp",15),
 ("renin","renal","Renin release","Juxtaglomerular cells · Afferent arteriole","-0.5*pressure-0.6*hydration-0.4*plasma_vol+0.2*sympathetic-0.2*sodium_load+0.3*gi_fluid_loss+0.3*dehydration-0.3*anp+0.2*pregnancy+0.5*hemorrhage",30),
 ("angiotensin","renal","Angiotensin II formation","Plasma · Renal interstitium","0.6*renin",20),
 ("aldosterone","renal","Aldosterone release","Adrenal cortex zona glomerulosa cells","0.7*angiotensin+0.3*potassium_shift+0.3*gi_fluid_loss-0.3*anp+0.2*pregnancy",60),
 ("adh","renal","ADH/vasopressin release","Hypothalamic supraoptic nucleus neurons","0.6*osmotic_drive-0.4*hydration+0.4*gi_fluid_loss+0.2*nausea+0.3*hypovolemia+0.3*hemorrhage-0.3*plasma_vol+0.3*serotonin_tone",15),
 ("aqp2","renal","Collecting duct aquaporin-2 translocation","Collecting duct principal cells","0.7*adh+0.1*nausea",25),
 ("enac","renal","ENaC sodium reabsorption","Collecting duct principal cells","0.7*aldosterone+0.2*acid_load",45),
 ("renal_acid","renal","Renal acid excretion","Proximal tubule epithelial cells","0.5*acid_load+0.2*k_depletion",90),
 ("epo","renal","Erythropoietin EPO secretion","Renal interstitial fibroblasts","0.5*oxygen_deficit+0.4*hypoxia+0.4*hemorrhage-0.2*hydration",60),
 ("anp","renal","Atrial natriuretic peptide ANP","Atrial cardiomyocytes","0.6*hydration+0.4*pressure",10),
 ("urine_output","renal","Urine production volume","Glomerulus · Tubular epithelial cells","0.5*gfr-0.5*aqp2+0.4*anp-0.2*aldosterone+0.3*prostaglandin",15),
 ("renal_bloodflow","renal","Renal blood flow regulation","Afferent arteriole smooth muscle cells","0.4*cardiac_output-0.4*sympathetic-0.4*angiotensin+0.2*anp",10),
 ("kaliuresis","renal","Potassium excretion drive","Collecting duct principal cells","0.5*aldosterone+0.3*potassium_shift+0.2*acid_load+0.4*enac",20),
 # ── Electrolyte and acid-base ────────────────────────────────────────────
 ("sodium_load","electrolyte","Extracellular sodium load","Extracellular fluid","salt-0.2*gfr+0.2*enac-0.3*anp-0.3*vomiting-0.3*sweat-0.3*diarrhea-0.3*prostaglandin",30),
 ("potassium_shift","electrolyte","K⁺ intracellular movement","Muscle cell · Hepatocytes","potassium+0.2*work-0.2*insulin_signal-0.3*aldosterone-0.3*k_depletion",15),
 ("osmotic_drive","electrolyte","Plasma osmotic pressure drive","Hypothalamic osmoreceptors","0.5*sodium_load-0.5*hydration+0.4*gi_fluid_loss+0.3*dehydration+0.2*hemorrhage-0.4*plasma_vol+0.5*urine_output",20),
 ("acid_load","electrolyte","Acid load","Tissue · Plasma","0.4*lactate+0.5*co2_load-0.4*renal_acid-0.3*bicarbonate+0.3*diarrhea+0.2*ketogenesis-0.3*vomiting",20),
 ("bicarbonate","electrolyte","Bicarbonate buffering","Plasma · Tubules","-0.4*acid_load+0.3*renal_acid+0.3*vomiting-0.2*diarrhea",45),
 ("magnesium","electrolyte","Mg²⁺ balance response","Renal tubular epithelial cells · Osteocytes","0.2*protein-0.4*diarrhea-0.3*vomiting-0.2*stress",120),
 ("phosphate","electrolyte","Phosphate balance response","Renal proximal tubule epithelial cells","0.3*protein+0.3*resorption-0.4*pth-0.2*insulin_signal",90),
 ("k_depletion","electrolyte","Potassium depletion signal","Gastrointestinal tract · Renal tubules","0.4*diarrhea+0.4*vomiting+0.3*kaliuresis-0.3*potassium-0.2*magnesium",60),
 # ── Hepatic metabolism ────────────────────────────────────────────────────
 ("glucagon","hepatic","Glucagon release","Pancreatic alpha cell","0.3*fasting+0.25*catecholamine-0.4*glucose_signal+0.5*hypoglycemia",25),
 ("glycogenolysis","hepatic","Glycogenolysis","Hepatocytes","0.5*glucagon+0.3*catecholamine-0.2*insulin_signal+0.4*hypoglycemia",15),
 ("gluconeogenesis","hepatic","Gluconeogenesis","Hepatocytes","0.3*glucagon+0.2*cortisol+0.15*lactate-0.2*insulin_signal+0.4*hypoglycemia",45),
 ("ketogenesis","hepatic","Ketogenesis","Hepatocyte mitochondria","0.4*fasting+0.3*lipolysis+0.2*glucagon",120),
 ("urea_cycle","hepatic","Urea cycle load","Hepatocyte mitochondria","0.5*protein+0.2*cortisol+0.3*bigmeal",90),
 ("albumin","hepatic","Albumin synthesis","Hepatocytes","0.2*protein-0.4*il6-0.2*hepatic_load",240),
 ("lipogenesis","hepatic","Hepatic de novo lipogenesis","Hepatocytes","0.5*insulin_signal+0.3*hyperglycemia+0.3*feeding-0.3*fasting-0.2*glucagon",60),
 ("vldl","hepatic","VLDL lipid secretion","Hepatocytes","0.5*lipogenesis+0.4*fatty_acid+0.2*feeding",90),
 ("cholesterol_syn","hepatic","Cholesterol synthesis","Hepatocytes","0.3*lipogenesis+0.3*insulin_signal+0.2*fat-0.2*scfa",240),
 ("cyp450","hepatic","CYP450 drug metabolism activity","Hepatocyte endoplasmic reticulum","0.8*drug_exposure+0.1*smoking",30),
 ("glutathione","hepatic","Glutathione redox defense","Hepatocytes","0.3*protein-0.4*drug_exposure-0.3*systemic_inflammation+0.2*scfa",120),
 ("hepatic_load","hepatic","Hepatic metabolic load","Hepatocytes","0.5*drug_exposure+0.3*lipogenesis+0.2*urea_cycle+0.2*systemic_inflammation",60),
 ("bilirubin","hepatic","Bilirubin processing load","Hepatocytes · Reticuloendothelial system","0.4*hepatic_load+0.3*hemorrhage+0.2*erythropoiesis",120),
 # ── Fat and energy storage ─────────────────────────────────────────
 ("lipolysis","adipose","Lipolysis","Adipocytes","0.4*catecholamine+0.2*fasting-0.3*insulin_signal+0.2*ghrelin+0.2*cold_exposure",20),
 ("fatty_acid","adipose","Free fatty acid supply","Adipocytes · Plasma","0.6*lipolysis+0.3*fat-0.3*oxidation-0.2*lpl",25),
 ("leptin","adipose","Leptin signaling","Adipocytes","0.2*feeding+0.2*meal-0.4*fasting",120),
 ("adiponectin","adipose","Adiponectin signaling","Adipocytes","-0.3*tnf+0.1*fasting",180),
 ("lpl","adipose","Lipoprotein lipase fat storage","Adipocytes · Capillary endothelial cells","0.5*insulin_signal+0.3*feeding-0.4*fasting-0.2*work",30),
 ("browning","adipose","Beige fat thermogenic activity","Beige adipocytes","0.6*cold_exposure+0.4*cold+0.4*sympathetic+0.2*thyroid",90),
 ("leptin_resistance","adipose","Leptin resistance signaling","Hypothalamic POMC neurons","0.3*tnf+0.2*pregnancy+0.3*leptin-0.2*adiponectin",240),
 # ── Muscle and cellular energy ─────────────────────────────────────────
 ("atp_demand","muscle","Cellular ATP demand","Skeletal muscle · Cardiomyocytes","work+0.1*thyroid+0.2*shivering+0.1*cognitive_load",5),
 ("ampk","muscle","AMPK activation","Skeletal muscle cells","0.6*atp_demand+0.2*oxygen_deficit-0.2*oxidation+0.2*adiponectin",10),
 ("oxidation","muscle","Mitochondrial oxidation","Skeletal muscle mitochondria","0.6*ampk+0.2*fatty_acid-0.3*oxygen_deficit+0.2*thyroid",15),
 ("lactate","muscle","Lactate production","Skeletal muscle cells","0.6*atp_demand+0.3*oxygen_deficit-0.4*oxidation",8),
 ("mtor","muscle","mTOR protein synthesis signaling","Skeletal muscle cells","0.4*protein+0.3*igf1-0.3*ampk-0.2*cortisol-0.2*insulin_resistance",60),
 ("muscle_glucose_uptake","muscle","Muscle glucose uptake activity","Skeletal muscle cells","0.4*insulin_signal+0.5*work+0.4*ampk+0.2*hyperglycemia-0.3*insulin_resistance",10),
 ("muscle_glycogen","muscle","Muscle glycogen repletion response","Skeletal muscle cells","0.3*insulin_signal+0.3*feeding-0.5*work-0.3*catecholamine",60),
 ("fatigue","muscle","Peripheral muscle fatigue","Skeletal muscle cells","0.5*atp_demand+0.4*lactate-0.4*oxidation+0.2*sleep_deprivation+0.2*hypoglycemia",20),
 ("doms","muscle","Delayed-onset muscle soreness and muscle fiber recovery","Skeletal muscle satellite cells · Macrophage","0.6*max(0,work-0.4)+0.2*injury+0.2*tnf",480),
 ("protein_breakdown","muscle","Muscle protein breakdown catabolism","Skeletal muscle cells","0.4*cortisol+0.3*fasting+0.3*tnf-0.4*mtor-0.3*insulin_signal",120),
 ("shivering","muscle","Shivering thermogenesis","Skeletal muscle cells","0.7*cold+0.5*cold_exposure+0.3*pyrogen",3),
 # ── Immune and inflammation ────────────────────────────────────────────────
 ("nfkb","immune","NF-κB inflammatory transcription","Macrophage · Dendritic cell","infection+injury+0.25*barrier_damage-0.3*cortisol-0.2*resolution+0.3*smoking+0.2*pollution+0.3*vaccination",15),
 ("tnf","immune","TNF-α surge","Macrophage · T cell","0.6*nfkb+0.2*macrophage-0.25*resolution",20),
 ("il6","immune","IL-6 waveform","Macrophage · Fibroblasts · Myocyte","0.5*nfkb+0.15*work+0.2*macrophage+0.1*menstrual_flow-0.2*resolution",25),
 ("tcell","immune","T cell activation","T lymphocytes","0.4*infection+0.3*vaccination+0.2*nfkb+0.2*dendritic-0.2*cortisol",120),
 ("bcell","immune","B cells and antibody production","B lymphocyte · Plasma cell","0.4*tcell+0.2*infection+0.3*vaccination+0.2*dendritic",180),
 ("complement","immune","Complement cascade","Plasma complement proteins","0.5*infection+0.2*bcell+0.2*ige",45),
 ("crp","immune","CRP acute phase response","Hepatocytes","0.7*il6+0.2*il1",60),
 ("resolution","immune","Inflammation resolution and resolvin pathway","M2 macrophages · Regulatory T cells","0.3*nfkb+0.2*tcell+0.3*il10",90),
 ("neutrophil","immune","Neutrophil recruitment and phagocytic activity","Neutrophil","0.6*infection+0.4*injury+0.4*nfkb+0.2*cortisol",30),
 ("macrophage","immune","Macrophage M1 inflammatory activity","Macrophage","0.5*nfkb+0.3*infection+0.3*injury-0.3*resolution",40),
 ("nk_cell","immune","NK cell activity","NK lymphocytes","0.4*infection+0.3*il6+0.2*interferon-0.3*cortisol",60),
 ("dendritic","immune","Dendritic cell antigen presentation","Dendritic cell","0.5*infection+0.3*nfkb+0.4*vaccination+0.2*allergen",60),
 ("ige","immune","IgE-mediated allergic reaction","B lymphocyte · Mast cells","0.7*allergen+0.3*bcell",120),
 ("eosinophil","immune","Eosinophil activity","Eosinophils","0.5*allergen+0.4*ige",90),
 ("il1","immune","IL-1β inflammatory signaling","Macrophage · Monocyte","0.6*nfkb+0.3*injury+0.3*infection",25),
 ("il10","immune","IL-10 anti-inflammatory regulation","Regulatory T cells · M2 macrophages","0.5*resolution+0.3*cortisol+0.2*tcell",60),
 ("interferon","immune","Interferon antiviral signaling","Plasmacytoid dendritic cells · Infected cells","0.6*infection+0.2*nfkb",45),
 ("histamine","immune","Mast cell histamine secretion","Mast cells · Basophils","0.8*allergen+0.5*ige+0.2*complement+0.2*injury",15),
 ("prostaglandin","immune","Prostaglandin E2 production","Macrophage · Fibroblasts · Vascular endothelial cells","0.5*injury+0.4*nfkb+0.3*pain+0.3*menstrual_flow",20),
 ("pyrogen","immune","Fever original signal","Hypothalamic preoptic area · Macrophage","0.5*il6+0.5*il1+0.4*prostaglandin+0.3*tnf",25),
 ("systemic_inflammation","immune","Systemic inflammatory load","Macrophage · Vascular endothelial cells","0.4*tnf+0.4*il6+0.3*nfkb+0.2*complement-0.4*resolution",45),
 ("mucosal_iga","immune","Mucosal IgA immunity","Mucosal plasma cells","0.4*scfa+0.3*infection+0.2*barrier_damage-0.2*stress",180),
 ("anaphylaxis_risk","immune","Anaphylaxis risk signal","Mast cells · Basophils","0.5*ige+0.4*histamine+0.3*allergen",10),
 # ── Blood · bone marrow ────────────────────────────────────────────────
 ("erythropoiesis","hematologic","Bone marrow erythropoiesis","Bone marrow erythroid progenitor cells","0.7*epo+0.3*hypoxia+0.2*hemorrhage-0.3*systemic_inflammation-0.2*iron_deficiency",720),
 ("oxygen_carrying","hematologic","Oxygen carrying capacity","Red blood cells","0.5*erythropoiesis+0.3*iron_store-0.5*hemorrhage-0.2*systemic_inflammation",480),
 ("iron_store","hematologic","Iron storage · ferritin","Hepatocytes · Macrophage","0.3*protein-0.5*hemorrhage-0.4*menstrual_flow-0.2*il6",720),
 ("iron_deficiency","hematologic","Iron deficiency signal","Bone marrow erythroid progenitor cells","-0.6*iron_store",720),
 ("leukocyte","hematologic","Leukocyte mobilization","Bone marrow granulocytic lineage · Leukocytes","0.5*infection+0.3*nfkb+0.2*cortisol+0.2*stress",45),
 ("thrombopoiesis","hematologic","Platelet production","Bone marrow megakaryocytes","0.4*il6+0.3*hemorrhage+0.2*erythropoiesis",480),
 ("plasma_vol","hematologic","Relative change in plasma volume","Plasma","0.5*hydration+0.4*transfusion-0.5*hemorrhage+0.2*albumin-0.3*gi_fluid_loss-0.6*dehydration+0.4*pregnancy-0.5*urine_output",45),
 # ── Hemostasis · coagulation ────────────────────────────────────────────────
 ("platelet","hemostasis","Platelet activation","Platelets · Damaged endothelium","0.6*injury+0.2*tnf+0.3*vwf+0.2*thrombin",10),
 ("thrombin","hemostasis","Thrombin generation","Coagulation cascade proteases","0.5*platelet+0.3*injury-0.2*fibrinolysis+0.2*vwf-0.3*anticoagulant",15),
 ("fibrin","hemostasis","Fibrin formation","Fibrinogen → fibrin","0.6*thrombin-0.4*fibrinolysis",20),
 ("fibrinolysis","hemostasis","Fibrinolysis","Plasmin","0.4*fibrin+0.1*hemorrhage",45),
 ("endothelial_activation","hemostasis","Vascular endothelial activation","Vascular endothelial cells","0.4*tnf+0.3*il6+0.3*injury+0.3*histamine+0.2*smoking",20),
 ("vwf","hemostasis","vWF-mediated platelet adhesion","Vascular endothelial cells · Megakaryocytes","0.6*endothelial_activation+0.4*injury+0.2*hemorrhage",25),
 ("anticoagulant","hemostasis","Antithrombin · protein C anticoagulation","Hepatocytes · Vascular endothelial cells","0.3*protein-0.4*systemic_inflammation-0.3*thrombin",60),
 ("consumption_coag","hemostasis","Consumptive coagulation load","Platelets · Vascular endothelial cells","0.4*systemic_inflammation+0.4*thrombin-0.4*fibrinolysis-0.3*anticoagulant",40),
 # ── Lymph · spleen ────────────────────────────────────────────────
 ("lymph_drainage","lymphatic","Lymph return","Lymphatic endothelial cells · Skeletal muscle pump","0.4*work+0.3*hydration+0.3*il6-0.3*immobility",30),
 ("spleen","lymphatic","Splenic blood filtration · immune surveillance","Splenic macrophages · Lymphocytes","0.4*infection+0.3*hemorrhage+0.2*erythropoiesis",120),
 ("lymphocyte_traffic","lymphatic","Lymphocyte circulation · lymph node mobilization","Lymph node · Lymphocytes","0.4*infection+0.4*dendritic-0.3*cortisol+0.2*vaccination",90),
 # ── Digestion and intestinal environment ───────────────────────────────────────────
 ("gastric_acid","digestive","Gastric acid secretion","Parietal cell","0.4*feeding+0.3*meal+0.2*bigmeal+0.2*parasympathetic-0.2*sympathetic+0.2*smoking",15),
 ("amylase","digestive","Pancreatic amylase","Pancreatic acinar cell","0.5*feeding+0.3*meal",20),
 ("lipase","digestive","Pancreatic lipase","Pancreatic acinar cell","0.5*fat+0.2*feeding+0.3*meal",20),
 ("protease","digestive","Protease","Pancreatic acinar cell","0.5*protein+0.3*meal",20),
 ("bile","digestive","Bile secretion","Hepatocytes · Cholangiocytes","0.5*fat+0.2*feeding+0.3*meal",30),
 ("barrier_damage","digestive","Intestinal wall damage","Intestinal epithelial cells · Intercellular junction","0.3*tnf+0.2*stress-0.3*scfa-0.3*probiotic_drive+0.2*smoking",60),
 ("scfa","digestive","Short-chain fatty acid fermentation","Intestinal anaerobic microorganisms","0.5*fiber+0.4*probiotic_drive-0.2*smoking",180),
 ("ghrelin","digestive","Ghrelin hunger signal","Gastric X/A-like endocrine cell","0.6*fasting-0.4*feeding-0.3*meal-0.2*satiety+0.2*stress",45),
 ("satiety","digestive","Satiety and meal termination signal","Intestinal L cells · Vagal afferent fibers","0.5*feeding+0.5*meal+0.3*protein+0.3*fiber+0.3*leptin-0.4*ghrelin+0.2*bigmeal",20),
 ("motility","digestive","Gastrointestinal peristalsis","Intestinal smooth muscle cell · Interstitial cell of Cajal","0.3*feeding+0.3*meal+0.2*parasympathetic-0.3*stress-0.2*sympathetic+0.2*fiber-0.4*constipation",15),
 ("splanchnic_flow","digestive","Splanchnic blood flow redistribution","Splanchnic arterial smooth muscle cell","0.5*feeding+0.5*meal+0.2*cardiac_output-0.3*sympathetic-0.2*hypovolemia",10),
 ("emetic_drive","digestive","Vomiting center drive","Medullary area postrema · Chemoreceptor trigger zone","0.7*nausea+0.5*vomiting+0.3*infection+0.2*pain+0.2*pregnancy+0.2*drug_exposure",10),
 ("gi_fluid_loss","digestive","Gastrointestinal fluid loss","Intestinal epithelial cells","0.7*vomiting+0.8*diarrhea+0.2*infection",30),
 ("intestinal_gas","digestive","Intestinal gas production","Gut microbial community","0.4*scfa+0.3*fiber+0.3*bigmeal+0.2*constipation",60),
 ("probiotic_drive","digestive","Probiotic microbiota support","Gut microbial community","0.8*probiotic",240),
 ("appetite","digestive","Appetite drive","Hypothalamic arcuate nucleus NPY/AgRP · POMC neurons","0.6*ghrelin-0.4*satiety-0.3*leptin+0.2*hypoglycemia",30),
 ("bile_stasis","digestive","Cholestasis","Gallbladder smooth muscle cells","0.4*cholesterol_syn-0.4*bile+0.3*fasting+0.2*pregnancy",240),
 # ── Circadian rhythm and sleep ──────────────────────────────────────────────
 ("retinal_light","circadian","Retinal light input","Retinal ganglion photoreceptor cell","light-0.8*darkness",5),
 ("scn","circadian","SCN clock drive","Hypothalamic suprachiasmatic nucleus neurons","0.6*retinal_light-0.2*phase_mismatch",30),
 ("melatonin","circadian","Melatonin window","Pinealocyte","-0.5*scn+0.4*darkness+0.3*sleep-0.2*phase_mismatch",45),
 ("sleep_pressure","circadian","Sleep pressure S process","Basal forebrain · Homeostatic sleep drive","sleep_deprivation-0.7*sleep-0.15*melatonin+0.2*stress+0.2*phase_mismatch+0.2*mental_fatigue",120),
 ("phase_mismatch","circadian","Circadian phase shift","SCN neurons","1.0*circadian_shift+0.3*sleep_deprivation-0.3*scn",240),
 ("sleep_quality","circadian","Sleep quality index","Hypothalamus · Brainstem sleep circuit","0.6*sleep-0.4*stress-0.3*light-0.3*apnea-0.35*phase_mismatch-0.2*pain",60),
 ("rem_sleep","circadian","REM sleep tendency","Pons · Brainstem REM circuit","0.4*sleep+0.3*sleep_deprivation-0.3*stress+0.2*melatonin",90),
 ("deep_sleep","circadian","Slow-wave sleep tendency","Basal forebrain · Thalamic neurons","0.5*sleep+0.4*sleep_pressure-0.3*arousal+0.2*adenosine",90),
 # ── Skin, body temperature, and recovery ─────────────────────────────────────────
 ("thermosensor","skin","Skin temperature sensing input","Skin thermoreceptors","temperature_signal+0.5*heat-0.4*cold_exposure",4),
 ("sweat","skin","Sweating","Eccrine sweat gland cells","max(0,0.6*thermosensor+0.3*work)+0.3*estrogen_decline",5),
 ("wound_repair","skin","Wound healing · fibrosis","Fibroblasts · Keratinocytes","0.3*injury+0.2*resolution-0.2*cortisol+0.2*angiogenesis",180),
 ("skin_bloodflow","skin","Skin blood flow · flushing","Skin capillary smooth muscle cells","0.5*heat_signal+0.4*thermosensor+0.3*histamine-0.4*cold",8),
 ("piloerection","skin","Goosebumps · piloerector contraction","Piloerector smooth muscle cells","0.6*cold+0.5*cold_exposure+0.3*stress",5),
 ("skin_damage","skin","Skin barrier damage","Keratinocytes","0.5*injury+0.4*uv+0.3*infection+0.2*pollution",90),
 ("pruritus","skin","Itch signal","Skin C-fiber sensory nerves","0.6*histamine+0.4*allergen+0.3*skin_damage",15),
 ("uv_damage","skin","UV epidermal damage","Keratinocytes · Melanocytes","0.7*uv",120),
 ("vitamin_d_synthesis","skin","Vitamin D precursor photosynthesis","Keratinocytes","0.7*uv-0.2*uv_damage",180),
 ("angiogenesis","skin","Wound angiogenesis","Vascular endothelial cells","0.4*injury+0.3*oxygen_deficit+0.2*igf1+0.2*il6",240),
 ("scar_formation","skin","Scar · matrix deposition","Fibroblasts","0.4*injury+0.3*wound_repair+0.2*tnf",480),
 # ── Bladder · urination ────────────────────────────────────────────────
 ("bladder_fill","urinary","Bladder fullness","Bladder transitional epithelium · Detrusor muscle","0.6*urine_output+0.4*hydration",60),
 ("micturition","urinary","Micturition reflex","Bladder detrusor muscle · Pudendal nerve","0.6*bladder_fill-0.3*stress",30),
 ("urinary_stasis","urinary","Urinary retention","Bladder · Urethral epithelial cells","0.4*bladder_fill-0.5*micturition+0.2*constipation+0.2*immobility",120),
 # ── Bone and calcium ──────────────────────────────────────────────────
 ("calcium_signal","bone","Blood calcium deviation","Plasma ionized calcium","0.3*calcitriol+0.2*resorption-0.2*formation-calcium_deficit-0.4*calcium",20),
 ("pth","bone","PTH release","Parathyroid chief cells","-0.6*calcium_signal+0.2*phosphate+0.2*estrogen_decline",30),
 ("calcitriol","bone","Calcitriol activity","Proximal tubule 1α-hydroxylase","0.5*pth+0.1*gfr+0.4*vitamin_d_synthesis",90),
 ("resorption","bone","Bone resorption","Osteoclast","0.4*pth+0.2*tnf+0.3*estrogen_decline-0.2*mechanical_load",240),
 ("formation","bone","Bone formation","Osteoblast","0.2*work+0.2*igf1-0.2*cortisol+0.3*mechanical_load",240),
 ("mechanical_load","bone","Mechanical load bone stimulation","Osteocyte · Bone marrow stromal cells","0.6*work+0.3*orthostasis-0.5*immobility",120),
 ("fracture_risk","bone","Fracture risk signal","Osteocyte · Osteoclast","0.4*resorption-0.4*formation+0.3*estrogen_decline-0.2*mechanical_load",720),
 ("cartilage","bone","Joint cartilage load","Chondrocytes","0.3*work+0.3*tnf+0.2*injury-0.2*igf1",240),
 # ── Reproductive endocrinology ────────────────────────────────────────────────
 ("gnrh","reproductive","GnRH pulsatility","Hypothalamic GnRH neurons","-0.3*cortisol-0.3*gonadal_steroid-0.2*fasting-0.4*prolactin+0.3*estrogen_decline",30),
 ("gonadotropin","reproductive","LH/FSH release","Pituitary gonadotroph cells","0.6*gnrh",60),
 ("gonadal_steroid","reproductive","Gonadal steroids","Ovary · Testicular cells","0.5*gonadotropin-0.5*estrogen_decline+0.4*pregnancy",120),
 ("menstrual_flow","reproductive","Menstrual bleeding · endometrial shedding","Endometrial spiral arteries","0.8*menstrual",720),
 ("estrogen_decline","reproductive","Estrogen withdrawal response","Ovarian follicular cells","0.7*menopause+0.3*menstrual_flow-0.2*gonadal_steroid",480),
 ("libido","reproductive","Libido · gonadotropic response","Hypothalamus · Limbic neurons","0.4*gonadal_steroid-0.4*stress-0.3*sleep_deprivation-0.3*prolactin",180),
 ("spermatogenesis","reproductive","Spermatogenesis activity","Testicular seminiferous tubule Sertoli cells","0.5*gonadotropin+0.4*gonadal_steroid-0.3*heat_signal-0.2*stress",720),
 ("milk_production","reproductive","Milk production","Mammary gland acinar cells","0.7*prolactin+0.6*lactation+0.3*oxytocin-0.2*stress",240),
]

def _source(system):
    s=SOURCES.get(system)
    return {"id":f"ref:{system}","title":s["title"]+" - system structure reference","url":s["url"]} if s else None

MODULES=[{"id":r[0],"system":r[1],"label":r[2],"cells":r[3].split(" · "),"target":r[4],"tau_min":r[5],"unit":"dimensionless","initial":0,
 "source":_source(r[1]),"status":"reduced_uncalibrated"} for r in ROWS]

# Each whole-body process exposes an additive port that additively adjusts its own target signal.
BODY_PORTS={f"body_{m['id']}":{"node":"body:"+m["id"],"label":m["label"],"mode":"additive_target"} for m in MODULES}


def body_catalog():
    systems={s[0]:{"id":s[0],"label":s[1],"scope":s[2],"modules":[]} for s in SYSTEMS}
    for m in MODULES:
        systems[m["system"]]["modules"].append({k:m[k] for k in ("id","system","label","cells","target","tau_min","unit","initial","source","status")})
    return {"version":"0.6","module_count":len(MODULES),"system_count":len(SYSTEMS),
      "scope":"Input connects only initial anchors; subsequent propagation is deterministically performed by the node-node connection graph declared here. Absolute concentrations or values are not calculated.",
      "gaps":["Does not represent cell-level factor-specific interactions or probabilistic state transitions.",
        "All coefficients are in the form from the source, but not calibrated to real human values.","No adjustment for other races, ages, or comorbidities.",
        "Without complete anatomical coordinates between organs, the graph shows only causal direction, not distance."],
      "systems":list(systems.values())}
