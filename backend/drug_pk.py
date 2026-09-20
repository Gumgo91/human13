"""Literature-based drug PK parameter table.

Each entry: f = oral bioavailability (0-1), vd_l_kg = apparent volume of
distribution (L/kg), t_half_h = elimination half-life (h), t_max_h = typical
time to peak (h). Unlisted drugs use assumptions (F=1, Vd=plasma volume 3 L,
default elimination half-life) and are labeled as such. Values are
textbook-level representatives that do not reflect inter-individual or
formulation differences — this is not a PBPK prediction.

Sources: Goodman & Gilman's Pharmacological Basis of Therapeutics 13th;
Rowland & Tozer Clinical Pharmacokinetics 5th; Davies 1998 Clin
Pharmacokinet; Fredholm 1999 Pharmacol Rev; Scheen 1996 Clin Pharmacokinet;
Spyker 1977 Antimicrob Agents Chemother.
"""

# entity (lowercase) → {"f", "vd_l_kg", "t_half_h", "t_max_h", "ref"}
PK_TABLE = {
    "ibuprofen":    { "t_max_h":1.5,"f": 0.80, "vd_l_kg": 0.14, "t_half_h": 2.0,  "ref": "Davies 1998 Clin Pharmacokinet 34:101"},
    "caffeine":     { "t_max_h":1.0,"f": 0.99, "vd_l_kg": 0.60, "t_half_h": 5.0,  "ref": "Fredholm 1999 Pharmacol Rev 51:83"},
    "acetaminophen":{ "t_max_h":0.75,"f": 0.80, "vd_l_kg": 0.90, "t_half_h": 2.5,  "ref": "Rowland & Tozer Clin Pharmacokinet"},
    "paracetamol":  { "t_max_h":0.75,"f": 0.80, "vd_l_kg": 0.90, "t_half_h": 2.5,  "ref": "Rowland & Tozer Clin Pharmacokinet"},
    "metformin":    { "t_max_h":2.5,"f": 0.55, "vd_l_kg": 3.00, "t_half_h": 5.0,  "ref": "Scheen 1996 Clin Pharmacokinet 31:359"},
    "amoxicillin":  { "t_max_h":1.5,"f": 0.80, "vd_l_kg": 0.30, "t_half_h": 1.2,  "ref": "Spyker 1977 AAC 11:132"},
    "propranolol":  { "t_max_h":2.0,"f": 0.25, "vd_l_kg": 3.90, "t_half_h": 3.9,  "ref": "G&G 13th; Kornhauser 1975"},
    "atorvastatin": { "t_max_h":1.5,"f": 0.14, "vd_l_kg": 4.00, "t_half_h": 14.0, "ref": "Lennernas 2003 Clin Pharmacokinet"},
    "ethanol":      { "t_max_h":0.75,"f": 1.00, "vd_l_kg": 0.55, "t_half_h": 4.0,  "ref": "Norberg 2003 Clin Pharmacokinet"},
    "alcohol":      { "t_max_h":0.75,"f": 1.00, "vd_l_kg": 0.55, "t_half_h": 4.0,  "ref": "Norberg 2003 Clin Pharmacokinet"},
    "nicotine":     { "t_max_h":0.15,"f": 1.00, "vd_l_kg": 2.60, "t_half_h": 2.0,  "ref": "Benowitz 2009 CPT 85:158"},
    "sertraline":   { "t_max_h":6.0,"f": 0.44, "vd_l_kg": 20.0, "t_half_h": 26.0, "ref": "DeVane 2002 Clin Pharmacokinet"},
    "morphine":     { "t_max_h":1.0,"f": 0.30, "vd_l_kg": 3.20, "t_half_h": 2.0,  "ref": "Lotsch 2002 Clin Pharmacokinet"},
    "omeprazole":   { "t_max_h":2.5,"f": 0.60, "vd_l_kg": 0.30, "t_half_h": 1.0,  "ref": "Andersson 1996"},
    "warfarin":     { "t_max_h":2.0,"f": 0.95, "vd_l_kg": 0.14, "t_half_h": 36.0, "ref": "Holford 1986 Clin Pharmacokinet"},
    "aspirin":      { "t_max_h":1.0,"f": 0.50, "vd_l_kg": 0.15, "t_half_h": 0.3,  "ref": "Rowland 1967 (salicylate)"},
    "diphenhydramine":{ "t_max_h":2.0,"f": 0.60, "vd_l_kg": 4.5, "t_half_h": 8.0, "ref": "Simons 1990 J Allergy Clin Immunol"},
    "semaglutide":  { "t_max_h":36.0,"f": 1.00, "vd_l_kg": 0.12, "t_half_h": 165.0,"ref": "Knudsen 2019 (SC, depot)"},
    "tirzepatide":  { "t_max_h":36.0,"f": 1.00, "vd_l_kg": 0.15, "t_half_h": 120.0,"ref": "Jastreboff 2022 SURMOUNT PK"},
    "insulin":      { "t_max_h":1.0,"f": 1.00, "vd_l_kg": 0.15, "t_half_h": 0.1,  "ref": "SC depot absorption model"},
    "epinephrine":  { "t_max_h":0.1,"f": 1.00, "vd_l_kg": 0.20, "t_half_h": 0.05, "ref": "G&G 13th catecholamine"},
    "albuterol":    { "t_max_h":0.2,"f": 1.00, "vd_l_kg": 2.20, "t_half_h": 4.0,  "ref": "inhaled pulmonary absorption"},
    "salbutamol":   { "t_max_h":0.2,"f": 1.00, "vd_l_kg": 2.20, "t_half_h": 4.0,  "ref": "inhaled pulmonary absorption"},
    "levothyroxine":{ "t_max_h":3.0,"f": 0.70, "vd_l_kg": 0.15, "t_half_h": 168.0,"ref": "Fish 1987"},
    "sildenafil":   { "t_max_h":1.0,"f": 0.40, "vd_l_kg": 1.50, "t_half_h": 4.0,  "ref": "Nichols 2002 BJCP 53:5S"},
    "naloxone":     { "t_max_h":0.1,"f": 1.00, "vd_l_kg": 2.00, "t_half_h": 1.1,  "ref": "IV/immediate; Fishman 1973"},
    "furosemide":   { "t_max_h":1.0,"f": 0.60, "vd_l_kg": 0.15, "t_half_h": 1.5,  "ref": "Ponto 1990"},
    "melatonin":    { "t_max_h":0.5,"f": 0.15, "vd_l_kg": 0.70, "t_half_h": 0.75, "ref": "DeMuro 2000"},
    "gabapentin":   { "t_max_h":3.0,"f": 0.60, "vd_l_kg": 0.80, "t_half_h": 6.0,  "ref": "Bockbrader 2010 Clin PK"},
    "tramadol":     { "t_max_h":2.0,"f": 0.70, "vd_l_kg": 2.60, "t_half_h": 6.0,  "ref": "Grond 2004 Clin PK"},
    "allopurinol":  { "t_max_h":1.0,"f": 0.80, "vd_l_kg": 0.60, "t_half_h": 1.5,  "ref": "Murrell 1986"},
    "ondansetron":  { "t_max_h":2.0,"f": 0.60, "vd_l_kg": 1.90, "t_half_h": 3.5,  "ref": "Roila 1998"},
    "clopidogrel":  { "t_max_h":1.0,"f": 0.50, "vd_l_kg": 0.40, "t_half_h": 6.0,  "ref": "prodrug; Caplain 1999"},
    "losartan":     { "t_max_h":1.0,"f": 0.33, "vd_l_kg": 0.50, "t_half_h": 2.0,  "ref": "Lo 1995"},
    "lisinopril":   { "t_max_h":6.0,"f": 0.25, "vd_l_kg": 1.20, "t_half_h": 12.0, "ref": "Beermann 1988"},
    "morphine sulfate": { "t_max_h":1.0,"f": 0.30, "vd_l_kg": 3.20, "t_half_h": 2.0, "ref": "Lotsch 2002"},
    "prednisone":   { "t_max_h":1.5,"f": 0.80, "vd_l_kg": 0.80, "t_half_h": 3.0,  "ref": "prodrug→prednisolone"},
    "donepezil":    { "t_max_h":4.0,"f": 1.00, "vd_l_kg": 12.0, "t_half_h": 70.0, "ref": "Rogers 1998"},
    "montelukast":  { "t_max_h":3.0,"f": 0.64, "vd_l_kg": 0.16, "t_half_h": 4.0,  "ref": "Knorr 2000"},
    "loperamide":   { "t_max_h":5.0,"f": 0.40, "vd_l_kg": 11.0, "t_half_h": 11.0, "ref": "Heykants 1974"},
    "amiodarone":   { "t_max_h":4.0,"f": 0.40, "vd_l_kg": 66.0, "t_half_h": 960.0,"ref": "extreme Vd; Holt 1983"},
    "digoxin":      { "t_max_h":1.5,"f": 0.70, "vd_l_kg": 7.00, "t_half_h": 38.0, "ref": "Iisalo 1977"},
}

DEFAULT_VD_L = 3.0  # assumed plasma volume — Vd proxy for unlisted drugs

# Brand names / common synonyms → generic key. Deterministic lookup only;
# no substring fuzzing (e.g. "alcohol" must not match "alcohol dehydrogenase").
ALIASES = {
    "mounjaro": "tirzepatide", "zepbound": "tirzepatide",
    "ozempic": "semaglutide", "wegovy": "semaglutide", "rybelsus": "semaglutide",
    "tylenol": "acetaminophen", "advil": "ibuprofen", "motrin": "ibuprofen",
    "glucophage": "metformin", "lipitor": "atorvastatin", "zoloft": "sertraline",
    "ventolin": "albuterol", "proventil": "albuterol", "viagra": "sildenafil",
    "imodium": "loperamide", "benadryl": "diphenhydramine", "synthroid": "levothyroxine",
    "prilosec": "omeprazole", "coumadin": "warfarin", "entresto": "sacubitril",
    "propecia": "finasteride", "amoxil": "amoxicillin", "narcan": "naloxone",
    "lasix": "furosemide", "zestril": "lisinopril", "cozaar": "losartan",
    "plavix": "clopidogrel", "aricept": "donepezil", "singulair": "montelukast",
    "zofran": "ondansetron", "neurontin": "gabapentin", "ultram": "tramadol",
    "zyloprim": "allopurinol", "cordarone": "amiodarone", "lanoxin": "digoxin",
    "deltasone": "prednisone", "ethanol": "ethanol", "ethyl alcohol": "ethanol",
    "booze": "alcohol", "coffee": "caffeine",
}


def lookup(entity):
    """Find PK parameters by exact name or declared alias. None if absent."""
    if not entity:
        return None
    e = entity.strip().casefold()
    e = ALIASES.get(e, e)
    return PK_TABLE.get(e)
