"""Independent external-evidence check of discovery_report.json.

Confronts each non-generic surfaced (entity, module, direction) with FDA
drug-label text fetched from the openFDA label API. This stage is separate
from the discovery harness on purpose: the harness only reports structural
outputs; this script is the outside-evidence confrontation.

Ground truth: openFDA label fields (adverse_reactions, warnings,
boxed_warning, indications, clinical_pharmacology, drug_interactions).

Matching: MODULE_TERMS maps module ids to clinical/MedDRA-style terms with
an expected sign. The map is written from module definitions (what each
state means physiologically), not from model output. A module with no
label-expressible terminology is 'unmappable' and is NOT counted against
the model.

Honesty caveats (also written into the report):
- Absence of a label term is not evidence the model is wrong: labels name
  clinical endpoints, not intermediate physiology (a label says
  'hyperglycemia', never 'lipolysis').
- Label text mixes indications, adverse effects, and monitoring language;
  a match is directional only where the term itself is directional
  ('hyperkalemia' vs 'hypokalemia').
- Result is a coverage estimate, not a correctness proof.
"""
import asyncio
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "benchmark" / "discovery_report.json"
CACHE = ROOT / "benchmark" / "label_cache.json"
OUT = ROOT / "benchmark" / "label_check_report.json"

# module -> [(clinical term, expected sign)]. sign None = direction-neutral
# presence match. Written a priori from module semantics.
MODULE_TERMS = {
    "acid_load": [("acidosis", "+")],
    "adh": [("syndrome of inappropriate antidiuretic", "+"), ("siadh", "+"),
            ("hyponatremia", "+"), ("diabetes insipidus", "-")],
    "airway": [("bronchospasm", "+"), ("bronchoconstriction", "+"),
               ("wheezing", "+"), ("bronchodilat", "-")],
    "anaphylaxis_risk": [("anaphylaxis", "+"), ("anaphylactic", "+")],
    "anticoagulant": [("bleeding", "+"), ("hemorrhage", "+"),
                      ("thrombosis", "-"), ("anticoagulant", None)],
    "arousal": [("insomnia", "+"), ("somnolence", "-"), ("sedation", "-"),
                ("drowsiness", "-")],
    "arrhythmia_risk": [("arrhythmia", "+"), ("torsade", "+"),
                        ("qt prolongation", "+"), ("ventricular tachycardia", "+"),
                        ("proarrhythmia", "+")],
    "bcell": [("lymphopenia", "-"), ("lymphocytopenia", "-"),
              ("lymphocytosis", "+")],
    "bicarbonate": [("alkalosis", "+"), ("acidosis", "-")],
    "bladder_fill": [("urinary retention", "+"), ("urine retention", "+")],
    "catecholamine": [("tachycardia", "+"), ("palpitation", "+")],
    "cholesterol_syn": [("hypercholesterolemia", "+"), ("hyperlipidemia", "+"),
                        ("cholesterol increased", "+"), ("dyslipidemia", "+"),
                        ("ldl", None), ("cholesterol", None)],
    "co2_load": [("hypercapnia", "+"), ("respiratory depression", "+"),
                 ("hypoventilation", "+"), ("co2 retention", "+"),
                 ("carbon dioxide retention", "+")],
    "cognitive_load": [("cognitive impairment", "+"), ("confusion", "+"),
                       ("memory impairment", "+"), ("delirium", "+")],
    "consumption_coag": [("disseminated intravascular coagulation", "+"),
                         ("dic", "+"), ("thrombosis", "+")],
    "contractility": [("heart failure", "-"), ("cardiac failure", "-"),
                      ("cardiomyopathy", "-"), ("inotropic", None)],
    "coronary_flow": [("myocardial ischemia", "-"), ("angina", "-"),
                      ("coronary vasospasm", "-")],
    "cortisol": [("cushing", "+"), ("hypercortisolism", "+"),
                 ("adrenal insufficiency", "-")],
    "cough": [("cough", "+")],
    "crp": [("c-reactive", "+")],
    "deep_sleep": [("somnolence", "+"), ("sedation", "+"),
                   ("drowsiness", "+"), ("insomnia", "-")],
    "endothelial_activation": [("vasculitis", "+"), ("endothel", "+")],
    "epo": [("polycythemia", "+"), ("erythrocytosis", "+"),
            ("hemoglobin increased", "+")],
    "fatigue": [("fatigue", "+"), ("asthenia", "+")],
    "fibrin": [("thrombosis", "+"), ("thromboembol", "+")],
    "gastric_acid": [("dyspepsia", "+"), ("gastritis", "+"),
                     ("gastroesophageal reflux", "+"), ("hyperacidity", "+")],
    "gfr": [("renal impairment", "-"), ("renal failure", "-"),
            ("nephrotoxicity", "-"), ("decreased glomerular", "-"),
            ("acute kidney injury", "-")],
    "headache": [("headache", "+"), ("migraine", "+")],
    "hrv": [("heart rate variability", None)],
    "hypoxic_drive": [("hypoxia", "+"), ("hypoxemia", "+"), ("dyspnea", "+")],
    "il1": [("interleukin-1", "+"), ("pyrexia", "+")],
    "il6": [("interleukin-6", "+")],
    "insulin_resistance": [("hyperglycemia", "+"), ("diabetes mellitus", "+"),
                           ("glucose intolerance", "+"), ("insulin resistance", "+"),
                           ("hypoglycemia", "-")],
    "interferon": [("interferon", "+"), ("influenza-like", "+")],
    "k_depletion": [("hypokalemia", "+"), ("hypokalaemia", "+"),
                    ("potassium depletion", "+"), ("hyperkalemia", "-"),
                    ("hyperkalaemia", "-")],
    "kaliuresis": [("hypokalemia", "+"), ("kaliuresis", "+"),
                   ("urinary potassium", "+")],
    "ketogenesis": [("ketoacidosis", "+"), ("ketosis", "+"),
                    ("ketone", "+")],
    "lactate": [("lactic acidosis", "+"), ("lactate", "+")],
    "leukocyte": [("leukopenia", "-"), ("leukocytosis", "+"),
                  ("neutropenia", "-"), ("agranulocytosis", "-")],
    "lipolysis": [("lipolysis", "+"), ("weight loss", "+"),
                  ("free fatty acid", "+")],
    "mental_fatigue": [("fatigue", "+"), ("somnolence", "+")],
    "micturition": [("urinary frequency", "+"), ("pollakiuria", "+"),
                    ("urinary retention", "-")],
    "mood": [("depression", "-"), ("depressed mood", "-"), ("euphoria", "+"),
             ("suicidal", "-"), ("anxiety", "-")],
    "motility": [("diarrhea", "+"), ("diarrhoea", "+"), ("constipation", "-"),
                 ("delayed gastric emptying", "-"), ("gastroparesis", "-")],
    "mucus": [("rhinorrhea", "+"), ("mucus", "+"), ("sputum", "+"),
              ("expectoration", "+")],
    "muscle_glucose_uptake": [("hypoglycemia", "+"), ("hyperglycemia", "-")],
    "neuroinflammation": [("neuroinflammation", "+"), ("aseptic meningitis", "+")],
    "neutrophil": [("neutropenia", "-"), ("agranulocytosis", "-"),
                   ("neutrophilia", "+"), ("leukocytosis", "+")],
    "nk_cell": [("immunosuppression", "-"), ("infection", "-")],
    "orexin": [("insomnia", "+"), ("somnolence", "-"), ("narcolepsy", "-")],
    "orthostatic_stress": [("orthostatic hypotension", "+"),
                           ("postural hypotension", "+"), ("dizziness", "+"),
                           ("syncope", "+"), ("lightheadedness", "+")],
    "osmotic_drive": [("thirst", "+"), ("polydipsia", "+")],
    "oxygen_deficit": [("hypoxia", "+"), ("hypoxemia", "+"),
                       ("oxygen desaturation", "+")],
    "pain": [("hyperalgesia", "+"), ("neuropathic pain", "+"), ("pain", None)],
    "peripheral_perfusion": [("flushing", "+"), ("peripheral edema", "+"),
                             ("cold extremities", "-"), ("raynaud", "-"),
                             ("peripheral ischemia", "-"), ("vasodilatation", "+")],
    "piloerection": [("piloerection", "+"), ("gooseflesh", "+")],
    "plasma_vol": [("edema", "+"), ("fluid retention", "+"),
                   ("dehydration", "-"), ("volume depletion", "-"),
                   ("hypovolemia", "-")],
    "platelet": [("thrombocytosis", "+"), ("thrombocytopenia", "-")],
    "potassium_shift": [("hyperkalemia", "+"), ("hyperkalaemia", "+"),
                        ("hypokalemia", "-"), ("hypokalaemia", "-")],
    "preoptic": [("pyrexia", "+"), ("fever", "+"), ("hyperthermia", "+")],
    "protein_breakdown": [("muscle wasting", "+"), ("myopathy", "+"),
                          ("rhabdomyolysis", "+"), ("muscle weakness", "+")],
    "pruritus": [("pruritus", "+"), ("itching", "+")],
    "pulmonary_vr": [("pulmonary hypertension", "+"),
                     ("pulmonary arterial hypertension", "+")],
    "pyrogen": [("pyrexia", "+"), ("fever", "+"), ("antipyretic", "-")],
    "renal_bloodflow": [("renal ischemia", "-"), ("renal perfusion", None)],
    "renin": [("renin", "+"), ("hyperreninemia", "+")],
    "resp_muscle_fatigue": [("respiratory failure", "+"),
                            ("respiratory muscle", None)],
    "shivering": [("shivering", "+"), ("chills", "+"), ("rigors", "+")],
    "skin_bloodflow": [("flushing", "+"), ("pallor", "-"),
                       ("skin blanching", "-"), ("erythema", "+")],
    "sleep_quality": [("insomnia", "-"), ("somnolence", "+"),
                      ("sleep disorder", "-")],
    "sodium_load": [("hypernatremia", "+"), ("hyponatremia", "-"),
                    ("sodium retention", "+")],
    "sweat": [("sweating", "+"), ("hyperhidrosis", "+"), ("anhidrosis", "-"),
              ("perspiration", "+")],
    "sympathetic": [("tachycardia", "+"), ("palpitation", "+"),
                    ("hypertension", "+"), ("bradycardia", "-")],
    "syncope_risk": [("syncope", "+"), ("fainting", "+")],
    "systemic_inflammation": [("sepsis", "+"), ("inflammation", None)],
    "tcell": [("lymphopenia", "-"), ("immunosuppression", "-"),
              ("lymphocytosis", "+")],
    "thermosensor": [("pyrexia", "+"), ("fever", "+"), ("hypothermia", "-"),
                     ("antipyretic", "-")],
    "thrombin": [("thrombosis", "+"), ("thromboembol", "+")],
    "thyroid": [("hyperthyroidism", "+"), ("thyrotoxicosis", "+"),
                ("hypothyroidism", "-")],
    "tnf": [("tumor necrosis factor", "+"), ("tnf-alpha", "+")],
    "urinary_stasis": [("urinary retention", "+"), ("urinary stasis", "+")],
    "urine_output": [("polyuria", "+"), ("diuresis", "+"),
                     ("urinary frequency", "+"), ("oliguria", "-"),
                     ("anuria", "-"), ("urinary retention", "-")],
    "ventilation": [("respiratory depression", "-"), ("hypoventilation", "-"),
                    ("hyperventilation", "+"), ("dyspnea", "-")],
    "vldl": [("hypertriglyceridemia", "+"), ("triglyceride", "+"),
             ("hyperlipidemia", "+")],
    "vwf": [("von willebrand", "+")],
    "wound_repair": [("wound healing", "+"), ("impaired wound healing", "-")],
}

LABEL_FIELDS = ["adverse_reactions", "warnings", "warnings_and_cautions",
                "boxed_warning", "indications_and_usage",
                "clinical_pharmacology", "drug_interactions", "precautions"]

ALIAS = {"heparin": "heparin sodium", "nicotine": "nicotine"}


def fetch_label_text(entity):
    """Return concatenated openFDA label text, or None if not found."""
    names = [entity, ALIAS.get(entity, entity)]
    for name in dict.fromkeys(names):
        for field in ("openfda.generic_name", "openfda.substance_name"):
            q = urllib.parse.quote(f'{field}:"{name}"')
            url = f"https://api.fda.gov/drug/label.json?search={q}&limit=1"
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    data = json.loads(r.read().decode())
                rec = data["results"][0]
                parts = []
                for f in LABEL_FIELDS:
                    v = rec.get(f)
                    if isinstance(v, list):
                        parts.append(" ".join(str(x) for x in v))
                return " ".join(parts).lower() or None
            except Exception:
                continue
    return None


def match(text, mid, sign):
    """Score one (entity,module) against label text."""
    terms = MODULE_TERMS.get(mid)
    if not terms:
        return "unmappable", ""
    hits, contra, present = [], [], []
    for term, tsign in terms:
        if term in text:
            if tsign is None:
                present.append(term)
            elif tsign == sign:
                hits.append(term)
            else:
                contra.append(term)
    if hits and not contra:
        return "supported", ";".join(hits + present)
    if contra and not hits:
        return "contradicted", ";".join(contra + present)
    if hits and contra:
        return "mixed", ";".join(hits) + "||" + ";".join(contra)
    if present:
        return "term_present", ";".join(present)
    return "no_label_term", ""


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    surfaced = [s for s in report["surfaced"] if not s["generic"]]
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    entities = sorted({s["entity"] for s in surfaced})
    missing = [e for e in entities if e not in cache]
    for e in missing:
        cache[e] = {"text": fetch_label_text(e)}
        time.sleep(0.3)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    for s in surfaced:
        rec = cache.get(s["entity"], {})
        text = rec.get("text")
        if not text:
            s["label_verdict"], s["label_terms"] = "no_label", ""
            continue
        s["label_verdict"], s["label_terms"] = match(text, s["module"], s["direction"])

    counts = {}
    per_entity = {}
    for s in surfaced:
        counts[s["label_verdict"]] = counts.get(s["label_verdict"], 0) + 1
        per_entity.setdefault(s["entity"], {}).setdefault(s["label_verdict"], 0)
        per_entity[s["entity"]][s["label_verdict"]] += 1

    # Only directional matches are scoreable: a directional term was found
    # and it either agrees (supported), disagrees (contradicted) or both
    # (mixed). term_present / no_label_term carry no directional signal.
    checkable = counts.get("supported", 0) + counts.get("contradicted", 0) \
        + counts.get("mixed", 0)
    out = {
        "ground_truth": "openFDA drug label text (adverse_reactions, warnings, "
                        "boxed_warning, indications, clinical_pharmacology)",
        "caveats": [
            "no_label_term != model error: labels name clinical endpoints, "
            "not intermediate physiology",
            "unmappable modules have no label-expressible terminology and are "
            "not scored",
            "presence-based matching; direction only via directional terms",
        ],
        "n_surfaced_nongeneric": len(surfaced),
        "entities_with_label": sum(1 for e in entities if cache.get(e, {}).get("text")),
        "entities_no_label": sorted(e for e in entities if not cache.get(e, {}).get("text")),
        "verdict_counts": counts,
        "precision_on_checkable": (
            round(counts.get("supported", 0) / checkable, 4) if checkable else None),
        "per_entity": per_entity,
        "surfaced": surfaced,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"label check: {counts}")
    print(f"supported/checkable = {counts.get('supported',0)}/{checkable}")
    for s in surfaced:
        v = s["label_verdict"]
        if v in ("contradicted", "mixed", "supported"):
            print(f" {v:12s} {s['entity']:16s} -> {s['module']:20s} {s['direction']} "
                  f"[{s['label_terms'][:60]}]")


if __name__ == "__main__":
    main()
