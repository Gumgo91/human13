"""Evidence accounting, independent of LLM assertions and numerical success.

Methodology references are not parameter evidence or regulatory certification.
The report describes this implementation's actual records, including omissions.
"""
from functools import lru_cache
from importlib.metadata import version
import platform
import hashlib
from pathlib import Path

GUIDANCE = [
    {"id": "fda_qsp_mabel_2026", "title": "FDA QSP-Based MABEL Dose Selection in FIH Trials",
     "url": "https://www.fda.gov/media/193230/download", "date": "2026-06", "status": "draft",
     "scope": "First-in-human MABEL dose selection", "sections": "III-V, printed pages 3-10"},
    {"id": "ich_m15_2026", "title": "FDA / ICH M15 General Principles for Model-Informed Drug Development",
     "url": "https://www.fda.gov/media/184747/download", "date": "2026-06", "status": "final",
     "scope": "Planning, evaluation, and reporting of model-informed drug development", "sections": "II-IV, Appendix 1-2"},
]

# Public controls, not the complete set of constants embedded in native modules.
CONTROLS = {
    "body_mass_kg": ("body mass", "kg", 30, 200),
    "activity_met": ("Unspecified exercise intensity", "MET", 1, 16),
    "gastric_half_min": ("Gastric emptying half-life", "min", 1, 300),
    "absorption_half_min": ("Absorption half-life", "min", 1, 300),
    "elimination_half_min": ("Central elimination half-life", "min", 1, 2880),
    "sglt1_scale": ("SGLT1 intestinal influx capacity", "dimensionless", 0, 3),
    "glut2_scale": ("GLUT2 portal-vein transfer capacity", "dimensionless", 0, 3),
    "glut5_scale": ("GLUT5 fructose transport capacity", "dimensionless", 0, 3),
    "sucrase_scale": ("Sucrase hydrolysis capacity", "dimensionless", 0, 3),
    "glut4_scale": ("GLUT4 peripheral utilization capacity", "dimensionless", 0, 3),
    "insulin_secretion_scale": ("Insulin secretion capacity", "dimensionless", 0, 3),
    "llm_scale": ("LLM -> existing-process coupling strength", "dimensionless", 0, 2),
}


@lru_cache(maxsize=1)
def software_versions():
    return {"python": platform.python_version(), **{name: version(name) for name in
            ("numpy", "scipy", "pint", "pydantic", "rdkit", "libroadrunner")}}


def sensitivity_parameters(result):
    generated = [{"id": "model:" + p["id"], "label": p.get("label", p["name"]),
                  "value": p["value"], "unit": p["unit"], "low": p["low"], "high": p["high"],
                  "origin": "generated_prior"} for p in result.get("generated", {}).get("parameters", [])]
    settings = result["plan"]["settings"]
    return generated + [{"id": "setting:" + key, "label": label, "value": settings[key],
                         "unit": unit, "low": low, "high": high, "origin": "scenario_setting"}
                        for key, (label, unit, low, high) in CONTROLS.items() if key in settings]


def parameter_ledger(result):
    """Never upgrade a citation on a node into evidence for a numerical value."""
    rows = []
    for p in sensitivity_parameters(result):
        rows.append({**p, "selected_value": p["value"], "candidates": [
            {"value": p["value"], "unit": p["unit"], "origin": p["origin"],
             "source_ids": [], "species": None, "experimental_system": None,
             "evidence_status": "unverified"}],
            "selection_rationale": ("Coefficient of a stored generated model, or a user-edited coefficient. No measured fit."
                                    if p["origin"] == "generated_prior" else "Stored scenario setting. No measured value confirmed."),
            "range_meaning": "Exploratory allowed range. Not a confidence interval or population distribution.",
            "alternative_evidence_status": "not_collected"})
    for node in result.get("nodes", []):
        if node.get("generated"): continue
        for index, p in enumerate(node.get("parameters", [])):
            rows.append({"id": f"node:{node['id']}:{index}", "label": p["name"], "unit": p["unit"],
                         "selected_value": p["value"], "origin": p.get("status", "unspecified"),
                         "node_id": node["id"], "candidates": [],
                         "context_sources": node.get("sources", []),
                         "selection_rationale": "Coefficient declared on the node. Node references were not verified as measured evidence for individual values.",
                         "alternative_evidence_status": "not_collected"})
    return {"coverage": "Generated parameters, public settings, and node-declared coefficients. Exhaustive extraction of code-internal constants is incomplete.",
            "complete": False, "rows": rows}


def build_study_report(result):
    generated = result.get("generated", {})
    numeric = result.get("validation", {})
    compiled = generated.get("status") == "compiled"
    checks = [
        {"id": "integration", "label": "Integration, finiteness, nonnegative states", "status": "passed" if numeric.get("nonnegative_check") else "not_recorded",
         "detail": f"{numeric.get('solver', 'not recorded')} - rtol={numeric.get('rtol')} - atol={numeric.get('atol')}. Applied to declared nonnegative states."},
        {"id": "generated", "label": "Generated-equation checks", "status": "passed" if compiled else "not_applied",
         "detail": " / ".join(generated.get("checks", [])) if compiled else "No generated model was run."},
        {"id": "balance", "label": "Declared conservation ledgers", "status": ("passed" if all(b["passed"] for b in numeric["balances"]) else "failed") if numeric.get("balances") else "not_declared",
         "detail": f"{len(numeric.get('balances', []))} ledger(s). Conservation of biological processes without a ledger is not guaranteed."},
        {"id": "zero_input", "label": "Zero-dose / zero-stimulus behavior", "status": "partial" if compiled else "not_evaluated",
         "detail": "Checks initial zero-stimulus derivatives of generated relative states and initial native couplings. The full zero-dose trajectory check is separate." if compiled else "The full zero-stimulus trajectory was not compared in this run."},
        {"id": "saturation", "label": "High-dose target saturation", "status": "not_evaluated",
         "detail": "No validated dose-response experiment on target occupancy, expression, or free concentration."},
    ]
    plan = result["plan"]
    return {
        "schema_version": "1.0", "guidance": GUIDANCE,
        "interpretation": "Development evidence record informed by FDA methodology documents. Not an FDA qualification or approval decision.",
        "question_of_interest": f"How do observations change over time when the mechanisms and assumptions declared in '{result['title']}' are applied?",
        "context_of_use": {"origin": "application_default", "purpose": "Exploratory physiology simulation for comparing mechanisms and assumptions",
                           "population": "Virtual subject specified by body mass only. No species, disease, age, or individual calibration.",
                           "decision_role": "Auxiliary tool for finding pathways and high-impact coefficients to verify in follow-up work",
                           "outside_scope": "Patient prediction, FIH/MABEL dose selection, prescribing or regulatory decisions"},
        "model_risk": {"status": "not_assessed", "model_influence": None, "wrong_decision_consequence": None,
                       "rationale": "The context, impact, and consequences of real decisions were not evaluated. No automatic risk score is assigned."},
        "verification": {"checks": checks, "balances": numeric.get("balances", []),
                         "reference_scope": numeric.get("reference_scope")},
        "calibration": {"status": "not_performed", "datasets": [], "parameter_changes": [],
                        "detail": "This integrated model has no measured fit, estimation objective, or calibration history."},
        "independent_validation": {"status": "not_performed", "datasets": [], "acceptance_criteria": [],
                                   "detail": "Predictions were not compared against independent data unused for development or calibration."},
        "applicability": [{"id": m["id"], "label": m["label"], "status": "unmodeled" if m["values"] is None else "not_empirically_validated",
                           "detail": m["explanation"]} for m in result["metrics"]],
        "parameters": parameter_ledger(result), "sensitivity_parameters": sensitivity_parameters(result),
        "uncertainty": {"parameter": "A local sensitivity analysis varying one selected coefficient at a time can be run.",
                        "structural": "Removing LLM->equation couplings and removing all LLM extensions can be compared. Not an exhaustive evaluation of alternative biological structures.",
                        "population": "Not evaluated. No virtual-population distribution, covariance, or species-difference data.",
                        "intervals": "Not computed. Exploratory parameter ranges must not be read as predictive confidence intervals."},
        "gaps": ["Missing evaluation of full relevant pathways, off-targets, and target expression/turnover is needed.",
                 "Per-parameter source values, assay conditions, species, alternative values, and selection rationale must be collected.",
                 "Calibration and independent-validation data must be separated, and per-observation tolerances predefined.",
                 "Beyond local analysis, interactions, joint uncertainty, alternative model structures, and population variability must be evaluated."],
        "refinement_plan": "When new measurements exist, first assign calibration vs independent-validation roles, then evaluate observation-prediction differences in a new model version preserving the existing plan.",
        "reproducibility": {"plan_hash": result["plan_hash"], "engine_hash": plan.get("engine_hash"),
                            "engine_version": plan.get("engine_version"), "source_hash": plan.get("source_hash"),
                            "report_builder_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "software_at_run": plan.get("software"),
                            "note": "Runs without a recorded software version are not retroactively attributed to the current version."},
    }
