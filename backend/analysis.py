"""Deterministic local perturbation analysis of a saved model. No LLM calls.

OAT changes one exposed parameter at a time. Structural comparisons are
ablations of already-declared hypotheses, not alternative-model validation.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
from pydantic import Field, model_validator
from .contracts import StrictModel, Interpretation, Expansion, RunSettings
from .credibility import sensitivity_parameters, build_study_report
from .simulation import simulate

ANALYSES = Path(__file__).parent / "data" / "analyses"


class AnalysisRequest(StrictModel):
    parameter_ids: list[str] = Field(default_factory=list, max_length=12)
    fraction: float = Field(default=.2, gt=0, le=.5)
    include_structural: bool = True

    @model_validator(mode="after")
    def unique_parameters(self):
        if len(set(self.parameter_ids)) != len(self.parameter_ids):
            raise ValueError("The same parameter cannot be selected twice.")
        if not self.parameter_ids and not self.include_structural:
            raise ValueError("Select either a parameter or a structural comparison.")
        return self


def metrics_view(result):
    return {"time": result["time"], "metrics": result["metrics"], "plan_hash": result["plan_hash"]}


def compare_metrics(baseline, scenario):
    """Compare in native observation units; absent outputs stay absent, never 0."""
    other = {m["id"]: m for m in scenario["metrics"]}
    comparisons = []
    for m in baseline["metrics"]:
        candidate = other.get(m["id"])
        row = {"id": m["id"], "label": m["label"], "unit": m["unit"]}
        if m["values"] is None or not candidate or candidate["values"] is None:
            comparisons.append({**row, "status": "unavailable", "max_absolute_delta": None,
                                "detail": "One model has no quantitative output, so no difference was computed."})
            continue
        if candidate["unit"] != m["unit"]:
            raise ValueError("The compared observations have different units.")
        a = np.asarray(m["values"])
        b = np.interp(baseline["time"], scenario["time"], candidate["values"])
        delta = b - a
        index = int(np.argmax(np.abs(delta)))
        comparisons.append({**row, "status": "compared", "max_absolute_delta": float(abs(delta[index])),
                            "signed_delta_at_max": float(delta[index]), "time_at_max_min": baseline["time"][index],
                            "final_delta": float(delta[-1]), "rmse_between_simulations": float(np.sqrt(np.mean(delta**2)))})
    return comparisons


def change_parameter(settings, parameter_id, value):
    payload = settings.model_dump()
    kind, key = parameter_id.split(":", 1)
    if kind == "model": payload["model_parameters"][key] = value
    else: payload[key] = value
    return RunSettings.model_validate(payload)


def analyze_snapshot(snapshot, request):
    saved = snapshot["result"]
    interpretation = Interpretation.model_validate(saved["interpretation"])
    settings = RunSettings.model_validate(saved["plan"]["settings"])
    expansion = Expansion.model_validate(snapshot["expansion"]) if snapshot.get("expansion") else None
    compounds = saved["compounds"]
    # Recompute with current numerical code, preserving the saved model itself.
    baseline = simulate(interpretation, settings, compounds, expansion)
    catalog = {p["id"]: p for p in sensitivity_parameters(baseline)}
    unknown = set(request.parameter_ids) - set(catalog)
    if unknown: raise ValueError("Unknown or disabled parameter: " + ", ".join(sorted(unknown)))
    planned = []
    for key in request.parameter_ids:
        p = catalog[key]
        step = request.fraction * (abs(p["value"]) if p["value"] != 0 else p["high"] - p["low"])
        for side, proposed in (("low", p["value"] - step), ("high", p["value"] + step)):
            value = max(p["low"], min(p["high"], proposed))
            planned.append({"id": f"{key}:{side}", "kind": "parameter", "parameter_id": key,
                            "label": f"{p['label']} · {'decrease' if side == 'low' else 'increase'}",
                            "baseline_value": p["value"], "value": value, "unit": p["unit"],
                            "changed": value != p["value"]})
    structural_available = bool(baseline.get("generated", {}).get("status") == "compiled" or
                                any(c["status"] == "compiled" for c in baseline.get("coupling", [])))
    if request.include_structural and structural_available:
        planned.extend([
            {"id": "native_coupling_off", "kind": "structural", "label": "Remove LLM -> existing-process couplings", "changed": settings.llm_scale != 0},
            {"id": "extension_off", "kind": "structural", "label": "Remove all LLM extensions", "changed": True},
        ])
    # Record the analysis plan before any perturbation result is observed.
    analysis_plan = {"created_at": datetime.now(timezone.utc).isoformat(), "request": request.model_dump(),
                     "baseline_plan_hash": baseline["plan_hash"], "scenarios": planned,
                     "method": "One coefficient varied at a time by ±fraction. At a zero baseline the allowed span x fraction is used; values are clipped at bounds.",
                     "criteria": "Unit, finiteness, range, and declared-conservation checks are preserved. A physiological error tolerance is not defined.",
                     "comparison": "Linear interpolation onto the baseline time grid. Max absolute difference in observation units, its time, the end difference, and between-curve RMSE.",
                     "scope": "Local assumed sensitivity. Not a confidence interval, independent validation, or global/population uncertainty analysis."}
    scenarios = []
    for planned_scenario in planned:
        row = dict(planned_scenario)
        if not row["changed"]:
            scenarios.append({**row, "status": "unchanged", "detail": "No effective change because of bounds or fixed values.", "comparisons": []})
            continue
        try:
            variant = settings.model_copy(deep=True)
            if row["kind"] == "parameter": variant = change_parameter(settings, row["parameter_id"], row["value"])
            elif row["id"] == "native_coupling_off": variant.llm_scale = 0
            else: variant.enable_llm_coupling = False
            result = simulate(interpretation, variant, compounds, expansion)
            scenarios.append({**row, "status": "computed", **metrics_view(result),
                              "comparisons": compare_metrics(baseline, result), "verification": result["validation"]})
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            # Failed scenarios remain in the report; never silently discard them.
            scenarios.append({**row, "status": "failed", "detail": str(error)[:400], "comparisons": []})
    return {"schema_version": "1.0", "run_id": saved.get("run_id"),
            "created_at": datetime.now(timezone.utc).isoformat(), "analysis_plan": analysis_plan,
            "analyzer_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "saved_plan_hash": saved["plan_hash"], "engine_changed": saved["plan"].get("engine_hash") != baseline["plan"].get("engine_hash"),
            "baseline": metrics_view(baseline), "model_plan": baseline["plan"], "study_report": build_study_report(baseline),
            "scenarios": scenarios, "unselected_parameters": sorted(set(catalog) - set(request.parameter_ids)),
            "structural_status": "included" if request.include_structural and structural_available else "not_available" if request.include_structural else "not_requested",
            "llm_calls": 0, "limitations": ["All other coefficients are held at their stored values. Coefficient interactions and covariance are not evaluated.",
                "The structural comparison removes LLM couplings already built; the ablated model is not necessarily correct.",
                "A zero difference does not prove the mechanism is irrelevant. Check inactive pathways, small local changes, and observation ranges.",
                "Scenarios failing numeric checks are kept and marked as failures, not dropped from results."]}


def save_analysis(report):
    analysis_id = uuid.uuid4().hex[:16]
    report["analysis_id"] = analysis_id
    ANALYSES.mkdir(parents=True, exist_ok=True)
    path = ANALYSES / (analysis_id + ".json")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)
    return report
