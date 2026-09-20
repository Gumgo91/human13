import copy
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend import app as api
from backend import analysis as analysis_module
from backend.analysis import AnalysisRequest, analyze_snapshot, compare_metrics
from backend.contracts import RunSettings, Interpretation, Intervention
from backend.credibility import build_study_report
from backend.simulation import simulate
from test_equation_engine import relay, event_plan


def snapshot(model=None):
    model = model or relay()
    result = simulate(event_plan(), RunSettings(horizon_min=60), {}, model)
    result["run_id"] = "1234567890abcdef"
    return {"result": result, "expansion": model.model_dump(), "llm_records": []}


def test_numeric_success_never_promotes_empirical_evidence_or_citations():
    data = snapshot()
    data["result"]["nodes"][0]["sources"] = [{"id": "invented", "title": "context", "url": "https://example.org"}]
    r = build_study_report(data["result"])
    assert r["verification"]["checks"][0]["status"] == "passed"
    assert r["calibration"]["status"] == r["independent_validation"]["status"] == "not_performed"
    assert r["model_risk"]["status"] == "not_assessed"
    assert r["model_risk"]["model_influence"] is None
    assert not r["parameters"]["complete"]
    generated = [p for p in r["parameters"]["rows"] if p["origin"] == "generated_prior"]
    assert generated and all(not p["candidates"][0]["source_ids"] for p in generated)
    assert all(p["candidates"][0]["species"] is None for p in generated)
    assert all(m["status"] != "validated" for m in r["applicability"])
    assert r["guidance"][0]["status"] == "draft"
    assert r["reproducibility"]["software_at_run"]["scipy"]
    del data["result"]["plan"]["software"]
    assert build_study_report(data["result"])["reproducibility"]["software_at_run"] is None


def test_oat_is_deterministic_changes_one_parameter_preserves_snapshot_and_ablation():
    data = snapshot()
    before = copy.deepcopy(data)
    request = AnalysisRequest(parameter_ids=["model:heart_gain"], fraction=.2)
    result = analyze_snapshot(data, request)
    again = analyze_snapshot(data, request)
    assert data == before
    assert result["baseline"] == again["baseline"]
    assert result["scenarios"] == again["scenarios"]
    assert result["llm_calls"] == 0
    scenarios = {r["id"]: r for r in result["scenarios"]}
    low = scenarios["model:heart_gain:low"]
    high = scenarios["model:heart_gain:high"]
    assert low["value"] == pytest.approx(.08) and high["value"] == pytest.approx(.12)
    assert all(s["status"] == "computed" for s in scenarios.values())
    def metric(s, key): return next(m["values"] for m in s["metrics"] if m["id"] == key)
    assert max(metric(low, "hr")) < max(metric(result["baseline"], "hr")) < max(metric(high, "hr"))
    assert np.max(np.abs(np.array(metric(scenarios["native_coupling_off"], "hr")) - 72)) < 1e-7
    assert metric(scenarios["native_coupling_off"], "model:obs:novel_output") == pytest.approx(metric(result["baseline"], "model:obs:novel_output"), abs=1e-6)
    missing = next(c for c in scenarios["extension_off"]["comparisons"] if c["id"] == "model:obs:novel_output")
    assert missing["status"] == "unavailable" and missing["max_absolute_delta"] is None
    assert result["analysis_plan"]["baseline_plan_hash"] == result["baseline"]["plan_hash"]


def test_zero_negative_and_fixed_parameters_are_reported_without_zero_division():
    model = relay()
    model.model.parameters[2].value = 0
    data = snapshot(model)
    r = analyze_snapshot(data, AnalysisRequest(parameter_ids=["model:heart_gain"], include_structural=False))
    assert [s["value"] for s in r["scenarios"]] == [-.2, .2]
    model.model.parameters[2].value = -.1
    model.model.parameters[2].low = -.1
    model.model.parameters[2].high = -.1
    r = analyze_snapshot(snapshot(model), AnalysisRequest(parameter_ids=["model:heart_gain"], include_structural=False))
    assert all(s["status"] == "unchanged" for s in r["scenarios"])


def test_failed_scenarios_are_kept_and_not_repaired_or_silently_dropped():
    model = relay()
    model.model.states[0].upper = 1.01
    data = snapshot(model)
    r = analyze_snapshot(data, AnalysisRequest(parameter_ids=["model:strength"], fraction=.5, include_structural=False))
    assert r["scenarios"][0]["status"] == "computed"
    assert r["scenarios"][1]["status"] == "failed"
    assert r["scenarios"][1]["detail"] and r["scenarios"][1]["comparisons"] == []


def test_comparison_uses_matching_times_and_preserves_missing_outputs():
    def result(times, values): return {"time": times, "metrics": [{"id": "x", "label": "X", "unit": "mg", "values": values}]}
    base = result([0, 1, 2], [0, 1, 2])
    change = compare_metrics(base, result([0, .5, 2], [0, 1, 4]))[0]
    assert change["max_absolute_delta"] == 2 and change["time_at_max_min"] == 2
    assert compare_metrics(base, result([0, 2], None))[0]["status"] == "unavailable"


def test_native_sugar_transport_sensitivity_preserves_amount_ledgers():
    plan = Interpretation(title="Sugar 50g 섭취", interventions=[
        Intervention(kind="nutrition", label="Sugar 50g", entity="sucrose", quantity=50, unit="g", route="oral")])
    result = simulate(plan, RunSettings(horizon_min=120), {}, None)
    result["run_id"] = "fedcba0987654321"
    data = {"result": result, "expansion": None}
    report = analyze_snapshot(data, AnalysisRequest(parameter_ids=["setting:sglt1_scale", "setting:glut2_scale"]))
    assert report["structural_status"] == "not_available"
    assert len(report["scenarios"]) == 4
    for scenario in report["scenarios"]:
        assert scenario["status"] == "computed"
        assert scenario["verification"]["balances"]
        assert all(b["passed"] for b in scenario["verification"]["balances"])
        glucose = next(c for c in scenario["comparisons"] if c["id"] == "glucose")
        assert glucose["max_absolute_delta"] > .001


def test_analysis_api_stores_reproducible_report_without_llm_and_rejects_invalid_ids(monkeypatch, tmp_path):
    data = snapshot()
    monkeypatch.setattr(api, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(analysis_module, "ANALYSES", tmp_path / "analyses")
    def forbidden(*a, **k): raise AssertionError("Analysis must never call LLM")
    monkeypatch.setattr(api.llm, "expand", forbidden)
    monkeypatch.setattr(api.llm, "interpret", forbidden)
    api.RUNS.mkdir()
    path = api.RUNS / (data["result"]["run_id"] + ".json")
    path.write_text(json.dumps(data), encoding="utf-8")
    with TestClient(api.app) as client:
        url = "/api/runs/" + data["result"]["run_id"]
        assert client.get(url).json()["study_report"]["independent_validation"]["status"] == "not_performed"
        r = client.post(url + "/analysis", json={"parameter_ids": ["model:heart_gain"]})
        assert r.status_code == 200, r.text
        report = r.json()
        assert client.get("/api/analyses/" + report["analysis_id"]).json() == report
        assert json.loads(path.read_text()) == data
        assert client.post(url + "/analysis", json={"parameter_ids": ["model:missing"]}).status_code == 422
        assert client.post(url + "/analysis", json={"parameter_ids": ["model:tau"] * 2}).status_code == 422
        assert client.post(url + "/analysis", json={"fraction": "Infinity"}).status_code == 422
        assert client.post(url + "/analysis", json={"fraction": -.1}).status_code == 422
        assert client.get("/api/analyses/not-a-run").status_code == 404
        assert client.get("/api/guidance").json()["references"][0]["status"] == "draft"
