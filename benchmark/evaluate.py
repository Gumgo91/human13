"""Deterministic benchmark: given correct initial anchors, does the declared
network produce the expected downstream response?

This evaluates the NETWORK, not the LLM — scenarios carry `given_channels`
(what a correct interpretation should select) so the run is reproducible
offline. LLM anchor selection is scored separately via `llm_eval.py`.

Usage:  python benchmark/evaluate.py [--compounds cache_dir] [-o report.json]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402

PEAK_THRESHOLD = 0.02   # |relative deviation| counted as "responded"
QUIET_THRESHOLD = 0.01


def load_scenarios(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_compounds(cache_dir):
    """Resolve any cached compound records so drug scenarios can use them."""
    cache = Path(cache_dir)
    if not cache.exists():
        return {}
    out = {}
    for p in cache.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        q = rec.get("query")
        if q:
            out[q] = rec
    return out


def direction(traces, mid):
    v = np.asarray(traces["body:" + mid], dtype=float)
    peak_pos, peak_neg = float(v.max()), float(v.min())
    return "+" if peak_pos >= -peak_neg else "-", max(peak_pos, -peak_neg)


def run_scenario(sc, compounds_by_entity):
    # negated:true emulates an interpreter that found the stimulus explicitly
    # negated — it emits no anchors (negation handling is the LLM's job).
    channels = [] if sc.get("negated") else sc.get("given_channels")
    iv = Intervention(kind=sc["kind"], label=sc["text"], entity=sc.get("entity"),
                      quantity=sc.get("quantity"), unit=sc.get("unit"),
                      route=sc.get("route", "unspecified"),
                      duration_min=sc.get("duration_min", 30),
                      body_channels=channels,
                      outcome_nodes=sc.get("outcome_nodes"))
    comp = {}
    if sc.get("entity") and sc["entity"] in compounds_by_entity:
        comp = {"0": compounds_by_entity[sc["entity"]]}
    return simulate(Interpretation(title=sc["id"], interventions=[iv]),
                    RunSettings(horizon_min=240), comp)


def evaluate(scenarios, compounds_by_entity):
    results = []
    for sc in scenarios:
        try:
            res = run_scenario(sc, compounds_by_entity)
        except Exception as exc:
            results.append({"id": sc["id"], "error": str(exc)})
            continue
        traces = res["traces"]
        hits, misses = [], []
        for mid, want in (sc.get("expect_responded") or {}).items():
            key = "body:" + mid
            if key not in traces:
                misses.append({"module": mid, "reason": "no_trace"})
                continue
            sign, peak = direction(traces, mid)
            # "?" = direction-agnostic response (competing mechanisms leave the sign undetermined).
            ok = peak > PEAK_THRESHOLD and (want == "?" or sign == want)
            (hits if ok else misses).append(
                {"module": mid, "want": want, "sign": sign, "peak": round(peak, 4)})
        quiet_fails = []
        for mid in sc.get("expect_quiet") or []:
            key = "body:" + mid
            if key in traces and float(np.max(np.abs(traces[key]))) > QUIET_THRESHOLD:
                quiet_fails.append(mid)
        outcome = {"claimed": bool(sc.get("outcome_nodes")),
                   "found": any(p["found"] for p in res["body"].get("outcome_paths", []))}
        results.append({"id": sc["id"], "n_expected": len(sc.get("expect_responded") or {}),
                        "hits": len(hits), "hit_detail": hits, "misses": misses,
                        "quiet_violations": quiet_fails, "outcome": outcome,
                        "anchors": len(res["body"].get("drug_anchors", [])),
                        "engine_hash": res["plan"].get("engine_hash")})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(ROOT / "benchmark" / "scenarios.jsonl"))
    ap.add_argument("--compounds", default=str(ROOT / "backend" / "data" / "cache"))
    ap.add_argument("-o", "--out", default=str(ROOT / "benchmark" / "report.json"))
    args = ap.parse_args()
    scenarios = load_scenarios(args.scenarios)
    compounds = load_compounds(args.compounds)
    results = evaluate(scenarios, compounds)

    scored = [r for r in results if "error" not in r and r.get("n_expected")]
    recall = (sum(r["hits"] for r in scored) /
              sum(r["n_expected"] for r in scored)) if scored else 0.
    quiet_viol = sum(len(r.get("quiet_violations", [])) for r in results)
    claimed = [r for r in results if r.get("outcome", {}).get("claimed")]
    outcome_found = sum(r["outcome"]["found"] for r in claimed)
    errors = [r for r in results if "error" in r]

    report = {"n_scenarios": len(scenarios),
              "module_recall": round(recall, 4),
              "quiet_violations": quiet_viol,
              "outcome_paths": {"claimed": len(claimed), "found": outcome_found},
              "errors": errors, "results": results}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"scenarios={len(scenarios)} module_recall={recall:.1%} "
          f"quiet_violations={quiet_viol} outcome_found={outcome_found}/{len(claimed)} "
          f"errors={len(errors)}")
    for r in results:
        if r.get("misses") or r.get("quiet_violations") or r.get("error"):
            print(" MISS", r["id"], r.get("error") or r["misses"], r.get("quiet_violations"))


if __name__ == "__main__":
    main()
