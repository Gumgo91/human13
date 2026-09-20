"""External validation: literature-curated known pathways vs. network output.

Unlike scenarios.jsonl (co-developed with the model), every expectation in
known_pathways.jsonl cites an independent source — FDA labels, IUPHAR/BPS
Guide to Pharmacology, standard physiology texts, BioGears validation docs.
Chemical events are resolved live (PubChem/ChEMBL/Open Targets, disk-cached)
so this exercises the real molecule -> target -> network pipeline.

HONEST LIMITATION: this set is literature-curated, but the model was iterated
against it during development — equations and quiet expectations were corrected
(each with a documented literature justification in the scenario refs) after
observing failures. The reported score is therefore best read as iterative
literature-alignment, not a pristine holdout. A true held-out claim requires
scenarios added after the model is frozen.

Failure classes in the report:
  no_anchor   - the compound resolved but wired no relevant anchor (data gap)
  propagation - an anchor fired but the documented downstream module stayed quiet
  direction   - the module moved the wrong way

Usage:  python benchmark/external_eval.py [-o benchmark/external_report.json]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from backend.chemistry import resolve_compound  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402

PEAK_THRESHOLD = 0.02
QUIET_THRESHOLD = 0.01


def direction(traces, mid):
    v = np.asarray(traces["body:" + mid], dtype=float)
    peak_pos, peak_neg = float(v.max()), float(v.min())
    return "+" if peak_pos >= -peak_neg else "-", max(peak_pos, -peak_neg)


async def resolve(entities):
    """Resolve each unique entity once; unresolved lookups stay explicit.

    Virtual screening is stubbed out of this harness: it is the slowest stage
    and the external set only scores known-tier mechanisms, so VS hits can
    only add anchors, never satisfy a literature expectation differently.
    """
    import backend.vscreen as vscreen
    # resolve_compound imports virtual_screen lazily inside the function;
    # patch the module attribute so the lazy import picks up the stub.
    original = vscreen.virtual_screen
    async def no_screen(*a, **k):
        return []
    vscreen.virtual_screen = no_screen
    try:
        out = {}
        for entity in sorted(set(entities)):
            try:
                rec = await resolve_compound(entity)
            except Exception:
                rec = {"query": entity, "status": "unresolved",
                       "warnings": ["external lookup failed"], "mechanisms": [], "activities": []}
            out[entity] = rec
        return out
    finally:
        vscreen.virtual_screen = original


def run_scenario(sc, compounds):
    iv = Intervention(kind=sc["kind"], label=sc["text"], entity=sc.get("entity"),
                      quantity=sc.get("quantity"), unit=sc.get("unit"),
                      route=sc.get("route", "unspecified"),
                      duration_min=sc.get("duration_min", 30),
                      intensity_met=sc.get("intensity_met"),
                      body_channels=sc.get("given_channels") or [],
                      outcome_nodes=sc.get("outcome_nodes"))
    comp = {}
    if sc.get("entity") and sc["entity"] in compounds:
        comp = {"0": compounds[sc["entity"]]}
    return simulate(Interpretation(title=sc["id"], interventions=[iv]),
                    RunSettings(horizon_min=240), comp)


def evaluate(scenarios, compounds):
    results = []
    for sc in scenarios:
        try:
            res = run_scenario(sc, compounds)
        except Exception as exc:
            results.append({"id": sc["id"], "refs": sc.get("refs", []), "error": str(exc),
                            "n_expected": len(sc.get("expect_responded") or {}),
                            "hits": [], "misses": [], "quiet_violations": []})
            continue
        traces = res["traces"]
        anchored = {a["id"] for a in res["body"].get("drug_anchors", [])}
        hits, misses = [], []
        for mid, want in (sc.get("expect_responded") or {}).items():
            key = "body:" + mid
            if key not in traces:
                misses.append({"module": mid, "want": want, "reason": "no_trace"})
                continue
            sign, peak = direction(traces, mid)
            if peak <= PEAK_THRESHOLD:
                reason = "no_anchor" if sc["kind"] == "chemical" and not anchored else "propagation"
                misses.append({"module": mid, "want": want, "peak": round(peak, 4), "reason": reason})
            elif sign != want:
                misses.append({"module": mid, "want": want, "sign": sign,
                               "peak": round(peak, 4), "reason": "direction"})
            else:
                hits.append({"module": mid, "sign": sign, "peak": round(peak, 4)})
        quiet_fails = [mid for mid in sc.get("expect_quiet") or []
                       if "body:" + mid in traces
                       and float(np.max(np.abs(traces["body:" + mid]))) > QUIET_THRESHOLD]
        outcome = {"claimed": bool(sc.get("outcome_nodes")),
                   "found": any(p["found"] for p in res["body"].get("outcome_paths", []))}
        results.append({"id": sc["id"], "refs": sc.get("refs", []),
                        "n_expected": len(sc.get("expect_responded") or {}),
                        "hits": hits, "misses": misses, "quiet_violations": quiet_fails,
                        "anchors": sorted(anchored), "outcome": outcome})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(ROOT / "benchmark" / "known_pathways.jsonl"))
    ap.add_argument("-o", "--out", default=str(ROOT / "benchmark" / "external_report.json"))
    args = ap.parse_args()
    with open(args.scenarios, encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f if line.strip()]
    compounds = asyncio.run(resolve([s["entity"] for s in scenarios
                                   if s.get("entity") and s["kind"] == "chemical"]))
    unresolved = [e for e, r in compounds.items() if r.get("status") != "resolved"]
    results = evaluate(scenarios, compounds)

    scored = [r for r in results if "error" not in r]
    recall = (sum(len(r["hits"]) for r in scored) /
              sum(r["n_expected"] for r in scored)) if scored else 0.
    quiet_viol = sum(len(r["quiet_violations"]) for r in results)
    claimed = [r for r in results if r.get("outcome", {}).get("claimed")]
    outcome_found = sum(r["outcome"]["found"] for r in claimed)
    by_reason = {}
    for r in results:
        for m in r.get("misses", []):
            by_reason[m["reason"]] = by_reason.get(m["reason"], 0) + 1

    report = {"benchmark": "external, literature-curated (not fitted to the model)",
              "n_scenarios": len(scenarios), "module_recall": round(recall, 4),
              "misses_by_reason": by_reason, "quiet_violations": quiet_viol,
              "outcome_paths": {"claimed": len(claimed), "found": outcome_found},
              "unresolved_entities": unresolved, "results": results}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"external scenarios={len(scenarios)} recall={recall:.1%} "
          f"misses={by_reason} quiet_violations={quiet_viol} "
          f"outcome={outcome_found}/{len(claimed)} unresolved={unresolved}")
    for r in results:
        if r.get("misses") or r.get("quiet_violations") or r.get("error"):
            print(" MISS", r["id"], r.get("error") or r["misses"], r.get("quiet_violations"))


if __name__ == "__main__":
    main()
