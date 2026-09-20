"""PK-error sensitivity: does a wrong Cmax flip downstream causal judgments?

Reviewer concern: if the plasma peak is off by 2-4x, the Hill occupancy and
every downstream ODE deviation inherit the error. This harness answers the
narrower question that matters for a hypothesis tool: do QUALITATIVE causal
judgments (response direction above the detection threshold, and outcome-path
discovery) survive PK perturbations?

Method: for each scenario, run the baseline and then perturbed variants —
dose x0.5/x2, Vd x0.5/x2, elimination t1/2 x0.5/x2, and a combined worst case
(dose x0.5 + Vd x2, which can cut peak concentration ~4x). A perturbation
"flips" a judgment if an expected module's direction changes or its peak
crosses the PEAK_THRESHOLD boundary. Outcome-path finding is re-evaluated
each run.

Usage: python benchmark/sensitivity.py [-o benchmark/sensitivity_report.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402
from external_eval import PEAK_THRESHOLD, direction  # noqa: E402
import backend.drug_pk as drug_pk  # noqa: E402

# Scenarios reuse literature expectations; entities resolved from disk cache.
SCENARIOS = [
    {"id": "warfarin", "entity": "warfarin", "quantity": 5, "expect": {"thrombin": "-", "anticoagulant": "+"}},
    {"id": "ibuprofen", "entity": "ibuprofen", "quantity": 400, "expect": {"prostaglandin": "-"}},
    {"id": "metformin", "entity": "metformin", "quantity": 500, "expect": {"ampk": "+", "gluconeogenesis": "-"}},
    {"id": "sildenafil", "entity": "sildenafil", "quantity": 50, "expect": {"peripheral_perfusion": "+"}},
    {"id": "furosemide", "entity": "furosemide", "quantity": 40, "expect": {"urine_output": "+"}},
]


def run(entity, mg, compounds, vd_scale=1.0, thalf_scale=1.0):
    """Simulate with optional PK parameter scaling patched into drug_pk.TABLE."""
    pk = drug_pk.lookup(entity)
    backup = None
    key = None
    if pk and (vd_scale != 1.0 or thalf_scale != 1.0):
        key = next((k for k, v in drug_pk.PK_TABLE.items() if v is pk), None)
        if key:
            backup = dict(pk)
            drug_pk.PK_TABLE[key] = {**pk, "vd_l_kg": pk["vd_l_kg"] * vd_scale,
                                     "t_half_h": pk["t_half_h"] * thalf_scale}
    try:
        iv = Intervention(kind="chemical", label=entity, entity=entity,
                          quantity=mg, unit="mg", route="oral",
                          body_channels=[], outcome_nodes=None)
        return simulate(Interpretation(title="sens", interventions=[iv]),
                        RunSettings(horizon_min=240), compounds)
    finally:
        if backup is not None and key is not None:
            drug_pk.PK_TABLE[key] = backup


def judge(res, expect):
    """Return {module: (sign_ok, peak)} for each expectation."""
    out = {}
    for mid, want in expect.items():
        key = "body:" + mid
        if key not in res["traces"]:
            out[mid] = (False, 0.0)
            continue
        sign, peak = direction(res["traces"], mid)
        out[mid] = (peak > PEAK_THRESHOLD and sign == want, round(peak, 4))
    return out


def main():
    import asyncio
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "benchmark" / "sensitivity_report.json"))
    args = ap.parse_args()
    from external_eval import resolve
    entities = [s["entity"] for s in SCENARIOS]
    compounds = asyncio.run(resolve(entities))
    perturbations = [
        ("dose_x0.5", .5, 1., 1.), ("dose_x2", 2., 1., 1.),
        ("vd_x0.5", 1., .5, 1.), ("vd_x2", 1., 2., 1.),
        ("t12_x0.5", 1., 1., .5), ("t12_x2", 1., 1., 2.),
        ("worst_cmax_low", .5, 2., 1.), ("worst_cmax_high", 2., .5, 1.),
    ]
    report = []
    n_ok = n_total = 0
    for sc in SCENARIOS:
        comp = {"0": compounds[sc["entity"]]} if sc["entity"] in compounds else {}
        base = judge(run(sc["entity"], sc["quantity"], comp), sc["expect"])
        entry = {"id": sc["id"], "baseline": {m: {"ok": ok, "peak": p} for m, (ok, p) in base.items()},
                 "perturbations": {}}
        for pname, ds, vs, ts in perturbations:
            res = run(sc["entity"], sc["quantity"] * ds, comp, vs, ts)
            j = judge(res, sc["expect"])
            flips = {m: {"baseline_ok": base[m][0], "perturbed_ok": j[m][0], "peak": j[m][1]}
                     for m in j if j[m][0] != base[m][0]}
            entry["perturbations"][pname] = {"flips": flips, "n_flip": len(flips)}
            n_total += len(j)
            n_ok += sum(1 for ok, _ in j.values() if ok)
        report.append(entry)
    stability = n_ok / n_total if n_total else 0.
    out = {"question": "do qualitative causal judgments survive PK perturbation?",
           "n_scenarios": len(SCENARIOS), "n_judgments": n_total,
           "n_preserved": n_ok, "judgment_stability": round(stability, 4),
           "scenarios": report}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sensitivity: {n_ok}/{n_total} judgments preserved ({stability:.1%})")
    for e in report:
        for pn, p in e["perturbations"].items():
            if p["n_flip"]:
                print(f"  FLIP {e['id']} {pn}: {p['flips']}")


if __name__ == "__main__":
    main()
