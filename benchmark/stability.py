"""Numerical stability verification for the 205 relative-deviation ODEs.

Every module integrates dx/dt = [3*tanh(u/3) - x] / tau. Analytic bound:
|3*tanh(u/3)| < 3 for any input u, and the state function is a contraction
(dF/dx = -1/tau < 0), so each module is driven toward a bounded steady state —
no blow-up is possible for bounded inputs. This harness verifies that
numerically under maximal simultaneous stress:

  1. Boundedness: |x| stays < ~3.1 for all modules under stacked extreme events.
  2. Finiteness: no NaN/inf anywhere.
  3. No persistent oscillation: late-window derivative sign changes stay low
     (homeostatic relaxation, not a limit cycle).
  4. Homeostasis: after events end, deviations relax back toward baseline.

Usage: python benchmark/stability.py
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from backend.body_library import MODULES  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402
from external_eval import resolve  # noqa: E402

BOUND = 3.05  # |3*tanh| < 3 plus numerical slack


def stress_plan():
    """Stack extreme simultaneous events: max channel drives + multiple drugs."""
    ch = ["heat", "hemorrhage", "meal", "salt", "dehydration", "stress",
          "infection", "hypoxia", "cold_exposure", "fasting"]
    ch2 = ["smoking", "orthostasis", "vomiting", "diarrhea", "injury",
           "sleep_deprivation", "pollution", "asthma", "immobility", "bigmeal"]
    events = [Intervention(kind="activity", label="extreme combined stress A",
                           duration_min=240, body_channels=ch, outcome_nodes=None),
              Intervention(kind="activity", label="extreme combined stress B",
                           duration_min=240, body_channels=ch2, outcome_nodes=None)]
    events += [Intervention(kind="chemical", label=e, entity=e, quantity=q, unit="mg",
                            route="oral", body_channels=[], outcome_nodes=None)
               for e, q in (("furosemide", 200), ("propranolol", 160), ("morphine", 60),
                            ("epinephrine", 1), ("insulin", 20))]
    return Interpretation(title="max stress", interventions=events)


def main():
    ents = ["furosemide", "propranolol", "morphine", "epinephrine", "insulin"]
    resolved = asyncio.run(resolve(ents))
    plan = stress_plan()
    compounds = {}
    for i, iv in enumerate(plan.interventions):
        if iv.kind == "chemical" and iv.entity in resolved:
            compounds[str(i)] = resolved[iv.entity]

    res = simulate(plan, RunSettings(horizon_min=1440), compounds)  # 24 h max
    traces = res["traces"]
    module_ids = [m["id"] for m in MODULES]

    # 1+2) boundedness & finiteness
    peaks, nonfinite = {}, []
    for mid in module_ids:
        v = np.asarray(traces["body:" + mid], dtype=float)
        peaks[mid] = float(np.max(np.abs(v)))
        if not np.all(np.isfinite(v)):
            nonfinite.append(mid)
    over = {m: round(p, 4) for m, p in peaks.items() if p > BOUND}

    # 3) persistent-oscillation screen: derivative sign changes in the last
    #    third. A relaxed system shows ~0-2; a limit cycle shows alternation.
    osc = {}
    for mid in module_ids:
        v = np.asarray(traces["body:" + mid], dtype=float)
        tail = v[int(2 * len(v) / 3):]
        d = np.diff(tail)
        crossings = int(np.sum(d[1:] * d[:-1] < 0))
        amp = float(np.ptp(tail))
        if crossings > 4 and amp > 0.01:
            osc[mid] = {"crossings": crossings, "late_amplitude": round(amp, 4)}

    # 4) homeostasis: stimulus ends at t=240; after 24 h deviations should
    #    relax for all but the slowest taus.
    end_dev = {mid: float(abs(traces["body:" + mid][-1])) for mid in module_ids}
    relaxed = sum(1 for v in end_dev.values() if v < 0.1)
    top_residual = sorted(end_dev.items(), key=lambda kv: -kv[1])[:8]

    report = {
        "analytic_bound": "|3*tanh(u/3)| < 3 and dF/dx = -1/tau < 0: contractive, bounded",
        "n_modules": len(module_ids), "horizon_min": 1440,
        "max_abs_deviation": round(max(peaks.values()), 4),
        "modules_over_bound": over, "nonfinite": nonfinite,
        "oscillating_modules": osc,
        "homeostasis": {"modules_within_0.1_at_end": relaxed,
                        "fraction": round(relaxed / len(module_ids), 4),
                        "top_residuals": [(k, round(v, 4)) for k, v in top_residual]},
    }
    Path(ROOT / "benchmark" / "stability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"stability: max|x|={report['max_abs_deviation']} "
          f"over_bound={len(over)} nonfinite={len(nonfinite)} "
          f"oscillators={len(osc)} relaxed={relaxed}/{len(module_ids)}")
    if over:
        print("  OVER:", over)
    if osc:
        print("  OSC:", osc)
    print("  residuals:", top_residual)


if __name__ == "__main__":
    main()
