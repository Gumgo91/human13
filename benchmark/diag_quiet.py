"""Diagnose external-benchmark quiet violations: which upstream terms drive
each module that was expected to stay quiet."""
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from backend.body_library import MODULES  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402
from external_eval import resolve  # noqa: E402

EXPR = {m["id"]: m["target"] for m in MODULES}
TERM = re.compile(r"([+-]?\s*\d*\.?\d+)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)")


def contributions(mid, traces):
    """Return [(input, coef, peak_contrib)] for each term of module mid."""
    out = []
    for m in TERM.finditer(EXPR.get(mid, "")):
        coef, name = float(m.group(1).replace(" ", "")), m.group(2)
        key = "body:" + name
        if key in traces:
            peak = float(np.max(np.abs(traces[key])))
            out.append((name, coef, round(coef * peak, 4)))
    return sorted(out, key=lambda t: -abs(t[2]))


def main():
    scenarios = [json.loads(l) for l in open(ROOT / "benchmark" / "known_pathways.jsonl", encoding="utf-8") if l.strip()]
    compounds = asyncio.run(resolve([s["entity"] for s in scenarios
                                     if s.get("entity") and s["kind"] == "chemical"]))
    for sc in scenarios:
        iv = Intervention(kind=sc["kind"], label=sc["text"], entity=sc.get("entity"),
                          quantity=sc.get("quantity"), unit=sc.get("unit"),
                          route=sc.get("route", "unspecified"),
                          duration_min=sc.get("duration_min", 30),
                          intensity_met=sc.get("intensity_met"),
                          body_channels=sc.get("given_channels") or [],
                          outcome_nodes=sc.get("outcome_nodes"))
        comp = {"0": compounds[sc["entity"]]} if sc.get("entity") in compounds else {}
        res = simulate(Interpretation(title=sc["id"], interventions=[iv]),
                       RunSettings(horizon_min=240), comp)
        traces = res["traces"]
        for mid in sc.get("expect_quiet") or []:
            key = "body:" + mid
            if key not in traces:
                continue
            peak = float(np.max(np.abs(traces[key])))
            if peak > 0.01:
                sign = "+" if float(np.max(traces[key])) >= -float(np.min(traces[key])) else "-"
                top = contributions(mid, traces)[:4]
                print(f"{sc['id']} :: {mid} {sign} peak={peak:.4f} <- {top}")
                for chain in res["body"].get("pathways", []):
                    ids = [e["id"] for e in chain]
                    if f"body:{mid}" in ids:
                        print("   path:", " -> ".join(e["id"].replace("body:", "") for e in chain))
                        break


if __name__ == "__main__":
    main()
