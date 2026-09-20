"""Hidden-pathway discovery harness.

The paper's scientific claim: the engine surfaces non-obvious downstream
effects that are NOT directly anchored by the drug's target profile — they
emerge only through body-network propagation. For each chemical scenario we
report modules that respond without being an anchor target, with the dominant
upstream chain. Each surfaced (entity, module) pair is classified against
documented pharmacology/physiology:

  documented   - indirect pathway is published (citable source)
  generic      - universal drug-exposure cascade (detox/hepatic), not a finding
  spurious     - no literature support (honest false positive)

Filters: body modules only (no primitives/rates), peak > 0.1, and the module
must not be an anchor target nor a benchmark expectation.

Usage: python benchmark/discovery.py [-o benchmark/discovery_report.json]
"""
import argparse
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

MODULE_IDS = {m["id"] for m in MODULES}
# Modules driven by the universal drug_exposure primitive — every chemical
# moves these, so they are a designed generic cascade, not a finding.
GENERIC = {"cyp450", "glutathione", "hepatic_load", "bilirubin", "emetic_drive",
           "dopamine_reward"}


def dominant_path(res, mid):
    for chain in res["body"].get("pathways", []):
        if f"body:{mid}" in [e["id"] for e in chain]:
            return " -> ".join(e["id"].replace("body:", "") for e in chain)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "benchmark" / "discovery_report.json"))
    args = ap.parse_args()
    scenarios = []
    for fname in ("known_pathways.jsonl", "holdout.jsonl"):
        for l in open(ROOT / "benchmark" / fname, encoding="utf-8"):
            s = json.loads(l)
            if s.get("kind") == "chemical" and s.get("entity"):
                scenarios.append(s)
    entities = sorted({s["entity"] for s in scenarios})
    compounds = asyncio.run(resolve(entities))

    surfaced = []
    for sc in scenarios:
        iv = Intervention(kind="chemical", label=sc["text"], entity=sc["entity"],
                          quantity=sc.get("quantity") or 1, unit=sc.get("unit") or "mg",
                          route=sc.get("route", "oral"), duration_min=sc.get("duration_min", 30),
                          body_channels=sc.get("given_channels") or [], outcome_nodes=None)
        comp = {"0": compounds[sc["entity"]]} if sc["entity"] in compounds else {}
        res = simulate(Interpretation(title=sc["id"], interventions=[iv]),
                       RunSettings(horizon_min=240), comp)
        traces = res["traces"]
        anchored = {a["id"] for a in res["body"].get("drug_anchors", [])}
        expected = set(sc.get("expect_responded") or {})
        for mid in MODULE_IDS:
            if mid in anchored or mid in expected or "body:" + mid not in traces:
                continue
            v = np.asarray(traces["body:" + mid], dtype=float)
            peak = float(np.max(np.abs(v)))
            if peak <= 0.1:
                continue
            sign = "+" if float(v.max()) >= -float(v.min()) else "-"
            surfaced.append({"scenario": sc["id"], "entity": sc["entity"], "module": mid,
                             "direction": sign, "peak": round(peak, 4),
                             "via": dominant_path(res, mid),
                             "generic": mid in GENERIC})
    generic = [s for s in surfaced if s["generic"]]
    specific = [s for s in surfaced if not s["generic"]]
    report = {"claim": "non-anchored modules respond via network propagation only",
              "note": "no post-hoc literature annotation is applied; every "
                      "surfaced entry is an unvalidated model output to be "
                      "checked against external evidence",
              "n_scenarios": len(scenarios), "n_surfaced": len(surfaced),
              "n_generic": len(generic), "n_specific": len(specific),
              "surfaced": surfaced}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"discovery: {len(surfaced)} surfaced | generic={len(generic)} "
          f"specific={len(specific)}")
    for s in specific:
        print(f" {s['entity']:16s} -> {s['module']:22s} {s['direction']} "
              f"({s['peak']}) via {s['via'][:70]}")


if __name__ == "__main__":
    main()
