"""Deep case study: COX selectivity -> opposing thrombotic direction.

celecoxib (COX-2 selective, live ChEMBL lookup) vs ibuprofen (nonselective,
built-in profile). Both reach platelet/thrombin through the SAME two target
rows - the difference is which arm carries the measured potency:

  COX-1 activity -> platelet+ arm   (platelet TXA2, pro-aggregation)
  COX-2 activity -> platelet- arm   (endothelial PGI2, anti-aggregation, x0.5)
                 -> coronary_flow+, pulmonary_vr- (vasodilator arm)

Inhibiting COX-1 suppresses aggregation; inhibiting the COX-2 arm REMOVES
the anti-aggregatory PGI2 signal -> net prothrombotic shift.

Ablations show each edge is load-bearing, not decoration:
  A) delete the PGI2 arm from the COX-2 anchor row -> celecoxib loses its
     prothrombotic direction (platelet goes quiet)
  B) strip COX-1/generic-COX records from ibuprofen, keep only COX-2 ->
     ibuprofen flips prothrombotic (becomes celecoxib-like)

Usage:  python benchmark/case_study.py [-o benchmark/case_study_report.json]
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import backend.drug_targets as dt  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402
from benchmark.external_eval import resolve, direction  # noqa: E402

WATCH = ["platelet", "thrombin", "fibrin", "coronary_flow", "pulmonary_vr",
         "prostaglandin", "consumption_coag"]


def peaks(traces):
    return {m: {"sign": direction(traces, m)[0],
                "peak": round(direction(traces, m)[1], 3)}
            for m in WATCH if "body:" + m in traces}


def run(entity, record, outcome=None):
    iv = Intervention(kind="chemical", label=f"took {entity}", entity=entity,
                      quantity=400, unit="mg", route="oral", duration_min=30,
                      body_channels=[], outcome_nodes=outcome)
    return simulate(Interpretation(title=entity, interventions=[iv]),
                    RunSettings(horizon_min=240), {"0": record})


def describe(entity, record, res):
    targets = []
    for a in record.get("activities", []):
        if re.search(r"cox|cyclooxygenase|ptgs|prostaglandin", str(a.get("target_pref_name") or ""), re.I):
            targets.append({"target": a.get("target_pref_name"),
                            "type": a.get("standard_type"),
                            "value": a.get("standard_value"),
                            "units": a.get("standard_units"),
                            "source": "ChEMBL activity record"})
    for m in record.get("mechanisms", []):
        targets.append({"target": m.get("mechanism_of_action") or m.get("target_name"),
                        "type": m.get("action_type"), "source": "mechanism record"})
    anchors = [{"module": a.get("id"), "sign": a.get("sign"), "gain": a.get("gain"),
                "target": a.get("target"), "action": a.get("action"),
                "affinity_m": a.get("affinity_m"), "basis": a.get("basis")}
               for a in res["body"].get("drug_anchors", [])]
    return {"entity": entity, "targets": targets, "anchors": anchors,
            "peaks": peaks(res["traces"]),
            "outcome_paths": res["body"].get("outcome_paths", [])}


def ablate_cox2_row():
    """Remove the endothelial PGI2 arm from the COX-2 anchor row."""
    for i, (pat, alist, note) in enumerate(dt.TARGET_ANCHORS):
        if "cox-?2" in pat or "ptgs2" in pat:
            stripped = [a for a in alist if a[0] not in
                        ("platelet", "coronary_flow", "pulmonary_vr")]
            dt.TARGET_ANCHORS[i] = (pat, stripped, note)
            return stripped
    raise RuntimeError("cox-2 anchor row not found")


def ablate_cox1_row():
    """Remove the platelet TXA2 arm: delete COX-1 + generic-COX anchor rows."""
    markers = ("cox-?1", "ptgs1", "cyclooxygenase-?1",
               "prostaglandin.?endo", "ptgs\\b")
    removed, keep = [], []
    for row in dt.TARGET_ANCHORS:
        (removed if any(m in row[0] for m in markers) else keep).append(row)
    dt.TARGET_ANCHORS[:] = keep
    return [(r[0], r[1]) for r in removed]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "benchmark" / "case_study_report.json"))
    args = ap.parse_args()
    recs = await resolve(["celecoxib", "ibuprofen"])
    out = {}

    base_cel = run("celecoxib", recs["celecoxib"], outcome=["platelet", "thrombin"])
    base_ibu = run("ibuprofen", recs["ibuprofen"], outcome=["platelet", "thrombin"])
    out["celecoxib"] = describe("celecoxib", recs["celecoxib"], base_cel)
    out["ibuprofen"] = describe("ibuprofen", recs["ibuprofen"], base_ibu)

    # Ablation A: celecoxib without the endothelial PGI2 arm
    stripped = ablate_cox2_row()
    abl_a = run("celecoxib", recs["celecoxib"])
    out["ablation_A_no_pgi2_arm"] = {
        "removed_anchors": stripped,
        "celecoxib_platelet": peaks(abl_a["traces"]).get("platelet"),
        "celecoxib_thrombin": peaks(abl_a["traces"]).get("thrombin"),
        "interpretation": "without the PGI2 arm the prothrombotic signal should collapse"}

    # restore the row before ablation B
    import importlib
    importlib.reload(dt)

    # Ablation B: network without the COX-1/generic-COX platelet arm.
    # Ibuprofen then inhibits only the COX-2 arm -> should flip prothrombotic.
    removed = ablate_cox1_row()
    abl_b = run("ibuprofen", recs["ibuprofen"])
    out["ablation_B_no_cox1_arm"] = {
        "removed_anchor_rows": [r[0] for r in removed],
        "ibuprofen_platelet": peaks(abl_b["traces"]).get("platelet"),
        "ibuprofen_thrombin": peaks(abl_b["traces"]).get("thrombin"),
        "interpretation": "without the COX-1 arm, residual COX-2 inhibition "
                          "should flip ibuprofen prothrombotic (celecoxib-like)"}

    out["mechanism_summary"] = (
        "Same network, same two COX rows. Direction is decided by which arm "
        "carries measured potency (ChEMBL selectivity), not by a drug->effect table.")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    for drug in ("celecoxib", "ibuprofen"):
        p = out[drug]["peaks"]
        print(f"{drug}: platelet {p['platelet']['sign']}{p['platelet']['peak']} "
              f"thrombin {p['thrombin']['sign']}{p['thrombin']['peak']} "
              f"coronary {p['coronary_flow']['sign']}{p['coronary_flow']['peak']}")
    print("ablation A celecoxib platelet:", out["ablation_A_no_pgi2_arm"]["celecoxib_platelet"])
    print("ablation B ibuprofen platelet:", out["ablation_B_no_cox1_arm"]["ibuprofen_platelet"])


if __name__ == "__main__":
    asyncio.run(main())
