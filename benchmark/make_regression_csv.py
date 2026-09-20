"""Emit results/regression_before_after.csv from archived reports vs the
post-change rerun files in raw/*_postfix.json."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "human13_ncs_validation_v3"

PAIRS = [
    ("internal", "benchmark/report.json", "raw/internal_postfix.json"),
    ("external", "benchmark/external_report.json", "raw/external_postfix.json"),
    ("holdout1", "benchmark/holdout_report.json", "raw/holdout1_postfix.json"),
    ("holdout2", "benchmark/holdout2_report.json", "raw/holdout2_postfix.json"),
    ("holdout3", "benchmark/holdout3_report.json", "raw/holdout3_postfix.json"),
    ("case_study", "benchmark/case_study_report.json", "raw/case_study_postfix.json"),
    ("sensitivity", "benchmark/sensitivity_report.json", "raw/sensitivity_postfix.json"),
    ("assimilation", "benchmark/assimilation_report.json", "raw/assimilation_postfix.json"),
]


def flat(d, p=""):
    """Flatten nested dicts AND numeric lists (index-keyed) so trajectory
    arrays are compared elementwise, not skipped."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, p + str(k) + ".")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, p + str(i) + ".")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield p[:-1], d


def main():
    rows = []
    for name, arch_rel, new_rel in PAIRS:
        arch_f, new_f = ROOT / arch_rel, V3 / new_rel
        if not new_f.exists():
            rows.append({"harness": name, "status": "NOT_RUN",
                         "archived": arch_rel, "rerun": new_rel})
            continue
        a = json.load(open(arch_f, encoding="utf-8"))
        n = json.load(open(new_f, encoding="utf-8"))
        fa, fn = dict(flat(a)), dict(flat(n))
        diffs = {k: (fa[k], fn[k]) for k in fa if k in fn
                 and abs(fa[k] - fn[k]) > 1e-9}
        only_a = set(fa) - set(fn)
        only_n = set(fn) - set(fa)
        recall_a = a.get("module_recall")
        recall_n = n.get("module_recall")
        quiet_a = a.get("quiet_violations")
        quiet_n = n.get("quiet_violations")
        status = ("IDENTICAL" if not diffs and not only_a and not only_n
                  else ("DEGRADED" if recall_n is not None
                        and recall_a is not None and recall_n < recall_a
                        else "DIFFERS"))
        rows.append({"harness": name, "archived": arch_rel,
                     "rerun": new_rel, "status": status,
                     "recall_archived": recall_a, "recall_new": recall_n,
                     "quiet_archived": quiet_a, "quiet_new": quiet_n,
                     "n_numeric_diffs": len(diffs),
                     "diffs": json.dumps(diffs)[:400],
                     "keys_only_archived": len(only_a),
                     "keys_only_new": len(only_n)})
    out = V3 / "results" / "regression_before_after.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['harness']:<14} {r['status']:<10} "
              f"recall {r.get('recall_archived')} -> {r.get('recall_new')} "
              f"quiet {r.get('quiet_archived')} -> {r.get('quiet_new')} "
              f"diffs={r.get('n_numeric_diffs')}")


if __name__ == "__main__":
    main()
