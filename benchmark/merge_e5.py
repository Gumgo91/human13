"""Merge e5 CSVs: the first e5 run scored propranolol with a body:-only
score_states (native 'hr' silently skipped). After the fix, a
propranolol-only rerun writes the corrected rows; merge those with the
furosemide rows preserved from the first run.

Usage: merge_e5.py <first_run_csv> <propranolol_rerun_csv> <final_csv>
"""
import csv
import sys


def main():
    first, rerun, final = sys.argv[1], sys.argv[2], sys.argv[3]
    furo = [r for r in csv.DictReader(open(first, encoding="utf-8"))
            if r.get("drug") == "furosemide"]
    prop = [r for r in csv.DictReader(open(rerun, encoding="utf-8"))
            if r.get("drug") == "propranolol"]
    rows = prop + furo
    with open(final, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"merged: {len(prop)} propranolol + {len(furo)} furosemide rows")


if __name__ == "__main__":
    main()
