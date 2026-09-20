"""Generate MANIFEST.csv — sha256 of every file in the v4 package."""
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "human13_ncs_validation_v4"


def main():
    rows = []
    for p in sorted(PKG.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.csv":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rows.append({"file": str(p.relative_to(PKG)).replace("\\", "/"),
                     "bytes": p.stat().st_size,
                     "sha256": h.hexdigest()})
    out = PKG / "MANIFEST.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"MANIFEST.csv: {len(rows)} files -> {out}")


if __name__ == "__main__":
    main()
