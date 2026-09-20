"""PK sanity check: compare our generic compartment plasma curve against
published Cmax/Tmax for a few well-studied drugs. This does NOT tune the
model — it reports how far the exploratory parameters sit from literature,
as an honest deviation record for the paper's limitations section.

Usage:  python benchmark/pk_check.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.drug_pk import lookup as pk_lookup, DEFAULT_VD_L  # noqa: E402
from backend.simulation import simulate  # noqa: E402

BODY_MASS_KG = 70.

# Literature values (approximate; citations for the paper's reference list).
LITERATURE = [
    {"entity": "ibuprofen", "dose_mg": 400, "route": "oral",
     "lit_cmax_mg_L": (30., 40.), "lit_tmax_h": (1.0, 2.0),
     "ref": "Davies 1998 Clin Pharmacokinet 34:101"},
    {"entity": "caffeine", "dose_mg": 200, "route": "oral",
     "lit_cmax_mg_L": (4., 8.), "lit_tmax_h": (0.5, 1.5),
     "ref": "Fredholm 1999 Pharmacol Rev 51:83"},
    {"entity": "acetaminophen", "dose_mg": 1000, "route": "oral",
     "lit_cmax_mg_L": (10., 25.), "lit_tmax_h": (0.5, 1.0),
     "ref": "Prescott 1980 oral paracetamol kinetics"},
    {"entity": "metformin", "dose_mg": 500, "route": "oral",
     "lit_cmax_mg_L": (1., 2.), "lit_tmax_h": (2., 3.),
     "ref": "Scheen 1996 Clin Pharmacokinet 31:359"},
    {"entity": "amoxicillin", "dose_mg": 500, "route": "oral",
     "lit_cmax_mg_L": (5.5, 11.), "lit_tmax_h": (1., 2.),
     "ref": "Spyker 1977 Antimicrob Agents Chemother"},
]


def main():
    rows = []
    for lit in LITERATURE:
        iv = Intervention(kind="chemical", entity=lit["entity"], label=lit["entity"],
                          quantity=lit["dose_mg"], unit="mg", route=lit["route"],
                          duration_min=30, body_channels=[])
        res = simulate(Interpretation(title=lit["entity"], interventions=[iv]),
                       RunSettings(horizon_min=480), {})
        t = np.asarray(res["time"]); plasma = np.asarray(res["traces"]["drug0:plasma"])
        pk = pk_lookup(lit["entity"])
        vd = pk["vd_l_kg"] * BODY_MASS_KG if pk else DEFAULT_VD_L
        cmax = float(plasma.max()) / vd                # mg -> mg/L (literature Vd)
        tmax = float(t[int(plasma.argmax())]) / 60.     # min -> h
        lo, hi = lit["lit_cmax_mg_L"]; tlo, thi = lit["lit_tmax_h"]
        rows.append({"entity": lit["entity"], "dose_mg": lit["dose_mg"], "vd_L": round(vd, 1),
                     "pk_ref": (pk or {}).get("ref"),
                     "our_cmax": round(cmax, 2), "lit_cmax": [lo, hi],
                     "cmax_in_range": lo <= cmax <= hi,
                     "our_tmax_h": round(tmax, 2), "lit_tmax_h": [tlo, thi],
                     "tmax_in_range": tlo <= tmax <= thi, "ref": lit["ref"]})
        print(f"{lit['entity']:14s} Cmax ours={cmax:6.2f} lit={lo}-{hi} "
              f"{'OK' if lo <= cmax <= hi else 'OFF'} | Tmax ours={tmax:4.1f}h lit={tlo}-{thi}h")
    out = ROOT / "benchmark" / "pk_report.json"
    out.write_text(json.dumps({"model": "1st-order compartments + literature F/Vd/t½", "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
