"""Raw-only regeneration of every v4 result table + paired bootstrap CIs.

Reads ONLY stored artifacts:
  raw/runs/<exp>/.../*.npz          method trajectories (all states)
  raw/trials/<scen>/seed*.npz       immutable truth + shared obs + params
  raw/prcp_extracted/*.npz          per-subject PRCP 1-min series
and writes results/*.csv that match the run-time tables. No in-memory
objects from the runs are needed: scoring is recomputed from npz files.
`persistence` rows are reconstructed from the stored observation stream
(last shared obs held constant) — also deterministic from raw.

Comparisons are PAIRED at the seed/subject level (never pooled across
timepoints, methods, or states): bootstrap resamples trial ids, and we
report the mean paired difference, its 95% CI, and the fraction of
trials in which the candidate method is worse than the reference.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from benchmark.da_v4 import (score_states, score_persistence, summarize,  # noqa: E402
                             state_key, write_csv)

RAW = ROOT / "human13_ncs_validation_v4" / "raw"
RES = ROOT / "human13_ncs_validation_v4" / "results"
CFG = ROOT / "human13_ncs_validation_v4" / "configs"


def _cut(exp, default=120.0):
    p = CFG / f"{exp}_used.json"
    if p.exists():
        return float(json.load(open(p, encoding="utf-8"))["cut"])
    return default

PANELS_HEM = {
    "full": ["sympathetic", "plasma_vol", "osmotic_drive", "urine_output",
             "gfr", "sodium_load", "ventilation", "preload"],
    "upstream": ["plasma_vol", "osmotic_drive", "preload", "sympathetic"],
    "downstream": ["urine_output", "kaliuresis"],
    "downstream4": ["urine_output", "kaliuresis", "gfr", "sodium_load"]}
HIDDEN_HEM = ["renin", "angiotensin", "aldosterone", "adh", "aqp2", "edema"]
HORIZON = 240.0

DRUG_PANELS = {
    "propranolol": {"observed": ["contractility", "hrv", "sympathetic",
                                 "svr", "hr", "airway", "lipolysis",
                                 "renin"],
                    "hidden": ["aldosterone", "angiotensin",
                               "coronary_flow"]},
    "furosemide": {"observed": ["urine_output", "sodium_load",
                                "kaliuresis", "plasma_vol", "gfr",
                                "osmotic_drive", "adh", "renin"],
                   "hidden": ["aldosterone", "angiotensin", "edema"]}}


def _load_run(path):
    z = np.load(path, allow_pickle=True)
    return {"time": z["times"].tolist(),
            "traces": {k[len("state::"):]: z[k] for k in z.files
                       if k.startswith("state::")}}


def _load_trial(scen, seed):
    z = np.load(RAW / "trials" / scen / f"seed{seed}.npz",
                allow_pickle=True)
    truth = {"time": z["times"].tolist(),
             "traces": {k[len("state::"):]: z[k] for k in z.files
                        if k.startswith("state::")}}
    obs = {}
    for k in z.files:
        if k.startswith("obs::"):
            _, panel, m = k.split("::", 2)
            obs.setdefault(panel, {})[m] = z[k].tolist()
    meta = json.loads(bytes(z["meta"].tobytes()).decode())
    return truth, obs, meta, z["obs_times"].tolist()


def _score_run(truth, run, states, cut, horizon):
    return score_states(truth, run, states, cut, horizon)


def _persist_per(truth, obs_vals, obs_times, cut, horizon):
    idx = [i for i, t in enumerate(obs_times) if t < cut]
    held = {m: v[idx[-1]] for m, v in obs_vals.items()} if idx else {}
    return score_persistence(truth, held, cut, horizon)


def aggregate_e2():
    rows, long_rows = [], []
    base = RAW / "runs" / "e2"
    if not base.exists():
        return
    for scen_dir in sorted(base.iterdir()):
        for seed_dir in sorted(scen_dir.iterdir()):
            seed = int(seed_dir.name.replace("seed", ""))
            truth, obs, meta, obs_times = _load_trial(scen_dir.name, seed)
            for cut_dir in sorted(seed_dir.iterdir(),
                                  key=lambda p: int(p.name[3:])):
                cut = float(cut_dir.name[3:])
                saved = {npz.stem: npz for npz in cut_dir.glob("*.npz")}
                for meth in saved.keys() | {"persistence"}:
                    npz = saved.get(meth)
                    run_id = (f"e2/{scen_dir.name}/seed{seed}/"
                              f"cut{int(cut)}/{meth}")
                    if meth == "persistence":
                        per = _persist_per(truth, obs["full"],
                                           obs_times, cut, HORIZON)
                    else:
                        run = _load_run(npz)
                        per = _score_run(
                            truth, run,
                            PANELS_HEM["full"] + HIDDEN_HEM,
                            cut, HORIZON)
                    for ep_name, ep in [("observed", PANELS_HEM["full"]),
                                        ("hidden", HIDDEN_HEM)]:
                        s = summarize({m: per[m] for m in ep if m in per},
                                      ep, truth, HORIZON, cut)
                        rows.append({"experiment": "e2", "trial": seed,
                                     "method": meth,
                                     "scenario": scen_dir.name,
                                     "cut": int(cut), "dur": meta["dur"],
                                     "panel": ep_name, **s,
                                     "run_id": run_id})
                    for m, v in per.items():
                        long_rows.append({"experiment": "e2", "trial": seed,
                                          "method": meth, "state": m,
                                          "eval_panel": "all",
                                          "cut": int(cut),
                                          "rmse": v["rmse"],
                                          "rmse_norm": v["rmse_norm"],
                                          "unit": v["unit"],
                                          "run_id": run_id})
    write_csv(RES / "forecast_metrics_v4.csv", rows)
    write_csv(RES / "forecast_per_state_v4.csv", long_rows)
    return rows


def aggregate_e3():
    rows, long_rows = [], []
    base = RAW / "runs" / "e3"
    if not base.exists():
        return
    for scen_dir in sorted(base.iterdir()):
        for seed_dir in sorted(scen_dir.iterdir()):
            seed = int(seed_dir.name.replace("seed", ""))
            truth, obs, meta, obs_times = _load_trial(scen_dir.name, seed)
            for mm_dir in sorted(seed_dir.iterdir()):
                mm = float(mm_dir.name[2:])
                saved = {npz.stem: npz for npz in mm_dir.glob("*.npz")}
                for meth in saved.keys() | {"persistence"}:
                    npz = saved.get(meth)
                    run_id = (f"e3/{scen_dir.name}/seed{seed}/"
                              f"mm{mm}/{meth}")
                    if meth == "persistence":
                        per = _persist_per(truth, obs["full"],
                                           obs_times, _cut("e3"), HORIZON)
                    else:
                        run = _load_run(npz)
                        per = _score_run(
                            truth, run,
                            PANELS_HEM["full"] + HIDDEN_HEM,
                            120.0, HORIZON)
                    for ep_name, ep in [("observed", PANELS_HEM["full"]),
                                        ("hidden", HIDDEN_HEM)]:
                        s = summarize({m: per[m] for m in ep if m in per},
                                      ep, truth, HORIZON, _cut("e3"))
                        rows.append({"experiment": "e3", "trial": seed,
                                     "method": meth, "mismatch": mm,
                                     "panel": ep_name, **s,
                                     "run_id": run_id})
                    for m, v in per.items():
                        long_rows.append({"experiment": "e3", "trial": seed,
                                          "method": meth, "state": m,
                                          "eval_panel": "all",
                                          "mismatch": mm,
                                          "rmse": v["rmse"],
                                          "rmse_norm": v["rmse_norm"],
                                          "unit": v["unit"],
                                          "run_id": run_id})
    write_csv(RES / "robustness_metrics_v4.csv", rows)
    write_csv(RES / "robustness_per_state_v4.csv", long_rows)
    return rows


def aggregate_e4():
    rows, long_rows = [], []
    base = RAW / "runs" / "e4"
    if not base.exists():
        return
    all_obs = sorted(set().union(*PANELS_HEM.values()))
    for scen_dir in sorted(base.iterdir()):
        for seed_dir in sorted(scen_dir.iterdir()):
            seed = int(seed_dir.name.replace("seed", ""))
            truth, obs, meta, obs_times = _load_trial(scen_dir.name, seed)
            for npz in sorted(seed_dir.glob("*.npz")):
                pname = npz.stem.replace("nudge_", "")
                run = _load_run(npz)
                per = _score_run(truth, run, all_obs + HIDDEN_HEM,
                                 _cut("e4"), HORIZON)
                run_id = (f"e4/{scen_dir.name}/seed{seed}/"
                          f"nudge_{pname}")
                for ep_name, ep in [("observed_union", all_obs),
                                    ("hidden", HIDDEN_HEM)]:
                    s = summarize({m: per[m] for m in ep if m in per},
                                  ep, truth, HORIZON, _cut("e4"))
                    rows.append({"experiment": "e4", "trial": seed,
                                 "method": f"nudge_{pname}",
                                 "obs_panel": pname,
                                 "eval_panel": ep_name, **s,
                                 "n_obs_states": len(PANELS_HEM[pname]),
                                 "run_id": run_id})
                for m, v in per.items():
                    long_rows.append({"experiment": "e4", "trial": seed,
                                      "method": f"nudge_{pname}",
                                      "state": m, "eval_panel": "all",
                                      "obs_panel": pname,
                                      "rmse": v["rmse"],
                                      "rmse_norm": v["rmse_norm"],
                                      "unit": v["unit"],
                                      "run_id": run_id})
    write_csv(RES / "observation_panel_metrics_v4.csv", rows)
    write_csv(RES / "observation_panel_per_state_v4.csv", long_rows)
    return rows


def aggregate_e5():
    rows, long_rows = [], []
    base = RAW / "runs" / "e5"
    if not base.exists():
        return
    for drug_dir in sorted(base.iterdir()):
        drug = drug_dir.name
        panels = DRUG_PANELS[drug]
        states = sorted(set(panels["observed"]) | set(panels["hidden"]))
        for seed_dir in sorted(drug_dir.iterdir()):
            seed = int(seed_dir.name.replace("seed", ""))
            truth, obs, meta, obs_times = _load_trial(drug, seed)
            for mm_dir in sorted(seed_dir.iterdir()):
                mm = float(mm_dir.name[2:])
                for npz in sorted(mm_dir.glob("*.npz")):
                    arm = npz.stem
                    run = _load_run(npz)
                    per = _score_run(truth, run, states, _cut("e5"), 240.0)
                    run_id = (f"e5/{drug}/seed{seed}/mm{mm}/{arm}")
                    for ep_name, ep in [("observed", panels["observed"]),
                                        ("hidden", panels["hidden"])]:
                        s = summarize({m: per[m] for m in ep if m in per},
                                      ep, truth, 240.0, _cut("e5"))
                        rows.append({"experiment": "e5", "trial": seed,
                                     "method": arm, "drug": drug,
                                     "mismatch": mm,
                                     "eval_panel": ep_name, **s,
                                     "run_id": run_id})
                    for m, v in per.items():
                        long_rows.append({"experiment": "e5", "trial": seed,
                                          "method": arm, "state": m,
                                          "eval_panel": "all",
                                          "drug": drug, "mismatch": mm,
                                          "rmse": v["rmse"],
                                          "rmse_norm": v["rmse_norm"],
                                          "unit": v["unit"],
                                          "run_id": run_id})
    write_csv(RES / "joint_drug_assimilation_v4.csv", rows)
    write_csv(RES / "joint_drug_per_state_v4.csv", long_rows)
    return rows


def aggregate_prcp():
    rows = []
    base = RAW / "runs" / "prcp"
    ext = RAW / "prcp_extracted"
    if not base.exists():
        return
    for subj_dir in sorted(base.iterdir()):
        z = np.load(ext / f"{subj_dir.name}.npz", allow_pickle=True)
        grid, hr, mp = z["grid_min"], z["hr"], z["map"]
        for npz in sorted(subj_dir.glob("*.npz")):
            r = np.load(npz, allow_pickle=True)
            meta = json.loads(bytes(r["meta"].tobytes()).decode())
            cut = meta["cut_min"]
            base_hr, base_map = meta["baseline"]
            tm, h, p = r["time"], r["dhr"], r["dmap"]
            m = grid >= cut
            eh = (hr[m] - base_hr) - np.interp(grid[m], tm, h)
            em = (mp[m] - base_map) - np.interp(grid[m], tm, p)
            eh = eh[np.isfinite(eh)]
            em = em[np.isfinite(em)]
            rows.append({"experiment": "prcp", "subject": subj_dir.name,
                         "method": npz.stem, "cut_min": cut,
                         "dhr_rmse": float(np.sqrt((eh ** 2).mean()))
                         if len(eh) else None,
                         "dmap_rmse": float(np.sqrt((em ** 2).mean()))
                         if len(em) else None,
                         "n_bins": int(m.sum()),
                         "run_id": f"prcp/{subj_dir.name}/{npz.stem}"})
    write_csv(RES / "prcp_metrics_v4.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# paired bootstrap — seed/subject-level only
# ---------------------------------------------------------------------------
def paired_bootstrap(rows, key_a, key_b, group, metric="pooled_rmse",
                     panel=None, n_boot=10000, seed=0):
    """Paired difference key_b - key_a within each group element (seed or
    subject). Positive delta means key_b is WORSE (larger error).
    `group` may be a column name or a tuple of column names — pairing must
    key every repeated condition (e.g. trial AND cut), otherwise dict
    overwrite silently keeps only the last value per trial."""
    rng = np.random.default_rng(seed)
    pa, pb = {}, {}
    for r in rows:
        if panel is not None and r.get("panel", r.get("eval_panel")) != panel:
            continue
        g = (tuple(r.get(k) for k in group) if isinstance(group, tuple)
             else r.get(group))
        v = r.get(metric)
        if v is None or v == "":
            continue
        if r["method"] == key_a:
            pa[g] = v
        elif r["method"] == key_b:
            pb[g] = v
    common = sorted(set(pa) & set(pb))
    if len(common) < 3:
        return None
    diffs = np.array([pb[g] - pa[g] for g in common])
    boot = rng.choice(diffs, size=(n_boot, len(diffs)),
                      replace=True).mean(1)
    return {"a": key_a, "b": key_b, "panel": panel, "metric": metric,
            "n_pairs": len(diffs),
            "mean_delta": float(diffs.mean()),
            "ci95_lo": float(np.percentile(boot, 2.5)),
            "ci95_hi": float(np.percentile(boot, 97.5)),
            "frac_worse": float((diffs > 0).mean()),
            "pooled_a": float(np.mean([pa[g] for g in common])),
            "pooled_b": float(np.mean([pb[g] for g in common]))}


def bootstrap_tables():
    out = []
    e2 = aggregate_e2() or []
    e3 = aggregate_e3() or []
    e4 = aggregate_e4() or []
    e5 = aggregate_e5() or []
    prcp = aggregate_prcp() or []
    for panel in ("observed", "hidden"):
        for m in ("nudge", "jump", "persistence"):
            r = paired_bootstrap(e2, "blind", m, ("trial", "cut"),
                                 panel=panel)
            if r:
                r["experiment"] = "e2"
                out.append(r)
    for panel in ("observed", "hidden"):
        for m in ("nudge", "jump", "decoupled_nudge"):
            r = paired_bootstrap(e3, "blind", m, ("trial", "mismatch"),
                                 panel=panel)
            if r:
                r["experiment"] = "e3"
                out.append(r)
    for panel in ("observed_union", "hidden"):
        for m in ("nudge_upstream", "nudge_downstream",
                  "nudge_downstream4"):
            r = paired_bootstrap(e4, "nudge_full", m, "trial",
                                 metric="mean_state_rmse",
                                 panel=panel)
            if r:
                r["experiment"] = "e4"
                out.append(r)
    for drug in DRUG_PANELS:
        sub = [r for r in e5 if r.get("drug") == drug]
        for panel in ("observed", "hidden"):
            for a, b in (("full_open", "ablated_open"),
                         ("full_open", "full_assim"),
                         ("ablated_open", "ablated_assim")):
                r = paired_bootstrap(sub, a, b, ("trial", "mismatch"),
                                     panel=panel)
                if r:
                    r["experiment"] = "e5"
                    r["drug"] = drug
                    out.append(r)
    for a, b in (("blind_raw", "blind_cal"), ("blind_raw", "nudge_cal"),
                 ("blind_raw", "persistence"), ("blind_raw", "naive_d0")):
        r = paired_bootstrap(prcp, a, b, "subject", metric="dhr_rmse")
        if r:
            r["experiment"] = "prcp"
            out.append(r)
        r = paired_bootstrap(prcp, a, b, "subject", metric="dmap_rmse")
        if r:
            r["experiment"] = "prcp"
            r["metric"] = "dmap_rmse"
            out.append(r)
    write_csv(RES / "paired_bootstrap_v4.csv", out)
    return out


if __name__ == "__main__":
    for name, fn in [("e2", aggregate_e2), ("e3", aggregate_e3),
                     ("e4", aggregate_e4), ("e5", aggregate_e5),
                     ("prcp", aggregate_prcp)]:
        rows = fn()
        print(f"{name}: {'n/a' if rows is None else len(rows)} rows")
    boot = bootstrap_tables()
    print(f"bootstrap comparisons: {len(boot)}")
