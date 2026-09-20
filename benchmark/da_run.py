"""v3 data-assimilation validation experiments (E2-E5).

Fixed-test-condition suite. Every experiment consumes a config JSON so the
conditions are recorded, not chosen after seeing results.

    python benchmark/da_run.py --exp e2 --config human13_ncs_validation_v3/configs/da_e2.json

Outputs (under config["out_dir"]):
    results/<exp>_metrics.csv      per-trial metrics
    raw/trials/<exp>/t<NN>_<method>.npz   full 205-state trajectories
    raw/trials/<exp>/t<NN>_params.json    perturbation factors / update counts
    logs/<exp>.log
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from benchmark.da_core import (  # noqa: E402
    MODULE_IDS, RMSE_FLOOR, _interp, arm_blind, arm_enkf, arm_jump,
    arm_nudge, arm_persistence, decoupled_modules, perturbed_modules,
    run_sim, score_persistence, score_states, summarize)

# --------------------------------------------------------------------------
# pre-declared panels and latent sets (config records which were used)
# --------------------------------------------------------------------------
OBSERVED_FULL = ["sympathetic", "plasma_vol", "osmotic_drive", "urine_output",
                 "gfr", "sodium_load", "ventilation", "preload"]
OBSERVED_UPSTREAM = ["plasma_vol", "osmotic_drive", "preload", "sympathetic"]
OBSERVED_DOWNSTREAM = ["urine_output", "kaliuresis"]   # reward-only (oliguria)
PANELS = {"full": OBSERVED_FULL, "upstream": OBSERVED_UPSTREAM,
          "downstream": OBSERVED_DOWNSTREAM}

LATENT_HEM = ["renin", "angiotensin", "aldosterone", "adh", "aqp2",
              "kaliuresis", "edema"]   # 'thirst' absent from MODULES (recorded)

SCENARIOS = {
    "hemorrhage":  {"channels": ["hemorrhage"],  "dur": (90, 150), "horizon": 240},
    "dehydration": {"channels": ["dehydration"], "dur": (120, 240), "horizon": 360},
    "heat":        {"channels": ["heat"],        "dur": (30, 90),  "horizon": 240},
}

DRUG_SCENARIOS = {
    "propranolol": {"dose_mg": 40, "observed": ["contractility", "hrv",
                                              "sympathetic", "svr", "hr",
                                              "airway", "lipolysis", "renin"],
                    "latent": ["renin", "aldosterone", "lipolysis", "airway"],
                    "horizon": 240},
    "furosemide":  {"dose_mg": 40, "observed": ["urine_output", "sodium_load",
                                              "kaliuresis", "plasma_vol", "gfr",
                                              "osmotic_drive", "adh", "renin"],
                    "latent": ["renin", "aldosterone", "adh", "edema"],
                    "horizon": 240},
}


def _save_traces(out, exp, trial, method, res):
    d = Path(out) / "raw" / "trials" / exp
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"t{trial:02d}_{method}.npz",
                        times=np.asarray(res["time"], float),
                        **{m: np.asarray(res["traces"]["body:" + m])
                           for m in MODULE_IDS})
    return str(d / f"t{trial:02d}_{method}.npz")


def _row(exp, trial, method, **kw):
    r = {"experiment": exp, "trial": trial, "method": method}
    r.update(kw)
    return r


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# E2: rigorous forecast evaluation at fixed cuts
# --------------------------------------------------------------------------
def run_e2(cfg, out):
    sc = SCENARIOS[cfg["scenario"]]
    rows = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        dur = float(rng.uniform(*sc["dur"]))
        truth = run_sim([{"channels": sc["channels"], "duration_min": dur}],
                        sc["horizon"])
        for cut in cfg["cuts"]:
            blind = run_sim([], sc["horizon"])
            rec = {}
            for meth in cfg["methods"]:
                if meth == "blind":
                    res, meta = blind, {}
                elif meth == "nudge":
                    res, meta = arm_nudge(truth, OBSERVED_FULL, cut,
                                          sc["horizon"], rng, cfg["noise_sd"],
                                          obs_every=cfg["obs_every"],
                                          gain=cfg["gain"])
                elif meth == "jump":
                    res, meta = arm_jump(truth, OBSERVED_FULL, cut,
                                         sc["horizon"], rng, cfg["noise_sd"],
                                         obs_every=cfg["obs_every"])
                elif meth == "persistence":
                    held, meta = arm_persistence(truth, OBSERVED_FULL, cut,
                                                 rng, cfg["noise_sd"],
                                                 obs_every=cfg["obs_every"])
                    res = None
                if meth == "persistence":
                    per = score_persistence(truth, held, cut, sc["horizon"])
                else:
                    per = score_states(truth, res, OBSERVED_FULL + LATENT_HEM,
                                       cut, sc["horizon"])
                    rec[meth] = _save_traces(out, "e2", seed, meth, res)
                for panel_name, panel in [("observed", OBSERVED_FULL),
                                          ("latent", LATENT_HEM)]:
                    s = summarize(per, panel, truth, sc["horizon"], cut)
                    rows.append(_row("e2", seed, meth, scenario=cfg["scenario"],
                                     cut=cut, panel=panel_name,
                                     pooled_rmse=s["pooled_rmse"],
                                     n_obs=len([t for t in
                                                np.arange(cfg["obs_every"],
                                                          sc["horizon"] + 1,
                                                          cfg["obs_every"])
                                                if t <= cut]),
                                     n_active=s["n_active"],
                                     active_rmse=s["active_rmse"],
                                     raw_path=rec.get(meth, ""),
                                     **meta))
    _write_csv(Path(out) / "results" / "forecast_metrics.csv", rows)


# --------------------------------------------------------------------------
# E3: repeated trials x model mismatch x methods
# --------------------------------------------------------------------------
def run_e3(cfg, out):
    sc = SCENARIOS[cfg["scenario"]]
    rows = []
    dec = decoupled_modules() if "decoupled" in json.dumps(cfg["methods"]) else None
    for trial in range(cfg["n_trials"]):
        seed = cfg["seeds"][trial]
        rng = np.random.default_rng(seed)
        dur = float(rng.uniform(*sc["dur"]))
        truth = run_sim([{"channels": sc["channels"], "duration_min": dur}],
                        sc["horizon"])
        for mm_level in cfg["mismatch"]:
            twin, prec = (perturbed_modules(rng, mm_level)
                          if mm_level > 0 else (None, None))
            for meth in cfg["methods"]:
                if meth == "blind":
                    res, meta, mods = arm_blind([], sc["horizon"]), {}, twin
                elif meth == "nudge":
                    res, meta = arm_nudge(truth, OBSERVED_FULL, cfg["cut"],
                                          sc["horizon"], rng, cfg["noise_sd"],
                                          obs_every=cfg["obs_every"],
                                          gain=cfg["gain"], modules=twin)
                    mods = twin
                elif meth == "jump":
                    res, meta = arm_jump(truth, OBSERVED_FULL, cfg["cut"],
                                         sc["horizon"], rng, cfg["noise_sd"],
                                         obs_every=cfg["obs_every"],
                                         modules=twin)
                    mods = twin
                elif meth == "persistence":
                    held, meta = arm_persistence(truth, OBSERVED_FULL,
                                                 cfg["cut"], rng,
                                                 cfg["noise_sd"],
                                                 obs_every=cfg["obs_every"])
                    res, mods = None, None
                elif meth == "decoupled_blind":
                    res, meta, mods = arm_blind([], sc["horizon"],
                                              modules=dec), {}, dec
                elif meth == "decoupled_nudge":
                    res, meta = arm_nudge(truth, OBSERVED_FULL, cfg["cut"],
                                          sc["horizon"], rng, cfg["noise_sd"],
                                          obs_every=cfg["obs_every"],
                                          gain=cfg["gain"], modules=dec)
                    mods = dec
                elif meth == "enkf":
                    traj, meta = arm_enkf(truth, OBSERVED_FULL, cfg["cut"],
                                          sc["horizon"], rng, cfg["noise_sd"],
                                          n_members=cfg.get("enkf_members", 16),
                                          obs_every=cfg["obs_every"],
                                          mismatch=max(mm_level, 0.1))
                    # ensemble mean scored as 'res' via a pseudo-trace
                    res = {"time": meta["times"],
                           "traces": {"body:" + m: meta["ensemble_mean"][j]
                                      for j, m in enumerate(MODULE_IDS)}}
                    meta = {"n_members": meta["n_members"],
                            "updates_applied": len(meta["analyses"])}
                    mods = "enkf_members"
                else:
                    continue
                if meth == "persistence":
                    per = score_persistence(truth, held, cfg["cut"],
                                            sc["horizon"])
                else:
                    per = score_states(truth, res,
                                       OBSERVED_FULL + LATENT_HEM,
                                       cfg["cut"], sc["horizon"])
                for panel_name, panel in [("observed", OBSERVED_FULL),
                                          ("latent", LATENT_HEM)]:
                    s = summarize(per, panel, truth, sc["horizon"], cfg["cut"])
                    rows.append(_row("e3", trial, meth,
                                     mismatch=mm_level, seed=seed,
                                     dur=dur, panel=panel_name,
                                     pooled_rmse=s["pooled_rmse"],
                                     n_active=s["n_active"],
                                     active_rmse=s["active_rmse"],
                                     **{k: v for k, v in meta.items()
                                        if not isinstance(v, (list, dict))}))
                if cfg.get("save_traces") and res is not None and \
                        meth in cfg.get("trace_methods", []):
                    _save_traces(out, "e3", trial,
                                 f"{meth}_mm{mm_level}", res)
            # EnKF on designated trials only (expensive); ensemble gives
            # uncertainty estimates -> report interval coverage/width too
            if trial in cfg.get("enkf_trials", []) and \
                    mm_level in cfg.get("enkf_mismatch", [0.1]):
                traj, meta = arm_enkf(
                    truth, OBSERVED_FULL, cfg["cut"], sc["horizon"], rng,
                    cfg["noise_sd"], n_members=cfg.get("enkf_members", 16),
                    obs_every=cfg["obs_every"], mismatch=mm_level)
                res = {"time": meta["times"],
                       "traces": {"body:" + m: meta["ensemble_mean"][j]
                                  for j, m in enumerate(MODULE_IDS)}}
                per = score_states(truth, res, OBSERVED_FULL + LATENT_HEM,
                                   cfg["cut"], sc["horizon"])
                # 95% interval coverage on the forecast window
                gtimes = np.asarray(meta["times"], float)
                ttimes = np.asarray(truth["time"], float)
                fm = (gtimes > cfg["cut"]) & (gtimes <= sc["horizon"])
                cov, wid = {}, {}
                for j, m in enumerate(MODULE_IDS):
                    sd = np.asarray(meta["ensemble_sd"][j])[fm]
                    tv = _interp(ttimes, truth["traces"]["body:" + m],
                                 gtimes[fm])
                    mu = np.asarray(meta["ensemble_mean"][j])[fm]
                    cov[m] = float(np.mean(np.abs(tv - mu) <= 1.96 * sd))
                    wid[m] = float(np.mean(1.96 * sd))
                for panel_name, panel in [("observed", OBSERVED_FULL),
                                          ("latent", LATENT_HEM)]:
                    s = summarize(per, panel, truth, sc["horizon"], cfg["cut"])
                    rows.append(_row(
                        "e3", trial, "enkf", mismatch=mm_level, seed=seed,
                        dur=dur, panel=panel_name,
                        pooled_rmse=s["pooled_rmse"], n_active=s["n_active"],
                        active_rmse=s["active_rmse"],
                        coverage95=float(np.mean([cov[m] for m in panel
                                                  if m in cov])),
                        width95=float(np.mean([wid[m] for m in panel
                                               if m in wid])),
                        n_members=meta["n_members"],
                        updates_applied=len(meta["analyses"])))
                if cfg.get("save_traces"):
                    _save_traces(out, "e3", trial, f"enkf_mm{mm_level}", res)
                    pd = Path(out) / "raw" / "trials" / "e3"
                    with open(pd / f"t{trial:02d}_enkf_ens_mm{mm_level}.json",
                              "w", encoding="utf-8") as f:
                        json.dump({"times": meta["times"],
                                   "mean": meta["ensemble_mean"],
                                   "sd": meta["ensemble_sd"]}, f)
            if prec:
                pd = Path(out) / "raw" / "trials" / "e3"
                pd.mkdir(parents=True, exist_ok=True)
                with open(pd / f"t{trial:02d}_params_mm{mm_level}.json", "w",
                          encoding="utf-8") as f:
                    json.dump(prec, f)
    _write_csv(Path(out) / "results" / "robustness_metrics.csv", rows)


# --------------------------------------------------------------------------
# E4: observation panels
# --------------------------------------------------------------------------
def run_e4(cfg, out):
    sc = SCENARIOS[cfg["scenario"]]
    rows = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        truth = run_sim([{"channels": sc["channels"],
                          "duration_min": cfg.get("dur", 120)}], sc["horizon"])
        for pname, panel in PANELS.items():
            res, meta = arm_nudge(truth, panel, cfg["cut"], sc["horizon"],
                                  rng, cfg["noise_sd"],
                                  obs_every=cfg["obs_every"],
                                  gain=cfg["gain"])
            per = score_states(truth, res,
                               OBSERVED_FULL + LATENT_HEM,
                               cfg["cut"], sc["horizon"])
            _save_traces(out, "e4", seed, f"nudge_{pname}", res)
            for ep_name, ep in [("observed_full_panel", OBSERVED_FULL),
                                ("latent", LATENT_HEM)]:
                s = summarize(per, ep, truth, sc["horizon"], cfg["cut"])
                rows.append(_row("e4", seed, f"nudge_{pname}",
                                 obs_panel=pname, eval_panel=ep_name,
                                 pooled_rmse=s["pooled_rmse"],
                                 n_active=s["n_active"],
                                 active_rmse=s["active_rmse"],
                                 per_state=json.dumps(per)))
    _write_csv(Path(out) / "results" / "observation_panel_metrics.csv",
               rows)


# --------------------------------------------------------------------------
# E5: drug information x observation update (2x2)
# --------------------------------------------------------------------------
def run_e5(cfg, out):
    import asyncio
    from backend.chemistry import resolve_compound
    rows = []
    for drug, dcfg in DRUG_SCENARIOS.items():
        if drug not in cfg["drugs"]:
            continue
        comp = asyncio.run(resolve_compound(drug))
        truth_ev = [{"chemical": drug, "dose_mg": dcfg["dose_mg"],
                     "duration_min": 120}]
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(seed)
            truth = run_sim(truth_ev, dcfg["horizon"], compounds={drug: comp})
            for mm_level in cfg["mismatch"]:
                twin, _ = (perturbed_modules(rng, mm_level)
                           if mm_level > 0 else (None, None))
                for drug_info in [True, False]:
                    for assim in [True, False]:
                        arm = f"{'full' if drug_info else 'ablated'}_{'assim' if assim else 'open'}"
                        ev_t = truth_ev if drug_info else []
                        if assim:
                            res, meta = arm_nudge(
                                truth, dcfg["observed"], cfg["cut"],
                                dcfg["horizon"], rng, cfg["noise_sd"],
                                obs_every=cfg["obs_every"], gain=cfg["gain"],
                                modules=twin, compounds={drug: comp},
                                events=ev_t)
                        else:
                            res = run_sim(ev_t, dcfg["horizon"], modules=twin,
                                          compounds={drug: comp})
                            meta = {}
                        states = sorted(set(dcfg["observed"]) |
                                        set(dcfg["latent"]))
                        per = score_states(truth, res, states, cfg["cut"],
                                           dcfg["horizon"])
                        for ep_name, ep in [("observed", dcfg["observed"]),
                                            ("latent", dcfg["latent"])]:
                            s = summarize(per, ep, truth, dcfg["horizon"],
                                          cfg["cut"])
                            rows.append(_row(
                                "e5", seed, arm, drug=drug,
                                mismatch=mm_level, eval_panel=ep_name,
                                pooled_rmse=s["pooled_rmse"],
                                n_active=s["n_active"],
                                active_rmse=s["active_rmse"],
                                **{k: v for k, v in meta.items()
                                   if not isinstance(v, (list, dict))}))
                        if cfg.get("save_traces"):
                            _save_traces(out, "e5", seed,
                                         f"{drug}_{arm}_mm{mm_level}", res)
    _write_csv(Path(out) / "results" / "joint_drug_assimilation_metrics.csv",
               rows)


EXPS = {"e2": run_e2, "e3": run_e3, "e4": run_e4, "e5": run_e5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    out = cfg["out_dir"]
    Path(out, "results").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    EXPS[args.exp](cfg, out)
    with open(Path(out, "logs", f"{args.exp}.log"), "a", encoding="utf-8") as f:
        f.write(f"{args.exp} config={args.config} elapsed={time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
