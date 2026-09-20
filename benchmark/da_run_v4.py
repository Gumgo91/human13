"""v4 DA experiment runner — every method consumes the shared trial
artifact (one truth, one obs stream, one parameter realisation per seed).

Panels (pre-declared, disjoint enforced):
  hemorrhage obs: full8 / upstream4 / downstream2 (legacy) + downstream4
    (equal-N comparison vs upstream4, declared before results)
  hidden (hemorrhage): {renin,angiotensin,aldosterone,adh,aqp2,edema} —
    fixed states minus the union of ALL observed panels
  propranolol obs: {contractility,hrv,sympathetic,svr,hr,airway,lipolysis,
    renin}; hidden declared {aldosterone,angiotensin,coronary_flow}
  furosemide obs: {urine_output,sodium_load,kaliuresis,plasma_vol,gfr,
    osmotic_drive,adh,renin}; hidden declared {aldosterone,angiotensin,
    edema}

    python benchmark/da_run_v4.py --exp e2 --config <cfg.json>
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from benchmark.da_v4 import (  # noqa: E402
    METHODS, MODULE_IDS, RMSE_FLOOR, assert_disjoint, cfg_hash,
    decoupled_modules, gen_trial, noise_sd_for, perturbed_modules,
    save_run, save_trial, score_persistence, score_states, state_key,
    summarize, unique_run_path, write_csv, _seed_stream)

PANELS_HEM = {
    "full": ["sympathetic", "plasma_vol", "osmotic_drive", "urine_output",
             "gfr", "sodium_load", "ventilation", "preload"],
    "upstream": ["plasma_vol", "osmotic_drive", "preload", "sympathetic"],
    "downstream": ["urine_output", "kaliuresis"],
    # equal-N arm declared pre-results: 4 output-side states vs upstream4
    "downstream4": ["urine_output", "kaliuresis", "gfr", "sodium_load"]}
HIDDEN_HEM = ["renin", "angiotensin", "aldosterone", "adh", "aqp2", "edema"]

SCENARIOS = {
    "hemorrhage": {"channels": ["hemorrhage"], "dur": (90, 150),
                   "horizon": 240}}

DRUG_SCENARIOS = {
    "propranolol": {"dose_mg": 40, "horizon": 240,
                    "observed": ["contractility", "hrv", "sympathetic",
                                 "svr", "hr", "airway", "lipolysis",
                                 "renin"],
                    "hidden": ["aldosterone", "angiotensin",
                               "coronary_flow"]},
    "furosemide": {"dose_mg": 40, "horizon": 240,
                   "observed": ["urine_output", "sodium_load",
                                "kaliuresis", "plasma_vol", "gfr",
                                "osmotic_drive", "adh", "renin"],
                   "hidden": ["aldosterone", "angiotensin", "edema"]}}


def _per_state_rows(exp, trial_id, method, per, eval_panel, **kw):
    rows = []
    for m, v in per.items():
        rows.append({"experiment": exp, "trial": trial_id,
                     "method": method, "state": m, "eval_panel": eval_panel,
                     "rmse": v["rmse"], "mse": v["mse"],
                     "rmse_norm": v["rmse_norm"], "unit": v["unit"], **kw})
    return rows


def _trial_artifact(cfg, scenario, sc, seed, panels):
    tdir = Path(cfg["out_dir"]) / "raw" / "trials" / scenario
    tpath = tdir / f"seed{seed}.npz"
    if tpath.exists():  # reuse immutable artifact — identical stream
        z = np.load(tpath, allow_pickle=True)
        meta = json.loads(bytes(z["meta"].tobytes()).decode())
        truth = {"time": z["times"].tolist(),
                 "traces": {k[len("state::"):]: z[k]
                            for k in z.files if k.startswith("state::")}}
        obs = {}
        for p in panels:
            vals = {m: z[f"obs::{p}::{m}"].tolist() for m in panels[p]
                    if f"obs::{p}::{m}" in z.files}
            obs[p] = {"times": z["obs_times"].tolist(), "values": vals,
                      "noise_sd": {m: noise_sd_for(m) for m in vals}}
        trial = {"scenario": scenario, "seed": seed, "dur": meta["dur"],
                 "events": meta.get("events", []),
                 "truth": truth, "obs": obs, "params": {},
                 "enkf_members": [], "enkf_ic": np.zeros(
                     (cfg.get("enkf_members", 16), len(MODULE_IDS))),
                 "seeds": meta["seeds"], "obs_times": z["obs_times"].tolist()}
        return trial, meta
    trial = gen_trial(scenario, sc, seed, panels, cfg)
    save_trial(trial, tpath)
    return trial, {"dur": trial["dur"]}


def _need_params(trial, scenario, sc, seed, mm_level, cfg):
    """Parameter realisations are generated per-trial once and kept in the
    in-memory trial dict; the immutable file holds the record."""
    if str(mm_level) not in trial["params"]:
        rng_p = _seed_stream(seed, "params")
        for lv in cfg["mismatch"]:
            if lv > 0 and str(lv) not in trial["params"]:
                mods, rec = perturbed_modules(rng_p, lv)
                trial["params"][str(lv)] = {"modules": mods, "record": rec}
    ent = trial["params"].get(str(mm_level))
    return (ent["modules"], ent["record"]) if ent else (None, None)


def _need_enkf(trial, cfg):
    if not trial["enkf_members"]:
        rng_e = _seed_stream(trial["seed"], "enkf")
        members = []
        for _ in range(cfg.get("enkf_members", 16)):
            mods, rec = perturbed_modules(rng_e, cfg.get("enkf_spread", 0.15))
            members.append({"modules": mods, "record": rec})
        trial["enkf_members"] = members
        trial["enkf_ic"] = rng_e.normal(
            0, cfg.get("enkf_ic_sd", 0.02),
            size=(len(members), len(MODULE_IDS)))


def _eval_panels(panel_key):
    return [("observed", panel_key), ("hidden", HIDDEN_HEM)]


def run_e2(cfg, out):
    sc = SCENARIOS[cfg["scenario"]]
    assert_disjoint(PANELS_HEM, HIDDEN_HEM)
    rows, long_rows = [], []
    for seed in cfg["seeds"]:
        trial, _ = _trial_artifact(cfg, cfg["scenario"], sc, seed, PANELS_HEM)
        truth = trial["truth"]
        for cut in cfg["cuts"]:
            for meth in cfg["methods"]:
                t0 = time.time()
                if meth == "persistence":
                    held, meta = METHODS[meth](trial, "full", cut,
                                             sc["horizon"], None, cfg)
                    per = score_persistence(truth, held, cut, sc["horizon"])
                    res = None
                else:
                    res, meta = METHODS[meth](trial, "full", cut,
                                              sc["horizon"], None, cfg)
                    per = score_states(truth, res,
                                       PANELS_HEM["full"] + HIDDEN_HEM,
                                       cut, sc["horizon"])
                rt = time.time() - t0
                run_id = f"e2/{cfg['scenario']}/seed{seed}/cut{cut}/{meth}"
                if res is not None:
                    save_run(unique_run_path(out, run_id), res)
                for ep_name, ep in _eval_panels("full"):
                    panel = (PANELS_HEM["full"] if ep_name == "observed"
                             else HIDDEN_HEM)
                    s = summarize({m: per[m] for m in panel if m in per},
                                  panel, truth, sc["horizon"], cut)
                    rows.append({"experiment": "e2", "trial": seed,
                                 "method": meth, "scenario": cfg["scenario"],
                                 "cut": cut, "dur": trial["dur"],
                                 "panel": ep_name, **s,
                                 "runtime_s": round(rt, 3),
                                 "run_id": run_id,
                                 **{k: v for k, v in meta.items()
                                    if not isinstance(v, (list, dict,
                                                          np.ndarray))}})
                long_rows += _per_state_rows(
                    "e2", seed, meth, per, "all", cut=cut,
                    dur=trial["dur"], run_id=run_id)
    write_csv(Path(out) / "results" / "forecast_metrics_v4.csv", rows)
    write_csv(Path(out) / "results" / "forecast_per_state_v4.csv", long_rows)


def run_e3(cfg, out):
    sc = SCENARIOS[cfg["scenario"]]
    assert_disjoint(PANELS_HEM, HIDDEN_HEM)
    rows, long_rows = [], []
    for seed in cfg["seeds"]:
        trial, _ = _trial_artifact(cfg, cfg["scenario"], sc, seed, PANELS_HEM)
        truth = trial["truth"]
        for mm_level in cfg["mismatch"]:
            twin, prec = _need_params(trial, cfg["scenario"], sc, seed,
                                      mm_level, cfg)
            dec = (decoupled_modules(base=twin)
                   if any(m.startswith("decoupled") for m in cfg["methods"])
                   else None)
            for meth in cfg["methods"]:
                if meth == "enkf" and \
                        mm_level not in cfg.get("enkf_mismatch", []):
                    continue
                mods = twin
                meth_base = meth
                if meth.startswith("decoupled_"):
                    mods = dec
                    meth_base = meth[len("decoupled_"):]
                t0 = time.time()
                run_id = (f"e3/{cfg['scenario']}/seed{seed}/mm{mm_level}/"
                          f"{meth}")
                run_path = Path(out) / "raw" / "runs" / f"{run_id}.npz"
                # --resume: reuse the stored artifact for this run_id instead
                # of recomputing. Never overwrites — EnKF still reruns because
                # its diagnostics (analyses/coverage) are not recomputable
                # from a saved trajectory alone.
                if cfg.get("resume") and meth != "enkf" \
                        and meth != "persistence" and run_path.exists():
                    z = np.load(run_path, allow_pickle=True)
                    res = {"time": z["times"].tolist(),
                           "traces": {k[len("state::"):]: z[k]
                                      for k in z.files
                                      if k.startswith("state::")}}
                    per = score_states(truth, res,
                                       PANELS_HEM["full"] + HIDDEN_HEM,
                                       cfg["cut"], sc["horizon"])
                    meta = {}
                    res_loaded = True
                else:
                    res_loaded = False
                    if meth == "enkf":
                        _need_enkf(trial, cfg)
                    if meth == "persistence":
                        held, meta = METHODS[meth_base](trial, "full", cfg["cut"],
                                                 sc["horizon"], mods, cfg)
                        per = score_persistence(truth, held, cfg["cut"],
                                                sc["horizon"])
                        res = None
                    else:
                        res, meta = METHODS[meth_base](trial, "full", cfg["cut"],
                                                  sc["horizon"], mods, cfg)
                        per = score_states(truth, res,
                                           PANELS_HEM["full"] + HIDDEN_HEM,
                                           cfg["cut"], sc["horizon"])
                rt = time.time() - t0
                extra = None
                if meth == "enkf":
                    gtimes = np.asarray(meta["times"], float)
                    ttimes = np.asarray(truth["time"], float)
                    fm = (gtimes > cfg["cut"]) & (gtimes <= sc["horizon"])
                    cov, wid = {}, {}
                    for j, m in enumerate(MODULE_IDS):
                        sd = np.asarray(meta["ensemble_sd"][j])[fm]
                        tv = np.interp(gtimes[fm], ttimes,
                                       truth["traces"]["body:" + m])
                        mu = np.asarray(meta["ensemble_mean"][j])[fm]
                        cov[m] = float(np.mean(np.abs(tv - mu) <= 1.96 * sd))
                        wid[m] = float(np.mean(2 * 1.96 * sd))  # full width
                    meta["coverage95"] = cov
                    meta["width95_full"] = wid
                    extra = {"ens_mean": meta["ensemble_mean"],
                             "ens_sd": meta["ensemble_sd"],
                             "analyses": np.frombuffer(json.dumps(
                                 meta["analyses"]).encode(), np.uint8)}
                if res is not None and not res_loaded:
                    save_run(unique_run_path(out, run_id), res, extra)
                for ep_name, ep in _eval_panels("full"):
                    panel = (PANELS_HEM["full"] if ep_name == "observed"
                             else HIDDEN_HEM)
                    s = summarize({m: per[m] for m in panel if m in per},
                                  panel, truth, sc["horizon"], cfg["cut"])
                    row = {"experiment": "e3", "trial": seed,
                           "method": meth, "mismatch": mm_level,
                           "seed": seed, "dur": trial["dur"],
                           "panel": ep_name, **s,
                           "runtime_s": round(rt, 3), "run_id": run_id,
                           **{k: v for k, v in meta.items()
                              if not isinstance(v, (list, dict,
                                                    np.ndarray))}}
                    if meth == "enkf":
                        row["coverage95"] = float(np.mean(
                            [meta["coverage95"][m] for m in panel]))
                        row["width95_full"] = float(np.mean(
                            [meta["width95_full"][m] for m in panel]))
                    rows.append(row)
                long_rows += _per_state_rows(
                    "e3", seed, meth, per, "all", cut=cfg["cut"],
                    mismatch=mm_level, dur=trial["dur"], run_id=run_id)
    write_csv(Path(out) / "results" / "robustness_metrics_v4.csv", rows)
    write_csv(Path(out) / "results" / "robustness_per_state_v4.csv",
              long_rows)


def run_e4(cfg, out):
    sc = SCENARIOS[cfg["scenario"]]
    assert_disjoint(PANELS_HEM, HIDDEN_HEM)
    rows, long_rows = [], []
    for seed in cfg["seeds"]:
        trial, _ = _trial_artifact(cfg, cfg["scenario"], sc, seed, PANELS_HEM)
        truth = trial["truth"]
        for pname in cfg["panels"]:
            res, meta = METHODS["nudge"](trial, pname, cfg["cut"],
                                         sc["horizon"], None, cfg)
            per = score_states(truth, res,
                               sorted(set().union(*PANELS_HEM.values())) +
                               HIDDEN_HEM, cfg["cut"], sc["horizon"])
            run_id = f"e4/{cfg['scenario']}/seed{seed}/nudge_{pname}"
            save_run(unique_run_path(out, run_id), res)
            for ep_name, ep in [("observed_union",
                                 sorted(set().union(*PANELS_HEM.values()))),
                                ("hidden", HIDDEN_HEM)]:
                s = summarize({m: per[m] for m in ep if m in per}, ep,
                              truth, sc["horizon"], cfg["cut"])
                rows.append({"experiment": "e4", "trial": seed,
                             "method": f"nudge_{pname}", "obs_panel": pname,
                             "eval_panel": ep_name, **s,
                             "n_obs_states": len(PANELS_HEM[pname]),
                             "run_id": run_id,
                             **{k: v for k, v in meta.items()
                                if not isinstance(v, (list, dict))}})
            long_rows += _per_state_rows(
                "e4", seed, f"nudge_{pname}", per, "all",
                obs_panel=pname, run_id=run_id)
    write_csv(Path(out) / "results" / "observation_panel_metrics_v4.csv",
              rows)
    write_csv(Path(out) / "results" / "observation_panel_per_state_v4.csv",
              long_rows)


def run_e5(cfg, out):
    from backend.chemistry import resolve_compound
    rows, long_rows = [], []
    for drug, dcfg in DRUG_SCENARIOS.items():
        if drug not in cfg["drugs"]:
            continue
        assert_disjoint({"obs": dcfg["observed"]}, dcfg["hidden"])
        comp = asyncio.run(resolve_compound(drug))
        sc = dict(dcfg)
        sc["compounds"] = {drug: comp}
        panels = {"obs": dcfg["observed"]}
        for seed in cfg["seeds"]:
            trial, _ = _trial_artifact(cfg, drug, sc, seed, panels)
            truth = trial["truth"]
            for mm_level in cfg["mismatch"]:
                twin, prec = _need_params(trial, drug, sc, seed, mm_level,
                                          cfg)
                for drug_info in (True, False):
                    for assim in (True, False):
                        arm = (f"{'full' if drug_info else 'ablated'}_"
                               f"{'assim' if assim else 'open'}")
                        ev = trial["events"] if drug_info else []
                        t0 = time.time()
                        if assim:
                            obs = []
                            o = trial["obs"]["obs"]
                            for ti, t in enumerate(o["times"]):
                                if t >= cfg["cut"]:
                                    break
                                for m in o["values"]:
                                    obs.append({
                                        "time_min": float(t),
                                        "state": state_key(m),
                                        "value": o["values"][m][ti],
                                        "gain": cfg["gain"],
                                        "mode": "nudge"})
                            obs += [{"time_min": float(cfg["cut"]),
                                     "state": state_key(m), "value": 0.0,
                                     "gain": 0.0, "mode": "nudge"}
                                    for m in dcfg["observed"]]
                            from benchmark.da_core import run_sim
                            res = run_sim(ev, dcfg["horizon"],
                                          observations=obs, modules=twin,
                                          compounds={drug: comp})
                            meta = {"updates_applied":
                                    len(obs) - len(dcfg["observed"]),
                                    "n_release": len(dcfg["observed"])}
                        else:
                            from benchmark.da_core import run_sim
                            res = run_sim(ev, dcfg["horizon"], modules=twin,
                                          compounds={drug: comp})
                            meta = {"updates_applied": 0}
                        states = sorted(set(dcfg["observed"]) |
                                        set(dcfg["hidden"]))
                        per = score_states(truth, res, states, cfg["cut"],
                                           dcfg["horizon"])
                        run_id = (f"e5/{drug}/seed{seed}/mm{mm_level}/{arm}")
                        save_run(unique_run_path(out, run_id), res)
                        for ep_name, ep in [("observed", dcfg["observed"]),
                                            ("hidden", dcfg["hidden"])]:
                            s = summarize({m: per[m] for m in ep
                                           if m in per}, ep, truth,
                                          dcfg["horizon"], cfg["cut"])
                            rows.append({"experiment": "e5", "trial": seed,
                                         "method": arm, "drug": drug,
                                         "mismatch": mm_level,
                                         "eval_panel": ep_name, **s,
                                         "runtime_s": round(t0 := 0 or
                                                            time.time() - t0,
                                                            3),
                                         "run_id": run_id, **meta})
                        long_rows += _per_state_rows(
                            "e5", seed, arm, per, "all", drug=drug,
                            mismatch=mm_level, run_id=run_id)
    write_csv(Path(out) / "results" / "joint_drug_assimilation_v4.csv",
              rows)
    write_csv(Path(out) / "results" / "joint_drug_per_state_v4.csv",
              long_rows)


EXPS = {"e2": run_e2, "e3": run_e3, "e4": run_e4, "e5": run_e5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="reuse stored run_id artifacts instead of "
                         "recomputing them (never overwrites)")
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    if args.resume:
        cfg["resume"] = True
    out = cfg["out_dir"]
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "results").mkdir(exist_ok=True)
    (Path(out) / "configs").mkdir(exist_ok=True)
    cfg["_config_sha256_12"] = cfg_hash(cfg)
    json.dump(cfg, open(Path(out) / "configs" /
                        f"{args.exp}_used.json", "w"), indent=2)
    t0 = time.time()
    EXPS[args.exp](cfg, out)
    print(f"[{args.exp}] done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
