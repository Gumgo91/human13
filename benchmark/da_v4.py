"""v4 data-assimilation harness — shared-stream, immutable-trial design.

Fixes vs v3 (each defect reproduced by a failing test first):
- one truth + one observation stream + one parameter realisation per trial,
  stored immutably; every method consumes the SAME stored values
- primary protocol: ALL methods use observations strictly t < cut
- observed/hidden panels are disjoint by construction (asserted)
- two RMSE definitions kept distinct: mean_state_rmse (v3 'pooled') and
  pooled_rmse = sqrt(mean of per-state MSE); native-unit states are scored
  in original units plus a declared-scale normalisation, never raw-pooled
- EnKF: prior covariance via declared initial-condition spread, signed
  states never clipped, shared observations + stochastic member
  perturbation; diagnostics (gain norm, innovation, spread) stored
- mismatch is two-sided U(1-d,1+d); blind/paired methods share the twin
- unique run_id artifacts; overwriting an existing run file raises

Usage:  python benchmark/da_v4.py --exp e2 --config <cfg.json>
"""
import argparse
import ast
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import backend.body_network as bn  # noqa: E402
from backend.body_library import MODULES  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402
from benchmark.da_core import run_sim, state_key  # noqa: E402

MODULE_IDS = [m["id"] for m in MODULES]
MODULE_SET = set(MODULE_IDS)
N_STATES = len(MODULE_IDS)

RMSE_FLOOR = 0.005
ACTIVE_PEAK = 0.02

# declared normalisation scales (nominal baselines, fixed before results)
STATE_SCALES = {"hr": 72.0, "svr": 17.0, "sv": 70.0, "map": 88.68,
                "temperature": 37.0}
# declared observation noise sd per state class
NOISE_SD = {"native_hr": 2.0, "native_svr": 0.5, "native_sv": 2.0,
            "body": 0.03}


def noise_sd_for(state):
    if state in STATE_SCALES:
        return NOISE_SD.get("native_" + state, 0.03)
    return NOISE_SD["body"]


def scale_for(state):
    return STATE_SCALES.get(state, 1.0)


# --------------------------------------------------------------------------
# AST expression transforms (P0-5)
# --------------------------------------------------------------------------
def _terms(node):
    """Flatten an additive expression into (sign, node) terms."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _terms(node.left)
        right = [(not s if isinstance(node.op, ast.Sub) else s, n)
                 for s, n in _terms(node.right)]
        return left + right
    return [(True, node)]


def _build(terms):
    node = None
    for sign, n in terms:
        if node is None:
            node = n if sign else ast.UnaryOp(ast.USub(), n)
        else:
            node = ast.BinOp(node, ast.Add() if sign else ast.Sub(), n)
    return node if node is not None else ast.Constant(0.0)


class _Perturber(ast.NodeTransformer):
    """Multiply every numeric coefficient of a symbolic term by a fresh
    draw from factor_fn(). Bare symbolic terms get factor*term. Pure
    additive constants and max()/min() bounds are preserved."""
    def __init__(self, factor_fn):
        self.factor_fn = factor_fn
        self.factors = []

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and \
                    not isinstance(node.right, ast.Constant):
                f = self.factor_fn()
                self.factors.append(f)
                node.left = ast.Constant(node.left.value * f)
            elif isinstance(node.right, ast.Constant) and \
                    not isinstance(node.left, ast.Constant):
                f = self.factor_fn()
                self.factors.append(f)
                node.right = ast.Constant(node.right.value * f)
        return node


def perturb_expr(expr, factor_fn):
    """Scale every coefficient in expr; implicit-1 symbolic additive terms
    gain an explicit perturbed coefficient. Pure constants untouched."""
    tree = ast.parse(expr, mode="eval").body
    terms = _terms(tree)
    new_terms = []
    p = _Perturber(factor_fn)
    for sign, node in terms:
        node2 = p.visit(node)
        # bare symbolic term (no numeric coefficient): wrap with factor
        if not _has_scaled_literal(node2):
            f = factor_fn()
            p.factors.append(f)
            node2 = ast.BinOp(ast.Constant(f), ast.Mult(), node2)
        new_terms.append((sign, node2))
    out = ast.unparse(_build(new_terms))
    return out, p.factors


def _has_scaled_literal(node):
    """True if the term already carries a Constant multiplied in."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return isinstance(node.left, ast.Constant) or \
            isinstance(node.right, ast.Constant)
    return False


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def decouple_expr(expr, module_set):
    """Remove every additive term containing a module symbol, preserving
    constants, channels and primitive terms — inside max()/min() too."""
    def filt(node):
        if isinstance(node, ast.Call):
            args = [filt(a) for a in node.args]
            node.args = [a if a is not None else ast.Constant(0.0)
                         for a in args]
            return node
        if isinstance(node, ast.BinOp) and isinstance(node.op,
                                                    (ast.Add, ast.Sub)):
            kept = []
            for sign, t in _terms(node):
                t2 = filt(t)
                if t2 is not None and not (_names(t2) & module_set):
                    kept.append((sign, t2))
            return _build(kept)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            l, r = filt(node.left), filt(node.right)
            if l is None or r is None:
                return None
            return ast.BinOp(l, node.op, r)
        if isinstance(node, ast.UnaryOp):
            o = filt(node.operand)
            return ast.UnaryOp(node.op, o) if o is not None else None
        if isinstance(node, ast.Name) and node.id in module_set:
            return None
        return node
    tree = ast.parse(expr, mode="eval").body
    out = filt(tree)
    return ast.unparse(out if out is not None else ast.Constant(0.0))


def perturbed_modules(rng, level, two_sided=True):
    """Twin model: factor ~ U(1-d,1+d) (two-sided, default) or U(1,1+d)
    legacy increase-only. level=0 returns a copy identical to nominal.
    Sign and positivity are preserved by construction (|f-1|<1)."""
    mods, rec = [], []
    for m in MODULES:
        mm = dict(m)
        tau_f = 1 + (rng.random() * 2 - 1) * level if two_sided \
            else 1 + level * rng.random()
        assert tau_f > 0, "tau factor must stay positive"
        mm["tau_min"] = m["tau_min"] * tau_f
        expr, cf = perturb_expr(m["target"], lambda: (
            1 + (rng.random() * 2 - 1) * level if two_sided
            else 1 + level * rng.random()))
        mm["target"] = expr
        mods.append(mm)
        rec.append({"id": m["id"], "tau_factor": tau_f,
                    "coef_factors": cf, "parsed": True,
                    "expr_after": expr})
    return mods, rec


def decoupled_modules(base=None):
    """Decoupled twin built on `base` (same parameter realisation — the
    perturbed modules when paired with a mismatch run, nominal otherwise).
    Every expression must process; a parse failure raises, never '0'."""
    base = base if base is not None else MODULES
    mods, failures = [], []
    for m in base:
        mm = dict(m)
        try:
            mm["target"] = decouple_expr(m["target"], MODULE_SET)
        except Exception as e:  # explicit failure, not silent 0
            failures.append((m["id"], str(e)))
        mods.append(mm)
    if failures:
        raise ValueError(f"decoupling failed on {failures}")
    return mods


# --------------------------------------------------------------------------
# trial artifacts: one truth + obs + params per (scenario, seed), immutable
# --------------------------------------------------------------------------
def _seed_stream(seed, tag):
    """Deterministic per-(seed, purpose) RNG stream.

    `hash(tag)` is NOT usable here: Python string hashing is randomized per
    process (PYTHONHASHSEED), so trials could never be reproduced across
    invocations. Use a stable digest of the tag instead."""
    tag_int = int.from_bytes(
        hashlib.sha256(tag.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, tag_int]))


def gen_trial(scenario, sc, seed, panels, cfg):
    """Generate the immutable trial: truth trajectory (ALL states incl
    natives), per-panel shared observations, parameter realisations."""
    rng_t = _seed_stream(seed, "truth")
    rng_o = _seed_stream(seed, "obs")
    rng_p = _seed_stream(seed, "params")
    rng_e = _seed_stream(seed, "enkf")
    dur = (float(rng_t.uniform(*sc["dur"])) if isinstance(sc.get("dur"), tuple)
           else float(sc.get("dur", 120)))
    events = ([{"channels": sc["channels"], "duration_min": dur}]
              if "channels" in sc else
              [{"chemical": scenario, "dose_mg": sc["dose_mg"],
                "duration_min": sc.get("drug_dur", 120)}])
    truth = run_sim(events, sc["horizon"], compounds=sc.get("compounds"))
    times = np.asarray(truth["time"], float)
    obs_times = [t for t in np.arange(cfg["obs_every"], sc["horizon"] + 1,
                                     cfg["obs_every"])]
    obs = {}
    for pname, panel in panels.items():
        vals = {}
        for m in panel:
            key = state_key(m)
            if key not in truth["traces"]:
                raise KeyError(f"panel state {key} missing from truth")
            sd = noise_sd_for(m)
            vals[m] = [float(truth["traces"][key][
                int(np.searchsorted(times, t))] +
                rng_o.normal(0, sd)) for t in obs_times]
        obs[pname] = {"times": list(map(float, obs_times)), "values": vals,
                      "noise_sd": {m: noise_sd_for(m) for m in panel}}
    params = {}
    for mm_level in cfg["mismatch"]:
        if mm_level > 0:
            mods, rec = perturbed_modules(rng_p, mm_level)
            params[str(mm_level)] = {"modules": mods, "record": rec}
    enkf_members = []
    for _ in range(cfg.get("enkf_members", 16)):
        mods, rec = perturbed_modules(rng_e, cfg.get("enkf_spread", 0.15))
        enkf_members.append({"modules": mods, "record": rec})
    enkf_ic = rng_e.normal(0, cfg.get("enkf_ic_sd", 0.02),
                           size=(cfg.get("enkf_members", 16), N_STATES))
    return {"scenario": scenario, "seed": seed, "dur": dur, "events": events,
            "truth": truth, "obs": obs, "params": params,
            "enkf_members": enkf_members, "enkf_ic": enkf_ic,
            "seeds": {"seed": seed, "streams":
                      {"truth": "seed+'truth'", "obs": "seed+'obs'",
                       "params": "seed+'params'", "enkf": "seed+'enkf'"}},
            "obs_times": obs_times}


def save_trial(trial, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable trial exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=trial["truth"]["time"],
        **{f"state::{k}": np.asarray(v)
           for k, v in trial["truth"]["traces"].items()},
        **{f"obs::{p}::{m}": np.asarray(v["values"][m])
           for p, v in trial["obs"].items() for m in v["values"]},
        obs_times=np.asarray(trial["obs_times"], float),
        meta=np.frombuffer(json.dumps({
            "scenario": trial["scenario"], "seed": trial["seed"],
            "dur": trial["dur"], "events": trial["events"],
            "seeds": trial["seeds"],
            "params": {k: v["record"] for k, v in trial["params"].items()},
            "enkf_members": [m["record"] for m in trial["enkf_members"]],
            "enkf_ic": trial["enkf_ic"].tolist()}).encode(), np.uint8))
    return str(path)


def unique_run_path(root, run_id):
    path = Path(root) / "raw" / "runs" / f"{run_id}.npz"
    if path.exists():
        raise FileExistsError(f"run_id collision: {run_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# methods — all consume the SAME stored observation stream (t < cut)
# --------------------------------------------------------------------------
def _obs_updates(trial, panel, cut, mode, gain):
    """Stored shared observations strictly before cut -> update dicts."""
    obs = trial["obs"][panel]
    out = []
    for t, ti in zip(obs["times"], range(len(obs["times"]))):
        if t >= cut:
            break
        for m in obs["values"]:
            out.append({"time_min": float(t), "state": state_key(m),
                        "value": obs["values"][m][ti], "gain": gain,
                        "mode": mode})
    return out


def _last_obs(trial, panel, cut):
    obs = trial["obs"][panel]
    idx = [i for i, t in enumerate(obs["times"]) if t < cut]
    if not idx:
        return {}
    i = idx[-1]
    return {m: obs["values"][m][i] for m in obs["values"]}


def release_updates(panel, t):
    return [{"time_min": float(t), "state": state_key(m), "value": 0.0,
             "gain": 0.0, "mode": "nudge"} for m in panel]


def method_blind(trial, panel, cut, horizon, modules, cfg):
    res = run_sim([], horizon, modules=modules)
    return res, {"updates_applied": 0}


def method_nudge(trial, panel, cut, horizon, modules, cfg):
    obs = _obs_updates(trial, panel, cut, "nudge", cfg["gain"])
    rel = release_updates(trial["obs"][panel]["values"].keys(), cut)
    res = run_sim([], horizon, observations=obs + rel, modules=modules)
    return res, {"updates_applied": len(obs), "n_release": len(rel),
                 "update_times": [o["time_min"] for o in obs]}


def method_jump(trial, panel, cut, horizon, modules, cfg):
    """Initialization baseline: the single most recent shared observation
    (t < cut) applied as one hard set at t=cut, then free evolution."""
    last = _last_obs(trial, panel, cut)
    upd = [{"time_min": float(cut), "state": state_key(m), "value": v,
            "gain": 1.0, "mode": "set"} for m, v in last.items()]
    res = run_sim([], horizon, observations=upd, modules=modules)
    return res, {"updates_applied": len(upd), "jump_time": cut,
                 "source_obs_time": max(t for t in trial["obs"][panel]["times"]
                                        if t < cut) if last else None}


def method_persistence(trial, panel, cut, horizon, modules, cfg):
    """Last shared observation (t < cut) held constant for t > cut.
    Latent estimate NA by definition."""
    last = _last_obs(trial, panel, cut)
    return last, {"held_from": max(t for t in trial["obs"][panel]["times"]
                                   if t < cut) if last else None}


def method_enkf(trial, panel, cut, horizon, modules, cfg):
    """Stochastic EnKF. Members: pre-stored perturbed-module realisations +
    initial-condition spread (declared ic_sd) so prior covariance is
    nonzero at the first analysis. Observations are the SHARED stream;
    each member adds its own N(0, R) perturbation. Signed dimensionless
    states are never clipped. Diagnostics recorded per analysis."""
    rng_e = _seed_stream(trial["seed"], "enkf_pert")
    members = trial["enkf_members"]
    n = len(members)
    obs = trial["obs"][panel]
    obs_idx = [MODULE_IDS.index(m) for m in obs["values"]]
    obs_times = [t for t in obs["times"] if t < cut]
    updates = [[] for _ in range(n)]
    for i in range(n):  # declared initial-condition spread at t=0
        for j, m in enumerate(MODULE_IDS):
            updates[i].append({"time_min": 0.0, "state": "body:" + m,
                               "value": float(trial["enkf_ic"][i, j]),
                               "gain": 1.0, "mode": "set"})
    analyses = []
    for t in obs_times:
        ti = obs["times"].index(t)
        X = np.zeros((N_STATES, n))
        for i, mm in enumerate(members):
            res = run_sim([], t, observations=updates[i],
                          modules=mm["modules"])
            X[:, i] = np.array([res["traces"]["body:" + m][-1]
                                for m in MODULE_IDS])
        Y = X[obs_idx, :]
        mx = X.mean(1, keepdims=True)
        my = Y.mean(1, keepdims=True)
        Ax, Ay = X - mx, Y - my
        Pxy = Ax @ Ay.T / (n - 1)
        Pyy = Ay @ Ay.T / (n - 1)
        R = np.diag([noise_sd_for(m) ** 2 for m in obs["values"]])
        K = Pxy @ np.linalg.inv(Pyy + R)
        z = np.array([obs["values"][m][ti] for m in obs["values"]])
        D = z[:, None] + np.column_stack(
            [rng_e.normal(0, [noise_sd_for(m) for m in obs["values"]])
             for _ in range(n)])
        Xa = X + K @ (D - Y)   # signed states unclipped (P0-3)
        analyses.append({
            "time": t, "gain_norm": float(np.linalg.norm(K)),
            "innovation": (z - my.ravel()).tolist(),
            "prior_mean": mx.ravel().tolist(),
            "prior_sd": Ax.std(1).tolist(),
            "post_mean": Xa.mean(1).tolist(),
            "post_sd": Xa.std(1).tolist(),
            "members": Xa.tolist()})
        for i in range(n):
            for j, m in enumerate(MODULE_IDS):
                updates[i].append({"time_min": float(t), "state": "body:" + m,
                                   "value": float(Xa[j, i]), "gain": 1.0,
                                   "mode": "set"})
    grid = np.linspace(0, horizon, 481)
    ens = []
    for i, mm in enumerate(members):
        res = run_sim([], horizon, observations=updates[i],
                      modules=mm["modules"])
        tt = np.asarray(res["time"], float)
        ens.append(np.array([np.interp(grid, tt, res["traces"]["body:" + m])
                             for m in MODULE_IDS]))
    ens = np.stack(ens)
    res = {"time": grid.tolist(),
           "traces": {"body:" + m: ens.mean(0)[j]
                      for j, m in enumerate(MODULE_IDS)}}
    meta = {"updates_applied": len(obs_times) * N_STATES,
            "n_members": n, "analyses": analyses,
            "ensemble_mean": ens.mean(0), "ensemble_sd": ens.std(0),
            "times": grid}
    return res, meta


METHODS = {"blind": method_blind, "nudge": method_nudge,
           "jump": method_jump, "persistence": method_persistence,
           "enkf": method_enkf}


# --------------------------------------------------------------------------
# scoring — unified state-key resolver, explicit missing, two RMSE defs
# --------------------------------------------------------------------------
def _interp(yt, y, x):
    return np.interp(x, np.asarray(yt, float), np.asarray(y, float))


def score_states(truth, test, states, t0, t1):
    """Per-state errors over (t0, t1]: raw RMSE in the state's own units +
    normalised RMSE using the declared scale. Missing states raise."""
    times = np.asarray(truth["time"], float)
    mask = (times > t0) & (times <= t1)
    if not mask.any():
        raise ValueError(f"empty evaluation window ({t0},{t1}]")
    out, missing = {}, []
    t_times = np.asarray(test["time"], float)
    for m in states:
        key = state_key(m)
        if key not in truth["traces"] or key not in test["traces"]:
            missing.append(key)
            continue
        e = np.asarray(truth["traces"][key])[mask] - _interp(
            t_times, test["traces"][key], times[mask])
        mse = float((e ** 2).mean())
        out[m] = {"rmse": float(np.sqrt(mse)), "mse": mse,
                  "rmse_norm": float(np.sqrt(mse) / scale_for(m)),
                  "unit": ("dimensionless" if m not in STATE_SCALES else m)}
    if missing:
        raise KeyError(f"states missing from traces: {missing}")
    return out


def score_persistence(truth, held, t0, t1):
    times = np.asarray(truth["time"], float)
    mask = (times > t0) & (times <= t1)
    out = {}
    for m, v in held.items():
        key = state_key(m)
        if key not in truth["traces"]:
            raise KeyError(f"held state {key} missing from truth")
        e = np.asarray(truth["traces"][key])[mask] - v
        mse = float((e ** 2).mean())
        out[m] = {"rmse": float(np.sqrt(mse)), "mse": mse,
                  "rmse_norm": float(np.sqrt(mse) / scale_for(m)),
                  "unit": ("dimensionless" if m not in STATE_SCALES else m)}
    return out


def summarize(per_state, panel, truth, horizon, cut):
    """Both RMSE definitions over the dimensionless subset + separate
    native-unit states. pooled_rmse = sqrt(mean mse); mean_state_rmse =
    mean of per-state rmse (the v3 'pooled')."""
    dim = [m for m in panel
           if m in per_state and per_state[m]["unit"] == "dimensionless"]
    nat = [m for m in panel
           if m in per_state and per_state[m]["unit"] != "dimensionless"]
    mses = [per_state[m]["mse"] for m in dim]
    times = np.asarray(truth["time"], float)
    mask = (times > cut) & (times <= horizon)
    active = [m for m in dim
              if np.abs(np.asarray(truth["traces"][state_key(m)])[mask]).max()
              >= ACTIVE_PEAK]
    norm = [per_state[m]["rmse_norm"] for m in panel if m in per_state]
    return {"pooled_rmse": float(np.sqrt(np.mean(mses))) if mses else None,
            "mean_state_rmse": float(np.mean([per_state[m]["rmse"]
                                              for m in dim])) if dim else None,
            "n_states": len(dim), "n_native": len(nat),
            "norm_rmse": float(np.mean(norm)) if norm else None,
            "n_active": len(active),
            "active_rmse": float(np.mean([per_state[m]["rmse"]
                                          for m in active])) if active else None,
            "native_per_state": {m: per_state[m] for m in nat}}


def assert_disjoint(panels, hidden):
    union = set().union(*map(set, panels.values()))
    overlap = union & set(hidden)
    assert not overlap, f"observed/hidden overlap: {sorted(overlap)}"
    return True


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_run(path, res, extra=None):
    d = {"times": np.asarray(res["time"], float),
         **{f"state::{k}": np.asarray(v)
            for k, v in res["traces"].items()}}
    if extra:
        d.update(extra)
    np.savez_compressed(path, **d)
    return str(path)


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True,
                                     default=str).encode()).hexdigest()[:12]
