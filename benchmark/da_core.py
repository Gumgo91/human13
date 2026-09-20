"""Shared machinery for the v3 data-assimilation validation experiments.

Model variants (parameter mismatch, coupling removal), observation
generation, the EnKF comparison method, and panel/interval scoring.

All evaluation choices are fixed a priori:
- forecast score uses only t > t_cut; reconstruction (t <= t_cut) is separate
- nudge observations strictly before t_cut (a zero-length trailing update
  contributes nothing and is excluded); jump/EnKF analyses may use the
  observation at t_cut itself (instantaneous update, logged separately)
- nudges release exactly at t_cut, not cut+epsilon
- persistence = last available observation held constant; it produces no
  latent estimate (NA), distinct from baseline-at-homeostasis
- percent improvement only where baseline RMSE >= RMSE_FLOOR, else NA
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import backend.body_network as bn  # noqa: E402
from backend.body_library import MODULES  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402

MODULE_IDS = [m["id"] for m in MODULES]
MODULE_SET = set(MODULE_IDS)
N_STATES = len(MODULE_IDS)

RMSE_FLOOR = 0.005       # below this baseline error, report absolute diff, not %
ACTIVE_PEAK = 0.02       # pre-declared 'responding state' threshold (|truth peak|)


# --------------------------------------------------------------------------
# model variants
# --------------------------------------------------------------------------
_TERM_RE = re.compile(r"([+-]?)\s*(?:(\d+(?:\.\d+)?)\s*\*\s*)?([A-Za-z_][\w.:]*)")


def _split_terms(expr):
    """Parse 'c1*s1+c2*s2-s3' into [(sign, coef, symbol), ...]; None on residue."""
    terms, pos = [], 0
    for m in _TERM_RE.finditer(expr):
        if expr[pos:m.start()].strip() not in ("", "+"):
            return None
        sign, coef, sym = m.groups()
        terms.append([sign or "+", float(coef) if coef else 1.0, sym])
        pos = m.end()
    if expr[pos:].strip():
        return None
    return terms


def perturbed_modules(rng, level):
    """Twin model with positive multiplicative perturbation of every time
    constant and every connection coefficient: factor = 1 + U(0, level),
    preserving signs and positivity. Returns (modules, record)."""
    mods, rec = [], []
    for m in MODULES:
        mm = dict(m)
        tau_f = 1 + level * rng.random()
        mm["tau_min"] = m["tau_min"] * tau_f
        terms = _split_terms(m["target"])
        cf = []
        if terms is not None:
            parts = []
            for sign, coef, sym in terms:
                f = 1 + level * rng.random()
                cf.append(round(f, 5))
                parts.append(f"{sign}{coef * f:g}*{sym}")
            mm["target"] = "".join(parts)
        mods.append(mm)
        rec.append({"id": m["id"], "tau_factor": round(tau_f, 5),
                    "coef_factors": cf, "parsed": terms is not None})
    return mods, rec


def decoupled_modules():
    """Twin model with all module->module coupling removed: only direct
    stimulus/channel and primitive terms remain, so observation updates
    cannot propagate between modules."""
    mods = []
    for m in MODULES:
        mm = dict(m)
        terms = _split_terms(m["target"])
        if terms is not None:
            keep = [(s, c, y) for s, c, y in terms if y not in MODULE_SET]
            mm["target"] = ("".join(f"{s}{c:g}*{y}" if c != 1.0 else f"{s}{y}"
                                    for s, c, y in keep)) or "0"
        else:
            mm["target"] = "0"
        mods.append(mm)
    return mods


# --------------------------------------------------------------------------
# simulation driver
# --------------------------------------------------------------------------
def run_sim(events, horizon, observations=None, modules=None, compounds=None):
    """simulate() with an optional perturbed/decoupled module set.

    Event dicts: {"channels": [...], "duration_min": d} for channel events,
    {"chemical": name, "dose_mg": x} for drug events (compounds keyed by
    event index as simulate() expects)."""
    iv, comp = [], dict(compounds or {})
    for i, e in enumerate(events):
        if isinstance(e, Intervention):
            iv.append(e)
        elif "chemical" in e:
            iv.append(Intervention(kind="chemical", label=e["chemical"],
                                   entity=e["chemical"],
                                   quantity=e.get("dose_mg"), unit="mg",
                                   duration_min=e.get("duration_min", 120)))
            if e["chemical"] in comp:
                comp[str(i)] = comp[e["chemical"]]
        else:
            iv.append(Intervention(kind="other", label="event", entity=None,
                                   start_min=e.get("start_min", 0),
                                   duration_min=e.get("duration_min", 120),
                                   body_channels=e["channels"]))
    if modules is None:
        return simulate(Interpretation(title="da", interventions=iv),
                        RunSettings(horizon_min=horizon), comp,
                        observations=observations)
    orig = bn.MODULES
    bn.MODULES = modules
    try:
        return simulate(Interpretation(title="da", interventions=iv),
                        RunSettings(horizon_min=horizon), comp,
                        observations=observations)
    finally:
        bn.MODULES = orig


def module_vec(res, mids=MODULE_IDS):
    return np.array([res["traces"]["body:" + m][-1] for m in mids])


def state_key(m):
    """Module ids become 'body:<id>'; native state names pass through."""
    return "body:" + m if m in MODULE_SET else m


def gen_observations(truth, observed, obs_times, rng, noise_sd, mode="nudge",
                     gain=0.15):
    """Observation list from truth at given times + Gaussian noise."""
    times = np.asarray(truth["time"], float)
    out = []
    for t in obs_times:
        k = int(np.searchsorted(times, t))
        for m in observed:
            key = state_key(m)
            if key not in truth["traces"]:
                continue
            v = float(truth["traces"][key][k]) + rng.normal(0, noise_sd)
            out.append({"time_min": float(t), "state": key,
                        "value": v, "gain": gain, "mode": mode})
    return out


def release_observations(observed, t):
    return [{"time_min": float(t), "state": state_key(m), "value": 0.0,
             "gain": 0.0, "mode": "nudge"} for m in observed]


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def arm_blind(events, horizon, modules=None, compounds=None):
    return run_sim(events, horizon, modules=modules, compounds=compounds)


def arm_nudge(truth, observed, cut, horizon, rng, noise_sd,
              obs_every=30, gain=0.15, modules=None, compounds=None,
              events=None):
    """Continuous nudging: observations strictly < cut (zero-length trailing
    updates are excluded a priori), release exactly at cut."""
    obs_times = [t for t in np.arange(obs_every, horizon + 1, obs_every) if t < cut]
    obs = gen_observations(truth, observed, obs_times, rng, noise_sd,
                           mode="nudge", gain=gain)
    obs += release_observations(observed, cut)
    res = run_sim(events or [], horizon, observations=obs, modules=modules,
                  compounds=compounds)
    return res, {"updates_applied": len(obs), "obs_times": obs_times,
                 "release": cut}


def arm_jump(truth, observed, cut, horizon, rng, noise_sd, obs_every=30,
             modules=None, compounds=None, events=None):
    """Instantaneous state jump: a single hard set at t_cut to the most
    recent observation (t <= cut), then free evolution."""
    obs_times = [t for t in np.arange(obs_every, horizon + 1, obs_every) if t <= cut]
    obs = gen_observations(truth, observed, obs_times, rng, noise_sd,
                           mode="set", gain=1.0)
    last = [o for o in obs if o["time_min"] == obs_times[-1]]
    upd = [{"time_min": float(cut), "state": o["state"], "value": o["value"],
            "gain": 1.0, "mode": "set"} for o in last]
    res = run_sim(events or [], horizon, observations=upd, modules=modules,
                  compounds=compounds)
    return res, {"updates_applied": len(upd), "jump_time": cut}


def arm_persistence(truth, observed, cut, rng, noise_sd, obs_every=30):
    """Last available observation held constant for t > cut. No latent
    estimate exists (NA) — this is not baseline-at-homeostasis."""
    obs_times = [t for t in np.arange(obs_every, cut + 1, obs_every)]
    times = np.asarray(truth["time"], float)
    vals = {}
    t_last = obs_times[-1]
    k = int(np.searchsorted(times, t_last))
    for m in observed:
        vals[m] = float(truth["traces"]["body:" + m][k]) + rng.normal(0, noise_sd)
    return vals, {"held_from": t_last}


# --------------------------------------------------------------------------
# EnKF (stochastic ensemble Kalman filter, Evensen 1994/2003)
# Members differ by perturbed model parameters (model-error representation,
# same distribution as the mismatch condition); observation noise R is the
# known obs sd. No inflation, no localization (documented choices).
# --------------------------------------------------------------------------
def arm_enkf(truth, observed, cut, horizon, rng, noise_sd, n_members=16,
             obs_every=30, mismatch=0.1, events=None, compounds=None):
    member_mods, member_recs = [], []
    for i in range(n_members):
        mm, rec = perturbed_modules(rng, mismatch)
        member_mods.append(mm); member_recs.append(rec)
    obs_times = [t for t in np.arange(obs_every, horizon + 1, obs_every) if t <= cut]
    obs_idx = [MODULE_IDS.index(m) for m in observed]
    updates = [[] for _ in range(n_members)]
    analyses = []
    for t in obs_times:
        X = np.zeros((N_STATES, n_members))
        for i, mm in enumerate(member_mods):
            res = run_sim(events or [], t, observations=updates[i],
                          modules=mm, compounds=compounds)
            X[:, i] = module_vec(res)
        Y = X[obs_idx, :]
        Ax = X - X.mean(1, keepdims=True)
        Ay = Y - Y.mean(1, keepdims=True)
        Pxy = Ax @ Ay.T / (n_members - 1)
        Pyy = Ay @ Ay.T / (n_members - 1)
        K = Pxy @ np.linalg.inv(Pyy + noise_sd ** 2 * np.eye(len(observed)))
        times_t = np.asarray(truth["time"], float)
        kt = int(np.searchsorted(times_t, t))
        z = np.array([truth["traces"][state_key(m)][kt] for m in observed])
        zpert = z[:, None] + rng.normal(0, noise_sd, size=Y.shape)
        Xa = np.clip(X + K @ (zpert - Y), 0.0, None)  # nonnegative projection
        analyses.append({"time": t, "kalman_gain_diag": np.diag(K).tolist()})
        for i in range(n_members):
            for j, m in enumerate(MODULE_IDS):
                updates[i].append({"time_min": float(t), "state": "body:" + m,
                                   "value": float(Xa[j, i]), "gain": 1.0,
                                   "mode": "set"})
    grid = np.linspace(0, horizon, 481)   # common evaluation grid
    traj, ens = [], []
    for i, mm in enumerate(member_mods):
        res = run_sim(events or [], horizon, observations=updates[i],
                      modules=mm, compounds=compounds)
        traj.append(res)
        tt = np.asarray(res["time"], float)
        ens.append(np.array([np.interp(grid, tt, res["traces"]["body:" + m])
                             for m in MODULE_IDS]))
    ens = np.stack(ens)          # (n_members, n_states, len(grid))
    return traj, {"n_members": n_members, "mismatch": mismatch,
                  "analyses": analyses, "member_params": member_recs,
                  "ensemble_mean": ens.mean(0).tolist(),
                  "ensemble_sd": ens.std(0).tolist(),
                  "times": grid.tolist()}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _interp(y_times, y, x):
    return np.interp(x, np.asarray(y_times, float), np.asarray(y, float))


def score_states(truth, test, mids, t0, t1):
    """Per-state RMSE of `test` vs `truth` over (t0, t1], test interpolated
    onto the truth grid (arms may carry different boundaries)."""
    times = np.asarray(truth["time"], float)
    mask = (times > t0) & (times <= t1)
    if not mask.any():
        return {}
    out = {}
    t_times = np.asarray(test["time"], float)
    for m in mids:
        key = state_key(m)
        if key not in truth["traces"] or key not in test["traces"]:
            continue
        e = np.asarray(truth["traces"][key])[mask] - _interp(
            t_times, test["traces"][key], times[mask])
        out[m] = float(np.sqrt((e ** 2).mean()))
    return out


def score_persistence(truth, held, t0, t1):
    times = np.asarray(truth["time"], float)
    mask = (times > t0) & (times <= t1)
    out = {}
    for m, v in held.items():
        e = np.asarray(truth["traces"]["body:" + m])[mask] - v
        out[m] = float(np.sqrt((e ** 2).mean()))
    return out


def summarize(per_state, panel, truth, horizon, cut):
    """Pooled RMSE + active-state subanalysis over a panel for the full
    forecast interval."""
    vals = [per_state[m] for m in panel if m in per_state]
    pooled = float(np.mean(vals)) if vals else None
    times = np.asarray(truth["time"], float)
    mask = (times > cut) & (times <= horizon)
    active = [m for m in panel
              if m in per_state and np.abs(
                  np.asarray(truth["traces"][state_key(m)])[mask]).max()
              >= ACTIVE_PEAK]
    act_vals = [per_state[m] for m in active]
    return {"pooled_rmse": pooled, "n_states": len(vals),
            "n_active": len(active),
            "active_rmse": float(np.mean(act_vals)) if act_vals else None,
            "active_states": active}
