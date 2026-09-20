"""PRCP external validation — Physiologic Response to Changes in Posture.

PhysioNet prcp/1.0.0 (Heldt & Mark, Comput Cardiol 2003; ODC-BY 1.0):
10 healthy subjects, ECG+ABP+tilt-angle at 250 Hz, ~55 min records with
slow tilt, rapid tilt, and stand-up maneuvers (two rounds each).

One-axis cardiovascular validation only — this is NOT clinical whole-body
or drug-pathway validation. Declared protocol (fixed before results):

- signal extraction: beat annotations (wqrs -> HR, wabp + ABP waveform ->
  SBP/DBP -> MAP = DBP + (SBP-DBP)/3), median-binned to a 1-min grid
- baseline: median of the first 5 record minutes (pre-intervention data,
  applied identically to every method); all comparisons on DEVIATIONS
  (dHR, dMAP) because model baselines (72 bpm, 88.68 mmHg) are fixed and
  subject baselines are not model inputs
- event mapping: each episode window -> one `orthostasis` channel event
  (scale 1 = declared unit stimulus; the channel has no quantity
  normalization). Episode windows come from the record's own .anI
  annotations; no universal sampling rate is assumed.
- output calibration: the reduced-form model gain is declared
  uncalibrated, so a SINGLE affine gain g per output channel
  (dHR, dMAP) is fit by least squares on the 9 TRAINING subjects of each
  LOSO fold and applied to the held-out subject's model forecast.
  Raw (g=1) output is reported alongside as its own arm.
- forecast protocol per test subject: assimilate shared observations
  strictly t < cut (cut = start of the final episode block), forecast
  t >= cut. Methods: blind_raw / blind_cal / nudge_cal / persistence /
  naive(=0). Observation channel is hr only (map is a derived quantity,
  not a state); dMAP is therefore an unconstrained downstream forecast.
- no test-subject data beyond the pre-cut observation stream is used for
  fitting; gains and cut are fixed before metrics are computed.

Outputs: per-subject trajectories npz + per-episode and per-subject
metrics CSV + calibration record. Raw artifacts are sufficient to
regenerate every table.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from backend.contracts import Interpretation, Intervention, RunSettings  # noqa: E402
from backend.simulation import simulate  # noqa: E402
from benchmark.da_v4 import write_csv, unique_run_path  # noqa: E402

PRCP_DIR = ROOT / "human13_ncs_validation_v4" / "raw" / "prcp"
OUT_DIR = ROOT / "human13_ncs_validation_v4"

# Annotation vocabulary differs per record (Initiate/Conclude vs
# Initiation/Conclusion vs Start/End, rapid vs fast tilt, Stand up vs
# Stand-up, misspellings like 'Conslusion'). Classification uses keyword
# rules on the normalized note, never an exact string.
def _cls(note):
    """Classify an .anI note -> (maneuver, mark) where mark is
    up/down_start/down_end/start/end. Handles per-record vocabulary,
    misspellings ('Conslusion'), truncations ('supin', 'dow'), and the
    'Initiation of <tilt> down' boundary markers."""
    n = "".join(ch if ch.isalnum() else " " for ch in note.lower())
    n = " ".join(n.split())
    start = any(w in n.split() for w in
                ("initiate", "initiation", "start", "begin"))
    end = any(w in n for w in
              ("conclude", "conclusion", "conslusion", "end"))
    up = " up" in f" {n}" or n.endswith("up")
    down = ("down" in n or "dow" in n or "back" in n)
    for m, kind in (("slow_tilt", "slow"), ("rapid_tilt", None)):
        fast = kind is None and ("rapid" in n or "fast" in n)
        if "tilt" in n and (kind in n if kind else fast):
            if up and start:
                return (m, "up")
            if down and start:
                return (m, "down_start")
            if down and end:
                return (m, "down_end")
            if up and end:
                return (m, "up_end")
    if "stand" in n and up and not any(
            w in n for w in ("back", "transition", "return", "down")):
        return ("stand", "start")
    if ("back" in n or "transition" in n or "return" in n) and "supin" in n:
        return ("stand", "end")
    return (None, None)


MANEUVERS = ["slow_tilt", "rapid_tilt", "stand"]
DOWN_FALLBACK_S = {"slow_tilt": 15.0, "rapid_tilt": 5.0}
BIN_MIN = 1.0
BASE_MIN = 5.0
NUDGE_GAIN = 0.3


def _import_wfdb():
    try:
        import wfdb
    except ImportError as e:
        raise SystemExit("wfdb package required: pip install wfdb") from e
    return wfdb


def extract_subject(record):
    """Header + annotations + beat-derived 1-min HR/MAP series."""
    wfdb = _import_wfdb()
    hdr = wfdb.rdheader(str(record))
    fs = float(hdr.fs)
    wq = wfdb.rdann(str(record), "wqrs")
    wa = wfdb.rdann(str(record), "wabp")
    ann = wfdb.rdann(str(record), "anI")
    abp = wfdb.rdrecord(str(record), channels=[
        hdr.sig_name.index("ABP")]).p_signal[:, 0]

    hr_t = wq.sample[1:] / fs
    hr_v = 60.0 / (np.diff(wq.sample) / fs)
    beats = wa.sample
    bt, sbp, dbp = [], [], []
    for i in range(len(beats) - 1):
        seg = abp[beats[i]:beats[i + 1]]
        if len(seg) > 3:
            bt.append(beats[i] / fs)
            sbp.append(float(seg.max()))
            dbp.append(float(seg.min()))
    bt = np.asarray(bt)
    map_v = np.asarray(dbp) + (np.asarray(sbp) - np.asarray(dbp)) / 3.0

    dur_min = hdr.sig_len / fs / 60.0
    edges = np.arange(0, dur_min + BIN_MIN, BIN_MIN)
    grid = (edges[:-1] + edges[1:]) / 2.0

    def binned(st, sv):
        out = np.full(len(grid), np.nan)
        idx = np.digitize(st / 60.0, edges) - 1
        for i in range(len(grid)):
            vals = sv[idx == i]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                out[i] = np.median(vals)
        return out

    notes = [str(n) for n in ann.aux_note]
    tagged = [(s / fs, _cls(n)) for s, n in zip(ann.sample, notes)]
    # deduplicate repeated markers of the same kind within 60 s (double
    # taps and 'failed attempt' re-annotations) keeping the first
    dedup = []
    for s, mk in sorted(tagged):
        if dedup and mk == dedup[-1][1] and s - dedup[-1][0] < 60:
            continue
        dedup.append((s, mk))
    events, dropped = [], []
    for name in ("slow_tilt", "rapid_tilt"):
        ups = [s for s, (m, k) in dedup if m == name and k == "up"]
        for i, s in enumerate(ups):
            next_up = ups[i + 1] if i + 1 < len(ups) else np.inf
            d_end = min([x for x, (m, k) in dedup
                         if m == name and k == "down_end" and s < x
                         and x < next_up], default=None)
            d_start = min([x for x, (m, k) in dedup
                           if m == name and k == "down_start" and s < x
                           and x < next_up], default=None)
            e = d_end if d_end is not None else (
                d_start + DOWN_FALLBACK_S[name]
                if d_start is not None else None)
            if e is not None and e > s:
                events.append({"type": name, "start_s": float(s),
                               "end_s": float(e)})
            else:
                dropped.append({"type": name, "start_s": float(s)})
    starts = [s for s, (m, k) in dedup if m == "stand" and k == "start"]
    ends = [s for s, (m, k) in dedup if m == "stand" and k == "end"]
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else np.inf
        e = min([x for x in ends if s < x < nxt], default=None)
        if e is not None:
            events.append({"type": "stand", "start_s": float(s),
                           "end_s": float(e)})
        else:
            dropped.append({"type": "stand", "start_s": float(s)})
    events.sort(key=lambda e: e["start_s"])
    return {"record": record.name, "fs": fs, "dur_min": dur_min,
            "sig_names": list(hdr.sig_name), "grid_min": grid,
            "hr": binned(hr_t, hr_v), "map": binned(bt, map_v),
            "events": events, "dropped": dropped,
            "n_beats": int(len(wq.sample))}


def baseline(sub):
    m = sub["grid_min"] < BASE_MIN
    return (float(np.nanmedian(sub["hr"][m])),
            float(np.nanmedian(sub["map"][m])))


def model_events(sub):
    return [{"channels": ["orthostasis"],
             "start_min": e["start_s"] / 60.0,
             "duration_min": (e["end_s"] - e["start_s"]) / 60.0}
            for e in sub["events"]]


def _sim(evs, horizon, observations=None):
    iv = [Intervention(kind="other", label="orthostatic challenge",
                       start_min=e["start_min"],
                       duration_min=e["duration_min"],
                       body_channels=e["channels"]) for e in evs]
    return simulate(Interpretation(title="prcp", interventions=iv),
                    RunSettings(horizon_min=horizon), {},
                    observations=observations)


def _series(res):
    t = np.asarray(res["time"], float)
    hr = np.asarray(res["traces"]["hr"], float) - 72.0
    mp = np.asarray(res["traces"]["map"], float) - (72 * 70 / 1000 * 17 + 3)
    return t, hr, mp


def fit_gains(train, sims):
    """Least-squares affine gain per channel on train subjects only.
    g = <data, model> / <model, model> over all post-baseline bins."""
    num = {"dhr": 0.0, "dmap": 0.0}
    den = {"dhr": 0.0, "dmap": 0.0}
    for sub in train:
        t_mod, dhr, dmp = sims[sub["record"]]
        base = baseline(sub)
        g = sub["grid_min"]
        m = g >= BASE_MIN
        mhr = np.interp(g[m], t_mod, dhr)
        mmp = np.interp(g[m], t_mod, dmp)
        dhr_d = sub["hr"][m] - base[0]
        dmp_d = sub["map"][m] - base[1]
        ok = np.isfinite(dhr_d)
        num["dhr"] += float((dhr_d[ok] * mhr[ok]).sum())
        den["dhr"] += float((mhr[ok] ** 2).sum())
        ok = np.isfinite(dmp_d)
        num["dmap"] += float((dmp_d[ok] * mmp[ok]).sum())
        den["dmap"] += float((mmp[ok] ** 2).sum())
    return {"dhr": num["dhr"] / den["dhr"] if den["dhr"] > 0 else 1.0,
            "dmap": num["dmap"] / den["dmap"] if den["dmap"] > 0 else 1.0}


def _forecast_rmse(sub, base, t_mod, dhr_mod, dmap_mod, cut_min):
    g = sub["grid_min"]
    mask = g >= cut_min
    e_hr = (sub["hr"][mask] - base[0]) - np.interp(
        g[mask], t_mod, dhr_mod)
    e_mp = (sub["map"][mask] - base[1]) - np.interp(
        g[mask], t_mod, dmap_mod)
    e_hr = e_hr[np.isfinite(e_hr)]
    e_mp = e_mp[np.isfinite(e_mp)]
    return {"dhr_rmse": float(np.sqrt((e_hr ** 2).mean())) if len(e_hr) else None,
            "dmap_rmse": float(np.sqrt((e_mp ** 2).mean())) if len(e_mp) else None,
            "n_bins": int(mask.sum())}


def run_prcp(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    subs = [extract_subject(PRCP_DIR / r.stem) for r in
            sorted(PRCP_DIR.glob("*.hea"))
            if (PRCP_DIR / (r.stem + ".dat")).exists()]
    assert len(subs) == 10, f"expected 10 PRCP records, found {len(subs)}"
    (out_dir / "raw" / "prcp_extracted").mkdir(parents=True, exist_ok=True)
    for s in subs:
        np.savez_compressed(
            out_dir / "raw" / "prcp_extracted" / f"{s['record']}.npz",
            grid_min=s["grid_min"], hr=s["hr"], map=s["map"],
            meta=np.frombuffer(json.dumps(
                {"record": s["record"], "fs": s["fs"],
                 "dur_min": s["dur_min"], "events": s["events"],
                 "dropped": s["dropped"],
                 "sig_names": s["sig_names"],
                 "n_beats": s["n_beats"]}).encode(), np.uint8))
    # one open-loop model run per subject is shared by every method arm —
    # the SAME event stream and the SAME truth baseline everywhere
    sims = {}
    for sub in subs:
        sims[sub["record"]] = _series(_sim(model_events(sub),
                                         sub["dur_min"]))
    rows, ep_rows, cal_rows = [], [], []
    for held in subs:
        train = [s for s in subs if s["record"] != held["record"]]
        gains = fit_gains(train, sims)
        base = baseline(held)
        cut = max(e["start_s"] for e in held["events"]) / 60.0
        horizon = held["dur_min"]
        g = held["grid_min"]
        obs_idx = [i for i, t in enumerate(g)
                   if BASE_MIN <= t < cut and np.isfinite(held["hr"][i])]
        obs = [{"time_min": float(g[i]), "state": "hr",
                "value": float(held["hr"][i] - base[0] + 72.0),
                "gain": NUDGE_GAIN, "mode": "nudge"} for i in obs_idx]

        runs = {}
        t_mod, dhr, dmp = sims[held["record"]]
        runs["blind_raw"] = ((t_mod, dhr, dmp), {"n_obs": 0, "gain": 1.0})
        runs["blind_cal"] = ((t_mod, gains["dhr"] * dhr,
                              gains["dmap"] * dmp),
                             {"n_obs": 0, "gain": dict(gains)})
        res = _sim(model_events(held), horizon, observations=obs)
        tn, hn, mn = _series(res)
        runs["nudge_cal"] = ((tn, gains["dhr"] * hn, gains["dmap"] * mn),
                             {"n_obs": len(obs), "gain": dict(gains)})
        # persistence: last pre-cut observed deviation held constant
        if obs_idx:
            li = obs_idx[-1]
            p_hr = float(held["hr"][li] - base[0])
            p_mp = float(held["map"][li] - base[1]) \
                if np.isfinite(held["map"][li]) else 0.0
        else:
            p_hr = p_mp = 0.0
        pt = np.array([0.0, horizon])
        runs["persistence"] = ((pt, np.array([p_hr, p_hr]),
                                np.array([p_mp, p_mp])),
                               {"n_obs": 1, "gain": 1.0})
        runs["naive_d0"] = ((pt, np.zeros(2), np.zeros(2)),
                            {"n_obs": 0, "gain": 1.0})

        cal_rows.append({"held_out": held["record"],
                         "gain_dhr": gains["dhr"],
                         "gain_dmap": gains["dmap"], "cut_min": cut,
                         "n_obs": len(obs)})
        for meth, ((tm, h, p), meta) in runs.items():
            sc = _forecast_rmse(held, base, tm, h, p, cut)
            run_id = f"prcp/{held['record']}/{meth}"
            rp = unique_run_path(out_dir, run_id)
            np.savez_compressed(
                rp, time=tm, dhr=h, dmap=p,
                meta=np.frombuffer(json.dumps(
                    {"subject": held["record"], "method": meth,
                     "cut_min": cut, "gains": gains,
                     "baseline": base, "n_obs": meta["n_obs"]}
                ).encode(), np.uint8))
            rows.append({"experiment": "prcp", "subject": held["record"],
                         "method": meth, "cut_min": round(cut, 3),
                         "baseline_hr": round(base[0], 2),
                         "baseline_map": round(base[1], 2),
                         "gain_dhr": round(gains["dhr"], 4),
                         "gain_dmap": round(gains["dmap"], 4),
                         "n_obs": meta["n_obs"], "run_id": run_id, **sc})
        for e in held["events"]:
            s0, s1 = e["start_s"] / 60, e["end_s"] / 60
            if s0 < cut:
                continue
            m = (g >= s0) & (g < s1 + 2)
            if not m.any():
                continue
            for meth, ((tm, h, p), _m2) in runs.items():
                eh = (held["hr"][m] - base[0]) - np.interp(g[m], tm, h)
                em = (held["map"][m] - base[1]) - np.interp(g[m], tm, p)
                eh = eh[np.isfinite(eh)]
                em = em[np.isfinite(em)]
                ep_rows.append({"subject": held["record"],
                                "episode": e["type"],
                                "start_min": round(s0, 3),
                                "method": meth,
                                "dhr_rmse": float(np.sqrt((eh**2).mean()))
                                if len(eh) else None,
                                "dmap_rmse": float(np.sqrt((em**2).mean()))
                                if len(em) else None,
                                "n_bins": int(m.sum())})
        got = {r["method"]: r["dhr_rmse"] for r in rows[-5:]}
        print(f"[prcp] {held['record']} g_hr={gains['dhr']:.2f} "
              f"g_map={gains['dmap']:.2f} " +
              " ".join(f"{k}={v:.2f}" for k, v in got.items()),
              flush=True)

    res_dir = out_dir / "results"
    res_dir.mkdir(exist_ok=True)
    write_csv(res_dir / "prcp_metrics_v4.csv", rows)
    write_csv(res_dir / "prcp_episode_metrics_v4.csv", ep_rows)
    write_csv(res_dir / "prcp_calibration_v4.csv", cal_rows)
    return rows, ep_rows, cal_rows


if __name__ == "__main__":
    t0 = time.time()
    run_prcp()
    print(f"done in {time.time()-t0:.0f}s")
