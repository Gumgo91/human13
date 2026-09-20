"""v4 summary figures from results/*.csv — PNG + PDF + source data."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "human13_ncs_validation_v4" / "results"
FIG = ROOT / "human13_ncs_validation_v4" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _rows(name):
    p = RES / name
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def _mean(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
    return float(np.mean(v)) if v else np.nan


def fig_e2():
    rows = [r for r in _rows("forecast_metrics_v4.csv")
            if r["experiment"] == "e2"]
    if not rows:
        return
    methods = ["blind", "nudge", "jump", "persistence"]
    cuts = sorted({float(r["cut"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    src = {}
    for ax, panel in zip(axes, ("observed", "hidden")):
        for m in methods:
            ys = [_mean([r for r in rows if r["method"] == m
                         and float(r["cut"]) == c and r["panel"] == panel],
                        "pooled_rmse") for c in cuts]
            src[f"e2_{panel}_{m}"] = ys
            ax.plot(cuts, ys, marker="o", label=m)
        ax.set_xlabel("cut (min)")
        ax.set_title(f"e2 {panel}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("pooled RMSE")
    fig.suptitle("E2 forecast vs observation cutoff (paired seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "e2_forecast_vs_cut.png", dpi=150)
    fig.savefig(FIG / "e2_forecast_vs_cut.pdf")
    json.dump({"cuts": cuts, "series": src},
              open(FIG / "e2_forecast_vs_cut.source.json", "w"), indent=1)
    plt.close(fig)


def fig_e3():
    rows = [r for r in _rows("robustness_metrics_v4.csv")
            if r["experiment"] == "e3"]
    if not rows:
        return
    methods = ["blind", "nudge", "jump", "persistence", "decoupled_nudge",
               "enkf"]
    mms = sorted({float(r["mismatch"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    src = {}
    for ax, panel in zip(axes, ("observed", "hidden")):
        for m in methods:
            ys = [_mean([r for r in rows if r["method"] == m
                         and float(r["mismatch"]) == mm
                         and r["panel"] == panel], "pooled_rmse")
                  for mm in mms]
            if all(np.isnan(ys)):
                continue
            src[f"e3_{panel}_{m}"] = ys
            ax.plot([str(x) for x in mms], ys, marker="o", label=m)
        ax.set_xlabel("two-sided mismatch d")
        ax.set_title(f"e3 {panel}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("pooled RMSE")
    fig.suptitle("E3 robustness under U(1-d,1+d) parameter mismatch")
    fig.tight_layout()
    fig.savefig(FIG / "e3_mismatch.png", dpi=150)
    fig.savefig(FIG / "e3_mismatch.pdf")
    json.dump({"mismatch": mms, "series": src},
              open(FIG / "e3_mismatch.source.json", "w"), indent=1)
    plt.close(fig)


def fig_e4():
    rows = [r for r in _rows("observation_panel_metrics_v4.csv")
            if r["experiment"] == "e4"]
    if not rows:
        return
    panels = ["full", "upstream", "downstream", "downstream4"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(panels))
    w = 0.38
    src = {}
    for k, ep in enumerate(("observed_union", "hidden")):
        ys = [_mean([r for r in rows if r["obs_panel"] == p
                     and r["eval_panel"] == ep], "mean_state_rmse")
              for p in panels]
        src[ep] = ys
        ax.bar(x + (k - .5) * w, ys, w, label=ep)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n({n})" for p, n in
                        zip(panels, [8, 4, 2, 4])])
    ax.set_ylabel("mean_state_rmse")
    ax.set_title("E4 observation-panel choice (equal-N comparison)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "e4_panels.png", dpi=150)
    fig.savefig(FIG / "e4_panels.pdf")
    json.dump({"panels": panels, "series": src},
              open(FIG / "e4_panels.source.json", "w"), indent=1)
    plt.close(fig)


def fig_e5():
    rows = [r for r in _rows("joint_drug_assimilation_v4.csv")
            if r["experiment"] == "e5"]
    if not rows:
        return
    arms = ["full_open", "full_assim", "ablated_open", "ablated_assim"]
    drugs = sorted({r["drug"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    src = {}
    for ax, drug in zip(axes, drugs):
        x = np.arange(len(arms))
        w = 0.38
        for k, ep in enumerate(("observed", "hidden")):
            ys = [_mean([r for r in rows if r["drug"] == drug
                         and r["method"] == a
                         and r["eval_panel"] == ep], "pooled_rmse")
                  for a in arms]
            src[f"{drug}_{ep}"] = ys
            ax.bar(x + (k - .5) * w, ys, w, label=ep)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
        ax.set_title(drug)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("pooled RMSE")
    fig.suptitle("E5 joint drug+physiology 2x2 (information x assimilation)")
    fig.tight_layout()
    fig.savefig(FIG / "e5_joint_2x2.png", dpi=150)
    fig.savefig(FIG / "e5_joint_2x2.pdf")
    json.dump({"arms": arms, "series": src},
              open(FIG / "e5_joint_2x2.source.json", "w"), indent=1)
    plt.close(fig)


def fig_prcp():
    rows = [r for r in _rows("prcp_metrics_v4.csv")]
    if not rows:
        return
    methods = ["blind_raw", "blind_cal", "nudge_cal", "persistence",
               "naive_d0"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    w = 0.38
    src = {}
    for k, metric in enumerate(("dhr_rmse", "dmap_rmse")):
        ys = [_mean([r for r in rows if r["method"] == m], metric)
              for m in methods]
        src[metric] = ys
        ax.bar(x + (k - .5) * w, ys, w, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("RMSE (native units)")
    ax.set_title("PRCP external validation — posture response forecasts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "prcp_methods.png", dpi=150)
    fig.savefig(FIG / "prcp_methods.pdf")
    json.dump({"methods": methods, "series": src},
              open(FIG / "prcp_methods.source.json", "w"), indent=1)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_e2, fig_e3, fig_e4, fig_e5, fig_prcp):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__}: {e}")
    print("figures ->", FIG)
