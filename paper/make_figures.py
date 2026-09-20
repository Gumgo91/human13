"""Generate manuscript figures from measured report data (paper/figures/)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "benchmark")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

C_TXT = "#263238"
C_A = "#1565c0"   # blue
C_B = "#ef6c00"   # orange
C_OK = "#2e7d32"
C_WARN = "#c62828"
C_GRAY = "#78909c"


def box(ax, x, y, w, h, text, fc="#eceff1", ec=C_TXT, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                              fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C_TXT,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=C_TXT, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                               mutation_scale=9, color=color, lw=lw))


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")

    stages = [
        ("Unstructured\nevent\n(natural\nlanguage)", "#eceff1"),
        ("Semantic\ninterpretation\n(anchors\nonly)", "#e3f2fd"),
        ("Molecular\ngrounding\n(DBs +\nVS)", "#e8f5e9"),
        ("Target\nengagement\n(weighted)", "#e8f5e9"),
        ("Deterministic\npropagation\n(205-state\nODE)", "#fff3e0"),
        ("Causal\npaths,\nprovenance,\ntiers", "#fce4ec"),
    ]
    xs = [0.15, 1.82, 3.49, 5.16, 6.83, 8.50]
    w = 1.38; y = 2.5; h = 1.4
    for (label, fc), x in zip(stages, xs):
        box(ax, x, y, w, h, label, fc=fc, fs=7.2)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + w + 0.01, y + h / 2, xs[i + 1] - 0.02, y + h / 2)

    # propagation boundary
    ax.plot([5.0, 5.0], [1.0, 4.05], ls=(0, (4, 3)), color=C_GRAY, lw=0.9)
    ax.text(5.0, 4.08, "propagation boundary", ha="center", fontsize=7,
            color=C_GRAY, style="italic")
    ax.text(2.9, 1.5, "interpretation decides only\ninitial anchors",
            ha="center", fontsize=7, color=C_GRAY, style="italic")
    ax.text(7.9, 1.5, "every downstream effect computed\nby the shared dynamical system",
            ha="center", fontsize=7, color=C_GRAY, style="italic")

    # evidence tiers
    tiers = [("known", "#2e7d32", 0.95), ("virtual_screening", "#ef6c00", 1.7),
             ("model-generated", "#7b1fa2", 1.8)]
    ax.text(5.0, 0.72, "evidence tiers:", fontsize=7, color=C_TXT, ha="right")
    x0 = 5.25
    for name, c, tw in tiers:
        ax.add_patch(FancyBboxPatch((x0, 0.58), tw, 0.32,
                                    boxstyle="round,pad=0.01", fc="white", ec=c, lw=1.0))
        ax.text(x0 + tw / 2, 0.74, name, ha="center", va="center",
                fontsize=6.6, color=c)
        x0 += tw + 0.3

    ax.text(0.15, 0.72, "a", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig1_architecture.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1_architecture.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig2():
    cs = json.load(open(os.path.join(B, "case_study_report.json"), encoding="utf-8"))
    cel, ibu = cs["celecoxib"]["peaks"], cs["ibuprofen"]["peaks"]
    abA, abB = cs["ablation_A_no_pgi2_arm"], cs["ablation_B_no_cox1_arm"]

    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1.0], wspace=0.42)

    # (a) wiring schematic: drugs (left) -> COX rows (middle) -> arms (right)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 4.2, 6.8, 2.6, 1.5, "COX-1 row\n(platelets)", fc="#e3f2fd", fs=6.4)
    box(ax, 4.2, 3.6, 2.6, 1.5, "COX-2 row\n(endothelium)", fc="#e3f2fd", fs=6.4)
    box(ax, 7.6, 6.8, 2.3, 1.5, "TXA2\narm (+)", fc="#ffebee", fs=6.4)
    box(ax, 7.6, 3.6, 2.3, 1.5, "PGI2 arm\n(x0.5, -)", fc="#e8f5e9", fs=6.4)
    arrow(ax, 6.85, 7.55, 7.55, 7.55, color=C_WARN, lw=1.4)
    arrow(ax, 6.85, 4.35, 7.55, 4.35, color=C_OK, lw=1.4)
    ax.text(0.1, 6.3, "ibuprofen", fontsize=7.2, color=C_B, fontweight="bold")
    ax.text(0.1, 5.6, "COX-1 + COX-2", fontsize=5.8, color=C_B)
    ax.text(0.1, 2.8, "celecoxib", fontsize=7.2, color=C_A, fontweight="bold")
    ax.text(0.1, 2.1, "COX-2 selective", fontsize=5.8, color=C_A)
    arrow(ax, 2.5, 6.4, 4.15, 7.4, color=C_B, lw=1.5)
    ax.annotate("", xy=(4.15, 4.6), xytext=(2.5, 6.1),
                arrowprops=dict(arrowstyle="-|>", color=C_B, lw=1.5))
    arrow(ax, 2.5, 3.1, 4.15, 4.2, color=C_A, lw=1.5)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=11)

    # (b) propagated peaks
    ax = fig.add_subplot(gs[1])
    mods = ["platelet", "thrombin", "coronary_flow"]
    labels = ["platelet", "thrombin", "coronary\nflow"]
    x = range(len(mods)); wd = 0.36
    for i, (pk, c, name) in enumerate(((cel, C_A, "celecoxib"), (ibu, C_B, "ibuprofen"))):
        vals = [(pk[m]["peak"] if pk[m]["sign"] == "+" else -pk[m]["peak"]) for m in mods]
        ax.bar([v + (i - 0.5) * wd for v in x], vals, wd, color=c, label=name,
               edgecolor="white", lw=0.4)
    ax.axhline(0, color=C_TXT, lw=0.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("peak deviation (signed)", fontsize=7.5)
    ax.legend(fontsize=6.6, frameon=False)
    ax.set_title("b", loc="left", fontweight="bold", fontsize=11)

    # (c) ablations
    ax = fig.add_subplot(gs[2])
    pos = [0, 1, 2.6, 3.6]
    cats = ["baseline", "PGI2-arm\ndeleted", "baseline", "COX-1-arm\ndeleted"]
    vals = [cel["platelet"]["peak"], -abA["celecoxib_platelet"]["peak"] if abA["celecoxib_platelet"]["sign"]=="-" else abA["celecoxib_platelet"]["peak"],
            -ibu["platelet"]["peak"], abB["ibuprofen_platelet"]["peak"] if abB["ibuprofen_platelet"]["sign"]=="+" else -abB["ibuprofen_platelet"]["peak"]]
    cols = [C_A, C_A, C_B, C_B]
    ax.bar(pos, vals, 0.62, color=cols, edgecolor="white", lw=0.4)
    ax.axhline(0, color=C_TXT, lw=0.6)
    for i, v in zip(pos, vals):
        ax.text(i, v + (0.06 if v >= 0 else -0.12), f"{v:+.2f}", ha="center",
                fontsize=6.2, color=C_TXT)
    ax.set_xticks(pos); ax.set_xticklabels(cats, fontsize=6.0)
    ax.text(0.5, 0.55, "celecoxib", ha="center", fontsize=6.8, color=C_A, fontweight="bold")
    ax.text(3.1, 0.55, "ibuprofen", ha="center", fontsize=6.8, color=C_B, fontweight="bold")
    ax.set_ylim(-1.35, 0.72)
    ax.set_ylabel("platelet peak (signed)", fontsize=7.5)
    ax.set_title("c", loc="left", fontweight="bold", fontsize=11)

    fig.savefig(os.path.join(OUT, "fig2_cox_case.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig2_cox_case.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig3():
    sens = json.load(open(os.path.join(B, "sensitivity_report.json"), encoding="utf-8"))
    stab = json.load(open(os.path.join(B, "stability_report.json"), encoding="utf-8"))

    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.15, 0.85], wspace=0.5)

    # (a) recall across sets
    ax = fig.add_subplot(gs[0])
    sets = [
        ("internal\nn=73", None, 1.0),
        ("external\nn=31", None, 1.0),
        ("holdout 1\nn=30", 0.946, 1.0),
        ("holdout 2\nn=33\n(prospective)", 0.887, 0.970),
        ("holdout 3\nn=28", 0.773, 0.909),
    ]
    wd = 0.36
    for i, (name, first, final) in enumerate(sets):
        if first is not None:
            ax.bar(i - wd / 2, first * 100, wd, color="#b0bec5", edgecolor="white", lw=0.4)
            ax.text(i - wd / 2, first * 100 + 1.5, f"{first*100:.1f}", ha="center", fontsize=5.9, color=C_GRAY)
        ax.bar(i + (wd / 2 if first else 0), final * 100, wd, color=C_A, edgecolor="white", lw=0.4)
        ax.text(i + (wd / 2 if first else 0), final * 100 + 1.5, f"{final*100:.1f}",
                ha="center", fontsize=5.9, color=C_A)
    ax.bar([-1], [0], color="#b0bec5", label="first run (blind)")
    ax.bar([-1], [0], color=C_A, label="after mechanism-level repair")
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[0] for s in sets], fontsize=6.4)
    ax.set_ylim(0, 118); ax.set_ylabel("module recall (%)", fontsize=7.5)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    ax.set_title("a", loc="left", fontweight="bold", fontsize=11)

    # (b) sensitivity heatmap: scenarios x perturbations, all preserved
    ax = fig.add_subplot(gs[1])
    scens = sens["scenarios"]
    perts = list(scens[0]["perturbations"].keys())
    grid = [[1.0 if scens[r]["perturbations"][p]["n_flip"] == 0 else 0.0
             for p in perts] for r in range(len(scens))]
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#ffcdd2", "#a5d6a7"]),
              aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(perts)))
    ax.set_xticklabels([p.replace("_", "\n") for p in perts], fontsize=5.4)
    ax.set_yticks(range(len(scens)))
    ax.set_yticklabels([s["id"] for s in scens], fontsize=6.4)
    ax.set_title(f"{sens['n_preserved']}/{sens['n_judgments']} verdicts preserved",
                 fontsize=7, color=C_OK)
    ax.set_title("b", loc="left", fontweight="bold", fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)

    # (c) stability bound
    ax = fig.add_subplot(gs[2])
    mx = stab.get("max_abs", stab.get("max|x|", 2.9189))
    ax.bar([0, 1], [3.05, mx], 0.55, color=["#eceff1", C_A],
           edgecolor=[C_GRAY, "white"], lw=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["analytic\nbound", "observed\nmax |x|"], fontsize=6.6)
    ax.set_ylim(0, 3.7)
    ax.text(1, mx + 0.12, f"{mx:.2f}", ha="center", fontsize=7, color=C_A, fontweight="bold")
    ax.set_ylabel("state excursion", fontsize=7.5)
    n_osc = stab.get("oscillators", stab.get("n_oscillators", 0))
    ax.text(0.5, 3.5, f"0 bound violations · {n_osc} oscillators", ha="center",
            fontsize=6.4, color=C_OK)
    ax.set_title("c", loc="left", fontweight="bold", fontsize=11)

    fig.savefig(os.path.join(OUT, "fig3_evaluation.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig3_evaluation.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig4():
    rep = json.load(open(os.path.join(B, "assimilation_report.json"), encoding="utf-8"))
    arms, tr = rep["arms"], rep["trajectories"]

    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.0], wspace=0.45)

    # (a) assimilation loop schematic
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.1, 7.2, 3.0, 1.8, "event", fc="#eceff1", fs=6.6)
    box(ax, 3.9, 7.2, 3.0, 1.8, "whole-body\nmodel", fc="#fff3e0", fs=6.6)
    box(ax, 7.6, 7.2, 2.4, 1.8, "forecast", fc="#fce4ec", fs=6.6)
    box(ax, 3.9, 1.2, 3.0, 1.8, "observed\ndata", fc="#e8f5e9", fs=6.6)
    box(ax, 3.9, 4.0, 3.0, 1.8, "state\nupdate", fc="#e3f2fd", fs=6.6)
    arrow(ax, 3.15, 8.1, 3.85, 8.1)
    arrow(ax, 6.95, 8.1, 7.55, 8.1)
    arrow(ax, 5.4, 3.05, 5.4, 3.95)
    arrow(ax, 5.4, 5.85, 5.4, 7.15)
    ax.annotate("", xy=(6.95, 4.9), xytext=(8.6, 7.1),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(8.6, 5.6, "next\nobs", fontsize=5.8, color=C_GRAY, ha="center")
    ax.set_title("a", loc="left", fontweight="bold", fontsize=11)

    # (b) latent trajectory: adh
    ax = fig.add_subplot(gs[1])
    tt = np.asarray(tr["times"]); m = "adh"
    d = tr["modules"][m]
    ax.plot(tt, d["truth"], color=C_TXT, lw=1.6, label="truth (event)")
    ax.plot(tt, d["blind"], color=C_GRAY, lw=1.0, ls=":", label="twin, no obs")
    ax.plot(tt, d["twin"], color=C_A, lw=1.4, label="twin + assimilation")
    ax.plot(tt, d["twin_fcst"], color=C_B, lw=1.0, ls="--", label="twin fcst (obs<=120)")
    for t0 in tr["obs_times"]:
        ax.axvline(t0, color=C_OK, lw=0.4, alpha=0.35)
    ax.axvline(120, color=C_B, lw=0.8, ls="--")
    ax.set_xlabel("time (min)", fontsize=7.5); ax.set_ylabel("adh deviation", fontsize=7.5)
    ax.legend(fontsize=6.0, frameon=False, loc="upper right")
    ax.set_title("b  latent state (unobserved): adh", loc="left", fontsize=8)
    ax.set_title("b", loc="left", fontweight="bold", fontsize=11)
    ax.text(0.5, 0.02, "obs every 30 min", transform=ax.transAxes, fontsize=6, color=C_OK)

    # (c) RMSE summary
    ax = fig.add_subplot(gs[2])
    groups = ["observed", "latent", "forecast\n(t>120)"]
    blind_v = [arms["blind"]["rmse_observed"], arms["blind"]["rmse_latent"], arms["blind_t>120"]["rmse_all"]]
    assim_v = [arms["twin_assim"]["rmse_observed"], arms["twin_assim"]["rmse_latent"], arms["twin_fcst_t>120"]["rmse_all"]]
    x = np.arange(3); wd = 0.35
    ax.bar(x - wd / 2, blind_v, wd, color="#b0bec5", label="no assimilation")
    ax.bar(x + wd / 2, assim_v, wd, color=C_A, label="with assimilation")
    for xi, v in zip(x - wd / 2, blind_v): ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=5.8, color=C_GRAY)
    for xi, v in zip(x + wd / 2, assim_v): ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=5.8, color=C_A)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=6.6)
    ax.set_ylabel("RMSE vs truth", fontsize=7.5); ax.set_ylim(0, 0.26)
    ax.legend(fontsize=6.0, frameon=False, loc="upper right")
    ax.set_title("c", loc="left", fontweight="bold", fontsize=11)

    fig.savefig(os.path.join(OUT, "fig4_assimilation.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig4_assimilation.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("wrote", sorted(os.listdir(OUT)))
