"""Generate MANIFEST.csv (sha256 of every packaged file) and the v3 figure
set + figure source-data CSVs. Excludes secrets/caches by design."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "human13_ncs_validation_v3"
sys.path.insert(0, str(ROOT))

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "node_modules"}
EXCLUDE_EXT = {".pyc", ".pyo", ".env"}


def manifest():
    rows = []
    for p in sorted(V3.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.csv":
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in EXCLUDE_EXT or p.name == ".env":
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"path": str(p.relative_to(V3)).replace("\\", "/"),
                     "bytes": p.stat().st_size, "sha256": h})
    with open(V3 / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {len(rows)} files")


def figure_source_data():
    """Emit figure source-data CSVs from the metrics files."""
    src = V3 / "figures" / "source_data"
    src.mkdir(parents=True, exist_ok=True)
    for name in ["forecast_metrics", "robustness_metrics",
                 "observation_panel_metrics",
                 "joint_drug_assimilation_metrics",
                 "reproduction_comparison", "affinity_mapping_before_after"]:
        f = V3 / "results" / f"{name}.csv"
        if f.exists():
            (src / f"{name}.csv").write_bytes(f.read_bytes())
    print("figure source data staged")


def figures():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib unavailable — figures NOT_RUN")
        return
    figd = V3 / "figures"
    figd.mkdir(exist_ok=True)

    # --- Fig A: E2 forecast RMSE by method x cut -------------------------
    rows = list(csv.DictReader(
        open(V3 / "results" / "forecast_metrics.csv", encoding="utf-8")))
    cuts = sorted({r["cut"] for r in rows}, key=float)
    methods = ["blind", "persistence", "jump", "nudge"]
    for panel in ["observed", "latent"]:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        for mi, m in enumerate(methods):
            xs, ys, es = [], [], []
            for c in cuts:
                v = [float(r["pooled_rmse"]) for r in rows
                     if r["cut"] == c and r["method"] == m
                     and r["panel"] == panel and r["pooled_rmse"]]
                if v:
                    xs.append(float(c)); ys.append(sum(v) / len(v))
                    es.append(np.std(v) / np.sqrt(len(v)))
            ax.errorbar([x + mi * 2.5 for x in xs], ys, yerr=es,
                        marker="o", ms=4, lw=1.4, capsize=3, label=m)
        ax.set_xlabel("forecast cut (min)"); ax.set_ylabel("RMSE (t>cut)")
        ax.set_title(f"E2 forecast — {panel} states")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(figd / f"fig_e2_{panel}.png", dpi=300)
        fig.savefig(figd / f"fig_e2_{panel}.pdf")
        plt.close(fig)

    # --- Fig B: E4 panels ------------------------------------------------
    rows = list(csv.DictReader(open(
        V3 / "results" / "observation_panel_metrics.csv",
        encoding="utf-8")))
    panels = ["full", "upstream", "downstream"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    w, x = 0.35, np.arange(2)
    for pi, p in enumerate(panels):
        for xi, ep in enumerate(["observed_full_panel", "latent"]):
            v = [float(r["pooled_rmse"]) for r in rows
                 if r["obs_panel"] == p and r["eval_panel"] == ep
                 and r["pooled_rmse"]]
            ax.bar(x[xi] + (pi - 1) * w, sum(v) / len(v), w * 0.9,
                   label=p if xi == 0 else None,
                   color=["#4c72b0", "#55a868", "#c44e52"][pi])
    ax.set_xticks(x); ax.set_xticklabels(["observed", "latent"])
    ax.set_ylabel("RMSE (t>cut)"); ax.set_title("E4 observation panels")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figd / "fig_e4_panels.png", dpi=300)
    fig.savefig(figd / "fig_e4_panels.pdf")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    figure_source_data()
    figures()
    manifest()
