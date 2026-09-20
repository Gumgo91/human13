# Paper package

Manuscript and supporting materials for the Human13 architecture paper.

## Contents

| File | Contents |
|---|---|
| `manuscript.md` | Full draft: title, abstract, introduction, results, discussion, methods, references, table/figure legends |
| `tables.md` | Tables 1–4 with measured values (evaluation sets, residuals, evidence tiers, robustness) |
| `supplementary.md` | Tables S1–S5 + Note 1 (developmental-set miss classification, residuals, thresholds, DA protocol, LLM-comparison detail) |
| `make_figures.py` | Figure generator; reads measured values from `benchmark/*_report.json` |
| `figures/` | fig1 architecture, fig2 COX case study, fig3 evaluation/robustness, fig4 data assimilation (PNG 300 dpi + PDF) |

Regenerate figures: `./.venv/Scripts/python.exe paper/make_figures.py`

## Source data (`benchmark/`)

| Harness | Report | What it measures |
|---|---|---|
| `evaluate.py` | `report.json` | internal regression, 73 scenarios |
| `external_eval.py` | `external_report.json` | literature-curated alignment, 31 scenarios |
| `external_eval.py --set holdout*.jsonl` | `holdout*_report.json` | blind sets (30/33/28) |
| `sensitivity.py` | `sensitivity_report.json` | PK perturbation, 56 verdicts |
| `stability` harness | `stability_report.json` | boundedness under maximal load |
| `pk_check.py` | `pk_report.json` | Tmax/Cmax vs literature |
| `label_check.py` | `label_check_report.json` | openFDA falsification, 79/154 |
| `discovery.py` | `discovery_report.json` | non-anchored propagated surface, 698 (432 non-generic) |
| `case_study.py` | `case_study_report.json` | celecoxib vs ibuprofen + ablations |
| `llm_compare.py` | `llm_compare_report.json` | Human13 vs LLM-only vs LLM+grounding |
| `assimilation_eval.py` | `assimilation_report.json` | data-assimilation OSSE |
| — | `VALIDATION.md` | consolidated validation summary |

Scenario files: `scenarios.jsonl` (internal), `known_pathways.jsonl` (external),
`holdout.jsonl`, `holdout2.jsonl` (prospective), `holdout3.jsonl`.

## Reporting rules (kept honest)

- holdout2 is the prospective set: its answer key was authored blind before
  any model output; both first-run (88.7%) and post-repair (97.0%) values are
  reported. holdout1/holdout3 are developmental sets, classified per miss in
  Table S1.
- Remaining misses are reported as data/scope limitations, never patched to
  satisfy expectations (Table S2).
- LLM-comparison metrics are reported on completed runs only; the arm's call
  failure rate is stated explicitly.
- Ablation numbers are mechanistic sensitivity evidence, not validation.
