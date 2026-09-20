# Tables

## Table 1. Evaluation sets and outcomes

| Set | n | Role | First-run recall | Current recall | Quiet violations |
|---|---|---|---|---|---|
| internal (`scenarios.jsonl`) | 73 | regression guard | — | 100% | 0 |
| external (`known_pathways.jsonl`) | 31 | literature-aligned | — | 100% | 4 residual (Table S3) |
| holdout 1 (`holdout.jsonl`) | 30 | developmental blind | 94.6% | 100% | 0 |
| **holdout 2 (`holdout2.jsonl`)** | **33** | **prospective blind** | **88.7%** | **97.0%** | 2 (Table S2) |
| holdout 3 (`holdout3.jsonl`) | 28 | developmental blind | 77.3% | 90.9% | 0 |

First-run values are recorded before any repair guided by that set; post-repair scores are reported on the same frozen expectations. Per-miss classification for the developmental sets is in Table S1.

## Table 2. Prospective-set residuals (n = 33)

| Item | Outcome | Interpretation |
|---|---|---|
| atenolol → airway | quiet violation | measured β2 affinity exists (Kd 1175 nM); model tracks the record more faithfully than the idealized β1-selective expectation |
| dexamethasone → sodium_load | quiet violation | measured weak COX/PLA2 activity → prostaglandin suppression → renal sodium retention; consistent with steroid-associated hypertension |
| bumetanide → kaliuresis | peak 0.0176, threshold 0.02 | borderline sub-threshold, correct direction |
| sauna → adh | peak 0.0082, threshold 0.02 | borderline sub-threshold, correct direction |
| filgrastim, pembrolizumab, vitamin D | unresolved entities | biologics / non-compound inputs |

## Table 3. Evidence tiers

| Tier | Contents | Example |
|---|---|---|
| known | database mechanism records, measured activities, curated profiles with citations | ChEMBL COX-2 IC50, IUPHAR action type, FDA-label profile row |
| virtual_screening | ligand-based similarity hits | USRCAT + ECFP4 screen |
| model-generated (llm) | network-propagated hypotheses with no direct evidence | propagated downstream modules |

Every reported edge displays its tier; the weakest link bounds the confidence of a reported causal chain.

## Table 4. Robustness summary

| Test | Result |
|---|---|
| PK sensitivity (dose, Vd, t½ ×0.5–×2, worst-case Cmax ×4/÷4) | 56/56 qualitative verdicts preserved |
| stability (max load, 24 h) | max \|x\| = 2.92 vs bound 3.05; 0 violations, 0 non-finite, 0 oscillators |
| PK accuracy vs literature | Tmax 4/5 in range, Cmax 1/5 in range |
| openFDA label falsification | 79/154 directionally checkable pairs agree; residuals classified (Table S3) |
| discovery surface | 698 non-anchored propagated responses, 432 non-generic |
| LLM comparison (n = 28) | reproducibility 28/28 vs 8/28; claimed-edge support 90.9% vs 12.5% |
| data assimilation (OSSE) | observed RMSE −46%, latent RMSE −25%, forecast RMSE −66% vs uninformed twin |
