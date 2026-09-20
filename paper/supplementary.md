# Supplementary Information

## Supplementary Table S1. Developmental blind-set outcomes and per-miss classification

All three blind sets used the same protocol: expectations were authored from drug labels and mechanism references before observing model output; every first-run miss was classified as an implementation defect (repaired at mechanism level) or a data/scope limitation (reported, not patched).

### Holdout 1 (n = 30): first-run 94.6% → post-repair 100%

| Item | First-run outcome | Classification | Resolution |
|---|---|---|---|
| azithromycin → QT axis | sign inverted | defect: hERG anchor sign | mechanism-level repair |
| antithrombin → anticoagulant | path absent | defect: incomplete wiring | mechanism-level repair |
| heparin | entity unresolved | limitation: biologic, no public compound record | reported |

### Holdout 3 (n = 28): first-run 77.3% → post-repair 90.9%

| Item | First-run outcome | Classification | Resolution |
|---|---|---|---|
| guanfacine → sympathetic | sign inverted | defect: adrenergic word-order target pattern | pattern repair (all α/β rows) |
| bromocriptine → prolactin | no anchor | defect: D2 synonym coverage ("D2-like", "D(2)") | pattern repair |
| linaclotide → cGMP/gut | no anchor | defect: GC-C synonym ("heat-stable enterotoxin receptor") | pattern repair |
| oxybutynin, atropine → mucus | no response | defect: cholinergic secretion drive absent | wiring repair (`mucus +0.4*parasympathetic`) |
| mirtazapine → arousal | sign `+` vs `-` | limitation: record lacks H1 activity that dominates clinical sedation | reported (data gap) |
| carbamazepine → adh | no anchor | limitation: hormone-release effect, no receptor target | reported (scope) |
| alteplase → fibrinolysis/fibrin | no anchor | limitation: protein therapeutic unresolvable by small-molecule lookup | reported (coverage) |
| adalimumab, rituximab | entities unresolved | limitation: antibody therapeutics | reported |

Quiet-violation residual introduced by the secretion repair: losartan → mucus −0.018 (weak propagated signal; retained).

## Supplementary Table S2. Prospective set residual detail (n = 33)

| Item | Outcome | Interpretation |
|---|---|---|
| atenolol → airway (quiet violation) | weak β2 response propagated | measured β2 affinity exists (Kd 1175 nM); the label itself advises caution — model tracks the record more faithfully than the idealized β1-selective expectation |
| dexamethasone → sodium_load (quiet violation) | sodium retention propagated | measured weak COX/PLA2 activity → renal prostaglandin suppression → retention; consistent with documented steroid-associated hypertension |
| bumetanide → kaliuresis | peak 0.0176 (threshold 0.02) | borderline sub-threshold propagation, correct direction |
| sauna → adh | peak 0.0082 (threshold 0.02) | borderline sub-threshold propagation, correct direction |
| filgrastim, pembrolizumab, vitamin D | entities unresolved | biologics / non-compound inputs outside lookup coverage |

## Supplementary Table S3. External set residual quiet signals (n = 31)

| Item | Peak | Note |
|---|---|---|
| sleep deprivation → urine_output | +0.014 | circadian ADH suppression via serotonergic afferent; weak, documented |
| prednisone → micturition | −0.040 | mineralocorticoid water retention → reduced voiding; physiologically plausible, clinically negligible |
| heat → browning | small | thermogenic browning wiring; weak residual |
| losartan → mucus | −0.018 | parasympathetic shift → secretion; introduced by cholinergic-secretion repair |

## Supplementary Table S4. Evaluation harness thresholds

| Quantity | Threshold |
|---|---|
| responded module | \|peak\| > 0.02 with expected sign |
| quiet module | \|peak\| ≤ 0.01 |
| discovery-surface hit | non-anchored module \|peak\| > 0.1 |
| LLM-compare sign call | \|peak\| > 0.1 |

## Supplementary Table S5. Data-assimilation protocol and results

| Quantity | Value |
|---|---|
| truth event | hemorrhage, 120 min, horizon 240 min |
| observable panel (8) | sympathetic, plasma_vol, osmotic_drive, urine_output, gfr, sodium_load, ventilation, preload |
| latent panel | renin, angiotensin, aldosterone, adh, aqp2, kaliuresis, thirst, edema |
| update rule | continuous nudging γ(z − x), γ = 0.15/min, last-observation-hold |
| observation noise | Gaussian, sd 0.03 (dimensionless) |
| observed RMSE | 0.185 (blind) → 0.100 (assimilated) |
| latent RMSE | 0.232 → 0.175 (every latent module improved) |
| post-release forecast RMSE (t > 120) | 0.095 (blind) → 0.032 (assimilated) |

## Supplementary Note 1. LLM comparison detail

Arm sizes n = 28 scenarios, three runs each (deepseek/deepseek-v4-flash-0731). Human13: 28/28 reproducible sign vectors by construction, 90.9% of claimed links backed by real graph edges, 100% provenance coverage, 0 errors. LLM-only: identical sign vectors 8/28, claimed-edge support 12.5%, 74/84 calls failed or produced unparseable output; on completed runs it reached the documented direction for well-known compounds. LLM + molecular records: edge support 13.0%, provenance text citations 91.3%, reproducibility unchanged. Direction accuracy is reported on completed runs only.
