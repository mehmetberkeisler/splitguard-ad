# SplitGuard-AD — project record

Generated 2026-10-03 15:30 UTC by `scripts/generate_project_record.py`, which reads every value below from the artefact that produced it. Re-run it after any rebuild and the record follows. No number here is typed by hand.

This is the single reference for what the project measured, what it holds on disk, and how to check any of it. For the readable account of what it all means, see `PROJECT_EXPLAINER.md`; for the argument as submitted, the manuscript.

---

## 1. The claim

> Patient-wise evaluation is only as reliable as the provenance used to define patient identity. Degrading that provenance admits patient leakage through a split that still looks patient-aware, and the admitted leakage inflates apparent discrimination.

Three controlled interventions and one observed case support it. The chain is: **degraded provenance → surviving leakage → evaluation optimism**.

## 2. The headline measurement

ADNI1, 220 CN/AD participants, 1,123 scans, one 2D coronal-centre slice per scan, ResNet-18, five seeds.

| Protocol | Test AUROC | SD | Seeds |
| --- | --- | --- | --- |
| random | 0.947 | 0.024 | 5 |
| subject_only | 0.837 | 0.035 | 5 |
| component_safe | 0.829 | 0.022 | 5 |

Source: `reports/tables/adni/adni_inflation_gap.csv`

## 3. Every robustness arm

The total gap is Protocol A − Protocol C. The component marginal is B − C, the quantity the leakage graph adds over participant grouping, and it is the paper's reported null.

| Arm | Total gap | Paired-seed CI | Hierarchical CI | Component marginal | Its CI |
| --- | --- | --- | --- | --- | --- |
| Primary (ADNI1, ResNet-18) | +0.118 | [+0.096, +0.145] | [+0.061, +0.181] | +0.008 | [-0.033, +0.042] |
| Converter-inclusive | +0.142 | [+0.131, +0.158] | [+0.081, +0.208] | -0.026 | [-0.100, +0.048] |
| MT1 excluded | +0.128 | [+0.097, +0.159] | [+0.072, +0.191] | +0.002 | [-0.017, +0.028] |
| DenseNet-121 | +0.146 | [+0.134, +0.161] | [+0.087, +0.209] | +0.011 | [-0.026, +0.053] |
| Size-balanced | +0.136 | [+0.098, +0.159] | [+0.070, +0.209] | +0.026 | [-0.040, +0.068] |
| Exact label linkage | +0.117 | [+0.072, +0.166] | [+0.046, +0.195] | +0.024 | [-0.017, +0.065] |
| Volumetric (3D) | +0.235 | — | [+0.145, +0.335] | — | — |

Every marginal interval includes zero. That is the expected result under intact provenance and is reported as a finding, not a shortfall: where participant identity is complete and correct, a leakage graph reduces to participant grouping.

## 4. The three controlled interventions

### 4.1 Label permutation — the mechanism

Diagnosis labels permuted across participants, once over the cohort so a participant straddling a boundary keeps one label. No disease association survives.

| Protocol | Permuted AUROC | 95% CI | Final train loss | Final val AUROC |
| --- | --- | --- | --- | --- |
| random | 0.878 | [0.837, 0.919] | 0.068 | 0.863 |
| subject_only | 0.521 | [0.427, 0.614] | 0.037 | 0.453 |
| component_safe | 0.427 | [0.320, 0.533] | 0.045 | 0.481 |

Seed-paired A − C: **+0.451** [+0.357, +0.546], positive on all five seeds.

All three protocols fit the permuted training set to a comparable loss. Only the random split's memorisation transfers, because its test partition holds the same patients. Sanity check: the permuted label coincides with the true diagnosis at the chance rate 48.3%, 55.1%, 50.3% under the three protocols.

Source: `reports/tables/adni/adni_permutation_null.json`

### 4.2 Contamination dose-response

Injecting a measured share of training participants into the test partition while the training set is held fixed.

| Architecture | Slope per unit contamination | 95% CI | R² | Seeds |
| --- | --- | --- | --- | --- |
| densenet121 | +0.0906 | [+0.042, +0.140] | 0.640 | 5 |
| resnet18 | +0.1251 | [+0.087, +0.164] | 0.853 | 5 |

Fit: iterated feasible GLS with a random intercept per seed and a cluster-robust sandwich standard error, so the seed is the unit rather than the run.

The two architectures give different slopes on the same data, which is why the manuscript reports this as an experiment-specific calibration and not a transportable coefficient.

Source: `reports/tables/adni/adni_dose_response.json`, `adni_dose_response_mixed_effects.json`

### 4.3 Provenance degradation and structured corruption

Three corruption operators at four intensities, five seeds, both protocols: 120 trained cells. `drop` removes an identifier visibly; `split` deals one participant into two keys; `merge` relabels two participants to one. The last two leave no field missing, which is what redistribution actually does.

| Operator | λ | Straddling B | Straddling C | Prevented | AUROC B | AUROC C | Paired Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drop | 0.0 | 0 | 0 | n/a | — | — | — |
| drop | 0.1 | 16.8 | 8.4 | 50.0% | 0.847 | 0.814 | +0.034 |
| drop | 0.25 | 31.2 | 19.8 | 36.5% | 0.857 | 0.812 | +0.045 |
| drop | 0.5 | 85.2 | 67 | 21.4% | 0.945 | 0.902 | +0.043 |
| drop | 1.0 | 182.6 | 125.4 | 31.3% | 0.956 | 0.959 | -0.004 |
| split | 0.0 | 0 | 0 | n/a | — | — | — |
| split | 0.1 | 1.2 | 0.6 | 50.0% | 0.850 | 0.836 | +0.014 |
| split | 0.25 | 1.6 | 0.8 | 50.0% | 0.868 | 0.853 | +0.015 |
| split | 0.5 | 4 | 1 | 75.0% | 0.845 | 0.806 | +0.038 |
| split | 1.0 | 61.6 | 6.4 | 89.6% | 0.877 | 0.852 | +0.025 |
| merge | 0.0 | 0 | 0 | n/a | — | — | — |
| merge | 0.1 | 0 | 0 | n/a | 0.841 | 0.841 | +0.000 |
| merge | 0.25 | 0 | 0 | n/a | 0.806 | 0.806 | +0.000 |
| merge | 0.5 | 0 | 0 | n/a | 0.815 | 0.815 | +0.000 |
| merge | 1.0 | 0 | 0 | n/a | 0.808 | 0.808 | +0.000 |

Across the 8 cells where the partitions differ, 7 favour the component-safe grouping; mean +0.0262, sign test p = 0.0352, signed-rank p = 0.0078. These are descriptive: the cells share seeds and a base manifest. Only 2 cells have an interval excluding zero.

Source: `reports/tables/adni/adni_provenance_stress_test.json`, `adni_provenance_degradation.json`

## 5. The observed case: Tier-1

A widely used public 2D AD benchmark, content-matched against OASIS-1.

| Quantity | Value |
| --- | --- |
| Redistributed images | 6400 |
| OASIS-1 volumes searched | 436 |
| Unambiguous matches | 6400 |
| Source participants recovered | 200 |
| Weakest correct match (self-correlation) | 0.9996 |
| Strongest match to a different participant | 0.9623 |
| Folder labels agreeing with source CDR | 6400 |
| Filename keys merging participants | 175 |
| Participants straddling the release's own split | 85 |
| Test images whose participant is in training | 647 of 960 |

The decisive figure is the separation between the last correct match and the best incorrect one. No threshold between them changes an assignment. Tier-1 is a provenance-damaged redistribution of Tier-2, not an independent cohort: all its participants are also in OASIS-1.

Source: `reports/tables/tier1_ground_truth_audit.json`

## 6. Threats, measured rather than argued

### Checkpoint selection

The protocols do not hold comparable validation partitions, and the checkpoint is chosen on validation AUROC. Selection optimism is the best validation AUROC minus the test AUROC.

| Protocol | Best val | Test | Selection optimism | 95% CI | Val size |
| --- | --- | --- | --- | --- | --- |
| random | 0.973 | 0.947 | +0.026 | [-0.011, +0.062] | 168 |
| subject_only | 0.867 | 0.837 | +0.030 | [-0.013, +0.072] | 171 |
| component_safe | 0.803 | 0.829 | -0.026 | [-0.062, +0.011] | 165 |

Spread across the three: 0.0552. Every interval spans zero, and the component-safe value is negative. Under permuted labels, where validation carries no signal at all, the same quantity reaches +0.159 for component-safe — that is the no-signal worst case, not the reported one.

### Label provenance

| Universe | Scans | Exact visit key | Date fallback | Date share |
| --- | --- | --- | --- | --- |
| all_scans | 2182 | 1695 | 485 | 22.2% |
| cnad_universe | 1412 | 1115 | 297 | 21.0% |
| primary_arm | 1123 | 841 | 282 | 25.1% |
| converter_components_excluded | 289 | 274 | 15 | 5.2% |

The primary arm is the worst case, because the converter components it excludes are covered by the visit key almost perfectly. The exact-linkage arm retrains on the exact-key scans alone; its gap is in the table in section 3.

## 7. What is on disk

| Location | Holds |
| --- | --- |
| `reports/tables/adni/` | 31 JSON artefacts, one per analysis, plus per-arm CSVs |
| `reports/tables/` | 31 cross-cohort artefacts |
| `reports/audits/` | 30 per-cohort audit summaries and reports |
| `release/` | three-tier redistributable manifests and `RELEASE_MANIFEST.json` |
| `paper/` | manuscript and supplement sources and PDFs, 12 figures, generated tables |
| `scripts/` | 68 Python scripts |
| `tests/` | 7 test modules |

Withheld by the ADNI data-use agreement, regenerable by a reader with their own access: `data/` (imaging and manifests), `runs/` and `runs_gpu/` (weights and per-image predictions), the per-scan linkage audit rows, the Tier-1 to OASIS participant mapping, and the Tier-3 PBKDF2 salt, which lives outside the repository entirely.

## 8. The code

| Group | Count | What it does |
| --- | --- | --- |
| `build_*.py` | 9 | manifest and leakage-graph builders, one per cohort |
| `make_*.py` | 3 | frozen component-safe split generators |
| `run_*.py` | 8 | experiment runners: inflation gap, permutation null, dose response, provenance degradation and corruption, biometric probes |
| `analyze_*.py` | 5 | second-pass analyses that fold a trained result back into its artefact |
| `bootstrap_*.py` | 1 | paired-seed bootstrap |
| `hierarchical_*.py` | 1 | subject x seed hierarchical bootstrap |
| `generate_*.py` | 11 | figure and table generators |
| `audit_*.py` | 3 | contamination and ground-truth audits |
| `train_*.py` | 2 | the trainers, 2D and volumetric |
| `verify_*.py` | 1 | manuscript number verification |
| `validate_*.py` | 1 | journal submission requirements |
| `check_*.py` | 1 | environment drift |
| `gpu_*.py` | 2 | GPU orchestration and postprocessing |
| `compare_*.py` | 1 | drift against the frozen run tree |

## 9. Verification, run just now

| Gate | Status | Result |
| --- | --- | --- |
| `verify_paper_numbers.py` | pass | All 521 checks passed: every source value appears in the manuscript. |
| `validate_submission.py` | pass | All submission requirements met. |
| `unittest discover` | pass | OK |

The suite is checked by mutation rather than by its own green light: breaking union-find, the identifier grouping, the component label rule, each corruption operator, the seeded tie-break and the permutation design produces seven mutants, and the suite fails on all seven (`python3 scripts/generate_project_record.py` does not run this; see `PROJECT_EXPLAINER.md` §6).

A fourth command is not a gate but explains the others:

```bash
python3 scripts/check_environment.py   # has this machine drifted from the pins?
```

It exists because drift takes its worst shape when the checks stay green: none of the three gates imports `sklearn` or `scipy`, so all three pass on a machine where every training script is unrunnable.

The pins it checks against are read off `runs_gpu/gpu_environment.txt`, the package list frozen on the GPU node before training, so a mismatch is a real difference from the environment the published numbers came from rather than from a version chosen in advance.

## 10. Compute provenance

Total **14.8 GPU-hours** across four rented nodes, all in one code state: most arms on a cloud H100, the 50-cell dose matrix on an RTX PRO 4500, the volumetric arm on an RTX 2000 Ada, and the exact-linkage arm, the 120-cell corruption matrix and the re-run permutation control on an RTX 4090.

ADNI data was deleted from every rented node before termination, verified by a filesystem sweep returning no match for `*adni*`, `*splitguard*` or `gpu_bundle*`.

---

*Regenerate with* `python3 scripts/generate_project_record.py`.
