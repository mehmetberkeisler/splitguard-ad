# Re-run comparison: published artefacts vs regenerated

Old: `runs_frozen/` (116 runs) — trained before the DataLoader rewiring in 877ca72, mostly before the first commit of the training code.
New: `runs/` (196 runs) — one code state, one device.

Matched runs: 116. New-only: 80. Missing from the re-run: 0.

## `adni` — 15 runs

Mean absolute change +0.0235; largest single change +0.0836.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni/inflation_gap_seed0/component_safe` | 0.8174 | 0.8133 | -0.0041 |
| `adni/inflation_gap_seed0/random` | 0.9142 | 0.9101 | -0.0041 |
| `adni/inflation_gap_seed0/subject_only` | 0.8496 | 0.8301 | -0.0195 ⚠️ |
| `adni/inflation_gap_seed1/component_safe` | 0.8254 | 0.8546 | +0.0292 ⚠️ |
| `adni/inflation_gap_seed1/random` | 0.9556 | 0.9478 | -0.0078 |
| `adni/inflation_gap_seed1/subject_only` | 0.8030 | 0.8645 | +0.0615 ⚠️ |
| `adni/inflation_gap_seed2/component_safe` | 0.7675 | 0.8511 | +0.0836 ⚠️ |
| `adni/inflation_gap_seed2/random` | 0.9422 | 0.9497 | +0.0075 |
| `adni/inflation_gap_seed2/subject_only` | 0.8017 | 0.7847 | -0.0170 ⚠️ |
| `adni/inflation_gap_seed3/component_safe` | 0.8183 | 0.8134 | -0.0049 |
| `adni/inflation_gap_seed3/random` | 0.9670 | 0.9528 | -0.0142 ⚠️ |
| `adni/inflation_gap_seed3/subject_only` | 0.8609 | 0.8345 | -0.0264 ⚠️ |
| `adni/inflation_gap_seed4/component_safe` | 0.8684 | 0.8119 | -0.0565 ⚠️ |
| `adni/inflation_gap_seed4/random` | 0.9650 | 0.9756 | +0.0106 ⚠️ |
| `adni/inflation_gap_seed4/subject_only` | 0.8777 | 0.8724 | -0.0053 |

## `adni_densenet121` — 15 runs

Mean absolute change +0.0180; largest single change -0.0487.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni_densenet121/inflation_gap_seed0/component_safe` | 0.8372 | 0.8186 | -0.0186 ⚠️ |
| `adni_densenet121/inflation_gap_seed0/random` | 0.9402 | 0.9491 | +0.0089 |
| `adni_densenet121/inflation_gap_seed0/subject_only` | 0.8327 | 0.8194 | -0.0133 ⚠️ |
| `adni_densenet121/inflation_gap_seed1/component_safe` | 0.8731 | 0.8373 | -0.0358 ⚠️ |
| `adni_densenet121/inflation_gap_seed1/random` | 0.9663 | 0.9700 | +0.0037 |
| `adni_densenet121/inflation_gap_seed1/subject_only` | 0.7849 | 0.8053 | +0.0204 ⚠️ |
| `adni_densenet121/inflation_gap_seed2/component_safe` | 0.8075 | 0.8223 | +0.0148 ⚠️ |
| `adni_densenet121/inflation_gap_seed2/random` | 0.9589 | 0.9686 | +0.0097 |
| `adni_densenet121/inflation_gap_seed2/subject_only` | 0.7980 | 0.7881 | -0.0099 |
| `adni_densenet121/inflation_gap_seed3/component_safe` | 0.8196 | 0.8079 | -0.0117 ⚠️ |
| `adni_densenet121/inflation_gap_seed3/random` | 0.9688 | 0.9558 | -0.0130 ⚠️ |
| `adni_densenet121/inflation_gap_seed3/subject_only` | 0.8135 | 0.8386 | +0.0251 ⚠️ |
| `adni_densenet121/inflation_gap_seed4/component_safe` | 0.8284 | 0.7981 | -0.0303 ⚠️ |
| `adni_densenet121/inflation_gap_seed4/random` | 0.9792 | 0.9732 | -0.0060 |
| `adni_densenet121/inflation_gap_seed4/subject_only` | 0.9361 | 0.8874 | -0.0487 ⚠️ |

## `adni_dose_response` — 51 runs

Mean absolute change +0.0000; largest single change +0.0000.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni_dose_response/densenet121/seed0_overlap0.0/baseline_seed0` | 0.8489 | 0.8489 | +0.0000 |
| `adni_dose_response/densenet121/seed0_overlap0.25/baseline_seed0` | 0.8416 | 0.8416 | +0.0000 |
| `adni_dose_response/densenet121/seed0_overlap0.50/baseline_seed0` | 0.8459 | 0.8459 | +0.0000 |
| `adni_dose_response/densenet121/seed0_overlap0.75/baseline_seed0` | 0.8937 | 0.8937 | +0.0000 |
| `adni_dose_response/densenet121/seed0_overlap1.0/baseline_seed0` | 0.9410 | 0.9410 | +0.0000 |
| `adni_dose_response/densenet121/seed1_overlap0.0/baseline_seed1` | 0.8476 | 0.8476 | +0.0000 |
| `adni_dose_response/densenet121/seed1_overlap0.25/baseline_seed1` | 0.8579 | 0.8579 | +0.0000 |
| `adni_dose_response/densenet121/seed1_overlap0.50/baseline_seed1` | 0.8875 | 0.8875 | +0.0000 |
| `adni_dose_response/densenet121/seed1_overlap0.75/baseline_seed1` | 0.9451 | 0.9451 | +0.0000 |
| `adni_dose_response/densenet121/seed1_overlap1.0/baseline_seed1` | 0.9323 | 0.9323 | +0.0000 |
| `adni_dose_response/densenet121/seed2_overlap0.0/baseline_seed2` | 0.9136 | 0.9136 | +0.0000 |
| `adni_dose_response/densenet121/seed2_overlap0.25/baseline_seed2` | 0.9074 | 0.9074 | +0.0000 |
| `adni_dose_response/densenet121/seed2_overlap0.50/baseline_seed2` | 0.9057 | 0.9057 | +0.0000 |
| `adni_dose_response/densenet121/seed2_overlap0.75/baseline_seed2` | 0.9473 | 0.9473 | +0.0000 |
| `adni_dose_response/densenet121/seed2_overlap1.0/baseline_seed2` | 0.9457 | 0.9457 | +0.0000 |
| `adni_dose_response/densenet121/seed3_overlap0.0/baseline_seed3` | 0.9060 | 0.9060 | +0.0000 |
| `adni_dose_response/densenet121/seed3_overlap0.25/baseline_seed3` | 0.8827 | 0.8827 | +0.0000 |
| `adni_dose_response/densenet121/seed3_overlap0.50/baseline_seed3` | 0.9506 | 0.9506 | +0.0000 |
| `adni_dose_response/densenet121/seed3_overlap0.75/baseline_seed3` | 0.9443 | 0.9443 | +0.0000 |
| `adni_dose_response/densenet121/seed3_overlap1.0/baseline_seed3` | 0.9624 | 0.9624 | +0.0000 |
| `adni_dose_response/densenet121/seed4_overlap0.0/baseline_seed4` | 0.8605 | 0.8605 | +0.0000 |
| `adni_dose_response/densenet121/seed4_overlap0.25/baseline_seed4` | 0.9063 | 0.9063 | +0.0000 |
| `adni_dose_response/densenet121/seed4_overlap0.50/baseline_seed4` | 0.8886 | 0.8886 | +0.0000 |
| `adni_dose_response/densenet121/seed4_overlap0.75/baseline_seed4` | 0.9546 | 0.9546 | +0.0000 |
| `adni_dose_response/densenet121/seed4_overlap1.0/baseline_seed4` | 0.8602 | 0.8602 | +0.0000 |
| `adni_dose_response/resnet18/seed0_overlap0.0/baseline_seed0` | 0.8118 | 0.8118 | +0.0000 |
| `adni_dose_response/resnet18/seed0_overlap0.25/baseline_seed0` | 0.8211 | 0.8211 | +0.0000 |
| `adni_dose_response/resnet18/seed0_overlap0.5/baseline_seed0` | 0.8130 | 0.8130 | +0.0000 |
| `adni_dose_response/resnet18/seed0_overlap0.50/baseline_seed0` | 0.8271 | 0.8271 | +0.0000 |
| `adni_dose_response/resnet18/seed0_overlap0.75/baseline_seed0` | 0.8723 | 0.8723 | +0.0000 |
| `adni_dose_response/resnet18/seed0_overlap1.0/baseline_seed0` | 0.9562 | 0.9562 | +0.0000 |
| `adni_dose_response/resnet18/seed1_overlap0.0/baseline_seed1` | 0.8548 | 0.8548 | +0.0000 |
| `adni_dose_response/resnet18/seed1_overlap0.25/baseline_seed1` | 0.8728 | 0.8728 | +0.0000 |
| `adni_dose_response/resnet18/seed1_overlap0.50/baseline_seed1` | 0.9078 | 0.9078 | +0.0000 |
| `adni_dose_response/resnet18/seed1_overlap0.75/baseline_seed1` | 0.9291 | 0.9291 | +0.0000 |
| `adni_dose_response/resnet18/seed1_overlap1.0/baseline_seed1` | 0.9384 | 0.9384 | +0.0000 |
| `adni_dose_response/resnet18/seed2_overlap0.0/baseline_seed2` | 0.8527 | 0.8527 | +0.0000 |
| `adni_dose_response/resnet18/seed2_overlap0.25/baseline_seed2` | 0.8270 | 0.8270 | +0.0000 |
| `adni_dose_response/resnet18/seed2_overlap0.50/baseline_seed2` | 0.9131 | 0.9131 | +0.0000 |
| `adni_dose_response/resnet18/seed2_overlap0.75/baseline_seed2` | 0.8860 | 0.8860 | +0.0000 |
| `adni_dose_response/resnet18/seed2_overlap1.0/baseline_seed2` | 0.9283 | 0.9283 | +0.0000 |
| `adni_dose_response/resnet18/seed3_overlap0.0/baseline_seed3` | 0.8520 | 0.8520 | +0.0000 |
| `adni_dose_response/resnet18/seed3_overlap0.25/baseline_seed3` | 0.8702 | 0.8702 | +0.0000 |
| `adni_dose_response/resnet18/seed3_overlap0.50/baseline_seed3` | 0.9126 | 0.9126 | +0.0000 |
| `adni_dose_response/resnet18/seed3_overlap0.75/baseline_seed3` | 0.9306 | 0.9306 | +0.0000 |
| `adni_dose_response/resnet18/seed3_overlap1.0/baseline_seed3` | 0.9418 | 0.9418 | +0.0000 |
| `adni_dose_response/resnet18/seed4_overlap0.0/baseline_seed4` | 0.8157 | 0.8157 | +0.0000 |
| `adni_dose_response/resnet18/seed4_overlap0.25/baseline_seed4` | 0.8639 | 0.8639 | +0.0000 |
| `adni_dose_response/resnet18/seed4_overlap0.50/baseline_seed4` | 0.9055 | 0.9055 | +0.0000 |
| `adni_dose_response/resnet18/seed4_overlap0.75/baseline_seed4` | 0.9176 | 0.9176 | +0.0000 |
| `adni_dose_response/resnet18/seed4_overlap1.0/baseline_seed4` | 0.9475 | 0.9475 | +0.0000 |

## `adni_inflation_smoke` — 3 runs

Mean absolute change +0.0000; largest single change +0.0000.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni_inflation_smoke/inflation_gap_seed0/component_safe` | 0.7262 | 0.7262 | +0.0000 |
| `adni_inflation_smoke/inflation_gap_seed0/random` | 0.7791 | 0.7791 | +0.0000 |
| `adni_inflation_smoke/inflation_gap_seed0/subject_only` | 0.7857 | 0.7857 | +0.0000 |

## `adni_no_mt1` — 15 runs

Mean absolute change +0.0163; largest single change -0.0516.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni_no_mt1/inflation_gap_seed0/component_safe` | 0.8711 | 0.8672 | -0.0039 |
| `adni_no_mt1/inflation_gap_seed0/random` | 0.9552 | 0.9548 | -0.0004 |
| `adni_no_mt1/inflation_gap_seed0/subject_only` | 0.8000 | 0.8417 | +0.0417 ⚠️ |
| `adni_no_mt1/inflation_gap_seed1/component_safe` | 0.8229 | 0.8524 | +0.0295 ⚠️ |
| `adni_no_mt1/inflation_gap_seed1/random` | 0.9358 | 0.9383 | +0.0025 |
| `adni_no_mt1/inflation_gap_seed1/subject_only` | 0.8126 | 0.8424 | +0.0298 ⚠️ |
| `adni_no_mt1/inflation_gap_seed2/component_safe` | 0.8435 | 0.7919 | -0.0516 ⚠️ |
| `adni_no_mt1/inflation_gap_seed2/random` | 0.9739 | 0.9622 | -0.0117 ⚠️ |
| `adni_no_mt1/inflation_gap_seed2/subject_only` | 0.7983 | 0.7886 | -0.0097 |
| `adni_no_mt1/inflation_gap_seed3/component_safe` | 0.8179 | 0.8203 | +0.0024 |
| `adni_no_mt1/inflation_gap_seed3/random` | 0.9836 | 0.9603 | -0.0233 ⚠️ |
| `adni_no_mt1/inflation_gap_seed3/subject_only` | 0.8223 | 0.8153 | -0.0070 |
| `adni_no_mt1/inflation_gap_seed4/component_safe` | 0.8329 | 0.8252 | -0.0077 |
| `adni_no_mt1/inflation_gap_seed4/random` | 0.9733 | 0.9826 | +0.0093 |
| `adni_no_mt1/inflation_gap_seed4/subject_only` | 0.8633 | 0.8766 | +0.0133 ⚠️ |

## `adni_pilot_n163` — 1 runs

Mean absolute change +0.0000; largest single change +0.0000.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni_pilot_n163/baseline_seed0` | 0.8542 | 0.8542 | +0.0000 |

## `adni_smoke_n2182` — 1 runs

Mean absolute change +0.0000; largest single change +0.0000.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni_smoke_n2182/baseline_seed0` | 0.8179 | 0.8179 | +0.0000 |

## `adni_with_converters` — 15 runs

Mean absolute change +0.0286; largest single change +0.0836.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni_with_converters/inflation_gap_seed0/component_safe` | 0.7440 | 0.7805 | +0.0365 ⚠️ |
| `adni_with_converters/inflation_gap_seed0/random` | 0.9532 | 0.9527 | -0.0005 |
| `adni_with_converters/inflation_gap_seed0/subject_only` | 0.8363 | 0.8588 | +0.0225 ⚠️ |
| `adni_with_converters/inflation_gap_seed1/component_safe` | 0.8235 | 0.7992 | -0.0243 ⚠️ |
| `adni_with_converters/inflation_gap_seed1/random` | 0.9438 | 0.9381 | -0.0057 |
| `adni_with_converters/inflation_gap_seed1/subject_only` | 0.6661 | 0.7497 | +0.0836 ⚠️ |
| `adni_with_converters/inflation_gap_seed2/component_safe` | 0.8664 | 0.8184 | -0.0480 ⚠️ |
| `adni_with_converters/inflation_gap_seed2/random` | 0.9505 | 0.9574 | +0.0069 |
| `adni_with_converters/inflation_gap_seed2/subject_only` | 0.8776 | 0.8836 | +0.0060 |
| `adni_with_converters/inflation_gap_seed3/component_safe` | 0.8614 | 0.8446 | -0.0168 ⚠️ |
| `adni_with_converters/inflation_gap_seed3/random` | 0.9653 | 0.9783 | +0.0130 ⚠️ |
| `adni_with_converters/inflation_gap_seed3/subject_only` | 0.7057 | 0.7450 | +0.0393 ⚠️ |
| `adni_with_converters/inflation_gap_seed4/component_safe` | 0.7756 | 0.8324 | +0.0568 ⚠️ |
| `adni_with_converters/inflation_gap_seed4/random` | 0.9608 | 0.9576 | -0.0032 |
| `adni_with_converters/inflation_gap_seed4/subject_only` | 0.7719 | 0.7058 | -0.0661 ⚠️ |

## New arms (no published counterpart)

- `adni_permutation_null/permutation_null_seed0/component_safe` — 0.4981
- `adni_permutation_null/permutation_null_seed0/random` — 0.4445
- `adni_permutation_null/permutation_null_seed0/subject_only` — 0.4400
- `adni_permutation_null/permutation_null_seed1/component_safe` — 0.4387
- `adni_permutation_null/permutation_null_seed1/random` — 0.5869
- `adni_permutation_null/permutation_null_seed1/subject_only` — 0.5217
- `adni_permutation_null/permutation_null_seed2/component_safe` — 0.4537
- `adni_permutation_null/permutation_null_seed2/random` — 0.5475
- `adni_permutation_null/permutation_null_seed2/subject_only` — 0.5734
- `adni_permutation_null/permutation_null_seed3/component_safe` — 0.5828
- `adni_permutation_null/permutation_null_seed3/random` — 0.5101
- `adni_permutation_null/permutation_null_seed3/subject_only` — 0.5224
- `adni_permutation_null/permutation_null_seed4/component_safe` — 0.6120
- `adni_permutation_null/permutation_null_seed4/random` — 0.4377
- `adni_permutation_null/permutation_null_seed4/subject_only` — 0.5403
- `adni_provenance_degradation/seed0_del0/component_safe` — 0.7679
- `adni_provenance_degradation/seed0_del0/subject_only` — 0.7679
- `adni_provenance_degradation/seed0_del10/component_safe` — 0.8208
- `adni_provenance_degradation/seed0_del10/subject_only` — 0.8311
- `adni_provenance_degradation/seed0_del100/component_safe` — 0.9404
- `adni_provenance_degradation/seed0_del100/subject_only` — 0.9565
- `adni_provenance_degradation/seed0_del25/component_safe` — 0.9291
- `adni_provenance_degradation/seed0_del25/subject_only` — 0.8332
- `adni_provenance_degradation/seed0_del50/component_safe` — 0.8901
- `adni_provenance_degradation/seed0_del50/subject_only` — 0.9100
- `adni_provenance_degradation/seed1_del0/component_safe` — 0.6945
- `adni_provenance_degradation/seed1_del0/subject_only` — 0.6945
- `adni_provenance_degradation/seed1_del10/component_safe` — 0.8351
- `adni_provenance_degradation/seed1_del10/subject_only` — 0.9291
- `adni_provenance_degradation/seed1_del100/component_safe` — 0.9812
- `adni_provenance_degradation/seed1_del100/subject_only` — 0.9403
- `adni_provenance_degradation/seed1_del25/component_safe` — 0.8769
- `adni_provenance_degradation/seed1_del25/subject_only` — 0.8017
- `adni_provenance_degradation/seed1_del50/component_safe` — 0.8898
- `adni_provenance_degradation/seed1_del50/subject_only` — 0.9122
- `adni_provenance_degradation/seed2_del0/component_safe` — 0.8017
- `adni_provenance_degradation/seed2_del0/subject_only` — 0.8017
- `adni_provenance_degradation/seed2_del10/component_safe` — 0.8429
- `adni_provenance_degradation/seed2_del10/subject_only` — 0.8420
- `adni_provenance_degradation/seed2_del100/component_safe` — 0.8907
- `adni_provenance_degradation/seed2_del100/subject_only` — 0.9128
- `adni_provenance_degradation/seed2_del25/component_safe` — 0.8507
- `adni_provenance_degradation/seed2_del25/subject_only` — 0.8810
- `adni_provenance_degradation/seed2_del50/component_safe` — 0.8663
- `adni_provenance_degradation/seed2_del50/subject_only` — 0.8847
- `adni_provenance_degradation/seed3_del0/component_safe` — 0.7632
- `adni_provenance_degradation/seed3_del0/subject_only` — 0.7632
- `adni_provenance_degradation/seed3_del10/component_safe` — 0.7941
- `adni_provenance_degradation/seed3_del10/subject_only` — 0.8578
- `adni_provenance_degradation/seed3_del100/component_safe` — 0.9434
- `adni_provenance_degradation/seed3_del100/subject_only` — 0.9581
- `adni_provenance_degradation/seed3_del25/component_safe` — 0.8938
- `adni_provenance_degradation/seed3_del25/subject_only` — 0.8279
- `adni_provenance_degradation/seed3_del50/component_safe` — 0.9728
- `adni_provenance_degradation/seed3_del50/subject_only` — 0.9238
- `adni_provenance_degradation/seed4_del0/component_safe` — 0.8273
- `adni_provenance_degradation/seed4_del0/subject_only` — 0.8273
- `adni_provenance_degradation/seed4_del10/component_safe` — 0.8541
- `adni_provenance_degradation/seed4_del10/subject_only` — 0.8673
- `adni_provenance_degradation/seed4_del100/component_safe` — 0.9214
- `adni_provenance_degradation/seed4_del100/subject_only` — 0.9647
- `adni_provenance_degradation/seed4_del25/component_safe` — 0.8375
- `adni_provenance_degradation/seed4_del25/subject_only` — 0.9069
- `adni_provenance_degradation/seed4_del50/component_safe` — 0.8517
- `adni_provenance_degradation/seed4_del50/subject_only` — 0.8575
- `adni_size_balanced/inflation_gap_seed0/component_safe` — 0.7541
- `adni_size_balanced/inflation_gap_seed0/random` — 0.9101
- `adni_size_balanced/inflation_gap_seed0/subject_only` — 0.8301
- `adni_size_balanced/inflation_gap_seed1/component_safe` — 0.7949
- `adni_size_balanced/inflation_gap_seed1/random` — 0.9478
- `adni_size_balanced/inflation_gap_seed1/subject_only` — 0.8645
- `adni_size_balanced/inflation_gap_seed2/component_safe` — 0.8887
- `adni_size_balanced/inflation_gap_seed2/random` — 0.9497
- `adni_size_balanced/inflation_gap_seed2/subject_only` — 0.7847
- `adni_size_balanced/inflation_gap_seed3/component_safe` — 0.7876
- `adni_size_balanced/inflation_gap_seed3/random` — 0.9528
- `adni_size_balanced/inflation_gap_seed3/subject_only` — 0.8345
- `adni_size_balanced/inflation_gap_seed4/component_safe` — 0.8318
- `adni_size_balanced/inflation_gap_seed4/random` — 0.9756
- `adni_size_balanced/inflation_gap_seed4/subject_only` — 0.8724

## Verdict

36 run(s) moved by at least 0.010 AUROC. Each one has to be traced into the manuscript before submission: the per-seed tables, the bootstrap intervals derived from them, and any sentence that interprets the affected arm.

| run | published | regenerated | delta |
|---|---|---|---|
| `adni/inflation_gap_seed2/component_safe` | 0.7675 | 0.8511 | +0.0836 |
| `adni_with_converters/inflation_gap_seed1/subject_only` | 0.6661 | 0.7497 | +0.0836 |
| `adni_with_converters/inflation_gap_seed4/subject_only` | 0.7719 | 0.7058 | -0.0661 |
| `adni/inflation_gap_seed1/subject_only` | 0.8030 | 0.8645 | +0.0615 |
| `adni_with_converters/inflation_gap_seed4/component_safe` | 0.7756 | 0.8324 | +0.0568 |
| `adni/inflation_gap_seed4/component_safe` | 0.8684 | 0.8119 | -0.0565 |
| `adni_no_mt1/inflation_gap_seed2/component_safe` | 0.8435 | 0.7919 | -0.0516 |
| `adni_densenet121/inflation_gap_seed4/subject_only` | 0.9361 | 0.8874 | -0.0487 |
| `adni_with_converters/inflation_gap_seed2/component_safe` | 0.8664 | 0.8184 | -0.0480 |
| `adni_no_mt1/inflation_gap_seed0/subject_only` | 0.8000 | 0.8417 | +0.0417 |
| `adni_with_converters/inflation_gap_seed3/subject_only` | 0.7057 | 0.7450 | +0.0393 |
| `adni_with_converters/inflation_gap_seed0/component_safe` | 0.7440 | 0.7805 | +0.0365 |
| `adni_densenet121/inflation_gap_seed1/component_safe` | 0.8731 | 0.8373 | -0.0358 |
| `adni_densenet121/inflation_gap_seed4/component_safe` | 0.8284 | 0.7981 | -0.0303 |
| `adni_no_mt1/inflation_gap_seed1/subject_only` | 0.8126 | 0.8424 | +0.0298 |
| `adni_no_mt1/inflation_gap_seed1/component_safe` | 0.8229 | 0.8524 | +0.0295 |
| `adni/inflation_gap_seed1/component_safe` | 0.8254 | 0.8546 | +0.0292 |
| `adni/inflation_gap_seed3/subject_only` | 0.8609 | 0.8345 | -0.0264 |
| `adni_densenet121/inflation_gap_seed3/subject_only` | 0.8135 | 0.8386 | +0.0251 |
| `adni_with_converters/inflation_gap_seed1/component_safe` | 0.8235 | 0.7992 | -0.0243 |
| `adni_no_mt1/inflation_gap_seed3/random` | 0.9836 | 0.9603 | -0.0233 |
| `adni_with_converters/inflation_gap_seed0/subject_only` | 0.8363 | 0.8588 | +0.0225 |
| `adni_densenet121/inflation_gap_seed1/subject_only` | 0.7849 | 0.8053 | +0.0204 |
| `adni/inflation_gap_seed0/subject_only` | 0.8496 | 0.8301 | -0.0195 |
| `adni_densenet121/inflation_gap_seed0/component_safe` | 0.8372 | 0.8186 | -0.0186 |
| `adni/inflation_gap_seed2/subject_only` | 0.8017 | 0.7847 | -0.0170 |
| `adni_with_converters/inflation_gap_seed3/component_safe` | 0.8614 | 0.8446 | -0.0168 |
| `adni_densenet121/inflation_gap_seed2/component_safe` | 0.8075 | 0.8223 | +0.0148 |
| `adni/inflation_gap_seed3/random` | 0.9670 | 0.9528 | -0.0142 |
| `adni_no_mt1/inflation_gap_seed4/subject_only` | 0.8633 | 0.8766 | +0.0133 |
| `adni_densenet121/inflation_gap_seed0/subject_only` | 0.8327 | 0.8194 | -0.0133 |
| `adni_densenet121/inflation_gap_seed3/random` | 0.9688 | 0.9558 | -0.0130 |
| `adni_with_converters/inflation_gap_seed3/random` | 0.9653 | 0.9783 | +0.0130 |
| `adni_densenet121/inflation_gap_seed3/component_safe` | 0.8196 | 0.8079 | -0.0117 |
| `adni_no_mt1/inflation_gap_seed2/random` | 0.9739 | 0.9622 | -0.0117 |
| `adni/inflation_gap_seed4/random` | 0.9650 | 0.9756 | +0.0106 |

Re-derive every downstream artefact from the regenerated runs (bootstraps, operating points, cost-of-leakage, subject-level aggregation) rather than editing the manuscript by hand, then run `scripts/verify_paper_numbers.py` to confirm the two agree.
