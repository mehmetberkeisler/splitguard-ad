# SplitGuard-AD

> **Paper.** "SplitGuard-AD: A Leakage-Audit and Prevention Framework for
> Alzheimer's Disease MRI Deep Learning Benchmarks."
> Mehmet Berke Isler, Irem Ilter, Mehmet Kemal Ozdemir. 2026.
> Licence: Apache-2.0.

A leakage-audit and prevention framework for Alzheimer's disease MRI deep
learning benchmarks. SplitGuard-AD reconstructs subject, session,
near-duplicate, and longitudinal edges in the data; emits a frozen
component-safe train / validation / test partition; and turns leakage
detection into a reproducible pre-training gate with a binary GO/NO-GO
decision rule.

## Citation

If you use SplitGuard-AD in your work, please cite:

```bibtex
@article{isler2026splitguardad,
  title   = {{SplitGuard-AD}: A Leakage-Audit and Prevention Framework for
             {Alzheimer's} Disease {MRI} Deep Learning Benchmarks},
  author  = {Isler, Mehmet Berke and Ilter, Irem and Ozdemir, Mehmet Kemal},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
  note    = {Submitted to Medical Image Analysis}
}
```

The framework is designed around three concrete validation cohorts:

- **Tier 1** — a public 2D JPEG benchmark (worst-case scenario, inferred
  pseudo-subjects from filename conventions).
- **Tier 2** — OASIS-1 (cross-sectional clinical-research cohort, true
  subject and session metadata, CDR labels).
- **Tier 3** — ADNI1: Complete 3Yr 1.5T (longitudinal clinical-research
  cohort, visit-level diagnosis through DXSUM, multi-phase coverage).

## Evidence streams

Beyond the basic inflation-gap measurement, the framework includes
three further evidence streams used in the paper:

- **Sex-related performance disparity that leakage hides** — paired-bootstrap
  female-minus-male AUROC delta of +0.106 to +0.109 under both honest
  protocols, indistinguishable from zero under the leaky benchmark.
  Class-prevalence control rules out the most obvious confound; age × sex
  interaction localises the gap to the older male stratum.
  See `scripts/subgroup_analysis_adni.py`.
- **Clinical cost-of-leakage translation** — converts the AUROC inflation
  into missed-diagnosis counts per 1000 screened patients at literature-
  cited prevalence anchors (Rajan 2021 US 75-84 stratum = 13.8%; PROMPT
  2024 tertiary memory clinic = 58.9%). Headline: ~3.2× understatement of
  clinical harm at the screening operating point.
  See `scripts/clinical_cost_of_leakage_adni.py`.
- **Leakage dose-response stress test** — controlled 5×5×2 subject-
  substitution matrix on ADNI yields a near-linear AUROC-vs-overlap
  relationship: AUROC ≈ 0.832 + 0.106·p on ResNet-18 (R²=0.69). Every 10pp
  of test-subject overlap inflates reported AUROC by ~0.011.
  See `scripts/inject_leakage_split.py` + `scripts/run_dose_response_matrix.sh`.

Plus infrastructure: a `DataLoader num_workers` fix that's critical on CUDA
(67× speedup) and a one-shot RunPod setup script
(`scripts/runpod_setup_and_run.sh`) for the dose-response matrix on a
rented 4090 (~$2, ~1h40min wall-clock).

## Quick start

The framework targets Python 3.10+. PyTorch with the MPS or CUDA backend
is recommended.

```bash
git clone https://github.com/mehmetberkeisler/splitguard-ad
cd splitguard-ad
pip install -r requirements.txt
```

Data is not redistributed; see `docs/DATA_ACCESS.md` for how to obtain
each tier from its original provider under that provider's licence
terms.

### Tier 1 — Public JPEG benchmark

```bash
python3 scripts/build_current_dataset_manifest.py
python3 scripts/build_current_leakage_graph.py
python3 scripts/make_current_splitguard_split.py
python3 scripts/run_inflation_gap.py --epochs 30
```

### Tier 2 — OASIS-1

```bash
python3 scripts/build_oasis1_pipeline.py --extract --slices
python3 scripts/run_oasis1_inflation_gap.py --epochs 20 --seed 42
```

### Tier 3 — ADNI1: Complete 3Yr 1.5T

Drop the 10 LONI IDA archive zips into `data/raw/adni/downloads/` and
the clinical CSVs (DXSUM, PTDEMOG, MMSE, DATADIC, ROSTER, REGISTRY,
VISITS, MRIMETA, MRI3META, MRIQC) into `data/raw/adni/study_files/`.
See `docs/ADNI_LABEL_ONTOLOGY.md` for the exact column names assumed by
the manifest builder.

```bash
# Stream-extract each zip and cache the coronal-centre slice as a PNG
# (~54 MB total vs ~77 GB of NIfTI volumes).
python3 scripts/preprocess_adni_volumes_to_slices.py

# Gated pipeline: inventory -> manifest -> leakage graph -> 5-seed
# component-safe splits -> audit. Fails fast at any gate.
python3 scripts/run_adni_pipeline.py --skip-extract

# Three-protocol inflation-gap experiment (5 seeds, 15 epochs).
python3 scripts/run_adni_inflation_gap.py --seeds 0 1 2 3 4 --epochs 15

# Paired-seed and subject x seed hierarchical bootstrap.
python3 scripts/bootstrap_adni_inflation_gap.py
python3 scripts/hierarchical_bootstrap_adni.py

# Optional sensitivity arms.
python3 scripts/make_adni_splitguard_split.py \
    --include-mixed-by-majority \
    --split-dir data/splits/adni_with_converters
python3 scripts/run_adni_inflation_gap.py \
    --arch densenet121 \
    --output-root runs/adni_densenet121 \
    --seeds 0 1 2 3 4 --epochs 15

# Optional probes.
python3 scripts/subgroup_analysis_adni.py
python3 scripts/run_biometric_probe_adni.py
```

#### Follow-up experiments

The dose-response stress test and the clinical cost-of-leakage analysis
both run on the converter-inclusive ADNI sensitivity arm and reuse the
per-image test predictions that the patched training pipeline persists.

```bash
# Clinical cost-of-leakage: translates AUROC gap into missed-diagnosis
# counts per 1000 screened patients at literature-cited prevalence
# anchors. Uses the predictions written by run_adni_inflation_gap.py.
python3 scripts/clinical_cost_of_leakage_adni.py \
    --prev-pop 0.138 \
    --prev-pop-citation "Rajan et al. 2021, Alz&Dement, doi:10.1002/alz.12362" \
    --prev-clinic 0.589 \
    --prev-clinic-citation "Thomas G et al. 2024 PROMPT registry (Alz&Dement), PMC11716007"
python3 scripts/generate_cost_of_leakage_figure.py

# Dose-response stress test: inject controlled test-subject overlap at
# five levels and fit AUROC = f(overlap). 50 training runs (5 seeds x
# 5 overlap levels x 2 archs) — typically ~25 hours on RTX 4090.
# A one-shot RunPod driver (scripts/runpod_setup_and_run.sh) handles
# environment setup, smoke test, full matrix, and result bundling.
for seed in 0 1 2 3 4; do
  for overlap in 0.0 0.25 0.50 0.75 1.0; do
    python3 scripts/inject_leakage_split.py \
        --base data/splits/adni_with_converters/adni_splitguard_seed${seed}.csv \
        --overlap $overlap --seed $seed \
        --output data/splits/adni_dose_response/seed${seed}_overlap${overlap}.csv
  done
done
bash scripts/run_dose_response_matrix.sh
python3 scripts/analyze_dose_response.py
python3 scripts/generate_dose_response_figure.py
```

## Repository layout

```
splitguard-ad/
├── LICENSE                # Apache-2.0
├── README.md
├── requirements.txt
├── scripts/               # Framework implementation
│   ├── build_*.py         # manifest builders (per tier)
│   ├── make_*_split.py    # frozen component-safe split generators
│   ├── run_*.py           # experiment runners (inflation gap, probes)
│   ├── bootstrap_*.py     # paired-seed and hierarchical bootstrap
│   ├── generate_*.py      # figure and table generators (work on local
│   │                      # output files; not bundled with the repo)
│   └── audit_*.py         # contamination / overlap audits
├── tests/                 # contract tests
└── docs/
    ├── ADNI_LABEL_ONTOLOGY.md  # phase-aware ADNI label resolution rule
    └── DATA_ACCESS.md          # pointers to each tier's data provider
```

The framework writes its intermediate artifacts (raw data, manifests,
splits, audits, per-image predictions, model checkpoints, figures,
tables, manuscript) to local directories that are intentionally
excluded from this repository. The release boundary is code only.

## Citation

A peer-reviewed description of the framework is in preparation. Pending
publication, please cite this repository:

```bibtex
@misc{splitguardad,
  author       = {Isler, Mehmet Berke and Ilter, Irem},
  title        = {{SplitGuard-AD}: A Leakage-Audit and Prevention
                  Framework for {A}lzheimer's Disease {MRI} Deep
                  Learning Benchmarks},
  year         = {2026},
  url          = {https://github.com/mehmetberkeisler/splitguard-ad}
}
```

## License

Apache License 2.0. See `LICENSE` for the full text.
