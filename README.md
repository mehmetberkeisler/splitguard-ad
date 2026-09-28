# SplitGuard-AD

> **Paper.** "Data Provenance and Evaluation Validity in Alzheimer's Disease
> MRI Deep Learning: Controlled Leakage and Provenance-Degradation
> Experiments (SplitGuard-AD)."
> Mehmet Berke Isler, Irem Ilter, Mehmet Kemal Ozdemir. 2026. Under review.
> Licence: Apache-2.0.

A leakage-audit and prevention framework for Alzheimer's disease MRI deep
learning benchmarks. SplitGuard-AD reconstructs subject, session,
near-duplicate, and longitudinal edges in the data; emits a frozen
component-safe train / validation / test partition; and turns leakage
detection into a reproducible pre-training gate with a binary GO/NO-GO
decision rule.

## Citation

A peer-reviewed description is under review. Pending publication, cite the
repository:

```bibtex
@misc{isler2026splitguardad,
  author = {Isler, Mehmet Berke and Ilter, Irem and Ozdemir, Mehmet Kemal},
  title  = {Data Provenance and Evaluation Validity in {Alzheimer's} Disease
            {MRI} Deep Learning: Controlled Leakage and
            Provenance-Degradation Experiments ({SplitGuard-AD})},
  year   = {2026},
  url    = {https://github.com/mehmetberkeisler/splitguard-ad}
}
```

The paper rests on two source cohorts and one redistribution of one of them:

- **Tier 1** — the widely used public Kaggle 2D JPEG benchmark. Content
  matching shows it to be 200 OASIS-1 participants at 32 axial slices each,
  with folder labels equal to CDR and participant identity replaced by a
  filename index that merges and splits participants. It is the paper's
  observed case of provenance damage.
  `scripts/audit_tier1_ground_truth.py` and
  `scripts/tier1_split_rules_vs_truth.py` recover the identities and measure
  every splitting rule against them from your own Kaggle and OASIS-1 copies.
- **Tier 2** — OASIS-1 (cross-sectional clinical-research cohort, true
  subject and session metadata, CDR labels).
- **Tier 3** — ADNI1: Complete 3Yr 1.5T (longitudinal clinical-research
  cohort, visit-level diagnosis through DXSUM, multi-phase coverage).

## Evidence streams

Beyond the inflation gap itself, the paper reports:

- **Provenance damage, controlled and observed** — identifier deletion at
  controlled rates on ADNI (`scripts/run_provenance_degradation.py`) and the
  recovered ground truth of the Tier-1 redistribution.
- **Leakage dose-response** — AUROC as a function of injected test-set
  participant overlap (`scripts/inject_leakage_split.py`,
  `scripts/analyze_dose_response.py`).
- **Clinical cost of leakage** — missed diagnoses per 1,000 screened at a
  fixed specificity and literature-cited prevalence anchors
  (`scripts/clinical_cost_of_leakage_adni.py`).
- **Subgroup disparity that leakage hides** — sex-stratified AUROC under each
  protocol (`scripts/subgroup_analysis_adni.py`).
- **Identity decodability** — a linear probe on frozen features, with image
  and whole-session hold-out (`scripts/run_biometric_probe_adni.py`).

The numbers live in the manuscript and in `reports/`, and
`scripts/verify_paper_numbers.py` holds one to the other; this README does
not repeat them, so it cannot go stale.

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

### Deciding how to evaluate, before you split

`design_experiment.py` reads a manifest and reports what evaluation protocol
the cohort can actually support: which leakage channels its identifiers can
close and which stay open, whether the leakage graph reaches participant
granularity, and what a naive image-level split would cost at minimum.

```bash
python3 scripts/design_experiment.py --manifest my_cohort.csv
```

The cost figure is a floor rather than a forecast. It prices the
participant-identity channel through the dose-response calibration, which was
fitted by varying test-subject overlap with the training set held fixed, so
slice-level and near-duplicate leakage are not in it. Against the two cohorts
where the total gap was measured it behaves accordingly: +0.125 predicted
against +0.142 measured on the converter-inclusive ADNI1 arm the calibration
was fitted on, and against +0.233 on the Tier-1 benchmark, where adjacent
slices of one volume add a channel the slope does not price. On the
converter-excluding ADNI1 arm the measured gap is +0.118, so there the floor is
tight rather than conservative. Use it to decide whether leakage is worth
worrying about on your data, not to predict your own number.

### Bringing your own cohort

The Tier-1 scripts are not specific to the benchmark they were written
for. Given a manifest CSV, they build the leakage graph, emit a frozen
component-safe split, and write an audit report. `image_id` is the only
column strictly required; the rest are used when present and default to
empty when absent.

| Column | Used for |
| --- | --- |
| `image_id` | **Required.** Row key. |
| `subject_id` | `same_subject` edges. Without it every image is its own component and the split degrades to a random one. |
| `session_id` | `same_session` edges. Independent of `subject_id`, so it still binds scans when the patient index is missing. |
| `relative_path` | Deterministic row ordering, and the near-duplicate hash when the file exists on disk. |
| `raw_class_label` | Class-stratified assignment. Any label set works; the four benchmark classes are stratified first, anything else after, in sorted order. |
| `binary_label` | Reported class balance. |

```bash
# 1. Leakage graph: union-find over the identifier columns present,
#    plus near-duplicate detection when relative_path resolves to a file.
python3 scripts/build_current_leakage_graph.py \
    --manifest my_cohort.csv \
    --components my_components.csv \
    --summary-json my_graph_summary.json

# 2. Component-safe split. Byte-reproducible for a given seed.
python3 scripts/make_current_splitguard_split.py \
    --manifest my_cohort.csv \
    --components my_components.csv \
    --output my_split_seed42.csv \
    --audit my_split_audit.md \
    --seed 42
```

The audit reports partition overlap by subject, session and component;
class balance; and the component-size distribution per partition. That
last one is worth reading rather than skimming. Partitions that are
disjoint can still be drawn from different kinds of participant: the
splitter fills validation and test first with the largest components,
which on a longitudinal cohort routes participants with longer records
away from training. Our own frozen ADNI manifests trip the warning, and
the Limitations section of the accompanying paper explains why it is
reported rather than enforced.

Not covered here: reading DICOM or NIfTI and building the manifest in
the first place. `scripts/build_adni_manifest.py` and
`scripts/build_current_dataset_manifest.py` do that for the two cohorts
in the paper and are worth reading as templates, but a new cohort needs
its own ingest.

### Tier 1 — Redistributed public benchmark

```bash
python3 scripts/build_current_dataset_manifest.py
python3 scripts/build_current_leakage_graph.py
python3 scripts/make_current_splitguard_split.py
# Recover participant identity against your OASIS-1 copy, then measure every
# grouping rule against it and write the filename-key and true-participant splits.
python3 scripts/audit_tier1_ground_truth.py
python3 scripts/tier1_split_rules_vs_truth.py
python3 scripts/run_inflation_gap.py --split data/splits/tier1_truth/tier1_true_participant_seed42.csv \
    --safe-label true_participant --epochs 30
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

# DUA-safe release: emit a hashed component-safe manifest for Tier 3
# (PBKDF2-HMAC-SHA256 of subject IDs at 600,000 iterations under a
# secret per-release salt, plus component_size, binary label, and
# non-identifying acquisition fields; participant-level identifiers are
# dropped). Key stretching rather than a plain digest is required, not
# merely prudent: ADNI PTIDs come from a space of roughly 10^7, so a
# single-round hash of one is invertible by enumeration in under a
# second. The salt is read from SPLITGUARD_RELEASE_SALT and has no
# default. A downstream auditor can verify component-safety, class
# balance, and site-level Cramer's V without holding ADNI data.
python3 scripts/build_hashed_manifest_tier3.py \
    --split data/splits/adni/adni_splitguard_seed0.csv \
    --output data/splits/adni_hashed_manifest_seed0.csv
```

#### Reproducing every training result on one GPU

Every AUROC in the manuscript comes from `scripts/gpu_program.py` in one
code state: the Tier-1 recovered-participant protocols, every ADNI arm,
OASIS-1 under both backbones, the permutation null, the size-balanced
control, both identity probes, the provenance AUROC arm and the
dose-response matrix. All of them but the dose-response matrix ran on one
H100; the matrix was rerun in the same code state on a second node, and the
paper says so. One stage the programme can run is still not reported: the
volumetric arm, whose cost is measured in `docs/GPU_RUNBOOK.md` section 5. The programme skips anything
already done, stops before a command that would exceed the cap in dollars or
minutes, runs commands concurrently with `--workers`, and packs predictions
and metrics (never weights) for download. `docs/GPU_RUNBOOK.md` is the full procedure, including the ADNI
Data Use Agreement steps.

```bash
python3 scripts/pack_gpu_bundle.py --volumes-root /path/to/ADNI1_preprocessed_128  # on your machine
bash scripts/gpu_setup.sh                                                      # on the GPU node
python scripts/gpu_program.py --smoke
python scripts/gpu_program.py --usd-per-hour 0.69 --budget-usd 10
bash scripts/rebuild_after_gpu.sh                                              # back on your machine
```

`rebuild_after_gpu.sh` promotes the downloaded runs, reruns every analysis
(bootstraps, dose-response fits, cost of leakage, subgroups, operating points),
regenerates the tables, figures and the manuscript's number file, compares
against the previous runs, and runs `scripts/verify_paper_numbers.py`.

## Regenerating the figures

`requirements.txt` pins `numpy==1.26.4` and `matplotlib==3.9.2`, and the figures
in `paper/` were produced under those versions. Matplotlib renders text and
antialiasing slightly differently across releases, so a figure regenerated under
a different matplotlib is visually equivalent but not byte-identical to the one
committed. Install the pinned versions before regenerating if you need the
committed files reproduced exactly:

```bash
pip install -r requirements.txt
python3 scripts/generate_leakage_taxonomy_figure.py
python3 scripts/generate_figures.py
python3 scripts/generate_cross_cohort_figure.py
python3 scripts/generate_arm_forest_figure.py
python3 scripts/generate_operating_point_figure.py
python3 scripts/generate_dose_response_figure.py
python3 scripts/generate_provenance_figure.py
python3 scripts/generate_cost_of_leakage_figure.py
python3 scripts/generate_subgroup_figure.py
python3 scripts/generate_adni_paper_tables.py
```

Figure files are numbered by the position the figure takes in the manuscript,
so `paper/fig07_arm_forest.pdf` is Figure 7. Figures 2 and 3 are drawn inline
in TikZ and have no file. `paper/figS1_subgroup_auroc.pdf` belongs to the
supplementary series.

The numbers a figure draws come from `reports/`, not from the figure script, so
`scripts/verify_paper_numbers.py` holds the manuscript to the same artefacts
regardless of which matplotlib drew them.

## Building the manuscript

The manuscript targets the Journal of Imaging Informatics in Medicine and uses
Springer Nature's `sn-jnl` class. Springer distributes that class under its own
terms, so it is fetched rather than redistributed here:

```bash
curl -sL -o sn.zip "https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12"
unzip -j sn.zip 'sn-article-template/sn-jnl.cls' 'sn-article-template/bst/*.bst' -d paper/
```

Then build and check both documents:

```bash
python3 scripts/verify_paper_numbers.py          # every number against its artefact
python3 scripts/validate_submission.py --build   # structure, declarations, refs, margins
```

`validate_submission.py` encodes the journal's published requirements: abstract
length, keyword count, section order, the declarations it asks for, LLM use
documented in the Methods, a reference list that resolves both ways, no float
that the text never mentions, no number still rendering as `??`, and no line
running into the margin in either document.

## Scripts that support work outside this paper

Four scripts in `scripts/` produce results the manuscript does not report, and
are kept because the manuscript names each as future work:

| Script | What it would add |
| --- | --- |
| `run_adni_permutation_null.py` | Label-permutation null control. Permutes diagnosis labels per participant within each partition, giving an empirical noise floor for a test partition of this size and a positive control for whether leakage alone can manufacture a gap. |
| `run_provenance_degradation.py` | The provenance-degradation sweep: the structural arm and, at each deletion level, a trained AUROC arm. Both are reported in the paper. |
| `compare_reruns.py` | Compares a regenerated run tree against the published one, arm by arm, and flags any AUROC that moved by 0.01 or more. |
| `train_adni_3d.py`, `preprocess_adni1_to_3d.py` | A 3D volumetric arm. The paper states repeatedly that all of its findings are properties of a 2D coronal-centre-slice representation and that a volumetric replication is the highest-priority architectural follow-on. |

`train_adni_3d.py` is the one arm `scripts/gpu_program.py` can run but the
paper does not report: it needs the 128^3 volume bundle on the node, which the
reported run did not have time to upload.
Treat their outputs as unpublished.

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
├── paper/                 # manuscript sources, figures, generated tables
├── release/               # per-tier redistributable manifests
├── tests/                 # contract tests
└── docs/
    ├── ADNI_LABEL_ONTOLOGY.md  # phase-aware ADNI label resolution rule
    ├── DATA_ACCESS.md          # pointers to each tier's data provider
    └── GPU_RUNBOOK.md          # the single-GPU reproduction procedure
```

## What is released, and what is not

Released here: every script, the manuscript and supplementary sources with
their figures, the generated number and table files the manuscript reads,
aggregate result tables, the leakage-graph component tables, the Tier-1 and
Tier-2 split manifests, hashed Tier-3 manifests, and the tests.

Kept local, deliberately: raw and preprocessed images, per-image predictions,
model checkpoints, the ADNI download inventory, the Tier-1 image-to-OASIS-1
mapping, and the per-release hashing salt. ADNI's Data Use Agreement does not
permit redistributing participant-level data even in processed form, model
weights trained on it carry the same restriction, and the Tier-1 mapping is
regenerable from a user's own Kaggle and OASIS-1 downloads with
`scripts/audit_tier1_ground_truth.py`. Everything withheld is reproducible
from the released code plus the data each provider grants directly.


## License

Apache License 2.0. See `LICENSE` for the full text.
