# Data Access and Rebuilding Policy

This repository does not redistribute MRI images, OASIS archives,
extracted Analyze/NIfTI volumes, derived slice images, subject/session
manifests, or per-image prediction files.

The public release contains code, paper sources, aggregate result
summaries, figures, and audit summaries. Dataset users must obtain the
underlying data from the original providers and rebuild local manifests
and splits with the scripts in this repository.

## Public JPEG Benchmark

The public JPEG benchmark is treated as a weak-label stress test for
evaluation leakage. Users should download it from the original public
source and place it locally according to the paths expected by the
manifest-building scripts.

Rebuild locally:

```bash
python3 scripts/build_current_dataset_manifest.py
python3 scripts/build_current_leakage_graph.py
python3 scripts/make_current_splitguard_split.py
```

## OASIS-1

OASIS-1 data should be obtained directly from the official OASIS access
site. This project uses the raw cross-sectional archives plus the
demographic/clinical and reliability spreadsheets as local inputs.

Rebuild locally:

```bash
python3 scripts/build_oasis1_pipeline.py --extract --slices
```

The OASIS script selectively extracts only one processed masked T88 volume
pair per session and avoids full archive expansion.

## What Is Safe to Publish

- Source code and experiment scripts.
- Paper LaTeX source and generated aggregate figures.
- Aggregate JSON metrics and audit summaries.
- Documentation explaining how to obtain and rebuild data locally.

## What Is Not Published

- Raw MRI images or archives.
- Extracted OASIS `.img` / `.hdr` files.
- Derived slice images.
- Subject/session manifests and split CSVs.
- Per-image or per-slice prediction CSVs.
- Model checkpoints.

