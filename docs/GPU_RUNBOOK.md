# GPU run: one code state, one budget

Every training result in the manuscript is regenerated on one GPU node, in one
code state, with per-image predictions persisted. This file is the whole
procedure: what to upload, what to run, what to bring back, and what to delete.

ADNI-derived images are covered by the LONI Data Use Agreement. They may be
processed on a rented node, but the node must be a **RunPod Secure Cloud**
instance (or an equivalent single-tenant provider), the data must be deleted
from it before the pod is terminated, and **no model weights are published**,
because weights trained on participant-level data are themselves a
redistribution risk. The results archive this run produces contains
predictions, metrics and logs; it never contains weights.

## 1. What runs, and what it costs

`scripts/gpu_program.py` runs the stages below in order, skips anything whose
outputs already exist, and stops before starting a command that would push the
bill past the cap. Timings are measured CUDA wall times of these same scripts,
with margin; the programme re-measures on the node and uses its own timings
from the second command of each kind onwards.

| Stage | What it answers | Trainings |
|---|---|---|
| `tier1_truth` | Tier 1 under random, filename grouping and true participants | 15 |
| `adni_primary`, `adni_converters`, `adni_no_mt1`, `adni_densenet121` | the headline gap and its robustness arms | 60 |
| `tier2_oasis1` | OASIS-1 replication, both backbones | 20 |
| `adni_null` | label-permutation null | 15 |
| `adni_size_balanced` | gap under a size-balanced component assignment | 15 |
| `identity_probe` | identity decodability, per image and per session | 2 probes |
| `adni_provenance` | protection under identifier deletion | 50 |
| `adni_3d` | the gap on 128³ volumes, not 2D slices | 15 |
| `adni_dose_response` | AUROC per unit injected overlap | 50 |

At $0.69/h (Secure Cloud RTX 4090) the conservative projection for all of it
is about **$5.3** including the volumetric arm, inside a $10 budget with a
$1.50 reserve for start-up, setup, upload and download. `--list` prints the
current projection; timings are re-measured on the node as the run proceeds.

## 2. On your machine, before renting anything

```bash
# Only needed once, and only for the volumetric arm: give each frozen split
# the three protocol columns and drop scans whose volume was never downloaded.
python3 scripts/make_protocol_split_columns.py \
    --volumes-root /path/to/ADNI1_preprocessed_128

python3 scripts/pack_gpu_bundle.py \
    --volumes-root /path/to/ADNI1_preprocessed_128
```

This writes `gpu_bundle_2d.tar` (about 180 MB: every split file and every
image they reference) and `gpu_bundle_3d.tar` (about 6.3 GB: the 1,018
preprocessed volumes the volumetric arm needs) and prints a SHA-256 for each.
Skip `--volumes-root` to leave the volumetric arm out.

Upload of the 6.3 GB bundle is billed as pod time, so start the upload
immediately after the pod boots and run `scripts/gpu_setup.sh` while it
transfers.

## 3. On the pod

```bash
git clone https://github.com/mehmetberkeisler/splitguard-ad.git
cd splitguard-ad
bash scripts/gpu_setup.sh              # venv, pinned wheels, CUDA check, environment record
source .venv-gpu/bin/activate

tar -xf /workspace/gpu_bundle_2d.tar   # extracts into data/ and the image folders
tar -xf /workspace/gpu_bundle_3d.tar   # optional: data/adni3d_128/

python scripts/gpu_program.py --smoke   # one epoch, one seed per stage, a few minutes
python scripts/gpu_program.py --list
python scripts/gpu_program.py --usd-per-hour 0.69 --budget-usd 10
```

`--smoke` is not optional in practice: it exercises every stage's data paths
and imports for the price of a few minutes, and a stage that fails there would
otherwise fail after an hour of billed training.

The full run prints one line per command and writes
`runs_gpu/gpu_ledger.json` (billed seconds so far) and
`runs/gpu_program_log.jsonl` (per-command status, minutes and spend). It is
safe to interrupt: rerun the same command and it resumes at the first command
whose outputs are missing, carrying the ledger forward.

If the programme reports `skipped_budget` or `stopped_at_cap`, the remaining
stages are the lowest-priority ones by design. Either accept that, or rerun
with a higher `--budget-usd` after checking the provider's own meter.

## 4. Bring the results back, then delete the data

```bash
# on the pod, at the end of the run
ls -lh gpu_results.tar.gz
```

Download `gpu_results.tar.gz` (predictions, metrics, logs, summaries; no
weights), then, still on the pod:

```bash
rm -rf data Alzheimer_MRI_4_classes_dataset oasis1 runs_gpu/*/*/*/best_state.pt \
       runs/checkpoints /workspace/gpu_bundle_*.tar
```

and terminate the pod, which releases its disk. Both steps are required by the
Data Use Agreement: deleting the archive is not enough if the pod's volume
survives.

## 5. The two stages this release does not report

The published run covers every stage except two, and both are one command on
the next node.

**The volumetric arm** (15 trainings, about 15 minutes at `--workers 12`,
under $1). It is the only stage that needs the 6.3 GB volume bundle, which at
a typical home upstream takes far longer to transfer than to train, so start
the upload first and run everything else while it lands:

```bash
# once, on your machine (needs the drive holding ADNI1_preprocessed_128)
python3 scripts/make_protocol_split_columns.py --volumes-root /path/to/ADNI1_preprocessed_128
python3 scripts/pack_gpu_bundle.py --volumes-root /path/to/ADNI1_preprocessed_128
scp -P <port> gpu_bundle_3d.tar root@<host>:/workspace/     # ~6.3 GB, start it early

# on the node, after the bundle lands
tar -xf /workspace/gpu_bundle_3d.tar        # extracts into data/adni3d_128/
python scripts/gpu_program.py --only adni_3d --workers 12 --time-budget-min 30
```

`--preprocessed-root` defaults to `data/adni3d_128`, so nothing else changes.
Each volumetric command takes four scheduling slots, because a $128^3$ batch
costs several times a slice batch. When the results come back,
`scripts/rebuild_after_gpu.sh` fills `\VolGapTotal` and `\VolGapCI`, and the
Limitations paragraph that currently calls the arm future work has to be
rewritten to report it.

**The dose-response matrix** (50 trainings, about 20 minutes at
`--workers 12`). The released numbers come from the earlier CUDA sweep, which
is why the Acknowledgements name it as the one exception to the single code
state. Rerunning it on the same node as everything else removes that
exception:

```bash
python scripts/gpu_program.py --only adni_dose_response --workers 12 --time-budget-min 30
```

Do not run this one on a different backend than the rest: a third hardware
backend would replace one stated exception with another. Run it beside the
volumetric arm on one node, or leave it as it is.

## 6. Back on your machine

```bash
tar -xf gpu_results.tar.gz
python3 scripts/gpu_postprocess.py                 # summary JSON + LaTeX macros
python3 scripts/compare_reruns.py --old runs --new runs_gpu \
        --markdown reports/rerun_comparison.md     # what moved against the frozen numbers
python3 scripts/verify_paper_numbers.py            # every manuscript number against its artefact
```

`gpu_postprocess.py` writes `reports/gpu/summary.json` and
`paper/tables/gpu_numbers.tex`. The manuscript reads its GPU-dependent numbers
from those macros, so no value is transcribed by hand, and any number whose run
did not complete renders as `??` instead of silently keeping a stale value.
