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
| `adni_null` | label-permutation positive control | 15 |
| `adni_size_balanced` | gap under a size-balanced component assignment | 15 |
| `identity_probe` | identity decodability, per image and per session | 2 probes |
| `adni_provenance` | protection under identifier deletion | 50 |
| `adni_3d` | the gap on 128³ volumes, not 2D slices | 15 |
| `adni_dose_response` | AUROC per unit injected overlap | 50 |
| `adni_exact_linkage` | the gap on exact-visit-key labels only | 15 |
| `adni_stress` | AUROC under each structured-corruption cell | 120 |

What it actually cost, summed across the four nodes these stages ran on, is
**14.8 GPU-hours** (`runs_gpu/gpu_ledger.json`), which at $0.69/h is about
**$10** if you run all of it on one RTX 4090. The volumetric arm is two thirds
of that on its own (see section 5), so budget it separately and leave it out
if you only need the 2D results: everything else fits inside $4. `--list`
prints the current projection before anything runs, and timings are
re-measured on the node as the run proceeds.

The ledger counts only the node it was written on, so carry it across nodes by
hand: add the `billed_seconds` together and concatenate the
`gpu_program_log.jsonl` files, or `\GpuHours{}` reports one node's slice as the
whole total.

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

One trap is worth stating before the commands: most rented images ship a
working CUDA `torch`, and installing anything that declares a torch dependency
(`monai` does) silently replaces it with a CPU wheel, after which every command
fails at launch. `scripts/gpu_setup.sh` keeps the image's build and installs
the rest with `--no-deps`, and `gpu_program.py` refuses to start if a required
module is missing or if `torch` can no longer see the GPU.

```bash
git clone https://github.com/mehmetberkeisler/splitguard-ad.git
cd splitguard-ad
bash scripts/gpu_setup.sh              # keeps the image's CUDA torch, adds the rest, records the environment
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
weights) and check it unpacks on your machine. Only then, on the pod, delete
the checkout and the bundle outright rather than pruning inside it: the run
tree holds per-image predictions keyed by ADNI image identifiers, which are as
much ADNI-derived data as the slices are.

```bash
rm -rf ~/splitguard /workspace/gpu_bundle_*.tar
find / -xdev \( -iname "*adni*" -o -iname "*splitguard*" -o -iname "gpu_bundle*" \) 2>/dev/null
```

The `find` must print nothing. Then terminate the pod, which releases its
disk. Both steps are required by the Data Use Agreement: deleting the archive
is not enough if the pod's volume survives.

## 5. The stages that needed their own node

Every stage this programme can run is now reported in the manuscript. Five of
them did not produce their reported result on the original H100: the
dose-response matrix, the volumetric arm, the exact-linkage arm, the
structured-corruption matrix, and `adni_null`, whose first run was superseded
when its permutation was corrected. The cost of that is three further hardware
backends, which the Declarations state.

The dose-response matrix was the first: it was rerun in the
published code state (50 trainings, 18 minutes billed at `--workers 10` on an
RTX PRO 4500, $0.21). What that rerun cost the manuscript is a second hardware
backend, which the Declarations now state.

**The volumetric arm** (15 trainings) was the expensive one, far
more expensive than an earlier estimate in this file claimed. Measured on an
RTX PRO 4500 (32 GB) at `--batch-size 4`, one 3D ResNet-18 step on a $128^3$
volume costs:

| configuration | ms/step | peak GPU memory | stage total |
|---|---|---|---|
| deterministic fp32 (what the trainer sets) | 633 | 9.2 GB allocated, 14.7 GB reserved | ~10 h |
| non-deterministic fp32 (`cudnn.benchmark`) | 270 | 9.2 GB | ~4 h |
| non-deterministic bf16 autocast | 167 | 5.0 GB | ~2.5 h |

So budget ten GPU-hours, not the fifteen minutes this section used to promise,
and expect one job at a time: two concurrent commands reserve 29 GB of a 32 GB
card and the second one dies in `batch_norm` with an out-of-memory error.
`SLOTS["adni_3d"] = 4` is therefore the floor, not a cushion — pass
`--workers 4`.

```bash
# once the volume bundle has landed and been extracted
python scripts/gpu_program.py --smoke --only adni_3d --workers 4   # exercises all three protocols
python scripts/gpu_program.py --only adni_3d --workers 4 --time-budget-min 700
```

Read the first command's `runs_gpu/logs/adni_3d_00.log` before letting the
stage run: on the `random` protocol validation AUROC has to climb toward the
leaky 2D level, because 100% subject overlap is the easy case. A validation
AUROC that falls as the training loss falls is the signature of a metric
orientation bug, which is how the one this trainer used to have was found.

When the results come back, `scripts/rebuild_after_gpu.sh` fills
`\VolGapTotal` and `\VolGapCI`, both of which the manuscript now uses.

**The exact-linkage arm** (15 trainings) and **the structured-corruption
matrix** (120 trainings) ran together on an RTX 4090, as did the re-run of
`adni_null` after its permutation was corrected. The corruption matrix has two
halves and only one needs a GPU: `run_provenance_stress_test.py --emit-splits`
measures the contamination each cell admits on a CPU in seconds and writes the
120 corrupted manifests, the stage trains one model per manifest, and
`analyze_provenance_stress_auroc.py` folds the trained half back in as a
seed-paired difference. Run that last script, and
`analyze_permutation_null.py`, after promoting the runs or the manuscript's
numbers will be checked against a half-filled artefact;
`rebuild_after_gpu.sh` does both for you.

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
