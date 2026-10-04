# Contributing

This is a reference implementation and audit pipeline for a published
experiment, not a general-purpose library, so the bar for a change is not "does
it work" but "can the published numbers still be reproduced after it".

## The test philosophy

> Every scientific bug that could change a published result gets a regression
> test, and the test is validated by reintroducing the bug.

The second half matters as much as the first. A guard that cannot fail is worse
than no guard, because it reads as coverage. Four guards added to this
repository in one sitting were vacuous on arrival: they searched raw LaTeX
source for phrases that the typesetter had wrapped across lines, so none of
them could ever match. They were only discovered by deliberately reintroducing a
phrase one of them forbade and watching the suite stay green.

So when you add a check, break the thing it checks and confirm it fails. The
mutation results quoted in the README were produced that way.

## What a change must not do

- Change a published number without changing the artefact it comes from.
  `scripts/verify_paper_numbers.py` holds 534 numeric claims to their sources
  and fails on any disagreement.
- Introduce a second implementation of something in `src/splitguard_ad/`. The
  package exists because the leakage graph had two missing-value rules and one
  of them was wrong.
- Widen an exception handler. Narrow ones are a deliberate choice here, and
  each broad one that remains is classified in its own comment as ignorable,
  warn, or fail.
- Add a dependency to the package core. It imports only the standard library,
  which is why the verification gates run on a bare interpreter and a reviewer
  can check every number without a scientific stack. CI asserts this.

## Before you open a pull request

```bash
pip install -e '.[dev]'
pre-commit run --all-files
python -m splitguard_ad.verify
```

If you touched anything in the manifest, graph or splitting path, also
regenerate an artefact and compare it byte for byte against the committed one:

```bash
python3 scripts/build_current_leakage_graph.py \
    --manifest data/manifests/current_jpeg_manifest.csv \
    --components /tmp/components.csv --near-dupes /tmp/dupes.csv \
    --audit /tmp/audit.md --summary-json /tmp/summary.json
cmp /tmp/components.csv data/manifests/current_jpeg_leakage_components.csv
```

A refactor of that path is only finished when that comparison is silent. Every
extraction into `src/splitguard_ad/` in this repository was accepted on that
basis, including one where a retyped line had quietly added a resample filter
and would have changed every perceptual hash.

## Comments

Write the reason, not the operation.

```python
# This test exists because the volumetric trainer previously inverted AUROC.
```

is worth keeping. `# calculate AUROC` is not. Where a comment records a past
defect, point at the regression test, the artefact, or the manuscript section
that depends on it, so the next person can tell whether the constraint still
applies.
