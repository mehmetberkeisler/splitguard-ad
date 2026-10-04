# Manifest specification

The manifest is the only thing SplitGuard needs about your cohort. It is a CSV
with one row per image, and it is deliberately permissive: the whole point of
the framework is to work on cohorts whose identity evidence is incomplete, so
most columns are optional and a missing column disables the edges that depend
on it rather than failing the run.

## Required

| Column | Meaning |
| --- | --- |
| `image_id` | Unique within the manifest. The node identity in the leakage graph. |

That is the only hard requirement. A manifest with nothing else produces one
component per image, which is the correct answer when no identity evidence
survives at all: every image is its own participant as far as anything in the
file can tell.

## Optional, and what each one buys

| Column | Edge family it enables |
| --- | --- |
| `subject_id` | `same_subject`. The participant key, when there is one. |
| `session_id` | `same_session`. Joins images acquired in one visit. |
| `series_uid` | `same_series_uid`. Joins images from one acquisition series. |
| `acq_date` | `same_acq_date`, scoped within a participant. |
| `relative_path` | Near-duplicate detection, when the path resolves to a file. |

Each present column contributes edges; absent ones contribute none. This is
what makes the three provenance regimes observable before training. If
`subject_id` is the only identifier and it is damaged, the graph has no
redundancy to fall back on and will say so.

## Label columns

| Column | Meaning |
| --- | --- |
| `raw_class_label` | The cohort's own class, used for stratification. |
| `binary_label` | The two-class label the trainer uses. |

Components must be label-pure: every image in one component must carry the same
`raw_class_label` and the same `binary_label`. A component that is not pure
raises `ValueError` rather than being silently assigned, because a component
spanning two classes means either the labels or the identity evidence is wrong,
and guessing which would corrupt the split either way.

## Missing-value semantics

This is the part that matters most, and it is the part a naive implementation
gets wrong.

These values are treated as **absent**, not as identifiers:

```
""   "na"   "n/a"   "none"   "null"   "unknown"   "nan"   "-"   "?"
```

Matching is case-insensitive and surrounding whitespace is stripped.

The reason is not tidiness. Grouping is identity evidence, and a literal
`unknown` is the absence of evidence. If it were treated as a shared key, every
image whose participant is unrecorded would be merged into one enormous
fictional participant, the audit would then report a leakage-free split, and
the split would be leakage-free with respect to a participant who does not
exist. A provenance-aware splitter that did this would manufacture exactly the
failure this project exists to measure. `tests/test_leakage_graph.py` holds the
behaviour, and disabling it makes three tests fail.

## A valid minimal manifest

```csv
image_id,subject_id,session_id,raw_class_label,binary_label
p01_s1,p01,p01_visit1,CN,CN
p01_s2,p01,p01_visit2,CN,CN
p02_s1,p02,p02_visit1,AD,AD
p02_s2,p02,p02_visit2,AD,AD
```

Two participants, two images each, two components.

## A manifest with damaged provenance

```csv
image_id,subject_id,session_id,raw_class_label,binary_label
p01_s1,p01,p01_visit1,CN,CN
p01_s2,unknown,p01_visit2,CN,CN
```

`subject_id` is absent for the second row, so it contributes no participant
edge. `session_id` still shares the `p01` stem, so the graph recovers the
relation from the surviving column. Grouping on `subject_id` alone would not.

## An invalid manifest

```csv
image_id,subject_id,raw_class_label,binary_label
p01_s1,p01,CN,CN
p01_s2,p01,AD,AD
```

One participant, two classes, so the component is not label-pure:

```
ValueError: SplitGuard v0 requires pure-label components.
Component <id> has raw=['AD', 'CN'], binary=['AD', 'CN'].
```

## Building a graph and a split

```bash
splitguard graph --manifest my_cohort.csv --components components.csv
splitguard split --manifest my_cohort.csv --components components.csv \
    --output split.csv --seed 0
```

Or from Python, which is the same code:

```python
from splitguard_ad import UnionFind, normalise_identifier, assign_splits
```

## Audit levels

The audit reports findings at four levels and only one of them refuses:

| Level | Meaning |
| --- | --- |
| `FAIL` | The split is invalid. A component has images in two partitions. |
| `WARN` | Valid but statistically awkward: component-size or class-mix imbalance. |
| `INFO` | A descriptive diagnostic with no pass or fail reading. |
| `PASS` | A requirement that was checked and satisfied. |

The decision is `GO` unless something `FAIL`s. Promoting a `WARN` to a refusal
would make the auditor reject splits that are correct for the purpose it
exists to serve, and the largest-first allocation order makes some imbalance
routine. `splitguard_ad.audit` holds this.
