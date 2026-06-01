# ADNI Label Ontology (v1)

**Ontology version:** v1
**Drafted:** 2026-05-26
**Author:** Mehmet Berke Isler (drafted with Claude Code against the local
ADNI study-file export dated 22–23 May 2026)
**Status:** Draft. Unblocks G2.5 of the ADNI pipeline runner
([scripts/run_adni_pipeline.py](../scripts/run_adni_pipeline.py)). Two
known manifest-builder bugs (§7) must be patched before G3a will return
non-zero CN/AD counts.
**Authoritative parent:** [docs/SPLITGUARD_AD_FORWARD_PLAN.md §7.4](SPLITGUARD_AD_FORWARD_PLAN.md).

This document is the contract between the downloaded ADNI study CSVs and
the SplitGuard manifest. The manifest builder
([scripts/build_adni_manifest.py](../scripts/build_adni_manifest.py)) must
implement the rules below verbatim.

---

## 1. Scope

The current ADNI batch is `ADNI1: Complete 3Yr 1.5T` — 163 valid subjects,
219 NiFTI scans, on disk under
`data/raw/adni/images/ADNI1_Complete_3Yr_1.5T/`. Clinical metadata covers
those 163 subjects across multiple ADNI phases (901 ADNI1 visits, 265
ADNI2, 73 ADNIGO, 47 ADNI3, 13 ADNI4 — 1,299 total in DXSUM).

**Task 1 (primary, this ontology):** CN vs AD binary classification with
visit-level diagnosis. MCI rows are kept in the manifest but excluded
from the primary task.

**Task 2 (deferred):** CN vs MCI vs AD. Same visit-level rule, MCI
re-included. Requires the manifest builder + splitter to land first.

**Future task (out of scope here):** MCI converter vs non-converter.
Needs an explicit conversion definition (use `DXCHANGE` for ADNIGO+),
minimum follow-up horizon, and censoring rules.

---

## 2. Source tables (local paths under `data/raw/adni/study_files/`)

| Purpose | File | Key columns |
|---|---|---|
| Diagnosis | `DXSUM_22May2026.csv` | PHASE, PTID, RID, VISCODE, VISCODE2, EXAMDATE, DIAGNOSIS, DXNORM, DXMCI, DXAD, DXCONFID |
| Roster | `enrollment/ROSTER_22May2026.csv` | PTID ↔ RID |
| Visits | `enrollment/VISITS_22May2026.csv`, `enrollment/REGISTRY_22May2026.csv` | VISCODE / VISCODE2 definitions |
| Demographics | `PTDEMOG_22May2026.csv` | PHASE, PTID, RID, PTGENDER, PTDOBYY, PTEDUCAT, PTETHCAT, PTRACCAT |
| MRI acquisition (1.5T) | `download/MRIMETA_23May2026.csv` | PHASE, PTID, RID, VISCODE, VISCODE2, EXAMDATE, FIELD_STRENGTH, MMSCOUT, MMSMPRAGE, MMRMPRAGE, MOTION |
| MRI acquisition (3T) | `download/MRI3META_23May2026.csv` | same shape as MRIMETA |
| MRI QC | `download/MRIQC_23May2026.csv` | per-image QC verdicts |
| Cognitive screen | `MMSE_22May2026.csv` | MMSCORE |
| Data dictionary | `DATADIC_22May2026.csv` | PHASE × TBLNAME × FLDNAME → CODE meanings |

**ADNIMERGE2.tar.gz** at project root is the R-package distribution
(binary `.rda` files), not used by this pipeline. The raw CSVs above are
authoritative.

---

## 3. Canonical labels

| Label | Meaning | Use in Task 1 |
|---|---|---|
| `CN` | Cognitively normal | Negative class |
| `MCI` | Mild cognitive impairment | Excluded |
| `AD` | Alzheimer disease (dementia) | Positive class |
| `unknown` | Diagnosis missing, ambiguous, or non-codable per §4 | Excluded |

`unknown` rows are **kept in the manifest** with `label_source=""` and
`label_confidence="missing"` so the splitter can later exclude them
explicitly. Never silently drop.

---

## 4. Visit-level diagnosis resolution

Applied per DXSUM row, in this priority order. Stop at the first rule
that returns a value.

### 4.1 Modern phases (ADNI2, ADNIGO, ADNI3, ADNI4)

1. **`DIAGNOSIS`** (CRF code, all phases ≥ ADNI2):
   - `1` → `CN`
   - `2` → `MCI`
   - `3` → `AD` (DATADIC labels this as "Dementia"; collapsed to AD per the SplitGuard manifest convention)
   - empty or non-{1,2,3} → fall through to 4.2

2. **`DXCHANGE`** (ADNIGO/ADNI2/ADNI3 only): collapse transitions to the destination state.
   - `1` (Stable: NL) → `CN`
   - `2` (Stable: MCI) → `MCI`
   - `3` (Stable: Dementia) → `AD`
   - `4` (Conversion: NL → MCI) → `MCI`
   - `5` (Conversion: MCI → Dementia) → `AD`
   - `6` (Conversion: NL → Dementia) → `AD`
   - `7` (Reversion: MCI → NL) → `CN`
   - `8` (Reversion: Dementia → MCI) → `MCI`
   - `9` (Reversion: Dementia → NL) → `CN`

### 4.2 ADNI1 phase (legacy mutually-exclusive flags)

ADNI1 visits have no `DIAGNOSIS` value in this DXSUM export and no
`DXCURREN` column. Use the three legacy flags:

```
DXAD  == "1"  →  AD
DXMCI == "1"  →  MCI
DXNORM== "1"  →  CN
otherwise     →  unknown   (skip; do not infer from DXOTHDEM or DXCONFID)
```

Per the ADNI1 CRF, these three flags are mutually exclusive (the
distribution in our 163 subjects' 901 ADNI1 visits is: 311 CN, 350 MCI,
240 AD — no mixed cases, zero unknown). The sentinel `-4` means
"not applicable" and is treated as "flag not set."

### 4.3 Disambiguation

- If a single visit has multiple DXSUM rows (rare, possible after
  database re-entry), keep the row with the most recent `USERDATE`.
- `DXOTHDEM=1` (other dementia, not AD) → do **not** assign to AD; label
  as `unknown` and exclude from Task 1. Record in
  `label_confidence="other_dementia"` for downstream filtering.
- `DXCONFID` (1=Uncertain, 2=Mildly Confident, 3=Moderately Confident,
  4=Highly Confident) is recorded as `label_confidence` for ADNI1 rows
  but does **not** alter the label.

### 4.4 Conversion handling within Task 1

A subject's per-scan diagnosis can change over follow-up (e.g. CN at
baseline, AD by year 3). For the CN-vs-AD task we assign **the diagnosis
of the visit whose `EXAMDATE` is closest to the scan's acquisition date**
(see §5 for the alignment policy). The leakage graph still keeps all of
a subject's scans in one component, so CN-baseline and AD-followup scans
from the same subject are never split across train/test.

Implication: a single subject can contribute both CN-labeled and
AD-labeled scans across visits. This is intentional; do not collapse
to a subject-level "ever-AD" rule unless the manuscript framing
requires it.

---

## 5. Scan → visit alignment (image-to-label join)

The .nii path encodes only the **acquisition timestamp**, not the
viscode:

```
ADNI1_Complete_3Yr_1.5T/<PTID>/MPR__GradWarp__B1_Correction__N3__Scaled/
  <YYYY-MM-DD_HH_MM_SS.0>/I<image_uid>/
    ADNI_<PTID>_MR_..._S<series_uid>_I<image_uid>.nii
```

So the join goes **path → MRIMETA → DXSUM**:

1. From path: extract `subject_id = PTID` and `acq_date = YYYY-MM-DD`.
2. Restrict MRIMETA + MRI3META to rows where `PTID == subject_id`.
3. Find the MRIMETA row whose `EXAMDATE` is closest to `acq_date` for
   that subject. Tolerance: ≤ 180 days. If no MRIMETA row exists for the
   subject, or all rows fall outside the tolerance, record
   `viscode=""`, `field_strength=""`, `label_source=""`,
   `label_confidence="no_mri_meta_match"`, and `diagnosis_group=unknown`.
4. From the matched MRIMETA row read `VISCODE` (preferring `VISCODE2`).
5. Restrict DXSUM to rows where `(RID, VISCODE)` (preferring `VISCODE2`)
   matches. If multiple, take the row whose `EXAMDATE` is closest to the
   scan `acq_date`. If none, fall back to the DXSUM row with the
   smallest `|EXAMDATE − acq_date|` for that subject regardless of
   viscode (record `label_confidence="date_only_match"`).

The 180-day tolerance is the documented ADNI convention for pairing
scans to clinical visits in retrospective studies. Tighten to 90 days
later if QC review surfaces stale-label cases.

---

## 6. Manifest columns produced

Per [scripts/build_adni_manifest.py:75-97](../scripts/build_adni_manifest.py),
plus the new fields this ontology requires:

| Column | Source / rule |
|---|---|
| `image_id` | Hash of `relative_path` (stable, 12-char) |
| `subject_id` | `PTID` from path |
| `session_id` | `f"{PTID}__{VISCODE2 or VISCODE or acq_date}"` |
| `source_dataset` | constant `"adni"` |
| `scanner_field_strength` | MRIMETA `FIELD_STRENGTH` (1.5 / 3.0 / unknown) |
| `age` | computed at scan time from PTDEMOG `PTDOBYY` and `acq_date` |
| `sex` | PTDEMOG `PTGENDER` (1→M, 2→F) |
| `diagnosis_group` | per §4, joined per §5 |
| `slice_index` | empty for ADNI (volumetric scans, not 2D slices) |
| `qc_flag` | `MOTION`/`MALFUNC` from MRIMETA collapsed to `"pass"`/`"fail"`/`"unknown"`; if MRIQC has a per-image entry, prefer that |
| `ptid` | path PTID |
| `rid` | derived from PTID (numeric tail) |
| `viscode` | MRIMETA `VISCODE2` or `VISCODE` |
| `image_uid` | path `I<id>` |
| `series_uid` | path `S<id>` |
| `acq_date` | path `YYYY-MM-DD` |
| `modality` | constant `"MR"` for this batch |
| `image_path` | absolute path |
| `relative_path` | path relative to `data/raw/adni/images/` |
| `label_source` | one of `DIAGNOSIS`, `DXCHANGE`, `DXNORM`, `DXMCI`, `DXAD`, or `""` |
| `label_confidence` | one of `clinical_table`, `date_only_match`, `other_dementia`, `no_mri_meta_match`, `missing` |

---

## 7. Manifest-builder bugs that block G3a

The current [scripts/build_adni_manifest.py](../scripts/build_adni_manifest.py)
implements ~70% of this ontology. Two patches are required for G3a to
return non-zero CN/AD counts on this export.

### Bug 7.1 — Diagnosis parser ignores ADNI1 legacy flags

`diagnosis_from_row()` at lines 183–198 currently checks only
`DIAGNOSIS`, `DXCURREN`, `DXCHANGE`, `DX_BL`. Add a fallback at the end
implementing §4.2:

```python
# §4.2 ADNI1 legacy flags (mutually exclusive)
if pick(row, "dxad") == "1":
    return "AD", "DXAD", "clinical_table"
if pick(row, "dxmci") == "1":
    return "MCI", "DXMCI", "clinical_table"
if pick(row, "dxnorm") == "1":
    return "CN", "DXNORM", "clinical_table"
```

Without this patch, 901/1,299 of our DXSUM rows label as `unknown` and
G3a fails its "no CN or AD rows" check.

### Bug 7.2 — MRIMETA join uses non-existent column

`build_manifest_row()` at lines 286–293 does
`meta = mri_meta.get(image_uid) or mri_meta.get(series_uid)`, but the
MRIMETA / MRI3META exports do not contain an image-UID or series-UID
column (only PHASE/PTID/RID/VISCODE/VISCODE2/EXAMDATE/FIELD_STRENGTH/…).
The lookup always returns `{}`, so `viscode` is empty and the
`(rid, viscode)` DXSUM join silently misses every row.

Replace the lookup with the path → MRIMETA-by-(PTID, EXAMDATE) join
documented in §5. Pseudocode:

```python
# Index MRIMETA by PTID with rows sorted by EXAMDATE
mri_by_ptid: dict[str, list[dict]] = defaultdict(list)
for row in mri_rows_all:
    mri_by_ptid[row["ptid"]].append(row)
for rows in mri_by_ptid.values():
    rows.sort(key=lambda r: r["examdate"])

# Per-image: pick the row with min |EXAMDATE - acq_date|, within 180 days
def closest_mri_row(ptid: str, acq_date: str) -> dict:
    rows = mri_by_ptid.get(ptid, [])
    if not rows or not acq_date:
        return {}
    target = date.fromisoformat(acq_date)
    best, best_delta = {}, timedelta(days=10**6)
    for r in rows:
        try:
            d = date.fromisoformat(r["examdate"])
        except ValueError:
            continue
        delta = abs(d - target)
        if delta < best_delta:
            best, best_delta = r, delta
    return best if best_delta <= timedelta(days=180) else {}
```

Then use the returned row's `VISCODE2` (or `VISCODE`) for the DXSUM
join, and its `FIELD_STRENGTH` for the manifest column.

---

## 8. Acceptance for this ontology (G2.5 → G3a)

The ontology is "done" for v1 when, after the §7 patches:

- `python3 scripts/build_adni_manifest.py` exits 0.
- `data/manifests/adni/adni_manifest.csv` contains one row per .nii under
  `data/raw/adni/images/` (≥ 219 rows for the current batch).
- `reports/audits/adni/adni_manifest_summary.json` shows:
  - `n_images >= 219`
  - `diagnosis_counts.CN >= 1` and `diagnosis_counts.AD >= 1` (G3a
    runner check)
  - `label_confidence_counts.missing` < 10% of rows
  - `qc_flag_counts.fail` and `field_strength` distributions are sane
    (~all 1.5T for this batch).

---

## 9. Open questions parked for v2

1. **Cross-phase label consistency.** A subject CN in ADNI1 may convert
   to AD in ADNI2. v1 takes the closest-date visit, which is correct
   per-scan but may produce label trajectories that look noisy in
   plots. Decide whether to publish per-scan trajectories or a
   subject-level "trajectory class" alongside.
2. **MRIQC integration.** `MRIQC_23May2026.csv` is 30 MB and has not yet
   been schema-mapped against `download/MRIMETA_*.csv`. v1 reads
   MRIMETA's `MOTION` / `MALFUNC`; v2 should prefer per-image MRIQC
   verdicts when available.
3. **Demographics for non-baseline visits.** PTDEMOG is collected once
   per subject in ADNI1 (mostly at screening). v1 propagates the
   demographic row across all visits and computes `age` at scan from
   `PTDOBYY` + `acq_date`; this is the standard ADNI convention but
   should be flagged in the manuscript Methods.
4. **Image-UID linkage.** If a future ADNI download includes the
   image-index table (e.g. `IDA_MR_METADATA_LISTING.csv`), the join
   should prefer image-UID → viscode over date-proximity. The §5
   fallback stays as a safety net.

---

## 10. Changelog

- **2026-05-26 v1:** First version. Grounded in `DXSUM_22May2026.csv`,
  `DATADIC_22May2026.csv`, `MRIMETA_23May2026.csv`, and the 163-subject
  ADNI1 image batch under `data/raw/adni/images/ADNI1_Complete_3Yr_1.5T/`.
  Identifies two manifest-builder bugs (§7) blocking G3a.
