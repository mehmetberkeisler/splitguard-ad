# Security and privacy

This repository is built around a cohort whose data-use agreement forbids
redistribution. The boundary is therefore enforced in code and tested, not left
to a `.gitignore` and good intentions, because a `.gitignore` does not fail a
build and the consequence of getting one rule wrong cannot be undone once a
repository is public.

## What may be released

| Tier | What ships | Why |
| --- | --- | --- |
| Tier 1, public JPEG benchmark | Frozen component-safe split manifests at image-path level, plus leakage-graph edge lists | The benchmark is public and its filename key is the artefact under study, so publishing it is the point |
| Tier 2, OASIS-1 | Frozen split manifests indexed by the public OASIS subject and session codes | Those codes are already public; no derived paths and no clinical fields |
| Tier 3, ADNI1 | Hashed component manifests only | Nothing that is not hashed or aggregate |
| All tiers | Code, paper sources, figures, aggregate result tables, audit summaries, the recorded GPU environment | None of it is participant-level |

## What must never be released

- Raw or preprocessed imaging of any tier.
- Per-image predictions, which are participant-level by construction.
- Model checkpoints trained on ADNI, which carry the same restriction as the
  data they were trained on.
- The ADNI download inventory.
- The recovered Tier-1 to OASIS-1 participant mapping. Tier 1 is a
  redistribution of OASIS-1, and that mapping is the identity the redistribution
  destroyed. Publishing the filename key is the contribution; publishing the
  recovered truth would undo the de-identification of the source cohort.
- The per-release hashing salt.
- Any real cohort participant identifier, including in test fixtures. That one
  is easy to get wrong: a PTID-shaped literal is all a hashing test needs, and
  a real one discloses that that participant is in the cohort.

## How identifiers are transformed

Tier 3 participant identifiers are replaced by
`PBKDF2-HMAC-SHA256(subject_id, salt)` at **600,000 iterations**, 32-byte
output, hex encoded.

Key stretching is required rather than merely prudent. ADNI PTIDs come from a
space of roughly ten million, so a single-round digest of one is invertible by
enumeration in under a second. The iteration count is a parameter of the
privacy claim the manuscript makes, so it is pinned in
`splitguard_ad.release.privacy` and asserted by the test suite rather than left
to a library default.

The salt lives outside the repository, is read from `SPLITGUARD_RELEASE_SALT`,
and has no default. A release records only its fingerprint, so two releases can
be compared without either revealing the salt that would invert their hashes.

## What the checks actually verify

`python -m splitguard_ad.verify` runs all of these.

`tests/test_release_safety.py`:

- the same participant hashes the same way within a release, and differently
  under a different salt
- distinct participants do not collide, and a digest never contains its input
- the iteration count has not been silently lowered
- Tier 3 ships only permitted columns and never a raw participant key
- Tier 2 ships no paths and no clinical fields
- Tier 1 never ships the recovered OASIS mapping
- the release manifest records a salt fingerprint and not a salt
- the salt appears in no released file, searched as bytes rather than decoded
  text, so a salt embedded in a binary artefact cannot be skipped quietly
- no model weights, per-image predictions or imaging are tracked
- no file git would ship names a participant in the cohort

`scripts/validate_submission.py` additionally checks that each tier's release
directory exists, that every file the manifest counts is present, that every
present file is counted, and that the ADNI release carries only de-identified
columns.

## Preparing a clean public release

See `RELEASE_CHECKLIST.md`. The short version:

```bash
SPLITGUARD_RELEASE_SALT=... python3 scripts/build_release_artefacts.py
python -m splitguard_ad.verify
git status --porcelain          # read every line before staging
```

Without the salt the ADNI rows are skipped and the manifest records their
absence rather than implying they shipped.

## Reporting a problem

If you believe this repository discloses participant-level data, please contact
the corresponding author listed in `CITATION.cff` rather than opening a public
issue.
