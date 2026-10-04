# Release checklist

Run `python -m splitguard_ad.verify` first. It covers most of this list and
tells you which part failed. The boxes below are the ones a command cannot
decide for you.

## Automated, and enforced

- [ ] `python -m splitguard_ad.verify` reports `OVERALL STATUS: PASS`
- [ ] no raw or preprocessed imaging is tracked
- [ ] no model checkpoint is tracked
- [ ] no per-image prediction file is tracked
- [ ] the ADNI download inventory is not tracked
- [ ] the recovered Tier-1 to OASIS-1 mapping is not tracked
- [ ] the release salt appears in no released file
- [ ] no tracked file names a participant in the cohort
- [ ] every manuscript number matches the artefact that produced it
- [ ] the submission requirements pass against freshly built PDFs
- [ ] the full test suite passes

Those eleven are checked by `tests/test_release_safety.py`,
`scripts/verify_paper_numbers.py` and `scripts/validate_submission.py`. If one
fails, the verify command names it.

## Judgement, and yours alone

- [ ] the environment matches `requirements-lock.txt`, or you have accepted
      that regenerated figures and retrained arms will not reproduce the
      released artefacts
- [ ] `git status --porcelain` read line by line before staging, not skimmed
- [ ] the commit hash recorded in the release notes
- [ ] `LICENSE` present and correct
- [ ] `CITATION.cff` lists the authors and the DOI
- [ ] the Zenodo DOI minted and written into the Data Availability statement
- [ ] every co-author has approved the release contents
- [ ] the release tarball unpacked somewhere clean and verified there, not only
      in the working tree it was built from

The last one has caught a real problem before: a bundle that passed every gate
in place ran twenty fewer checks once unpacked, because four artefacts it
needed were gitignored.

## Preparing the public release tree

```bash
SPLITGUARD_RELEASE_SALT=... python3 scripts/build_release_artefacts.py
python -m splitguard_ad.verify
git archive --format=tar --prefix=SplitGuard-AD/ HEAD | tar -x -C /tmp/release
cd /tmp/release/SplitGuard-AD && python -m splitguard_ad.verify
```

Building the archive from `HEAD` rather than from the working tree is
deliberate: it excludes everything untracked by construction, which is where
the never-publish files live, instead of relying on a copy command's exclude
list being complete.
