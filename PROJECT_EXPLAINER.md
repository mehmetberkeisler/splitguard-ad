# SplitGuard-AD, explained

A plain-language account of what this project found, what it does not show,
how the pieces fit together, and what is left to do. Written for you, not for
a reviewer. Nothing here is a claim the manuscript does not also make.

---

## 1. The one-sentence version

> Everyone agrees you should split medical imaging data by patient. That
> advice quietly assumes you *know* who the patient is. This project tests
> what happens when you don't, and shows that the assumption fails in ways
> that inflate reported performance.

That is the paper. Everything else supports it.

---

## 2. Why this is a real contribution

It helps to be clear about what is *not* new, because a reviewer will start
there:

| Claim | Status |
| --- | --- |
| Random image-level splitting inflates performance | known since Wen 2020, Yagis 2021 |
| Splitting by patient is safer | known |
| Repeated scans of one head let a model recognise the patient | known in principle |
| Connected components can enforce grouping | elementary |

If the paper is read as any of those, it gets rejected. The contribution is
one step further back:

> Patient-wise splitting is only as good as the patient identifier. Identifiers
> get deleted, fragmented, merged and redistributed. **Nobody had measured what
> that does.**

This project measures it on both links of the chain:

```
degraded provenance  →  leakage survives the split  →  reported score inflates
```

---

## 3. The four results that carry the paper

### 3.1 The gap is real: **+0.118** AUROC

On ADNI1, a random image split scores 0.947; a component-safe split scores
0.829. The difference is +0.118, with a paired-seed interval of
[+0.096, +0.145]. It replicates on OASIS-1, under DenseNet-121, on a 3D
volumetric arm, and across six robustness arms.

This part is solid but, on its own, not novel.

### 3.2 The mechanism, proved not argued: **0.878 on permuted labels**

This is the strongest experiment in the paper and it is worth understanding
properly.

We took the diagnosis labels and **shuffled them across participants**. Each
participant keeps one label, but the label now has nothing to do with their
brain. There is no disease signal left to learn — none.

Then we retrained all three protocols:

| Protocol | AUROC on permuted labels |
| --- | --- |
| Random split | **0.878** |
| Subject-only split | 0.521 (chance) |
| Component-safe split | 0.427 (chance) |

The random split still scores 0.878 with no disease information available.
It cannot be learning pathology, because there is none. It is recognising
patients it already saw in training.

The training curves make it sharper still: **all three protocols fit the
training set equally well** (final loss 0.068 / 0.037 / 0.045). Only the
random split's memorisation *transfers*, because its test set contains the
same people.

This is why the experiment matters. Without it, a sceptic says "maybe the
random split just makes an easier test set." With it, that reading is closed.

### 3.3 The dose: **+0.125** AUROC per unit of contamination

Substituting a measured share of test scans with other scans of training
participants raises AUROC roughly linearly. More leakage, more optimism.

**The correction that matters here.** For several drafts this said "holding the
training set fixed", and that was wrong. `inject_leakage_split.py:177` moves
each substituted scan *out* of training, so evaluated training rows fall from
1,041 to 783 across the dose axis, 24.8% of them. The donor participant stays
in training, which is why donors need two or more scans, so the leakage is
real; the training set simply is not fixed. The evaluated test set also grows,
203 to 258 rows, with the AD share going 47.7% to 54.3%. An external
adversarial review caught this on 2026-10-03 and the code confirmed it. The
slope is therefore descriptive of the substitution procedure, not a causal
price per unit of leakage, and five places in the manuscript that said
otherwise, including the abstract, were corrected.

**Second caveat, already stated:** the slope is not a universal constant.
DenseNet-121 gives a different slope on the same data. It is an
experiment-specific calibration, and the manuscript says so in those words.

### 3.4 It happens for real: the Tier-1 case

A widely used public 2D Alzheimer's benchmark turns out, by content matching,
to be a redistribution of 200 OASIS-1 participants at 32 slices each. Its
filename index merges and splits the original participants.

The consequence: if you do everything right — group by the only identifier
the release gives you — **70% of your test images still share a participant
with training**, and nothing inside the release can tell you.

The forensic evidence is now a table in the Supplement. The decisive number
is the separation: the weakest *correct* image match correlates at 0.9996,
while the strongest match to any *different* participant reaches only 0.9623.
No threshold between those changes a single assignment.

---

## 4. The honest weak spot, and how it is handled

**The leakage graph — the thing the project is named after — has never been
shown to help on real data.**

| Where measured | Component marginal | Includes zero? |
| --- | --- | --- |
| Six intact ADNI arms | −0.026 to +0.026 | yes, all six |
| Tier-1, real provenance damage | +0.027 | yes |

On Tier-1, the paper itself says the component layer "does not recover what
the identifier lost," because the redistribution destroyed the session and
series identifiers too.

The graph demonstrably helps only under *simulated* corruption, and even there
2 of 8 cells have intervals excluding zero, each by 0.001.

### Why this is not fatal

Because the paper now says it, up front, and frames it correctly through the
**three-regime model** that opens the Discussion:

- **Regime I — intact provenance.** Graph ≡ participant grouping. The null is
  the *design target*. A provenance-aware splitter that disagreed with
  participant grouping on clean data would be wrong.
- **Regime II — degraded but redundant.** The key is damaged, other relations
  survive. The graph prevents 49% of the leakage at 10% identifier loss, 90%
  when one participant is split across two keys. **This is the paper's
  subject.**
- **Regime III — destroyed.** No surviving relation. Tier-1. An information
  limit, not a software limitation — and the paper claims nothing here.

The reframed question is not "is the graph better?" but "which regime is your
dataset in?", which you can answer *before training* from identifiers alone.

---

## 5. What was found and fixed along the way

These matter because they show the verification machinery works, and several
would have reached reviewers:

| Problem | How it was caught |
| --- | --- |
| The null control was **broken**: it permuted labels inside each partition, so a participant straddling train/test got two different labels, making memorisation undetectable. It returned chance everywhere and read as a clean null. | Reading the code against its own docstring |
| The volumetric trainer computed **1 − AUROC** and selected the least-trained epoch | Regression test against the analysis pipeline |
| The graphical abstract still showed **+0.129 / 0.949 / 0.819 / 382 subjects / 98.3%** — four superseded values and one explicitly retracted — and it uploads *with* the submission | Sweeping tracked PDFs, not just the manuscript |
| A published number was wrong: double rounding turned 0.86546 into **0.866** | A verifier mismatch after an unrelated cut |
| `check_absent`, the guard holding 27 retired values out of the paper, searched raw source — so any phrase LaTeX wrapped across a line was **invisible to it** | Adding a guard, then testing that it fired. It didn't. |
| The generic graph builder grouped on raw `subject_id`, so blank or `unknown` values would merge every unlabelled scan into one fictional component — and the audit would then *pass* the split | External review, confirmed latent on the shipped cohort |
| Six `§??` in the supplement — references into the main manuscript, which compiles separately and can never resolve them | Reading the built PDF rather than the source |
| An installed `scikit-learn` older than the pin against a newer numpy leaves `sklearn.metrics` unimportable, because it is a C extension built for the other ABI. Every training script reaches `--help` and then dies at its first metric call, while all three gates keep passing, because none imports it | Trying to run the fixed-epoch experiment |
| `requirements.txt` described a stack that **had never run**. The GPU node took the image's own CUDA wheels and froze what resolved; six of seven load-bearing pins named different versions, and MONAI, which the volumetric arm needs, was not pinned at all | Diffing the pins against `runs_gpu/gpu_environment.txt` |
| That recorded environment, which the manuscript cites as the provenance of every AUROC, was **gitignored**, so no reader could see it | Checking whether the cited file was actually released |

The last one is the pattern worth remembering: **the checks stayed green while
the pipeline was broken.** `scripts/check_environment.py` now exists for that.

---

## 6. How to check any of this yourself

```bash
python3 scripts/verify_paper_numbers.py          # 521 numbers vs their source artefacts
python3 scripts/validate_submission.py --build   # 24 journal requirements, compiles both PDFs
python3 -m unittest discover -s tests            # 104 tests, ~4 seconds
python3 scripts/check_environment.py             # has this machine drifted from the pins?
```

The first three all pass, and they pass **inside the distributed zip**, not
only in this checkout. That means a reader with the bundle and no ADNI access
can verify every number in the paper.

The test suite is validated by mutation rather than by its own green light:
breaking union-find, the identifier grouping, the component label rule, each
corruption operator, the seeded tie-break and the permutation design produces
seven mutants, and the suite fails on all seven.

---

## 7. What is left

### Yours alone

1. **Nothing is committed or pushed.** ~75 changed paths.
2. **Zenodo DOI** for the release.
3. **İrem Ilter's co-author review**; funding and COI confirmation.
4. **JIIM's graphical-abstract specification** — the current file uses
   Elsevier's 2:1 format from an earlier submission plan.

### Recommended, not blocking

5. **The fixed-epoch run.** Both reviews call it the one new experiment worth
   doing. I added the `--checkpoint-rule final` flag so it is one command, but
   could not run it: the environment drift above breaks sklearn, and fixing it
   properly needs a clean install I did not want to make without asking. What
   exists instead is a *measurement* of the concern: with real labels,
   selection optimism is +0.026 / +0.030 / −0.026 across the three protocols,
   every interval spans zero, and it is *negative* for component-safe. The
   confound the reviews infer exists only in the no-signal regime.

   To run it on a GPU node:
   ```bash
   python3 scripts/run_adni_inflation_gap.py --seeds 0 1 2 3 4 --epochs 15 \
       --checkpoint-rule final --output-root runs/adni_fixed_epoch
   ```

6. **Length.** 58 pages, ~25,800 words. Both reviews want 25–30% cut; we are
   at roughly 6%. The demotions they list are mostly done — clinical cost is
   163 words with the apparatus in Supplement S2, subgroup is 336 words
   pointing at S6. What remains is Methods and Results, which both reviews say
   to *keep*. Cutting further means removing verified material.

   My recommendation: submit at this length. If the editor asks, the taxonomy
   (§2.1) and the audit-instrument detail in §2.2 are the first 8–10 pages to
   move, and neither carries a result.

---

## 8. If a reviewer says...

| "..." | The answer, already in the paper |
| --- | --- |
| "Patient leakage is already known." | The contribution is the provenance dependency, not the leakage. §1, Discussion opening. |
| "Your graph doesn't beat subject-wise splitting." | Correct, and expected — Regime I is the design target. §5.1 (three regimes). |
| "The 0.125 slope isn't general." | Agreed, stated: experiment-specific, and DenseNet gives a different slope on the same data. |
| "You hid a composition problem." | It is reported beside the headline, not in Limitations, and the size-balanced control gives +0.136 against +0.118. |
| "Five seeds is too few." | Every interval is labelled descriptive; the permutation control is a mechanism demonstration, not a significance test. |
| "Checkpoint selection confounds the protocols." | Measured: +0.026 / +0.030 / −0.026, all intervals spanning zero. Limitations. |
| "How do you know Tier-1 is OASIS?" | Supplementary forensic table: 6,400/6,400 unambiguous, 0.9996 vs 0.9623 separation. |
| "Is Tier-1 an independent cohort?" | No, and the paper says so twice. All 200 participants are also in Tier 2. |
| "Your clinical numbers are speculative." | Demoted to one paragraph; the arithmetic is in Supplement S2 and labelled not a deployment estimate. |
| "Is this a mature library?" | No — "an open-source reference implementation and audit pipeline, not a general-purpose library." |

---

## 9. The thing worth being proud of

Most papers about evaluation rigour do not subject themselves to it. This one
does, and it kept catching itself: the component-size defect was found by the
framework auditing its own output; the broken null control was found by
reading code against its own stated purpose; a published digit was found by a
verifier that checks 521 numbers against the artefacts that produced them.

That self-auditing quality is, in my view, the paper's best feature and the
thing most likely to earn a reviewer's trust. It is worth protecting in
revision — do not let a request to shorten remove the places where the paper
reports what went wrong.
