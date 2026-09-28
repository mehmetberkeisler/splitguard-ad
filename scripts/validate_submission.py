#!/usr/bin/env python3
"""Check the manuscript against the target journal's stated requirements.

``scripts/verify_paper_numbers.py`` checks that every number in the manuscript
matches the artefact it came from. This script checks the things that are not
numbers: the structure, the front matter, the declarations, the bibliography and
the build. The requirements encoded here are the ones the Journal of Imaging
Informatics in Medicine publishes for research papers (Springer, journal 10278):

* Abstract of 150-250 words, 4-6 keywords.
* Sections Introduction, Materials and Methods, Results, Discussion,
  Conclusions, Acknowledgements, in that order.
* References numbered in order of citation.
* Statements for funding, competing interests, author contributions, ethics
  approval, consent, and use of an LLM -- the last documented in the Methods.

It also fails on things no journal wants to receive: a float nobody refers to,
a reference that does not resolve, a cross-reference to a label that does not
exist, a number still rendering as ``??``, or a page whose text runs into the
margin.

Usage
-----
    python3 scripts/validate_submission.py            # checks, no build
    python3 scripts/validate_submission.py --build    # also compiles and reads the log
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "splitguard_ad.tex"
SUPP = ROOT / "paper" / "SplitGuard-AD_Supplementary_Material.tex"
BIB = ROOT / "paper" / "references.bib"
TABLE_DIR = ROOT / "paper" / "tables"

REQUIRED_SECTIONS = ["Introduction", "Materials and Methods", "Results",
                     "Discussion", "Conclusions"]
REQUIRED_STATEMENTS = {
    "Funding": r"\\paragraph\*?\{Funding\}",
    "Competing interests": r"\\paragraph\*?\{Competing interests\}",
    "Author contributions": r"\\paragraph\*?\{Author contributions\}",
    "Ethics approval": r"\\paragraph\*?\{Ethics approval",
    "Consent": r"\\paragraph\*?\{Consent",
    "Use of generative AI": r"\\paragraph\*?\{Use of generative AI\}",
    "Data and code availability": r"\\paragraph\*?\{Data and code availability\}",
}


class Report:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.failures: list[tuple[str, str]] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        (self.passes.append(label) if ok else self.failures.append((label, detail)))
        return ok


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def abstract_words(text: str) -> int:
    m = re.search(r"\\abstract\{", text)
    if not m:
        return -1
    depth, i = 1, m.end()
    while i < len(text) and depth:
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
        i += 1
    body = text[m.end(): i - 1]
    body = re.sub(r"\\[a-zA-Z]+\*?(\{[^{}]*\})?", " ", body)
    body = re.sub(r"[{}$~\\]", " ", body)
    return len([w for w in body.split() if any(c.isalnum() for c in w)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="compile the manuscript and fail on errors or margin overruns")
    args = ap.parse_args()

    raw = PAPER.read_text(encoding="utf-8")
    text = strip_comments(raw)
    r = Report()

    # ── Front matter ────────────────────────────────────────────────────
    r.check("document class is Springer Nature sn-jnl",
            "{sn-jnl}" in text,
            "the target journal is a Springer title; convert the class")
    n_words = abstract_words(text)
    r.check(f"abstract is 150-250 words (it is {n_words})", 150 <= n_words <= 250,
            f"{n_words} words")
    kw = re.search(r"\\keywords\{(.*?)\}", text, re.S)
    n_kw = len([k for k in kw.group(1).split(",") if k.strip()]) if kw else 0
    r.check(f"4-6 keywords (there are {n_kw})", 4 <= n_kw <= 6, f"{n_kw} keywords")

    # ── Structure ───────────────────────────────────────────────────────
    found = [m.group(1) for m in re.finditer(r"\\section\{([^}]*)\}", text)]
    r.check("required sections present and in order",
            found[:len(REQUIRED_SECTIONS)] == REQUIRED_SECTIONS,
            f"found {found[:len(REQUIRED_SECTIONS)]}")
    r.check("Acknowledgements section present",
            bool(re.search(r"\\section\*?\{Acknowledgements\}", text)))

    # ── Declarations ────────────────────────────────────────────────────
    for label, pattern in REQUIRED_STATEMENTS.items():
        r.check(f"declaration: {label}", bool(re.search(pattern, text)),
                "required on submission")
    # The journal asks for LLM use in the Methods, not only in the Declarations.
    methods = text.split(r"\section{Results}")[0]
    r.check("LLM use documented in the Methods section",
            "large-language-model" in methods.lower() or "large language model" in methods.lower(),
            "the journal requires this in Methods")

    # ── Bibliography ────────────────────────────────────────────────────
    cited = {k.strip() for m in re.finditer(r"\\cite[tp]?\{([^}]*)\}", text)
             for k in m.group(1).split(",") if k.strip()}
    defined = set(re.findall(r"^@\w+\{([^,]+),", BIB.read_text(encoding="utf-8"), re.M))
    r.check(f"every citation resolves ({len(cited)} keys)", not (cited - defined),
            f"missing from references.bib: {sorted(cited - defined)}")
    r.check("no uncited bibliography entries", not (defined - cited),
            f"never cited: {sorted(defined - cited)}")

    # ── Cross-references ────────────────────────────────────────────────
    label_sources = [text, strip_comments(SUPP.read_text(encoding="utf-8"))]
    label_sources += [strip_comments(p.read_text(encoding="utf-8"))
                      for p in sorted(TABLE_DIR.glob("*.tex"))]
    labels = {m.group(1) for s in label_sources for m in re.finditer(r"\\label\{([^}]*)\}", s)}
    refs = {m.group(2) for s in label_sources
            for m in re.finditer(r"\\(ref|autoref|eqref)\{([^}]*)\}", s)}
    r.check(f"every cross-reference resolves ({len(refs)} targets)", not (refs - labels),
            f"undefined: {sorted(refs - labels)}")

    # Every float has to be referred to somewhere, or the reader never meets it.
    float_labels = {m.group(1) for m in re.finditer(r"\\label\{((?:fig|tab):[^}]*)\}", text)}
    float_labels |= {m.group(1) for p in sorted(TABLE_DIR.glob("*.tex"))
                     for m in re.finditer(r"\\label\{((?:fig|tab):[^}]*)\}",
                                          strip_comments(p.read_text(encoding="utf-8")))}
    orphans = sorted(float_labels - refs)
    r.check("every figure and table is referenced in the text", not orphans,
            f"never referenced: {orphans}")

    # ── Generated numbers ───────────────────────────────────────────────
    macros = TABLE_DIR / "gpu_numbers.tex"
    pending = {m.group(1) for m in
               re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{[^}]*\\pending", macros.read_text())}
    used_pending = sorted(n for n in pending if f"\\{n}" in text)
    r.check("no manuscript number renders as ??", not used_pending,
            f"still pending: {used_pending}")

    # ── Build ───────────────────────────────────────────────────────────
    if args.build:
        # Both documents are submitted, so both are built and both are held to
        # the same standard.
        for doc, name in ((PAPER, "manuscript"), (SUPP, "supplement")):
            proc = subprocess.run(["tectonic", "-X", "compile", doc.name],
                                  cwd=doc.parent, capture_output=True, text=True)
            log = proc.stdout + proc.stderr
            r.check(f"{name} compiles without errors", proc.returncode == 0,
                    "\n".join(l for l in log.splitlines() if l.startswith("error"))[:400])
            overfull = sorted({l.split("warning: ")[-1] for l in log.splitlines()
                               if "Overfull \\hbox" in l})
            r.check(f"no {name} text runs into the margin", not overfull,
                    "\n      ".join(overfull))

    # ── Report ──────────────────────────────────────────────────────────
    print(f"Checked {len(r.passes) + len(r.failures)} submission requirements\n")
    for label in r.passes:
        print(f"  ok   {label}")
    if r.failures:
        print(f"\nFAILED ({len(r.failures)}):")
        for label, detail in r.failures:
            print(f"  FAIL {label}")
            if detail:
                print(f"       {detail}")
        return 1
    print("\nAll submission requirements met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
