"""Structured provenance-corruption operators: drop, split, merge.

These are controlled stress tests of identifiable provenance failure modes.
They are not estimates of how often such errors occur in clinical
repositories, and the manuscript says so where it reports them.

Each operator must leave the cohort and its ground truth untouched and change
only the observed identifier, which the test suite asserts: no scan created or
lost, no label modified, intensity zero an exact identity, and determinism
under a given seed. The last matters because Python salts ``hash()`` per
process, so any operator reaching for it would be irreproducible across runs.
"""

from __future__ import annotations

from collections import defaultdict

from .metrics import TRUE_SUBJECT

__all__ = ["corrupt_drop", "corrupt_split", "corrupt_merge", "MECHANISMS"]


def _tag(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Copy rows and record untouched ground truth before any corruption."""
    out = []
    for row in rows:
        new = dict(row)
        new[TRUE_SUBJECT] = row["subject_id"]
        out.append(new)
    return out


def corrupt_drop(rows, intensity, rng):
    """Each scan independently loses its subject identifier."""
    out = _tag(rows)
    for row in out:
        if rng.random() < intensity:
            row["subject_id"] = "unknown"
    return out


def corrupt_split(rows, intensity, rng):
    """One participant is dealt into two pseudo-identifiers.

    The scans keep an identifier, so nothing looks missing; a subject-wise
    splitter simply believes there are two patients where there is one. This is
    the failure the Tier-1 filename key exhibits for 179 of its 200
    participants.
    """
    out = _tag(rows)
    by_subject = defaultdict(list)
    for row in out:
        by_subject[row[TRUE_SUBJECT]].append(row)
    for subject, subject_rows in by_subject.items():
        if len(subject_rows) < 2 or rng.random() >= intensity:
            continue
        shuffled = subject_rows[:]
        rng.shuffle(shuffled)
        cut = max(1, len(shuffled) // 2)
        for row in shuffled[:cut]:
            row["subject_id"] = f"{subject}__a"
        for row in shuffled[cut:]:
            row["subject_id"] = f"{subject}__b"
    return out


def corrupt_merge(rows, intensity, rng):
    """Two participants are relabelled to one pseudo-identifier.

    Again nothing is missing. A subject-wise splitter over-groups, which is
    conservative for leakage but destroys the participant count; the leakage
    graph inherits the same wrong key, so this mechanism is expected to hurt
    both protocols equally. Reporting a mechanism where the graph does not help
    is the point of running three.
    """
    out = _tag(rows)
    subjects = sorted({row[TRUE_SUBJECT] for row in out})
    rng.shuffle(subjects)
    merged: dict[str, str] = {}
    # Pairs the shuffled subjects. With an odd count the two slices differ in
    # length by one and the leftover subject is deliberately left unmerged,
    # so strict=True would raise rather than catch anything.
    for left, right in zip(subjects[0::2], subjects[1::2]):  # noqa: B905
        if rng.random() < intensity:
            merged[left] = merged[right] = f"{left}+{right}"
    for row in out:
        if row[TRUE_SUBJECT] in merged:
            row["subject_id"] = merged[row[TRUE_SUBJECT]]
    return out


MECHANISMS = {"drop": corrupt_drop, "split": corrupt_split, "merge": corrupt_merge}
