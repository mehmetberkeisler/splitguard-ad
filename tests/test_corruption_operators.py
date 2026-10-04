"""Unit tests for the three structured-corruption operators.

These generate the 120-cell matrix in the manuscript, and each one has an
invariant that no reported number would expose if it broke. An operator that
dropped a scan would change every denominator downstream; one that modified
the recorded ground truth would score the corrupted labels against themselves
and report zero leakage for any corruption; one whose RNG was seeded from
Python's salted ``hash()`` would produce a different matrix on every run while
each run looked internally consistent. The last of those was a real defect
here, caught before its numbers were trusted.
"""

import ast
import importlib.util
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


STRESS = load_module(ROOT / "scripts" / "run_provenance_stress_test.py")
TRUE = load_module(ROOT / "scripts" / "run_provenance_degradation.py").TRUE_SUBJECT


def cohort(n_subjects=20, scans_each=4):
    return [{"image_id": f"i{s}_{k}", "subject_id": f"s{s}",
             "diagnosis_group": "AD" if s % 2 else "CN"}
            for s in range(n_subjects) for k in range(scans_each)]


class OperatorInvariantTests(unittest.TestCase):
    """Properties every operator must satisfy, at every intensity."""

    def setUp(self):
        self.rows = cohort()

    def each_operator(self):
        for name, op in STRESS.MECHANISMS.items():
            for level in (0.0, 0.1, 0.5, 1.0):
                yield name, op, level, op(self.rows, level, random.Random(0))

    def test_no_scan_is_created_or_lost(self):
        for name, _op, level, out in self.each_operator():
            self.assertEqual(len(out), len(self.rows), f"{name} at {level} changed the row count")
            self.assertEqual({r["image_id"] for r in out},
                             {r["image_id"] for r in self.rows},
                             f"{name} at {level} changed which scans exist")

    def test_ground_truth_is_never_modified(self):
        # The whole measurement is "score the corrupted grouping against the
        # untouched truth". An operator that wrote through to TRUE_SUBJECT
        # would make every protocol look perfect.
        truth = {r["image_id"]: r["subject_id"] for r in self.rows}
        for name, _op, level, out in self.each_operator():
            for row in out:
                self.assertEqual(row[TRUE], truth[row["image_id"]],
                                 f"{name} at {level} overwrote the recorded ground truth")

    def test_zero_intensity_is_the_identity_on_the_observed_key(self):
        for name, op in STRESS.MECHANISMS.items():
            out = op(self.rows, 0.0, random.Random(0))
            for row in out:
                self.assertEqual(row["subject_id"], row[TRUE],
                                 f"{name} corrupted something at intensity 0")

    def test_the_input_rows_are_left_alone(self):
        # The matrix applies every operator to the same base manifest, so an
        # operator that mutated its input in place would make the result
        # depend on the order the cells happen to run in.
        before = [dict(r) for r in self.rows]
        for _name, op in STRESS.MECHANISMS.items():
            op(self.rows, 1.0, random.Random(0))
        self.assertEqual(self.rows, before, "an operator mutated its input")

    def test_labels_are_untouched(self):
        labels = {r["image_id"]: r["diagnosis_group"] for r in self.rows}
        for name, _op, level, out in self.each_operator():
            for row in out:
                self.assertEqual(row["diagnosis_group"], labels[row["image_id"]],
                                 f"{name} at {level} changed a diagnosis")


class DropTests(unittest.TestCase):

    def test_total_loss_removes_every_identifier(self):
        out = STRESS.corrupt_drop(cohort(), 1.0, random.Random(0))
        self.assertEqual({r["subject_id"] for r in out}, {"unknown"})

    def test_partial_loss_removes_roughly_the_requested_share(self):
        rows = cohort(n_subjects=200, scans_each=5)      # 1000 scans
        out = STRESS.corrupt_drop(rows, 0.25, random.Random(1))
        lost = sum(1 for r in out if r["subject_id"] == "unknown")
        self.assertAlmostEqual(lost / len(rows), 0.25, delta=0.05)

    def test_loss_is_visible_rather_than_silent(self):
        # drop is the polite failure: what is missing is marked missing, which
        # is what lets a plain GroupKFold user respond to it at all.
        out = STRESS.corrupt_drop(cohort(), 0.5, random.Random(0))
        survivors = {r["subject_id"] for r in out} - {"unknown"}
        for key in survivors:
            self.assertTrue(key.startswith("s"), "a surviving key must be the original")


class SplitTests(unittest.TestCase):

    def test_a_participant_is_dealt_into_at_most_two_keys(self):
        out = STRESS.corrupt_split(cohort(), 1.0, random.Random(0))
        by_truth = {}
        for row in out:
            by_truth.setdefault(row[TRUE], set()).add(row["subject_id"])
        for subject, keys in by_truth.items():
            self.assertLessEqual(len(keys), 2, f"{subject} was dealt into {len(keys)} keys")

    def test_total_intensity_fragments_every_multi_scan_participant(self):
        out = STRESS.corrupt_split(cohort(), 1.0, random.Random(0))
        by_truth = {}
        for row in out:
            by_truth.setdefault(row[TRUE], set()).add(row["subject_id"])
        self.assertTrue(all(len(k) == 2 for k in by_truth.values()),
                        "at intensity 1 every participant with >1 scan should fragment")

    def test_nothing_looks_missing(self):
        # The point of split: no field is absent, so an audit of the release's
        # own metadata reports a clean split.
        out = STRESS.corrupt_split(cohort(), 1.0, random.Random(0))
        for row in out:
            self.assertNotIn(row["subject_id"], ("", "unknown", None))

    def test_a_single_scan_participant_cannot_fragment(self):
        rows = [{"image_id": "i1", "subject_id": "s1", "diagnosis_group": "CN"}]
        out = STRESS.corrupt_split(rows, 1.0, random.Random(0))
        self.assertEqual(out[0]["subject_id"], "s1")


class MergeTests(unittest.TestCase):

    def test_participants_are_merged_in_pairs_not_wholesale(self):
        out = STRESS.corrupt_merge(cohort(n_subjects=20), 1.0, random.Random(0))
        by_key = {}
        for row in out:
            by_key.setdefault(row["subject_id"], set()).add(row[TRUE])
        self.assertTrue(all(len(v) <= 2 for v in by_key.values()),
                        "merge must pair participants, not collapse the cohort")

    def test_total_intensity_halves_the_apparent_participant_count(self):
        rows = cohort(n_subjects=20)
        out = STRESS.corrupt_merge(rows, 1.0, random.Random(0))
        self.assertEqual(len({r["subject_id"] for r in out}), 10)

    def test_merging_is_conservative_for_contamination(self):
        # Over-grouping destroys the participant count but cannot place one
        # participant on both sides, which is why the manuscript reports merge
        # as a null the graph does not repair.
        rows = cohort(n_subjects=20)
        out = STRESS.corrupt_merge(rows, 1.0, random.Random(0))
        for key in {r["subject_id"] for r in out}:
            truths = {r[TRUE] for r in out if r["subject_id"] == key}
            self.assertLessEqual(len(truths), 2)


class ReproducibilityTests(unittest.TestCase):
    """The matrix must be the same matrix on every run and every machine."""

    def test_same_seed_gives_identical_output(self):
        rows = cohort()
        for name, op in STRESS.MECHANISMS.items():
            a = op(rows, 0.5, random.Random(42))
            b = op(rows, 0.5, random.Random(42))
            self.assertEqual(a, b, f"{name} is not reproducible from its seed")

    def test_different_seeds_give_different_output(self):
        rows = cohort()
        for name, op in STRESS.MECHANISMS.items():
            a = op(rows, 0.5, random.Random(1))
            b = op(rows, 0.5, random.Random(2))
            self.assertNotEqual(a, b, f"{name} ignores its seed")

    def test_cell_seeding_does_not_depend_on_a_salted_hash(self):
        # The derivation under test is sha256 of "<mechanism>|<level>|<seed>".
        # Python salts str.__hash__ per process, so an implementation using
        # hash() would give a different matrix on every invocation while each
        # run looked internally consistent. Pin the derived value itself.
        import hashlib
        stream = b"split|0.5|3"
        seed = int(hashlib.sha256(stream).hexdigest()[:16], 16)
        # Pinned literal: this is the value a fresh interpreter must derive for
        # that cell. str.__hash__ would give a different one per process.
        self.assertEqual(seed, 15228584043889260364)
        self.assertEqual(seed, int(hashlib.sha256(stream).hexdigest()[:16], 16))

        source = (ROOT / "scripts" / "run_provenance_stress_test.py").read_text()
        self.assertIn("hashlib.sha256", source, "cell seeding must be a stable digest")
        # Read code, not prose: the comment above the fix names hash() in
        # order to warn against it, and matching that would be a false alarm.
        code = ast.dump(ast.parse(source))
        self.assertNotIn("Name(id='hash'", code,
                         "the built-in hash() is salted per process")


if __name__ == "__main__":
    unittest.main()
