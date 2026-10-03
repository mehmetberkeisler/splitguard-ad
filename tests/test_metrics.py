"""Unit tests for the two measurements every claim in the paper reduces to.

AUROC is the reported quantity, and residual leakage is what the protocols are
compared on. Both have a property that makes silent failure easy: AUROC's
plausible range covers its own complement, so an orientation flip reads as a
mediocre model rather than a bug (it shipped once, in the volumetric trainer);
and residual leakage returns a small number whether the split is clean or the
counting is wrong.
"""

import importlib.util
import math
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


POST = load_module(ROOT / "scripts" / "gpu_postprocess.py")
DEGRADE = load_module(ROOT / "scripts" / "run_provenance_degradation.py")


def mann_whitney_auc(pairs):
    """AUROC from first principles, for use as an independent oracle.

    Counts concordant pairs directly rather than ranking, so it shares no code
    path with the implementation under test.
    """
    pos = [s for y, s in pairs if y == 1]
    neg = [s for y, s in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


class AurocTests(unittest.TestCase):

    def test_perfect_ranking_is_one(self):
        self.assertEqual(POST.auroc([(0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)]), 1.0)

    def test_perfectly_inverted_ranking_is_zero(self):
        # The volumetric trainer returned this value for a good model.
        self.assertEqual(POST.auroc([(1, 0.1), (1, 0.2), (0, 0.8), (0, 0.9)]), 0.0)

    def test_constant_scores_are_one_half(self):
        self.assertEqual(POST.auroc([(0, 0.5), (1, 0.5), (0, 0.5), (1, 0.5)]), 0.5)

    def test_ties_take_the_mid_rank_not_an_arbitrary_order(self):
        # Scores {0.1, 0.5} negative and {0.5, 0.9} positive: three of the four
        # pairs are concordant and one is tied, so 3.5/4. Sorting the tied pair
        # one way gives 1.0 and the other 0.75; the mid-rank gives neither.
        self.assertAlmostEqual(POST.auroc([(0, 0.1), (0, 0.5), (1, 0.5), (1, 0.9)]),
                               0.875, places=12)

    def test_single_class_is_undefined_rather_than_guessed(self):
        # nan, not 0.5: a partition with one class present has no AUROC, and
        # returning the chance value would let it average into a reported mean.
        self.assertTrue(math.isnan(POST.auroc([(1, 0.1), (1, 0.9)])))
        self.assertTrue(math.isnan(POST.auroc([(0, 0.1), (0, 0.9)])))

    def test_matches_an_independent_oracle_on_random_inputs_with_ties(self):
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(2, 40)
            pairs = [(rng.randint(0, 1), round(rng.random(), 1)) for _ in range(n)]
            if len({y for y, _ in pairs}) < 2:
                continue
            self.assertAlmostEqual(POST.auroc(pairs), mann_whitney_auc(pairs), places=12)

    def test_is_invariant_to_monotone_rescaling_of_scores(self):
        rng = random.Random(11)
        pairs = [(rng.randint(0, 1), rng.random()) for _ in range(40)]
        rescaled = [(y, math.exp(3 * s) + 1) for y, s in pairs]
        self.assertAlmostEqual(POST.auroc(pairs), POST.auroc(rescaled), places=12)

    def test_is_invariant_to_input_order(self):
        rng = random.Random(3)
        pairs = [(rng.randint(0, 1), round(rng.random(), 2)) for _ in range(30)]
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        self.assertAlmostEqual(POST.auroc(pairs), POST.auroc(shuffled), places=12)


def rows_with(assignment):
    """One scan per entry: (image_id, true_subject, partition)."""
    rows, mapping = [], {}
    for image_id, subject, phase in assignment:
        rows.append({"image_id": image_id, DEGRADE.TRUE_SUBJECT: subject,
                     "subject_id": subject})
        mapping[image_id] = phase
    return rows, mapping


class ResidualLeakageTests(unittest.TestCase):

    def test_a_clean_split_admits_nothing(self):
        rows, assign = rows_with([("i1", "s1", "train"), ("i2", "s1", "train"),
                                  ("i3", "s2", "test"), ("i4", "s2", "test")])
        got = DEGRADE.residual_subject_leakage(rows, assign)
        self.assertEqual(got["n_subjects_straddling_partitions"], 0)
        self.assertEqual(got["test_scan_contamination"], 0.0)
        self.assertEqual(got["test_train_subject_overlap"], 0.0)

    def test_one_participant_on_both_sides_is_counted_once(self):
        rows, assign = rows_with([("i1", "s1", "train"), ("i2", "s1", "test"),
                                  ("i3", "s2", "test")])
        got = DEGRADE.residual_subject_leakage(rows, assign)
        self.assertEqual(got["n_subjects_straddling_partitions"], 1)
        self.assertEqual(got["n_images_in_straddling_subjects"], 2,
                         "both scans of the straddling participant count")
        self.assertEqual(got["n_test_subjects_also_in_train"], 1)

    def test_scan_share_and_participant_share_are_different_quantities(self):
        # One contaminated participant with a single test scan, against a clean
        # participant with many: the participant share is high and the scan
        # share is low. The paper composes the dose slope through the scan
        # share precisely because it is the smaller, more conservative one.
        rows, assign = rows_with(
            [("i0", "s1", "train"), ("i1", "s1", "test")]
            + [(f"j{k}", "s2", "test") for k in range(9)])
        got = DEGRADE.residual_subject_leakage(rows, assign)
        self.assertEqual(got["test_train_subject_overlap"], 0.5, "1 of 2 test participants")
        self.assertEqual(got["test_scan_contamination"], 0.1, "1 of 10 test scans")
        self.assertLess(got["test_scan_contamination"], got["test_train_subject_overlap"])

    def test_straddling_across_train_and_validation_counts_as_straddling(self):
        # A participant split between train and val never touches the test
        # partition, so it contaminates nothing, but the split still failed to
        # hold them together. Both facts are reported separately.
        rows, assign = rows_with([("i1", "s1", "train"), ("i2", "s1", "val"),
                                  ("i3", "s2", "test")])
        got = DEGRADE.residual_subject_leakage(rows, assign)
        self.assertEqual(got["n_subjects_straddling_partitions"], 1)
        self.assertEqual(got["test_scan_contamination"], 0.0)

    def test_a_fully_contaminated_split_reports_one(self):
        rows, assign = rows_with([("i1", "s1", "train"), ("i2", "s1", "test"),
                                  ("i3", "s2", "train"), ("i4", "s2", "test")])
        got = DEGRADE.residual_subject_leakage(rows, assign)
        self.assertEqual(got["test_scan_contamination"], 1.0)
        self.assertEqual(got["test_train_subject_overlap"], 1.0)

    def test_an_empty_test_partition_does_not_divide_by_zero(self):
        rows, assign = rows_with([("i1", "s1", "train"), ("i2", "s2", "train")])
        got = DEGRADE.residual_subject_leakage(rows, assign)
        self.assertEqual(got["test_scan_contamination"], 0.0)
        self.assertEqual(got["n_test_scans"], 0)


if __name__ == "__main__":
    unittest.main()
