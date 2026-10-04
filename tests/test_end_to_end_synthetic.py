"""The whole framework on a synthetic cohort, with no ADNI or OASIS access.

Every other test here checks one stage. This one walks the chain the paper
rests on, manifest to graph to components to split to audit, on twelve images
nobody needs permission to read. It is the test a reviewer can run on a laptop
to see that SplitGuard does what the manuscript says, and the only test that
would fail if two correct stages were wired together wrongly.

The cohort is small enough to reason about by hand:

    20 participants, 10 CN and 10 AD, so the classes are balanced
    most have 2 sessions, four have 3, so component sizes vary
    44 images in total
    p05 carries a deliberate provenance fault: its second session was
      registered under a fresh participant key, so the identifier column says
      21 participants where the cohort has 20

The first draft of this test used the twelve images a review suggested, and
every one of them landed in train: with six images per class the validation
target rounds to one image while the smallest component holds two, so the
splitter correctly allocated nothing. The audit then returned GO because there
was no boundary for anything to straddle, and the test passed while proving
nothing. The cohort is sized so all three partitions are non-empty, and
``test_all_three_partitions_are_used`` fails if that ever stops being true.

That last row is the paper's subject in miniature. Grouping on the supplied
participant key puts p05's two sessions in different components, which is how a
careful user who grouped by the only identifier available still ends up with
the same head on both sides of a split. The session key survives, so the graph
recovers the relation; that is regime II, degraded with redundancy.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splitguard_ad import (  # noqa: E402
    AuditReport,
    Finding,
    Level,
    UnionFind,
    assign_splits,
    build_components,
    normalise_identifier,
)

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


# Four participants are imaged three times rather than twice, so component
# sizes differ. Without that the largest-first rule has nothing to discriminate
# on, the seeded tie-break is never reached, and every seed gives one answer.
LONGITUDINAL = {"p03", "p08", "p12", "p17"}


def synthetic_manifest() -> list[dict[str, str]]:
    """Forty-four images over twenty participants, with one fragmented key."""
    rows = []
    labels = {f"p{i:02d}": ("CN" if i <= 10 else "AD") for i in range(1, 21)}
    for participant, label in labels.items():
        for session in range(1, 4 if participant in LONGITUDINAL else 3):
            # The fault: p05's second session was filed under a new key.
            observed = "p21_orphan" if (participant == "p05" and session == 2) else participant
            rows.append({
                "image_id": f"{participant}_s{session}",
                "subject_id": observed,
                "session_id": f"{participant}_visit{session}",
                "true_participant": participant,
                "raw_class_label": label,
                "binary_label": label,
            })
    return rows


def build_graph(rows, use_session_edges: bool):
    """Join images on surviving identity evidence; return image -> component."""
    uf = UnionFind([r["image_id"] for r in rows])
    by_subject, by_session = {}, {}
    for row in rows:
        subject = normalise_identifier(row.get("subject_id"))
        if subject is not None:
            if subject in by_subject:
                uf.union(by_subject[subject], row["image_id"], "same_subject")
            by_subject[subject] = row["image_id"]
        if use_session_edges:
            # One participant's sessions share a visit stem, which is the
            # redundancy the fragmented participant key no longer carries.
            stem = (normalise_identifier(row.get("session_id")) or "").split("_visit")[0]
            if stem:
                if stem in by_session:
                    uf.union(by_session[stem], row["image_id"], "same_session_stem")
                by_session[stem] = row["image_id"]
    return {r["image_id"]: uf.find(r["image_id"]) for r in rows}


def component_rows(rows, membership):
    return [{
        "image_id": r["image_id"],
        "component_id": membership[r["image_id"]],
        "subject_id": r["subject_id"],
        "raw_class_label": r["raw_class_label"],
        "binary_label": r["binary_label"],
        "component_primary_reason": "same_subject",
    } for r in rows]


def audit(rows, membership, assignment) -> AuditReport:
    """The one blocking condition, plus advisory composition findings."""
    findings = []
    phases = {}
    for row in rows:
        phases.setdefault(membership[row["image_id"]], set()).add(
            assignment[membership[row["image_id"]]])
    straddling = sorted(c for c, p in phases.items() if len(p) > 1)
    findings.append(Finding(Level.FAIL, "component straddles a partition boundary",
                            ", ".join(straddling))
                    if straddling else
                    Finding(Level.PASS, "no component straddles a partition boundary"))

    # A participant on both sides is leakage even when every component is whole,
    # which is exactly what the fragmented key causes.
    by_true = {}
    for row in rows:
        by_true.setdefault(row["true_participant"], set()).add(
            assignment[membership[row["image_id"]]])
    leaked = sorted(p for p, ph in by_true.items() if len(ph) > 1)
    findings.append(Finding(Level.FAIL, "true participant appears in two partitions",
                            ", ".join(leaked))
                    if leaked else
                    Finding(Level.PASS, "every true participant is confined to one partition"))

    counts = {}
    for row in rows:
        counts[assignment[membership[row["image_id"]]]] = \
            counts.get(assignment[membership[row["image_id"]]], 0) + 1
    findings.append(Finding(Level.INFO, "images by partition", json.dumps(counts, sort_keys=True)))
    return AuditReport(findings)


def run(use_session_edges: bool, seed: int = 0):
    rows = synthetic_manifest()
    membership = build_graph(rows, use_session_edges)
    components = build_components(component_rows(rows, membership))
    assignment, targets = assign_splits(components, seed=seed, ratios=RATIOS)
    return rows, membership, components, assignment, audit(rows, membership, assignment)


class SyntheticManifestTests(unittest.TestCase):
    """1 and 2: the manifest parses and the graph covers it."""

    def test_the_manifest_is_the_cohort_it_claims_to_be(self):
        rows = synthetic_manifest()
        expected = 2 * 20 + len(LONGITUDINAL)      # two sessions each, four have three
        self.assertEqual(len(rows), expected)
        self.assertEqual(len({r["true_participant"] for r in rows}), 20)
        self.assertEqual(sorted({r["binary_label"] for r in rows}), ["AD", "CN"])
        counts = {lab: sum(1 for r in rows if r["binary_label"] == lab)
                  for lab in ("CN", "AD")}
        self.assertEqual(counts["CN"], counts["AD"], "the synthetic classes must be balanced")

    def test_the_supplied_identifier_disagrees_with_the_truth(self):
        rows = synthetic_manifest()
        truth = len({r["true_participant"] for r in rows})
        self.assertEqual(len({r["subject_id"] for r in rows}), truth + 1,
                         "the fragmented key should report one participant too many")

    def test_every_image_lands_in_exactly_one_component(self):
        rows, membership, *_ = run(use_session_edges=True)
        self.assertEqual(set(membership), {r["image_id"] for r in rows})


class LeakageEdgeTests(unittest.TestCase):
    """3 and 4: the known relation is found, and the components are right."""

    def test_grouping_on_the_broken_key_alone_fragments_a_participant(self):
        _, membership, components, _, _ = run(use_session_edges=False)
        self.assertEqual(len(components), 21,
                         "the fragmented key should yield one component too many")
        self.assertNotEqual(membership["p05_s1"], membership["p05_s2"],
                            "p05's two sessions must be split by the broken key")

    def test_the_surviving_session_relation_recovers_the_participant(self):
        _, membership, components, _, _ = run(use_session_edges=True)
        self.assertEqual(len(components), 20, "the graph should recover twenty participants")
        self.assertEqual(membership["p05_s1"], membership["p05_s2"],
                         "the session edge must rejoin p05")

    def test_each_component_holds_exactly_one_participant_and_one_label(self):
        _, _, components, _, _ = run(use_session_edges=True)
        multi_key = [c for c in components if len(c["subject_ids"]) > 1]
        self.assertEqual(len(multi_key), 1,
                         "exactly one component should span two keys: the repaired one")
        self.assertEqual(sorted(multi_key[0]["subject_ids"]), ["p05", "p21_orphan"],
                         "and it should be p05 rejoined to its orphaned session")
        for c in components:
            self.assertIn(c["n_images"], (2, 3))
            self.assertIn(c["binary_label"], ("CN", "AD"))


class SplitIntegrityTests(unittest.TestCase):
    """5, 6 and 7: nothing straddles, the audit decides, diagnostics exist."""

    def test_no_component_appears_in_more_than_one_partition(self):
        rows, membership, _, assignment, _ = run(use_session_edges=True)
        seen = {}
        for row in rows:
            comp = membership[row["image_id"]]
            seen.setdefault(comp, set()).add(assignment[comp])
        straddling = [c for c, p in seen.items() if len(p) > 1]
        self.assertEqual(straddling, [])

    def test_the_audit_returns_go_when_the_graph_recovers_the_participant(self):
        *_, report = run(use_session_edges=True)
        self.assertEqual(report.decision, "GO", report.render())
        self.assertTrue(report.ok)
        self.assertEqual(report.failures, [])

    def test_the_audit_refuses_when_a_participant_crosses_the_boundary(self):
        """The negative case. Without it, GO above would prove nothing.

        Grouping on the broken key alone leaves p05 as two components, so the
        two halves of one head *may* be allocated to different partitions. On
        this cohort the allocator does not in fact do that at any seed in 200,
        because the fragments are single-image components and the chooser fills
        validation and test from larger ones. Waiting for luck would make this
        test silently vacuous, so the leak is constructed instead: the audit is
        handed the arrangement the broken key permits and must refuse it.
        """
        rows = synthetic_manifest()
        membership = build_graph(rows, use_session_edges=False)
        first, second = membership["p05_s1"], membership["p05_s2"]
        self.assertNotEqual(first, second,
                            "the broken key must leave p05 as two components")

        # Everything in train, except the orphaned half of p05 in test.
        assignment = dict.fromkeys(set(membership.values()), "train")
        assignment[second] = "test"

        report = audit(rows, membership, assignment)
        self.assertEqual(report.decision, "NO-GO", report.render())
        self.assertTrue(any("true participant" in f.check for f in report.failures),
                        "the refusal must name the participant-level leak")

    def test_the_same_arrangement_is_clean_once_the_graph_repairs_it(self):
        """The mirror of the test above, and the paper's claim in one assertion.

        The same allocation that leaks under the broken key cannot leak once
        the surviving session relation has rejoined p05, because there is then
        no second component to place on the other side.
        """
        rows = synthetic_manifest()
        membership = build_graph(rows, use_session_edges=True)
        self.assertEqual(membership["p05_s1"], membership["p05_s2"])
        assignment = dict.fromkeys(set(membership.values()), "train")
        assignment[membership["p05_s2"]] = "test"
        report = audit(rows, membership, assignment)
        self.assertEqual(report.decision, "GO", report.render())

    def test_all_three_partitions_are_used(self):
        """The guard on this whole class.

        If validation or test came out empty, no component could straddle a
        boundary, the audit would return GO for the wrong reason, and several
        tests above would pass while proving nothing. That is what the first
        draft of this file did.
        """
        rows, membership, _, assignment, _ = run(use_session_edges=True)
        used = {assignment[membership[r["image_id"]]] for r in rows}
        self.assertEqual(used, {"train", "val", "test"},
                         "a partition is empty, so the split tests are vacuous")

    def test_diagnostics_are_produced(self):
        *_, report = run(use_session_edges=True)
        info = report.at(Level.INFO)
        self.assertTrue(info, "the audit should report descriptive diagnostics")
        counts = json.loads(info[0].detail)
        self.assertEqual(sum(counts.values()), len(synthetic_manifest()),
                         "the diagnostics must account for every image")

    def test_composition_warnings_are_advisory_and_never_block(self):
        rows, membership, _, assignment, report = run(use_session_edges=True)
        warned = AuditReport(report.findings + [Finding(Level.WARN, "class mix spread 0.20")])
        self.assertEqual(warned.decision, "GO",
                         "a warning must not turn a valid split into a refusal")


class DeterminismTests(unittest.TestCase):
    """8: the same seed gives the same answer, twice."""

    def test_repeated_execution_is_identical(self):
        first = run(use_session_edges=True, seed=0)
        second = run(use_session_edges=True, seed=0)
        self.assertEqual(first[3], second[3], "assignment must be reproducible")
        self.assertEqual(first[1], second[1], "membership must be reproducible")

    def test_a_different_seed_can_give_a_different_assignment(self):
        base = run(use_session_edges=True, seed=0)[3]
        others = [run(use_session_edges=True, seed=s)[3] for s in range(1, 8)]
        self.assertTrue(any(o != base for o in others),
                        "every seed produced the same split, so the seed does nothing")


if __name__ == "__main__":
    unittest.main()
