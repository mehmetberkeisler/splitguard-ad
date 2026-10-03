"""Unit tests for the leakage graph: union-find, edge rules, component labels.

The graph is the framework's central claim. Every protocol-C split, every
contamination count and the whole provenance argument rest on two properties:
that the union-find computes a genuine transitive closure, and that each edge
rule fires on exactly the pairs it claims to. Both are easy to get subtly
wrong in ways no reported number would reveal -- a union that forgets to
compress a path still returns the right root, and an edge rule that skips
empty values looks identical to one that treats "" as a joinable key until a
cohort arrives with blank session identifiers.
"""

import csv
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


GRAPH = load_module(ROOT / "scripts" / "build_adni_leakage_graph.py")


def scan(image_id, subject, session="", series="", date="", dx="CN"):
    return {"image_id": image_id, "subject_id": subject, "session_id": session,
            "series_uid": series, "acq_date": date, "diagnosis_group": dx}


class UnionFindTests(unittest.TestCase):
    """The data structure, independent of what the edges mean."""

    def test_a_fresh_node_is_its_own_root(self):
        uf = GRAPH.UnionFind()
        uf.add("a")
        self.assertEqual(uf.find("a"), "a")

    def test_union_is_symmetric(self):
        left, right = GRAPH.UnionFind(), GRAPH.UnionFind()
        left.union("a", "b")
        right.union("b", "a")
        self.assertEqual(left.find("a"), left.find("b"))
        self.assertEqual(right.find("a"), right.find("b"))

    def test_union_is_transitive(self):
        uf = GRAPH.UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("d", "e")
        self.assertEqual(uf.find("a"), uf.find("c"), "a~b~c must be one component")
        self.assertNotEqual(uf.find("a"), uf.find("d"), "d~e must stay separate")

    def test_repeated_union_is_idempotent(self):
        uf = GRAPH.UnionFind()
        uf.union("a", "b")
        root = uf.find("a")
        for _ in range(10):
            uf.union("a", "b")
        self.assertEqual(uf.find("a"), root)
        self.assertEqual(uf.find("b"), root)

    def test_path_compression_shortens_paths_without_changing_membership(self):
        # Build a deep chain, then assert that find() both flattens the tree
        # and leaves every node in the same component. A compression bug that
        # reparented a node onto the wrong root would still return quickly.
        uf = GRAPH.UnionFind()
        chain = [f"n{i}" for i in range(50)]
        for a, b in zip(chain, chain[1:]):
            uf.union(a, b)

        root = uf.find(chain[0])
        self.assertEqual({uf.find(n) for n in chain}, {root},
                         "a chain must resolve to exactly one root")

        # After a full pass of find(), every node points at the root directly.
        depth = sum(1 for n in chain if uf.parent[n] != root and uf.parent[n] != n)
        self.assertEqual(depth, 0, "find() must leave every node parented to the root")

    def test_components_match_a_naive_closure_on_random_edges(self):
        # The property that matters, checked against an independent and
        # deliberately slow implementation: union-find must agree with
        # repeated label propagation on arbitrary edge sets.
        rng = random.Random(0)
        for trial in range(30):
            nodes = [f"n{i}" for i in range(rng.randint(2, 25))]
            edges = [(rng.choice(nodes), rng.choice(nodes))
                     for _ in range(rng.randint(0, 40))]
            uf = GRAPH.UnionFind()
            for n in nodes:
                uf.add(n)
            for a, b in edges:
                uf.union(a, b)
            fast = {}
            for n in nodes:
                fast.setdefault(uf.find(n), set()).add(n)

            label = {n: i for i, n in enumerate(nodes)}
            changed = True
            while changed:                     # naive closure
                changed = False
                for a, b in edges:
                    low = min(label[a], label[b])
                    if label[a] != low or label[b] != low:
                        label[a] = label[b] = low
                        changed = True
            slow = {}
            for n in nodes:
                slow.setdefault(label[n], set()).add(n)

            self.assertEqual(sorted(map(sorted, fast.values())),
                             sorted(map(sorted, slow.values())),
                             f"trial {trial}: union-find disagrees with label propagation")


class GroupByTests(unittest.TestCase):
    """Which values count as a joinable key."""

    def test_blank_and_unknown_keys_never_join(self):
        # A cohort with missing session identifiers must not collapse into one
        # component because "" equals "". This is the failure the provenance
        # experiment simulates, so the builder must not create it by accident.
        rows = [scan("i1", "s1", session=""), scan("i2", "s2", session=""),
                scan("i3", "s3", session="unknown"), scan("i4", "s4", session="unknown")]
        self.assertEqual(GRAPH.group_by(rows, "session_id"), {})

    def test_whitespace_is_stripped_before_grouping(self):
        rows = [scan("i1", "s1", session=" v1 "), scan("i2", "s2", session="v1")]
        groups = GRAPH.group_by(rows, "session_id")
        self.assertEqual(groups, {"v1": ["i1", "i2"]})

    def test_grouping_preserves_input_order(self):
        rows = [scan(f"i{i}", "s1") for i in range(5)]
        self.assertEqual(GRAPH.group_by(rows, "subject_id")["s1"],
                         ["i0", "i1", "i2", "i3", "i4"])


class EdgeRuleTests(unittest.TestCase):
    """Each rule fires on exactly the pairs it claims."""

    def components(self, rows):
        uf, counts, _edges = GRAPH.build_graph(rows)
        out = {}
        for row in rows:
            out.setdefault(uf.find(row["image_id"]), set()).add(row["image_id"])
        return sorted(map(sorted, out.values())), counts

    def test_distinct_participants_stay_separate(self):
        rows = [scan("i1", "s1"), scan("i2", "s2"), scan("i3", "s3")]
        comps, counts = self.components(rows)
        self.assertEqual(comps, [["i1"], ["i2"], ["i3"]])
        self.assertEqual(counts["same_subject"], 0)

    def test_same_subject_joins_every_scan_of_one_participant(self):
        rows = [scan("i1", "s1"), scan("i2", "s1"), scan("i3", "s1"), scan("i4", "s2")]
        comps, counts = self.components(rows)
        self.assertEqual(comps, [["i1", "i2", "i3"], ["i4"]])
        self.assertEqual(counts["same_subject"], 2, "n scans of one subject give n-1 edges")

    def test_shared_session_joins_across_participant_identifiers(self):
        # The case the framework exists for: the participant key says these are
        # two people, a non-subject rule says otherwise. Protocol B cannot see
        # this; the graph must.
        rows = [scan("i1", "s1", session="visit-A"), scan("i2", "s2", session="visit-A")]
        comps, counts = self.components(rows)
        self.assertEqual(comps, [["i1", "i2"]])
        self.assertGreaterEqual(counts["same_session"], 1)

    def test_shared_series_uid_joins_across_participants(self):
        rows = [scan("i1", "s1", series="SER9"), scan("i2", "s2", series="SER9")]
        comps, _ = self.components(rows)
        self.assertEqual(comps, [["i1", "i2"]])

    def test_rules_compose_transitively_across_different_edge_types(self):
        # i1-i2 by subject, i2-i3 by session: all three must land together even
        # though no single rule joins i1 to i3.
        rows = [scan("i1", "s1"), scan("i2", "s1", session="v"), scan("i3", "s9", session="v")]
        comps, _ = self.components(rows)
        self.assertEqual(comps, [["i1", "i2", "i3"]])


class ComponentLabelTests(unittest.TestCase):
    """A component's label decides whether it may enter the binary task."""

    def test_uniform_label_is_carried_through(self):
        self.assertEqual(GRAPH.label_component([scan("i1", "s", dx="AD"),
                                                scan("i2", "s", dx="AD")]), "AD")

    def test_conflicting_labels_are_flagged_not_resolved(self):
        # Choosing a majority here is what would silently relabel a converter.
        self.assertEqual(GRAPH.label_component([scan("i1", "s", dx="CN"),
                                                scan("i2", "s", dx="AD")]), "mixed_label")

    def test_unknown_alone_is_missing_not_mixed(self):
        self.assertEqual(GRAPH.label_component([scan("i1", "s", dx="unknown")]),
                         "label_missing")

    def test_unknown_does_not_make_a_uniform_component_mixed(self):
        self.assertEqual(GRAPH.label_component([scan("i1", "s", dx="AD"),
                                                scan("i2", "s", dx="unknown")]), "AD")


if __name__ == "__main__":
    unittest.main()


# ── The generic builder's missing-identifier contract ──────────────────────

GENERIC = load_module(ROOT / "scripts" / "build_current_leakage_graph.py")


class MissingIdentifierTests(unittest.TestCase):
    """Absence of evidence must not become evidence of identity.

    The generic builder accepts a bring-your-own-cohort manifest, where a
    subject or hash column can be blank or carry a placeholder. Grouping on
    those values directly would merge every unlabelled scan into one
    component, and the audit would then pass the resulting split: the
    partitions really are component-disjoint, because the component is
    fictional. That is the failure mode this paper exists to measure, arriving
    through the tool meant to prevent it.
    """

    def test_blank_and_placeholder_values_normalise_to_absent(self):
        for value in ("", "   ", None, "NA", "n/a", "N/A", "none", "null",
                      "unknown", "Unknown", "NaN", "-", "?"):
            self.assertIsNone(GENERIC.normalise_identifier(value),
                              f"{value!r} should count as absent")

    def test_real_identifiers_survive_and_are_stripped(self):
        self.assertEqual(GENERIC.normalise_identifier("999_S_9999"), "999_S_9999")
        self.assertEqual(GENERIC.normalise_identifier("  999_S_9999  "), "999_S_9999")
        self.assertEqual(GENERIC.normalise_identifier(42), "42")

    def test_a_value_that_merely_contains_a_token_is_kept(self):
        # "unknown_site_7" is an identifier, not a missing value.
        for value in ("unknown_site_7", "NAT-01", "none_of_the_above"):
            self.assertEqual(GENERIC.normalise_identifier(value), value)

    def test_records_with_absent_subjects_do_not_form_a_component(self):
        rec = GENERIC.Record
        fields = {f: "" for f in rec.__dataclass_fields__}
        def make(image_id, subject, sha):
            return rec(**{**fields, "image_id": image_id,
                          "subject_id": subject, "file_sha256": sha})
        records = [make("i1", "", "h1"), make("i2", "", "h2"),
                   make("i3", "unknown", "h3"), make("i4", "s9", "h4"),
                   make("i5", "s9", "h5")]
        uf, *_ = (GENERIC.build_blocking_components(records)
                  if callable(getattr(GENERIC, "build_blocking_components", None))
                  else (None,))
        roots = {r.image_id: uf.find(r.image_id) for r in records}
        self.assertNotEqual(roots["i1"], roots["i2"],
                            "two scans with no subject are not one patient")
        self.assertNotEqual(roots["i1"], roots["i3"],
                            "blank and 'unknown' are not the same patient")
        self.assertEqual(roots["i4"], roots["i5"],
                         "a real shared identifier must still join")


class ManifestContractTests(unittest.TestCase):
    """The README promises image_id is the only required column.

    The implementation demanded ten. A user bringing their own cohort would
    hit a wall of columns the documentation never mentioned, which is the
    opposite of the auditability the paper argues for.
    """

    def write(self, tmp, header, rows):
        path = Path(tmp) / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def test_a_manifest_of_image_id_alone_is_accepted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, ["image_id"], [{"image_id": f"i{i}"} for i in range(3)])
            records = GENERIC.read_manifest(path)
        self.assertEqual([r.image_id for r in records], ["i0", "i1", "i2"])
        self.assertEqual({r.subject_id for r in records}, {""})

    def test_a_manifest_without_image_id_is_rejected_by_name(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, ["subject_id"], [{"subject_id": "s1"}])
            with self.assertRaises(ValueError) as ctx:
                GENERIC.read_manifest(path)
        self.assertIn("image_id", str(ctx.exception))

    def test_an_absent_column_switches_its_edge_family_off(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, ["image_id"], [{"image_id": f"i{i}"} for i in range(4)])
            records = GENERIC.read_manifest(path)
        uf, *_ = GENERIC.build_blocking_components(records)
        roots = {uf.find(r.image_id) for r in records}
        self.assertEqual(len(roots), 4,
                         "with no identifiers every image is its own component")
