"""Unit tests for the release boundary: the ADNI data-use agreement, in code.

These rules are otherwise enforced by prose in the README and by a .gitignore,
neither of which fails a build. The consequence of getting one wrong is not a
wrong number in a table; it is redistributing participant-level data from a
cohort whose agreement forbids it, in a way that cannot be undone once the
repository is public.

The salt is deliberately not read here. These tests construct their own, so
the suite runs on a machine that has never held the real one.

Participant identifiers in these fixtures are synthetic by the same logic. A
PTID-shaped literal is all the hashing needs, and a real one would disclose
that that participant is in the cohort, which is the participant-level fact
the agreement restricts. Site 999 does not exist, so 999_S_9999 cannot
collide with anyone.
"""

import csv
import importlib.util
import json
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


TIER3 = load_module(ROOT / "scripts" / "build_hashed_manifest_tier3.py")

# The boundary is tier-dependent, exactly as the manuscript states it, and a
# blanket rule would be wrong in both directions. Tier 1 is a public benchmark
# whose filename key IS the thing under study, so publishing it is the point;
# what is withheld there is the recovered mapping to OASIS-1. Tier 2 ships the
# public OASIS codes but no paths and no clinical fields. Tier 3 ships nothing
# that is not hashed or aggregate.
TIER3_ALLOWED = {"subject_id_hash", "component_id", "component_size",
                 "binary_label", "scanner_field_strength", "modality", "split"}
TIER2_FORBIDDEN = {"cdr", "mmse", "educ", "education", "ses", "age", "sex",
                   "image_path", "relative_path", "path"}
TIER1_FORBIDDEN = {"oasis_subject_id", "oasis_session_id", "true_participant",
                   "ptid", "rid"}


class SubjectHashingTests(unittest.TestCase):
    """Key stretching, and the properties that make it worth doing."""

    SALT = "a-test-salt-not-the-release-one"

    def test_hashing_is_deterministic_under_one_salt(self):
        a = TIER3.hash_subject_id("999_S_9999", self.SALT)
        b = TIER3.hash_subject_id("999_S_9999", self.SALT)
        self.assertEqual(a, b, "the same participant must hash the same way within a release")

    def test_distinct_participants_do_not_collide(self):
        # Deliberately few: PBKDF2 at 600,000 iterations costs ~0.1 s per call
        # by design, and a slow suite is a suite that stops being run.
        digests = {TIER3.hash_subject_id(f"{i:03d}_S_{i:04d}", self.SALT) for i in range(8)}
        self.assertEqual(len(digests), 8)

    def test_changing_the_salt_changes_every_digest(self):
        # This is what makes the salt load-bearing: without it, the ~10^7 PTID
        # space is enumerable and the released hashes are invertible.
        a = TIER3.hash_subject_id("999_S_9999", self.SALT)
        b = TIER3.hash_subject_id("999_S_9999", self.SALT + "x")
        self.assertNotEqual(a, b)

    def test_the_digest_does_not_contain_the_input(self):
        subject = "999_S_9999"
        digest = TIER3.hash_subject_id(subject, self.SALT)
        self.assertNotIn(subject, digest)
        self.assertNotIn(subject.replace("_", ""), digest)

    def test_the_digest_is_a_fixed_width_hex_string(self):
        for subject in ("s", "999_S_9999", "x" * 500):
            digest = TIER3.hash_subject_id(subject, self.SALT)
            self.assertEqual(len(digest), 64, "32 bytes, hex encoded")
            self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_the_iteration_count_is_not_silently_lowered(self):
        # 600,000 is the parameter that makes enumeration expensive. A change
        # to it is a change to the privacy claim in the manuscript, so it is
        # pinned rather than left to a default.
        self.assertEqual(TIER3.PBKDF2_ITERATIONS, 600_000)


class ReleasedArtefactTests(unittest.TestCase):
    """What is actually on disk under release/, if a release has been built."""

    def setUp(self):
        self.release = ROOT / "release"
        if not self.release.is_dir():
            self.skipTest("no release/ directory in this checkout")

    def headers(self, tier):
        for path in sorted((self.release / tier).glob("*.csv")):
            with path.open(encoding="utf-8", newline="") as fh:
                yield path, {c.strip().lower() for c in next(csv.reader(fh), [])}

    def test_tier3_ships_only_hashed_or_aggregate_columns(self):
        offenders = [f"{p.name}: {sorted(h - TIER3_ALLOWED)}"
                     for p, h in self.headers("tier3_adni1") if h - TIER3_ALLOWED]
        self.assertEqual(offenders, [], "ADNI release carries a non-permitted column")

    def test_tier3_never_ships_a_raw_participant_identifier(self):
        for path, header in self.headers("tier3_adni1"):
            self.assertNotIn("subject_id", header,
                             f"{path.name} ships the raw key, not its hash")

    def test_tier2_ships_public_codes_but_no_paths_or_clinical_fields(self):
        offenders = [f"{p.name}: {sorted(h & TIER2_FORBIDDEN)}"
                     for p, h in self.headers("tier2_oasis1") if h & TIER2_FORBIDDEN]
        self.assertEqual(offenders, [], "OASIS-1 release carries a withheld column")

    def test_tier1_never_ships_the_recovered_oasis_mapping(self):
        # The filename key is published on purpose: it is the artefact under
        # study. The mapping back to OASIS participants is what is withheld.
        offenders = [f"{p.name}: {sorted(h & TIER1_FORBIDDEN)}"
                     for p, h in self.headers("tier1_public_benchmark") if h & TIER1_FORBIDDEN]
        self.assertEqual(offenders, [], "Tier-1 release leaks the recovered identities")

    def test_the_release_manifest_states_the_salt_is_withheld(self):
        manifest = self.release / "RELEASE_MANIFEST.json"
        if not manifest.is_file():
            self.skipTest("no RELEASE_MANIFEST.json")
        payload = json.loads(manifest.read_text())
        self.assertIn("tier3_salt_fingerprint", payload,
                      "a release must record which salt produced it")
        blob = json.dumps(payload).lower()
        self.assertNotIn("salt_value", blob)
        self.assertNotIn("release_salt", blob)

    def test_the_salt_itself_is_nowhere_in_the_release(self):
        # A fingerprint identifies the salt; the salt inverts the hashes.
        salt_file = Path.home() / ".splitguard" / "release_salt"
        if not salt_file.is_file():
            self.skipTest("no local salt to search for")
        secret = salt_file.read_text().strip()
        self.assertTrue(secret, "salt file is empty")
        # Read bytes, not decoded text: a salt embedded in a binary artefact
        # is exactly the case a text read would skip, and skipping it quietly
        # would make this test pass for the wrong reason.
        needle = secret.encode("utf-8")
        for path in sorted(self.release.rglob("*")):
            if path.is_file():
                self.assertNotIn(needle, path.read_bytes(),
                                 f"the release salt appears in {path.relative_to(ROOT)}")


class RepositoryBoundaryTests(unittest.TestCase):
    """Artefacts that must never become tracked files."""

    NEVER_TRACKED = [
        "reports/tables/adni/adni_inventory.csv",
        "data/manifests/adni/adni_linkage_audit.csv",
        "paper/KARAR_DEFTERI.md",
        "paper/CALISMA_ANATOMISI.md",
    ]

    def tracked_files(self):
        import subprocess
        out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest("not a git checkout")
        return set(out.stdout.split())

    def test_the_named_artefacts_are_not_tracked(self):
        tracked = self.tracked_files()
        for rel in self.NEVER_TRACKED:
            self.assertNotIn(rel, tracked, f"{rel} must not be committed")

    def test_no_model_weights_are_tracked(self):
        weights = sorted(f for f in self.tracked_files()
                         if f.endswith((".pt", ".pth", ".ckpt", ".safetensors")))
        self.assertEqual(weights, [], "model weights must not be committed")

    def test_no_per_image_prediction_file_is_tracked(self):
        preds = sorted(f for f in self.tracked_files()
                       if f.endswith("test_predictions.csv")
                       or f.endswith("val_predictions.csv"))
        self.assertEqual(preds, [], "per-image predictions must not be committed")

    def test_no_released_file_names_a_cohort_participant(self):
        """No file git would ship may contain a PTID that is in the cohort.

        This is the one boundary rule that cannot be checked from the released
        tree alone: deciding whether an identifier is real needs the inventory,
        which is local-only. So the test runs where the inventory exists and
        skips where it does not, which is the machine that would introduce the
        defect and the machine that could not, respectively.

        It was written because the test fixtures here used a real PTID.
        """
        inventory = ROOT / "reports" / "tables" / "adni" / "adni_inventory.csv"
        if not inventory.is_file():
            self.skipTest("no local ADNI inventory to compare against")
        import re
        import subprocess
        real = set(re.findall(r"\b\d{3}_S_\d{4}\b",
                              inventory.read_text(encoding="utf-8", errors="ignore")))
        self.assertTrue(real, "inventory holds no PTIDs; the pattern may have changed")

        out = subprocess.run(["git", "ls-files", "-c", "-o", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest("not a git checkout")

        offenders = []
        for rel in out.stdout.split("\n"):
            rel = rel.strip()
            if not rel:
                continue
            path = ROOT / rel
            if not path.is_file() or path.stat().st_size > 8_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            found = sorted(real.intersection(re.findall(r"\b\d{3}_S_\d{4}\b", text)))
            if found:
                offenders.append(f"{rel}: {found[:3]}")
        self.assertEqual(offenders, [],
                         "a file git would ship names a participant in the cohort")

    def test_no_raw_imaging_data_is_tracked(self):
        imaging = sorted(f for f in self.tracked_files()
                         if f.endswith((".nii", ".nii.gz", ".dcm", ".img", ".hdr"))
                         or f.startswith("data/preprocessed/"))
        self.assertEqual(imaging, [], "imaging data must not be committed")


if __name__ == "__main__":
    unittest.main()
