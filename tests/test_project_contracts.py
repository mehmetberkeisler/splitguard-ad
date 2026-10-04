import csv
import importlib.util
import random
import re
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class ProjectContractTests(unittest.TestCase):
    def test_filename_parsing_is_explicitly_pseudo_subject_scoped(self):
        manifest = load_module(ROOT / "scripts" / "build_current_dataset_manifest.py")

        parsed = manifest.parse_filename(
            Path("NonDemented/NonDemented_patient100_1 (100).jpg"),
            "NonDemented",
        )
        self.assertEqual(parsed["subject_id"], "NonDemented_patient100")
        self.assertEqual(parsed["subject_id_confidence"], "high_filename_parentheses")

        unparseable = manifest.parse_filename(Path("ModerateDemented/abc.jpg"), "ModerateDemented")
        self.assertTrue(unparseable["subject_id"].startswith("ModerateDemented_unique_"))
        self.assertEqual(unparseable["subject_id_confidence"], "low_unparseable_unique")

    def test_main_paper_has_required_sections_and_no_submission_blockers(self):
        text = (ROOT / "paper" / "splitguard_ad.tex").read_text(encoding="utf-8")
        # IMRAD, as the target venue expects. The pre-IMRAD headings this test
        # used to require ("Related Work", "The \\SGA{} Framework",
        # "Datasets", "Experiments", "Clinical Discussion") were merged into
        # Materials and Methods / Results / Discussion. The test kept passing
        # through that restructure only because it was reading a stale copy of
        # the manuscript, which is the failure mode it now guards against.
        for section in ["Introduction", "Materials and Methods", "Results",
                        "Discussion", "Conclusions"]:
            self.assertIn(rf"\section{{{section}}}", text)
        self.assertIn(r"\bibliography{references}", text)
        blocked_terms = ["TO" + "DO", "TO" + "DO@institution", "PLACE" + "HOLDER", "FIX" + "ME"]
        self.assertNotRegex(text, "|".join(blocked_terms))

        # Exactly one manuscript may live in paper/. A second .tex that also
        # looks like the paper is how an outdated draft gets compiled and
        # submitted by accident; one such copy sat beside the live file for
        # weeks and this test was reading it rather than the real manuscript.
        manuscripts = sorted(
            path.name for path in (ROOT / "paper").glob("*.tex")
            if "splitguard_ad" in path.name
        )
        self.assertEqual(manuscripts, ["splitguard_ad.tex"])

    def test_gpu_programme_flags_exist_in_the_scripts_it_calls(self):
        """Every flag the orchestrator passes must exist, or a stage fails after
        the node is already rented."""
        catalogue = load_module(ROOT / "scripts" / "gpu_program.py")
        declared = {}
        for stage in catalogue.stages("cuda", "data/adni3d_128").values():
            for command in stage:
                script = command["argv"][1]
                if not script.endswith(".py"):
                    continue
                if script not in declared:
                    source = (ROOT / script).read_text(encoding="utf-8")
                    declared[script] = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source))
                for token in command["argv"][2:]:
                    if token.startswith("--"):
                        self.assertIn(token, declared[script], f"{script} has no {token}")

    def test_manuscript_macros_are_all_generated(self):
        """Every generated number the manuscript uses must be defined by
        scripts/gpu_postprocess.py, and every \\input table must exist."""
        macros_file = ROOT / "paper" / "tables" / "gpu_numbers.tex"
        if not macros_file.exists():
            self.skipTest("gpu_numbers.tex not generated yet")
        defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros_file.read_text(encoding="utf-8")))
        for name in ("splitguard_ad.tex", "SplitGuard-AD_Supplementary_Material.tex"):
            text = (ROOT / "paper" / name).read_text(encoding="utf-8")
            body = "\n".join(line.split("%")[0] for line in text.splitlines())
            for used in set(re.findall(r"\\(TierOne[A-Za-z]*|Adni[A-Za-z]*|Oasis[A-Za-z]*|Vol[A-Za-z]*|Card[A-Za-z]*|Prov[A-Za-z]*|Dose[A-Za-z]*|Sens[A-Za-z]*|Composed[A-Za-z]*|Missed[A-Za-z]*|SizeBalanced[A-Za-z]*|ProbeLift[A-Za-z]*|GpuHours)", body)):
                self.assertIn(used, defined, f"{name} uses \\{used}, which nothing generates")
            for table in set(re.findall(r"\\input\{tables/([A-Za-z0-9_]+)\}", body)):
                self.assertTrue((ROOT / "paper" / "tables" / f"{table}.tex").exists(),
                                f"{name} inputs tables/{table}.tex, which does not exist")

    def test_main_paper_latex_references_resolve_locally(self):
        tex_path = ROOT / "paper" / "splitguard_ad.tex"
        text = tex_path.read_text(encoding="utf-8")
        bib_path = tex_path.parent / "references.bib"
        bib_text = bib_path.read_text(encoding="utf-8")
        bibitems = set(re.findall(r"@\w+\{([^,]+),", bib_text))
        cites = {
            key.strip()
            for group in re.findall(r"\\cite\{([^}]+)\}", text)
            for key in group.split(",")
        }
        self.assertTrue(cites)
        self.assertTrue(cites.issubset(bibitems))

        graphics = re.findall(r"\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}", text)
        self.assertTrue(graphics)
        for graphic in graphics:
            self.assertTrue((tex_path.parent / graphic).exists(), graphic)

    def test_split_manifest_schema_and_component_safety_when_available(self):
        split_path = ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv"
        if not split_path.exists():
            self.skipTest("Local split manifest is intentionally excluded from public release.")

        with split_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        required = {
            "image_id",
            "split",
            "component_id",
            "path",
            "relative_path",
            "raw_class_label",
            "binary_label",
            "subject_id",
            "subject_id_confidence",
            "split_policy",
        }
        self.assertTrue(rows)
        self.assertTrue(required.issubset(rows[0]))

        component_to_splits = {}
        for row in rows:
            self.assertIn(row["split"], {"train", "val", "test"})
            component_to_splits.setdefault(row["component_id"], set()).add(row["split"])

        leaking = {
            component_id: sorted(splits)
            for component_id, splits in component_to_splits.items()
            if len(splits) > 1
        }
        self.assertEqual(leaking, {})

    def test_every_trainer_orients_auroc_the_same_way_as_the_postprocessor(self):
        # The volumetric trainer once ranked scores descending, so its AUROC was
        # 1 - AUROC: validation looked anti-predictive and the best-val
        # checkpoint was the least-trained epoch. An orientation flip is silent
        # in a metric whose plausible range covers its own complement, so it has
        # to be pinned down by test rather than by reading the number.
        trainer = load_module(ROOT / "scripts" / "train_adni_3d.py")
        reference = load_module(ROOT / "scripts" / "gpu_postprocess.py")

        labels = [0, 0, 0, 1, 1, 1]
        self.assertEqual(trainer.auroc(labels, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), 1.0)
        self.assertEqual(trainer.auroc(labels, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), 0.0)
        self.assertEqual(trainer.auroc(labels, [0.5] * 6), 0.5)
        self.assertNotEqual(trainer.auroc([1, 1, 1], [0.1, 0.2, 0.3]),
                            trainer.auroc([1, 1, 1], [0.1, 0.2, 0.3]))   # nan

        rng = random.Random(0)
        for _ in range(200):
            n = rng.randint(4, 40)
            y = [rng.randint(0, 1) for _ in range(n)]
            if len(set(y)) < 2:
                continue
            scores = [round(rng.random(), 2) for _ in range(n)]      # deliberate ties
            self.assertAlmostEqual(trainer.auroc(y, scores),
                                   reference.auroc(list(zip(y, scores, strict=True))), places=12)

    def test_label_permutation_keeps_one_label_per_participant_across_partitions(self):
        # The null control permuted labels inside each partition, which gave a
        # participant who straddles train and test a different label on each
        # side. Identity memorisation then cannot pay off, every protocol lands
        # at chance, and the control reads as "leakage manufactures nothing"
        # when it has in fact been prevented from manufacturing anything. The
        # corrected permutation runs once over the cohort; nothing in the
        # numbers it produces would reveal a regression, so it is pinned here.
        null = load_module(ROOT / "scripts" / "run_adni_permutation_null.py")
        rows = [{"image_id": f"i{i}", "subject_id": f"s{i // 3}",
                 "diagnosis_group": "AD" if i % 2 else "CN"} for i in range(30)]
        splits = {"train": rows[:20], "val": rows[20:24], "test": rows[16:]}
        self.assertTrue({r["subject_id"] for r in splits["train"]}
                        & {r["subject_id"] for r in splits["test"]},
                        "the fixture must contain a straddling participant")

        moved = 0
        for seed in range(20):
            permuted = null.permute_splits(splits, random.Random(seed))
            if any(a["diagnosis_group"] != b["diagnosis_group"]
                   for a, b in zip(rows, permuted["train"] + permuted["val"], strict=True)):
                moved += 1
            labels = defaultdict(set)
            for phase_rows in permuted.values():
                for row in phase_rows:
                    labels[row["subject_id"]].add(row["diagnosis_group"])
            self.assertFalse([s for s, v in labels.items() if len(v) > 1],
                             "a participant carries two permuted labels")
            original = Counter(r["diagnosis_group"] for r in rows)
            union = {r["image_id"]: r["diagnosis_group"]
                     for phase_rows in permuted.values() for r in phase_rows}
            self.assertEqual(Counter(union.values()), original,
                             "the permutation must deal the same labels back out")

        # Label-preserving and consistent is also what the identity function
        # is, so require that some seeds actually move a label.
        self.assertGreater(moved, 0, "permute_splits never changed a label")


if __name__ == "__main__":
    unittest.main()


class AuditVerdictTests(unittest.TestCase):
    """A split can be identity-safe and still be a poor split.

    The gate returned one undifferentiated GO, so the frozen ADNI manifests
    passed every check while being stratified by component size. Blocking and
    advisory findings are now separate, and these tests pin the distinction:
    a warning must never block, and overlap must always block.
    """

    def setUp(self):
        self.mod = load_module(ROOT / "scripts" / "make_current_splitguard_split.py")

    def test_a_balanced_split_raises_nothing(self):
        self.assertEqual(self.mod.compositional_warnings(
            {"component_size_by_split": {"train": 5.0, "val": 5.1, "test": 5.0},
             "binary_share_by_split": {"train": 0.40, "val": 0.39, "test": 0.41}}), [])

    def test_the_frozen_adni_composition_is_flagged(self):
        # The real means from the frozen manifests: 4.65 / 7.50 / 6.11.
        warnings = self.mod.compositional_warnings(
            {"component_size_by_split": {"train": 4.65, "val": 7.50, "test": 6.11}})
        self.assertEqual(len(warnings), 1)
        self.assertIn("Component-size imbalance", warnings[0])
        self.assertIn("still leakage-free", warnings[0],
                      "a warning must say plainly that it does not block")

    def test_the_converter_arm_class_mix_is_flagged(self):
        # 41.3% / 20.7% / 37.5% AD by image, the defect Limitations reports.
        warnings = self.mod.compositional_warnings(
            {"binary_share_by_split": {"train": 0.413, "val": 0.207, "test": 0.375}})
        self.assertTrue(any("Class-mix imbalance" in w for w in warnings))

    def test_warnings_are_advisory_and_overlap_is_not(self):
        source = (ROOT / "scripts" / "make_current_splitguard_split.py").read_text()
        verdict = source[source.index("warnings = compositional_warnings"):
                         source.index("warning_block")]
        self.assertIn('if not summary["overlap_check_passed"]', verdict,
                      "overlap must be the only thing that produces NO-GO")
        no_go = verdict.count("NO-GO")
        self.assertEqual(no_go, 1, "exactly one branch may return NO-GO")
        self.assertIn("do not block training", verdict + source,
                      "the report must state that warnings do not block")
