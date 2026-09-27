import csv
import importlib.util
import re
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


if __name__ == "__main__":
    unittest.main()
