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
        text = (ROOT / "paper" / "splitguard_ad_paper.tex").read_text(encoding="utf-8")
        for section in ["Introduction", "Related Work", "The \\SGA{} Framework", "Datasets", "Experiments", "Clinical Discussion", "Conclusion"]:
            self.assertIn(rf"\section{{{section}}}", text)
        self.assertIn(r"\bibliography{references}", text)
        blocked_terms = ["TO" + "DO", "TO" + "DO@institution", "PLACE" + "HOLDER", "FIX" + "ME"]
        self.assertNotRegex(text, "|".join(blocked_terms))

    def test_main_paper_latex_references_resolve_locally(self):
        tex_path = ROOT / "paper" / "splitguard_ad_paper.tex"
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
