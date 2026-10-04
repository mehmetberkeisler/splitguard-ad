"""Tests for the repository as an artefact, rather than for any number in it.

Every check here exists because the thing it checks was once wrong, and in each
case the three verification gates stayed green while it was. They are cheap and
they are the kind of defect a reader meets first: a command in the README that
names a script that was deleted, a pin file describing a stack that never ran,
an interpreter whose metrics library cannot be imported.

The environment checks compare the pin files against
``runs_gpu/gpu_environment.txt``, the package list frozen on the GPU node
before training. That record, not the pin file, is the provenance of every
published number, so the pins are correct exactly when they agree with it.
"""

import ast
import importlib.util
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RECORD = ROOT / "runs_gpu" / "gpu_environment.txt"

# Distribution name -> the name you import it under, where they differ.
IMPORT_NAME = {"scikit-learn": "sklearn", "pillow": "PIL", "pyyaml": "yaml"}


def parse_pins(path: Path) -> dict[str, str]:
    """Read ``name==version`` lines, ignoring comments and markers."""
    pins = {}
    if not path.is_file():
        return pins
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9._-]+)==([^\s;]+)$", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def public(version: str) -> str:
    """2.8.0+cu128 -> 2.8.0. The local suffix names the build, not the release."""
    return version.split("+", 1)[0]


class ScriptsAreWellFormedTests(unittest.TestCase):
    """Syntax, for every script, in both languages."""

    def test_every_python_script_parses(self):
        broken = []
        for path in sorted(SCRIPTS.glob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                broken.append(f"{path.name}: {exc}")
        self.assertEqual(broken, [], "scripts that do not parse")

    def test_every_shell_script_parses(self):
        # This caught a real one: a stage script edited while bash was part way
        # through executing it, which left an unbalanced block at the tail.
        broken = []
        for path in sorted(SCRIPTS.glob("*.sh")):
            out = subprocess.run(["bash", "-n", str(path)],
                                 capture_output=True, text=True)
            if out.returncode != 0:
                broken.append(f"{path.name}: {out.stderr.strip()}")
        self.assertEqual(broken, [], "shell scripts that do not parse")


class NoDanglingScriptReferencesTests(unittest.TestCase):
    """A documented command must name a script that exists.

    Only the released surface is searched. Local planning notes are gitignored,
    outlive the scripts they describe on purpose, and are not what a reader runs.
    """

    SEARCHED = ("README.md", "PROJECT_EXPLAINER.md", "PROJECT_RECORD.md",
                "environment.yml", "requirements.txt", "requirements-gpu.txt")

    @staticmethod
    def released(paths):
        """Drop anything .gitignore excludes, when git can tell us.

        In a clone or in the distributed bundle the ignored files are simply
        absent, so the filter is a no-op there; in a working checkout it is
        what keeps local planning notes from failing a test about the
        released surface.
        """
        paths = [p for p in paths if p.is_file()]
        if not paths:
            return []
        try:
            out = subprocess.run(
                ["git", "check-ignore", "--stdin"], cwd=ROOT, text=True,
                input="\n".join(str(p) for p in paths), capture_output=True)
        except OSError:
            return paths
        if out.returncode not in (0, 1):      # 128: not a git checkout
            return paths
        ignored = {line.strip() for line in out.stdout.splitlines() if line.strip()}
        return [p for p in paths if str(p) not in ignored]

    def sources(self):
        candidates = [ROOT / rel for rel in self.SEARCHED]
        for sub in ("docs", "scripts", "tests"):
            for pattern in ("*.md", "*.py", "*.sh"):
                candidates.extend(sorted((ROOT / sub).glob(pattern)))
        return self.released(candidates)

    def test_every_referenced_script_exists(self):
        present = {p.name for p in SCRIPTS.glob("*.*")}
        dangling = []
        for path in self.sources():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name in re.findall(r"scripts/([A-Za-z0-9_.-]+\.(?:py|sh))", text):
                if name not in present:
                    dangling.append(f"{path.relative_to(ROOT)} -> scripts/{name}")
        self.assertEqual(sorted(set(dangling)), [],
                         "released files naming a script that does not exist")


class ManuscriptCitationsResolveTests(unittest.TestCase):
    """Every repository path the manuscript names must actually ship.

    A reader who follows a \\texttt{...} path in the paper to the release and
    finds nothing there has caught the paper in a false statement about its own
    artefacts. This was real: the subject-level aggregation paragraph pointed at
    reports/tables/oasis1_subject_level_summary.json, which was gitignored.

    The file existed on disk, which is why no earlier check noticed. The test
    compares against what git would ship, not against the working tree.
    """

    PATH_RE = re.compile(r"^[\w./-]+\.(py|sh|csv|json|txt|tex|md|yml)$")

    def cited_paths(self) -> set[str]:
        paths = set()
        for name in ("splitguard_ad.tex", "SplitGuard-AD_Supplementary_Material.tex"):
            doc = ROOT / "paper" / name
            if not doc.is_file():
                continue
            text = doc.read_text(encoding="utf-8", errors="ignore")
            for literal in re.findall(r"\\texttt\{([^}]*?)\}", text):
                # LaTeX escapes underscores and percent signs in \texttt.
                candidate = literal.replace("\\_", "_").replace("\\%", "%").strip()
                if "/" in candidate and self.PATH_RE.match(candidate):
                    paths.add(candidate)
        return paths

    def test_every_path_the_manuscript_names_is_shipped(self):
        cited = self.cited_paths()
        if not cited:
            self.skipTest("no manuscript sources in this checkout")
        out = subprocess.run(["git", "ls-files", "-c", "-o", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest("not a git checkout")
        shipped = set(out.stdout.split())
        missing = sorted(p for p in cited if p not in shipped)
        self.assertEqual(missing, [],
                         "the manuscript names a path the release does not contain")

    def test_the_check_actually_found_some_paths(self):
        """Guard against the regex silently matching nothing."""
        cited = self.cited_paths()
        if not (ROOT / "paper" / "splitguard_ad.tex").is_file():
            self.skipTest("no manuscript sources in this checkout")
        self.assertGreater(len(cited), 5,
                           "the citation scan found almost nothing, so it proves nothing")


class PinsMatchTheRecordedEnvironmentTests(unittest.TestCase):
    """The pin files must describe the environment the results came from."""

    def setUp(self):
        if not RECORD.is_file():
            self.skipTest("no runs_gpu/gpu_environment.txt in this checkout")
        self.node = {}
        for line in RECORD.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z0-9._-]+)==(.+)$", line.strip())
            if m:
                self.node[m.group(1).lower().replace("_", "-")] = m.group(2)
        self.assertTrue(self.node, "the record holds no pinned versions")

    def assert_agrees(self, path: Path):
        pins = parse_pins(path)
        self.assertTrue(pins, f"{path.name} declares no pins")
        wrong = [f"{d}: pinned {v}, node ran {self.node[d]}"
                 for d, v in sorted(pins.items())
                 if d in self.node and public(self.node[d]) != public(v)]
        self.assertEqual(wrong, [],
                         f"{path.name} disagrees with the recorded GPU environment")

    def test_requirements_agrees_with_the_record(self):
        self.assert_agrees(ROOT / "requirements.txt")

    def test_the_lock_file_agrees_with_the_record(self):
        # requirements-lock.txt is what a reviewer installs to reproduce a
        # published number, so it is the file that must not drift.
        self.assert_agrees(ROOT / "requirements-lock.txt")

    def test_gpu_requirements_agrees_with_the_record(self):
        # This file drives the venv fallback in gpu_setup.sh. If it drifts from
        # the record, the fallback silently builds a different stack than the
        # one that produced the numbers.
        self.assert_agrees(ROOT / "requirements-gpu.txt")

    def test_the_training_stack_is_pinned_at_all(self):
        pins = parse_pins(ROOT / "requirements.txt")
        for dist in ("torch", "torchvision", "scikit-learn", "numpy", "monai"):
            self.assertIn(dist, pins,
                          f"{dist} runs on the node but is not pinned")


class DependencyFilesAgreeTests(unittest.TestCase):
    """requirements.txt, requirements-gpu.txt and environment.yml are one set.

    They drifted apart once already: the conda spec and the GPU fallback kept
    pins that the main file had moved past, so which stack you got depended on
    which file you happened to install from.
    """

    @staticmethod
    def conda_pins(path: Path) -> dict[str, str]:
        pins = {}
        if not path.is_file():
            return pins
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*-\s*([A-Za-z0-9._-]+)==([^\s;]+)$", line)
            if m:
                pins[m.group(1).lower().replace("_", "-")] = m.group(2)
        return pins

    def test_environment_yml_matches_requirements(self):
        req = parse_pins(ROOT / "requirements.txt")
        env = self.conda_pins(ROOT / "environment.yml")
        if not env:
            self.skipTest("no environment.yml in this checkout")
        self.assertEqual(sorted(env), sorted(req),
                         "environment.yml and requirements.txt list different packages")
        clashes = [f"{k}: requirements {req[k]} vs environment.yml {env[k]}"
                   for k in sorted(req) if public(req[k]) != public(env[k])]
        self.assertEqual(clashes, [], "the two dependency files disagree on a version")

    def test_the_lock_file_and_requirements_are_the_same_set(self):
        req = parse_pins(ROOT / "requirements.txt")
        lock = parse_pins(ROOT / "requirements-lock.txt")
        if not lock:
            self.skipTest("no requirements-lock.txt in this checkout")
        self.assertEqual(sorted(lock), sorted(req),
                         "the lock file and requirements.txt list different packages")
        clashes = [f"{k}: requirements {req[k]} vs lock {lock[k]}"
                   for k in sorted(req) if public(req[k]) != public(lock[k])]
        self.assertEqual(clashes, [], "the lock file and requirements.txt disagree")

    def test_the_package_declares_ranges_not_pins(self):
        """pyproject must not pin, or installing it would fight the lock file.

        The two files answer different questions: the ranges are what the
        package needs to run, the lock is the environment the published numbers
        came from. If pyproject pinned, `pip install -e .` would silently
        become a reproduction attempt.
        """
        try:
            import tomllib
        except ImportError:  # tomllib is 3.11+; the package floor is 3.10
            self.skipTest("tomllib needs Python 3.11; the package supports 3.10")
        path = ROOT / "pyproject.toml"
        if not path.is_file():
            self.skipTest("no pyproject.toml in this checkout")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data["project"]
        self.assertEqual(project["dependencies"], [],
                         "the core must stay dependency-free")
        pinned = [d for group in project.get("optional-dependencies", {}).values()
                  for d in group if "==" in d]
        self.assertEqual(pinned, [],
                         "an extra pins a version; ranges belong here and pins in the lock file")

    def test_gpu_requirements_does_not_contradict_requirements(self):
        req = parse_pins(ROOT / "requirements.txt")
        gpu = parse_pins(ROOT / "requirements-gpu.txt")
        if not gpu:
            self.skipTest("no requirements-gpu.txt in this checkout")
        clashes = [f"{k}: requirements {req[k]} vs gpu {gpu[k]}"
                   for k in sorted(set(req) & set(gpu))
                   if public(req[k]) != public(gpu[k])]
        self.assertEqual(clashes, [], "the GPU fallback would install a different stack")


class NoVestigialPinsTests(unittest.TestCase):
    """A pin for a package nothing imports is a claim about nothing.

    Three such pins survived a change of target journal here: the scripts that
    used them were deleted and the pins stayed, so `pip install -r` kept
    fetching them and check_environment.py kept reporting on them.
    """

    def imported_names(self):
        names = set()
        for path in list(SCRIPTS.glob("*.py")) + list((ROOT / "tests").glob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module.split(".")[0])
        return names

    def test_every_pin_is_imported_somewhere(self):
        imported = self.imported_names()
        unused = []
        for dist in sorted(parse_pins(ROOT / "requirements.txt")):
            mod = IMPORT_NAME.get(dist, dist.replace("-", "_"))
            if mod not in imported:
                unused.append(dist)
        self.assertEqual(unused, [],
                         "pinned but imported by nothing in scripts/ or tests/")


class CheckEnvironmentBehaviourTests(unittest.TestCase):
    """The drift checker itself, since it is the gate that guards the others."""

    def setUp(self):
        path = SCRIPTS / "check_environment.py"
        if not path.is_file():
            self.skipTest("check_environment.py absent")
        spec = importlib.util.spec_from_file_location("chkenv", path)
        self.mod = importlib.util.module_from_spec(spec)
        sys.modules["chkenv"] = self.mod
        spec.loader.exec_module(self.mod)

    def test_a_cuda_build_matches_its_upstream_pin(self):
        # The node's wheels report 2.8.0+cu128. Treating that as drift would
        # flag the one environment that is correct by definition.
        self.assertEqual(self.mod.public("2.8.0+cu128"), "2.8.0")
        self.assertEqual(self.mod.public("0.23.0+cu128"), "0.23.0")

    def test_a_plain_version_is_unchanged(self):
        self.assertEqual(self.mod.public("1.9.1"), "1.9.1")

    def test_it_reads_the_projects_own_pins(self):
        pins = self.mod.pinned(ROOT / "requirements.txt")
        self.assertIn("torch", pins)
        self.assertNotIn("", pins)


if __name__ == "__main__":
    unittest.main()
