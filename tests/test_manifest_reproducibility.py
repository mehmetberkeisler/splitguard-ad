"""Byte-reproducibility contract tests for SplitGuard-AD manifests.

The framework's headline claim is that a single ``--seed`` argument
produces byte-identical component-safe split manifests across runs and
platforms. These tests exercise the pieces of the split generators that
are responsible for that determinism:

* ``choose_subset_by_size`` in :mod:`make_current_splitguard_split` and
  :mod:`make_adni_splitguard_split` — the greedy bin-packer whose sort
  key was hardened with a deterministic tie-break on ``component_id``.

The tests deliberately construct component lists with **repeated
``n_images`` values** to exercise the tie-break path: without the
``(-n_images, component_id)`` key, two components with the same size
would be ordered by the input list's arbitrary order, and different
runs could produce different subsets. With the tie-break in place, the
selected subset is a pure function of the inputs.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class BinPackingDeterminismTests(unittest.TestCase):
    """Same input list → same output subset, regardless of input order."""

    def _components(self, sizes_and_ids):
        return [{"component_id": cid, "n_images": n} for n, cid in sizes_and_ids]

    def test_current_choose_subset_is_stable_across_input_orderings(self):
        module = _load_module(ROOT / "scripts" / "make_current_splitguard_split.py")
        components_a = self._components(
            [(30, "c_alpha"), (30, "c_beta"), (30, "c_gamma"), (20, "c_delta"), (10, "c_eps")]
        )
        components_b = list(reversed(components_a))
        target = 60
        chosen_a = module.choose_subset_by_size(components_a, target)
        chosen_b = module.choose_subset_by_size(components_b, target)
        self.assertEqual(
            chosen_a,
            chosen_b,
            "choose_subset_by_size must be a deterministic function of "
            "component identity and size; input list order should not "
            "matter. If this fails, the ADNI bin-packing tie-break is "
            "gone and the manifest is no longer byte-reproducible.",
        )

    def test_adni_choose_subset_is_stable_across_input_orderings(self):
        module = _load_module(ROOT / "scripts" / "make_adni_splitguard_split.py")
        components_a = self._components(
            [(40, "adni_a"), (40, "adni_b"), (40, "adni_c"), (25, "adni_d"), (15, "adni_e")]
        )
        components_b = list(reversed(components_a))
        target = 80
        chosen_a = module.choose_subset_by_size(components_a, target)
        chosen_b = module.choose_subset_by_size(components_b, target)
        self.assertEqual(
            chosen_a,
            chosen_b,
            "ADNI choose_subset_by_size must resolve ties on "
            "``component_id``; without the tie-break, two components of "
            "identical size are ordered arbitrarily and the same seed "
            "may produce different splits across runs.",
        )

    def test_adni_choose_subset_prefers_lexicographically_lower_component_id_on_ties(self):
        """Tie-break rule: after ``-n_images``, sort by ``component_id`` ascending.
        This test locks in the specific rule so a future refactor cannot
        silently change which component is selected on a tie."""
        module = _load_module(ROOT / "scripts" / "make_adni_splitguard_split.py")
        # Two components of size 50 — target=50 must pick exactly one.
        # With ``component_id`` ascending as the tie-break, "adni_a" wins.
        components = self._components([(50, "adni_b"), (50, "adni_a"), (30, "adni_c")])
        chosen = module.choose_subset_by_size(components, target=50)
        self.assertEqual(chosen, {"adni_a"})


if __name__ == "__main__":  # pragma: no cover - convenience runner
    unittest.main()
