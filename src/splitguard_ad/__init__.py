"""SplitGuard-AD: provenance-aware splitting and leakage auditing.

Patient-wise splitting is only as good as the identifier that defines a
patient. This package is the reusable part of that argument: build a leakage
graph from whatever identity evidence a cohort still carries, take its
connected components as the indivisible units, allocate whole components to
partitions, and audit the result before training rather than after.

    from splitguard_ad import UnionFind, assign_splits, AuditReport

The paper-specific experiments live in ``scripts/`` and read these modules, so
there is one implementation of the core and not two. This is a reference
implementation and audit pipeline rather than a general-purpose library.
"""

from __future__ import annotations

from .audit import AuditReport, Finding, Level
from .corruption import MECHANISMS, corrupt_drop, corrupt_merge, corrupt_split
from .graph import (
                    MISSING_TOKENS,
                    Record,
                    UnionFind,
                    component_maps,
                    dhash,
                    hamming,
                    normalise_identifier,
                    stable_token,
)
from .metrics import auroc, residual_subject_leakage
from .splitting import (
                    assign_splits,
                    build_components,
                    choose_subset_by_size,
                    class_order,
                    class_targets,
                    compositional_warnings,
)

__version__ = "1.0.0"

__all__ = [
    "AuditReport", "Finding", "Level",
    "MECHANISMS", "corrupt_drop", "corrupt_merge", "corrupt_split",
    "MISSING_TOKENS", "Record", "UnionFind", "component_maps", "dhash",
    "hamming", "normalise_identifier", "stable_token",
    "auroc", "residual_subject_leakage",
    "assign_splits", "build_components", "choose_subset_by_size",
    "class_order", "class_targets", "compositional_warnings",
    "__version__",
]
