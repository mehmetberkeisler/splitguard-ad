"""Component-safe splitting: allocate whole components, never split one.

Moved verbatim from ``scripts/make_current_splitguard_split.py``. The ordering
rule is deliberate and is the source of a known limitation the manuscript
reports: validation and test are filled from the largest components first, so a
leakage-free split is not automatically a compositionally balanced one. That is
why ``compositional_warnings`` is advisory and separate from the single
blocking condition, which is cross-split component overlap.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

__all__ = ["class_order", "class_targets", "choose_subset_by_size",
           "build_components", "assign_splits", "compositional_warnings",
           "RAW_CLASS_ORDER", "SPLIT_ORDER"]


RAW_CLASS_ORDER = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


SPLIT_ORDER = ["train", "val", "test"]


def class_order(components: list[dict]) -> list[str]:
    """Stratification order: the known classes first, then whatever else is present."""
    present = {component["raw_class_label"] for component in components}
    known = [label for label in RAW_CLASS_ORDER if label in present]
    return known + sorted(present - set(RAW_CLASS_ORDER))


def class_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train = round(total * ratios["train"])
    val = round(total * ratios["val"])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def choose_subset_by_size(components: list[dict], target: int) -> set[str]:
    """Return component IDs with total image count closest to target.

    The DP iteration order depends on the input list order, so callers
    that pass a differently-ordered list can end up with different
    subsets on ties. We therefore start by sorting the components on
    ``(-n_images, component_id)`` so this function is a pure function of
    the component set (identities + sizes), independent of the caller's
    ordering. This guarantee is a prerequisite for byte-reproducible
    manifests.
    """
    if target <= 0:
        return set()

    sorted_components = sorted(
        components,
        key=lambda component: (
            -int(component["n_images"]),
            # Seeded rank, not component_id — see assign_splits() for why.
            int(component.get("_shuffle_rank", 0)),
        ),
    )

    # dp[sum] = tuple(component_ids)
    dp: dict[int, tuple[str, ...]] = {0: ()}
    for component in sorted_components:
        component_id = component["component_id"]
        size = component["n_images"]
        additions: dict[int, tuple[str, ...]] = {}
        for current_sum, chosen in dp.items():
            new_sum = current_sum + size
            if new_sum not in dp and new_sum not in additions:
                additions[new_sum] = (*chosen, component_id)
        dp.update(additions)

    best_sum = min(dp, key=lambda value: (abs(value - target), value > target, value))
    return set(dp[best_sum])


def build_components(component_rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in component_rows:
        groups[row["component_id"]].append(row)

    components = []
    for component_id, rows in groups.items():
        raw_labels = sorted({row["raw_class_label"] for row in rows})
        binary_labels = sorted({row["binary_label"] for row in rows})
        if len(raw_labels) != 1 or len(binary_labels) != 1:
            raise ValueError(
                "SplitGuard v0 requires pure-label components. "
                f"Component {component_id} has raw={raw_labels}, binary={binary_labels}."
            )
        components.append(
            {
                "component_id": component_id,
                "n_images": len(rows),
                "raw_class_label": raw_labels[0],
                "binary_label": binary_labels[0],
                "subject_ids": sorted({row["subject_id"] for row in rows}),
                "primary_reason": rows[0]["component_primary_reason"],
            }
        )
    return components


def assign_splits(components: list[dict], seed: int, ratios: dict[str, float]) -> tuple[dict[str, str], list[dict]]:
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    target_rows = []

    components_by_raw_class: dict[str, list[dict]] = defaultdict(list)
    for component in components:
        components_by_raw_class[component["raw_class_label"]].append(component)

    for raw_class in class_order(components):
        class_components = components_by_raw_class.get(raw_class, [])
        if not class_components:
            continue

        shuffled = class_components[:]
        rng.shuffle(shuffled)
        # Freeze the seeded order into an explicit key, then sort by size with
        # that rank as the tie-break. This keeps the ordering byte-identical
        # for a given seed WITHOUT making it seed-independent: tie-breaking on
        # ``component_id`` instead would impose a total order that erases the
        # shuffle, leaving the seed inert and every per-seed manifest identical.
        for rank, item in enumerate(shuffled):
            item["_shuffle_rank"] = rank
        shuffled.sort(
            key=lambda item: (-int(item["n_images"]), int(item["_shuffle_rank"])),
        )

        total_images = sum(component["n_images"] for component in shuffled)
        targets = class_targets(total_images, ratios)

        val_components = choose_subset_by_size(shuffled, targets["val"])
        remaining_after_val = [
            component for component in shuffled if component["component_id"] not in val_components
        ]
        test_components = choose_subset_by_size(remaining_after_val, targets["test"])

        for component in shuffled:
            component_id = component["component_id"]
            if component_id in val_components:
                split = "val"
            elif component_id in test_components:
                split = "test"
            else:
                split = "train"
            assignments[component_id] = split

        achieved = Counter()
        component_counts = Counter()
        for component in shuffled:
            split = assignments[component["component_id"]]
            achieved[split] += component["n_images"]
            component_counts[split] += 1

        for split in SPLIT_ORDER:
            target_rows.append(
                {
                    "raw_class_label": raw_class,
                    "split": split,
                    "target_images": targets[split],
                    "achieved_images": achieved[split],
                    "component_count": component_counts[split],
                }
            )

    return assignments, target_rows


def compositional_warnings(summary: dict) -> list[str]:
    """Advisory findings: true of the split, but not reasons to refuse it.

    Disjointness is the only hard constraint. These are the properties that
    make a disjoint split a poor one, and the audit stayed silent about them
    until the frozen ADNI manifests turned out to have one.
    """
    out: list[str] = []
    sizes = summary.get("component_size_by_split") or {}
    if len(sizes) >= 2:
        means = {k: v for k, v in sizes.items() if v}
        if means and max(means.values()) > 1.5 * min(means.values()):
            spread = ", ".join(f"{k} {v:.2f}" for k, v in sorted(means.items()))
            out.append(
                "Component-size imbalance across partitions (mean images per "
                f"component: {spread}). Participants with longer records may be "
                "routed away from training. The split is still leakage-free.")
    shares = summary.get("binary_share_by_split") or {}
    if len(shares) >= 2 and max(shares.values()) - min(shares.values()) > 0.10:
        spread = ", ".join(f"{k} {100*v:.1f}%" for k, v in sorted(shares.items()))
        out.append(
            f"Class-mix imbalance across partitions ({spread}). A threshold "
            "selected on validation may not transfer to test.")
    return out
