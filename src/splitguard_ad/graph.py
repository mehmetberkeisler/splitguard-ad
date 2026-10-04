"""The leakage graph: identifier edges, union-find, connected components.

This is the reusable core of SplitGuard. A manifest names images and whatever
identity evidence survives for each one; the graph joins images that any
surviving identifier says belong to the same participant, and the connected
components of that graph are the units a split must not break.

The code here was moved verbatim out of ``scripts/build_current_leakage_graph.py``
rather than rewritten, so the package holds the same implementation that
produced every number in the manuscript. That script now imports from here, so
there is one implementation and not two.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MISSING_TOKENS", "normalise_identifier", "stable_token", "Record",
           "UnionFind", "component_maps", "dhash", "hamming"]


MISSING_TOKENS = {"", "na", "n/a", "none", "null", "unknown", "nan", "-", "?"}


def normalise_identifier(value: object) -> str | None:
    """Return the identifier, or None when the field is effectively absent.

    Grouping is identity evidence. A blank, a placeholder or a literal
    "unknown" is the absence of evidence, and treating absence as a shared key
    is how a provenance-aware splitter would manufacture the very failure this
    paper measures.
    """
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING_TOKENS else text


def stable_token(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class Record:
    image_id: str
    path: str
    relative_path: str
    raw_class_label: str
    binary_label: str
    subject_id: str
    subject_id_confidence: str
    subject_parse_status: str
    file_sha256: str


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = dict.fromkeys(items, 0)
        self.reasons: dict[str, set[str]] = defaultdict(set)

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str, reason: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            self.reasons[root_left].add(reason)
            return

        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.reasons[root_left].update(self.reasons[root_right])
        self.reasons[root_left].add(reason)
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def component_maps(records: list[Record], uf: UnionFind) -> tuple[dict[str, str], dict[str, list[Record]], dict[str, str]]:
    root_to_records: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        root_to_records[uf.find(record.image_id)].append(record)

    root_to_component_id: dict[str, str] = {}
    root_to_reason: dict[str, str] = {}
    for root, group in root_to_records.items():
        image_ids = sorted(record.image_id for record in group)
        component_id = f"comp_{stable_token(';'.join(image_ids))}"
        root_to_component_id[root] = component_id

        reasons = sorted(uf.reasons.get(root, set()))
        if not reasons and len(group) == 1:
            reason = "singleton"
        elif reasons:
            reason = "+".join(reasons)
        else:
            reason = "implicit_group"
        root_to_reason[root] = reason

    image_to_component_id = {
        record.image_id: root_to_component_id[uf.find(record.image_id)]
        for record in records
    }
    component_id_to_records = {
        root_to_component_id[root]: group for root, group in root_to_records.items()
    }
    component_id_to_reason = {
        root_to_component_id[root]: reason for root, reason in root_to_reason.items()
    }
    return image_to_component_id, component_id_to_records, component_id_to_reason


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def dhash(path: Path, hash_size: int = 16) -> int:
    """Difference hash of an image, for near-duplicate detection.

    The body is the original implementation unchanged, including the default
    resample filter: naming one explicitly would change every hash, and with it
    every near-duplicate edge the manuscript reports. Pillow is imported here
    rather than at module scope so importing the graph core costs nothing on a
    machine that only needs union-find, which is every machine running the
    verification gates.
    """
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size))
        pixels = list(img.getdata())

    bits = 0
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits
