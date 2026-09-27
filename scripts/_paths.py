"""Path display helpers shared by the pipeline scripts.

Every script in this directory reports where it wrote something, and the
natural way to do that is ``path.relative_to(PROJECT_ROOT)`` so the message
stays short. That call raises ``ValueError`` the moment the path is not under
the project root -- which happens as soon as anyone points ``--output`` or
``--audit-dir`` at a directory of their own, a scratch disk, or a mounted
share. The work has already been done and written by that point, so the
failure is purely in the reporting, but it still produces a traceback and a
non-zero exit status, which is enough to break any pipeline wrapping the
script and enough to make a first-time user think the run failed.

``display_path`` degrades to the absolute path instead of raising.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def display_path(path: Path | str, root: Path = PROJECT_ROOT) -> str:
    """Path relative to `root` when it is inside it, else the absolute path."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(p.resolve())


def resolve_data_path(row: dict, keys: tuple[str, ...] = ("path", "image_path"),
                      relative_key: str = "relative_path", root: Path = PROJECT_ROOT) -> Path:
    """The file a manifest row names, wherever the repository now lives.

    Manifests record the absolute path on the machine that built them, which
    does not exist anywhere else: a clone, a reviewer's laptop or a GPU node
    all fail on the first image. The recorded path is used when it resolves;
    otherwise the repository-relative path the manifest also carries is tried
    against ``root``. If neither exists the recorded path is returned so the
    error names the file that was expected.
    """
    recorded = [Path(row[k]) for k in keys if row.get(k)]
    for candidate in recorded:
        if candidate.exists():
            return candidate
    if row.get(relative_key):
        candidate = Path(root) / row[relative_key]
        if candidate.exists():
            return candidate
    if recorded:
        return recorded[0]
    raise KeyError(f"manifest row has none of {keys + (relative_key,)}")
