"""No-test-access guard.

``enter_search_mode()`` is one-way for the life of the process: after it, any
attempt to open a test-partition file through :func:`check_path` raises
:class:`TestAccessError`. Every loader in this package calls ``check_path``
before reading, so a search run cannot see test labels, test text, test
embeddings or test score caches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

#: Files and directories that belong to the test partition. A path is blocked if
#: it equals one of these or lies inside one of them, or if its file name
#: starts with ``test`` / contains ``_test_`` (score caches written here).
TEST_PATHS = (
    ROOT / "preprocess" / "go_ppi_splits" / "test.jsonl",
    ROOT / "preprocess" / "go_withholding" / "splits" / "test.jsonl",
    ROOT / "results" / "publication_phase1_fixedval" / "prepared_embeddings"
    / "go_ppi" / "test_c65a1d2ca0e5c16a_sentence-transformers_all-mpnet-base-v2",
    ROOT / "results" / "publication_phase1_fixedval" / "go_ppi_fusion_learning_curve.json",
    ROOT / "results" / "publication_phase1_fixedval"
    / "go_ppi_fusion_learning_curve_ndcg_selected.json",
)


class TestAccessError(PermissionError):
    """Raised when search mode tries to read a test-partition artefact."""

    __test__ = False  # not a pytest class


_SEARCH_MODE = False


def enter_search_mode() -> None:
    """Block test access for the rest of this process (cannot be undone)."""
    global _SEARCH_MODE
    _SEARCH_MODE = True


def in_search_mode() -> bool:
    return _SEARCH_MODE


def _is_test_path(path: Path, blocked: Iterable[Path] = TEST_PATHS) -> bool:
    p = Path(path).resolve()
    for b in blocked:
        b = b.resolve()
        if p == b or b in p.parents:
            return True
    name = p.name
    return name.startswith("test") or "_test_" in name or name.startswith("scores_test")


def check_path(path: Path, split: str | None = None) -> None:
    """Raise in search mode if ``path`` (or the ``split`` name) is test data."""
    if not _SEARCH_MODE:
        return
    if split == "test" or _is_test_path(Path(path)):
        raise TestAccessError(
            f"search mode forbids test access: {path} (split={split!r})")


def check_split(split: str) -> None:
    if _SEARCH_MODE and split != "validation":
        raise TestAccessError(f"search mode forbids split {split!r}")
