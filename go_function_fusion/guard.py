"""Test-read guard and search log (protocol §10.3; pattern of ``go_ppi_fusion/guard.py``).

``enter_search_mode()`` is one-way for the life of the process. After it:

* every loader of this package calls :func:`check_path` / :func:`check_split`
  before reading and raises :class:`TestAccessError` on a test artefact;
* a ``sys.addaudithook`` hook checks every ``open()`` of the process (numpy,
  pandas, torch and the imported ``go_function`` loaders included), so a search
  run is *unable* to open a blocked file even through code that does not call
  the guard.

Blocked in search mode (repo-relative):

* ``dataset/go_function/dgz/*/test_data.pkl``, ``predictions_*.pkl``,
  ``test_diamond.res``;
* ``results/go_function_fusion*/test/**`` and the remote test Diamond output
  (``results/go_function_fusion*/diamond/*test*``);
* every file under ``results/go_function_fusion*/tp_bases/`` (Tier B source (b),
  §4.2: ``go_function.train.run`` writes the base's own test ``eval.json``,
  ``scores.npz``, ``eval_inputs.npz``, ``run.json`` there) except ``checkpoint.pt``
  and ``tp_base_provenance.json``;
* every file under ``results/go_function/`` except ``specs/**.json``,
  ``protocol_pins.json``, ``gates/*.json`` and the Tier B ``checkpoint.pt`` files
  registered with :func:`allow_checkpoints` before search mode (directory listings
  stay allowed: listing is not an ``open`` event);
* under ``dataset/``, ``preprocess/`` and ``results/``: any file whose name starts
  with ``test`` or contains ``_test``;
* the ``test`` split of a label npz: :mod:`go_function_fusion.data` reads train
  and valid members only and calls ``check_split`` (the npz is hash-verified as a
  whole file).
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Set

from go_ppi_fusion.guard import TestAccessError as _PpiTestAccessError

ROOT = Path(__file__).resolve().parents[1]

BLOCKED_GLOBS = (
    "dataset/go_function/dgz/*/test_data.pkl",
    "dataset/go_function/dgz/*/predictions_*.pkl",
    "dataset/go_function/dgz/*/test_diamond.res",
    "results/go_function_fusion*/test",
    "results/go_function_fusion*/test/*",
    "results/go_function_fusion*/diamond/*test*",
)
GF_RESULTS = "results/go_function/"
GF_ALLOWED_GLOBS = ("results/go_function/specs/*.json", "results/go_function/protocol_pins.json",
                    "results/go_function/gates/*.json")
TP_BASES_GLOB = "results/go_function_fusion*/tp_bases/*"
TP_BASES_ALLOWED = ("checkpoint.pt", "tp_base_provenance.json")
NAME_RULE_DIRS = ("dataset/", "preprocess/", "results/")
ALLOWED_SPLITS = ("train", "valid", "validation")


class TestAccessError(_PpiTestAccessError):
    """Raised when search mode tries to read a test-partition artefact."""

    __test__ = False  # not a pytest class


_SEARCH_MODE = False
_HOOK_INSTALLED = False
_ALLOWED_CHECKPOINTS: Set[str] = set()


def _rel(path: Any) -> Optional[str]:
    """Repo-relative posix path, or None for a path outside the repo."""
    try:
        p = Path(os.fsdecode(path))
    except TypeError:
        return None
    if not p.is_absolute():
        p = Path.cwd() / p
    p = Path(os.path.normpath(p))
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return p.resolve().relative_to(ROOT.resolve()).as_posix()
        except (ValueError, OSError):
            return None


def is_test_path(path: Any) -> bool:
    """True if ``path`` is blocked in search mode (see module docstring)."""
    rel = _rel(path)
    if rel is None:
        return False
    if any(fnmatch.fnmatchcase(rel, g) for g in BLOCKED_GLOBS):
        return True
    if fnmatch.fnmatchcase(rel, TP_BASES_GLOB):
        return rel.rsplit("/", 1)[-1] not in TP_BASES_ALLOWED
    if rel.startswith(GF_RESULTS):
        if rel in _ALLOWED_CHECKPOINTS:
            return False
        return not any(fnmatch.fnmatchcase(rel, g) for g in GF_ALLOWED_GLOBS)
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith(NAME_RULE_DIRS) and (name.startswith("test") or "_test" in name):
        return True
    return False


def allow_checkpoints(rel_paths: Iterable[str]) -> None:
    """Register the Tier B ``checkpoint.pt`` files named at G-TP (before search mode)."""
    if _SEARCH_MODE:
        raise TestAccessError("allow_checkpoints must be called before enter_search_mode()")
    for r in rel_paths:
        r = str(r)
        if not (r.startswith(GF_RESULTS) and r.endswith("/checkpoint.pt")):
            raise ValueError(f"only results/go_function/**/checkpoint.pt can be allowed: {r}")
        _ALLOWED_CHECKPOINTS.add(r)


def _audit(event: str, args) -> None:
    if not _SEARCH_MODE or event != "open" or not args:
        return
    path = args[0]
    if isinstance(path, int) or path is None:
        return
    if is_test_path(path):
        raise TestAccessError(f"search mode forbids opening {path}")


def enter_search_mode() -> None:
    """Block test access for the rest of this process (cannot be undone)."""
    global _SEARCH_MODE, _HOOK_INSTALLED
    _SEARCH_MODE = True
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True


def in_search_mode() -> bool:
    return _SEARCH_MODE


def check_path(path: Any, split: Optional[str] = None) -> None:
    """Raise in search mode if ``path`` (or the ``split`` name) is test data."""
    if not _SEARCH_MODE:
        return
    if split is not None:
        check_split(split)
    if is_test_path(path):
        raise TestAccessError(f"search mode forbids test access: {path} (split={split!r})")


def check_split(split: str) -> None:
    if _SEARCH_MODE and split not in ALLOWED_SPLITS:
        raise TestAccessError(f"search mode forbids split {split!r}")


# ------------------------------------------------------------------ search log
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SearchLog:
    """Append-only ``search_log.jsonl``: one JSON object per event (selection
    cells, guard state, violations). Paths in payloads must be repo-relative."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        rec = {"utc": utc_now(), "event": event, "search_mode": _SEARCH_MODE, **payload}
        line = json.dumps(rec, sort_keys=True, default=str)
        if "/Users/" in line or str(ROOT) in line:
            raise ValueError("search log entries must not contain absolute paths")
        with open(self.path, "a") as fh:
            fh.write(line + "\n")

    def violation(self, exc: BaseException) -> None:
        msg = re.sub(r"/Users/[^\s'\"]*", "<path>", str(exc).replace(str(ROOT) + "/", ""))
        self.log("guard_violation", error=type(exc).__name__, message=msg)
