#!/usr/bin/env python3
"""Isolation import-scan guard for the ORCA package (Milestone 0 gate).

Feature: orca, Task 1.5 (Milestone 0 gate).

Requirement 7.6 (Isolation constraint)
--------------------------------------
    THE OSCAR_Pipeline modules SHALL contain no import of or reference to the
    ``orca/`` package, so that removing the ``orca/`` package leaves the
    pre-ORCA career path fully operational.

This test statically scans every OSCAR / non-ORCA *production* Python module in
the repository and asserts that none of them import from or reference the
top-level ``orca`` package. ORCA integrates with the shared pipeline only
through its three new seams (the ``make_loss_engine`` factory, the
negative-selector injection point, and the phase orchestrator), all of which
live *inside* the ``orca/`` package — never by adding ``import orca`` call
sites into OSCAR modules.

Scanning is done with the AST so that string mentions of "orca" in comments,
docstrings, or path literals do not trip the guard: only real ``import orca``
/ ``from orca ...`` statements count.

Excluded from the scan (legitimately allowed to reference ORCA):
  * the ``orca/`` package itself,
  * all test trees (``tests/``, ``**/tests/``, ``test_*.py``),
  * virtualenvs / caches / VCS / build dirs.

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_isolation_import_scan.py

Requirements: 7.6
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import pytest

# tests/orca/<this file> -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Directory names that are never OSCAR production code and are allowed to
# reference ORCA (or are simply irrelevant to the isolation guard).
_EXCLUDED_DIR_NAMES = {
    "orca",            # the ORCA package itself (allowed to self-reference)
    "tests",           # test trees may import orca freely
    ".venv",
    "venv",
    ".git",
    ".hypothesis",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
}

# ORCA entrypoint scripts (top-level runners) legitimately import the ``orca/``
# package — they exist ONLY to launch ORCA. They are not part of the OSCAR
# career pipeline, so removing ``orca/`` is expected to remove them too. The
# isolation guard concerns the OSCAR *pipeline* modules (Requirement 7.6), so
# these entrypoints are excluded by name.
_ALLOWED_ORCA_ENTRYPOINT_PREFIXES = ("run_orca", "probe_reliability")


def _is_orca_entrypoint(path: Path) -> bool:
    """True for top-level ORCA runner scripts (e.g. ``run_orca_training.py``)."""
    return any(path.name.startswith(p) for p in _ALLOWED_ORCA_ENTRYPOINT_PREFIXES)


def _iter_oscar_modules() -> List[Path]:
    """Yield every non-ORCA, non-test production ``.py`` file in the repo."""
    modules: List[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        parts = set(path.relative_to(REPO_ROOT).parts)
        if parts & _EXCLUDED_DIR_NAMES:
            continue
        # Defensively skip stray test files that live outside a tests/ dir.
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        # ORCA entrypoint runners are allowed to import orca (not OSCAR pipeline).
        if _is_orca_entrypoint(path):
            continue
        modules.append(path)
    return modules


def _orca_imports(path: Path) -> List[str]:
    """Return a list of ``orca``-package import statements found in ``path``.

    Uses the AST so only genuine ``import``/``from`` statements count; string
    or comment mentions of "orca" are ignored. Files that fail to parse are
    reported as an explicit finding rather than silently skipped.
    """
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - surfaces as a test failure
        return [f"<unparseable: {exc}>"]

    findings: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "orca" or alias.name.startswith("orca."):
                    findings.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # node.level > 0 is a relative import; those can't reach top-level orca.
            if node.level == 0 and (module == "orca" or module.startswith("orca.")):
                names = ", ".join(alias.name for alias in node.names)
                findings.append(f"from {module} import {names}")
    return findings


def test_oscar_modules_do_not_import_orca() -> None:
    """No OSCAR / non-test production module imports the ``orca`` package (Req 7.6)."""
    offenders: List[Tuple[str, List[str]]] = []
    scanned = 0
    for module in _iter_oscar_modules():
        scanned += 1
        imports = _orca_imports(module)
        if imports:
            offenders.append((str(module.relative_to(REPO_ROOT)), imports))

    # Guard against a broken scan silently passing (e.g. wrong root).
    assert scanned > 0, "isolation scan found no OSCAR modules to inspect"

    assert not offenders, (
        "Isolation constraint violated (Requirement 7.6): the following OSCAR "
        "modules import the orca/ package. ORCA must integrate only through its "
        "own seams so the pre-ORCA career path works with orca/ removed:\n"
        + "\n".join(f"  - {mod}: {stmts}" for mod, stmts in offenders)
    )


def test_contrastive_learning_package_is_orca_free() -> None:
    """The core OSCAR pipeline package specifically references no orca imports.

    ``contrastive_learning`` is the package ORCA layers on top of (loss engine,
    trainer, data structures). It is the highest-risk place for an accidental
    ``import orca`` to creep in, so it gets its own focused assertion.
    """
    pkg_root = REPO_ROOT / "contrastive_learning"
    assert pkg_root.is_dir(), "expected contrastive_learning/ package to exist"

    offenders: List[Tuple[str, List[str]]] = []
    for module in pkg_root.rglob("*.py"):
        if "__pycache__" in module.parts:
            continue
        imports = _orca_imports(module)
        if imports:
            offenders.append((str(module.relative_to(REPO_ROOT)), imports))

    assert not offenders, (
        "contrastive_learning/ must not import orca/ (Requirement 7.6):\n"
        + "\n".join(f"  - {mod}: {stmts}" for mod, stmts in offenders)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
