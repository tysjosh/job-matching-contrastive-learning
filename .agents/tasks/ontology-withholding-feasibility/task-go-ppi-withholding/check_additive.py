"""Prove that the withholding study's edits to shared files are additive (plan item 1).

Compares each pre-edit snapshot in ``pre_edit/`` with the current file using
``difflib.SequenceMatcher`` over lines and exits 0 only if:

* ``batch_processor.py`` is byte-identical;
* ``trainer.py``, ``data_structures.py`` and ``go_ontology.py`` show only
  ``equal``/``insert`` opcodes;
* in ``negative_selector.py`` every ``replace`` opcode is (a) one of the two
  ``if self.tier_sampling == "random_window":`` -> ``if use_random_window:``
  lines, or (b) a line inside ``__init__``'s ``logger.info`` call, the
  ``__init__`` signature or ``from_config`` that only GAINS characters and
  mentions ``unannotated_anchor_window`` (design section 5.6); at most two (a)s;
* in ``loss_engine.py`` every ``replace`` is a single-line wrap of a ``return``
  statement inside ``compute_loss`` into ``return self._with_go_relation(<old
  expression>, ...)`` (characters only gained); the file is either unchanged or
  has exactly three such wraps (design section 5.3);
* every ``+`` line of each saved user ``.diff`` is still present in the current
  file, in order.

A replace block may also contain pure inserted lines (difflib can merge an
adjacent insertion into a replace); those are matched by aligning each old line
of the block with an allowed modified new line, in order.

Exit 0 on success, 1 on any violation (each printed).
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

ROOT = Path("/Users/olukotunjosh/Downloads/CDCL")
HERE = Path(__file__).resolve().parent
PRE = HERE / "pre_edit"

FILES = {
    "batch_processor.py": ROOT / "contrastive_learning/batch_processor.py",
    "loss_engine.py": ROOT / "contrastive_learning/loss_engine.py",
    "trainer.py": ROOT / "contrastive_learning/trainer.py",
    "data_structures.py": ROOT / "contrastive_learning/data_structures.py",
    "negative_selector.py": ROOT / "go_ppi_domain/negative_selector.py",
    "go_ontology.py": ROOT / "go_ppi_domain/go_ontology.py",
}
USER_DIFFS = ("batch_processor.py", "loss_engine.py", "trainer.py")
INSERT_ONLY = ("trainer.py", "data_structures.py", "go_ontology.py")


def _gains_only(old: str, new: str) -> bool:
    """True when ``new`` is ``old`` with characters inserted (none removed/changed)."""
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return all(tag in ("equal", "insert") for tag, *_ in sm.get_opcodes())


def _function_span(lines: list[str], header_re: str) -> tuple[int, int]:
    """[start, end) line indices of the first ``def`` matching ``header_re``."""
    for i, line in enumerate(lines):
        if re.match(header_re, line):
            indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                s = lines[j]
                if s.strip() and (len(s) - len(s.lstrip())) <= indent and not s.lstrip().startswith(")"):
                    break
                j += 1
            return i, j
    return -1, -1


def _align(old_block: list[str], new_block: list[str], allowed) -> bool:
    """Every old line maps (in order) to a new line with ``allowed(old_idx, old, new)``."""
    j = 0
    for k, old in enumerate(old_block):
        while j < len(new_block) and not allowed(k, old, new_block[j]):
            j += 1
        if j == len(new_block):
            return False
        j += 1
    return True


def check_selector(old: list[str], new: list[str], errors: list[str]) -> None:
    init_span = _function_span(old, r"\s+def __init__\(")
    from_cfg_span = _function_span(old, r"\s+def from_config\(")
    n_window = 0
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "insert"):
            continue
        if tag == "delete":
            errors.append(f"negative_selector.py: deleted lines {i1 + 1}-{i2}")
            continue
        kinds: list[str] = []

        def allowed(k, o, n, i1=i1, kinds=kinds):
            idx = i1 + k
            if o.strip() == 'if self.tier_sampling == "random_window":':
                ok = n.strip() == "if use_random_window:" and (
                    len(o) - len(o.lstrip()) == len(n) - len(n.lstrip()))
                if ok:
                    kinds.append("window")
                return ok
            in_scope = (init_span[0] <= idx < init_span[1]
                        or from_cfg_span[0] <= idx < from_cfg_span[1])
            return in_scope and _gains_only(o, n) and "unannotated_anchor_window" in n

        if not _align(old[i1:i2], new[j1:j2], allowed):
            errors.append(f"negative_selector.py: disallowed replace at old lines "
                          f"{i1 + 1}-{i2}: {old[i1:i2]!r} -> {new[j1:j2]!r}")
        n_window += kinds.count("window")
    if n_window > 2:
        errors.append(f"negative_selector.py: {n_window} random_window tests replaced (max 2)")
    print(f"  negative_selector.py: random_window tests replaced = {n_window}")


def check_loss_engine(old: list[str], new: list[str], errors: list[str]) -> None:
    span = _function_span(old, r"\s+def compute_loss\(")
    wraps = 0
    changed = old != new
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "insert"):
            continue
        if tag == "delete":
            errors.append(f"loss_engine.py: deleted lines {i1 + 1}-{i2}")
            continue
        found: list[int] = []

        def allowed(k, o, n, i1=i1, found=found):
            idx = i1 + k
            s_old, s_new = o.strip(), n.strip()
            if not (span[0] <= idx < span[1] and s_old.startswith("return ")):
                return False
            expr = s_old[len("return "):]
            ok = (s_new.startswith(f"return self._with_go_relation({expr},")
                  and _gains_only(o, n)
                  and len(o) - len(o.lstrip()) == len(n) - len(n.lstrip()))
            if ok:
                found.append(idx)
            return ok

        if not _align(old[i1:i2], new[j1:j2], allowed):
            errors.append(f"loss_engine.py: disallowed replace at old lines "
                          f"{i1 + 1}-{i2}: {old[i1:i2]!r} -> {new[j1:j2]!r}")
        wraps += len(found)
    if changed and wraps != 3:
        errors.append(f"loss_engine.py: {wraps} compute_loss return wraps (need exactly 3 "
                      f"once the file is edited)")
    print(f"  loss_engine.py: changed={changed}, compute_loss return wraps = {wraps}")


def check_insert_only(name: str, old: list[str], new: list[str], errors: list[str]) -> None:
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    n_ins = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            n_ins += j2 - j1
        elif tag != "equal":
            errors.append(f"{name}: {tag} at old lines {i1 + 1}-{i2} -> new {j1 + 1}-{j2}")
    print(f"  {name}: inserted lines = {n_ins}")


def check_user_diff(name: str, current: list[str], errors: list[str]) -> None:
    plus = []
    for line in (PRE / f"{name}.diff").read_text(encoding="utf-8").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            plus.append(line[1:])
    stripped = [c.rstrip("\n") for c in current]
    j = 0
    for k, p in enumerate(plus):
        while j < len(stripped) and stripped[j] != p:
            j += 1
        if j == len(stripped):
            errors.append(f"{name}: user diff '+' line #{k + 1} missing or out of order: {p!r}")
            return
        j += 1
    print(f"  {name}: {len(plus)} user '+' lines present in order")


def main() -> int:
    errors: list[str] = []
    for name, path in FILES.items():
        snap = PRE / name
        if not snap.exists():
            errors.append(f"missing snapshot {snap}")
            continue
        old_b, new_b = snap.read_bytes(), path.read_bytes()
        old = old_b.decode("utf-8").splitlines(keepends=True)
        new = new_b.decode("utf-8").splitlines(keepends=True)
        if name == "batch_processor.py":
            if old_b != new_b:
                errors.append("batch_processor.py: not byte-identical to the snapshot")
            print(f"  batch_processor.py: byte-identical = {old_b == new_b}")
        elif name in INSERT_ONLY:
            check_insert_only(name, old, new, errors)
        elif name == "negative_selector.py":
            check_selector(old, new, errors)
        elif name == "loss_engine.py":
            check_loss_engine(old, new, errors)
        if name in USER_DIFFS:
            check_user_diff(name, new, errors)
    if errors:
        print("check_additive: FAIL")
        for e in errors:
            print("  -", e)
        return 1
    print("check_additive: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
