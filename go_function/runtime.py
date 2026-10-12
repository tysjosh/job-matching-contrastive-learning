"""Runtime helpers shared by every CLI (plan decision 2).

- ``ROOT``: the repository root, computed from this file's location.
- ``resolve_device``: ``auto`` gives ``cuda`` if available, else ``cpu``. MPS is
  never selected, and asking for it is an error (local runs are CPU only).
- ``repo_rel``: the repo-relative POSIX path written into every artifact; a
  path outside ROOT is an error, so no absolute path can leak into outputs.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, Union

from go_function.constants import EXIT_FATAL, EXIT_STOP

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ("auto", "cuda", "cpu")


class FatalError(RuntimeError):
    """Fatal input or invariant failure (exit 2)."""
    exit_code = EXIT_FATAL


class ProtocolStop(RuntimeError):
    """A preregistered rule says stop and ask the user (exit 3)."""
    exit_code = EXIT_STOP


def resolve_device(name: str) -> str:
    """Map ``auto|cuda|cpu`` to a torch device string. Never returns ``mps``."""
    if name not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {name!r} (MPS is never used)")
    if name == "cpu":
        return "cpu"
    import torch
    has_cuda = bool(torch.cuda.is_available())
    if name == "cuda":
        if not has_cuda:
            raise ValueError("device 'cuda' requested but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if has_cuda else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is in the venv
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(name: str, out_dir: Optional[Path] = None,
                  level: int = logging.INFO) -> logging.Logger:
    """INFO to stderr, and to ``<out_dir>/log.txt`` when given (design §15)."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, "_go_function", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        h._go_function = True  # type: ignore[attr-defined]
        root.addHandler(h)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "log.txt")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return logging.getLogger(name)


def repo_rel(path: Union[str, Path]) -> str:
    """Repo-relative POSIX path of ``path``; raises if it is outside ROOT."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError(f"path is outside the repository root: {p}") from None


def repo_path(rel: str) -> Path:
    """Absolute path of a repo-relative path written by ``repo_rel``."""
    if Path(rel).is_absolute():
        raise ValueError(f"expected a repo-relative path, got {rel}")
    return ROOT / rel
