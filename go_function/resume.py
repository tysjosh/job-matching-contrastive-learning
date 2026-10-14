"""Which run dirs are continued from a training snapshot (go_function.train module
docstring, "Checkpoint and resume"). Kept free of torch and of go_function.evaluate
so the sweep driver and the pilot (whose import probe forbids the evaluator) can
use it."""
from __future__ import annotations

import json
from pathlib import Path

RESUME_FILE = "resume.pt"


def resumable(out_dir: Path) -> bool:
    """A run dir that ``train.run`` continues instead of refusing: it holds resume.pt,
    no DONE.json, and no FAILED.json except one left by an interrupt."""
    out_dir = Path(out_dir)
    if not (out_dir / RESUME_FILE).exists() or (out_dir / "DONE.json").exists():
        return False
    fp = out_dir / "FAILED.json"
    if not fp.exists():
        return True
    try:
        return json.loads(fp.read_text()).get("error") == "KeyboardInterrupt"
    except ValueError:
        return False
