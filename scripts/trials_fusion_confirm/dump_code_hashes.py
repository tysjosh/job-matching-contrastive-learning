#!/usr/bin/env python3
"""Mac side, right before the step -1 rsync: dump ``fc.code_hashes()`` of the code tree.

``verify_transfer.py --code-manifest`` compares the transferred tree with this dump on
the remote, so a partial or stale rsync of an uncommitted study file is caught before
``make_pins.py`` records ``pins.code`` (review finding 4). Writes repo-relative paths only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trials_domain import fusion_confirm as fc  # noqa: E402

DEFAULT_OUT = ".agents/tasks/ontology-withholding-feasibility/code_hashes_trials_fusion.mac.json"


def main(argv=None) -> int:
    fc.require_hashseed()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    out = fc.refuse_existing(args.out)
    hashes = fc.code_hashes()
    body = {"schema": "trials_fusion_confirm/code_hashes/v1", "created_utc": fc.utc_now(),
            "globs": list(fc.CODE_GLOBS), "git": fc.git_record(), "code": hashes}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {fc.rel(out)}: {len(hashes)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
