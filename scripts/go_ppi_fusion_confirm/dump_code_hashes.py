#!/usr/bin/env python3
"""Mac side, right before the step −1 transfer: dump ``fc.code_hashes()`` of the code tree.

``verify_transfer.py --code-manifest`` compares the transferred tree with this dump on the
remote, so a partial or stale copy of a study file is caught before ``make_pins.py``
records ``pins.code`` (protocol_go_ppi_fusion.md §6.2). Writes repo-relative paths only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go_ppi_domain import fusion_confirm as fc  # noqa: E402


def main(argv=None) -> int:
    fc.require_hashseed()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path(fc.CODE_MANIFEST_REL))
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    hashes = fc.code_hashes()
    body = {"schema": "go_ppi_fusion_confirm/code_hashes/v1", "created_utc": fc.utc_now(),
            "globs": list(fc.CODE_GLOBS), "git": fc.git_record(), "code": hashes}
    out = fc.write_json(args.out, body)
    print(f"wrote {fc.rel(out)}: {len(hashes)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
