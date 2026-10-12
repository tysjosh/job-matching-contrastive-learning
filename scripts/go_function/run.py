"""Run one training run from a JSON spec (design §7; plan item 12).

  preflight -> train -> eval_inputs.npz -> eval.json -> scores.npz -> run.json -> DONE.json

The spec schema is design §7.1 plus ``mode`` (study | gate | pilot | smoke),
``max_train_batches`` (smoke only) and ``nf4_radius`` ("dgz"; "class" only for
the G4 diagnostic). Unknown fields are rejected. Outputs go to the spec's
``out_dir`` (repo-relative, under results/go_function/). Any failure after the
spec is validated writes ``FAILED.json`` with the stage (preflight, train, eval).

``--device`` is auto | cuda | cpu; ``auto`` resolves to cuda when available, else
cpu. MPS is never used.

Exit codes: 0 ok; 2 fatal (schema, pins, assertions, non-finite loss); 3 protocol
stop; 4 refused by state (missing pin or norm, output dir not empty).

Usage:
  .venv/bin/python scripts/go_function/run.py --spec results/go_function/specs/<x>.json --device cpu
  .venv/bin/python scripts/go_function/run.py --spec <spec> --device cuda --rescore   (sweep.py rescore)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA initialises

from go_function.constants import EXIT_FATAL, EXIT_REFUSED  # noqa: E402
from go_function.runtime import DEVICES, FatalError, ProtocolStop, resolve_device, setup_logging  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="run spec JSON")
    ap.add_argument("--device", choices=DEVICES, default="auto")
    ap.add_argument("--rescore", action="store_true",
                    help="re-run only the evaluation of a run with FAILED.json stage eval")
    args = ap.parse_args(argv)
    from go_function import train as T
    log = setup_logging("run")
    try:
        try:
            device = resolve_device(args.device)
        except ValueError as e:
            raise FatalError(str(e)) from None
        p = Path(args.spec)
        p = p if p.is_absolute() else Path.cwd() / p
        try:
            raw = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise FatalError(f"cannot read spec {p.name}: {e.__class__.__name__}") from None
        log.info("spec %s on device %s", p.name, device)
        return (T.rescore if args.rescore else T.run)(raw, device, log=log)
    except T.Refused as e:
        log.error("%s", e)
        return EXIT_REFUSED
    except ProtocolStop as e:
        log.error("%s", e)
        return e.exit_code
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main())
