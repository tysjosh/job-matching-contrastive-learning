"""Tier B source (b) (REMOTE ONLY; protocol §4.2): train a TP-trained ``all`` base out of
order from a GF materialized spec (read only, byte-identical, sha256 recorded) into a
fusion-owned directory, via the imported, unmodified ``go_function.train.run(...,
out_dir=...)``. Writes ``tp_base_provenance.json`` (actual directory, spec sha256,
``go_function/`` code hashes). The wrapper never opens the base's own test
``eval.json`` / ``scores.npz`` that ``train.run`` writes (§10.4).

Usage (remote):
  .venv/bin/python scripts/go_function_fusion/train_tp_base.py \\
      --spec results/go_function/specs/<aspect>/<feature>/<run>.json --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_pins as MP  # noqa: E402


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    a = ap.parse_args(argv)
    MP.refuse_on_mac("train_tp_base.py (no training on the Mac)")
    from go_function import train as T
    from go_function.pins import sha256_bytes, sha256_file
    from go_function.runtime import setup_logging
    spec_p = ROOT / a.spec
    if not MP.rel(spec_p).startswith("results/go_function/specs/"):
        raise MP.Fatal("--spec must be a GF materialized spec under results/go_function/specs/")
    raw_b = spec_p.read_bytes()
    spec = json.loads(raw_b)
    if spec.get("mode") != "study" or spec.get("pack") is not None or int(spec.get("seed", -1)) != 13:
        raise MP.Fatal("Tier B bases are study-mode 'all' runs with seed 13")
    out = ROOT / "results/go_function_fusion/tp_bases" / Path(spec["out_dir"]).name
    MP.assert_writable(out)
    code = {MP.rel(p): sha256_file(p) for p in sorted((ROOT / "go_function").glob("*.py"))}
    log = setup_logging("train_tp_base")
    rc = T.run(spec, a.device, log=log, out_dir=out)
    MP.write_json(out / "tp_base_provenance.json",
                  {"out_dir": MP.rel(out), "spec": MP.rel(spec_p), "spec_sha256": sha256_bytes(raw_b),
                   "go_function_code": code, "exit_code": rc,
                   "note": "go_function.train.run out_dir override ('tests only'), protocol §4.2, §16"})
    return rc


if __name__ == "__main__":
    sys.exit(MP.main_wrapper(run))
