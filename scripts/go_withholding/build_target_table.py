"""Build the frozen target table (design section 2.8, gate G2).

Preflight: ``frozen_hashes.json`` matches the converted/split files and the pins
file has a ``G1`` entry. Refuses to overwrite. Writes ``target_table.json``
(schema 2), ``target_table.json.sha256``, ``candidates.json`` (every term with
n(t) >= 40 and its rule flags) and ``append_pin("G2-targets", ...)``.

Packs are singletons (amendment AM1, protocol section 13): one target per
condition, so ``n_packs == n_targets`` and ``n_conditions = 5 * n_targets + 1``.

Exit 3 (protocol stop, ask the user): fewer than 10 matched pairs, or a
selected target failing the single-target removal cap or retention
feasibility. Exit 2: any other failure. The printed summary holds
counts only (protocol section 4: target names are not inspected before G2).

Usage (protocol section 10, G2)::

    "$PY" "$R/scripts/go_withholding/build_target_table.py" \\
      --converted "$R/preprocess/go_withholding/converted" \\
      --splits "$R/preprocess/go_withholding/splits" \\
      --obo "$R/dataset/go_ppi/bulk/go-basic.obo" \\
      --out "$R/preprocess/go_withholding/targets/target_table.json"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding_constants as wc  # noqa: E402
from go_ppi_domain import withholding_targets as wt  # noqa: E402

logger = logging.getLogger("build_target_table")

#: frozen_hashes key -> file (relative to converted / splits).
FROZEN_FILES = {"train_sha256": ("splits", "train.jsonl"),
                "validation_sha256": ("splits", "validation.jsonl"),
                "test_sha256": ("splits", "test.jsonl"),
                "negative_pools_sha256": ("splits", "negative_pools.jsonl"),
                "pairs_sha256": ("converted", "pairs.jsonl"),
                "proteins_sha256": ("converted", "proteins.jsonl"),
                "go_index_P_sha256": ("converted", "go_index_P.pkl")}


def preflight(converted: Path, splits: Path) -> Dict[str, Any]:
    frozen = json.loads((splits / "frozen_hashes.json").read_text(encoding="utf-8"))
    dirs = {"converted": converted, "splits": splits}
    for key, (where, name) in FROZEN_FILES.items():
        got = wc.sha256_file(dirs[where] / name)
        if got != frozen.get(key):
            raise RuntimeError(f"{dirs[where] / name} sha256 {got} != frozen_hashes[{key}]")
    if not any(e.get("gate") == "G1" for e in wc.read_pins()):
        raise RuntimeError(f"pins file {wc.pins_path()} has no G1 entry")
    return frozen


def run(converted: Path, splits: Path, obo: Path, out: Path,
        params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t0 = time.time()
    out_dir = out.parent
    sha_path = out.with_name(out.name + ".sha256")
    cand_path = out_dir / "candidates.json"
    for p in (out, sha_path, cand_path):
        wc.assert_writable_target(p)
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    frozen = preflight(converted, splits)
    inputs = {"train_sha256": frozen["train_sha256"], "test_sha256": frozen["test_sha256"],
              "validation_sha256": frozen["validation_sha256"],
              "negative_pools_sha256": frozen["negative_pools_sha256"],
              "go_index_P_sha256": frozen["go_index_P_sha256"],
              "obo_sha256": wc.sha256_file(obo), "code_sha256": wc.code_sha256()[0]}
    ctx = wt.load_context(converted, splits, obo)
    table, rows = wt.build_target_table(ctx, inputs, params=params)
    data = wt.table_json_bytes(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    digest = wc.sha256_file(out)
    sha_path.write_text(digest + "\n", encoding="utf-8")
    cand_path.write_text(json.dumps({"schema_version": 1, "params": table["params"],
                                     "rows": rows}, indent=1, sort_keys=True) + "\n",
                         encoding="utf-8")
    summary = {"target_table_sha256": digest, "candidates_sha256": wc.sha256_file(cand_path),
               "n_candidates": table["n_candidates"], "n_non_nested": table["n_non_nested"],
               "n_targets": table["n_targets"], "n_packs": table["n_packs"],
               "n_conditions": table["n_conditions"], "n_runs": table["n_runs"],
               "shortfall": table["shortfall"]}
    wc.append_pin("G2-targets", summary)
    summary.update({
        "n_low": table["n_low"], "n_high": table["n_high"],
        "n_pairs_accepted": table["n_pairs_accepted"],
        "strata_cuts": table["strata_cuts"], "balance": table["balance"],
        "replacement_queue_left": len(table["replacement_queue"]),
        "pack_removal_frac_k0": [round(p["removal_frac_k0"], 4) for p in table["packs"]],
        "pack_min_retention_eligible": [p["min_retention_eligible"] for p in table["packs"]],
        "wall_seconds": round(time.time() - t0, 1)})
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--converted", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--obo", type=Path, default=ROOT / "dataset/go_ppi/bulk/go-basic.obo")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        summary = run(Path(args.converted or wc.CONVERTED).resolve(),
                      Path(args.splits or wc.SPLITS).resolve(),
                      Path(args.obo).resolve(),
                      Path(args.out or (Path(wc.TARGETS) / "target_table.json")).resolve())
    except wt.TargetTableStop as exc:
        logger.error("PROTOCOL STOP (ask the user): %s", exc)
        return 3
    except Exception as exc:  # noqa: BLE001
        logger.error("build_target_table failed: %s: %s", type(exc).__name__, exc)
        return 2
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
