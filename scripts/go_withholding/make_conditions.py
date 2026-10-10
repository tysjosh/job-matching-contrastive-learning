"""Write the target-free validation set (G2) or every withholding condition (G3).

``--mode validation`` (design 3.2): ``validation.jsonl``, ``validation_positive.jsonl``,
``negative_pools_validation.jsonl`` and ``manifest.json`` with every validation
protein annotated with any selected target removed (any role);
``append_pin("G2-validation", ...)``.

``--mode conditions`` (design 3.3, 3.4): one dir per ``all`` and
``<pack>_k<k>`` (``target_table["n_conditions"]`` dirs), each with
``train.jsonl``, ``negative_pools.jsonl`` and ``withholding_manifest.json``.
Fatal (exit 2) on: invalid k/pack, k=0 removal above the cap, W meeting the
retained sets, not exactly k retained, residual W/V proteins after writing, a
retained protein lost from the pools, < 3 GB free disk, non-idempotent
regeneration, ``unannotated_anchors.ids`` disagreeing with the pruned index, or
a written-count != ``n_conditions``. ``append_pin("G3", ...)`` once all succeed.

A pin is appended only if the latest pin of that gate does not already hold the
same values (an idempotent re-run adds nothing).

Usage (protocol section 10)::

    "$PY" "$R/scripts/go_withholding/make_conditions.py" --mode validation \\
      --target-table "$R/preprocess/go_withholding/targets/target_table.json" \\
      --base-splits "$R/preprocess/go_withholding/splits" \\
      --out "$R/preprocess/go_withholding/validation_targetfree"
    "$PY" "$R/scripts/go_withholding/make_conditions.py" --mode conditions \\
      --target-table "$R/preprocess/go_withholding/targets/target_table.json" \\
      --base-splits "$R/preprocess/go_withholding/splits" \\
      --targetfree-validation "$R/preprocess/go_withholding/validation_targetfree" \\
      --out "$R/preprocess/go_withholding/conditions"
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding as wh  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("make_conditions")

#: Minimum free disk before writing conditions (design 3.3). Read at call time.
#: 5 GB covers the AM1 grid's ~4.4 GB of conditions (121 x ~36 MB); the pre-AM1
#: 3 GB guard was sized for 41 conditions (~1.5 GB).
MIN_FREE_BYTES = 5 * 10 ** 9


def _require_pin(gate: str) -> None:
    if not any(e.get("gate") == gate for e in wc.read_pins()):
        raise RuntimeError(f"pins file {wc.pins_path()} has no {gate} entry")


def _pin_once(gate: str, values: Dict[str, Any]) -> bool:
    latest = [e for e in wc.read_pins() if e.get("gate") == gate]
    if latest:
        prev = {k: v for k, v in latest[-1].items() if k not in ("gate", "time")}
        if prev == json.loads(json.dumps(values)):
            return False
    wc.append_pin(gate, values)
    return True


def load_table(table_path: Path) -> Dict[str, Any]:
    """The table, after checking its ``.sha256`` sidecar."""
    sidecar = table_path.with_name(table_path.name + ".sha256")
    digest = wc.sha256_file(table_path)
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError(f"{table_path} does not match {sidecar}")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if table.get("schema_version") != 2:
        raise RuntimeError(f"{table_path}: schema_version {table.get('schema_version')} != 2")
    return {"table": table, "sha256": digest}


def check_base(base_splits: Path) -> Dict[str, Any]:
    frozen = json.loads((base_splits / "frozen_hashes.json").read_text(encoding="utf-8"))
    for key, name in (("train_sha256", "train.jsonl"), ("validation_sha256", "validation.jsonl"),
                      ("negative_pools_sha256", "negative_pools.jsonl")):
        if wc.sha256_file(base_splits / name) != frozen[key]:
            raise RuntimeError(f"{base_splits / name} differs from frozen_hashes[{key}]")
    return frozen


def run_validation(table_path: Path, base_splits: Path, out: Path,
                   converted: Path) -> Dict[str, Any]:
    t = load_table(table_path)
    check_base(base_splits)
    _require_pin("G2-targets")
    base = wh.load_base(base_splits, converted)
    manifest = wh.build_targetfree_validation(t["table"], base_splits, out, base=base,
                                              table_sha=t["sha256"])
    pinned = _pin_once("G2-validation", {
        "manifest_sha256": manifest["manifest_sha256"], "target_table_sha256": t["sha256"],
        "V_size": manifest["V_size"], "records_kept": manifest["records_kept"],
        "pool_rows": manifest["pool_rows"], "sha256s": manifest["sha256s"]})
    return {"status": manifest["status"], "pinned": pinned,
            **{k: manifest[k] for k in ("V_size", "records_kept", "records_dropped",
                                        "positive_records", "pool_rows", "pool_rows_dropped")}}


def _free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def run_conditions(table_path: Path, base_splits: Path, targetfree: Path, out: Path,
                   converted: Path) -> Dict[str, Any]:
    from go_ppi_domain.go_ontology import GoIndex

    t = load_table(table_path)
    table = t["table"]
    check_base(base_splits)
    _require_pin("G2-validation")
    vm = json.loads((targetfree / "manifest.json").read_text(encoding="utf-8"))
    if vm.get("target_table_sha256") != t["sha256"]:
        raise RuntimeError("target-free validation was built from a different target table")
    free = _free_bytes(out)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"only {free / 1e9:.2f} GB free (< {MIN_FREE_BYTES / 1e9:.0f} GB)")
    wc.assert_writable_target(out)
    cond_ids = wh.condition_ids(table)
    if len(cond_ids) != int(table["n_conditions"]):
        raise RuntimeError(f"{len(cond_ids)} condition ids != n_conditions {table['n_conditions']}")
    base = wh.load_base(base_splits, converted)
    index = GoIndex.load(Path(converted) / "go_index_P.pkl")
    manifests: Dict[str, str] = {}
    stats: Dict[str, Any] = {}
    for cid in cond_ids:
        m = wh.build_condition(table, cid, base_splits, targetfree, out / cid, base=base,
                               table_sha=t["sha256"], index=index)
        manifests[cid] = m["manifest_sha256"]
        stats[cid] = {"status": m["status"], "withheld_count": m["withheld_count"],
                      "removal_frac": m["removal"]["removal_frac"],
                      "pool_member_share_removed": m["removal"]["pool_member_share_removed"],
                      "anchors_dropped_empty_pool": len(m["removal"]["anchors_dropped_empty_pool"]),
                      "unannotated": {k: m["unannotated_anchors"][k]
                                      for k in ("pool_anchors", "record_anchors", "records")}}
    written = sorted(p.name for p in Path(out).iterdir()
                     if p.is_dir() and (p / wh.MANIFEST).is_file())
    if written != sorted(cond_ids) or len(written) != int(table["n_conditions"]):
        raise RuntimeError(f"{len(written)} condition dirs written != n_conditions "
                           f"{table['n_conditions']}")
    pinned = _pin_once("G3", {"n_conditions": len(written), "manifests": manifests})
    return {"n_conditions": len(written), "pinned": pinned, "conditions": stats}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=["validation", "conditions"])
    ap.add_argument("--target-table", type=Path, default=None)
    ap.add_argument("--base-splits", type=Path, default=None)
    ap.add_argument("--targetfree-validation", type=Path, default=None)
    ap.add_argument("--converted", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    table = Path(args.target_table or (Path(wc.TARGETS) / "target_table.json")).resolve()
    base = Path(args.base_splits or wc.SPLITS).resolve()
    converted = Path(args.converted or wc.CONVERTED).resolve()
    try:
        if args.mode == "validation":
            summary = run_validation(table, base, Path(args.out or wc.VALIDATION_TF).resolve(),
                                     converted)
        else:
            summary = run_conditions(
                table, base, Path(args.targetfree_validation or wc.VALIDATION_TF).resolve(),
                Path(args.out or wc.CONDITIONS).resolve(), converted)
    except Exception as exc:  # noqa: BLE001 - every generator failure is fatal
        logger.error("make_conditions --mode %s failed: %s: %s", args.mode,
                     type(exc).__name__, exc)
        return 2
    summary["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
