"""Synthetic grid fixtures shared by the sweep-driver and statistics tests.

A singleton-pack target table (AM1), run dirs with synthetic ``lc_manifest.json``
that pass the design 12.3 group checks, ``eval/metrics.json`` and the
``DONE.json`` / ``GROUP_OK.json`` markers exactly as the sweep writes them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from go_ppi_domain import withholding as wh
from go_ppi_domain import withholding_constants as wc


def make_table(n_targets: int = 3) -> Dict[str, Any]:
    targets, packs = [], []
    for i in range(n_targets):
        go, pid = f"GO:{i + 1:07d}", f"p{i + 1:02d}"
        targets.append({"go_id": go, "pack_id": pid, "stratum": "low" if i % 2 == 0 else "high",
                        "S_sib": 100 * (i + 1), "S_2": 50 * (n_targets - i),
                        "test_anchors": [f"a{i}"], "nbr_test_proteins_sib": [f"n{i}"]})
        packs.append({"pack_id": pid, "targets": [go]})
    n_cond = 5 * n_targets + 1
    return {"schema_version": 2, "n_targets": n_targets, "n_packs": n_targets,
            "n_conditions": n_cond, "n_runs": 4 * 5 * n_cond, "targets": targets,
            "packs": packs}


def write_table(table: Dict[str, Any], targets_dir: Path) -> Path:
    targets_dir.mkdir(parents=True, exist_ok=True)
    path = targets_dir / "target_table.json"
    path.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (targets_dir / "target_table.json.sha256").write_text(wc.sha256_file(path) + "\n")
    return path


def lc_manifest(cond: str, cell: str, seed: int, epochs: int = 15,
                sel: Optional[Dict[str, str]] = None, **over) -> Dict[str, Any]:
    tiered = cell in ("neg_only", "both")
    sel = sel or {str(e): (f"B{e}" if tiered else f"A{e}") for e in range(epochs)}
    m = {
        "domain": "go_ppi", "seed": seed, "epochs": epochs, "train_records": 100,
        "pilot": False,
        "withholding": {"cond_id": cond, "cell": cell, "pack_id": wh.parse_cond_id(cond)[0],
                        "assertions": {"post_seen_ids_disjoint_W": "ok",
                                       "post_final_checkpoint": "ok"},
                        "selected_checkpoint": f"checkpoint_epoch_{epochs - 1}.pt",
                        "selected_checkpoint_sha256": "c" * 64},
        "epoch_stats": [{"epoch": e, "ok_batches": 10, "failed_batches": 0}
                        for e in range(epochs)],
        "negative_selection": {"selection_sha256_by_epoch": sel,
                               "calls_by_epoch": {str(e): 5 for e in range(epochs)},
                               "unannotated_anchor_calls_by_epoch": {str(e): 2
                                                                     for e in range(epochs)},
                               "unannotated_anchor_window": True,
                               "selector_none_returns": 0, "selector_exceptions": 0},
        "go_relation_loss": {"applied_batches": 10 if cell in ("loss_only", "both") else 0},
    }
    m.update(over)
    return m


def metrics_for(table: Dict[str, Any], run_id: str, cond: str,
                value: Callable[[str, str, Any, str, int], float]) -> Dict[str, Any]:
    pack, k = wh.parse_cond_id(cond)
    cell = run_id.split("__")[2]
    seed = int(run_id.rsplit("__s", 1)[1])
    per = {}
    for t in table["targets"]:
        v = value(t["go_id"], t["pack_id"], k, cell, seed)
        per[t["go_id"]] = {"hard_auc": v, "macro_query_hard_auc": v, "macro_mrr": v,
                           "queries": 1, "n_pos": 1, "n_weak": 1}
    packs = [p["pack_id"] for p in table["packs"]] if pack == "all" else [pack]
    coll = {p: {"hard_auc": 0.7 + 0.001 * seed / 100.0 + (0.002 if cell != "baseline" else 0),
                "queries": 3} for p in packs}
    return {"run_id": run_id, "cond_id": cond, "per_target": per, "collateral": coll,
            "global": {"hard_auc": 0.7}}


def write_run(runs: Path, run_id: str, manifest: Dict[str, Any],
              metrics: Optional[Dict[str, Any]] = None, done: bool = True,
              relation: Optional[Dict[str, Any]] = None) -> Path:
    d = runs / run_id
    (d / "eval").mkdir(parents=True, exist_ok=True)
    (d / "lc_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (d / manifest["withholding"]["selected_checkpoint"]).write_bytes(b"ckpt")
    if metrics is not None:
        (d / "eval" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if relation is not None:
        (d / "eval" / "relation.json").write_text(json.dumps(relation), encoding="utf-8")
    if done:
        (d / "DONE.json").write_text(json.dumps({
            "lc_manifest_sha256": wc.sha256_file(d / "lc_manifest.json"),
            "metrics_sha256": (wc.sha256_file(d / "eval" / "metrics.json")
                               if metrics is not None else None),
            "code_sha256": "code", "finished": "t"}), encoding="utf-8")
    return d


def write_group_ok(dirs: List[Path]) -> None:
    payload = {"run_ids": [d.name for d in dirs],
               "lc_manifest_sha256": {d.name: wc.sha256_file(d / "lc_manifest.json")
                                      for d in dirs},
               "checked": "t", "code_sha256": "code"}
    for d in dirs:
        (d / "GROUP_OK.json").write_text(json.dumps(payload), encoding="utf-8")


def write_group(runs: Path, table: Dict[str, Any], cond: str, seed: int,
                value: Optional[Callable] = None, checked: bool = True,
                relation_excess: Optional[Callable] = None) -> List[Path]:
    dirs = []
    for cell in wc.CELLS:
        rid = wc.run_id_for(cond, cell, seed)
        met = metrics_for(table, rid, cond, value) if value is not None else None
        rel = None
        if relation_excess is not None:
            rel = {"per_target": {t["go_id"]: {"excess": relation_excess(t["go_id"], cond, cell,
                                                                         seed)}
                                  for t in table["targets"]},
                   "global": {"spearman": 0.1}}
        dirs.append(write_run(runs, rid, lc_manifest(cond, cell, seed), met, relation=rel))
    if checked:
        write_group_ok(dirs)
    return dirs
