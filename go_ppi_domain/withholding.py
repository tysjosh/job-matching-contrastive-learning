"""Withholding generator and run-time condition assertions (design sections 3, 4.3).

A condition ``<pack>_k<k>`` withholds, for every target t of the pack, the
proteins W = U_t (T_train(t) minus R_k(t)), where T_train(t) is the target's
train-partition proteins in ANY role and R_k(t) the first k entries of its
retention order. The shared ``all`` condition withholds nothing. Under
amendment AM1 every pack holds exactly one target (one target per condition);
the code below stays generic over the table's packs.

Per condition (``build_condition``):

* train records keep base order; a record is dropped when its anchor OR its
  partner is in W (both directions);
* train pool rows with an anchor in W are dropped and W is removed from every
  remaining member list; retained proteins (not in W) keep their rows and stay
  pool members;
* empty-pool rule: a train anchor with no pool row, or with both lists empty
  after intersecting with the protein views, has its records dropped (also for
  ``all``); otherwise the selector would return ``None`` and the batch
  processor would train on its dummy negative;
* ``negative_pools.jsonl`` = filtered train rows, then the exact bytes of the
  shared target-free validation pool file.

Every written file is serialised as ``json.dumps(row, ensure_ascii=False) + "\\n"``
(the splitter's format; unchanged rows round-trip byte-identically). Generation
is idempotent: an existing output dir is regenerated in memory and must match
byte-for-byte, otherwise ``RuntimeError``. Every check failure raises
``RuntimeError`` naming the first 5 offending ids.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

from go_ppi_domain import withholding_constants as wc

K = Union[int, Literal["all"]]
MANIFEST = "withholding_manifest.json"


# ------------------------------------------------------------------ helpers
def _dump(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False) + "\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _first(ids: Iterable[str], n: int = 5) -> List[str]:
    return sorted(ids)[:n]


def _fail(check: str, ids: Iterable[str] = (), detail: str = "") -> None:
    ids = list(ids)
    msg = f"{check}: {detail}" if detail else check
    if ids:
        msg += f" (first ids: {_first(ids)})"
    raise RuntimeError(msg)


def _read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line if line.endswith("\n") else line + "\n" for line in fh if line.strip()]


def _record_pids(rec: dict) -> Tuple[str, str]:
    return str(rec["resume"]["protein_id"]), str(rec["job"]["partner_id"])


def _pool_proteins(rows: Iterable[dict]) -> Tuple[Set[str], Set[str]]:
    anchors: Set[str] = set()
    members: Set[str] = set()
    for r in rows:
        anchors.add(r["anchor_id"])
        members.update(r.get("weak_evidence", []))
        members.update(r.get("no_interaction", []))
    return anchors, members


def _undirected(records: Iterable[dict]) -> Set[Tuple[str, str]]:
    out = set()
    for r in records:
        a, b = _record_pids(r)
        out.add((a, b) if a <= b else (b, a))
    return out


def parse_cond_id(cond_id: str) -> Tuple[str, K]:
    """``"all"`` -> ("all", "all"); ``"p03_k5"`` -> ("p03", 5)."""
    if cond_id == "all":
        return "all", "all"
    pack, sep, k = cond_id.rpartition("_k")
    if not sep or not pack or not k.isdigit():
        raise RuntimeError(f"invalid cond_id {cond_id!r}")
    return pack, int(k)


def condition_ids(table: dict) -> List[str]:
    """``all`` then ``<pack>_k<k>`` for every table pack and k in ``KS``."""
    return ["all"] + [f"{p['pack_id']}_k{k}" for p in table["packs"] for k in wc.KS]


# ------------------------------------------------------------- withholding
def withheld_set(table: dict, pack_id: str, k: K) -> Dict[str, Any]:
    """``{"W": set, "R": {go_id: set}}``; ``k == "all"`` only for the ``all`` condition."""
    if k == "all":
        if pack_id != "all":
            raise RuntimeError(f"k='all' is only valid for cond 'all', got pack {pack_id!r}")
        return {"W": set(), "R": {}}
    if isinstance(k, bool) or not isinstance(k, int) or k not in wc.KS:
        raise RuntimeError(f"invalid k {k!r}; allowed {wc.KS} or 'all'")
    if pack_id not in {p["pack_id"] for p in table["packs"]}:
        raise RuntimeError(f"unknown pack id {pack_id!r}")
    targets = [t for t in table["targets"] if t["pack_id"] == pack_id]
    W: Set[str] = set()
    R: Dict[str, Set[str]] = {}
    for t in targets:
        order = list(t["retention_order"])
        if len(order) < k:
            _fail("retention order shorter than k", [t["go_id"]],
                  f"{len(order)} < {k}")
        R[t["go_id"]] = set(order[:k])
    for t in targets:
        W |= set(t["train_annotated_proteins"]) - R[t["go_id"]]
    retained = set().union(*R.values()) if R else set()
    if W & retained:
        _fail("W intersects the retained sets", W & retained)
    return {"W": W, "R": R}


def filter_train_records(records: Sequence[dict], W: Set[str]) -> Tuple[List[dict], List[dict]]:
    """Drop a record when its anchor OR its partner is withheld; base order kept."""
    kept, dropped = [], []
    for r in records:
        a, b = _record_pids(r)
        (dropped if a in W or b in W else kept).append(r)
    return kept, dropped


def filter_pools(pool_rows: Sequence[dict], W: Set[str]) -> Tuple[List[dict], Dict[str, Any]]:
    """Rows with an anchor in W removed; W removed from both member lists."""
    rows: List[dict] = []
    anchors_dropped = removed = total = 0
    for r in pool_rows:
        n = len(r.get("weak_evidence", [])) + len(r.get("no_interaction", []))
        total += n
        if r["anchor_id"] in W:
            anchors_dropped += 1
            continue
        new = dict(r)
        new["weak_evidence"] = [p for p in r.get("weak_evidence", []) if p not in W]
        new["no_interaction"] = [p for p in r.get("no_interaction", []) if p not in W]
        removed += n - len(new["weak_evidence"]) - len(new["no_interaction"])
        rows.append(new)
    return rows, {"pool_anchor_rows_dropped": anchors_dropped,
                  "pool_member_entries_removed": removed,
                  "pool_member_entries_total": total,
                  "pool_member_share_removed": removed / total if total else 0.0}


def drop_empty_pool_anchors(records: Sequence[dict], pools: Sequence[dict],
                            views: Set[str]) -> Tuple[List[dict], List[str]]:
    """Drop the records of anchors with no pool row or no member in ``views``."""
    usable = {r["anchor_id"] for r in pools
              if any(p in views for p in r.get("weak_evidence", []))
              or any(p in views for p in r.get("no_interaction", []))}
    kept, dropped = [], set()
    for r in records:
        a = _record_pids(r)[0]
        if a in usable:
            kept.append(r)
        else:
            dropped.add(a)
    return kept, sorted(dropped)


# ------------------------------------------------------------- base data
@dataclass
class BaseData:
    """Base (post-fix) split files, parsed once and shared across conditions."""
    train_path: Path
    pools_path: Path
    validation_path: Path
    train: List[dict]
    pools: List[dict]
    validation: List[dict]
    views: Set[str]
    empty_annotation: Set[str]          # proteins whose pruned go_uris is empty
    train_sha256: str
    pools_sha256: str
    validation_sha256: str


def load_base(base_splits: Path, converted: Optional[Path] = None) -> BaseData:
    base_splits = Path(base_splits)
    converted = Path(wc.CONVERTED if converted is None else converted)
    train_path = base_splits / "train.jsonl"
    pools_path = base_splits / "negative_pools.jsonl"
    validation_path = base_splits / "validation.jsonl"
    views: Set[str] = set()
    empty: Set[str] = set()
    for line in _read_lines(converted / "proteins.jsonl"):
        row = json.loads(line)
        views.add(row["protein_id"])
        if not row.get("go_uris"):
            empty.add(row["protein_id"])
    return BaseData(
        train_path=train_path, pools_path=pools_path, validation_path=validation_path,
        train=[json.loads(l) for l in _read_lines(train_path)],
        pools=[json.loads(l) for l in _read_lines(pools_path)],
        validation=[json.loads(l) for l in _read_lines(validation_path)],
        views=views, empty_annotation=empty,
        train_sha256=wc.sha256_file(train_path), pools_sha256=wc.sha256_file(pools_path),
        validation_sha256=wc.sha256_file(validation_path))


def _write_or_verify(out_dir: Path, files: Dict[str, bytes]) -> str:
    """Write ``files``; an existing non-empty dir must match exactly. Returns
    ``"written"`` or ``"unchanged"``."""
    out_dir = Path(out_dir)
    wc.assert_writable_target(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        present = sorted(p.name for p in out_dir.iterdir())
        if present != sorted(files):
            _fail("idempotent regeneration mismatch", [], f"{out_dir}: files {present} "
                  f"!= {sorted(files)}")
        for name, data in files.items():
            if (out_dir / name).read_bytes() != data:
                _fail("idempotent regeneration mismatch", [], f"{out_dir / name} differs")
        return "unchanged"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        tmp = out_dir / f".{name}.tmp"
        tmp.write_bytes(data)
        tmp.replace(out_dir / name)
    return "written"


def targetfree_V(table: dict) -> Set[str]:
    """V = U_t P_val(t) over every selected target."""
    out: Set[str] = set()
    for t in table["targets"]:
        out |= set(t["validation_proteins"])
    return out


def table_sha256(table_path: Path) -> str:
    return wc.sha256_file(Path(table_path))


# ---------------------------------------------------- target-free validation
def build_targetfree_validation(table: dict, base_splits: Path, out_dir: Path, *,
                                base: Optional[BaseData] = None,
                                table_sha: Optional[str] = None) -> Dict[str, Any]:
    """Design 3.2: validation records/pools with every V protein removed (any role)."""
    base = base or load_base(base_splits)
    V = targetfree_V(table)
    kept = [r for r in base.validation if not (set(_record_pids(r)) & V)]
    positive = [r for r in kept if int(r["metadata"]["go_grade"]) == 2]
    val_rows = [r for r in base.pools if r["split"] == "validation"]
    rows, stats = filter_pools(val_rows, V)
    files = {"validation.jsonl": "".join(_dump(r) for r in kept).encode("utf-8"),
             "validation_positive.jsonl": "".join(_dump(r) for r in positive).encode("utf-8"),
             "negative_pools_validation.jsonl": "".join(_dump(r) for r in rows).encode("utf-8")}
    anchors, members = _pool_proteins(rows)
    leaked = (anchors | members) & V
    if leaked:
        _fail("V protein in a target-free validation pool row", leaked)
    for r in kept:
        if set(_record_pids(r)) & V:
            _fail("V protein in a target-free validation record", set(_record_pids(r)) & V)
    manifest = {"schema_version": 1, "V_size": len(V),
                "records_kept": len(kept), "records_dropped": len(base.validation) - len(kept),
                "positive_records": len(positive), "pool_rows": len(rows),
                "pool_rows_dropped": stats["pool_anchor_rows_dropped"],
                "pool_member_entries_removed": stats["pool_member_entries_removed"],
                "target_table_sha256": table_sha,
                "base_validation_sha256": base.validation_sha256,
                "base_pools_sha256": base.pools_sha256,
                "sha256s": {name: _sha(data) for name, data in sorted(files.items())}}
    files["manifest.json"] = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    manifest["status"] = _write_or_verify(Path(out_dir), files)
    manifest["manifest_sha256"] = _sha(files["manifest.json"])
    return manifest


# ----------------------------------------------------------------- conditions
def _condition_files(table: dict, cond_id: str, base: BaseData, val_pool_bytes: bytes,
                     table_sha: Optional[str], val_pool_sha: str
                     ) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    pack_id, k = parse_cond_id(cond_id)
    ws = withheld_set(table, pack_id, k)
    W, R = ws["W"], ws["R"]
    train_rows = [r for r in base.pools if r["split"] == "train"]
    kept, dropped_w = filter_train_records(base.train, W)
    pools, pstats = filter_pools(train_rows, W)
    kept, empty_anchors = drop_empty_pool_anchors(kept, pools, base.views)
    train_bytes = "".join(_dump(r) for r in kept).encode("utf-8")
    pool_bytes = "".join(_dump(r) for r in pools).encode("utf-8") + val_pool_bytes
    base_pairs = _undirected(base.train)
    new_pairs = _undirected(kept)
    removed_pairs = len(base_pairs - new_pairs)

    targets = []
    for t in (x for x in table["targets"] if x["pack_id"] == pack_id):
        rk = R[t["go_id"]]
        P = set(t["train_proteins"])
        T = set(t["train_annotated_proteins"])
        targets.append({
            "go_id": t["go_id"], "retained": [p for p in t["retention_order"] if p in rk],
            "retained_count": len(rk), "withheld_positive": len(P - rk),
            "withheld_pool_only": len(T - P),
            "retained_positive_pairs_undirected": sum(1 for a, b in new_pairs
                                                      if a in rk or b in rk)})
    pool_anchors = {r["anchor_id"] for r in pools}
    record_anchors = {_record_pids(r)[0] for r in kept}
    un_pool = pool_anchors & base.empty_annotation
    un_rec = record_anchors & base.empty_annotation
    manifest = {
        "schema_version": 2, "cond_id": cond_id, "pack_id": pack_id, "k": k,
        "_k_type": "all" if k == "all" else "int",
        "target_table_sha256": table_sha, "base_train_sha256": base.train_sha256,
        "base_pools_sha256": base.pools_sha256,
        "targetfree_validation_pools_sha256": val_pool_sha,
        "retention_seed": table.get("params", {}).get("retention_seed", wc.RETENTION_SEED),
        "targets": targets,
        "withheld_proteins": sorted(W), "withheld_count": len(W),
        "removal": {
            "train_records_dropped": len(base.train) - len(kept),
            "train_records_dropped_withheld": len(dropped_w),
            "train_records_kept": len(kept),
            "undirected_pairs_removed": removed_pairs,
            "removal_frac": removed_pairs / len(base_pairs) if base_pairs else 0.0,
            "pool_anchor_rows_dropped": pstats["pool_anchor_rows_dropped"],
            "pool_member_entries_removed": pstats["pool_member_entries_removed"],
            "pool_member_share_removed": pstats["pool_member_share_removed"],
            "anchors_dropped_empty_pool": empty_anchors},
        "unannotated_anchors": {
            "pool_anchors": len(un_pool), "record_anchors": len(un_rec),
            "records": sum(1 for r in kept if _record_pids(r)[0] in base.empty_annotation),
            "ids": sorted(un_pool | un_rec)},
        "files": {"train.jsonl": {"sha256": _sha(train_bytes), "lines": len(kept)},
                  "negative_pools.jsonl": {"sha256": _sha(pool_bytes),
                                           "lines": len(pools) + val_pool_bytes.count(b"\n")}},
    }
    files = {"train.jsonl": train_bytes, "negative_pools.jsonl": pool_bytes,
             MANIFEST: (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")}
    return files, manifest


def check_condition_files(table: dict, manifest: dict, train_bytes: bytes,
                          pool_bytes: bytes, base: BaseData, V: Set[str]) -> Dict[str, Any]:
    """The design 3.4 checks on a condition's (written or in-memory) bytes."""
    pack_id, k = manifest["pack_id"], manifest["k"]
    ws = withheld_set(table, pack_id, k)
    W, R = ws["W"], ws["R"]
    records = [json.loads(l) for l in train_bytes.decode("utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in pool_bytes.decode("utf-8").splitlines() if l.strip()]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    other = [r for r in rows if r["split"] not in ("train", "validation")]
    if other:
        _fail("negative_pools.jsonl holds non-train/validation rows",
              [r["anchor_id"] for r in other])
    rec_proteins: Set[str] = set()
    for r in records:
        rec_proteins.update(_record_pids(r))
    if rec_proteins & W:
        _fail("residual W protein in train.jsonl", rec_proteins & W)
    ta, tm = _pool_proteins(train_rows)
    if (ta | tm) & W:
        _fail("residual W protein in a train pool row", (ta | tm) & W)
    va, vm = _pool_proteins(val_rows)
    if (va | vm) & V:
        _fail("V protein in a validation pool row", (va | vm) & V)
    if k != "all":
        cap = table.get("params", {}).get("pack_removal_cap", 0.08)
        if k == 0 and manifest["removal"]["removal_frac"] > cap:
            _fail("k=0 removal_frac above cap", [pack_id],
                  f"{manifest['removal']['removal_frac']:.4f} > {cap}")
        base_a, base_m = _pool_proteins(r for r in base.pools if r["split"] == "train")
        for t in (x for x in table["targets"] if x["pack_id"] == pack_id):
            rk = R[t["go_id"]]
            got = rec_proteins & set(t["train_proteins"])
            if got != rk or len(got) != k:
                _fail("not exactly k retained", got ^ rk,
                      f"target {t['go_id']}: {len(got)} present, k={k}")
            was = rk & (base_a | base_m)
            missing = was - (ta | tm)
            if missing:
                _fail("retained protein dropped from the pools", missing,
                      f"target {t['go_id']}")
    return {"train_records": len(records), "train_pool_rows": len(train_rows),
            "validation_pool_rows": len(val_rows)}


def build_condition(table: dict, cond_id: str, base_splits: Path, targetfree_val: Path,
                    out_dir: Path, *, base: Optional[BaseData] = None,
                    table_sha: Optional[str] = None, index=None) -> Dict[str, Any]:
    """Design 3.3: write (or verify) one condition dir; returns its manifest.

    ``index`` (the pruned ``GoIndex``), when given, enforces that
    ``unannotated_anchors.ids`` equals the written anchors with empty pruned
    ``terms_for`` (design 15).
    """
    base = base or load_base(base_splits)
    val_pool_path = Path(targetfree_val) / "negative_pools_validation.jsonl"
    val_pool_bytes = val_pool_path.read_bytes()
    files, manifest = _condition_files(table, cond_id, base, val_pool_bytes, table_sha,
                                       _sha(val_pool_bytes))
    V = targetfree_V(table)
    check_condition_files(table, manifest, files["train.jsonl"],
                          files["negative_pools.jsonl"], base, V)
    if index is not None:
        rows = [json.loads(l) for l in files["negative_pools.jsonl"].decode().splitlines()]
        anchors = {r["anchor_id"] for r in rows if r["split"] == "train"}
        anchors |= {_record_pids(json.loads(l))[0]
                    for l in files["train.jsonl"].decode().splitlines() if l.strip()}
        expect = sorted(a for a in anchors if not index.terms_for(a))
        if expect != manifest["unannotated_anchors"]["ids"]:
            _fail("unannotated_anchors.ids disagree with the pruned index",
                  set(expect) ^ set(manifest["unannotated_anchors"]["ids"]))
    status = _write_or_verify(Path(out_dir), files)
    # Residual scan after writing (reads the files back).
    out_dir = Path(out_dir)
    check_condition_files(table, manifest, (out_dir / "train.jsonl").read_bytes(),
                          (out_dir / "negative_pools.jsonl").read_bytes(), base, V)
    manifest = dict(manifest)
    manifest["status"] = status
    manifest["manifest_sha256"] = _sha(files[MANIFEST])
    return manifest


# ------------------------------------------------------- run-time assertions
def assert_condition(trainer, config, manifest_path, train_file, cell: str,
                     frozen: Mapping[str, Any]) -> Dict[str, Any]:
    """The ten design 4.3 condition checks; each failure raises ``RuntimeError``."""
    from go_ppi_domain.withholding_diagnostics import (
        EpochDiagnostics, SelectionRecorder, StrictEpochGuard)

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cond_dir = manifest_path.parent
    table_path = Path(wc.TARGETS) / "target_table.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if wc.sha256_file(table_path) != manifest["target_table_sha256"]:
        _fail("check 1: target table sha256 differs from the manifest", [str(table_path)])
    pack_id, k = manifest["pack_id"], manifest["k"]
    W = set(manifest["withheld_proteins"])
    if W != withheld_set(table, pack_id, k)["W"]:
        _fail("check 1: manifest withheld set differs from the target table",
              W ^ withheld_set(table, pack_id, k)["W"])
    V = targetfree_V(table)
    out: Dict[str, Any] = {}

    # 1. hashes and directories
    tf = Path(train_file)
    if wc.sha256_file(tf) != manifest["files"]["train.jsonl"]["sha256"]:
        _fail("check 1: train file sha256 differs from the manifest", [str(tf)])
    split_dir = Path(getattr(config, "go_ppi_split_dir", ""))
    pools_path = split_dir / "negative_pools.jsonl"
    if not pools_path.is_file() or (wc.sha256_file(pools_path)
                                    != manifest["files"]["negative_pools.jsonl"]["sha256"]):
        _fail("check 1: negative_pools.jsonl sha256 differs from the manifest", [str(pools_path)])
    if split_dir.resolve() != cond_dir.resolve():
        _fail("check 1: go_ppi_split_dir is not the manifest's directory", [str(split_dir)])
    conv = Path(getattr(config, "go_ppi_converted_dir", ""))
    if conv.resolve() != Path(wc.CONVERTED).resolve():
        _fail("check 1: go_ppi_converted_dir is not the study converted dir", [str(conv)])
    out["1_hashes"] = "ok"

    # 2. train file free of W
    seen: Set[str] = set()
    with open(tf, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                seen.update(_record_pids(json.loads(line)))
    if seen & W:
        _fail("check 2: withheld protein in the train file", seen & W)
    out["2_train_free_of_W"] = "ok"

    # 3. selector free of W
    bp = trainer.batch_processor
    selector = getattr(bp, "domain_negative_selector", None)
    if selector is None:
        _fail("check 3: no domain negative selector attached")
    pool_members: Set[str] = set()
    for pool in selector.pools.values():
        pool_members.update(pool.get("weak_evidence", []))
        pool_members.update(pool.get("no_interaction", []))
    for what, ids in (("pool anchors", set(selector.pools) & W),
                      ("pool members", pool_members & W),
                      ("views", set(selector.views) & W)):
        if ids:
            _fail(f"check 3: withheld protein in selector {what}", ids)
    out["3_selector_free_of_W"] = "ok"

    # 4. validation pools equal the target-free file
    vpath = Path(wc.VALIDATION_TF) / "negative_pools_validation.jsonl"
    if wc.sha256_file(vpath) != manifest["targetfree_validation_pools_sha256"]:
        _fail("check 4: target-free validation pool file sha256 differs", [str(vpath)])
    expected = {}
    for line in _read_lines(vpath):
        r = json.loads(line)
        expected[r["anchor_id"]] = (frozenset(r.get("weak_evidence", [])),
                                    frozenset(r.get("no_interaction", [])))
    bad = [a for a, (w, n) in expected.items()
           if a not in selector.pools
           or frozenset(selector.pools[a].get("weak_evidence", [])) != w
           or frozenset(selector.pools[a].get("no_interaction", [])) != n]
    if bad:
        _fail("check 4: selector validation pools differ from the target-free file", bad)
    train_anchors = {json.loads(l)["anchor_id"] for l in _read_lines(pools_path)
                     if json.loads(l)["split"] == "train"}
    extra = set(selector.pools) - train_anchors - set(expected)
    if extra:
        _fail("check 4: selector pools hold anchors outside the condition files", extra)
    va, vm = set(expected), set()
    for w, n in expected.values():
        vm |= w | n
    if (va | vm) & V:
        _fail("check 4: V protein in validation pools", (va | vm) & V)
    out["4_validation_pools"] = "ok"

    # 5. exactly k
    if k != "all":
        targets = {t["go_id"]: t for t in table["targets"]}
        for mt in manifest["targets"]:
            P = set(targets[mt["go_id"]]["train_proteins"])
            got = seen & P
            rk = set(mt["retained"])
            if got != rk or len(got) != int(k):
                _fail("check 5: not exactly k retained", got ^ rk, f"target {mt['go_id']}")
        out["5_exactly_k"] = "ok"
    else:
        out["5_exactly_k"] = "skipped (k=all)"

    # 6. frozen test file
    test_path = Path(wc.SPLITS) / "test.jsonl"
    if wc.sha256_file(test_path) != frozen["test_sha256"]:
        _fail("check 6: test.jsonl sha256 differs from frozen_hashes", [str(test_path)])
    out["6_test_hash"] = "ok"

    # 7. cell wiring
    weight = float(getattr(config, "go_ontology_loss_weight", 0.0))
    tiered = bool(getattr(config, "go_ppi_go_tiered_negatives", False))
    sampling = str(getattr(config, "go_ppi_tier_sampling", ""))
    table_obj = getattr(trainer.loss_engine, "go_relation_table", None)
    wiring = {
        "baseline": (False, "random_window", False, False),
        "loss_only": (False, "random_window", True, False),
        "neg_only": (True, "stochastic", False, True),
        "both": (True, "stochastic", True, True),
    }
    if cell not in wiring:
        _fail("check 7: unknown cell", [str(cell)])
    exp_tiered, exp_sampling, loss_on, matcher_on = wiring[cell]
    problems = []
    if tiered != exp_tiered:
        problems.append(f"go_ppi_go_tiered_negatives={tiered}")
    if sampling != exp_sampling:
        problems.append(f"go_ppi_tier_sampling={sampling}")
    if loss_on and not weight > 0:
        problems.append(f"go_ontology_loss_weight={weight} (need > 0)")
    if not loss_on and weight != 0.0:
        problems.append(f"go_ontology_loss_weight={weight} (need 0.0)")
    if getattr(selector, "unannotated_anchor_window", None) is not True:
        problems.append("selector.unannotated_anchor_window is not True")
    if getattr(config, "go_ppi_unannotated_anchor_window", None) is not True:
        problems.append("config.go_ppi_unannotated_anchor_window is not True")
    if (getattr(selector, "matcher", None) is not None) != matcher_on:
        problems.append(f"selector.matcher {'missing' if matcher_on else 'present'}")
    if (getattr(bp, "ontology_matcher_override", None) is not None) != matcher_on:
        problems.append(f"batch_processor.ontology_matcher_override "
                        f"{'missing' if matcher_on else 'present'}")
    if (table_obj is not None) != loss_on:
        problems.append(f"loss_engine.go_relation_table {'missing' if loss_on else 'present'}")
    if table_obj is not None:
        for attr in ("ontology_set_similarity", "terms_for_gene", "branch_distance"):
            if hasattr(table_obj, attr):
                problems.append(f"go_relation_table exposes {attr}")
    if problems:
        _fail(f"check 7: cell wiring for {cell}", [], "; ".join(problems))
    out["7_wiring"] = "ok"

    # 8. every GoIndex in use is leak-pruned
    metas = []
    matcher = getattr(selector, "matcher", None)
    if matcher is not None:
        metas.append(("selector.matcher.index", getattr(matcher.index, "meta", {})))
    if table_obj is not None:
        metas.append(("go_relation_table source", getattr(table_obj, "source_meta", {})))
    for name, meta in metas:
        if not (meta.get("leak_pruning") or {}).get("with_from_human_protein") is True:
            _fail("check 8: GoIndex not leak-pruned", [name])
    out["8_index_pruned"] = "ok"

    # 9. relation table protein set
    if table_obj is not None:
        ids = set(table_obj.ids)
        if ids - seen:
            _fail("check 9: relation table proteins outside the train records", ids - seen)
        if ids & W:
            _fail("check 9: withheld protein in the relation table", ids & W)
        out["9_relation_table"] = "ok"
    else:
        out["9_relation_table"] = "skipped (no table)"

    # 10. callbacks, in order
    cbs = list(getattr(trainer, "epoch_end_callbacks", ()) or ())

    def kind(cb) -> Optional[str]:
        if isinstance(getattr(cb, "__self__", None), SelectionRecorder):
            return "recorder"
        if isinstance(cb, EpochDiagnostics):
            return "diagnostics"
        if isinstance(cb, StrictEpochGuard):
            return "guard"
        return None

    kinds = [x for x in (kind(cb) for cb in cbs) if x is not None]
    if kinds != ["recorder", "diagnostics", "guard"]:
        _fail("check 10: epoch_end_callbacks", [], f"found {kinds}")
    out["10_callbacks"] = "ok"
    out.update({"cond_id": manifest["cond_id"], "withheld_count": len(W),
                "train_proteins": len(seen)})
    return out
