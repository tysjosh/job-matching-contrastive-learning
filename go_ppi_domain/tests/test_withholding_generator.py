"""Withholding generator, make_conditions and assert_condition (design 3, 4.3)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding as wh  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402
from go_ppi_domain.go_ontology import GoIndex  # noqa: E402
from scripts.go_withholding import make_conditions as mc  # noqa: E402

N_PER_TARGET = 27
HUBS = [f"h{k}" for k in range(5)]
GENERIC = ["g01", "g02", "g03", "g04"]


def _rec(a, b, grade=2):
    return {"resume": {"protein_id": a, "encoder_view": f"{a} — t", "skill_uris": [],
                       "coarse_uris": []},
            "job": {"partner_id": b, "encoder_view": f"{b} — t", "skill_uris": [],
                    "coarse_uris": [], "title": b, "grade": grade},
            "label": int(grade == 2),
            "metadata": {"resume_id": a, "protein_id": a, "partner_id": b,
                         "go_grade": grade}}


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_fixture(root: Path, n_targets: int = 6, removal_cap: float = 0.95) -> dict:
    """Base splits + converted dir + a hand-built schema-2 target table with
    singleton packs (amendment AM1: pack p{i+1:02d} holds target i only).

    Per target i: 27 positive proteins t{i}p{j} (records to hubs, every 9th also
    reversed hub -> p), a pool-only annotated anchor t{i}x0 and a pool-only
    annotated member t{i}x1. Extra anchors: e01 (pool only references a protein
    without a view -> always dropped), e02 (pool only holds t00p05).
    """
    root = Path(root)
    go = [f"GO:9{i:06d}" for i in range(n_targets)]
    pack_of = {}
    packs = [{"pack_id": f"p{j + 1:02d}", "targets": [t]} for j, t in enumerate(go)]
    for p in packs:
        for t in p["targets"]:
            pack_of[t] = p["pack_id"]

    train, pools = [], []
    direct = {}
    targets = []
    for i, t in enumerate(go):
        P = [f"t{i:02d}p{j:02d}" for j in range(N_PER_TARGET)]
        x0, x1 = f"t{i:02d}x0", f"t{i:02d}x1"
        for p in P + [x0, x1]:
            direct[p] = frozenset({t})
        for j, p in enumerate(P):
            hub = HUBS[(i + j) % len(HUBS)]
            train.append(_rec(p, hub))
            if j % 9 == 0:
                train.append(_rec(hub, p))
            pools.append({"anchor_id": p, "split": "train", "weak_evidence": [x1, "g01"],
                          "no_interaction": ["g02", "g03", P[(j + 1) % len(P)]]})
        pools.append({"anchor_id": x0, "split": "train", "weak_evidence": ["g01"],
                      "no_interaction": [x1]})
        order = sorted(P)
        random.Random(f"20251009|{t}").shuffle(order)
        targets.append({"go_id": t, "pack_id": pack_of[t], "train_proteins": sorted(P),
                        "train_annotated_proteins": sorted(P + [x0, x1]),
                        "retention_eligible": sorted(P), "retention_order": order,
                        "validation_proteins": [f"v{i % 10:02d}"]})
    for k, hub in enumerate(HUBS):
        pools.append({"anchor_id": hub, "split": "train",
                      "weak_evidence": ["g01", "t00p00", f"t{k:02d}x1"],
                      "no_interaction": ["g04"]})
    train.append(_rec("e01", "h0"))
    pools.append({"anchor_id": "e01", "split": "train", "weak_evidence": ["ghost"],
                  "no_interaction": []})
    train.append(_rec("e02", "h1"))
    pools.append({"anchor_id": "e02", "split": "train", "weak_evidence": ["t00p05"],
                  "no_interaction": []})
    val = []
    for k in range(10):
        a, b = f"v{k:02d}", f"v{(k + 1) % 10 + 10:02d}"
        val.append(_rec(a, b, 2))
        val.append(_rec(a, f"v{(k + 3) % 10 + 10:02d}", 1))
        pools.append({"anchor_id": a, "split": "validation",
                      "weak_evidence": [f"v{(k + 3) % 10 + 10:02d}", f"v{(k + 1) % 10:02d}"],
                      "no_interaction": [f"v{(k + 5) % 10 + 10:02d}"]})
    for k in range(10):
        direct[f"v{k:02d}"] = frozenset({go[k % n_targets]})

    splits, conv = root / "splits", root / "converted"
    _write_jsonl(splits / "train.jsonl", train)
    _write_jsonl(splits / "negative_pools.jsonl", pools)
    _write_jsonl(splits / "validation.jsonl", val)
    _write_jsonl(splits / "test.jsonl", [_rec("z1", "z2", 2)])
    proteins = set(direct) | set(HUBS) | set(GENERIC) | {"e01", "e02"} | \
        {f"v{k:02d}" for k in range(20)}
    _write_jsonl(conv / "proteins.jsonl", [
        {"protein_id": p, "gene": p, "encoder_view": f"{p} — t",
         "go_uris": sorted(direct.get(p, ())), "coarse_uris": [], "n_terms": 0}
        for p in sorted(proteins)])
    anc = {t: frozenset({t}) for t in go}
    index = GoIndex(anc, {t: 0 for t in go}, dict(direct), dict(direct), {}, {t: 1.0 for t in go},
                    {"aspect": "P", "leak_pruning": {"with_from_human_protein": True}})
    index.save(conv / "go_index_P.pkl")
    frozen = {"train_sha256": wc.sha256_file(splits / "train.jsonl"),
              "validation_sha256": wc.sha256_file(splits / "validation.jsonl"),
              "negative_pools_sha256": wc.sha256_file(splits / "negative_pools.jsonl"),
              "test_sha256": wc.sha256_file(splits / "test.jsonl")}
    (splits / "frozen_hashes.json").write_text(json.dumps(frozen), encoding="utf-8")
    table = {"schema_version": 2, "params": {"pack_size": 1, "pack_removal_cap": removal_cap,
                                             "retention_seed": 20251009},
             "n_targets": n_targets, "n_packs": len(packs),
             "n_conditions": 5 * len(packs) + 1, "packs": packs, "targets": targets}
    tdir = root / "targets"
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / "target_table.json"
    tpath.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (tdir / "target_table.json.sha256").write_text(wc.sha256_file(tpath) + "\n")
    return {"root": root, "splits": splits, "converted": conv, "table": table,
            "table_path": tpath, "index": index, "frozen": frozen}


@pytest.fixture()
def fx(tmp_path):
    f = make_fixture(tmp_path)
    f["base"] = wh.load_base(f["splits"], f["converted"])
    f["val_dir"] = tmp_path / "validation_targetfree"
    wh.build_targetfree_validation(f["table"], f["splits"], f["val_dir"], base=f["base"],
                                   table_sha=wc.sha256_file(f["table_path"]))
    return f


def _build(f, cond_id, out=None, **kw):
    out = out or (f["root"] / "conditions" / cond_id)
    m = wh.build_condition(f["table"], cond_id, f["splits"], f["val_dir"], out,
                           base=f["base"], table_sha=wc.sha256_file(f["table_path"]),
                           index=f["index"], **kw)
    records = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    pools = [json.loads(l) for l in (out / "negative_pools.jsonl").read_text().splitlines()]
    return m, records, pools, out


def _pool_proteins(pools, split="train"):
    a, m = set(), set()
    for r in pools:
        if r["split"] == split:
            a.add(r["anchor_id"])
            m.update(r["weak_evidence"])
            m.update(r["no_interaction"])
    return a, m


def _record_proteins(records):
    out = set()
    for r in records:
        out.update((r["resume"]["protein_id"], r["job"]["partner_id"]))
    return out


# ------------------------------------------------------------------ sets
def test_withheld_set_nested_and_typing(fx):
    table = fx["table"]
    assert wh.withheld_set(table, "all", "all") == {"W": set(), "R": {}}
    sets = {k: wh.withheld_set(table, "p01", k) for k in wc.KS}
    for t in table["packs"][0]["targets"]:
        r = [sets[k]["R"][t] for k in wc.KS]
        assert [len(x) for x in r] == list(wc.KS)
        assert r[1] <= r[2] <= r[3] <= r[4]                     # R1 c R5 c R10 c R25
    for k in wc.KS:
        retained = set().union(*sets[k]["R"].values())
        assert not sets[k]["W"] & retained
    assert len(sets[0]["W"]) == N_PER_TARGET + 2                  # one target per pack
    with pytest.raises(RuntimeError):
        wh.withheld_set(table, "p01", 3)
    with pytest.raises(RuntimeError):
        wh.withheld_set(table, "p01", "all")
    with pytest.raises(RuntimeError):
        wh.withheld_set(table, "p99", 0)
    short = json.loads(json.dumps(table))
    short["targets"][0]["retention_order"] = short["targets"][0]["retention_order"][:4]
    with pytest.raises(RuntimeError, match="shorter than k"):
        wh.withheld_set(short, "p01", 5)


def test_filter_functions():
    recs = [_rec("a", "b"), _rec("b", "a"), _rec("c", "d")]
    kept, dropped = wh.filter_train_records(recs, {"a"})
    assert kept == [recs[2]] and dropped == recs[:2]
    rows = [{"anchor_id": "a", "split": "train", "weak_evidence": ["x"], "no_interaction": []},
            {"anchor_id": "c", "split": "train", "weak_evidence": ["a", "x"],
             "no_interaction": ["a", "y"]}]
    out, stats = wh.filter_pools(rows, {"a"})
    assert out == [{"anchor_id": "c", "split": "train", "weak_evidence": ["x"],
                    "no_interaction": ["y"]}]
    assert stats["pool_anchor_rows_dropped"] == 1 and stats["pool_member_entries_removed"] == 2
    assert rows[1]["weak_evidence"] == ["a", "x"]                  # inputs not mutated
    kept, dropped = wh.drop_empty_pool_anchors(
        [_rec("c", "d"), _rec("e", "f")], out, views={"y"})
    assert [r["metadata"]["protein_id"] for r in kept] == ["c"] and dropped == ["e"]


# -------------------------------------------------------------- conditions
def test_both_directions_and_pool_only_removed_at_k0_and_k25(fx):
    for k in (0, 25):
        m, records, pools, _ = _build(fx, f"p01_k{k}")
        W = set(m["withheld_proteins"])
        assert not _record_proteins(records) & W
        anchors, members = _pool_proteins(pools)
        assert "t00x0" not in anchors | members                    # pool-only anchor
        assert "t00x1" not in anchors | members                    # pool-only member
        assert "t00x0" in W and "t00x1" in W
        assert not (anchors | members) & W
        # reversed records (hub -> withheld p) are dropped too
        assert not any(r["job"]["partner_id"] in W for r in records)
    # other targets' (packs') proteins untouched
    m, records, pools, _ = _build(fx, "p01_k0", out=fx["root"] / "again")
    assert "t01x0" in _pool_proteins(pools)[0]
    assert {"t01p00", "t05p00"} <= _record_proteins(records)
    assert {t["go_id"] for t in m["targets"]} == {fx["table"]["targets"][0]["go_id"]}


def test_exactly_k_and_retained_stay_in_pools(fx):
    base_a, base_m = _pool_proteins(fx["base"].pools)
    for k in wc.KS:
        m, records, pools, _ = _build(fx, f"p01_k{k}")
        present = _record_proteins(records)
        anchors, members = _pool_proteins(pools)
        assert len(m["targets"]) == 1
        for t, mt in zip(fx["table"]["targets"][:1], m["targets"]):
            assert mt["retained_count"] == k
            got = present & set(t["train_proteins"])
            assert got == set(t["retention_order"][:k]) == set(mt["retained"])
            assert mt["retained"] == t["retention_order"][:k]
            assert mt["withheld_positive"] == N_PER_TARGET - k
            assert mt["withheld_pool_only"] == 2
            if k == 5:
                was = set(mt["retained"]) & (base_a | base_m)
                assert was and was <= anchors | members
                assert not set(m["withheld_proteins"]) & (anchors | members)
            assert (mt["retained_positive_pairs_undirected"] > 0) == (k > 0)


def test_empty_pool_rule_and_removal_fraction(fx):
    m_all, rec_all, _, _ = _build(fx, "all")
    assert "e01" in m_all["removal"]["anchors_dropped_empty_pool"]
    assert "e01" not in {r["metadata"]["protein_id"] for r in rec_all}
    assert "e02" not in m_all["removal"]["anchors_dropped_empty_pool"]
    m0, rec0, _, _ = _build(fx, "p01_k0")
    assert {"e01", "e02"} <= set(m0["removal"]["anchors_dropped_empty_pool"])
    base_pairs = wh._undirected(fx["base"].train)
    new_pairs = wh._undirected(rec0)
    assert m0["removal"]["undirected_pairs_removed"] == len(base_pairs - new_pairs)
    assert m0["removal"]["removal_frac"] == pytest.approx(
        len(base_pairs - new_pairs) / len(base_pairs))
    assert m0["removal"]["train_records_kept"] == len(rec0)
    assert m0["removal"]["pool_anchor_rows_dropped"] == N_PER_TARGET + 1
    assert 0 < m0["removal"]["pool_member_share_removed"] < 1
    # unannotated anchors: hubs (records and pool rows); e01/e02 keep their pool
    # rows (their records are dropped), so they count as unannotated pool anchors.
    un = m0["unannotated_anchors"]
    assert un["ids"] == sorted(HUBS + ["e01", "e02"])
    assert un["pool_anchors"] == 7 and un["record_anchors"] == len(HUBS)
    assert un["records"] == sum(1 for r in rec0 if r["metadata"]["protein_id"] in HUBS)


def test_all_manifest_typing_and_bytes(fx):
    m, records, pools, out = _build(fx, "all")
    assert m["k"] == "all" and m["_k_type"] == "all" and m["pack_id"] == "all"
    assert m["targets"] == [] and m["withheld_proteins"] == [] and m["withheld_count"] == 0
    m5, *_ = _build(fx, "p02_k5")
    assert m5["k"] == 5 and m5["_k_type"] == "int"
    written = json.loads((out / wh.MANIFEST).read_text())
    assert written["k"] == "all"
    val_bytes = (fx["val_dir"] / "negative_pools_validation.jsonl").read_bytes()
    assert (out / "negative_pools.jsonl").read_bytes().endswith(val_bytes)
    # unchanged rows are byte-identical to the base lines (splitter format)
    base_lines = fx["splits"].joinpath("train.jsonl").read_text().splitlines(keepends=True)
    kept_lines = out.joinpath("train.jsonl").read_text().splitlines(keepends=True)
    assert set(kept_lines) <= set(base_lines)
    assert len(base_lines) - len(kept_lines) == 1                  # only e01 dropped


def test_idempotent_regeneration_and_mismatch(fx):
    m1, _, _, out = _build(fx, "p01_k1")
    assert m1["status"] == "written"
    m2, _, _, _ = _build(fx, "p01_k1")
    assert m2["status"] == "unchanged" and m2["manifest_sha256"] == m1["manifest_sha256"]
    (out / "train.jsonl").write_text((out / "train.jsonl").read_text() + "\n")
    with pytest.raises(RuntimeError, match="idempotent regeneration mismatch"):
        _build(fx, "p01_k1")


def test_check_condition_detects_violations(fx):
    m, _, _, out = _build(fx, "p01_k5")
    train = (out / "train.jsonl").read_bytes()
    pools = (out / "negative_pools.jsonl").read_bytes()
    V = wh.targetfree_V(fx["table"])
    wh.check_condition_files(fx["table"], m, train, pools, fx["base"], V)
    w = m["withheld_proteins"][0]
    bad = train + (json.dumps(_rec("h0", w)) + "\n").encode()
    with pytest.raises(RuntimeError, match="residual W protein in train.jsonl"):
        wh.check_condition_files(fx["table"], m, bad, pools, fx["base"], V)
    r = m["targets"][0]["retained"][0]
    lines = [l for l in train.decode().splitlines(keepends=True)
             if r not in (json.loads(l)["resume"]["protein_id"], json.loads(l)["job"]["partner_id"])]
    with pytest.raises(RuntimeError, match="not exactly k"):
        wh.check_condition_files(fx["table"], m, "".join(lines).encode(), pools, fx["base"], V)
    bad_pools = pools + (json.dumps({"anchor_id": "zz", "split": "train",
                                     "weak_evidence": [w], "no_interaction": []}) + "\n").encode()
    with pytest.raises(RuntimeError, match="residual W protein in a train pool row"):
        wh.check_condition_files(fx["table"], m, train, bad_pools, fx["base"], V)
    v = sorted(V)[0]
    bad_val = pools + (json.dumps({"anchor_id": v, "split": "validation",
                                   "weak_evidence": [], "no_interaction": []}) + "\n").encode()
    with pytest.raises(RuntimeError, match="V protein in a validation pool row"):
        wh.check_condition_files(fx["table"], m, train, bad_val, fx["base"], V)


def test_k0_removal_cap_and_unannotated_agreement(tmp_path):
    f = make_fixture(tmp_path, removal_cap=0.0001)
    base = wh.load_base(f["splits"], f["converted"])
    val = tmp_path / "vtf"
    wh.build_targetfree_validation(f["table"], f["splits"], val, base=base)
    with pytest.raises(RuntimeError, match="k=0 removal_frac above cap"):
        wh.build_condition(f["table"], "p01_k0", f["splits"], val, tmp_path / "c0", base=base)
    assert not (tmp_path / "c0").exists()
    wh.build_condition(f["table"], "p01_k1", f["splits"], val, tmp_path / "c1", base=base)
    wrong = GoIndex(f["index"].ancestors, f["index"].depth,
                    {**f["index"].direct, "h0": frozenset({"GO:9000000"})},
                    f["index"].closure, {}, f["index"].ic, f["index"].meta)
    with pytest.raises(RuntimeError, match="unannotated_anchors.ids disagree"):
        wh.build_condition(f["table"], "p01_k5", f["splits"], val, tmp_path / "c5",
                           base=base, index=wrong)


def test_targetfree_validation(fx):
    V = wh.targetfree_V(fx["table"])
    assert V == {f"v{k:02d}" for k in range(6)}
    recs = [json.loads(l) for l in (fx["val_dir"] / "validation.jsonl").read_text().splitlines()]
    assert recs and not _record_proteins(recs) & V
    pos = [json.loads(l) for l in
           (fx["val_dir"] / "validation_positive.jsonl").read_text().splitlines()]
    assert pos == [r for r in recs if r["metadata"]["go_grade"] == 2]
    rows = [json.loads(l) for l in
            (fx["val_dir"] / "negative_pools_validation.jsonl").read_text().splitlines()]
    a, m = _pool_proteins(rows, "validation")
    assert rows and not (a | m) & V
    man = json.loads((fx["val_dir"] / "manifest.json").read_text())
    assert man["V_size"] == 6 and man["records_kept"] == len(recs)
    assert man["positive_records"] == len(pos) and man["pool_rows"] == len(rows)


# ----------------------------------------------------------- make_conditions
# AM1: one target per condition -> 24 targets: 24 x 5 + 1 = 121 conditions;
# the 20-target size-rule shortfall: 20 x 5 + 1 = 101.
@pytest.mark.parametrize("n_targets, expected", [(24, 121), (20, 101)])
def test_make_conditions_counts(tmp_path, monkeypatch, n_targets, expected):
    f = make_fixture(tmp_path, n_targets=n_targets)
    assert f["table"]["n_conditions"] == expected and f["table"]["n_packs"] == n_targets
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    common = ["--target-table", str(f["table_path"]), "--base-splits", str(f["splits"]),
              "--converted", str(f["converted"])]
    val, cond = tmp_path / "vtf", tmp_path / "conditions"
    assert mc.main(["--mode", "validation", *common, "--out", str(val)]) == 2   # no G2-targets
    wc.append_pin("G2-targets", {"target_table_sha256": wc.sha256_file(f["table_path"])})
    assert mc.main(["--mode", "validation", *common, "--out", str(val)]) == 0
    args = ["--mode", "conditions", *common, "--targetfree-validation", str(val),
            "--out", str(cond)]
    assert mc.main(args) == 0
    dirs = sorted(p.name for p in cond.iterdir())
    assert len(dirs) == expected and dirs == sorted(wh.condition_ids(f["table"]))
    gates = [p["gate"] for p in wc.read_pins()]
    assert gates == ["G2-targets", "G2-validation", "G3"]
    g3 = wc.read_pins()[-1]
    assert g3["n_conditions"] == expected and len(g3["manifests"]) == expected
    # idempotent re-run: nothing rewritten, no duplicate pin
    assert mc.main(args) == 0
    assert [p["gate"] for p in wc.read_pins()] == gates
    # a disagreeing count is fatal
    bad = json.loads(f["table_path"].read_text())
    bad["n_conditions"] += 1
    f["table_path"].write_text(json.dumps(bad))
    (f["table_path"].parent / "target_table.json.sha256").write_text(
        wc.sha256_file(f["table_path"]))
    assert mc.main(args) == 2


def test_make_conditions_disk_guard(tmp_path, monkeypatch):
    f = make_fixture(tmp_path)
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    wc.append_pin("G2-validation", {})
    monkeypatch.setattr(mc, "MIN_FREE_BYTES", 10 ** 18)
    base = wh.load_base(f["splits"], f["converted"])
    val = tmp_path / "vtf"
    wh.build_targetfree_validation(f["table"], f["splits"], val, base=base,
                                   table_sha=wc.sha256_file(f["table_path"]))
    assert mc.main(["--mode", "conditions", "--target-table", str(f["table_path"]),
                    "--base-splits", str(f["splits"]), "--converted", str(f["converted"]),
                    "--targetfree-validation", str(val),
                    "--out", str(tmp_path / "c")]) == 2
    assert not (tmp_path / "c").exists()


# ----------------------------------------------------------- assert_condition
def _stub_trainer(f, cond_dir, cell="baseline"):
    from go_ppi_domain.negative_selector import GoPpiNegativeSelector
    from go_ppi_domain.withholding_diagnostics import (
        EpochDiagnostics, SelectionRecorder, StrictEpochGuard)

    kw = dict(tier_sampling="random_window", unannotated_anchor_window=True)
    sel = GoPpiNegativeSelector.from_split_dir(cond_dir, f["converted"], "train", **kw)
    vsel = GoPpiNegativeSelector.from_split_dir(cond_dir, f["converted"], "validation", **kw)
    sel.pools.update(vsel.pools)
    sel.views.update(vsel.views)
    rec = SelectionRecorder(sel)
    diag = EpochDiagnostics(cond_dir / "diag.jsonl", cond_dir / "train.jsonl", rec)
    guard = StrictEpochGuard(cell)
    trainer = SimpleNamespace(
        batch_processor=SimpleNamespace(domain_negative_selector=sel,
                                        ontology_matcher_override=None),
        loss_engine=SimpleNamespace(go_relation_table=None),
        epoch_end_callbacks=[rec.on_epoch_end, diag, guard])
    config = SimpleNamespace(go_ppi_split_dir=str(cond_dir),
                             go_ppi_converted_dir=str(f["converted"]),
                             go_ontology_loss_weight=0.0, go_ppi_go_tiered_negatives=False,
                             go_ppi_tier_sampling="random_window",
                             go_ppi_unannotated_anchor_window=True)
    return trainer, config


def test_assert_condition_happy_path_and_violations(fx, monkeypatch):
    monkeypatch.setattr(wc, "TARGETS", fx["table_path"].parent)
    monkeypatch.setattr(wc, "VALIDATION_TF", fx["val_dir"])
    monkeypatch.setattr(wc, "CONVERTED", fx["converted"])
    monkeypatch.setattr(wc, "SPLITS", fx["splits"])
    m, _, _, out = _build(fx, "p01_k5")
    trainer, config = _stub_trainer(fx, out)
    res = wh.assert_condition(trainer, config, out / wh.MANIFEST, out / "train.jsonl",
                              "baseline", fx["frozen"])
    assert res["5_exactly_k"] == "ok" and res["10_callbacks"] == "ok"
    assert res["9_relation_table"].startswith("skipped")

    # check 2: a withheld protein in the train file (sha check comes first -> doctor both)
    with pytest.raises(RuntimeError, match="check 1"):
        wh.assert_condition(trainer, config, out / wh.MANIFEST, fx["splits"] / "train.jsonl",
                            "baseline", fx["frozen"])
    # check 7: wiring
    with pytest.raises(RuntimeError, match="check 7"):
        wh.assert_condition(trainer, config, out / wh.MANIFEST, out / "train.jsonl",
                            "loss_only", fx["frozen"])
    # check 3: a withheld protein reaches the selector views
    trainer.batch_processor.domain_negative_selector.views[m["withheld_proteins"][0]] = {}
    with pytest.raises(RuntimeError, match="check 3"):
        wh.assert_condition(trainer, config, out / wh.MANIFEST, out / "train.jsonl",
                            "baseline", fx["frozen"])
    del trainer.batch_processor.domain_negative_selector.views[m["withheld_proteins"][0]]
    # check 10: callback order
    trainer.epoch_end_callbacks = list(reversed(trainer.epoch_end_callbacks))
    with pytest.raises(RuntimeError, match="check 10"):
        wh.assert_condition(trainer, config, out / wh.MANIFEST, out / "train.jsonl",
                            "baseline", fx["frozen"])
    trainer.epoch_end_callbacks.reverse()
    # check 6: frozen test hash
    with pytest.raises(RuntimeError, match="check 6"):
        wh.assert_condition(trainer, config, out / wh.MANIFEST, out / "train.jsonl",
                            "baseline", {"test_sha256": "0" * 64})
    # check 2 proper: doctor the train file and the manifest hash together
    doctored = out.parent / "doctored"
    doctored.mkdir()
    for name in ("negative_pools.jsonl",):
        (doctored / name).write_bytes((out / name).read_bytes())
    w = m["withheld_proteins"][0]
    data = (out / "train.jsonl").read_bytes() + (json.dumps(_rec("h0", w)) + "\n").encode()
    (doctored / "train.jsonl").write_bytes(data)
    man = json.loads((out / wh.MANIFEST).read_text())
    man["files"]["train.jsonl"]["sha256"] = wc.sha256_file(doctored / "train.jsonl")
    (doctored / wh.MANIFEST).write_text(json.dumps(man))
    config.go_ppi_split_dir = str(doctored)
    with pytest.raises(RuntimeError, match="check 2"):
        wh.assert_condition(trainer, config, doctored / wh.MANIFEST, doctored / "train.jsonl",
                            "baseline", fx["frozen"])
