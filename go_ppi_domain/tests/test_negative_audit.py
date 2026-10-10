"""audit_negatives.py (design 7, 14) on the synthetic study layout."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from go_ppi_domain import withholding as wh
from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.tests import _study_fixtures as sf
from scripts.go_withholding import audit_negatives as an


@pytest.fixture()
def study(tmp_path, monkeypatch):
    s = sf.build_study(tmp_path / "study")
    sf.patch_study(monkeypatch, s)
    return s


def _argv(cond, *extra):
    return ["--cond", cond, "--epochs", "0", "7", "14", "--independent", *extra]


def _out(s, cond, cell, seed=13):
    return json.loads((s["results"] / "audits" / f"neg__{cond}__{cell}__s{seed}.json")
                      .read_text())


def _rewrite_pools(s, cond, members):
    """Doctor the condition's train pool rows so every negative is drawn from ``members``."""
    path = s["conditions"] / cond / "negative_pools.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for r in rows:
        if r["split"] == "train":
            r["weak_evidence"] = list(members)
            r["no_interaction"] = list(members)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_audit_seed_override_shares_and_identity(study):
    assert json.loads((study["config_dir"] / "baseline.json").read_text())["training_seed"] == 42
    assert an.main(_argv("p01_k5")) == 0
    outs = {c: _out(study, "p01_k5", c) for c in wc.CELLS}
    for cell, o in outs.items():
        assert o["seed"] == 13 and o["selector_training_seed"] == 13      # not the config's 42
        assert set(o["by_epoch"]) == {"0", "7", "14"}
        row = o["by_epoch"]["7"]
        assert row["n"] > 0 and row["share_withheld_negatives"] == 0.0
        assert abs(row["share_target_annotated_negatives"]
                   - row["share_withheld_negatives"] - row["share_retained_negatives"]) <= 1e-12
        assert set(row) >= {"grade_mix", "simgic_mean", "indep_fn_rate",
                            "share_target_neighbour_negatives", "share_target_neighbour_anchors",
                            "share_withheld_negatives", "share_target_annotated_negatives",
                            "share_retained_negatives"}
    assert outs["baseline"]["selection_sha256_offline"] == outs["loss_only"]["selection_sha256_offline"]
    assert outs["neg_only"]["selection_sha256_offline"] == outs["both"]["selection_sha256_offline"]
    assert outs["baseline"]["selection_sha256_offline"] != outs["neg_only"]["selection_sha256_offline"]


def test_build_cell_selector_uses_the_audit_seed(study):
    cfg = wc.load_study_config(study["config_dir"] / "neg_only.json")
    assert cfg.training_seed == 42
    cfg.go_ppi_split_dir = str(study["conditions"] / "all")
    sel = an.build_cell_selector(cfg, 13)
    assert sel.training_seed == 13 and sel.matcher is not None


def test_withheld_gate_trips_on_doctored_pool(study):
    w = study["manifests"]["p01_k0"]["withheld_proteins"][0]
    _rewrite_pools(study, "p01_k0", [w, "g01"])
    assert an.main(_argv("p01_k0", "--cells", "baseline")) == 2
    assert not (study["results"] / "audits" / "neg__p01_k0__baseline__s13.json").exists()


def test_retained_negative_only_raises_descriptive_shares(study):
    r = study["manifests"]["p01_k5"]["targets"][0]["retained"][0]
    _rewrite_pools(study, "p01_k5", [r, "g01"])
    assert an.main(_argv("p01_k5", "--cells", "baseline", "neg_only")) == 0
    for cell in ("baseline", "neg_only"):
        row = _out(study, "p01_k5", cell)["by_epoch"]["0"]
        assert row["share_withheld_negatives"] == 0.0
        assert row["share_retained_negatives"] > 0
        assert row["share_target_annotated_negatives"] == pytest.approx(
            row["share_retained_negatives"], abs=1e-12)


def test_all_condition_has_no_gate_and_uses_all_targets(study):
    assert an.main(_argv("all", "--cells", "baseline", "--anchors", "20")) == 0
    o = _out(study, "all", "baseline")
    assert o["anchors"] == 20
    row = o["by_epoch"]["0"]
    assert row["share_withheld_negatives"] == 0.0 and row["share_retained_negatives"] == 0.0
    assert row["share_target_annotated_negatives"] > 0                   # all-pack union


def test_offline_identity_mismatch_is_fatal(study, monkeypatch, caplog):
    real = an.build_cell_selector
    seen = []

    def skewed(cfg, seed):
        seen.append(seed)
        return real(cfg, seed + (len(seen) == 2))      # loss_only gets another seed
    monkeypatch.setattr(an, "build_cell_selector", skewed)
    with caplog.at_level("ERROR", logger="audit_negatives"):
        assert an.main(_argv("p01_k5", "--cells", "baseline", "loss_only")) == 2
    assert "offline identity baseline == loss_only fails" in caplog.text


def test_index_preflight_is_fatal(study):
    frozen = json.loads((study["splits"] / "frozen_hashes.json").read_text())
    frozen["go_index_P_sha256"] = "0" * 64
    (study["splits"] / "frozen_hashes.json").write_text(json.dumps(frozen))
    assert an.main(_argv("p01_k5", "--cells", "baseline")) == 2


def test_independent_pair_mapping(tmp_path):
    info = tmp_path / "info.txt.gz"
    with gzip.open(info, "wt") as fh:
        fh.write("#string_protein_id\tpreferred_name\n9606.E1\tAAA\n9606.E2\tBBB\n9606.E3\tCCC\n")
    aliases = tmp_path / "aliases.txt.gz"
    with gzip.open(aliases, "wt") as fh:
        fh.write("#string_protein_id\talias\tsource\n"
                 "9606.E1\tENSG01\tEnsembl_gene\n9606.E2\tENSG02\tEnsembl_gene\n"
                 "9606.E3\tENSG03\tEnsembl_HPA\n9606.E3\tENSG03\tEnsembl_gene Ensembl_HGNC\n")
    huri = tmp_path / "HuRI.tsv"
    huri.write_text("ENSG01\tENSG02\nENSG02\tENSG03.5\nENSG01\tENSG99\nENSG01\tENSG01\n")
    m = an.ensg_to_symbols(aliases, info)
    assert m == {"ENSG01": {"AAA"}, "ENSG02": {"BBB"}, "ENSG03": {"CCC"}}
    pairs = an.load_independent_pairs([huri], m)
    assert pairs == {frozenset(("AAA", "BBB")), frozenset(("BBB", "CCC"))}


def test_audit_cell_fn_rate_and_invariant():
    class Sel:
        def select_batch_negatives(self, sample, cands, k, epoch):
            a = sample.metadata["resume_id"]
            return [{"partner_id": "N1", "grade": 1}, {"partner_id": "R1", "grade": 0}], [0, 0]
    sets = {"nbr": {"N1"}, "annotated": {"R1", "W1"}, "withheld": {"W1"}, "retained": {"R1"}}
    res = an.audit_cell(Sel(), [("A", "P"), ("B", "P")], [0], 7, sets,
                        lambda a, b: 0.5, {frozenset(("A", "N1"))}, pack_condition=True)
    row = res["by_epoch"]["0"]
    assert row["n"] == 4 and row["grade_mix"] == {"1": 0.5, "0": 0.5}
    assert row["indep_fn_rate"] == {"1": 0.5, "0": 0.0}
    assert row["share_target_neighbour_negatives"] == 0.5
    assert row["share_retained_negatives"] == row["share_target_annotated_negatives"] == 0.5
    assert row["simgic_mean"]["all"] == 0.5
    sets["annotated"].add("N1")                           # breaks annotated = W + R
    with pytest.raises(an.AuditError, match="invariant"):
        an.audit_cell(Sel(), [("A", "P")], [0], 7, sets, lambda a, b: 0.0, set(), True)
