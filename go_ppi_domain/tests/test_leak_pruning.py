"""Leak pruning: additive GAF filters, GoIndex.build options and text masking (design 1.1, 1.2)."""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go_ppi_domain import go_ontology  # noqa: E402
from go_ppi_domain import leak_pruning as lp  # noqa: E402
from go_ppi_domain.go_ontology import GoIndex, obo_descendants, parse_gaf, parse_obo  # noqa: E402
from go_ppi_domain.tests import _leak_fixtures as fx  # noqa: E402

PRE_EDIT = (ROOT / ".agents/tasks/ontology-withholding-feasibility/"
            "task-go-ppi-withholding/pre_edit/go_ontology.py")


def _pre_edit_module():
    if not PRE_EDIT.exists():
        pytest.skip("pre-edit go_ontology.py snapshot not available")
    spec = importlib.util.spec_from_file_location("_pre_edit_go_ontology", PRE_EDIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _filter_gaf(tmp_path: Path) -> Path:
    L = fx.gaf_line
    return fx.write_gaf(tmp_path / "f.gaf.gz", [
        L("AAA", "P00001", "GO:0000001"),                                  # kept
        L("AAA", "P00001", "GO:0000002", "IBA", "UniProtKB:P00002"),       # names BBB -> drop
        L("AAA", "P00001", "GO:0000003", "ISS", "UniProtKB:P00001-3"),     # self (isoform) -> keep
        L("AAA", "P00001", "GO:0000004", "IGI", "MGI:MGI:1|UniProtKB:P00002-2"),  # drop
        L("AAA", "P00001", "GO:0000005", "IMP", "UniProtKB:Q99999"),       # unmapped -> keep
        L("BBB", "P00002", "GO:0000005", "IDA", "ComplexPortal:CPX-1"),    # keep
        L("BBB", "P00002", "GO:0000002", "IDA", "UniProtKB:P00001,UniProtKB:P00002"),  # drop
        L("BBB", "P00002", "GO:0000001", "IPI", "UniProtKB:P00001"),       # IPI guard first
        L("BBB", "P00002", "GO:0000001", "IDA", qualifier="NOT|involved_in"),
        L("CCC", "P00003", "GO:0000010", "IDA", aspect="F"),               # MF root descendant
        L("CCC", "P00003", "GO:0005515", "IDA", aspect="F"),               # excluded term
        L("CCC", "P00003", "GO:0000011", "IEA", aspect="F"),
        L("CCC", "P00003", "GO:0000001", "IDA", taxon="taxon:10090"),
    ])


SYMBOL_MAP = {"P00001": "AAA", "P00002": "BBB", "P00003": "CCC"}


# ------------------------------------------------------------------ parse_gaf
def test_parse_gaf_defaults_identical_to_pre_edit(tmp_path):
    old = _pre_edit_module()
    gaf = _filter_gaf(tmp_path)
    for aspect in ("P", "F", "A"):
        assert parse_gaf(gaf, aspect) == old.parse_gaf(gaf, aspect)
    big = fx.write_bulk_dir(tmp_path / "bulk")["gaf"]
    assert parse_gaf(big, "P") == old.parse_gaf(big, "P")


def test_goindex_build_defaults_identical_to_pre_edit(tmp_path):
    old = _pre_edit_module()
    bulk = fx.write_bulk_dir(tmp_path / "bulk")
    new_idx = GoIndex.build(bulk["obo"], bulk["gaf"], "P")
    old_idx = old.GoIndex.build(bulk["obo"], bulk["gaf"], "P")
    assert "leak_pruning" not in new_idx.meta
    for attr in ("ancestors", "depth", "direct", "closure", "coarse", "ic", "meta"):
        assert getattr(new_idx, attr) == getattr(old_idx, attr), attr
    new_idx.save(tmp_path / "a.pkl")
    old_idx.save(tmp_path / "b.pkl")
    assert pickle.loads((tmp_path / "a.pkl").read_bytes()) == \
        pickle.loads((tmp_path / "b.pkl").read_bytes())


def test_parse_gaf_with_from_filter_keeps_self_reference(tmp_path):
    gaf = _filter_gaf(tmp_path)
    direct, stats = parse_gaf(gaf, "P", with_from_symbol_map=SYMBOL_MAP,
                              drop_with_from_human_protein=True)
    assert direct["AAA"] == {"GO:0000001", "GO:0000003", "GO:0000005"}
    assert direct["BBB"] == {"GO:0000005"}
    assert stats["drop_with_from_human_protein"] == 3
    base, base_stats = parse_gaf(gaf, "P")
    assert base_stats["kept"] - stats["kept"] == 3
    assert {k: v for k, v in stats.items() if k not in ("kept", "drop_with_from_human_protein")} \
        == {k: v for k, v in base_stats.items() if k != "kept"}


def test_parse_gaf_excluded_root_descendants(tmp_path):
    gaf = _filter_gaf(tmp_path)
    parents = parse_obo(fx.write_obo(tmp_path / "go.obo"))[0]
    assert obo_descendants(parents, "GO:0005515") == {"GO:0000010"}
    direct, stats = parse_gaf(gaf, "F", excluded_term_roots=["GO:0005515"],
                              descendants_of=lambda r: obo_descendants(parents, r))
    assert "CCC" not in direct
    assert stats["drop_excluded_root_descendant"] == 1
    # BP: generic implementation, GO:0000001 subtree (GO:0000002) dropped.
    direct, stats = parse_gaf(gaf, "P", excluded_term_roots=["GO:0000001"],
                              descendants_of=lambda r: obo_descendants(parents, r))
    assert stats["drop_excluded_root_descendant"] == 3
    assert "GO:0000002" not in direct.get("BBB", set())


def test_parse_gaf_option_errors(tmp_path):
    gaf = _filter_gaf(tmp_path)
    with pytest.raises(ValueError):
        parse_gaf(gaf, "P", drop_with_from_human_protein=True)
    with pytest.raises(ValueError):
        parse_gaf(gaf, "P", excluded_term_roots=["GO:0005515"])


def test_goindex_build_leak_pruning(tmp_path):
    bulk = fx.write_bulk_dir(tmp_path / "bulk")
    with pytest.raises(ValueError, match="leak_pruning requires with_from_symbol_map"):
        GoIndex.build(bulk["obo"], bulk["gaf"], "P",
                      leak_pruning={"with_from_human_protein": True,
                                    "excluded_term_roots": ["GO:0005515"]})
    acc = lp.human_accession_map(bulk["gaf"], bulk["aliases"], bulk["info"])
    pruned = GoIndex.build(bulk["obo"], bulk["gaf"], "P",
                           leak_pruning={"with_from_human_protein": True,
                                         "excluded_term_roots": ["GO:0005515"]},
                           with_from_symbol_map=acc)
    meta = pruned.meta["leak_pruning"]
    assert meta["with_from_human_protein"] is True
    assert meta["excluded_term_roots"] == ["GO:0005515"]
    assert meta["accession_map_size"] == len(acc)
    assert meta["gaf_stats"]["drop_with_from_human_protein"] == 15     # every 4th of 60
    assert meta["gaf_stats"]["drop_excluded_root_descendant"] == 0     # no-op for BP
    assert pruned.meta["gaf_stats"] == meta["gaf_stats"]
    assert pruned.closure_for(fx.sym_of(0)) != frozenset()


def test_regulates_and_obsolete_never_enter(tmp_path):
    parents, ns, _alt, _name = parse_obo(fx.write_obo(tmp_path / "go.obo"))
    assert parents["GO:0000003"] == {"GO:0008150"}
    assert "GO:0000099" not in ns


# ------------------------------------------------------------------ masking
def _lexicon(texts, aliases, **kw):
    return lp.build_mask_lexicon(texts, aliases, **kw)


def test_self_symbol_kept_partner_masked():
    aliases = {"TP53": {"TP53", "P53"}, "MDM2": {"MDM2", "HDM2"}}
    texts = {"TP53": "x", "MDM2": "y"}
    lex = _lexicon(texts, aliases, max_doc_freq=1.0)
    out, n = lp.mask_view_text("TP53", "TP53 (P53) is degraded by MDM2 and HDM2.", lex)
    assert out == "TP53 (P53) is degraded by [GENE] and [GENE]."
    assert n == 2


def test_whole_word_boundaries():
    lex = _lexicon({"TP53": "", "ABC": ""}, {"TP53": {"TP53"}, "ABC": {"ABC", "HLA-A"}},
                   max_doc_freq=1.0)
    out, n = lp.mask_view_text("ABC", "TP53-binding, TP530, xTP53, TP53.", lex)
    assert out == "[GENE]-binding, TP530, xTP53, [GENE]."
    assert n == 2
    out, _ = lp.mask_view_text("TP53", "binds HLA-A and HLA-AB; HLA-A-like", lex)
    assert out == "binds [GENE] and HLA-AB; [GENE]-like"


def test_case_sensitive_and_alias_filters():
    aliases = {"CAT": {"CAT", "cat", "AB", "Lo2"}, "DOG": {"DOG"}}
    lex = _lexicon({"CAT": "", "DOG": ""}, aliases, max_doc_freq=1.0)
    assert "cat" not in lex.owner          # needs an uppercase letter or digit
    assert "AB" not in lex.owner           # alias shorter than 3
    assert "Lo2" in lex.owner
    out, _ = lp.mask_view_text("DOG", "the cat sat on CAT and Lo2", lex)
    assert out == "the cat sat on [GENE] and [GENE]"


def test_doc_frequency_filter():
    texts = {f"G{i}": ("common ALIASX here" if i < 5 else "nothing") for i in range(100)}
    aliases = {f"G{i}": {f"G{i}"} for i in range(100)}
    aliases["G0"] |= {"ALIASX", "RARE1"}
    lex = _lexicon(texts, aliases, max_doc_freq=0.01)
    assert "ALIASX" not in lex.owner                  # 5% of texts
    assert "RARE1" in lex.owner                       # 0%
    assert lex.params["aliases_dropped_docfreq"] == ["ALIASX"]
    assert "G1" in lex.owner                          # primary symbols always eligible


def test_lexicon_json_roundtrip():
    lex = _lexicon({"AAA": "", "BBB": ""}, {"AAA": {"AAA", "A-1"}, "BBB": {"BBB"}},
                   max_doc_freq=1.0)
    back = lp.lexicon_from_json(json.loads(json.dumps(lp.lexicon_to_json(lex))))
    assert back.owner == lex.owner and back.alnum == lex.alnum and back.special == lex.special
    assert lp.mask_view_text("BBB", "AAA and A-1", back) == ("[GENE] and [GENE]", 2)


def test_alias_and_accession_maps(tmp_path):
    bulk = fx.write_bulk_dir(tmp_path / "bulk", n=8)
    al = lp.load_symbol_aliases(bulk["aliases"], bulk["info"], {"PRT00", "PRT01"})
    assert al["PRT00"] == {"PRT00", "ALT00", "CDS"}
    acc = lp.human_accession_map(bulk["gaf"], bulk["aliases"], bulk["info"])
    assert acc[fx.acc_of(3)] == "PRT03"
    other = lp.other_db_symbol_map(bulk["aliases"], bulk["info"])
    assert other["HGNC:1002"] == "PRT02"
    assert other[f"ENSEMBL_GeneID:ENSG{5:011d}"] == "PRT05"


# ------------------------------------------------------------------ rewrite
def _ref_proteins(path: Path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_rewrite_proteins_preserves_ids_order_prefix(tmp_path):
    from go_ppi_domain.tests._withholding_fixtures import build_index

    index = build_index({"BBB": ["T5"], "AAA": ["T2", "T6"]})
    rows = [
        {"protein_id": "BBB", "gene": "BBB", "encoder_view": "BBB — binds AAA strongly",
         "go_uris": ["OLD"], "coarse_uris": ["OLD"], "n_terms": 9},
        {"protein_id": "AAA", "gene": "AAA", "encoder_view": "AAA — partner of BBB; AAA itself",
         "go_uris": [], "coarse_uris": [], "n_terms": 0},
        {"protein_id": "CCC", "gene": "CCC", "encoder_view": "CCC — no terms é",
         "go_uris": ["X"], "coarse_uris": ["X"], "n_terms": 1},
    ]
    ref = tmp_path / "ref.jsonl"
    _ref_proteins(ref, rows)
    lex = _lexicon({"AAA": "", "BBB": "", "CCC": ""},
                   {"AAA": {"AAA"}, "BBB": {"BBB"}, "CCC": {"CCC"}}, max_doc_freq=1.0)
    out = tmp_path / "out.jsonl"
    stats = lp.rewrite_proteins(ref, out, index, lambda g, t: lp.mask_view_text(g, t, lex))
    new = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [r["protein_id"] for r in new] == ["BBB", "AAA", "CCC"]
    assert [list(r) for r in new] == [list(r) for r in rows]          # key order unchanged
    assert new[0]["encoder_view"] == "BBB — binds [GENE] strongly"
    assert new[1]["encoder_view"] == "AAA — partner of [GENE]; AAA itself"
    assert new[0]["go_uris"] == ["T5"] and new[0]["n_terms"] == 1
    assert new[0]["coarse_uris"] == sorted(index.coarse_for("BBB"))
    assert new[1]["go_uris"] == ["T2", "T6"] and new[1]["n_terms"] == 2
    assert new[2]["go_uris"] == [] and new[2]["coarse_uris"] == [] and new[2]["n_terms"] == 0
    assert "é" in out.read_text(encoding="utf-8")                      # ensure_ascii=False
    assert stats["n"] == 3 and stats["empty_go_uris"] == 1
    assert stats["replacements"]["total"] == 2


def test_rewrite_proteins_missing_prefix_is_fatal(tmp_path):
    from go_ppi_domain.tests._withholding_fixtures import build_index

    ref = tmp_path / "ref.jsonl"
    _ref_proteins(ref, [{"protein_id": "AAA", "gene": "AAA", "encoder_view": "AAA: text",
                         "go_uris": [], "coarse_uris": [], "n_terms": 0}])
    with pytest.raises(ValueError, match="does not start with"):
        lp.rewrite_proteins(ref, tmp_path / "o.jsonl", build_index({}),
                            lambda g, t: (t, 0))
