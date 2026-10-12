import numpy as np
import pytest

from go_function.constants import OWL_NOTHING
from go_function.normforms import (NormFormError, assert_ladder, load_normal_forms, select)

VOCAB = ["GO:0000010", "GO:0000011", "GO:0000012", "GO:0000013", "GO:0000014", "GO:0000016"]
TD = {g: i for i, g in enumerate(VOCAB)}


@pytest.fixture
def nf(mini_norm):
    return load_normal_forms(mini_norm, TD, expected_unparsed=1, expected_nothing=1)


def test_every_form_and_role_axiom(nf):
    assert nf.counts()["nf1"] == 2 and len(nf.nf2) == 2 and len(nf.nf3) == 2 and len(nf.nf4) == 8
    assert nf.unparsed == 1 and nf.nothing_lines == 1
    # NF1 rows in file order: 11 ⊑ 10, 12 ⊑ 10
    assert nf.nf1.tolist() == [[1, 0], [2, 0]]


def test_dgz_first_appearance_order(nf):
    zero = {c: i for c, i in nf.class_index.items() if i >= len(VOCAB)}
    assert list(zero) == ["GO:0000015", "GO:0000040", "GO:0000041", "GO:0000042",
                          "GO:0000020", "GO:0005515", OWL_NOTHING, "GO:0000043", "GO:0000044"]
    assert [zero[c] for c in zero] == list(range(6, 15))
    assert nf.rel_index == {"BFO:0000050": 0, "RO:0002211": 1, "RO:0002213": 2,
                            "RO:0002212": 3, "BFO:0000051": 4, "BFO:0000066": 5,
                            "RO:0002092": 6, "RO:0002093": 7}
    # NF3 is (rel, c, d); NF4 is (c, rel, d)
    assert nf.nf3.tolist()[0] == [0, 2, nf.class_index["GO:0000041"]]
    assert nf.nf4.tolist()[0] == [2, 0, 3]


def test_nothing_mapping_and_position(nf):
    i = nf.class_index[OWL_NOTHING]
    assert i == 12
    assert [r for r in nf.nf2.tolist() if r[2] == i] == [[1, 3, 12]]


def test_pinned_counts_enforced(mini_norm):
    with pytest.raises(NormFormError):
        load_normal_forms(mini_norm, TD)  # pinned 9 unparsed / 30 Nothing
    with pytest.raises(NormFormError):
        load_normal_forms(mini_norm, TD, expected_unparsed=1, expected_nothing=2)


def test_fatal_tokens(tmp_path):
    p = tmp_path / "bad.norm"
    p.write_text("GO_0000010 SubClassOf CHEBI_0000001\n")
    with pytest.raises(NormFormError):
        load_normal_forms(p, TD, expected_unparsed=None, expected_nothing=None)
    p.write_text("GO_0000010 SubClassOf RO_9999999 some GO_0000011\n")
    with pytest.raises(NormFormError):
        load_normal_forms(p, TD, expected_unparsed=None, expected_nothing=None)
    p.write_text("GO_0000010 SubClassOf Nothing\n")
    with pytest.raises(NormFormError):
        load_normal_forms(p, TD, expected_unparsed=None, expected_nothing=None)


def test_ladder(nf):
    counts = assert_ladder(nf)
    v1, v2, v3, v3c, v4 = (select(nf, a) for a in ("V1", "V2", "V3", "V3c", "V4"))
    assert (len(v1.nf3), len(v1.nf4), len(v1.nf2)) == (0, 0, 0)
    assert v1.rel_index == {} and len(v1.class_index) == len(VOCAB)
    assert v2.rel_index == {"BFO:0000050": 0}
    assert (len(v2.nf3), len(v2.nf4)) == (1, 1)
    assert v2.class_index["GO:0000041"] == len(VOCAB)  # re-indexed within the subset
    assert v3.rel_index == {"BFO:0000050": 0, "RO:0002211": 1, "RO:0002213": 2, "RO:0002212": 3}
    assert (len(v3.nf3), len(v3.nf4)) == (2, 4)
    assert set(v3c.rel_index) == set(v3.rel_index) and set(v3c.rel_index.values()) == {0}
    assert np.array_equal(v3c.nf1, v3.nf1) and v3c.class_index == v3.class_index
    assert set(v3c.nf3[:, 0]) == {0} and set(v3c.nf4[:, 1]) == {0}
    for s in (v1, v2, v3, v3c):
        assert OWL_NOTHING not in s.class_index and len(s.nf2) == 0
    assert v4.class_index == nf.class_index and v4.rel_index == nf.rel_index
    assert np.array_equal(v4.nf4, nf.nf4) and np.array_equal(v4.nf2, nf.nf2)
    assert counts["V6"] == counts["V4"]
    with pytest.raises(ValueError):
        select(nf, "V0")


def test_v3c_keeps_v3_nf4_radius_slots(nf):
    """Amendment A5 (review finding 4): V3c shares the relation vector only; its NF4
    radius index equals V3's per-relation ids, so C1 is single-factor."""
    v3, v3c, v4 = select(nf, "V3"), select(nf, "V3c"), select(nf, "V4")
    assert np.array_equal(v3c.nf4_rad, v3.nf4_rad)
    assert np.array_equal(v3.nf4_rad, v3.nf4[:, 1]) and len(set(v3.nf4_rad.tolist())) > 1
    assert set(v3c.nf4[:, 1].tolist()) == {0}                     # one shared relation vector
    assert np.array_equal(v4.nf4_rad, nf.nf4[:, 1])               # DGZ indexing unchanged


def test_v3_and_v3c_nf4_loss_read_the_same_radii(nf):
    """With identical class tables and the shared relation vector set equal to every
    V3 relation vector, V3 and V3c give the same NF4 loss: only relation typing differs."""
    import torch
    from go_function.model import build_model
    torch.manual_seed(0)
    v3 = build_model("V3", 4, len(VOCAB), select(nf, "V3"), hidden_dim=8)
    v3c = build_model("V3c", 4, len(VOCAB), select(nf, "V3c"), hidden_dim=8)
    with torch.no_grad():
        v3c.go_embed.weight.copy_(v3.go_embed.weight)
        v3c.go_rad.weight.copy_(v3.go_rad.weight)
        v3.rel_embed.weight[:] = v3.rel_embed.weight[0]
        v3c.rel_embed.weight[:] = v3.rel_embed.weight[0]
    v3.eval(), v3c.eval()
    # radii of classes 0-3 differ (random init), so a V3c that read class 0's radius
    # for every NF4 axiom (the pre-A5 behaviour) would give a different loss
    assert len(set(v3.go_rad.weight[:4, 0].tolist())) == 4
    a, b = v3.el_loss()[1]["nf4"], v3c.el_loss()[1]["nf4"]
    assert torch.allclose(a, b)


@pytest.mark.slow
@pytest.mark.needs_dgz
def test_real_go_norm_counts():
    from go_function.labels import read_terms
    from go_function.runtime import ROOT
    dgz = ROOT / "dataset" / "go_function" / "dgz"
    vocab = read_terms(dgz / "bp" / "terms.pkl")
    nf = load_normal_forms(dgz / "go.norm", {g: i for i, g in enumerate(vocab)})
    c = nf.counts()
    assert (c["nf1"], c["nf2"], c["nf3"], c["nf4"]) == (80942, 11841, 11810, 19594)
    assert c["unparsed"] == 9 and c["nothing_lines"] == 30
    assert set(nf.rel_index) == {"BFO:0000050", "BFO:0000051", "BFO:0000066", "RO:0002211",
                                 "RO:0002212", "RO:0002213", "RO:0002092", "RO:0002093"}
    assert_ladder(nf)
    v3, v3c = select(nf, "V3"), select(nf, "V3c")
    assert np.array_equal(v3.nf4_rad, v3c.nf4_rad) and len(v3c.nf4_rad) == len(v3c.nf4)
    assert sorted(set(v3c.nf4_rad.tolist())) == [0, 1, 2, 3]
