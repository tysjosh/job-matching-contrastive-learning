"""Model (design §6, plan item 11): forward shapes for every arm, masked BCE, DGZ's EL
formulas on hand-computed 3-class examples (both NF4 modes), V3c sharing, DGZ
checkpoint key mapping and mismatch, norm detection, DGZ score propagation."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from go_function import model as M
from go_function import normforms as NF
from go_function.constants import ARMS, OWL_NOTHING
from go_function.evaluate import propagate_max
from go_function.ontology import DgzRule

D = 8
N_IN = 5
VOCAB = ["GO:0000010", "GO:0000011", "GO:0000013", "GO:0000014"]
EPS_SCALE = 1.0 / math.sqrt(1.0 + 1e-5)   # BatchNorm1d in eval mode at init


@pytest.fixture
def nf_full(mini_norm):
    return NF.load_normal_forms(mini_norm, {g: i for i, g in enumerate(VOCAB)},
                                expected_unparsed=None, expected_nothing=None)


def _model(arm, nf_full, **kw):
    nf = NF.select(nf_full, arm) if arm in M.EL_ARMS else None
    text = None
    if arm == "V5":
        text = np.random.default_rng(0).normal(size=(len(VOCAB), M.TEXT_DIM))
    if arm == "V6":
        text = np.random.default_rng(0).normal(size=(len(nf.class_index), M.TEXT_DIM))
    torch.manual_seed(0)
    return M.build_model(arm, N_IN, len(VOCAB), nf, hidden_dim=D, text=text, **kw), nf


@pytest.mark.parametrize("arm", ARMS)
def test_forward_shapes_every_arm(arm, nf_full):
    m, nf = _model(arm, nf_full)
    x = torch.rand(3, N_IN)
    m.eval()
    assert m(x).shape == (3, len(VOCAB))
    if arm in M.EL_ARMS:
        total, parts = m.el_loss()
        assert set(parts) == {"nf1", "nf2", "nf3", "nf4"} and torch.isfinite(total)
        idx = torch.arange(len(VOCAB))
        assert torch.allclose(m.logits_for_classes(x, idx), m(x), atol=1e-6)
        assert m.rel_embed.weight.shape[0] == M.n_relations(nf.rel_index) + 1
    assert len(m.protein_parameters()) + len(m.term_parameters()) == len(list(m.parameters()))


def test_v6_owl_nothing_has_the_only_free_vector(nf_full):
    m, nf = _model("V6", nf_full)
    k = nf.class_index[OWL_NOTHING]
    assert m.nothing_index == k
    centre, radius = m._tables()
    assert torch.equal(centre[k], m.nothing_embed.weight[0])
    assert torch.equal(radius[k], m.nothing_rad.weight[0])
    other = [i for i in range(len(nf.class_index)) if i != k]
    assert torch.allclose(centre[other], m.proj(m.text)[other])
    assert not hasattr(m, "go_embed")
    _, parts = m.el_loss()
    parts["nf2"].backward()                           # NF2 holds the Nothing axiom
    assert m.nothing_embed.weight.grad is not None


def test_v1_to_v3c_have_no_owl_nothing_and_v3c_shares_one_vector(nf_full):
    for arm in ("V1", "V2", "V3", "V3c"):
        assert OWL_NOTHING not in NF.select(nf_full, arm).class_index
    v3, v3c = NF.select(nf_full, "V3"), NF.select(nf_full, "V3c")
    assert set(v3c.rel_index.values()) == {0} and len(set(v3.rel_index.values())) == 4
    assert set(v3c.nf3[:, 0].tolist()) <= {0} and set(v3c.nf4[:, 1].tolist()) == {0}
    m, _ = _model("V3c", nf_full)
    assert m.rel_embed.weight.shape[0] == 2          # shared vector + hasFunction


def test_masked_bce_empty_mask_equals_plain_bce():
    torch.manual_seed(1)
    z, y = torch.randn(4, 6), (torch.rand(4, 6) > 0.5).float()
    plain = F.binary_cross_entropy_with_logits(z, y)
    assert torch.allclose(M.masked_bce(z, y, None), plain)
    assert torch.allclose(M.masked_bce(z, y, torch.zeros(4, 6, dtype=torch.bool)), plain)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[:, 2] = True
    keep = ~mask
    want = F.binary_cross_entropy_with_logits(z[keep], y[keep])
    assert torch.allclose(M.masked_bce(z, y, mask), want)


# ------------------------------------------------------------------ EL by hand
def _el_world(nf4_radius):
    """3 classes (2 columns + 1 zero class), 1 relation, d = 2, eval-mode go_norm."""
    m = M.BilinearModel("V4", N_IN, 2, hidden_dim=2, n_zero=1, n_rels=1, nf4_radius=nf4_radius)
    with torch.no_grad():
        m.go_embed.weight.copy_(torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]]))
        m.go_rad.weight.copy_(torch.tensor([[1.0], [-2.0], [0.5]]))      # |r| is used
        m.rel_embed.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))  # rel 0, hasFunc
    m.set_normal_forms({"nf1": np.array([[0, 1]]), "nf2": np.array([[0, 1, 2]]),
                        "nf3": np.array([[0, 0, 1]]), "nf4": np.array([[1, 0, 2]])})
    return m.eval()


def _hand(nf4_radius):
    s, mg = EPS_SCALE, 0.1
    c = {0: np.array([0.0, 0.0]) * s, 1: np.array([3.0, 4.0]) * s, 2: np.array([1.0, 1.0]) * s}
    r = {0: 1.0, 1: 2.0, 2: 0.5}
    rel = np.array([1.0, 0.0])
    n = np.linalg.norm
    relu = lambda v: max(v, 0.0)  # noqa: E731
    nf1 = relu(n(c[0] - c[1]) + r[0] - r[1] - mg)
    nf2 = (relu(n(c[0] - c[1]) - (r[0] + r[1]) - mg) + relu(n(c[2] - c[0]) - r[0] - mg)
           + relu(n(c[2] - c[1]) - r[1] - mg))
    nf3 = relu(n(c[0] + rel - c[1]) + r[0] - r[1] - mg)
    rc = r[0] if nf4_radius == "dgz" else r[1]      # DGZ: radius of data[:, 1] (the relation id)
    nf4 = relu(n(c[1] - (c[2] + rel)) - (rc + r[2]) - mg)
    return {"nf1": nf1, "nf2": nf2, "nf3": nf3, "nf4": nf4}


@pytest.mark.parametrize("nf4_radius", ["dgz", "class"])
def test_el_matches_hand_computed_three_class_example(nf4_radius):
    m = _el_world(nf4_radius)
    total, parts = m.el_loss()
    want = _hand(nf4_radius)
    for f in ("nf1", "nf2", "nf3", "nf4"):
        assert float(parts[f].detach()) == pytest.approx(want[f], abs=1e-5), f
    assert float(total.detach()) == pytest.approx(sum(want.values()), abs=1e-5)


def test_nf4_modes_differ_and_empty_forms_are_zero():
    a, b = _el_world("dgz").el_loss()[1], _el_world("class").el_loss()[1]
    assert float(a["nf4"].detach()) != pytest.approx(float(b["nf4"].detach()))
    m = _el_world("dgz")
    m.set_normal_forms({"nf1": np.zeros((0, 2)), "nf2": np.zeros((0, 3)),
                        "nf3": np.zeros((0, 3)), "nf4": np.zeros((0, 3))})
    total, parts = m.el_loss()
    assert float(total.detach()) == 0.0


def test_score_is_dgz_forward(nf_full):
    m, _ = _model("V4", nf_full)
    m.eval()
    x = torch.rand(2, N_IN)
    h = m.net(x)
    g = m.go_embed.weight[:len(VOCAB)] + m.rel_embed.weight[m.has_func]
    want = h @ g.T + torch.abs(m.go_rad.weight[:len(VOCAB)]).view(1, -1)
    assert torch.allclose(m(x), want, atol=1e-6)


# ------------------------------------------------------------------ checkpoints
def _dgz_mlp_state(m):
    return {("net.2." + k[3:]) if k.startswith("fc.") else k: v for k, v in m.state_dict().items()}


def test_checkpoint_key_mapping_mlp_and_deepgozero(nf_full):
    src, _ = _model("V0", nf_full)
    sd = _dgz_mlp_state(src)
    assert "net.2.weight" in sd and not any(k.startswith("fc.") for k in sd)
    got = M.DgzCompatModel.load(None, kind="mlp", n_in=N_IN, n_terms=len(VOCAB), state_dict=sd)
    assert got.norm_type == "layernorm" and got.hidden_dim == D
    x = torch.rand(3, N_IN)
    src.eval()
    assert torch.allclose(got(x), src(x))
    src4, _ = _model("V4", nf_full)
    got4 = M.DgzCompatModel.load(None, kind="deepgozero", n_in=N_IN, n_terms=len(VOCAB),
                                 nf=nf_full, state_dict=src4.state_dict())
    src4.eval()
    assert torch.allclose(got4(x), src4(x))


def test_checkpoint_mismatch_lists_keys_and_shapes(nf_full):
    src4, _ = _model("V4", nf_full)
    sd = dict(src4.state_dict())
    sd.pop("rel_embed.weight")
    sd["go_rad.weight"] = torch.zeros(3, 1)
    sd["extra.weight"] = torch.zeros(1)
    with pytest.raises(M.CheckpointMismatch) as e:
        M.DgzCompatModel.load(None, kind="deepgozero", n_in=N_IN, n_terms=len(VOCAB),
                              nf=nf_full, state_dict=sd)
    assert e.value.missing == ["rel_embed.weight"] and e.value.unexpected == ["extra.weight"]
    assert e.value.shapes and e.value.shapes[0][0] == "go_rad.weight"
    assert e.value.exit_code == 2


def test_detect_norm():
    ln = M.BilinearModel("V0", N_IN, 2, hidden_dim=D, norm="layernorm").state_dict()
    bn = M.BilinearModel("V0", N_IN, 2, hidden_dim=D, norm="batchnorm1d").state_dict()
    assert M.detect_norm(ln) == "layernorm"
    assert M.detect_norm(bn) == "batchnorm1d"
    with pytest.raises(M.CheckpointMismatch):
        M.detect_norm({"net.0.linear.weight": torch.zeros(1)})
    got = M.DgzCompatModel.load(None, kind="mlp", n_in=N_IN, n_terms=2,
                                state_dict=_dgz_mlp_state(
                                    M.BilinearModel("V0", N_IN, 2, hidden_dim=D,
                                                    norm="batchnorm1d")))
    assert isinstance(got.net[0].layer_norm, torch.nn.BatchNorm1d)


def test_dgz_score_propagation_equals_dgz_loop(mini_obo):
    dgz = DgzRule(mini_obo)
    terms = ["GO:0000010", "GO:0000011", "GO:0000013", "GO:0000014", "GO:0000016",
             "GO:0000099", "GO:0008150"]
    preds = np.random.default_rng(3).random((5, len(terms))).astype(np.float32)
    src, dst = M.dgz_propagation_pairs(terms, dgz)
    got = propagate_max(torch.as_tensor(preds), src, dst).numpy()
    want = preds.copy()                       # deepgozero.py post-prediction loop
    tdict = {g: j for j, g in enumerate(terms)}
    for i in range(len(want)):
        prop = {}
        for g, j in tdict.items():
            for sup in dgz.closure(g):
                prop[sup] = max(prop.get(sup, -1.0), preds[i, j])
        for g, sc in prop.items():
            if g in tdict:
                want[i, tdict[g]] = sc
    assert np.array_equal(got, want)
    assert (got >= preds).all()
