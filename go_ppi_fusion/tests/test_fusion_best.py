"""Unit tests for the GO/PPI inference-time fusion library (CPU only, no test labels)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_fusion import data as D  # noqa: E402
from go_ppi_fusion import fusion as F  # noqa: E402
from go_ppi_fusion import guard  # noqa: E402
from go_ppi_fusion import metrics as M  # noqa: E402
from go_ppi_fusion import search as S  # noqa: E402
from scripts import evaluate_publication_fusion as epf  # noqa: E402


def synthetic(n_q=60, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    q, g = [], []
    for i in range(n_q):
        n = int(rng.integers(1, 25))
        q += [f"Q{i:03d}"] * n
        g += list(rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]))
    g = np.asarray(g)
    perm = rng.permutation(len(g))            # interleave queries like a real file
    q = np.asarray(q, dtype=object)[perm]
    g = g[perm]
    text = rng.normal(size=len(g)) + 0.8 * g
    if ties:
        text = np.round(text, 1)
    phi = np.clip(rng.normal(0.1, 0.1, size=len(g)) + 0.05 * g, 0, 1)
    miss = rng.random(len(g)) < 0.15
    phi[miss] = 0.0
    phis = {"simgic_P_unpruned": phi, "miss_P_unpruned": miss.astype(float),
            "na_P_unpruned": rng.integers(0, 30, len(g)).astype(float),
            "nb_P_unpruned": rng.integers(0, 30, len(g)).astype(float),
            "simgic_F_unpruned": np.clip(rng.normal(0.1, 0.1, len(g)), 0, 1),
            "miss_F_unpruned": np.zeros(len(g))}
    return q, g, text, phis


# ---------------------------------------------------------------- metrics
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_match_published(seed):
    q, g, s, _ = synthetic(seed=seed)
    mine = M.macro(M.per_query(s, M.QueryIndex(q, g)))
    c = epf.contrast_metrics(s, g, q)
    r = epf.ranking_metrics(s, g, q)
    for name in ("hard", "easy", "middle_vs_easy", "pooled"):
        assert mine[f"auc_{name}"] == pytest.approx(c["macro_query_auc"][name], abs=1e-12)
    for name in M.RANKING:
        assert mine[name] == pytest.approx(r["macro"][name], abs=1e-12)
    assert M.global_auc(s, g) == pytest.approx(epf.rank_auc(s[g == 2], s[g != 2]), abs=1e-12)


def test_metrics_per_query_order_is_np_unique():
    q, g, s, _ = synthetic(seed=3)
    qi = M.QueryIndex(q, g)
    pq = M.per_query(s, qi)
    rows = epf.ranking_metrics(s, g, q)["per_query"]
    assert [r["query"] for r in rows] == list(qi.queries)
    ap = np.array([np.nan if r["average_precision"] != r["average_precision"]
                   else r["average_precision"] for r in rows])
    np.testing.assert_allclose(pq["average_precision"], ap, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------- fusion
def _inp(q, g, phis, split="validation"):
    return F.Inputs(split, q, phis, M.QueryIndex(q, g), g)


def test_linear_zero_is_published_form():
    q, g, t, phis = synthetic()
    inp = _inp(q, g, phis)
    fam = F.make_family({"family": "linear", "phi": "simgic_P_unpruned", "missing": "zero"})
    st = fam.prepare(t, inp, np.ones(len(g), bool))
    zt = epf.z_apply(t, epf.z_stats(t))
    zp = epf.z_apply(phis["simgic_P_unpruned"], epf.z_stats(phis["simgic_P_unpruned"]))
    for w in (0.0, 0.3, 1.0):
        np.testing.assert_array_equal(fam.score(st, w, t, inp), (1 - w) * zt + w * zp)


def test_linear_impute_and_text_only_handle_missing():
    q, g, t, phis = synthetic()
    inp = _inp(q, g, phis)
    miss = phis["miss_P_unpruned"] > 0
    imp = F.make_family({"family": "linear", "phi": "simgic_P_unpruned", "missing": "impute"})
    st = imp.prepare(t, inp, np.ones(len(g), bool))
    zt = F.z_apply(t, st["t"])
    np.testing.assert_allclose(imp.score(st, 0.4, t, inp)[miss], 0.6 * zt[miss])
    # z stats of phi are fit on annotated records only
    assert st["p"][0] == pytest.approx(phis["simgic_P_unpruned"][~miss].mean())
    to = F.make_family({"family": "linear", "phi": "simgic_P_unpruned", "missing": "text_only"})
    np.testing.assert_allclose(to.score(st, 0.4, t, inp)[miss], zt[miss])


def test_within_query_rank_and_rank_fusion():
    from scipy.stats import rankdata
    q, g, t, phis = synthetic()
    qi = M.QueryIndex(q, g)
    r = F.within_query_rank(t, qi)
    rd = F.within_query_rank(t, qi, pct=False)
    for query in qi.queries[:10]:
        m = q == query
        np.testing.assert_allclose(r[m], rankdata(t[m]) / m.sum())
        np.testing.assert_allclose(rd[m], m.sum() + 1 - rankdata(t[m]))
    inp = _inp(q, g, phis)
    rank = F.make_family({"family": "rank", "phi": "simgic_P_unpruned"})
    # w = 0 keeps the text order within every anchor
    s0 = rank.score({}, 0.0, t, inp)
    assert M.macro(M.per_query(s0, qi))["ndcg@10"] == pytest.approx(
        M.macro(M.per_query(t, qi))["ndcg@10"])
    rrf = F.make_family({"family": "rrf", "phi": "simgic_P_unpruned", "k": 60})
    s1 = rrf.score({}, 0.0, t, inp)
    assert M.macro(M.per_query(s1, qi))["auc_pooled"] == pytest.approx(
        M.macro(M.per_query(t, qi))["auc_pooled"])
    z = F.within_query_z(t, qi)
    for query in qi.queries[:5]:
        assert abs(z[q == query].mean()) < 1e-9


# ---------------------------------------------------------------- stacker
@pytest.mark.parametrize("model", ["lr", "pairwise", "hgb"])
def test_stacker_fits_on_validation_fit_records_only(model):
    q, g, t, phis = synthetic(n_q=120)
    spec = {"family": "stacker", "model": model, "phi": ["simgic_P_unpruned", "simgic_F_unpruned"],
            "flags": ["miss_P_unpruned"], "counts": ["na_P_unpruned", "nb_P_unpruned"]}
    fam = F.make_family(spec)
    inp = _inp(q, g, phis)
    fit = np.zeros(len(g), bool)
    fit[: len(g) // 2] = True
    st = fam.prepare(t, inp, fit)
    s = fam.score(st, None, t, inp)
    # labels outside the fit mask cannot influence the fit
    g2 = g.copy()
    g2[~fit] = 2 - g2[~fit]
    st2 = fam.prepare(t, _inp(q, g2, phis), fit)
    np.testing.assert_allclose(fam.score(st2, None, t, inp), s)
    # a test-tagged input is refused
    with pytest.raises(PermissionError):
        fam.prepare(t, _inp(q, g, phis, split="test"), fit)


# ---------------------------------------------------------------- search
def test_search_cv_rewards_signal_and_refuses_test():
    q, g, t, phis = synthetic(n_q=150)
    phis = dict(phis)
    phis["simgic_P_unpruned"] = g + np.random.default_rng(5).normal(0, 0.5, len(g))
    phis["miss_P_unpruned"] = np.zeros(len(g))
    inp = _inp(q, g, phis)
    spec = {"family": "linear", "phi": "simgic_P_unpruned", "select_on": "ndcg@10", "step": 0.1}
    res = S.evaluate_variant(spec, [t, t + 0.1], inp)
    assert res["summary"]["cv_delta"]["auc_pooled"] > 0
    assert len(res["per_checkpoint"]) == 2
    fitted = S.final_fit(spec, [t], inp)
    assert fitted[0]["theta"] > 0
    with pytest.raises(guard.TestAccessError):
        S.evaluate_variant(spec, [t], _inp(q, g, phis, split="test"))
    with pytest.raises(guard.TestAccessError):
        S.final_fit(spec, [t], _inp(q, g, phis, split="test"))


def test_scope_groups():
    grid = D.checkpoint_grid()
    assert len(S.scope_groups("checkpoint", grid)) == 15
    assert [len(x) for x in S.scope_groups("fraction", grid)] == [3] * 5
    assert S.scope_groups("global", grid) == [list(range(15))]


# ---------------------------------------------------------------- alignment
def _rec(anchor, partner, grade, uris=("GO:1",)):
    return {"resume": {"protein_id": anchor, "skill_uris": list(uris)},
            "job": {"partner_id": partner, "grade": grade, "skill_uris": list(uris)},
            "metadata": {"resume_id": anchor}}


def test_assert_aligned_detects_mismatch():
    a = [_rec("A", "B", 2), _rec("A", "C", 0)]
    D.assert_aligned(a, [_rec("A", "B", 2, ()), _rec("A", "C", 0, ())])  # URIs may differ
    with pytest.raises(AssertionError):
        D.assert_aligned(a, [_rec("A", "C", 0), _rec("A", "B", 2)])
    with pytest.raises(AssertionError):
        D.assert_aligned(a, [_rec("A", "B", 1), _rec("A", "C", 0)])
    with pytest.raises(AssertionError):
        D.assert_aligned(a, a[:1])


def test_real_validation_pruned_alignment_and_uris():
    split = D.load_split("validation", with_embeddings=False, with_pruned=True)
    assert split.n == 5587 and len(set(split.queries)) == 697
    assert split.meta["sha256"].startswith("07e432fb")
    changed = sum(a != b for a, b in zip(split.uris_a, split.pruned_uris_a))
    assert changed > 0  # pruned URIs are really the pruned ones


# ---------------------------------------------------------------- guard
def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                          text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_guard_blocks_test_in_search_mode():
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from go_ppi_fusion import guard, data as D\n"
        "guard.enter_search_mode()\n"
        "D.load_split('validation', with_embeddings=False, with_pruned=False)\n"
        "for fn in (lambda: D.load_split('test', with_embeddings=False),\n"
        "           lambda: D.read_jsonl(D.SPLITS['test']),\n"
        "           lambda: D.read_jsonl(D.PRUNED_SPLITS['test']),\n"
        "           lambda: D.file_sha256(D.SPLITS['test']),\n"
        "           lambda: guard.check_path(guard.TEST_PATHS[2] / 'chunk_x.npz'),\n"
        "           lambda: guard.check_path(guard.TEST_PATHS[3])):\n"
        "    try:\n"
        "        fn(); print('LEAK'); sys.exit(3)\n"
        "    except guard.TestAccessError:\n"
        "        pass\n"
        "print('BLOCKED')\n")
    out = _run(code)
    assert out.returncode == 0, out.stderr
    assert "BLOCKED" in out.stdout


def test_guard_is_inactive_outside_search_mode():
    out = _run("import sys; sys.path.insert(0, '.')\n"
               "from go_ppi_fusion import guard\n"
               "guard.check_path(guard.TEST_PATHS[0]); print('OK')\n")
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr
