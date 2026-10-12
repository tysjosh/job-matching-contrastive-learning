"""Parity of go_function.vendored with the originals (design §13.2).

The only module in go_function that may import go_ppi_domain/scripts.go_withholding.
"""
import importlib.util

import numpy as np
import pytest

from go_function import vendored as V
from go_function.constants import ALPHA, FLOOR, UNSTABLE_SHARE
from go_function.runtime import ROOT


@pytest.fixture(scope="module")
def orig():
    spec = importlib.util.spec_from_file_location(
        "_orig_analyze_withholding", ROOT / "scripts" / "go_withholding" / "analyze_withholding.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def orig_seed():
    from go_ppi_domain import withholding_constants as wc
    return wc.ANALYSIS_SEED


def _X(seed=0, units=24, seeds=5, nan_share=0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.02, 0.05, size=(units, seeds))
    X[rng.random(X.shape) < nan_share] = np.nan
    return X


def test_constants_match_original(orig):
    assert (orig.FLOOR, orig.ALPHA, orig.UNSTABLE_SHARE) == (FLOOR, ALPHA, UNSTABLE_SHARE)
    assert orig.SCHEME == V.SCHEME


def test_jaccard():
    from go_ppi_domain.withholding_targets import jaccard
    cases = [(set(), set()), ({"a"}, {"a"}), ({"a", "b"}, {"b", "c"}), ({"a"}, set())]
    for a, b in cases:
        assert V.jaccard(a, b) == jaccard(a, b)


def test_draws_and_counts(orig, orig_seed):
    a = V.two_way_draws(8, 5, 300, orig_seed)
    b = orig.two_way_draws(8, 5, 300, orig_seed)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    d = np.random.default_rng(1).integers(0, 6, size=(10, 6))
    assert np.array_equal(V._counts(d, 6), orig._counts(d, 6))


def test_summarise(orig):
    rng = np.random.default_rng(3)
    T = rng.normal(size=200)
    keep = rng.random(200) > 0.1
    a = V._summarise(T, keep, 200, 0.1, 50, unstable_share=UNSTABLE_SHARE)
    b = orig._summarise(T, keep, 200, 0.1, 50)
    assert a == b


@pytest.mark.parametrize("data_seed", [0, 1, 2])
def test_two_way_bootstrap(orig, orig_seed, data_seed):
    X = _X(data_seed)
    unit_pack = [u % 8 for u in range(X.shape[0])]
    strata = [(1 if u % 3 == 0 else 0 if u % 3 == 1 else -1) for u in range(X.shape[0])]
    for st in (None, strata):
        a = V.two_way_bootstrap(X, unit_pack, 8, 400, seed=orig_seed,
                                unstable_share=UNSTABLE_SHARE, strata=st)
        b = orig.two_way_bootstrap(X, unit_pack, 8, 400, seed=orig_seed, strata=st)
        assert a == b


@pytest.mark.parametrize("data_seed", [0, 4])
def test_hierarchical_bootstrap(orig, orig_seed, data_seed):
    X = _X(data_seed, units=12)
    strata = [u % 2 for u in range(12)]
    for st in (None, strata):
        a = V.hierarchical_bootstrap(X, 150, seed=orig_seed, unstable_share=UNSTABLE_SHARE,
                                     strata=st)
        b = orig.hierarchical_bootstrap(X, 150, seed=orig_seed, strata=st)
        assert a == b


def test_holm(orig):
    rng = np.random.default_rng(9)
    for _ in range(20):
        p = {f"C{i}": (None if rng.random() < 0.2 else float(rng.random())) for i in range(7)}
        assert V.holm(p) == orig.holm(p)


def test_classify_and_ci(orig):
    rng = np.random.default_rng(11)
    cases = [{"unstable": True, "estimate": 0.1, "ci_low": 0.05, "ci_high": 0.2, "p_holm": 0.0},
             {"estimate": None, "ci_low": None, "ci_high": None},
             {"estimate": -0.05, "ci_low": -0.08, "ci_high": -0.01, "p_holm": 0.01},
             {"estimate": 0.05, "ci_low": 0.01, "ci_high": 0.08, "p_holm": 0.01},
             {"estimate": 0.0, "ci_low": -0.005, "ci_high": 0.005, "p_holm": 0.9},
             {"estimate": 0.005, "ci_low": 0.001, "ci_high": 0.02, "p_holm": 0.2}]
    for _ in range(200):
        lo = float(rng.normal(0, 0.03))
        hi = lo + float(abs(rng.normal(0, 0.03)))
        cases.append({"estimate": (lo + hi) / 2, "ci_low": lo, "ci_high": hi,
                      "p_holm": float(rng.random()), "unstable": bool(rng.random() < 0.1)})
    for c in cases:
        assert V.classify_contrast(c, floor=FLOOR, alpha=ALPHA) == orig.classify_contrast(c)
        assert V.ci_excludes_0(c) == orig.ci_excludes_0(c)


def test_vendored_takes_no_defaults():
    import inspect
    for fn, names in ((V.two_way_bootstrap, ("seed", "unstable_share")),
                      (V.hierarchical_bootstrap, ("seed", "unstable_share")),
                      (V._summarise, ("unstable_share",)),
                      (V.classify_contrast, ("floor", "alpha"))):
        sig = inspect.signature(fn)
        for n in names:
            assert sig.parameters[n].default is inspect.Parameter.empty
            assert sig.parameters[n].kind is inspect.Parameter.KEYWORD_ONLY
