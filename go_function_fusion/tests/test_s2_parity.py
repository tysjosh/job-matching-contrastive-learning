"""S2 parity on the GF smoke ``scores.npz``: F recomputed equals fusion_F.csv (label-free
columns, auc_V0 and score level; the auc_F column needs test labels and is checked on
the remote with --with-labels), and AP-T_S2 is computed for every V0 run."""
import importlib.util
import json

import numpy as np
import pytest

from go_function_fusion.tests.conftest import ROOT, SMOKE

pytestmark = pytest.mark.needs_data


def _s2():
    spec = importlib.util.spec_from_file_location(
        "s2_grid_fusion_t", ROOT / "scripts" / "go_function_fusion" / "s2_grid_fusion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def smoke_rows():
    S2 = _s2()
    from go_function.constants import ROOTS
    from go_function.labels import read_terms
    from go_function_fusion import data as D
    from go_function_fusion import priors as PR
    ctx = D.aspect_context("bp", ("train", "valid"))
    terms = read_terms(ROOT / D.DGZ / "bp" / "terms.pkl")
    vidx = {g: i for i, g in enumerate(terms)}
    reg = PR.neighbour_tables(ctx.ont, ctx.vocab, ctx.anc, ctx.desc, ctx.ntr, ctx.root,
                              "confirmatory", keys=("reg",))["reg"]
    table = json.loads((ROOT / D.PRE / "bp" / "targets/target_table.json").read_text())
    params = {"c": 2.0, "tau": 30.0, "label": S2.SMOKE_LABEL}
    rows, par = [], []
    for d in S2.discover(SMOKE / "runs"):
        r, p = S2.s2_run(d, table, vidx, ROOTS["bp"], reg, params, None)
        rows += r
        par += p
    return S2, rows, par


def test_f_recomputed_equals_fusion_F_csv(smoke_rows):
    S2, rows, par = smoke_rows
    res = S2.check_f_parity(par, SMOKE / "analysis" / "fusion_F.csv")
    assert res["passed"], res["misses"][:5]
    assert res["n_compared"] == res["n_reference_rows"] == 12
    assert max(p["score_maxdiff_vs_fuse"] for p in par) <= 1e-9


def test_apt_s2_for_every_v0_run_and_schema(smoke_rows):
    S2, rows, _ = smoke_rows
    v0 = [r for r in rows if r["variant"] == "V0"]
    assert len(v0) == 12 and {r["k"] for r in v0} == {0, 5}
    for r in v0:
        assert set(S2.SCHEMA_COLS) <= set(r)
        k = r["k"]
        assert r["w_t"] == pytest.approx(2.0 * 30.0 / (30.0 + k))
        assert r["label"] == "smoke, 2 epochs, non-informative"
        assert r["auc_APT_S2"] is None          # no test label is read on the Mac
    assert any(r["n_reg_used"] + r["n_reg_skipped"] > 0 for r in rows)

