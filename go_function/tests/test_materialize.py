"""Materialiser (plan item 15): block counts, the single-factor audit, E* refusal,
write-or-verify and repo-relative outputs."""
import importlib.util
import json
import logging
from pathlib import Path

import pytest

from go_function.constants import EXIT_OK, EXIT_REFUSED
from go_function.runtime import FatalError

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_materialize")


def _load():
    spec = importlib.util.spec_from_file_location("materialize_t",
                                                  ROOT / "scripts" / "go_function" / "materialize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _study(**star):
    s = json.loads((ROOT / "config" / "go_function" / "study.json").read_text())
    s["epochs_star"] = star or {"bp_esm2_650m": 12, "bp_interpro": 10, "mf_esm2_650m": 11,
                                "mf_interpro": 9}
    return s


def test_block_counts_core_a_total_524_bp_and_3_mf():
    """Protocol A7 (Core A): BP ESM-2 main block (6 arms x (all + 8 packs x k 0/1/5)
    x 3 seeds = 450) + the three V4 C3 cells at k = 0 (3 x 24), and the fusion study's
    Tier B `all` / seed-13 bases as ordinary cells."""
    s = _study()
    per, names = {}, {}
    for a in ("bp", "mf"):
        for f in ("esm2_650m", "interpro"):
            items, counts = M.build_specs(s, a, f)
            assert all(c["n"] == c["expected"] for c in counts.values())
            per[(a, f)] = len(items)
            names[(a, f)] = {n for _, n, _ in items}
            assert len(names[(a, f)]) == len(items)
    assert per == {("bp", "esm2_650m"): 522, ("bp", "interpro"): 2,
                   ("mf", "esm2_650m"): 1, ("mf", "interpro"): 2}
    items, counts = M.build_specs(s, "bp", "esm2_650m")
    assert counts["main"] == {"n": 450, "expected": 450}
    assert {sp["variant"] for _, _, sp in items} == {"V0", "V3", "V3c", "V4", "V5", "V6"}
    assert {sp["k"] for _, _, sp in items} == {0, 1, 5, "all"}
    assert {sp["seed"] for _, _, sp in items} == {13, 21, 42}
    assert all(sp["epochs"] == 12 for _, _, sp in items)
    tier_b = "__TP__W2__all__kall__s13"
    assert f"V0{tier_b}" in names[("bp", "esm2_650m")]
    assert names[("bp", "interpro")] == names[("mf", "interpro")] == {f"V0{tier_b}", f"V4{tier_b}"}
    assert names[("mf", "esm2_650m")] == {f"V0{tier_b}"}


def test_refused_without_estar(tmp_path):
    sp = tmp_path / "study.json"
    s = _study()
    s["epochs_star"] = {}
    sp.write_text(json.dumps(s))
    assert M.main(["--aspect", "bp", "--feature", "interpro"], spec_root=tmp_path / "specs",
                  study_path=sp) == EXIT_REFUSED
    assert not (tmp_path / "specs").exists()


def test_audit_passes_on_the_grid_and_trips_on_a_planted_difference():
    items, _ = M.build_specs(_study(), "bp", "esm2_650m")
    specs = [dict(s) for _, _, s in items]
    rows = {r["contrast"]: r for r in M.audit(specs)}
    assert rows["C3 (DGZ/WDGZrel - TP/W2)"]["n_pairs"] == 24
    assert rows["C3 (DGZ/WDGZrel - TP/W2)"]["declared"] == ["regime", "rule"]
    # planted: one V0 spec gets a different learning rate
    bad = [dict(s) for s in specs]
    i = next(i for i, s in enumerate(bad) if s["variant"] == "V0" and s["regime"] == "TP"
             and s["k"] == 0)
    bad[i]["lr"] = 1e-3
    with pytest.raises(FatalError, match="differ in"):
        M.audit(bad)
    # planted: a missing partner
    gone = [s for s in specs if not (s["rule"] == "W2drop" and s["pack"] == "p03"
                                     and s["seed"] == 21 and s["regime"] == "DGZ")]
    gone = [s for s in gone if not (s["rule"] == "W2" and s["regime"] == "DGZ"
                                    and s["variant"] == "V4" and s["pack"] == "p03"
                                    and s["k"] == 0 and s["seed"] == 21)]
    with pytest.raises(FatalError, match="no partner"):
        M.audit(gone)


def test_lambda_difference_implied_by_variant_is_not_a_second_factor():
    s = _study()
    a = M.make_spec(s, aspect="bp", feature="interpro", variant="V4", regime="TP", rule="W2",
                    pack="p01", k=0, seed=13, epochs=10, out_dir="results/go_function/x/a")
    b = M.make_spec(s, aspect="bp", feature="interpro", variant="V0", regime="TP", rule="W2",
                    pack="p01", k=0, seed=13, epochs=10, out_dir="results/go_function/x/b")
    assert a["lambda_el"] == 1.0 and b["lambda_el"] is None
    assert M.spec_diff(a, b) == {"variant"}
    c = dict(b, variant="V5")
    assert M.spec_diff(c, dict(b, lambda_el=2.0)) == {"variant", "lambda_el"}


def test_materialize_writes_or_verifies_and_is_repo_relative(tmp_path):
    s = _study()
    r1 = M.materialize(s, "mf", "interpro", tmp_path, LOG)
    assert r1["n_specs"] == 2 and r1["status"]["written"] == 2
    r2 = M.materialize(s, "mf", "interpro", tmp_path, LOG)
    assert r2["status"]["unchanged"] == 2 and r2["index"] == "unchanged"
    assert r1["spec_set_sha256"] == r2["spec_set_sha256"]
    idx = json.loads((tmp_path / "mf" / "interpro" / "index.json").read_text())
    assert idx["blocks"]["s1_tier_b_interpro"]["n"] == 2
    for p in (tmp_path / "mf" / "interpro").iterdir():
        txt = p.read_text()
        assert "/Users/" not in txt and str(ROOT) not in txt
    one = json.loads((tmp_path / "mf" / "interpro" / f"{idx['specs'][0]['name']}.json").read_text())
    assert one["out_dir"].startswith("results/go_function/runs/mf/interpro/")
    assert one["mode"] == "study" and one["selection"] == "final" and one["epochs"] == 9
    # a changed study (E*) makes the existing specs differ: fatal, nothing replaced
    with pytest.raises(FatalError):
        M.materialize(_study(mf_interpro=10), "mf", "interpro", tmp_path, LOG)
