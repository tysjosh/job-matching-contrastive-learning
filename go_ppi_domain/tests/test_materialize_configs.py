"""materialize_configs.py (design 12.1, 14; user decision B path rule).

``CONFIG_DIR`` is monkeypatched to ``tmp_path`` and the pins file is redirected,
so nothing is written into ``config/go_withholding/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from go_ppi_domain import withholding_constants as wc
from scripts.go_withholding import materialize_configs as mc

CELL_FILES = {"baseline.json", "loss_only.json", "neg_only.json", "both.json"}


@pytest.fixture()
def cfgdir(tmp_path, monkeypatch):
    d = tmp_path / "config_go_withholding"
    monkeypatch.setattr(wc, "CONFIG_DIR", d)
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    return d


def _load(d: Path, name: str) -> dict:
    return json.loads((d / name).read_text())


def test_lambda_float_writes_four_configs_and_one_pin(cfgdir):
    assert mc.main(["--lambda", "0.3"]) == 0
    names = {p.name for p in cfgdir.iterdir()}
    assert names == CELL_FILES | {"configs.sha256.json"}
    cfgs = {n: _load(cfgdir, n) for n in CELL_FILES}
    for name, cfg in cfgs.items():
        cell = name[:-5]
        assert cfg["go_ppi_unannotated_anchor_window"] is True
        assert cfg["_cell"] == cell and cfg["_study"] == "go_withholding_v4"
        assert "validation_path" not in cfg
        assert cfg["checkpoint_frequency"] == 1000000000
        assert cfg["go_similarity_mode"] == "simgic"
        assert cfg["go_aspect_weights"] == {"P": 1.0, "F": 1.0, "C": 1.0}
        assert cfg["go_ontology_loss_weight"] == (0.3 if cell in ("loss_only", "both") else 0.0)
        assert cfg["go_ppi_tier_sampling"] == ("stochastic" if cell in ("neg_only", "both")
                                               else "random_window")
    sha = _load(cfgdir, "configs.sha256.json")
    assert set(sha) == CELL_FILES
    assert all(sha[n] == wc.sha256_file(cfgdir / n) for n in CELL_FILES)
    pins = wc.read_pins()
    assert [p["gate"] for p in pins] == ["G4-configs"]
    assert set(pins[0]["files"]) == CELL_FILES and pins[0]["lambda"] == 0.3
    assert pins[0]["configs_sha256_json"] == wc.sha256_file(cfgdir / "configs.sha256.json")
    assert not (cfgdir / ".staging").exists()
    # a re-run (e.g. after an amendment) appends a second entry
    assert mc.main(["--lambda", "0.5"]) == 0
    assert [p["gate"] for p in wc.read_pins()] == ["G4-configs", "G4-configs"]


def test_lambda_none_writes_no_pin(cfgdir):
    assert mc.main(["--lambda", "none"]) == 0
    assert {p.name for p in cfgdir.iterdir()} == {"baseline.json", "neg_only.json",
                                                  "loss_pilot_template.json"}
    tpl = _load(cfgdir, "loss_pilot_template.json")
    assert tpl["go_ontology_loss_weight"] == 0.0 and tpl["_cell"] == "loss_only"
    assert wc.read_pins() == []


def test_paths_are_root_anchored_and_expand(cfgdir):
    assert mc.main(["--lambda", "0.3"]) == 0
    for name in CELL_FILES:
        raw = _load(cfgdir, name)
        text = (cfgdir / name).read_text()
        assert str(wc.ROOT) not in text                    # no machine-specific path
        for k, v in raw.items():
            if wc.is_path_key(k) and isinstance(v, str):
                assert v.startswith("$ROOT/"), (k, v)
        cfg = wc.load_study_config(cfgdir / name)
        assert cfg.go_ppi_split_dir == str(wc.ROOT / "preprocess/go_withholding/conditions/all")
        assert cfg.go_index_cache == str(wc.ROOT /
                                         "preprocess/go_withholding/converted/go_index_P.pkl")
        assert cfg.embedding_cache_path == str(wc.EMBED_CACHE)
        assert cfg.go_ppi_unannotated_anchor_window is True
        assert Path(cfg.go_obo_path).is_absolute()


def _doctor(monkeypatch, fn):
    real = mc.build_candidates

    def doctored(lam):
        c = real(lam)
        fn(c)
        return c
    monkeypatch.setattr(mc, "build_candidates", doctored)


@pytest.mark.parametrize("mutate", [
    lambda c: c["loss_only"].__setitem__("go_ppi_go_tiered_negatives", True),
    lambda c: c["neg_only"].__setitem__("go_ppi_unannotated_anchor_window", False),
    lambda c: c["both"].__setitem__("go_ppi_tier_window_frac", 0.5),
    lambda c: c["baseline"].__setitem__("learning_rate", 1e-3),
])
def test_factor_audit_refuses_and_writes_nothing(cfgdir, monkeypatch, mutate):
    _doctor(monkeypatch, mutate)
    assert mc.main(["--lambda", "0.3"]) == 2
    assert not cfgdir.exists() or not any(p.suffix == ".json" for p in cfgdir.iterdir())
    assert not (cfgdir / ".staging").exists()
    assert wc.read_pins() == []


def test_exact_diff_sets_match_design():
    assert mc.EXPECTED_PAIR_DIFF == {
        ("baseline", "loss_only"): {"go_ontology_loss_weight"},
        ("neg_only", "both"): {"go_ontology_loss_weight"},
        ("baseline", "neg_only"): {"go_ppi_go_tiered_negatives", "go_ppi_tier_sampling"},
        ("loss_only", "both"): {"go_ppi_go_tiered_negatives", "go_ppi_tier_sampling"},
    }


@pytest.mark.parametrize("value", ["preprocess/go_withholding/converted",
                                   "/abs/preprocess/go_withholding/converted"])
def test_non_root_anchored_path_refused(cfgdir, monkeypatch, value):
    _doctor(monkeypatch, lambda c: [cfg.__setitem__("go_ppi_converted_dir", value)
                                    for cfg in c.values()])
    assert mc.main(["--lambda", "0.3"]) == 2
    assert not cfgdir.exists() or not any(cfgdir.iterdir())


@pytest.mark.parametrize("bad", ["0", "-0.1", "nan", "inf"])
def test_bad_lambda_refused(cfgdir, bad):
    assert mc.main(["--lambda", bad]) == 2
    assert wc.read_pins() == []
