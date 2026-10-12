"""Pilot (plan item 14): E* formula on fabricated epochs.jsonl, the import probe,
and the lambda record never changing study.json's lambda."""
import importlib.util
import json
import logging
from pathlib import Path

import pytest

from go_function.constants import EXIT_OK

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_pilot")


def _load():
    spec = importlib.util.spec_from_file_location("pilot_t", ROOT / "scripts" / "go_function" / "pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load()


def _epochs(d: Path, valid, ratios=(0.5, 2.0)):
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"epoch": i, "valid_bce_target_free": v,
                         "grad_norms": [{"batch": 0, "ratio_protein": ratios[0],
                                         "ratio_term": ratios[1]}]}) for i, v in enumerate(valid)]
    (d / "epochs.jsonl").write_text("\n".join(lines) + "\n")
    (d / "DONE.json").write_text("{}\n")


def test_i_best_and_e_star_formula(tmp_path):
    _epochs(tmp_path / "a", [0.5, 0.4, 0.3, 0.35, 0.3])
    assert P.i_best(P.read_epochs(tmp_path / "a")) == 2          # first minimum
    _epochs(tmp_path / "b", [0.5, None])
    with pytest.raises(P.FatalError):
        P.i_best(P.read_epochs(tmp_path / "b"))


def test_e_star_values():
    assert P.e_star([9, 10, 11, 12, 13, 14]) == 13               # median(10..15) = 12.5 -> 13
    assert P.e_star([0, 1, 2, 0, 1, 2]) == 8                     # clipped below
    assert P.e_star([70, 80, 90, 70, 80, 90]) == 64              # clipped above
    assert P.e_star([19, 19, 19, 29, 29, 29]) == 25              # median 25


def test_final_ratios_mean_over_last_epoch(tmp_path):
    _epochs(tmp_path / "a", [0.5, 0.4], ratios=(0.25, 4.0))
    r = P.final_ratios(P.read_epochs(tmp_path / "a"))
    assert r == {"ratio_protein": 0.25, "ratio_term": 4.0}


def test_import_probe_never_loads_the_evaluator():
    mods = P.import_probe()
    assert "go_function.evaluate" not in mods and "go_function.train" not in mods
    P.assert_no_evaluator(mods)
    planted = P.import_probe("import go_function.evaluate")
    with pytest.raises(P.FatalError):
        P.assert_no_evaluator(planted)


def _fake_runner(valid_by_name):
    def runner(spec_path, out_dir, device, log_path):
        spec = json.loads(Path(spec_path).read_text())
        assert spec["mode"] == "pilot" and spec["selection"] == "final"
        assert device == "cpu"
        name = Path(out_dir).name
        vals = valid_by_name(name, spec)
        _epochs(Path(out_dir), vals)
        return 0
    return runner


def test_stages_write_choice_estar_and_pin_without_changing_lambda(tmp_path):
    study = json.loads((ROOT / "config" / "go_function" / "study.json").read_text())
    study["epochs_star"] = {}
    sp = tmp_path / "study.json"
    sp.write_text(json.dumps(study))
    pins = tmp_path / "pins.json"

    def vals(name, spec):
        n = spec["epochs"]
        best = 11 if "s13" in name else 13            # i_best + 1 = 12 or 14 -> median 13
        return [1.0 - 0.01 * min(i, best) + (0.02 if i > best else 0.0) for i in range(n)]

    p = P.Pilot("bp", "esm2_650m", "cpu", LOG, root=tmp_path / "pilots", study_path=sp,
                pins_path=pins, runner=_fake_runner(vals))
    rec = p.stage_estar()
    assert len(rec["runs"]) == 6
    assert rec["E_star"] == P.e_star([r["i_best"] for r in rec["runs"].values()]) == 14
    assert json.loads(sp.read_text())["epochs_star"]["bp_esm2_650m"] == rec["E_star"]
    lam = p.stage_lambda()
    assert len(lam["runs"]) == 16
    assert {r["lambda_el"] for r in lam["runs"].values()} == {0.1, 0.3, 1.0, 3.0}
    assert all(r["epochs"] == rec["E_star"] for r in lam["runs"].values())
    assert json.loads(sp.read_text())["lambda_el"] == study["lambda_el"]   # never changed
    assert p.maybe_pin() is None                                           # text record pending
    txt = p.stage_text()
    assert set(r["variant"] for r in txt["runs"].values()) == {"V5", "V6"}
    assert p.maybe_pin() == "written"
    pin = json.loads(pins.read_text())["pins"]["G7"]
    assert pin["bp_esm2_650m/E_star"] == rec["E_star"]
    for f in (tmp_path / "pilots").rglob("*.json"):
        assert "/Users/" not in f.read_text()
    # a re-run reuses DONE runs and an E* conflict is fatal
    assert p.stage_estar()["E_star"] == rec["E_star"]
    with pytest.raises(P.FatalError):
        p.set_estar(rec["E_star"] + 1)


def test_lambda_stage_refused_without_estar(tmp_path):
    study = json.loads((ROOT / "config" / "go_function" / "study.json").read_text())
    study["epochs_star"] = {}
    sp = tmp_path / "study.json"
    sp.write_text(json.dumps(study))
    p = P.Pilot("bp", "interpro", "cpu", LOG, root=tmp_path / "pilots", study_path=sp,
                runner=_fake_runner(lambda n, s: [0.5] * s["epochs"]))
    with pytest.raises(P.Refused):
        p.stage_lambda()
    assert p.stage_text() is None                     # InterPro has no text arms
