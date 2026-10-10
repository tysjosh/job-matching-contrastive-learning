"""lambda_pilot.py (design 12.2, 14): rounding, the confirm / re-estimate state
machine, the < 4 batches stop, the epoch assertion and the end-to-end wiring with
the runner subprocess and the measurement stubbed. One real (fake-encoder)
measurement exercises the strict checkpoint load and the gradient recipe.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.tests import _study_fixtures as sf
from scripts import run_learning_curve_point as runner
from scripts.go_withholding import lambda_pilot as lp
from scripts.go_withholding import materialize_configs as mc

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ rounding
@pytest.mark.parametrize("x, expected", [
    (0.1, 0.1), (0.14, 0.1), (0.15, 0.2), (0.24, 0.2), (0.25, 0.3), (0.4, 0.5),
    (0.7, 0.5), (0.75, 1.0), (7.0, 5.0), (8.0, 10.0), (3e-5, 3e-05), (0.0007, 0.0005),
    (2.2, 2.0), (123.0, 100.0),
])
def test_round_lambda_log_space(x, expected):
    assert lp.round_lambda(x) == expected
    assert repr(lp.round_lambda(x)) == repr(expected)        # no 0.30000000000000004


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_round_lambda_rejects(bad):
    with pytest.raises(ValueError):
        lp.round_lambda(bad)


# ------------------------------------------------------------- state machine
def _measure(table):
    calls = []

    def measure(lam):
        calls.append(lam)
        return {"r": table[lam]}
    return measure, calls


def test_choose_confirms_first_estimate():
    measure, calls = _measure({0.1: 0.05, 0.5: 0.22})
    out = lp.choose_lambda(0.1, measure)
    assert out["lambda"] == 0.5 and out["r0"] == 0.05 and out["r_confirm"] == 0.22
    assert calls == [0.1, 0.5]
    assert [a["role"] for a in out["attempts"]] == ["pilot", "confirm"]


def test_choose_reuses_identical_lambda_run():
    measure, calls = _measure({0.1: 0.2})                # lambda* == lambda0
    out = lp.choose_lambda(0.1, measure)
    assert out["lambda"] == 0.1 and calls == [0.1]       # trained once, confirmed from it


def test_choose_one_re_estimate_then_confirm():
    measure, calls = _measure({0.1: 0.05, 0.5: 0.5, 0.2: 0.18})
    out = lp.choose_lambda(0.1, measure)
    assert out["lambda"] == 0.2 and calls == [0.1, 0.5, 0.2]
    assert [a["role"] for a in out["attempts"]] == ["pilot", "confirm", "re-estimate"]


def test_choose_stops_after_failed_re_estimate():
    measure, calls = _measure({0.1: 0.05, 0.5: 0.5, 0.2: 0.6})
    with pytest.raises(lp.PilotStop, match="not confirmed"):
        lp.choose_lambda(0.1, measure)
    assert calls == [0.1, 0.5, 0.2]                      # exactly one re-estimate


@pytest.mark.parametrize("r0", [0.0, float("nan")])
def test_choose_stops_on_unusable_ratio(r0):
    measure, _ = _measure({0.1: r0})
    with pytest.raises(lp.PilotStop):
        lp.choose_lambda(0.1, measure)


# ------------------------------------------------------------ batches, epoch
def test_make_batches_drops_partial_and_needs_four():
    assert [len(b) for b in lp.make_batches(list(range(130)))] == [32] * 4
    assert lp.make_batches(list(range(130)))[3][-1] == 127
    with pytest.raises(lp.PilotStop, match="too few target-free validation batches"):
        lp.make_batches(list(range(127)))                # 3 full batches


def test_set_selector_epoch_asserts():
    calls = []

    class BP:
        current_epoch = 0

        def set_epoch(self, e):
            self.current_epoch = e

    trainer = SimpleNamespace(batch_processor=BP(),
                              loss_engine=SimpleNamespace(set_epoch=calls.append))
    assert lp.set_selector_epoch(trainer, 2) == 2 and calls == [2]
    trainer.batch_processor.set_epoch = lambda e: None   # does not move the curriculum
    trainer.batch_processor.current_epoch = 0
    with pytest.raises(RuntimeError, match="current_epoch 0 != 2"):
        lp.set_selector_epoch(trainer, 2)


def test_pilot_command_is_an_argument_list():
    cmd = lp.pilot_command(Path("/c/config.json"), Path("/cond/all"), Path("/out/l0.1"),
                           device="cuda")
    assert cmd[1] == str(ROOT / "scripts" / "run_learning_curve_point.py")
    for flag, value in (("--study-cell", "loss_only"), ("--seed", "42"), ("--epochs", "3"),
                        ("--withholding-manifest", "/cond/all/withholding_manifest.json"),
                        ("--device", "cuda")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--pilot" in cmd and all(isinstance(c, str) for c in cmd)


# ----------------------------------------------------------- end to end (stubs)
@pytest.fixture()
def pilot_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    cond = tmp_path / "conditions" / "all"
    cond.mkdir(parents=True)
    (cond / "withholding_manifest.json").write_text("{}")
    tpl = tmp_path / "loss_pilot_template.json"
    tpl.write_text(json.dumps({"go_ontology_loss_weight": 0.0, "_cell": "loss_only"}))
    vpos = tmp_path / "validation_positive.jsonl"
    vpos.write_text("")
    ratios = {0.1: 0.05, 0.5: 0.21}
    runs, mats = [], []

    def fake_run(lam, template, cond_dir, out, device):
        d = out / lp.lambda_label(lam)
        lp.write_pilot_config(template, d, lam)
        runs.append((lam, device))
        return d

    def fake_measure(run_dir, cond_dir, validation_positive, lam, device=None):
        return {"r": ratios[lam], "ratios_val_per_batch": [ratios[lam]] * 5, "n_records": 181,
                "n_batches": 5, "selector_epoch": 2}
    monkeypatch.setattr(lp, "run_pilot", fake_run)
    monkeypatch.setattr(lp, "measure_validation", fake_measure)
    monkeypatch.setattr(mc, "main", lambda argv: mats.append(argv) or 0)
    out = tmp_path / "lambda_pilot"
    argv = ["--out", str(out), "--template", str(tpl), "--condition", str(cond),
            "--validation-positive", str(vpos), "--device", "cpu"]
    return SimpleNamespace(out=out, argv=argv, runs=runs, mats=mats, ratios=ratios)


def test_main_writes_choice_pin_and_materializes(pilot_env):
    assert lp.main(pilot_env.argv) == 0
    choice = json.loads((pilot_env.out / "lambda_choice.json").read_text())
    assert choice["lambda"] == 0.5 and choice["lambda0"] == 0.1
    assert choice["r0"] == 0.05 and choice["r_confirm"] == 0.21
    assert (choice["batch_size"], choice["seed"], choice["epochs"],
            choice["selector_epoch"]) == (32, 42, 3, 2)
    assert set(choice["ratios_val_per_batch"]) == {"0.1", "0.5"}
    assert len(choice["pilot_run_dirs"]) == 2 and choice["code_sha256"]
    assert pilot_env.runs == [(0.1, "cpu"), (0.5, "cpu")]
    cfg = json.loads((pilot_env.out / "l0.1" / "config.json").read_text())
    assert cfg["go_ontology_loss_weight"] == 0.1
    pins = wc.read_pins()
    assert [p["gate"] for p in pins] == ["G4-lambda"] and pins[0]["lambda"] == 0.5
    assert pins[0]["lambda_choice_sha256"] == wc.sha256_file(pilot_env.out / "lambda_choice.json")
    assert pilot_env.mats == [["--lambda", "0.5"]]
    assert lp.main(pilot_env.argv) == 2                  # refuses to overwrite


def test_main_exit_3_on_protocol_stops(pilot_env, monkeypatch):
    def too_few(*a, **k):
        raise lp.PilotStop("too few target-free validation batches: 3")
    monkeypatch.setattr(lp, "measure_validation", too_few)
    assert lp.main(pilot_env.argv) == 3
    assert not (pilot_env.out / "lambda_choice.json").exists() and wc.read_pins() == []


def test_main_exit_3_when_not_confirmed(pilot_env):
    pilot_env.ratios.update({0.5: 0.5, 0.2: 0.9})
    assert lp.main(pilot_env.argv) == 3
    assert pilot_env.mats == [] and wc.read_pins() == []


# --------------------------------------------------- real measurement (fake MPNet)
def test_measure_validation_on_a_real_pilot_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("contrastive_learning.trainer.SentenceTransformer",
                        sf.fake_encoder_class())
    s = sf.build_study(tmp_path / "study", conds=("all",))
    sf.patch_study(monkeypatch, s)
    (s["results"] / "lambda_pilot" / "lambda_choice.json").unlink()
    run_dir = tmp_path / "pilot" / "l0.1"
    cfg = lp.write_pilot_config(s["config_dir"] / "loss_only.json", run_dir, 0.1)
    rc = runner.main(sf.runner_args(s, "loss_only", run_dir, config=cfg, seed=42,
                                    extra=["--pilot", "--epochs", "3"]))
    assert rc == 0, (run_dir / "run_error.json").read_text()
    assert (run_dir / "checkpoint_epoch_2.pt").is_file()
    vpos = s["val_dir"] / "validation_positive.jsonl"
    with pytest.raises(lp.PilotStop, match="too few"):     # 4 records < 4 x 32
        lp.measure_validation(run_dir, s["conditions"] / "all", vpos, 0.1)
    monkeypatch.setattr(lp, "BATCH_SIZE", 2)
    monkeypatch.setattr(lp, "MIN_BATCHES", 2)
    m = lp.measure_validation(run_dir, s["conditions"] / "all", vpos, 0.1)
    assert m["selector_epoch"] == 2 and m["n_batches"] == 2 and m["n_records"] == 4
    assert m["r"] > 0 and len(m["ratios_val_per_batch"]) == 2
    assert lp.train_diag_ratios(run_dir).keys() == {"0", "1", "2"}
    # strict load: a checkpoint with a missing key is refused
    import torch
    ck = torch.load(run_dir / "checkpoint_epoch_2.pt", weights_only=False)
    ck["model_state_dict"].pop(next(iter(ck["model_state_dict"])))
    torch.save(ck, run_dir / "checkpoint_epoch_2.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        lp.measure_validation(run_dir, s["conditions"] / "all", vpos, 0.1)
