"""Runner study path: CLI rules, preflight, condition and post-training checks,
lc_manifest blocks and the --device flag (design 4.1-4.4, 4.7, 14; user decision B).

In-process runs use the real runner, trainer, selector, loss engine and
callbacks on the synthetic study layout of ``_study_fixtures`` with a tiny
deterministic stand-in for the frozen SentenceTransformer (no MPNet).
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from go_ppi_domain import withholding as wh
from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.go_ontology import GoIndex
from go_ppi_domain.tests import _study_fixtures as sf
from go_ppi_domain.tests._withholding_fixtures import write_identity_fixture
from go_ppi_domain.tests.test_withholding_generator import _rec, _stub_trainer
from scripts import run_learning_curve_point as runner

ROOT = Path(__file__).resolve().parents[2]
LC_KEYS = {"domain", "config", "train_file", "train_records", "fraction", "seed", "epochs",
           "single_factor_audit", "attach", "train_result"}
STUDY_KEYS = LC_KEYS | {"withholding", "epoch_stats", "negative_selection", "go_relation_loss",
                        "projection_params", "code", "pilot", "device"}


@pytest.fixture()
def fake_encoder(monkeypatch):
    monkeypatch.setattr("contrastive_learning.trainer.SentenceTransformer",
                        sf.fake_encoder_class())


@pytest.fixture()
def study(tmp_path, monkeypatch):
    s = sf.build_study(tmp_path / "study")
    sf.patch_study(monkeypatch, s)
    return s


def _error(out: Path) -> dict:
    return json.loads((out / "run_error.json").read_text())


def _write_cfg(s, name, cell="baseline", **over) -> Path:
    path = s["config_dir"] / f"{name}.json"
    path.write_text(json.dumps(sf.cell_config(s, cell, **over), indent=2))
    return path


# ----------------------------------------------------------------- CLI rules
BASE_ARGV = ["--domain", "go_ppi", "--config", "/x.json", "--train-file", "/t.jsonl",
             "--output-dir", "/o", "--seed", "13"]
MANIFEST_ARGV = ["--withholding-manifest", "/m.json"]


@pytest.mark.parametrize("extra", [
    ["--pilot"],                                                      # pilot needs manifest
    ["--study-cell", "baseline"],
    ["--go-diagnostics", "/d.jsonl"],
    MANIFEST_ARGV + ["--go-diagnostics", "/d.jsonl"],                 # no --study-cell
    MANIFEST_ARGV + ["--study-cell", "baseline"],                     # no --go-diagnostics
    MANIFEST_ARGV + ["--study-cell", "baseline", "--go-diagnostics", "/d.jsonl",
                     "--validation-file", "/v.jsonl"],                # forbidden
    ["--study-cell", "nonsense"],
    ["--device", "mps"],
])
def test_cli_rules_are_argparse_errors(extra):
    with pytest.raises(SystemExit) as e:
        runner.main(BASE_ARGV + extra)
    assert e.value.code == 2


def test_cli_manifest_requires_go_ppi_domain():
    argv = ["--domain", "career"] + BASE_ARGV[2:] + MANIFEST_ARGV + [
        "--study-cell", "baseline", "--go-diagnostics", "/d.jsonl"]
    with pytest.raises(SystemExit) as e:
        runner.main(argv)
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        runner.main(["--domain", "career"] + BASE_ARGV[2:] + ["--go-ppi-split-dir", "/s"])


# ------------------------------------------------------------------ preflight
class _Spy:
    def __init__(self):
        self.calls = []


@pytest.fixture()
def no_build(monkeypatch):
    """Record (and refuse) any trainer construction or matcher build."""
    spy = _Spy()

    def trainer_init(self, *a, **k):
        spy.calls.append("trainer")
        raise AssertionError("trainer must not be built")

    def matcher(*a, **k):
        spy.calls.append("matcher")
        raise AssertionError("matcher must not be built")
    monkeypatch.setattr("contrastive_learning.trainer.ContrastiveLearningTrainer.__init__",
                        trainer_init)
    monkeypatch.setattr("go_ppi_domain.run_config.build_go_matcher", matcher)
    return spy


def _run_expect_preflight_error(s, tmp_path, cell, cfg, match, no_build, extra=()):
    out = tmp_path / "out"
    rc = runner.main(sf.runner_args(s, cell, out, config=cfg, extra=extra))
    err = _error(out)
    assert rc == 2 and err["stage"] == "preflight", err
    assert match in err["error"], err["error"]
    assert err["cell"] == cell and err["seed"] == 13
    assert no_build.calls == []
    assert not (out / "lc_manifest.json").exists()
    return err


def test_relative_path_key_in_json_rejected(study, tmp_path, no_build):
    cfg = _write_cfg(study, "rel", embedding_cache_path="embedding_cache/x.pt")
    _run_expect_preflight_error(study, tmp_path, "baseline", cfg,
                                "embedding_cache_path='embedding_cache/x.pt'", no_build)


def test_relative_cli_override_rejected(study, tmp_path, no_build):
    out = tmp_path / "out"
    argv = sf.runner_args(study, "baseline", out)
    argv[argv.index("--go-diagnostics") + 1] = "relative/diag.jsonl"
    assert runner.main(argv) == 2
    assert "--go-diagnostics" in _error(out)["error"] and no_build.calls == []


def test_absent_relative_trainingconfig_defaults_are_ignored(study, tmp_path):
    raw = json.loads((study["config_dir"] / "baseline.json").read_text())
    from contrastive_learning.data_structures import TrainingConfig
    rel_defaults = [f.name for f in dataclasses.fields(TrainingConfig)
                    if wc.is_path_key(f.name) and isinstance(f.default, str)
                    and f.default and not Path(f.default).is_absolute()
                    and f.name not in raw]
    assert rel_defaults, "fixture should leave some relative TrainingConfig path defaults"
    assert wc.config_path_problems(raw) == []
    args = runner.argparse.Namespace(
        go_ppi_split_dir=study["conditions"] / "all",
        train_file=study["conditions"] / "all" / "train.jsonl", output_dir=tmp_path / "o",
        withholding_manifest=study["conditions"] / "all" / wh.MANIFEST,
        go_diagnostics=tmp_path / "o" / "d.jsonl", seed=13, epochs=None,
        validation_file=None, pilot=False, study_cell="baseline", device=None)
    pre = runner.study_preflight(args, raw)
    assert pre["lambda_check"] == "ok" and pre["device_resolved"] in ("cpu", "cuda")
    # root-anchored values are accepted and expanded against ROOT
    raw2 = dict(raw, embedding_cache_path="$ROOT/embedding_cache/x.pt")
    pre2 = runner.study_preflight(args, raw2)
    assert pre2["config"].embedding_cache_path == str(wc.ROOT / "embedding_cache/x.pt")


def test_index_sha_mismatch_rejected_before_any_matcher(study, tmp_path, no_build):
    frozen_path = study["splits"] / "frozen_hashes.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["go_index_P_sha256"] = "0" * 64
    frozen_path.write_text(json.dumps(frozen))
    _run_expect_preflight_error(study, tmp_path, "neg_only",
                                study["config_dir"] / "neg_only.json",
                                "preflight 2: go_index_cache sha256", no_build)


def test_unpruned_index_rejected(study, tmp_path, no_build):
    idx = study["index"]
    bad = GoIndex(idx.ancestors, idx.depth, idx.direct, idx.closure, idx.coarse, idx.ic,
                  {"aspect": "P"})
    path = study["root"] / "unpruned" / "go_index_P.pkl"
    bad.save(path)
    frozen_path = study["splits"] / "frozen_hashes.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["go_index_P_sha256"] = wc.sha256_file(path)
    frozen_path.write_text(json.dumps(frozen))
    cfg = _write_cfg(study, "unpruned", go_index_cache=str(path))
    _run_expect_preflight_error(study, tmp_path, "baseline", cfg, "not leak-pruned", no_build)


def test_validation_path_rejected(study, tmp_path, no_build):
    cfg = _write_cfg(study, "val", validation_path=str(study["splits"] / "validation.jsonl"))
    _run_expect_preflight_error(study, tmp_path, "baseline", cfg, "preflight 3", no_build)


def test_lambda_equality_and_pilot_rules(study, tmp_path, no_build):
    cfg = _write_cfg(study, "wrong_lam", "loss_only", go_ontology_loss_weight=0.2)
    _run_expect_preflight_error(study, tmp_path, "loss_only", cfg,
                                "preflight 4: go_ontology_loss_weight 0.2 != 0.3", no_build)
    cfg = _write_cfg(study, "base_lam", "baseline", go_ontology_loss_weight=0.3)
    shutil.rmtree(tmp_path / "out")
    _run_expect_preflight_error(study, tmp_path, "baseline", cfg,
                                "preflight 4: go_ontology_loss_weight 0.3 != 0.0", no_build)
    (study["results"] / "lambda_pilot" / "lambda_choice.json").unlink()
    shutil.rmtree(tmp_path / "out")
    _run_expect_preflight_error(study, tmp_path, "loss_only",
                                study["config_dir"] / "loss_only.json",
                                "lambda_choice.json does not exist", no_build)


def test_pilot_skips_only_lambda_and_writes_pilot_true(study, tmp_path, fake_encoder):
    (study["results"] / "lambda_pilot" / "lambda_choice.json").unlink()
    cfg = _write_cfg(study, "pilot", "loss_only", go_ontology_loss_weight=0.1)
    out = tmp_path / "pilot"
    rc = runner.main(sf.runner_args(study, "loss_only", out, config=cfg, extra=["--pilot"]))
    assert rc == 0, _error(out)
    lc = json.loads((out / "lc_manifest.json").read_text())
    assert lc["pilot"] is True
    assert lc["withholding"]["assertions"]["preflight_lambda"] == "skipped (--pilot)"
    assert lc["go_relation_loss"]["weight"] == 0.1
    # --pilot does not relax anything else: a neg_only-wired config still fails check 7
    bad = _write_cfg(study, "pilot_bad", "neg_only", go_ontology_loss_weight=0.1)
    out2 = tmp_path / "pilot_bad"
    assert runner.main(sf.runner_args(study, "loss_only", out2, config=bad,
                                      extra=["--pilot"])) == 2
    assert _error(out2)["stage"] == "condition_assertions"


# ----------------------------------------------------- condition checks (4.3)
def _doctored_dir(s, cond, mutate_manifest=None, extra_record=None):
    src = s["conditions"] / cond
    dst = s["root"] / f"doctored_{cond}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    shutil.copy(src / "negative_pools.jsonl", dst / "negative_pools.jsonl")
    data = (src / "train.jsonl").read_bytes()
    if extra_record is not None:
        data += (json.dumps(extra_record) + "\n").encode()
    (dst / "train.jsonl").write_bytes(data)
    man = json.loads((src / wh.MANIFEST).read_text())
    man["files"]["train.jsonl"]["sha256"] = wc.sha256_file(dst / "train.jsonl")
    if mutate_manifest:
        mutate_manifest(man)
    (dst / wh.MANIFEST).write_text(json.dumps(man))
    return dst


def _table_stub(ids, pruned=True):
    meta = {"aspect": "P", "leak_pruning": {"with_from_human_protein": True}} if pruned else {}
    return SimpleNamespace(ids=list(ids), source_meta=meta)


def _check(s, case):
    """(trainer, config, manifest_path, train_file, cell, frozen) for a doctoring case."""
    cond = s["conditions"] / "p01_k5"
    man = json.loads((cond / wh.MANIFEST).read_text())
    W = man["withheld_proteins"]
    trainer, config = _stub_trainer(s, cond)
    sel = trainer.batch_processor.domain_negative_selector
    mpath, tfile, cell, frozen = cond / wh.MANIFEST, cond / "train.jsonl", "baseline", s["frozen"]
    if case == "1_train_sha":
        tfile = s["splits"] / "train.jsonl"
    elif case == "1_split_dir":
        config.go_ppi_split_dir = str(s["conditions"] / "all")
    elif case == "1_converted_dir":
        config.go_ppi_converted_dir = str(s["splits"])
    elif case == "2":
        d = _doctored_dir(s, "p01_k5", extra_record=_rec("h0", W[0]))
        mpath, tfile = d / wh.MANIFEST, d / "train.jsonl"
        config.go_ppi_split_dir = str(d)
    elif case == "3_members":
        next(iter(sel.pools.values()))["weak_evidence"].append(W[0])
    elif case == "4":
        val_anchor = sorted(a for a in sel.pools if a.startswith("v"))[0]
        sel.pools[val_anchor]["no_interaction"] = []
    elif case == "5":
        def drop(m):
            m["targets"][0]["retained"] = m["targets"][0]["retained"][1:]
        d = _doctored_dir(s, "p01_k5", mutate_manifest=drop)
        mpath, tfile = d / wh.MANIFEST, d / "train.jsonl"
        config.go_ppi_split_dir = str(d)
    elif case == "6":
        frozen = dict(frozen, test_sha256="0" * 64)
    elif case == "7_override":
        trainer.batch_processor.ontology_matcher_override = object()
    elif case == "7_flag":
        config.go_ppi_unannotated_anchor_window = False
    elif case in ("8", "9"):
        cell = "loss_only"
        config.go_ontology_loss_weight = 0.3
        ids = sorted({p for l in tfile.read_text().splitlines()
                      for p in (json.loads(l)["resume"]["protein_id"],
                                json.loads(l)["job"]["partner_id"])})
        trainer.loss_engine.go_relation_table = (
            _table_stub(ids, pruned=False) if case == "8" else _table_stub(ids + [W[0]]))
    elif case == "10":
        trainer.epoch_end_callbacks = trainer.epoch_end_callbacks[1:]
    return trainer, config, mpath, tfile, cell, frozen


@pytest.mark.parametrize("case", ["1_train_sha", "1_split_dir", "1_converted_dir", "2",
                                  "3_members", "4", "5", "6", "7_override", "7_flag", "8",
                                  "9", "10"])
def test_every_condition_check_fires(study, case):
    ok = wh.assert_condition(*_check(study, "none"))
    assert all(ok[f"{i}_{n}"] for i, n in [(1, "hashes"), (10, "callbacks")])
    check = case.split("_")[0]
    with pytest.raises(RuntimeError, match=rf"check {check}\b"):
        wh.assert_condition(*_check(study, case))


# ------------------------------------------------- post-training checks (4.4)
def _post_env(tmp_path, monkeypatch, epochs=2):
    out = tmp_path / "run"
    out.mkdir()
    torch.save({"epoch": epochs - 1}, out / f"checkpoint_epoch_{epochs - 1}.pt")
    monkeypatch.setattr(wc, "imported_repo_modules", lambda: ["go_ppi_domain/a.py"])
    trainer = SimpleNamespace(output_dir=out, loss_engine=SimpleNamespace(
        go_relation_seen_ids={"P1"}, go_relation_missing_ids=0))
    return (trainer, SimpleNamespace(num_epochs=epochs), {"withheld_proteins": ["W1"]},
            SimpleNamespace(none_returns=0, exceptions=0),
            SimpleNamespace(epochs=[{}] * epochs))


def test_post_training_happy_and_each_check(tmp_path, monkeypatch):
    trainer, config, manifest, rec, guard = _post_env(tmp_path, monkeypatch)
    files = ["go_ppi_domain/a.py"]
    post = runner.assert_post_training(trainer, config, manifest, rec, guard, code_files=files)
    assert post["selected_checkpoint"] == "checkpoint_epoch_1.pt"
    assert post["selected_checkpoint_sha256"] == wc.sha256_file(
        trainer.output_dir / "checkpoint_epoch_1.pt")
    assert set(post["checks"].values()) == {"ok"} and len(post["checks"]) == 7
    cases = [
        (lambda: trainer.loss_engine.go_relation_seen_ids.add("W1"), "saw withheld"),
        (lambda: setattr(trainer.loss_engine, "go_relation_missing_ids", 2), "missing_ids == 2"),
        (lambda: setattr(rec, "none_returns", 1), "returned None"),
        (lambda: setattr(rec, "exceptions", 1), "raised 1 times"),
        (lambda: setattr(guard, "epochs", [{}]), "saw 1 of 2 epochs"),
        (lambda: monkeypatch.setattr(wc, "imported_repo_modules",
                                     lambda: ["go_ppi_domain/a.py", "orca/x.py"]),
         "outside the code pin: ['orca/x.py']"),
        (lambda: (trainer.output_dir / "checkpoint_epoch_1.pt").unlink(), "missing"),
        (lambda: torch.save({"epoch": 0}, trainer.output_dir / "checkpoint_epoch_1.pt"),
         "records epoch 0"),
        (lambda: torch.save({}, trainer.output_dir / "checkpoint_epoch_0_emergency.pt"),
         "emergency checkpoint"),
    ]
    for mutate, match in cases:
        t2 = tmp_path / match.replace(" ", "_").replace("'", "").replace("/", "")[:30]
        t2.mkdir()
        mp = pytest.MonkeyPatch()
        trainer, config, manifest, rec, guard = _post_env(t2, mp)
        try:
            mutate()
            with pytest.raises(RuntimeError, match="post-training"):
                try:
                    runner.assert_post_training(trainer, config, manifest, rec, guard,
                                                code_files=files)
                except RuntimeError as exc:
                    assert match in str(exc), (match, str(exc))
                    raise
        finally:
            mp.undo()


# ---------------------------------------------------------- full study runs
def test_study_run_writes_every_manifest_block(study, tmp_path, fake_encoder):
    out = tmp_path / "loss_only"
    rc = runner.main(sf.runner_args(study, "loss_only", out, cond="p01_k5"))
    assert rc == 0, _error(out)
    lc = json.loads((out / "lc_manifest.json").read_text())
    assert set(lc) == STUDY_KEYS
    w = lc["withholding"]
    assert (w["cond_id"], w["pack_id"], w["k"], w["cell"]) == ("p01_k5", "p01", 5, "loss_only")
    assert w["manifest_sha256"] == wc.sha256_file(study["conditions"] / "p01_k5" / wh.MANIFEST)
    assert w["frozen_hashes_sha256"] == wc.sha256_file(study["splits"] / "frozen_hashes.json")
    assert w["selected_checkpoint"] == "checkpoint_epoch_0.pt"
    assert w["selected_checkpoint_sha256"] == wc.sha256_file(out / "checkpoint_epoch_0.pt")
    a = w["assertions"]
    for key in ("1_hashes", "4_validation_pools", "5_exactly_k", "7_wiring", "9_relation_table",
                "10_callbacks", "preflight_lambda", "post_import_coverage",
                "post_final_checkpoint"):
        assert a[key] == "ok", key
    assert a["counts"]["withheld_count"] == study["manifests"]["p01_k5"]["withheld_count"]
    assert lc["epochs"] == 1 and lc["pilot"] is False
    assert lc["epoch_stats"][0]["go_relation_applied"] == lc["epoch_stats"][0]["ok_batches"] > 0
    ns = lc["negative_selection"]
    assert set(ns) == {"selection_sha256_by_epoch", "calls_by_epoch",
                       "unannotated_anchor_calls_by_epoch", "unannotated_anchor_window",
                       "selector_none_returns", "selector_exceptions"}
    assert ns["unannotated_anchor_window"] is True and ns["selector_exceptions"] == 0
    assert set(lc["go_relation_loss"]) == {"weight", "table_proteins", "table_sha256",
                                           "seen_ids", "missing_ids", "skipped_batches",
                                           "applied_batches"}
    assert lc["go_relation_loss"]["missing_ids"] == 0 and lc["go_relation_loss"]["weight"] == 0.3
    assert set(lc["code"]) == {"git_head", "code_sha256", "files", "imported"}
    assert "scripts/run_learning_curve_point.py" in lc["code"]["imported"]
    assert lc["projection_params"]["numel"] > 0
    assert lc["device"]["resolved"] == "cpu" and lc["device"]["requested"] == "auto"
    text = (out / "lc_manifest.json").read_text()
    assert str(ROOT) not in text                       # user decision B: no machine path
    assert (out / "go_diagnostics.jsonl").is_file() and not (out / "run_error.json").exists()


def test_forced_go_relation_failure_exits_2(study, tmp_path, fake_encoder, monkeypatch):
    from go_ppi_domain import withholding_diagnostics as wd

    original = wd.EpochDiagnostics.__call__

    def detach_after_first_epoch(self, trainer, epoch, stats):
        original(self, trainer, epoch, stats)
        trainer.loss_engine.go_relation_table = None        # L_GO now raises per batch
    monkeypatch.setattr(wd.EpochDiagnostics, "__call__", detach_after_first_epoch)
    out = tmp_path / "forced"
    rc = runner.main(sf.runner_args(study, "loss_only", out, extra=["--epochs", "2"]))
    err = _error(out)
    assert rc == 2 and err["stage"] == "train"
    assert "StrictEpochGuard[loss_only] epoch 1" in err["error"]
    assert err["cond_id"] == "all" and not (out / "lc_manifest.json").exists()
    assert str(ROOT) not in (out / "run_error.json").read_text()


# --------------------------------------------------------------- device flag
def test_resolve_device_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runner.resolve_device("auto") == "cpu"
    assert runner.resolve_device(None) == "cpu"
    assert runner.resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="cuda requested"):
        runner.resolve_device("cuda")
    with pytest.raises(ValueError):
        runner.resolve_device("mps")


def test_device_scope_hides_cuda_only_while_building(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with runner.device_scope("cpu"):
        assert torch.cuda.is_available() is False
    assert torch.cuda.is_available() is True
    fn = torch.cuda.is_available
    with runner.device_scope("cuda"):
        assert torch.cuda.is_available is fn
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fn = torch.cuda.is_available
    with runner.device_scope("cpu"):                       # CPU machine: a no-op
        assert torch.cuda.is_available is fn


def test_device_record_detects_wrong_component():
    trainer = SimpleNamespace(device=torch.device("cpu"),
                              loss_engine=SimpleNamespace(device=torch.device("cuda")),
                              text_encoder=torch.nn.Linear(2, 2), model=torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="loss_engine"):
        runner.device_record("cpu", "cpu", trainer)
    trainer.loss_engine.device = torch.device("cpu")
    rec = runner.device_record("auto", "cpu", trainer)
    assert rec["components"] == {"trainer": "cpu", "loss_engine": "cpu",
                                 "text_encoder": "cpu", "projection": "cpu"}


def test_study_device_from_config_and_cli(study, tmp_path, fake_encoder, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _write_cfg(study, "devcpu", "baseline", training_device="cpu")
    out = tmp_path / "devcpu"
    assert runner.main(sf.runner_args(study, "baseline", out, config=cfg)) == 0, _error(out)
    dev = json.loads((out / "lc_manifest.json").read_text())["device"]
    assert dev["requested"] == "cpu" and dev["resolved"] == "cpu"
    out2 = tmp_path / "devcuda"
    assert runner.main(sf.runner_args(study, "baseline", out2,
                                      extra=["--device", "cuda"])) == 2
    err = _error(out2)
    assert err["stage"] == "preflight" and "cuda requested" in err["error"]


def _plain_config(tmp_path, fx) -> Path:
    raw = json.loads((sf.BASE_DIR / "randwin.json").read_text())
    raw.update({"num_epochs": 1, "batch_size": 16, "text_encoder_model": "fake",
                "embedding_cache_path": str(tmp_path / "emb.pt"),
                "go_ppi_split_dir": str(fx["split_dir"]),
                "go_ppi_converted_dir": str(fx["converted_dir"]),
                "go_index_cache": str(fx["index"]), "log_frequency": 1000,
                "checkpoint_frequency": 100000})
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(raw))
    return path


def test_default_path_unchanged_and_cpu_flag_byte_identical(tmp_path, fake_encoder,
                                                            monkeypatch):
    """Without --withholding-manifest: same manifest keys as before; --device cpu on a
    machine without CUDA trains byte-identically and only adds the device block."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fx = write_identity_fixture(tmp_path / "fx")
    cfg = _plain_config(tmp_path, fx)
    base = ["--domain", "go_ppi", "--config", str(cfg), "--train-file", str(fx["train_file"]),
            "--seed", "13"]
    assert runner.main(base + ["--output-dir", str(tmp_path / "a")]) == 0
    (tmp_path / "emb.pt").unlink()                         # same cold-cache start for b
    assert runner.main(base + ["--output-dir", str(tmp_path / "b"), "--device", "cpu"]) == 0
    la = json.loads((tmp_path / "a" / "lc_manifest.json").read_text())
    lb = json.loads((tmp_path / "b" / "lc_manifest.json").read_text())
    assert set(la) == LC_KEYS                              # unchanged default path
    assert set(lb) == LC_KEYS | {"device"} and lb["device"]["resolved"] == "cpu"
    assert la["attach"]["pool"] == lb["attach"]["pool"]
    ca = torch.load(tmp_path / "a" / "checkpoint_epoch_0.pt", weights_only=False)
    cb = torch.load(tmp_path / "b" / "checkpoint_epoch_0.pt", weights_only=False)
    assert ca["epoch_losses"] == cb["epoch_losses"]
    assert all(torch.equal(ca["model_state_dict"][k], cb["model_state_dict"][k])
               for k in ca["model_state_dict"])
