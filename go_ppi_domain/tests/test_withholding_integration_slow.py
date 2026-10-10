"""Slow integration tests with the real trainer and the real frozen MPNet
(design 5.4, 6, 14). Run with ``GO_WITHHOLDING_SLOW=1``; skipped otherwise.

Layout: a 9-target synthetic-but-real-format study (``_study_fixtures``) whose
conditions are written by ``make_conditions.py`` itself; the tests train on the
240-record ``p01_k0`` condition (4 batches of 64).

(a) the runner, one clean subprocess per cell (as the sweep runs it), 1 epoch:
    baseline == loss_only and neg_only == both selection hashes, baseline !=
    neg_only, every selector counter 0, L_GO applied exactly in the loss cells;
(b) diagnostics non-interference: identical epoch loss, state_dict and selection
    hash with and without the diagnostics callback;
(c) a forced L_GO failure (relation table detached after epoch 0) -> runner exit
    2 and ``run_error.json``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.tests import _study_fixtures as sf

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
MPNET = "sentence-transformers/all-mpnet-base-v2"
COND = "p01_k0"
PATCHED = ("TARGETS", "VALIDATION_TF", "CONVERTED", "SPLITS", "CONDITIONS", "STUDY_RES",
           "CONFIG_DIR")
BOOT = """
import json, sys
from pathlib import Path
sys.path.insert(0, {root!r})
from go_ppi_domain import withholding_constants as wc
for k, v in json.loads({paths!r}).items():
    setattr(wc, k, Path(v))
from scripts.run_learning_curve_point import main
sys.exit(main(json.loads({argv!r})))
"""


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    from scripts.go_withholding import make_conditions as mc

    root = tmp_path_factory.mktemp("slow_study")
    mp = pytest.MonkeyPatch()
    try:
        s = sf.build_study(root, n_targets=9, conds=(), text_encoder_model=MPNET)
        sf.patch_study(mp, s)
        shutil.rmtree(s["val_dir"])
        wc.append_pin("G2-targets", {"target_table_sha256": wc.sha256_file(s["table_path"])})
        common = ["--target-table", str(s["table_path"]), "--base-splits", str(s["splits"]),
                  "--converted", str(s["converted"])]
        assert mc.main(["--mode", "validation", *common, "--out", str(s["val_dir"])]) == 0
        assert mc.main(["--mode", "conditions", *common, "--targetfree-validation",
                        str(s["val_dir"]), "--out", str(s["conditions"])]) == 0
        for cell in wc.CELLS:
            cfg = sf.cell_config(s, cell, cond=COND, text_encoder_model=MPNET, batch_size=64)
            (s["config_dir"] / f"{cell}.json").write_text(json.dumps(cfg, indent=2))
        s["paths"] = {k: str(getattr(wc, k)) for k in PATCHED}
        s["pins"] = str(wc.pins_path())
    finally:
        mp.undo()
    n = sum(1 for _ in open(s["conditions"] / COND / "train.jsonl"))
    assert 200 <= n <= 256, n
    return s


@pytest.fixture()
def patched(layout, monkeypatch):
    sf.patch_study(monkeypatch, layout)
    return layout


def _env(s):
    env = dict(os.environ)
    env.update({"GO_WITHHOLDING_PINS": s["pins"], "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1"})
    return env


def _subprocess_run(s, cell, out, extra=()):
    argv = sf.runner_args(s, cell, out, cond=COND, extra=extra)
    code = BOOT.format(root=str(ROOT), paths=json.dumps(s["paths"]), argv=json.dumps(argv))
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_env(s),
                          capture_output=True, text=True, timeout=1200)


def test_a_real_trainer_selection_identity(layout, tmp_path):
    lcs = {}
    for cell in wc.CELLS:
        out = tmp_path / cell
        proc = _subprocess_run(layout, cell, out)
        assert proc.returncode == 0, (proc.stderr[-3000:],
                                      (out / "run_error.json").read_text()
                                      if (out / "run_error.json").exists() else None)
        lcs[cell] = json.loads((out / "lc_manifest.json").read_text())
    sel = {c: lc["negative_selection"]["selection_sha256_by_epoch"] for c, lc in lcs.items()}
    assert sel["baseline"] == sel["loss_only"]
    assert sel["neg_only"] == sel["both"]
    assert sel["baseline"] != sel["neg_only"]
    for cell, lc in lcs.items():
        ns = lc["negative_selection"]
        assert ns["selector_none_returns"] == 0 and ns["selector_exceptions"] == 0
        assert ns["unannotated_anchor_window"] is True
        assert lc["go_relation_loss"]["missing_ids"] == 0
        applied = lc["go_relation_loss"]["applied_batches"]
        assert (applied > 0) == (cell in ("loss_only", "both")), (cell, applied)
        assert lc["withholding"]["assertions"]["post_import_coverage"] == "ok"
        assert lc["device"]["resolved"] in ("cpu", "cuda")
        assert lc["attach"]["cache_check"]
        assert str(ROOT) not in json.dumps(lc)
    for key in ("unannotated_anchor_calls_by_epoch", "calls_by_epoch"):
        assert len({json.dumps(lc["negative_selection"][key], sort_keys=True)
                    for lc in lcs.values()}) == 1, key
    assert len({json.dumps([e["ok_batches"] for e in lc["epoch_stats"]])
                for lc in lcs.values()}) == 1
    assert len({lc["train_records"] for lc in lcs.values()}) == 1

    # Cross-feature seam (plan item 23): the sweep's group checks, its post-check
    # reader and the evaluator's checkpoint resolution accept the real runner output.
    from scripts.go_withholding import evaluate_withholding as ev
    from scripts.go_withholding import sweep

    run_ids = [wc.run_id_for(COND, cell, 13) for cell in wc.CELLS]
    res = sweep.group_checks([lcs[c] for c in wc.CELLS], run_ids, require_epochs=1)
    assert res["failures"] == [], res["failures"]
    for cell, lc in lcs.items():
        assert sweep._post_checks_ok(lc), cell
        ck = ev.resolve_checkpoint(tmp_path / cell)
        assert ck["epochs"] == 1 and ck["path"].name == "checkpoint_epoch_0.pt"


def _train_in_process(s, cell, out, with_diag):
    import go_ppi_domain.record_adapter  # noqa: F401
    from contrastive_learning.trainer import ContrastiveLearningTrainer
    from go_ppi_domain.withholding_diagnostics import (
        EpochDiagnostics, SelectionRecorder, StrictEpochGuard)
    from scripts.run_learning_curve_point import attach_go_ppi, attach_go_relation_loss
    from trials_domain.label_budget_runner import preencode_pool

    cfg = wc.load_study_config(s["config_dir"] / f"{cell}.json")
    cfg.training_seed = 13
    cond = s["conditions"] / COND
    train_file = cond / "train.jsonl"
    trainer = ContrastiveLearningTrainer(config=cfg, output_dir=str(out))
    attach_go_ppi(trainer, cfg)
    attach_go_relation_loss(trainer, cfg, train_file)
    manifest = json.loads((cond / "withholding_manifest.json").read_text())
    rec = SelectionRecorder(trainer.batch_processor.domain_negative_selector,
                            unannotated_anchors=manifest["unannotated_anchors"]["ids"])
    preencode_pool(trainer, trainer.batch_processor.domain_negative_selector)
    guard = StrictEpochGuard(cell)
    callbacks = [rec.on_epoch_end]
    if with_diag:
        callbacks.append(EpochDiagnostics(out / "go_diagnostics.jsonl", train_file, rec,
                                          weight=cfg.go_ontology_loss_weight))
    callbacks.append(guard)
    trainer.epoch_end_callbacks = callbacks
    trainer.train(str(train_file))
    return trainer, rec, guard


def test_b_diagnostics_do_not_interfere(patched, tmp_path):
    t0, r0, g0 = _train_in_process(patched, "both", tmp_path / "plain", with_diag=False)
    t1, r1, g1 = _train_in_process(patched, "both", tmp_path / "diag", with_diag=True)
    assert g0.epochs[0]["avg_loss"] == g1.epochs[0]["avg_loss"]
    s0, s1 = t0.model.state_dict(), t1.model.state_dict()
    assert all(torch.equal(s0[k], s1[k]) for k in s0)
    assert r0.by_epoch == r1.by_epoch and r0.calls_by_epoch == r1.calls_by_epoch
    assert r1.exceptions == r1.none_returns == 0
    row = json.loads((tmp_path / "diag" / "go_diagnostics.jsonl").read_text().splitlines()[0])
    assert row["error"] is None and row["drift_flag"] is False and row["n_batches"] > 0
    assert row["grad_norm_go_relation_unweighted"]["median"] > 0


def test_c_forced_go_relation_failure_exits_2(patched, tmp_path, monkeypatch):
    from go_ppi_domain import withholding_diagnostics as wd
    from scripts import run_learning_curve_point as runner

    original = wd.EpochDiagnostics.__call__

    def detach_after_first_epoch(self, trainer, epoch, stats):
        original(self, trainer, epoch, stats)
        trainer.loss_engine.go_relation_table = None
    monkeypatch.setattr(wd.EpochDiagnostics, "__call__", detach_after_first_epoch)
    out = tmp_path / "forced"
    rc = runner.main(sf.runner_args(patched, "loss_only", out, cond=COND,
                                    extra=["--epochs", "2"]))
    err = json.loads((out / "run_error.json").read_text())
    assert rc == 2 and err["stage"] == "train"
    assert "StrictEpochGuard[loss_only] epoch 1" in err["error"]
    assert err["cond_id"] == COND and err["cell"] == "loss_only" and err["seed"] == 13
    assert list(out.glob("*emergency*.pt")) and not (out / "lc_manifest.json").exists()
