"""StrictEpochGuard, SelectionRecorder and EpochDiagnostics (design 4.5, 4.6, 6, 14).

The trainer-level tests run the REAL ``ContrastiveLearningTrainer`` (real
``_train_epoch``, batch processor, loss engine and projection head) with a tiny
deterministic stand-in for the frozen SentenceTransformer, so no MPNet is loaded.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import go_ppi_domain.record_adapter  # noqa: F401  registers "go_ppi"
from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.trainer import ContrastiveLearningTrainer
from go_ppi_domain.tests._withholding_fixtures import (
    make_sample, small_selector, write_identity_fixture, write_small_fixture)
from go_ppi_domain.withholding_diagnostics import (
    EpochDiagnostics, SelectionRecorder, StrictEpochGuard, negative_selection_block,
    preserved_rng, projection_param_summary, projection_params)
from scripts.run_learning_curve_point import attach_go_ppi, attach_go_relation_loss

RANDWIN = (Path(__file__).resolve().parents[2]
           / "config" / "go_decomposition_go_ordinal" / "randwin.json")
DIM = 16


# ------------------------------------------------------------------ guard
def _engine(applied=0):
    return SimpleNamespace(loss_engine=SimpleNamespace(go_relation_applied_batches=applied))


def _stats(epoch=0, failed=0, ok=10, avg=1.0):
    return {"epoch": epoch, "failed_batches": failed, "ok_batches": ok, "avg_loss": avg}


@pytest.mark.parametrize("stats,match", [
    (_stats(failed=1), "failed batch"),
    (_stats(ok=0, avg=float("inf")), "no successful batch"),
    (_stats(avg=float("nan")), "non-finite"),
    (_stats(avg=float("inf")), "non-finite"),
])
def test_guard_raises_on_failed_or_bad_epochs(stats, match):
    with pytest.raises(RuntimeError, match=match):
        StrictEpochGuard("baseline")(_engine(), 0, stats)


def test_guard_loss_on_applied_share():
    guard = StrictEpochGuard("loss_only")
    trainer = _engine(applied=9)
    guard(trainer, 0, _stats(ok=10))                 # 9/10 >= 0.9
    trainer.loss_engine.go_relation_applied_batches = 17
    with pytest.raises(RuntimeError, match="applied in 8 of 10"):
        guard(trainer, 1, _stats(epoch=1, ok=10))    # delta 8 < 9
    assert [e["go_relation_applied"] for e in guard.epochs] == [9, 8]
    assert set(guard.epochs[0]) == {"epoch", "failed_batches", "ok_batches", "avg_loss",
                                    "go_relation_applied"}


@pytest.mark.parametrize("cell", ["baseline", "neg_only"])
def test_guard_off_cell_with_applied_raises(cell):
    guard = StrictEpochGuard(cell)
    guard(_engine(0), 0, _stats())
    with pytest.raises(RuntimeError, match="loss-off cell"):
        guard(_engine(1), 1, _stats(epoch=1))


# --------------------------------------------------------------- recorder
class _ListSelector:
    def __init__(self, mode="ok"):
        self.mode = mode
        self.unannotated_anchor_window = True

    def select_batch_negatives(self, anchor_sample, candidate_negatives, max_negatives,
                               epoch=None):
        if self.mode == "raise":
            raise ValueError("selector broke")
        if self.mode == "none":
            return None
        a = anchor_sample.metadata["resume_id"]
        return [{"partner_id": f"{a}-n{i}", "grade": i % 2} for i in range(3)], [5.0] * 3


def test_recorder_hash_is_order_independent():
    r1, r2 = SelectionRecorder(s1 := _ListSelector()), SelectionRecorder(s2 := _ListSelector())
    for a in ("x", "y", "z"):
        s1.select_batch_negatives(make_sample(a), [], 3, 0)
    for a in ("z", "x", "y"):
        s2.select_batch_negatives(make_sample(a), [], 3, 0)
    r1.on_epoch_end(None, 0, {})
    r2.on_epoch_end(None, 0, {})
    assert r1.by_epoch == r2.by_epoch and r1.calls_by_epoch == {"0": 3}
    s1.select_batch_negatives(make_sample("x"), [], 3, 1)
    r1.on_epoch_end(None, 1, {})
    assert r1.by_epoch["1"] != r1.by_epoch["0"] and r1.calls_by_epoch["1"] == 1


def test_recorder_counts_none_and_exceptions_and_reraises():
    sel = _ListSelector("none")
    rec = SelectionRecorder(sel)
    assert sel.select_batch_negatives(make_sample("x"), [], 3, 0) is None
    assert rec.none_returns == 1 and rec.calls == 1
    sel.mode = "raise"
    with pytest.raises(ValueError, match="selector broke"):
        sel.select_batch_negatives(make_sample("x"), [], 3, 0)
    assert rec.exceptions == 1


def test_recorder_ignores_paused_calls_but_counts_paused_exceptions():
    sel = _ListSelector()
    rec = SelectionRecorder(sel)
    with rec.pause():
        assert rec.paused
        sel.select_batch_negatives(make_sample("x"), [], 3, 0)
        sel.mode = "raise"
        with pytest.raises(ValueError):
            sel.select_batch_negatives(make_sample("x"), [], 3, 0)
    assert not rec.paused
    rec.on_epoch_end(None, 0, {})
    assert rec.calls == 0 and rec.calls_by_epoch == {"0": 0}
    assert rec.exceptions == 1
    assert rec.by_epoch["0"] == hashlib.sha256(b"").hexdigest()


def test_recorder_attribute_names_and_manifest_block(tmp_path):
    selector = small_selector(write_small_fixture(tmp_path), "neg_only", 13,
                              unannotated_anchor_window=True)
    rec = SelectionRecorder(selector, unannotated_anchors=["U"])
    for name in ("by_epoch", "calls_by_epoch", "unannotated_anchor_calls_by_epoch",
                 "none_returns", "exceptions"):
        assert hasattr(rec, name), name
    assert not hasattr(rec, "unannotated_calls_by_epoch")
    selector.select_batch_negatives(make_sample("U"), [], 6, 0)
    selector.select_batch_negatives(make_sample("A"), [], 6, 0)
    rec.on_epoch_end(None, 0, {})
    block = negative_selection_block(rec, selector)
    assert set(block) == {"selection_sha256_by_epoch", "calls_by_epoch",
                          "unannotated_anchor_calls_by_epoch", "unannotated_anchor_window",
                          "selector_none_returns", "selector_exceptions"}
    assert block["selection_sha256_by_epoch"] == rec.by_epoch
    assert block["calls_by_epoch"] == rec.calls_by_epoch == {"0": 2}
    assert block["unannotated_anchor_calls_by_epoch"] == {"0": 1}
    assert block["unannotated_anchor_window"] is selector.unannotated_anchor_window is True
    assert block["selector_none_returns"] == rec.none_returns == 0
    assert block["selector_exceptions"] == rec.exceptions == 0


def test_preserved_rng_restores_states():
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    with preserved_rng():
        random.random(); np.random.rand(); torch.rand(3)
    a = (random.random(), float(np.random.rand()), torch.rand(1).item())
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    assert a == (random.random(), float(np.random.rand()), torch.rand(1).item())


# ----------------------------------------------------- real-trainer tests
class FakeSentenceTransformer(torch.nn.Module):
    """Deterministic frozen-encoder stand-in: text -> fixed pseudo-random vector."""

    def __init__(self, _name=None):
        super().__init__()
        self.dummy = torch.nn.Linear(2, 2)
        self.max_seq_length = 512

    def get_sentence_embedding_dimension(self):
        return DIM

    def encode(self, texts, convert_to_tensor=True, device=None, show_progress_bar=False,
               normalize_embeddings=False, **_):
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "big")
            out.append(torch.randn(DIM, generator=torch.Generator().manual_seed(seed)))
        return torch.stack(out)


def _config(fx, tmp_path, cell="baseline", **over):
    raw = json.loads(RANDWIN.read_text())
    tiered = cell in ("neg_only", "both")
    raw.update({
        "num_epochs": 1, "batch_size": 16, "text_encoder_model": "fake",
        "enable_embedding_preload": False, "embedding_cache_path": str(tmp_path / "emb.pt"),
        "go_ppi_split_dir": str(fx["split_dir"]), "go_ppi_converted_dir": str(fx["converted_dir"]),
        "go_index_cache": str(fx["index"]), "training_seed": 13,
        "go_ppi_go_tiered_negatives": tiered,
        "go_ppi_tier_sampling": "stochastic" if tiered else "random_window",
        "go_ppi_unannotated_anchor_window": True,
        "go_ontology_loss_weight": 0.3 if cell in ("loss_only", "both") else 0.0,
        "log_frequency": 1000, "checkpoint_frequency": 100000,
    })
    raw.update(over)
    names = {f.name for f in dataclasses.fields(TrainingConfig)}
    return TrainingConfig(**{k: v for k, v in raw.items() if k in names})


@pytest.fixture
def fake_encoder(monkeypatch):
    monkeypatch.setattr("contrastive_learning.trainer.SentenceTransformer",
                        FakeSentenceTransformer)


def _wired_trainer(fx, tmp_path, cell, out, trainer_cls=ContrastiveLearningTrainer, **over):
    cfg = _config(fx, tmp_path, cell, **over)
    trainer = trainer_cls(config=cfg, output_dir=str(out))
    attach_go_ppi(trainer, cfg)
    attach_go_relation_loss(trainer, cfg, fx["train_file"])
    rec = SelectionRecorder(trainer.batch_processor.domain_negative_selector,
                            unannotated_anchors=fx["unannotated"])
    return trainer, cfg, rec


def test_projection_params_exclude_text_encoder(fake_encoder, tmp_path):
    fx = write_identity_fixture(tmp_path / "fx")
    trainer, _, _ = _wired_trainer(fx, tmp_path, "baseline", tmp_path / "run")
    params = projection_params(trainer)
    enc = {id(p) for p in trainer.text_encoder.parameters()}
    assert params and not any(id(p) in enc for p in params)
    summary = projection_param_summary(trainer)
    assert summary["numel"] == sum(p.numel() for p in params) and summary["names"]
    for p in trainer.model.parameters():
        p.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no trainable"):
        projection_params(trainer)


class _FailingBatchTrainer(ContrastiveLearningTrainer):
    def _train_batch(self, batch):
        raise RuntimeError("batch exploded")


def test_guard_turns_swallowed_batch_failure_into_train_error(fake_encoder, tmp_path):
    fx = write_identity_fixture(tmp_path / "fx")
    out = tmp_path / "run"
    trainer, _, rec = _wired_trainer(fx, tmp_path, "baseline", out,
                                     trainer_cls=_FailingBatchTrainer)
    guard = StrictEpochGuard("baseline")
    trainer.epoch_end_callbacks = [rec.on_epoch_end, guard]
    with pytest.raises(RuntimeError, match=r"StrictEpochGuard\[baseline\] epoch 0: 4 failed"):
        trainer.train(str(fx["train_file"]))
    assert guard.epochs[0]["failed_batches"] == 4 and guard.epochs[0]["ok_batches"] == 0
    assert list(out.glob("*_emergency*"))


def _train(fx, tmp_path, cell, out, with_diag):
    trainer, cfg, rec = _wired_trainer(fx, tmp_path, cell, out)
    guard = StrictEpochGuard(cell)
    callbacks = [rec.on_epoch_end]
    diag = None
    if with_diag:
        diag = EpochDiagnostics(out / "go_diagnostics.jsonl", fx["train_file"], rec,
                                n_batches=2, batch_size=16,
                                weight=cfg.go_ontology_loss_weight)
        callbacks.append(diag)
    callbacks.append(guard)
    trainer.epoch_end_callbacks = callbacks
    trainer.train(str(fx["train_file"]))
    return trainer, rec, guard, diag


@pytest.mark.parametrize("cell", ["baseline", "both"])
def test_diagnostics_do_not_interfere_and_write_schema_row(fake_encoder, tmp_path, cell):
    fx = write_identity_fixture(tmp_path / "fx")
    t0, r0, g0, _ = _train(fx, tmp_path, cell, tmp_path / "plain", with_diag=False)
    t1, r1, g1, diag = _train(fx, tmp_path, cell, tmp_path / "diag", with_diag=True)
    # Non-interference: same epoch loss, weights and selection hash.
    assert g0.epochs[0]["avg_loss"] == g1.epochs[0]["avg_loss"]
    s0, s1 = t0.model.state_dict(), t1.model.state_dict()
    assert all(torch.equal(s0[k], s1[k]) for k in s0)
    assert r0.by_epoch == r1.by_epoch and r0.calls_by_epoch == r1.calls_by_epoch
    assert r1.calls == r0.calls and r1.exceptions == r1.none_returns == 0
    # Loss-on cell really applied the term; counters untouched by diagnostics.
    applied = g1.epochs[0]["go_relation_applied"]
    assert applied == (g1.epochs[0]["ok_batches"] if cell == "both" else 0)
    assert t1.loss_engine.go_relation_applied_batches == applied
    assert t1.loss_engine._diagnostic_mode is False and t1.model.training is True
    assert "train" not in vars(t1.model)
    rows = [json.loads(l) for l in (tmp_path / "diag" / "go_diagnostics.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "epoch", "n_batches", "n_triplets", "lambda", "infonce_mean", "ranking_mean",
        "go_relation_mean", "max_component_drift", "drift_flag", "grad_norm_infonce",
        "grad_norm_ranking", "grad_norm_go_relation_unweighted",
        "ratio_weighted_go_to_infonce", "ratio_ranking_to_infonce",
        "grad_norm_total_preclip_eval_diag", "relation_skipped_batches", "epoch_stats",
        "elapsed_s", "error"}
    assert row["error"] is None and row["epoch"] == 0
    assert row["n_batches"] == 2 and row["n_triplets"] == 32
    assert row["max_component_drift"] <= 1e-6 and row["drift_flag"] is False
    assert row["lambda"] == (0.3 if cell == "both" else 0.0)
    assert row["grad_norm_infonce"]["median"] > 0
    assert row["grad_norm_go_relation_unweighted"]["median"] > 0  # logged in every cell
    assert len(row["ratio_weighted_go_to_infonce"]["per_batch"]) == 2
    assert (row["ratio_weighted_go_to_infonce"]["median"] > 0) == (cell == "both")
    assert row["epoch_stats"]["ok_batches"] == g1.epochs[0]["ok_batches"]
    assert diag.table is not None and t1.loss_engine.go_relation_table is not diag.table


def test_diagnostics_accumulate_no_grad_and_leave_optimizer(fake_encoder, tmp_path):
    fx = write_identity_fixture(tmp_path / "fx")
    trainer, cfg, rec = _wired_trainer(fx, tmp_path, "both", tmp_path / "run")
    trainer.optimizer.zero_grad(set_to_none=True)
    opt_state = json.dumps(trainer.optimizer.state_dict()["param_groups"])
    diag = EpochDiagnostics(tmp_path / "run" / "d.jsonl", fx["train_file"], rec,
                            n_batches=1, batch_size=16, weight=0.3)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    diag(trainer, 0, _stats())
    assert json.loads((tmp_path / "run" / "d.jsonl").read_text())["error"] is None
    assert all(p.grad is None for p in trainer.model.parameters())
    assert json.dumps(trainer.optimizer.state_dict()["param_groups"]) == opt_state
    assert not trainer.optimizer.state_dict()["state"]
    assert all(torch.equal(before[k], v) for k, v in trainer.model.state_dict().items())
    assert trainer.loss_engine.go_relation_applied_batches == 0


def test_diagnostics_errors_are_non_fatal(fake_encoder, tmp_path):
    fx = write_identity_fixture(tmp_path / "fx")
    trainer, cfg, rec = _wired_trainer(fx, tmp_path, "baseline", tmp_path / "run")
    diag = EpochDiagnostics(tmp_path / "run" / "d.jsonl", tmp_path / "missing.jsonl", rec)
    trainer.loss_engine._diagnostic_mode = False
    diag(trainer, 0, _stats())
    row = json.loads((tmp_path / "run" / "d.jsonl").read_text())
    assert row["error"].startswith("FileNotFoundError")
    assert trainer.loss_engine._diagnostic_mode is False and not rec.paused


def test_paused_selector_exception_is_counted(fake_encoder, tmp_path):
    fx = write_identity_fixture(tmp_path / "fx")
    trainer, cfg, rec = _wired_trainer(fx, tmp_path, "baseline", tmp_path / "run")

    def boom(*a, **k):
        raise ValueError("selector broke in diagnostics")
    rec._inner = boom
    diag = EpochDiagnostics(tmp_path / "run" / "d.jsonl", fx["train_file"], rec,
                            n_batches=1, batch_size=16)
    diag(trainer, 0, _stats())
    row = json.loads((tmp_path / "run" / "d.jsonl").read_text())
    assert rec.exceptions == 16 and rec.calls == 0   # fatal later, at assert_post_training
    assert row["epoch"] == 0                          # the row is still written
