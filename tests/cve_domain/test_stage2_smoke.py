"""Integration/smoke test for a tiny CVE Stage 2 supervised run (task 10.3).

Feature: cve-vulnerability-ranking
Requirements: 8.4, 8.5

Component under test: :class:`cve_domain.stage2.CVEStage2Trainer` — the additive
wiring layer (task 10.2) that loads a **Stage 1 checkpoint** and attaches the four
:class:`~cve_domain.supervised_heads.CVESupervisedHeads` over the **frozen**
pretrained embeddings (the frozen sentence-transformer encoder + the Stage 1
projection head), training only the supervised heads.

This test builds a tiny synthetic world in ``tmp_path``:

* A **synthetic Stage 1 checkpoint** — a dict whose ``model_state_dict`` carries
  the projection-head tensors (``projection_head.0.weight/bias`` sized
  ``hidden x 384`` and ``projection_head.3.weight/bias`` sized ``128 x hidden``)
  the way the Stage 1 ``CareerAwareContrastiveModel`` saves them. 384 is the
  ``all-MiniLM-L6-v2`` embedding width; 128 is the contrastive projection dim.
* Tiny ``CVE_View_Records`` (``train.jsonl`` + ``validation.jsonl``) carrying all
  four supervised labels — ``priority_score`` (regression), ``priority_band``
  (multiclass, ``>= 2`` distinct classes), ``in_kev`` and ``ransomware``
  (binary) — plus the ``encoder_view`` text the frozen encoder embeds.

Two layers of assertions so the wiring is always proven even without the encoder:

* ``test_prepare_detects_all_four_heads`` (always runs; no ``torch``/encoder
  load) — constructs the trainer against the synthetic checkpoint and asserts
  :meth:`CVEStage2Trainer.prepare` enables **all four** supervised heads from the
  labels present on the records (the Req 8.5 head set) with a ``>= 2``-class band
  vocabulary. The Req 8.6 missing-checkpoint guard means construction itself
  proves the checkpoint path is honored.

* ``test_stage2_end_to_end_attaches_four_frozen_heads`` (skips gracefully when
  the sentence-transformer encoder cannot be loaded) — runs the real tiny Stage 2
  fine-tune and asserts:
    - the Stage 1 checkpoint is loaded and its projection head reconstructed over
      the **frozen** encoder — no encoder parameter and no projection parameter is
      trainable, only the four heads are (Req 8.4, 9.1);
    - all four heads are attached and produce correctly-shaped outputs for the
      appropriate loss (regression -> ``(batch,)``, multiclass ->
      ``(batch, n_bands)``, binary -> ``(batch,)``; the trainer pairs these with
      MSE / CrossEntropy / BCEWithLogits — Req 8.5); and
    - a Stage 2 checkpoint (``stage2_best_checkpoint.pt``) and report
      (``stage2_report.json``) are produced.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_stage2_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Heavy torch/encoder path is gated for the whole module: the synthetic Stage 1
# checkpoint is built from torch tensors, so torch is required even to construct
# the fixture.
pytest.importorskip("torch")

import torch  # noqa: E402  (imported after importorskip)

from contrastive_learning.data_structures import TrainingConfig  # noqa: E402
from cve_domain.stage2 import (  # noqa: E402
    ALL_HEADS,
    STAGE2_BEST_CHECKPOINT,
    STAGE2_REPORT_FILENAME,
    CVEStage2Trainer,
)
from cve_domain.supervised_heads import (  # noqa: E402
    IN_KEV_HEAD,
    PRIORITY_BAND_HEAD,
    PRIORITY_SCORE_HEAD,
    RANSOMWARE_HEAD,
)


# ---------------------------------------------------------------------------
# Tiny synthetic world
# ---------------------------------------------------------------------------
_ENCODER_DIM = 384       # all-MiniLM-L6-v2 embedding width
_PROJECTION_HIDDEN = 64  # Stage 1 projection hidden width (arbitrary for the test)
_PROJECTION_DIM = 128    # Stage 1 contrastive projection dim


def _view_record(cve: str, band: str, score: float, in_kev: bool, ransom: bool) -> Dict[str, Any]:
    """A minimal CVE_View_Record carrying all four supervised labels."""
    return {
        "cve": cve,
        "encoder_view": (
            f"{cve}. Synthetic vulnerability in band {band}. CWE-79. "
            f"CVSS 7.5 HIGH. Affects some product."
        ),
        "nvd_published": "2024-01-01T00:00:00.000",
        "cve_labels": {
            "priority_score": score,
            "priority_band": band,
            "in_kev": in_kev,
            "ransomware": ransom,
        },
        "ontology": {"cwes": ["CWE-79"], "cpes": [], "vendors": ["acme"]},
    }


def _train_records() -> List[Dict[str, Any]]:
    # Two distinct bands so the multiclass head has >= 2 classes (Req 8.5).
    specs = [
        ("CVE-2024-1000", "high", 90.0, True, False),
        ("CVE-2024-1001", "low", 20.0, False, False),
        ("CVE-2024-1002", "high", 85.0, True, True),
        ("CVE-2024-1003", "low", 10.0, False, False),
        ("CVE-2024-1004", "high", 95.0, True, True),
        ("CVE-2024-1005", "low", 30.0, False, False),
    ]
    return [_view_record(*spec) for spec in specs]


def _validation_records() -> List[Dict[str, Any]]:
    specs = [
        ("CVE-2024-2000", "high", 88.0, True, False),
        ("CVE-2024-2001", "low", 15.0, False, False),
    ]
    return [_view_record(*spec) for spec in specs]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_stage1_checkpoint(path: Path) -> None:
    """Write a synthetic Stage 1 checkpoint with a loadable projection head.

    The ``model_state_dict`` contains the two Linear layers of the Stage 1
    projection head under the exact keys the Stage 2 loader expects
    (``projection_head.0.*`` sized ``hidden x 384`` and ``projection_head.3.*``
    sized ``128 x hidden``), so :class:`CVEStage2Trainer` can reconstruct the
    frozen projection over the 384-dim MiniLM embeddings.
    """
    state_dict = {
        "projection_head.0.weight": torch.randn(_PROJECTION_HIDDEN, _ENCODER_DIM),
        "projection_head.0.bias": torch.randn(_PROJECTION_HIDDEN),
        "projection_head.3.weight": torch.randn(_PROJECTION_DIM, _PROJECTION_HIDDEN),
        "projection_head.3.bias": torch.randn(_PROJECTION_DIM),
    }
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": state_dict,
            "val_loss": 0.123,
            "config": {"freeze_text_encoder": True, "domain_adapter": "cve"},
        },
        path,
    )


@pytest.fixture
def cve_stage2_world(tmp_path: Path) -> Dict[str, Any]:
    """Materialize a synthetic Stage 1 checkpoint + train/val splits in tmp_path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"
    checkpoint_path = data_dir / "best_checkpoint.pt"

    train = _train_records()
    validation = _validation_records()
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, validation)
    _make_stage1_checkpoint(checkpoint_path)

    return {
        "train": train,
        "validation": validation,
        "train_path": train_path,
        "val_path": val_path,
        "checkpoint_path": checkpoint_path,
        "output_dir": tmp_path / "stage2_out",
    }


def _make_config() -> TrainingConfig:
    """A tiny CVE Stage 2 Run_Config (few epochs, small batch, frozen encoder)."""
    return TrainingConfig(
        domain_adapter="cve",
        num_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        freeze_text_encoder=True,
    )


def _encoder_available(model_name: str) -> bool:
    """True when the sentence-transformer encoder can be constructed locally."""
    try:  # pragma: no cover - environment dependent
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_name)
        return True
    except Exception:  # pragma: no cover - offline / missing weights
        return False


# ---------------------------------------------------------------------------
# 1. Head detection wiring (checkpoint honored; no encoder load)
# ---------------------------------------------------------------------------
def test_prepare_detects_all_four_heads(cve_stage2_world: Dict[str, Any]) -> None:
    """Constructing against the Stage 1 checkpoint + prepare() enables all four heads.

    Proves the Req 8.6 checkpoint-existence guard is satisfied (construction
    succeeds only when the checkpoint exists) and that the four supervised heads
    of Req 8.5 are detected from the labels present on the records, with a
    ``>= 2``-class band vocabulary — all without loading the encoder.
    """
    config = _make_config()
    trainer = CVEStage2Trainer(
        config=config,
        output_dir=cve_stage2_world["output_dir"],
        stage1_checkpoint_path=str(cve_stage2_world["checkpoint_path"]),
    )

    # The checkpoint path is honored (Req 8.4/8.6): construction validated it exists.
    assert Path(trainer.stage1_checkpoint_path) == cve_stage2_world["checkpoint_path"]

    head_config = trainer.prepare(cve_stage2_world["train"])

    # All four supervised heads are enabled (Req 8.5).
    for head in ALL_HEADS:
        assert head_config.enabled.get(head) is True, f"{head} head should be enabled"
    assert not head_config.skip_reasons, "no head should be skipped for this data"

    # Multiclass band head has >= 2 distinct classes.
    assert head_config.band_vocabulary == ["high", "low"]


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    """A non-existent Stage 1 checkpoint stops before training (Req 8.6)."""
    config = _make_config()
    with pytest.raises(FileNotFoundError):
        CVEStage2Trainer(
            config=config,
            output_dir=tmp_path / "out",
            stage1_checkpoint_path=str(tmp_path / "does_not_exist.pt"),
        )


# ---------------------------------------------------------------------------
# 2. End-to-end: four heads over a frozen encoder + projection, checkpoint + report
# ---------------------------------------------------------------------------
def test_stage2_end_to_end_attaches_four_frozen_heads(
    cve_stage2_world: Dict[str, Any],
) -> None:
    """A tiny end-to-end Stage 2 run attaches four heads over frozen embeddings.

    Heavy path: requires a loadable sentence-transformer encoder. Skips gracefully
    when it is unavailable so the wiring test above still guards head detection.
    """
    config = _make_config()
    if not _encoder_available(config.text_encoder_model):
        pytest.skip(
            f"sentence-transformer encoder '{config.text_encoder_model}' "
            "is not available in this environment"
        )

    trainer = CVEStage2Trainer(
        config=config,
        output_dir=cve_stage2_world["output_dir"],
        stage1_checkpoint_path=str(cve_stage2_world["checkpoint_path"]),
    )

    result = trainer.train(
        train_path=cve_stage2_world["train_path"],
        validation_path=cve_stage2_world["val_path"],
    )

    # --- Req 8.4: Stage 1 checkpoint loaded, projection reconstructed frozen. ---
    assert Path(result.stage1_checkpoint_path) == cve_stage2_world["checkpoint_path"]
    assert trainer.projection is not None, "Stage 1 projection head must be loaded"
    assert trainer.projection.projection_dim == _PROJECTION_DIM
    # The projection input matches the 384-dim MiniLM embedding.
    assert trainer.projection.projection_head[0].in_features == _ENCODER_DIM

    # --- Req 9.1 / 8.4: encoder + projection frozen; only the heads train. ---
    assert all(
        not p.requires_grad for p in trainer.text_encoder.parameters()
    ), "no sentence-transformer parameter should be trainable (encoder frozen)"
    assert all(
        not p.requires_grad for p in trainer.projection.parameters()
    ), "the Stage 1 projection head must stay frozen"
    assert any(
        p.requires_grad for p in trainer.heads.parameters()
    ), "the supervised heads must be trainable"

    # --- Req 8.5: all four heads attached and enabled. ---
    enabled = trainer.heads.enabled_heads
    for head in ALL_HEADS:
        assert enabled[head] is True, f"{head} head should be attached"

    # The heads produce correctly-shaped outputs for their appropriate loss over
    # the frozen embeddings (regression/binary -> (batch,), multiclass ->
    # (batch, n_bands)); this is what MSE / BCEWithLogits / CrossEntropy consume.
    embeddings = trainer._embed_views([r["encoder_view"] for r in cve_stage2_world["train"]])
    assert embeddings.shape == (len(cve_stage2_world["train"]), _PROJECTION_DIM)
    outputs = trainer.heads(embeddings)
    batch = len(cve_stage2_world["train"])
    n_bands = len(result.head_configuration.band_vocabulary)
    assert outputs[PRIORITY_SCORE_HEAD].shape == (batch,)
    assert outputs[PRIORITY_BAND_HEAD].shape == (batch, n_bands)
    assert outputs[IN_KEV_HEAD].shape == (batch,)
    assert outputs[RANSOMWARE_HEAD].shape == (batch,)

    # --- A Stage 2 checkpoint + report were produced. ---
    assert result.best_checkpoint_path is not None
    checkpoint_path = cve_stage2_world["output_dir"] / STAGE2_BEST_CHECKPOINT
    report_path = cve_stage2_world["output_dir"] / STAGE2_REPORT_FILENAME
    assert checkpoint_path.exists(), "stage2 best-by-validation checkpoint must be saved"
    assert report_path.exists(), "stage2 report must be written"

    # The checkpoint records the enabled heads + the originating Stage 1 checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(checkpoint["enabled_heads"]) == set(ALL_HEADS)
    assert checkpoint["embedding_dim"] == _PROJECTION_DIM
    assert Path(checkpoint["stage1_checkpoint_path"]) == cve_stage2_world["checkpoint_path"]

    # The report enumerates all four enabled heads and no skips.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report["enabled_heads"]) == set(ALL_HEADS)
    assert report["skipped_heads"] == {}
    assert Path(report["stage1_checkpoint_path"]) == cve_stage2_world["checkpoint_path"]


# ---------------------------------------------------------------------------
# E0 base-embeddings baseline: end-to-end train -> predict -> evaluate
# ---------------------------------------------------------------------------
def _encoder_loadable(model_name: str) -> bool:
    try:  # pragma: no cover - environment dependent
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_name)
        return True
    except Exception:  # pragma: no cover
        return False


def test_base_embeddings_end_to_end_train_predict_evaluate(tmp_path: Path) -> None:
    """E0 path: heads over the frozen base encoder (no Stage 1), then predict+eval.

    Exercises the ``base_embeddings=True`` branch of ``_embed_views`` — the one the
    experiment smoke caught returning an inference-mode tensor (unusable for
    backward). Asserts training runs (the clone fix), all four heads attach over
    the raw encoder dim, ``predict_records`` yields per-CVE predictions, and the
    ``CVEEvaluationReporter`` produces a non-skipped ranking metric on the
    ground-truth labels.
    """
    config = TrainingConfig(domain_adapter="cve", num_epochs=1, batch_size=4,
                            learning_rate=0.01, freeze_text_encoder=True)
    if not _encoder_loadable(config.text_encoder_model):
        pytest.skip(f"encoder '{config.text_encoder_model}' not available")

    from cve_domain.evaluation_reporter import CVEEvaluationReporter

    train = _train_records()
    validation = _validation_records()
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "validation.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, validation)

    # No Stage 1 checkpoint required in base-embeddings mode.
    trainer = CVEStage2Trainer(config, tmp_path / "stage2_base", base_embeddings=True)
    result = trainer.train(train_path=str(train_path), validation_path=str(val_path))

    # Trained over the raw frozen encoder dim (no Stage 1 projection).
    assert trainer.base_embeddings is True
    assert trainer.projection is None
    assert trainer.embedding_dim == trainer.text_encoder.get_sentence_embedding_dimension()
    assert set(trainer.heads.enabled_heads) and all(trainer.heads.enabled_heads.values())
    assert result.best_checkpoint_path is not None

    # Predictions for every test record, consumed by the evaluation reporter.
    test_records = _validation_records()
    preds = trainer.predict_records(test_records)
    assert set(preds) == {r["cve"] for r in test_records}
    for p in preds.values():
        assert "ranking_score" in p and "in_kev" in p and "priority_band" in p
        assert isinstance(p["embedding"], list) and len(p["embedding"]) == trainer.embedding_dim

    report = CVEEvaluationReporter().evaluate(test_records, preds, output_dir=str(tmp_path / "eval"))
    assert report.status == "ok"
    assert report.ranking.get("skipped") is False          # priority_score present
    assert (tmp_path / "eval" / "evaluation_report.json").exists()
