"""Integration/smoke test for a tiny CVE Stage 1 contrastive run (task 9.2).

Feature: cve-vulnerability-ranking
Requirements: 8.1, 8.2, 8.3, 9.1

Component under test: :class:`cve_domain.stage1.Stage1ContrastivePretrainer` — the
thin wiring layer (task 9.1) that drives the *existing* CDCL
``ContrastiveLearningTrainer`` for CVE Stage 1 contrastive pretraining.

This test builds a tiny synthetic world of ``CVE_View_Records`` (split into
``train.jsonl`` + ``validation.jsonl``) plus a small ``cve_denominator_pools.jsonl``
in ``tmp_path`` and exercises the pretrainer end-to-end:

Two layers of assertions, so the wiring is always proven even when the heavy
encoder is unavailable:

* ``test_prepare_pairs_and_selection_wiring`` (always runs) — drives
  :meth:`Stage1ContrastivePretrainer.prepare` and asserts the
  preparation/pairing wiring produces the paired anchor/positive inputs (Req 8.1,
  8.2: each training example is an *anchor + ontology-related positive* real pair,
  no augmentation) and writes the per-split negative/positive selection reports.
  This needs no ``torch``/encoder.

* ``test_stage1_end_to_end_best_checkpoint_frozen_encoder`` (skips gracefully when
  the sentence-transformer encoder cannot be loaded) — runs the real trainer for
  a couple of tiny epochs and asserts:
    - a **best-by-validation checkpoint** (``best_checkpoint.pt``) is produced
      (Req 8.3), carrying the validation-loss metric it was selected by; and
    - the sentence-transformer encoder is **frozen** — no encoder parameter is
      trainable while the projection head is (Req 9.1).

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_stage1_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from contrastive_learning.data_structures import TrainingConfig
from cve_domain.stage1 import (
    PAIRED_TRAIN_FILENAME,
    PAIRED_VAL_FILENAME,
    POSITIVE_KEY,
    Stage1ContrastivePretrainer,
)


# ---------------------------------------------------------------------------
# Tiny synthetic CVE world
# ---------------------------------------------------------------------------
#
# Records within a split share an ontology signal (a common CWE) so the
# CVEPositiveSelector always resolves an in-split ontology-related sibling for
# every anchor (Req 8.2, 13.x). Train and validation are independent worlds
# (different shared CWE) since positive selection is per-split.

_TRAIN_CWE = "CWE-79"       # every train anchor shares this -> cascade level 2
_VAL_CWE = "CWE-89"         # every validation anchor shares this
_MAX_NEGATIVES = 3
_SPLIT_SEED = 123


def _view_record(cve: str, cwe: str, vendor: str, band: str) -> Dict[str, Any]:
    """A minimal but complete CVE_View_Record as produced by the converter."""
    return {
        "cve": cve,
        "encoder_view": (
            f"{cve}. Synthetic {vendor} vulnerability. {cwe}. "
            f"CVSS 7.5 HIGH. Affects {vendor} product."
        ),
        "nvd_published": "2024-01-01T00:00:00.000",
        "cve_labels": {
            "priority_score": 50.0,
            "priority_band": band,
            "in_kev": False,
            "ransomware": False,
        },
        "ontology": {"cwes": [cwe], "cpes": [], "vendors": [vendor]},
    }


def _train_records() -> List[Dict[str, Any]]:
    bands = ["high", "medium", "high", "low", "medium", "high"]
    return [
        _view_record(f"CVE-2024-1{idx:03d}", _TRAIN_CWE, f"vendor{idx}", bands[idx])
        for idx in range(6)
    ]


def _validation_records() -> List[Dict[str, Any]]:
    bands = ["high", "medium", "low", "high"]
    return [
        _view_record(f"CVE-2024-2{idx:03d}", _VAL_CWE, f"vendorv{idx}", bands[idx])
        for idx in range(4)
    ]


def _denominator_pools(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Give every anchor tiered pooled negatives drawn from the other CVEs.

    The other CVEs are spread across the three tiers so the selector never needs
    the same-split random fallback and always has ontology siblings left over for
    the positive selector (which excludes an anchor's selected negatives).
    """
    all_ids = [r["cve"] for r in records]
    pools: List[Dict[str, Any]] = []
    for anchor in all_ids:
        others = [c for c in all_ids if c != anchor]
        pools.append(
            {
                "cve": anchor,
                "hard_negatives": others[:1],
                "medium_negatives": others[1:2],
                "easy_negatives": others[2:],
            }
        )
    return pools


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


@pytest.fixture
def cve_world(tmp_path: Path) -> Dict[str, Any]:
    """Materialize train/validation splits + denominator pools in tmp_path."""
    train = _train_records()
    validation = _validation_records()
    all_records = train + validation

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"
    pools_path = data_dir / "cve_denominator_pools.jsonl"

    _write_jsonl(train_path, train)
    _write_jsonl(val_path, validation)
    _write_jsonl(pools_path, _denominator_pools(all_records))

    return {
        "train": train,
        "validation": validation,
        "train_path": train_path,
        "val_path": val_path,
        "pools_path": pools_path,
        "output_dir": tmp_path / "stage1_out",
    }


def _make_config(pools_path: Path) -> TrainingConfig:
    """A tiny CVE Stage 1 Run_Config (few epochs, small batch)."""
    return TrainingConfig(
        domain_adapter="cve",
        num_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        max_negatives_per_anchor=_MAX_NEGATIVES,
        split_seed=_SPLIT_SEED,
        validate_every_n_epochs=1,
        cve_denominator_pools_path=str(pools_path),
        # Keep the career pathway/global negative logic out of the picture; the
        # CVE tiered selector is injected separately.
        use_pathway_negatives=False,
        pathway_weight=0.0,
        loss_type="infonce",
    )


# ---------------------------------------------------------------------------
# 1. Preparation/pairing wiring (no encoder required)
# ---------------------------------------------------------------------------


def test_prepare_pairs_and_selection_wiring(cve_world: Dict[str, Any]) -> None:
    """prepare() produces paired anchor/positive inputs + selection reports.

    Validates the Stage 1 wiring of Req 8.1/8.2 without loading the encoder:
    every anchor is paired with an in-split ontology-related positive (a real
    pair — no augmentation), and both the negative- and positive-selection
    reports are written per split.
    """
    config = _make_config(cve_world["pools_path"])
    pretrainer = Stage1ContrastivePretrainer(
        config=config,
        output_dir=cve_world["output_dir"],
        denominator_pools_path=str(cve_world["pools_path"]),
    )

    prep = pretrainer.prepare(
        train_records=cve_world["train"],
        validation_records=cve_world["validation"],
    )

    # Config invariants enforced by the driver (Req 8.2 no augmentation, Req 9.1
    # frozen encoder default, "cve" adapter routing).
    assert config.domain_adapter == "cve"
    assert config.use_view_augmentation is False
    assert config.freeze_text_encoder is True

    # --- Paired inputs exist for both splits. ---
    assert prep.paired_train_path.exists()
    assert prep.paired_train_path.name == PAIRED_TRAIN_FILENAME
    assert prep.paired_val_path is not None
    assert prep.paired_val_path.exists()
    assert prep.paired_val_path.name == PAIRED_VAL_FILENAME

    # Every train anchor shares the common CWE -> all are paired (none excluded).
    assert prep.train_report.input_anchors == len(cve_world["train"])
    assert prep.train_report.paired_anchors == len(cve_world["train"])
    assert prep.train_report.excluded_anchors == 0
    assert prep.train_report.missing_positive_record == 0

    assert prep.val_report is not None
    assert prep.val_report.paired_anchors == len(cve_world["validation"])

    # --- Each paired line is a real anchor+positive pair. ---
    paired_lines = [
        json.loads(line)
        for line in prep.paired_train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(paired_lines) == prep.train_report.paired_anchors
    for row in paired_lines:
        assert POSITIVE_KEY in row, "each anchor carries a selected positive record"
        positive = row[POSITIVE_KEY]
        # The positive is a distinct, in-split CVE with its own encoder view.
        assert positive["cve"] != row["cve"]
        assert positive["encoder_view"].strip()
        assert row["encoder_view"].strip()

    # --- Selection reports written per split (Req 6.7 / 13.6). ---
    train_out = cve_world["output_dir"] / "train"
    val_out = cve_world["output_dir"] / "validation"
    assert (train_out / "negative_selection_report.json").exists()
    assert (train_out / "positive_selection_report.json").exists()
    assert (val_out / "negative_selection_report.json").exists()
    assert (val_out / "positive_selection_report.json").exists()

    # The view lookup covers the negative universe (union of loaded splits).
    expected_ids = {r["cve"] for r in cve_world["train"]} | {
        r["cve"] for r in cve_world["validation"]
    }
    assert set(prep.view_lookup.keys()) == expected_ids


# ---------------------------------------------------------------------------
# 2. End-to-end: best-by-validation checkpoint + frozen encoder
# ---------------------------------------------------------------------------


def _encoder_available(model_name: str) -> bool:
    """True when the sentence-transformer encoder can be constructed locally."""
    try:  # pragma: no cover - environment dependent
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_name)
        return True
    except Exception:  # pragma: no cover - offline / missing weights
        return False


def test_stage1_end_to_end_best_checkpoint_frozen_encoder(
    cve_world: Dict[str, Any],
) -> None:
    """A tiny end-to-end run yields a best-by-validation checkpoint with a frozen encoder.

    Heavy path: requires ``torch`` and a loadable sentence-transformer encoder.
    Skips gracefully when either is unavailable so the wiring test above still
    guards the pairing/selection behavior.
    """
    pytest.importorskip("torch")

    config = _make_config(cve_world["pools_path"])
    if not _encoder_available(config.text_encoder_model):
        pytest.skip(
            f"sentence-transformer encoder '{config.text_encoder_model}' "
            "is not available in this environment"
        )

    import torch  # noqa: F401  (imported after importorskip)

    pretrainer = Stage1ContrastivePretrainer(
        config=config,
        output_dir=cve_world["output_dir"],
        denominator_pools_path=str(cve_world["pools_path"]),
    )

    # Prepare pairs, then build the real trainer so we can inspect the encoder
    # freeze state (Req 9.1) before and after training.
    prep = pretrainer.prepare(
        train_records=cve_world["train"],
        validation_records=cve_world["validation"],
    )
    trainer = pretrainer.build_trainer(prep)

    # --- Req 9.1: encoder frozen, only the projection head trainable. ---
    assert trainer.freeze_text_encoder is True
    assert all(
        not p.requires_grad for p in trainer.text_encoder.parameters()
    ), "no sentence-transformer parameter should be trainable (encoder frozen)"
    assert any(
        p.requires_grad for p in trainer.model.parameters()
    ), "the projection head must remain trainable"

    # The trainer's validation path is wired to the paired validation input so
    # best-by-validation checkpointing runs (Req 8.3).
    assert Path(config.validation_path) == prep.paired_val_path

    # --- Run the tiny contrastive pretraining (Req 8.1). ---
    results = trainer.train(str(prep.paired_train_path))
    assert results is not None

    # --- Req 8.3: a best-by-validation checkpoint was produced. ---
    best_ckpt = cve_world["output_dir"] / "best_checkpoint.pt"
    assert best_ckpt.exists(), "best_checkpoint.pt (best-by-validation) must be saved"

    # The checkpoint carries the validation-loss metric it was selected by, plus
    # the frozen-encoder config it was trained under.
    checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    assert "val_loss" in checkpoint, "best checkpoint records the selecting val_loss"
    assert "model_state_dict" in checkpoint
    assert checkpoint["config"]["freeze_text_encoder"] is True
    assert checkpoint["config"]["domain_adapter"] == "cve"

    # Encoder remains frozen after training (no accidental unfreeze).
    assert all(not p.requires_grad for p in trainer.text_encoder.parameters())
