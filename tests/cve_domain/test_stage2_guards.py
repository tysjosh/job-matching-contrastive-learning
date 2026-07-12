"""Unit tests for the Stage 2 supervised fine-tuning guards (task 10.4).

Component under test: ``cve_domain/stage2.py`` — ``CVEStage2Trainer`` (the
missing-checkpoint hard stop) and ``detect_head_configuration`` (the absent-label
head-skip logic).

Two guards required by the spec are covered, both exercising only pure-Python
paths (no torch / no encoder load):

- **Missing Stage 1 checkpoint (Req 8.6).** Constructing ``CVEStage2Trainer`` with
  a Stage 1 checkpoint path that does not exist raises ``FileNotFoundError`` and
  the message reports the missing path. The check runs at construction time,
  before any training, so a misconfigured run fails fast.
- **Absent-label head skip (Req 8.7).** ``detect_head_configuration`` disables a
  head whose label is absent from every training record and records a skip reason:
  * no ``priority_score`` -> priority_score head skipped,
  * ``priority_band`` with fewer than 2 distinct classes -> band head skipped,
  * no ``in_kev`` / no ``ransomware`` -> those heads skipped,
  * all labels absent -> no heads enabled.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_stage2_guards.py

Requirements: 8.6, 8.7
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import pytest

from contrastive_learning.data_structures import TrainingConfig
from cve_domain.stage2 import (
    CVEStage2Trainer,
    detect_head_configuration,
    IN_KEV_HEAD,
    PRIORITY_BAND_HEAD,
    PRIORITY_SCORE_HEAD,
    RANSOMWARE_HEAD,
)


def _record(**labels: Any) -> Dict[str, Any]:
    """Build a minimal CVE_View_Record carrying the given supervised labels."""
    return {
        "cve": "CVE-0000-0000",
        "encoder_view": "some encoder view text",
        "cve_labels": dict(labels),
    }


# --------------------------------------------------------------------------- #
# Req 8.6: missing Stage 1 checkpoint -> hard stop before training
# --------------------------------------------------------------------------- #
class TestMissingCheckpointGuard:
    def test_nonexistent_checkpoint_raises_filenotfound_reporting_path(self, tmp_path):
        """Construction with a non-existent checkpoint path fails fast (Req 8.6)."""
        missing = tmp_path / "does_not_exist" / "stage1_best_checkpoint.pt"
        assert not missing.exists()

        with pytest.raises(FileNotFoundError) as exc_info:
            CVEStage2Trainer(
                config=TrainingConfig(),
                output_dir=tmp_path / "stage2_out",
                stage1_checkpoint_path=str(missing),
            )

        # The guard must report the exact missing path so the run is debuggable.
        assert str(missing) in str(exc_info.value)

    def test_missing_checkpoint_stops_before_torch_or_encoder(self, tmp_path):
        """The guard fires at construction, before any encoder/torch work (Req 8.6).

        No encoder should be loaded: the trainer's torch-heavy attributes are
        never populated because construction raises first.
        """
        missing = tmp_path / "nope.pt"
        with pytest.raises(FileNotFoundError):
            CVEStage2Trainer(
                config=TrainingConfig(),
                output_dir=tmp_path / "out",
                stage1_checkpoint_path=str(missing),
            )

    def test_existing_checkpoint_does_not_raise_at_construction(self, tmp_path):
        """A checkpoint path that exists passes the Req 8.6 guard (control case)."""
        ckpt = tmp_path / "stage1_best_checkpoint.pt"
        ckpt.write_bytes(b"not a real checkpoint")  # existence is all the guard checks
        out = tmp_path / "stage2_out"

        trainer = CVEStage2Trainer(
            config=TrainingConfig(),
            output_dir=out,
            stage1_checkpoint_path=str(ckpt),
        )

        assert trainer.stage1_checkpoint_path == str(ckpt)
        # torch-heavy state stays unbuilt until prepare()/train().
        assert trainer.text_encoder is None
        assert trainer.projection is None

    def test_no_checkpoint_configured_raises_valueerror(self, tmp_path):
        """No checkpoint path at all is a configuration error (ValueError)."""
        with pytest.raises(ValueError):
            CVEStage2Trainer(
                config=TrainingConfig(),
                output_dir=tmp_path / "out",
                stage1_checkpoint_path=None,
            )


# --------------------------------------------------------------------------- #
# Req 8.7: absent-label head skip via detect_head_configuration
# --------------------------------------------------------------------------- #
class TestAbsentLabelHeadSkip:
    def test_all_labels_present_enables_all_heads(self):
        """Sanity control: when every label is present, every head is enabled."""
        records = [
            _record(priority_score=90.0, priority_band="CRITICAL", in_kev=True, ransomware=True),
            _record(priority_score=10.0, priority_band="LOW", in_kev=False, ransomware=False),
        ]
        config = detect_head_configuration(records)

        assert config.enabled[PRIORITY_SCORE_HEAD] is True
        assert config.enabled[PRIORITY_BAND_HEAD] is True
        assert config.enabled[IN_KEV_HEAD] is True
        assert config.enabled[RANSOMWARE_HEAD] is True
        assert config.any_enabled() is True
        assert config.skip_reasons == {}
        assert config.band_vocabulary == ["CRITICAL", "LOW"]

    def test_absent_priority_score_skips_score_head_with_reason(self):
        """No priority_score anywhere -> score head disabled + skip reason (Req 8.7)."""
        records = [
            _record(priority_band="HIGH", in_kev=True, ransomware=False),
            _record(priority_band="LOW", in_kev=False, ransomware=False),
        ]
        config = detect_head_configuration(records)

        assert config.enabled[PRIORITY_SCORE_HEAD] is False
        assert PRIORITY_SCORE_HEAD in config.skip_reasons
        assert "priority_score" in config.skip_reasons[PRIORITY_SCORE_HEAD]
        # The other present labels stay enabled.
        assert config.enabled[PRIORITY_BAND_HEAD] is True
        assert config.enabled[IN_KEV_HEAD] is True

    def test_out_of_range_priority_score_counts_as_absent(self):
        """Only numeric priority_score within [0, 100] enables the head (Req 8.7)."""
        records = [
            _record(priority_score=150.0, priority_band="HIGH", in_kev=True),
            _record(priority_score="not-a-number", priority_band="LOW"),
        ]
        config = detect_head_configuration(records)

        assert config.enabled[PRIORITY_SCORE_HEAD] is False
        assert PRIORITY_SCORE_HEAD in config.skip_reasons

    def test_single_distinct_band_skips_band_head_with_reason(self):
        """priority_band with only one class is too few to train (Req 8.7)."""
        records = [
            _record(priority_score=50.0, priority_band="HIGH", in_kev=True, ransomware=False),
            _record(priority_score=60.0, priority_band="HIGH", in_kev=False, ransomware=False),
        ]
        config = detect_head_configuration(records)

        assert config.enabled[PRIORITY_BAND_HEAD] is False
        assert PRIORITY_BAND_HEAD in config.skip_reasons
        assert "one distinct class" in config.skip_reasons[PRIORITY_BAND_HEAD]
        assert config.band_vocabulary == ["HIGH"]

    def test_absent_priority_band_skips_band_head_with_reason(self):
        """No priority_band anywhere -> band head disabled + skip reason (Req 8.7)."""
        records = [
            _record(priority_score=50.0, in_kev=True, ransomware=False),
            _record(priority_score=60.0, in_kev=False, ransomware=True),
        ]
        config = detect_head_configuration(records)

        assert config.enabled[PRIORITY_BAND_HEAD] is False
        assert PRIORITY_BAND_HEAD in config.skip_reasons
        assert "absent" in config.skip_reasons[PRIORITY_BAND_HEAD]
        assert config.band_vocabulary == []

    def test_absent_in_kev_skips_in_kev_head_with_reason(self):
        """No in_kev anywhere -> in_kev head disabled + skip reason (Req 8.7)."""
        records = [
            _record(priority_score=50.0, priority_band="HIGH", ransomware=False),
            _record(priority_score=60.0, priority_band="LOW", ransomware=True),
        ]
        config = detect_head_configuration(records)

        assert config.enabled[IN_KEV_HEAD] is False
        assert IN_KEV_HEAD in config.skip_reasons
        assert "in_kev" in config.skip_reasons[IN_KEV_HEAD]
        # ransomware is present, so it stays enabled.
        assert config.enabled[RANSOMWARE_HEAD] is True

    def test_absent_ransomware_skips_ransomware_head_with_reason(self):
        """No ransomware anywhere -> ransomware head disabled + skip reason (Req 8.7)."""
        records = [
            _record(priority_score=50.0, priority_band="HIGH", in_kev=True),
            _record(priority_score=60.0, priority_band="LOW", in_kev=False),
        ]
        config = detect_head_configuration(records)

        assert config.enabled[RANSOMWARE_HEAD] is False
        assert RANSOMWARE_HEAD in config.skip_reasons
        assert "ransomware" in config.skip_reasons[RANSOMWARE_HEAD]

    def test_all_labels_absent_enables_no_heads(self):
        """Records with no labels at all -> no heads enabled (Req 8.7)."""
        records = [
            {"cve": "CVE-1", "encoder_view": "view one"},
            {"cve": "CVE-2", "encoder_view": "view two", "cve_labels": {}},
        ]
        config = detect_head_configuration(records)

        assert config.any_enabled() is False
        assert config.enabled[PRIORITY_SCORE_HEAD] is False
        assert config.enabled[PRIORITY_BAND_HEAD] is False
        assert config.enabled[IN_KEV_HEAD] is False
        assert config.enabled[RANSOMWARE_HEAD] is False
        # Every head has a recorded skip reason.
        for head in (PRIORITY_SCORE_HEAD, PRIORITY_BAND_HEAD, IN_KEV_HEAD, RANSOMWARE_HEAD):
            assert head in config.skip_reasons

    def test_empty_records_enables_no_heads(self):
        """No training records at all -> no heads enabled (Req 8.7 boundary)."""
        config = detect_head_configuration([])

        assert config.any_enabled() is False
        assert config.band_vocabulary == []
        for head in (PRIORITY_SCORE_HEAD, PRIORITY_BAND_HEAD, IN_KEV_HEAD, RANSOMWARE_HEAD):
            assert config.enabled[head] is False
            assert head in config.skip_reasons

    def test_label_present_on_only_one_record_enables_head(self):
        """A head is enabled when its label is present on at least one record (Req 8.7)."""
        records = [
            _record(priority_band="HIGH"),  # no score/in_kev/ransomware here
            _record(priority_score=42.0, priority_band="LOW", in_kev=True, ransomware=True),
        ]
        config = detect_head_configuration(records)

        # priority_score present on only the second record -> still enabled.
        assert config.enabled[PRIORITY_SCORE_HEAD] is True
        assert config.enabled[PRIORITY_BAND_HEAD] is True  # two distinct bands
        assert config.enabled[IN_KEV_HEAD] is True
        assert config.enabled[RANSOMWARE_HEAD] is True
