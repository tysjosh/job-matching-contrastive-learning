"""Unit tests for the CVE Run_Config guards and Run_Manifest recording (task 12.2).

Component under test: ``cve_domain/run_config.py`` — ``CVERunConfig`` (JSON loading
with the required-field and read/parse guards), ``RunManifest`` (provenance
recording + ``save()``), plus the ``load_run_config`` convenience wrapper and the
``CVERunConfigMissingFieldError`` / ``CVERunConfigReadError`` error types.

Coverage maps to the spec acceptance criteria:

- **Missing required field (Req 11.4).** A config dict / JSON missing any
  ``REQUIRED_FIELDS`` entry (e.g. ``split_strategy``, ``cve_csv_path``) raises
  ``CVERunConfigMissingFieldError`` naming the field, and ``from_dict`` stops
  *before* constructing / using the config (so an otherwise-invalid config still
  fails on the missing field first).
- **Invalid JSON / unreadable file (Req 11.5).** Malformed JSON, a non-existent
  file, and a non-object top-level JSON each raise ``CVERunConfigReadError``
  reporting the read/parse reason.
- **Unknown keys ignored.** Extra keys not in the schema don't break loading
  (``TrainingConfig`` convention).
- **Manifest recording (Req 11.2).** ``RunManifest.from_config`` records config
  values, input paths, seeds, output artifact paths, and the frozen-encoder
  setting, and ``save()`` writes ``run_manifest.json``.
- **Unfrozen-encoder recording (Req 9.2).** ``freeze_text_encoder=False`` is
  recorded as ``encoder_setting == "unfrozen"`` with ``unfrozen_override == True``.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_run_config_manifest.py

Requirements: 9.2, 11.2, 11.4, 11.5
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from cve_domain.run_config import (
    CVE_DOMAIN_ADAPTER,
    RUN_MANIFEST_FILENAME,
    CVERunConfig,
    CVERunConfigMissingFieldError,
    CVERunConfigReadError,
    RunManifest,
    load_run_config,
)


def _valid_config_dict(**overrides: Any) -> Dict[str, Any]:
    """A minimal dict carrying every ``CVERunConfig.REQUIRED_FIELDS`` entry."""
    base: Dict[str, Any] = {
        "domain_adapter": CVE_DOMAIN_ADAPTER,
        "split_strategy": "stratified",
        "split_seed": 7,
        "max_negatives_per_anchor": 12,
        "negative_tier_ratios": {"hard": 0.34, "medium": 0.33, "easy": 0.33},
        "freeze_text_encoder": True,
        "cve_csv_path": "data/cve.csv",
        "cve_profiles_path": "data/profiles.jsonl",
        "cve_denominator_pools_path": "data/pools.jsonl",
    }
    base.update(overrides)
    return base


def _write_json(path, payload: str) -> str:
    path.write_text(payload, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# Req 11.4: missing required field -> stop before training, report the field
# --------------------------------------------------------------------------- #
class TestMissingRequiredField:
    def test_all_required_fields_present_loads_successfully(self):
        """Control: a dict with every required field loads (no guard trips)."""
        config = CVERunConfig.from_dict(_valid_config_dict())
        assert isinstance(config, CVERunConfig)
        assert config.split_strategy == "stratified"
        assert config.cve_csv_path == "data/cve.csv"

    @pytest.mark.parametrize("field_name", list(CVERunConfig.REQUIRED_FIELDS))
    def test_each_missing_required_field_raises_naming_that_field(self, field_name):
        """Dropping any one required field raises, naming that field (Req 11.4)."""
        data = _valid_config_dict()
        del data[field_name]

        with pytest.raises(CVERunConfigMissingFieldError) as exc_info:
            CVERunConfig.from_dict(data)

        err = exc_info.value
        assert err.field_name == field_name
        assert field_name in str(err)
        assert field_name in err.missing_fields

    def test_split_strategy_missing_reports_split_strategy(self):
        """Named case from the task: missing split_strategy is reported (Req 11.4)."""
        data = _valid_config_dict()
        del data["split_strategy"]
        with pytest.raises(CVERunConfigMissingFieldError) as exc_info:
            CVERunConfig.from_dict(data)
        assert exc_info.value.field_name == "split_strategy"

    def test_cve_csv_path_missing_reports_cve_csv_path(self):
        """Named case from the task: missing cve_csv_path is reported (Req 11.4)."""
        data = _valid_config_dict()
        del data["cve_csv_path"]
        with pytest.raises(CVERunConfigMissingFieldError) as exc_info:
            CVERunConfig.from_dict(data)
        assert exc_info.value.field_name == "cve_csv_path"

    def test_multiple_missing_fields_reports_first_and_lists_all(self):
        """When several are missing, the first is named and all are listed (Req 11.4)."""
        data = _valid_config_dict()
        del data["split_strategy"]
        del data["freeze_text_encoder"]
        with pytest.raises(CVERunConfigMissingFieldError) as exc_info:
            CVERunConfig.from_dict(data)
        err = exc_info.value
        # First missing in schema order is split_strategy.
        assert err.field_name == "split_strategy"
        assert set(err.missing_fields) == {"split_strategy", "freeze_text_encoder"}

    def test_missing_field_check_runs_before_config_construction(self):
        """from_dict stops before *using* the values (Req 11.4).

        A config value that would fail ``TrainingConfig.__post_init__`` (here an
        invalid ``batch_size``) must NOT be reached: the missing-field guard fires
        first, so the caller sees the actionable missing-field error rather than a
        downstream construction error.
        """
        data = _valid_config_dict(batch_size=-5)  # would raise ValueError if constructed
        del data["split_strategy"]

        with pytest.raises(CVERunConfigMissingFieldError):
            CVERunConfig.from_dict(data)

    def test_non_dict_input_raises_read_error(self):
        """A non-dict payload is a read/parse-shaped failure (Req 11.5-adjacent)."""
        with pytest.raises(CVERunConfigReadError):
            CVERunConfig.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Req 11.5: unreadable / invalid JSON -> stop, report the reason
# --------------------------------------------------------------------------- #
class TestInvalidOrUnreadableJson:
    def test_nonexistent_file_raises_read_error_with_path_and_reason(self, tmp_path):
        """A missing file stops the run and reports the read failure (Req 11.5)."""
        missing = tmp_path / "no_such_config.json"
        with pytest.raises(CVERunConfigReadError) as exc_info:
            CVERunConfig.from_json(str(missing))
        err = exc_info.value
        assert err.file_path == str(missing)
        assert "cannot read file" in err.reason

    def test_malformed_json_raises_read_error_reporting_invalid_json(self, tmp_path):
        """Malformed JSON stops the run and reports the parse failure (Req 11.5)."""
        bad = _write_json(tmp_path / "bad.json", "{ this is not valid json ")
        with pytest.raises(CVERunConfigReadError) as exc_info:
            CVERunConfig.from_json(bad)
        assert "invalid JSON" in exc_info.value.reason

    def test_non_object_top_level_json_raises_read_error(self, tmp_path):
        """A JSON array/scalar at the top level is rejected (Req 11.5)."""
        arr = _write_json(tmp_path / "arr.json", "[1, 2, 3]")
        with pytest.raises(CVERunConfigReadError) as exc_info:
            CVERunConfig.from_json(arr)
        assert "object" in exc_info.value.reason

    def test_valid_json_missing_field_still_raises_missing_field_error(self, tmp_path):
        """Readable, parseable JSON missing a required field -> Req 11.4 error."""
        data = _valid_config_dict()
        del data["cve_profiles_path"]
        path = _write_json(tmp_path / "cfg.json", json.dumps(data))
        with pytest.raises(CVERunConfigMissingFieldError) as exc_info:
            CVERunConfig.from_json(path)
        assert exc_info.value.field_name == "cve_profiles_path"

    def test_load_run_config_wrapper_loads_valid_file(self, tmp_path):
        """The convenience wrapper loads a valid Run_Config JSON (Req 11.1)."""
        path = _write_json(tmp_path / "cfg.json", json.dumps(_valid_config_dict()))
        config = load_run_config(path)
        assert isinstance(config, CVERunConfig)
        assert config.split_seed == 7


# --------------------------------------------------------------------------- #
# Unknown keys ignored (TrainingConfig convention)
# --------------------------------------------------------------------------- #
class TestUnknownKeysIgnored:
    def test_extra_unknown_keys_do_not_break_loading(self):
        """Extra keys outside the schema are ignored, not fatal."""
        data = _valid_config_dict(
            some_future_field="ignored",
            another_unknown={"nested": True},
        )
        config = CVERunConfig.from_dict(data)
        assert isinstance(config, CVERunConfig)
        assert not hasattr(config, "some_future_field")

    def test_unknown_keys_ignored_when_loading_from_json(self, tmp_path):
        """Unknown keys are also ignored on the JSON load path."""
        data = _valid_config_dict(unexpected="value")
        path = _write_json(tmp_path / "cfg.json", json.dumps(data))
        config = load_run_config(path)
        assert config.max_negatives_per_anchor == 12


# --------------------------------------------------------------------------- #
# Req 11.2: manifest records config values, paths, seeds, outputs, encoder
# --------------------------------------------------------------------------- #
class TestManifestRecording:
    def test_from_config_records_all_required_provenance(self):
        """Manifest captures config, input paths, seeds, outputs, encoder (Req 11.2)."""
        config = CVERunConfig.from_dict(
            _valid_config_dict(cyber_kg_path="data/kg.jsonl", training_seed=99)
        )
        outputs = {
            "checkpoint": "runs/stage1/best.pt",
            "conversion_report": "runs/conversion_report.json",
        }
        manifest = RunManifest.from_config(
            config, output_artifact_paths=outputs, run_id="run-123"
        )

        assert manifest.run_id == "run-123"
        assert manifest.domain_adapter == CVE_DOMAIN_ADAPTER
        # Config values recorded in full.
        assert manifest.config["split_strategy"] == "stratified"
        assert manifest.config["max_negatives_per_anchor"] == 12
        # Input paths recorded.
        assert manifest.input_paths["cve_csv_path"] == "data/cve.csv"
        assert manifest.input_paths["cve_profiles_path"] == "data/profiles.jsonl"
        assert manifest.input_paths["cve_denominator_pools_path"] == "data/pools.jsonl"
        assert manifest.input_paths["cyber_kg_path"] == "data/kg.jsonl"
        # Seeds recorded.
        assert manifest.seeds["split_seed"] == 7
        assert manifest.seeds["training_seed"] == 99
        # Output artifact paths recorded verbatim.
        assert manifest.output_artifact_paths == outputs
        # Frozen-encoder setting recorded.
        assert manifest.freeze_text_encoder is True
        assert manifest.encoder_setting == "frozen"
        assert manifest.unfrozen_override is False

    def test_build_manifest_helper_matches_from_config(self):
        """CVERunConfig.build_manifest delegates to RunManifest.from_config."""
        config = CVERunConfig.from_dict(_valid_config_dict())
        manifest = config.build_manifest(
            output_artifact_paths={"out": "runs/out"}, run_id="r1"
        )
        assert isinstance(manifest, RunManifest)
        assert manifest.run_id == "r1"
        assert manifest.output_artifact_paths == {"out": "runs/out"}

    def test_to_dict_is_json_serializable_and_structured(self):
        """Manifest.to_dict is JSON-ready and nests the encoder block (Req 11.2)."""
        config = CVERunConfig.from_dict(_valid_config_dict())
        manifest = RunManifest.from_config(config, run_id="r2")
        payload = manifest.to_dict()

        # Round-trips through JSON without error.
        reloaded = json.loads(json.dumps(payload))
        assert reloaded["run_id"] == "r2"
        assert reloaded["encoder"]["encoder_setting"] == "frozen"
        assert reloaded["encoder"]["freeze_text_encoder"] is True
        assert reloaded["seeds"]["split_seed"] == 7
        assert "config" in reloaded and reloaded["config"]["domain_adapter"] == CVE_DOMAIN_ADAPTER

    def test_save_writes_run_manifest_json(self, tmp_path):
        """save() writes run_manifest.json under the output dir (Req 11.2)."""
        config = CVERunConfig.from_dict(_valid_config_dict())
        manifest = RunManifest.from_config(config, run_id="saved-run")
        out_dir = tmp_path / "nested" / "run_out"

        written_path = manifest.save(str(out_dir))

        expected = out_dir / RUN_MANIFEST_FILENAME
        assert written_path == str(expected)
        assert expected.exists()
        on_disk = json.loads(expected.read_text(encoding="utf-8"))
        assert on_disk["run_id"] == "saved-run"
        assert on_disk["encoder"]["encoder_setting"] == "frozen"


# --------------------------------------------------------------------------- #
# Req 9.2: unfrozen encoder is recorded as an explicit override
# --------------------------------------------------------------------------- #
class TestUnfrozenEncoderRecording:
    def test_unfrozen_config_records_unfrozen_setting_and_override(self):
        """freeze_text_encoder=False -> unfrozen + override flag (Req 9.2)."""
        config = CVERunConfig.from_dict(
            _valid_config_dict(freeze_text_encoder=False)
        )
        manifest = RunManifest.from_config(config)

        assert manifest.freeze_text_encoder is False
        assert manifest.encoder_setting == "unfrozen"
        assert manifest.unfrozen_override is True

    def test_unfrozen_setting_survives_serialization(self, tmp_path):
        """The unfrozen override is persisted to run_manifest.json (Req 9.2)."""
        config = CVERunConfig.from_dict(
            _valid_config_dict(freeze_text_encoder=False)
        )
        manifest = RunManifest.from_config(config)
        path = manifest.save(str(tmp_path / "unfrozen_run"))

        on_disk = json.loads(open(path, encoding="utf-8").read())
        assert on_disk["encoder"]["encoder_setting"] == "unfrozen"
        assert on_disk["encoder"]["unfrozen_override"] is True
        assert on_disk["encoder"]["freeze_text_encoder"] is False

    def test_frozen_default_records_frozen_without_override(self):
        """The frozen default (Req 9.3) records no override (contrast to Req 9.2)."""
        # freeze_text_encoder present + True (the frozen default value).
        config = CVERunConfig.from_dict(_valid_config_dict(freeze_text_encoder=True))
        manifest = RunManifest.from_config(config)
        assert manifest.encoder_setting == "frozen"
        assert manifest.unfrozen_override is False
