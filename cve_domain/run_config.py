"""CVE Run_Config + Run_Manifest — config-driven, auditable runs (Requirement 11).

This module implements the CVE-domain Run_Config and Run_Manifest described in the
design ("10. CVERunConfig + Run_Manifest"). It reuses the existing
:class:`~contrastive_learning.data_structures.TrainingConfig` conventions rather
than reinventing them:

* :class:`CVERunConfig` **extends** ``TrainingConfig`` (JSON-loadable, ``from_dict``
  ignores unknown keys) and inherits the additive CVE fields already added to
  ``TrainingConfig`` (task 1.1): ``domain_adapter``, ``split_strategy``,
  ``split_seed``, ``split_proportions``, ``max_negatives_per_anchor``,
  ``negative_tier_ratios``, ``freeze_text_encoder``, and the ``cve_*_path`` inputs.
* Loading enforces two guard rails required by Requirement 11:
    - a **missing required field** stops before training and reports the field name
      (Req 11.4) via :class:`CVERunConfigMissingFieldError`;
    - an **unreadable file or invalid JSON** stops before training and reports the
      read/parse reason (Req 11.5) via :class:`CVERunConfigReadError`.
* :class:`RunManifest` records the Run_Config values, input data paths, seeds,
  output artifact paths, and the frozen/unfrozen encoder setting (Req 11.2), and
  explicitly flags an **unfrozen override** so an unfrozen encoder run is recorded
  (Req 9.2). The frozen-encoder setting is exposed as a Run_Config field with a
  frozen default (Req 9.3), inherited from ``TrainingConfig.freeze_text_encoder``.

The Run_Config schema includes fields for the split strategy, split seed, maximum
negatives per anchor, per-tier negative ratios, and the frozen-encoder setting
(Req 11.6), all inherited from ``TrainingConfig``.

Design note on reuse: ``run_manifests/manifest_adapter.py`` translates *research*
YAML manifests into pipeline configs — a different concern from recording a single
run's provenance. This module therefore records a per-run manifest JSON artifact in
the same additive spirit, keeping CVE artifacts isolated from career-domain outputs
(Req 12.1).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from contrastive_learning.data_structures import TrainingConfig

# The domain adapter value that selects the CVE record adapter.
CVE_DOMAIN_ADAPTER = "cve"

# Standard Run_Manifest artifact filename.
RUN_MANIFEST_FILENAME = "run_manifest.json"


# --------------------------------------------------------------------------- #
# Errors (stop-before-training guards, Req 11.4 / 11.5)
# --------------------------------------------------------------------------- #
class CVERunConfigError(Exception):
    """Base class for Run_Config load failures that stop before training."""


class CVERunConfigMissingFieldError(CVERunConfigError):
    """Raised when a required Run_Config field is missing (Req 11.4).

    Attributes:
        field_name: The first required field found missing (reported to the user).
        missing_fields: All required fields found missing, in schema order.
    """

    def __init__(self, field_name: str, missing_fields: Optional[List[str]] = None) -> None:
        self.field_name = field_name
        self.missing_fields = list(missing_fields or [field_name])
        super().__init__(f"Missing required Run_Config field: {field_name!r}")


class CVERunConfigReadError(CVERunConfigError):
    """Raised when the Run_Config file cannot be read or parsed (Req 11.5).

    Attributes:
        file_path: The Run_Config path whose read/parse failed.
        reason: A human-readable description of the read or parse failure.
    """

    def __init__(self, file_path: str, reason: str) -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Cannot load Run_Config {file_path!r}: {reason}")


# --------------------------------------------------------------------------- #
# CVERunConfig
# --------------------------------------------------------------------------- #
@dataclass
class CVERunConfig(TrainingConfig):
    """CVE-domain Run_Config, extending ``TrainingConfig`` conventions (Req 11).

    All CVE fields (``domain_adapter``, ``split_strategy``, ``split_seed``,
    ``split_proportions``, ``max_negatives_per_anchor``, ``negative_tier_ratios``,
    ``freeze_text_encoder``, ``cve_*_path``) are inherited from ``TrainingConfig``
    (added in task 1.1). This subclass adds JSON loading with the Req 11.4 / 11.5
    guards and Run_Manifest recording; it introduces no new dataclass fields so
    existing configs continue to load unchanged.

    The frozen-encoder setting is exposed as the inherited ``freeze_text_encoder``
    field with a frozen default of ``True`` (Req 9.3).
    """

    # Fields a CVE Run_Config MUST include. Absence of any of these stops the run
    # before training and reports the field name (Req 11.4). This set combines the
    # Req 11.6 schema fields (split strategy, split seed, max negatives per anchor,
    # per-tier negative ratios, frozen-encoder setting) with the CVE input data
    # paths a run cannot execute without. Declared as ClassVar so the dataclass
    # machinery does not treat it as a field.
    REQUIRED_FIELDS: ClassVar[Tuple[str, ...]] = (
        # Req 11.6 schema fields.
        "split_strategy",
        "split_seed",
        "max_negatives_per_anchor",
        "negative_tier_ratios",
        "freeze_text_encoder",
        # CVE input data paths (a run cannot execute without these).
        "cve_csv_path",
        "cve_profiles_path",
        "cve_denominator_pools_path",
    )

    # ------------------------------------------------------------------ #
    # Loading (Req 11.4 / 11.5)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CVERunConfig":
        """Build a Run_Config from a dict, enforcing required fields (Req 11.4).

        Unknown keys are ignored (``TrainingConfig`` convention). A missing
        required field raises :class:`CVERunConfigMissingFieldError` naming the
        field, stopping before training.
        """
        if not isinstance(data, dict):
            raise CVERunConfigReadError("<dict>", "Run_Config must be a JSON object / dict")

        missing = [name for name in cls.REQUIRED_FIELDS if name not in data]
        if missing:
            # Report the first missing field by name (Req 11.4); carry the full
            # list for callers/tests that want it.
            raise CVERunConfigMissingFieldError(missing[0], missing)

        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, file_path: str) -> "CVERunConfig":
        """Load a Run_Config JSON file (Req 11.1), guarding read/parse (Req 11.5).

        Stops before training and reports the reason when the file cannot be read
        or cannot be parsed as valid JSON.
        """
        path = Path(file_path)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CVERunConfigReadError(str(file_path), f"cannot read file: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CVERunConfigReadError(str(file_path), f"invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise CVERunConfigReadError(
                str(file_path), "top-level JSON must be an object mapping field names to values"
            )
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def input_paths(self) -> Dict[str, Optional[str]]:
        """Return the CVE input data paths recorded on this config (Req 11.2)."""
        return {
            "cve_csv_path": self.cve_csv_path,
            "cve_profiles_path": self.cve_profiles_path,
            "cve_denominator_pools_path": self.cve_denominator_pools_path,
            "cyber_kg_path": self.cyber_kg_path,
        }

    def seeds(self) -> Dict[str, int]:
        """Return the seeds governing reproducible splits/selection (Req 11.2/11.3)."""
        return {
            "split_seed": int(self.split_seed),
            "training_seed": int(self.training_seed),
        }

    def build_manifest(
        self,
        output_artifact_paths: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
    ) -> "RunManifest":
        """Produce the Run_Manifest for this config (Req 11.2, 9.2)."""
        return RunManifest.from_config(
            self,
            output_artifact_paths=output_artifact_paths or {},
            run_id=run_id,
        )


# --------------------------------------------------------------------------- #
# Run_Manifest (Req 11.2, 9.2)
# --------------------------------------------------------------------------- #
@dataclass
class RunManifest:
    """Recorded provenance of a CVE run (see design "Run_Manifest").

    Captures the Run_Config values, input data paths, seeds, and output artifact
    paths (Req 11.2), and records the frozen/unfrozen encoder setting — explicitly
    flagging an unfrozen override so an unfrozen run is recorded (Req 9.2/9.3).

    Attributes:
        run_id: Optional caller-supplied identifier for the run.
        created_at: UTC ISO-8601 timestamp when the manifest was built.
        domain_adapter: The active Domain_Adapter (e.g. ``"cve"``).
        config: The full Run_Config values (``TrainingConfig.to_dict``).
        input_paths: The CVE input data paths (csv/profiles/pools/kg).
        seeds: The split/training seeds governing reproducibility.
        output_artifact_paths: Where run outputs are/will be written.
        freeze_text_encoder: The raw frozen-encoder setting from the config.
        encoder_setting: ``"frozen"`` or ``"unfrozen"`` (human-readable).
        unfrozen_override: ``True`` when the encoder is unfrozen (a deliberate
            override of the frozen default), recorded per Req 9.2.
    """

    run_id: Optional[str]
    created_at: str
    domain_adapter: str
    config: Dict[str, Any]
    input_paths: Dict[str, Optional[str]]
    seeds: Dict[str, int]
    output_artifact_paths: Dict[str, str]
    freeze_text_encoder: bool
    encoder_setting: str
    unfrozen_override: bool

    @classmethod
    def from_config(
        cls,
        config: CVERunConfig,
        output_artifact_paths: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
    ) -> "RunManifest":
        """Build a manifest from a Run_Config (Req 11.2, 9.2)."""
        frozen = bool(config.freeze_text_encoder)
        return cls(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            domain_adapter=config.domain_adapter,
            config=config.to_dict(),
            input_paths=config.input_paths(),
            seeds=config.seeds(),
            output_artifact_paths=dict(output_artifact_paths or {}),
            freeze_text_encoder=frozen,
            encoder_setting="frozen" if frozen else "unfrozen",
            # Default is frozen (Req 9.3); any unfrozen encoder is an explicit
            # override that must be recorded (Req 9.2).
            unfrozen_override=not frozen,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the manifest to a plain dict (JSON-ready)."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "domain_adapter": self.domain_adapter,
            "seeds": self.seeds,
            "input_paths": self.input_paths,
            "output_artifact_paths": self.output_artifact_paths,
            "encoder": {
                "freeze_text_encoder": self.freeze_text_encoder,
                "encoder_setting": self.encoder_setting,
                "unfrozen_override": self.unfrozen_override,
            },
            "config": self.config,
        }

    def save(self, output_dir: str, filename: str = RUN_MANIFEST_FILENAME) -> str:
        """Write the manifest JSON under ``output_dir`` and return its path."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        manifest_path = out_path / filename
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return str(manifest_path)


def load_run_config(file_path: str) -> CVERunConfig:
    """Load and validate a CVE Run_Config JSON file (Req 11.1, 11.4, 11.5).

    Convenience wrapper around :meth:`CVERunConfig.from_json`.
    """
    return CVERunConfig.from_json(file_path)
