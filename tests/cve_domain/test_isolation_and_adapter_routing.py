"""Integration test for domain isolation and adapter routing (task 12.5).

This test covers two end-to-end concerns from the CVE spec:

1. **Adapter routing via ``Run_Config.domain_adapter``** (Req 7.6, 12.4): the
   active Domain_Adapter is selected purely through the ``domain_adapter``
   Run_Config field, not through edits to shared source files. A config with
   ``domain_adapter="cve"`` routes the ``DataLoader`` to ``CVERecordAdapter``; a
   config with ``domain_adapter="career"`` (or an absent field -> default) routes
   to ``CareerDomainAdapter``. An unknown adapter name is rejected.

2. **Artifact-location isolation** (Req 12.1): the example CVE Run_Configs place
   all of their data, config, and output paths under the CVE-isolated locations
   (``cve_domain/`` and ``cybersecurity-vulnerability-ranking/``), separate from
   the career-domain artifact roots (``config/``, ``preprocess/``, ``results_*``,
   ``training_output/``).

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_isolation_and_adapter_routing.py

Requirements: 7.6, 12.1, 12.4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pytest

from contrastive_learning.data_loader import DataLoader
from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.domain_adapters import (
    CareerDomainAdapter,
    get_domain_adapter,
)

# Importing the CVE record adapter registers it under "cve" as an import-time
# side effect (task 7.2). The career adapter is registered when
# ``contrastive_learning.domain_adapters`` is imported.
from cve_domain.record_adapter import CVERecordAdapter
from cve_domain.run_config import load_run_config


# --------------------------------------------------------------------------- #
# Repo layout / config locations
# --------------------------------------------------------------------------- #
# tests/cve_domain/<this file> -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE1_CONFIG = REPO_ROOT / "cve_domain" / "configs" / "stage1_contrastive.json"
STAGE2_CONFIG = REPO_ROOT / "cve_domain" / "configs" / "stage2_supervised.json"

# Top-level directory names that belong to the *career* domain. CVE artifacts
# must never be placed under any of these (Req 12.1).
CAREER_ARTIFACT_ROOTS = {"config", "preprocess", "training_output"}
CAREER_ARTIFACT_PREFIXES = ("results_", "data_splits")

# CVE-isolated top-level roots (Req 12.1).
CVE_OUTPUT_ROOT = "cve_domain"
CVE_DATA_ROOT = "cybersecurity-vulnerability-ranking"


# Keys in the example Run_Config JSON that name output/artifact locations. These
# are carried in the JSON for run wiring but are not fields on ``TrainingConfig``
# (``from_dict`` ignores unknown keys), so they are read from the raw JSON here.
OUTPUT_PATH_KEYS = ("output_dir", "checkpoint_path", "validation_path")


def _first_segment(path_value: str) -> str:
    """Return the first path segment of a (relative) config path string."""
    return Path(path_value).parts[0] if Path(path_value).parts else ""


def _raw_output_paths(config_path: Path) -> List[str]:
    """Read the output/artifact path strings straight from the Run_Config JSON."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return [str(data[k]) for k in OUTPUT_PATH_KEYS if data.get(k)]


def _assert_not_under_career(paths: Iterable[str], *, label: str) -> None:
    """Assert none of ``paths`` fall under a career-domain artifact root."""
    for value in paths:
        first = _first_segment(value)
        assert first not in CAREER_ARTIFACT_ROOTS, (
            f"{label} path {value!r} is under career artifact root {first!r}"
        )
        assert not first.startswith(CAREER_ARTIFACT_PREFIXES), (
            f"{label} path {value!r} is under a career artifact root {first!r}"
        )


# --------------------------------------------------------------------------- #
# 1. Adapter routing via Run_Config.domain_adapter (Req 7.6, 12.4)
# --------------------------------------------------------------------------- #
class TestAdapterRouting:
    """Adapter selection is driven solely by the ``domain_adapter`` field."""

    def test_cve_config_routes_to_cve_adapter(self) -> None:
        """domain_adapter="cve" -> DataLoader uses CVERecordAdapter."""
        config = TrainingConfig(domain_adapter="cve")

        # Direct seam resolution.
        adapter = get_domain_adapter("cve", config)
        assert isinstance(adapter, CVERecordAdapter)
        assert adapter.name == "cve"

        # End-to-end through the DataLoader (the consumer of the seam).
        loader = DataLoader(config)
        assert isinstance(loader._adapter, CVERecordAdapter)
        assert loader._adapter.name == "cve"

    def test_career_config_routes_to_career_adapter(self) -> None:
        """domain_adapter="career" -> DataLoader uses CareerDomainAdapter."""
        config = TrainingConfig(domain_adapter="career")

        adapter = get_domain_adapter("career", config)
        assert isinstance(adapter, CareerDomainAdapter)
        assert adapter.name == "career"

        loader = DataLoader(config)
        assert isinstance(loader._adapter, CareerDomainAdapter)
        assert loader._adapter.name == "career"

    def test_absent_domain_adapter_defaults_to_career(self) -> None:
        """An absent domain_adapter field -> default career routing (Req 12.5)."""
        # TrainingConfig's default for the field is "career".
        default_config = TrainingConfig()
        assert default_config.domain_adapter == "career"

        loader = DataLoader(default_config)
        assert isinstance(loader._adapter, CareerDomainAdapter)
        assert loader._adapter.name == "career"

    def test_unknown_adapter_name_raises(self) -> None:
        """An unknown adapter name is rejected, listing registered adapters."""
        config = TrainingConfig(domain_adapter="does-not-exist")

        with pytest.raises(KeyError) as excinfo:
            get_domain_adapter("does-not-exist", config)
        message = str(excinfo.value)
        assert "does-not-exist" in message
        # The error lists the registered adapters to ease diagnosis.
        assert "career" in message and "cve" in message

        # The DataLoader surfaces the same failure when handed a bad config.
        with pytest.raises(KeyError):
            DataLoader(config)

    def test_example_configs_select_cve_adapter(self) -> None:
        """Both example CVE Run_Configs declare the CVE adapter (Req 7.6, 12.4)."""
        stage1 = load_run_config(str(STAGE1_CONFIG))
        stage2 = load_run_config(str(STAGE2_CONFIG))

        assert stage1.domain_adapter == "cve"
        assert stage2.domain_adapter == "cve"

        # And that field alone routes the loader to the CVE adapter.
        for cfg in (stage1, stage2):
            loader = DataLoader(cfg)
            assert isinstance(loader._adapter, CVERecordAdapter)
            assert loader._adapter.name == "cve"


# --------------------------------------------------------------------------- #
# 2. Artifact-location isolation (Req 12.1)
# --------------------------------------------------------------------------- #
class TestArtifactIsolation:
    """CVE configs keep all paths under CVE-isolated roots, away from career."""

    def test_config_files_live_under_cve_domain(self) -> None:
        """The example configs themselves are isolated under cve_domain/."""
        for cfg_path in (STAGE1_CONFIG, STAGE2_CONFIG):
            assert cfg_path.exists(), f"missing example config: {cfg_path}"
            rel = cfg_path.relative_to(REPO_ROOT)
            assert rel.parts[0] == CVE_OUTPUT_ROOT

    @pytest.mark.parametrize("config_path", [STAGE1_CONFIG, STAGE2_CONFIG])
    def test_output_paths_under_cve_domain(self, config_path: Path) -> None:
        """output_dir / checkpoint / validation paths live under cve_domain/."""
        output_paths = _raw_output_paths(config_path)
        assert output_paths, "expected output/artifact paths in the Run_Config"
        for value in output_paths:
            assert value, "expected an output path to be set"
            assert _first_segment(value) == CVE_OUTPUT_ROOT, (
                f"output path {value!r} is not under {CVE_OUTPUT_ROOT}/"
            )
        _assert_not_under_career(output_paths, label="output")

    @pytest.mark.parametrize("config_path", [STAGE1_CONFIG, STAGE2_CONFIG])
    def test_input_paths_under_cve_data_root(self, config_path: Path) -> None:
        """All CVE input data paths live under the CVE data root, not career."""
        cfg = load_run_config(str(config_path))
        input_paths = [v for v in cfg.input_paths().values() if v]
        assert input_paths, "expected CVE input data paths to be set"
        for value in input_paths:
            assert _first_segment(value) == CVE_DATA_ROOT, (
                f"input path {value!r} is not under {CVE_DATA_ROOT}/"
            )
        _assert_not_under_career(input_paths, label="input")

    @pytest.mark.parametrize("config_path", [STAGE1_CONFIG, STAGE2_CONFIG])
    def test_all_config_paths_isolated_from_career(self, config_path: Path) -> None:
        """No path on the CVE config lands under any career artifact root."""
        cfg = load_run_config(str(config_path))
        all_paths = [
            *_raw_output_paths(config_path),
            cfg.pretrained_model_path,
            *[v for v in cfg.input_paths().values()],
        ]
        # Drop unset values before checking isolation.
        present = [v for v in all_paths if v]
        _assert_not_under_career(present, label="config")
        # Every present path is under one of the two CVE-isolated roots.
        for value in present:
            assert _first_segment(value) in {CVE_OUTPUT_ROOT, CVE_DATA_ROOT}, (
                f"path {value!r} is outside the CVE-isolated roots"
            )
