"""Integration test for CVE run reproducibility at the Run_Config level (Req 11.3).

Requirement 11.3 (design "10. CVERunConfig + Run_Manifest"): *Same Run_Config +
seeds → identical splits and identical negative selections.*

This test ties together the two reproducibility guarantees that a single
``CVERunConfig`` governs through its ``split_seed``:

* the :class:`~cve_domain.data_splitter.CVEDataSplitter` (task 6.1) — running it
  twice from the same config produces byte-identical ``split_indices.json`` and
  identical per-split membership; and
* the :class:`~cve_domain.negative_selector.CVENegativeSelector` (task 7.4),
  reached through the :class:`~cve_domain.ontology_adapter.CVEOntologyAdapter`
  over the same ``cve_denominator_pools.jsonl`` — running
  ``select_for_split`` twice produces identical per-anchor negative selections.

It also asserts :class:`~cve_domain.positive_selector.CVEPositiveSelector`
reproducibility under the same seed (task 7.10), since that selector is likewise
seeded from the Run_Config ``split_seed``.

Everything is driven from a single ``CVERunConfig`` instance so the test exercises
reproducibility "at the Run_Config level" rather than through ad-hoc arguments:
the splitter, negative selector, and positive selector all read their strategy,
proportions, seed, tier ratios, and max-negatives from that one config.

Requirements: 11.3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

# Make the repository root importable when pytest is invoked from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cve_domain.data_splitter import SPLIT_NAMES, CVEDataSplitter  # noqa: E402
from cve_domain.negative_selector import CVENegativeSelector  # noqa: E402
from cve_domain.ontology_adapter import CVEOntologyAdapter  # noqa: E402
from cve_domain.positive_selector import CVEPositiveSelector  # noqa: E402
from cve_domain.run_config import CVERunConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic dataset builders
# --------------------------------------------------------------------------- #
# A small, deterministic pool of ontology tokens so that generated records share
# CWEs / CPEs / vendors — this makes both negative pools and positive selection
# meaningful (siblings actually exist) rather than trivially empty.
_BANDS = ["critical", "high", "medium", "low"]
_CWES = ["CWE-22", "CWE-79", "CWE-89", "CWE-416", "CWE-787"]
_VENDORS = ["jetbrains", "microsoft", "adobe", "cisco", "apache"]


def _build_records(n: int) -> List[Dict[str, Any]]:
    """Build ``n`` synthetic CVE_View_Records with overlapping ontology structure.

    IDs are unique so per-split membership is well defined. Bands cycle across the
    four canonical bands so the stratified strategy has multiple non-trivial
    strata. Ontology tokens are drawn from the small shared vocabularies above so
    anchors have real CWE / vendor siblings within a split.
    """
    records: List[Dict[str, Any]] = []
    for i in range(n):
        cve = f"CVE-2024-{i:07d}"
        cwe = _CWES[i % len(_CWES)]
        cwe2 = _CWES[(i + 1) % len(_CWES)]
        vendor = _VENDORS[i % len(_VENDORS)]
        records.append(
            {
                "cve": cve,
                "encoder_view": f"{cve}. synthetic vulnerability {i}. {cwe}.",
                "nvd_published": f"2024-01-01T00:{(i // 60) % 60:02d}:{i % 60:02d}",
                "cve_labels": {
                    "priority_band": _BANDS[i % len(_BANDS)],
                    "priority_score": float(i % 100),
                    "in_kev": (i % 2 == 0),
                    "ransomware": (i % 3 == 0),
                },
                "ontology": {
                    "cwes": [cwe, cwe2],
                    "cpes": [f"cpe:2.3:a:{vendor}:product{i % 7}:*:*:*:*:*:*:*:*"],
                    "vendors": [vendor],
                },
            }
        )
    return records


def _write_denominator_pools(records: List[Dict[str, Any]], path: Path) -> None:
    """Write a ``cve_denominator_pools.jsonl`` referencing ids within the dataset.

    Each anchor's tiers reference other real ids in the dataset (so negatives are
    valid and present), with deliberately uneven tier population so the selection
    draws from all three tiers.
    """
    ids = [r["cve"] for r in records]
    n = len(ids)
    lines: List[str] = []
    for idx, cve in enumerate(ids):
        # Deterministic, id-relative references that stay within the dataset and
        # never include the anchor itself.
        hard = [ids[(idx + off) % n] for off in (1, 2, 3) if ids[(idx + off) % n] != cve]
        medium = [ids[(idx + off) % n] for off in (4, 5, 6, 7) if ids[(idx + off) % n] != cve]
        easy = [ids[(idx + off) % n] for off in (8, 9, 10, 11, 12) if ids[(idx + off) % n] != cve]
        pool = {
            "cve": cve,
            "hard_negatives": hard,
            "medium_negatives": medium,
            "easy_negatives": easy,
        }
        lines.append(json.dumps(pool))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_config(pools_path: Path, seed: int = 4242) -> CVERunConfig:
    """Build a fully-specified CVERunConfig that governs the whole run.

    Includes every REQUIRED_FIELD so ``from_dict`` succeeds, plus the paths and
    seed the splitter / selectors read.
    """
    return CVERunConfig.from_dict(
        {
            "domain_adapter": "cve",
            "split_strategy": "stratified",
            "split_seed": seed,
            "split_proportions": {"train": 80, "validation": 10, "test": 10},
            "max_negatives_per_anchor": 8,
            "negative_tier_ratios": {"hard": 0.5, "medium": 0.3, "easy": 0.2},
            "freeze_text_encoder": True,
            "cve_csv_path": "unused.csv",
            "cve_profiles_path": "unused.jsonl",
            "cve_denominator_pools_path": str(pools_path),
        }
    )


# --------------------------------------------------------------------------- #
# Config-driven run helpers (everything derives from the one CVERunConfig)
# --------------------------------------------------------------------------- #
def _run_split_from_config(
    config: CVERunConfig, records: List[Mapping[str, Any]], out_dir: Path
) -> Dict[str, Any]:
    """Run the splitter using ONLY values read off the Run_Config."""
    splitter = CVEDataSplitter(
        strategy=config.split_strategy,
        proportions=config.split_proportions,
        seed=config.split_seed,
    )
    splitter.split_records(records, str(out_dir))

    indices = json.loads((out_dir / "split_indices.json").read_bytes())
    return {
        "splits": indices["splits"],
        "indices_bytes": (out_dir / "split_indices.json").read_bytes(),
    }


def _load_split_records(out_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Read back the JSONL for one split as records (input to the selectors)."""
    records: List[Dict[str, Any]] = []
    for line in (out_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _run_negatives_from_config(
    config: CVERunConfig, split_records: List[Mapping[str, Any]], out_dir: Path
) -> Dict[str, List[str]]:
    """Run negative selection via the ontology adapter, seeded from the config."""
    ontology_adapter = CVEOntologyAdapter(config.cve_denominator_pools_path)
    selector = CVENegativeSelector.from_config(ontology_adapter, config)
    return selector.select_for_split(split_records, output_dir=str(out_dir))


def _run_positives_from_config(
    config: CVERunConfig, split_records: List[Mapping[str, Any]], out_dir: Path
) -> Dict[str, str]:
    """Run positive selection seeded from the config's split seed."""
    selector = CVEPositiveSelector.from_config(config)
    return selector.select_for_split(split_records, output_dir=str(out_dir))


# --------------------------------------------------------------------------- #
# The reproducibility integration test (Req 11.3)
# --------------------------------------------------------------------------- #
def test_same_run_config_and_seed_reproduces_splits_and_negatives(tmp_path: Path) -> None:
    """Two runs from the same Run_Config produce identical splits AND negatives.

    This is the Req 11.3 guarantee end-to-end: a single ``CVERunConfig`` with a
    fixed ``split_seed`` drives (1) the splitter, (2) the negative selector via the
    ontology adapter, and (3) the positive selector — and every one of them is
    byte-for-byte reproducible across two independent runs.
    """
    records = _build_records(120)

    # One shared denominator-pools artifact + one shared config for both runs.
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_denominator_pools(records, pools_path)
    config = _make_config(pools_path, seed=4242)

    # ---- Run A ----------------------------------------------------------- #
    split_a = _run_split_from_config(config, records, tmp_path / "run_a_split")
    train_a = _load_split_records(tmp_path / "run_a_split", "train")
    negatives_a = _run_negatives_from_config(config, train_a, tmp_path / "run_a_neg")
    positives_a = _run_positives_from_config(config, train_a, tmp_path / "run_a_pos")

    # ---- Run B (independent objects, same config + seed) ----------------- #
    split_b = _run_split_from_config(config, records, tmp_path / "run_b_split")
    train_b = _load_split_records(tmp_path / "run_b_split", "train")
    negatives_b = _run_negatives_from_config(config, train_b, tmp_path / "run_b_neg")
    positives_b = _run_positives_from_config(config, train_b, tmp_path / "run_b_pos")

    # ---- Identical splits (Req 11.3, splitter reproducibility) ----------- #
    assert split_a["splits"] == split_b["splits"], (
        "per-split membership differs across identical-config runs"
    )
    assert split_a["indices_bytes"] == split_b["indices_bytes"], (
        "split_indices.json is not byte-identical across identical-config runs"
    )
    for name in SPLIT_NAMES:
        assert set(split_a["splits"][name]) == set(split_b["splits"][name])

    # Sanity: the negative-selection input (the train split) is itself identical.
    assert [r["cve"] for r in train_a] == [r["cve"] for r in train_b]

    # ---- Identical negative selections (Req 11.3) ------------------------ #
    assert negatives_a == negatives_b, (
        "per-anchor negative selections differ across identical-config runs"
    )
    # And the selection is non-trivial (the pools were populated), so the equality
    # above is meaningful rather than two empty maps.
    assert any(len(negs) > 0 for negs in negatives_a.values()), (
        "expected non-empty negative selections from populated pools"
    )

    # ---- Identical positive selections (Req 11.3, bonus — task 7.10) ----- #
    assert positives_a == positives_b, (
        "per-anchor positive selections differ across identical-config runs"
    )
    assert positives_a, "expected at least one resolved ontology positive"


def test_negative_selector_reproducible_on_repeated_calls(tmp_path: Path) -> None:
    """Calling ``select_for_split`` twice on one selector/config is identical (Req 11.3).

    Guards against hidden per-call state: the selector resets its report each run
    and derives per-anchor RNGs solely from ``(split_seed, anchor_cve)``, so a
    second call on the same split reproduces the first exactly.
    """
    records = _build_records(60)
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_denominator_pools(records, pools_path)
    config = _make_config(pools_path, seed=13)

    ontology_adapter = CVEOntologyAdapter(config.cve_denominator_pools_path)
    selector = CVENegativeSelector.from_config(ontology_adapter, config)

    first = selector.select_for_split(records)
    second = selector.select_for_split(records)

    assert first == second, "repeated select_for_split calls are not reproducible"
    # Every selection respects the config's max and excludes the anchor.
    for anchor, negs in first.items():
        assert anchor not in negs
        assert len(negs) == len(set(negs))
        assert len(negs) <= config.max_negatives_per_anchor
