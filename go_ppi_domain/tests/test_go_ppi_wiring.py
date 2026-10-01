"""Smoke tests for the GO/PPI domain wiring.

Checks the contracts that fail SILENTLY rather than loudly, which are the ones
that have actually cost this project time:

  * ``TrainingConfig.from_json`` drops undeclared keys, so a ``go_*`` field that
    was never declared as a dataclass field reads back as its hardcoded fallback
    and tuning it in the config does nothing.
  * An arm flagged as ontology-tiered but holding no matcher trains exactly as the
    baseline while reporting as the ontology arm.
  * A view slot missing ``encoder_view`` collapses every protein onto one
    embedding-cache key, so the encoder sees identical text for all of them.
  * ``career_distances`` must be non-negative or ``ContrastiveTriplet`` raises.

Run:
    .venv/bin/python3 -m pytest go_ppi_domain/tests/test_go_ppi_wiring.py -q
    .venv/bin/python3 go_ppi_domain/tests/test_go_ppi_wiring.py     # no pytest needed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONVERTED = ROOT / "preprocess" / "go_ppi"
SPLITS = ROOT / "preprocess" / "go_ppi_splits"
BASELINE_CONFIG = ROOT / "config" / "lc_go_ppi_baseline.json"
ONTNEG_CONFIG = ROOT / "config" / "lc_go_ppi_ontneg_only.json"
STOCH_CONFIG = ROOT / "config" / "lc_go_ppi_ontneg_stoch.json"


def _first_record(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    raise AssertionError(f"{path} is empty")


def test_config_fields_survive_from_json():
    """The go_* fields must be declared, or from_json silently drops them."""
    from contrastive_learning.data_structures import TrainingConfig

    raw = json.loads(ONTNEG_CONFIG.read_text())
    cfg = TrainingConfig.from_json(str(ONTNEG_CONFIG))
    for key in ("go_obo_path", "go_annotation_path", "go_index_cache", "go_aspect",
                "go_alpha", "go_max_hops", "go_ppi_split_dir",
                "go_ppi_converted_dir", "go_ppi_start_hard_ratio",
                "go_ppi_end_hard_ratio", "go_ppi_go_tiered_negatives",
                "go_ppi_go_score_cap"):
        assert hasattr(cfg, key), f"TrainingConfig has no field {key!r}"
        assert getattr(cfg, key) == raw[key], (
            f"{key}: config file says {raw[key]!r} but the loaded config has "
            f"{getattr(cfg, key)!r} — the field is probably undeclared")
    assert cfg.domain_adapter == "go_ppi"
    assert cfg.go_ppi_go_tiered_negatives is True
    print("OK  config fields survive from_json")


def test_single_factor_between_arms():
    """The baseline and diversity-preserving GO arm differ in one field."""
    a = json.loads(BASELINE_CONFIG.read_text())
    b = json.loads(STOCH_CONFIG.read_text())
    keys = {k for k in set(a) | set(b) if not k.startswith("_")}
    diff = {k for k in keys if a.get(k) != b.get(k)}
    assert diff == {"go_ppi_go_tiered_negatives"}, (
        f"arms differ in {sorted(diff)}; the delta could not be attributed to the "
        f"ontology alone")
    print("OK  single factor between arms: go_ppi_go_tiered_negatives")


def test_adapter_registers_and_builds_a_sample():
    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.domain_adapters import get_domain_adapter
    import go_ppi_domain.record_adapter  # noqa: F401  registers "go_ppi"

    cfg = TrainingConfig.from_json(str(BASELINE_CONFIG))
    adapter = get_domain_adapter("go_ppi", cfg)
    assert adapter.name == "go_ppi"

    record = _first_record(SPLITS / "train.jsonl")
    sample = adapter.build_sample(record, 0, cfg)
    assert sample is not None, "adapter returned None for a real train record"
    assert adapter.validate(sample), "adapter rejected its own sample"
    assert sample.label == "positive", "train records must all be grade-2 positives"
    # The grouping id is what grouped/ordinal batching and the selector key on.
    assert sample.metadata["resume_id"] == sample.resume["protein_id"]
    # encoder_view is load-bearing: without it every protein collapses onto one
    # embedding-cache key.
    assert sample.resume["encoder_view"].strip()
    assert sample.job["encoder_view"].strip()
    assert sample.resume["skill_uris"], "anchor carries no GO terms"
    print(f"OK  adapter built sample {sample.sample_id} "
          f"({len(sample.resume['skill_uris'])} anchor GO terms)")


def test_matcher_simgic_is_bounded_and_symmetric():
    from contrastive_learning.data_structures import TrainingConfig
    from go_ppi_domain.run_config import build_go_matcher

    cfg = TrainingConfig.from_json(str(ONTNEG_CONFIG))
    matcher = build_go_matcher(cfg)

    record = _first_record(SPLITS / "train.jsonl")
    a = record["resume"]["skill_uris"]
    b = record["job"]["skill_uris"]

    s = matcher.ontology_set_similarity(a, b)
    assert 0.0 <= s <= 1.0, f"simGIC out of range: {s}"
    assert abs(s - matcher.ontology_set_similarity(b, a)) < 1e-12, "simGIC asymmetric"
    assert matcher.ontology_set_similarity(a, a) > 0.99, "self-similarity should be ~1"
    assert matcher.ontology_set_similarity(a, []) == 0.0, "empty side must give 0.0"

    # branch_distance is a DISTANCE in [0,1] with 0.5 for unknown.
    d = matcher.branch_distance(record["resume"]["coarse_uris"],
                               record["job"]["coarse_uris"])
    assert 0.0 <= d <= 1.0, f"branch_distance out of range: {d}"
    assert matcher.branch_distance([], []) == 0.5, "empty coarse must give neutral 0.5"
    print(f"OK  simGIC={s:.4f} branch_distance={d:.4f}, {len(matcher.index)} GO terms")


def test_selector_returns_graded_negatives_with_valid_distances():
    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.domain_adapters import get_domain_adapter
    from go_ppi_domain.negative_selector import GoPpiNegativeSelector
    from go_ppi_domain.run_config import build_go_matcher
    import go_ppi_domain.record_adapter  # noqa: F401

    cfg = TrainingConfig.from_json(str(ONTNEG_CONFIG))
    matcher = build_go_matcher(cfg)
    selector = GoPpiNegativeSelector.from_config(cfg, "train", matcher=matcher)

    summary = selector.pool_summary()
    assert summary["topics"] > 0, "selector indexed 0 anchor pools"
    assert summary["hard_total"] > 0, "no grade-1 negatives available anywhere"

    adapter = get_domain_adapter("go_ppi", cfg)
    # Find a train record whose anchor actually has a pool.
    sample = None
    with open(SPLITS / "train.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            s = adapter.build_sample(json.loads(line), 0, cfg)
            if s is not None and s.metadata["resume_id"] in selector.pools:
                sample = s
                break
    assert sample is not None, "no train record maps onto a graded pool"

    out = selector.select_batch_negatives(sample, [], cfg.max_negatives_per_anchor, 0)
    assert out is not None, "selector returned None for an anchor that has a pool"
    negatives, distances = out
    assert negatives, "selector returned an empty negative list"
    assert len(negatives) == len(distances)
    # ContrastiveTriplet validates career_distances >= 0.
    assert all(d >= 0 for d in distances), f"negative distance emitted: {distances}"
    # The *10.0 convention shared with _select_ontology_negatives.
    assert all(d <= 10.0 for d in distances), f"distance above the 0-10 scale: {distances}"
    for n in negatives:
        assert n["encoder_view"].strip(), "negative view has no encoder_view"
        assert n["grade"] in (0, 1), f"unexpected negative grade {n['grade']}"
        assert n["original_label"] in ("potential_fit", "no_fit")
    grades = [n["grade"] for n in negatives]
    print(f"OK  selector: {len(negatives)} negatives, grades={grades}, "
          f"distances={[round(d, 3) for d in distances]}")
    print(f"    pools: {summary['topics']} anchors, {summary['hard_total']} grade-1, "
          f"{summary['easy_total']} grade-0, {summary['anchors_without_hard']} anchors "
          f"with no grade-1")


def test_determinism():
    """Selection must be reproducible for a fixed (seed, epoch, anchor)."""
    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.domain_adapters import get_domain_adapter
    from go_ppi_domain.negative_selector import GoPpiNegativeSelector
    import go_ppi_domain.record_adapter  # noqa: F401

    cfg = TrainingConfig.from_json(str(BASELINE_CONFIG))
    adapter = get_domain_adapter("go_ppi", cfg)
    record = _first_record(SPLITS / "train.jsonl")
    sample = adapter.build_sample(record, 0, cfg)

    def once():
        sel = GoPpiNegativeSelector.from_config(cfg, "train", matcher=None)
        out = sel.select_batch_negatives(sample, [], 7, 3)
        return None if out is None else [n["partner_id"] for n in out[0]]

    a, b = once(), once()
    assert a == b, f"non-deterministic selection: {a} vs {b}"
    print(f"OK  deterministic selection at epoch 3: {a}")


def test_splits_are_protein_disjoint():
    manifest = json.loads((SPLITS / "split_manifest.json").read_text())
    overlap = manifest["entity_overlap_between_splits"]
    assert manifest["strategy"] == "protein_disjoint"
    for pair, n in overlap.items():
        assert n == 0, f"{n} proteins shared between {pair}"
    print(f"OK  protein-disjoint splits, entity overlap {overlap}, "
          f"records {manifest['records']}")


if __name__ == "__main__":
    failures = 0
    for fn in (test_config_fields_survive_from_json,
               test_single_factor_between_arms,
               test_splits_are_protein_disjoint,
               test_adapter_registers_and_builds_a_sample,
               test_matcher_simgic_is_bounded_and_symmetric,
               test_selector_returns_graded_negatives_with_valid_distances,
               test_determinism):
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print()
    print("all wiring checks passed" if not failures else f"{failures} check(s) failed")
    sys.exit(1 if failures else 0)
