"""Run one point of a learning curve: (domain, arm, fraction, seed) -> checkpoint.

This exists because ``trials_domain.label_budget_runner`` answers a *different*
question and hardcodes the machinery for it. That study replaces the graded
negative pool with unjudged-corpus negatives, to ask whether an ontology can
substitute for relevance judgments. The learning curve here holds the graded pool
fixed and toggles only whether the ontology *orders* it, which is the contrast the
career side runs. Reusing the label-budget runner would have silently imported the
corpus-negative design into a table that claims to vary one factor.

Two domains, one code path
--------------------------
``career``  Nothing to wire. ``ontology_guided_negatives`` is read by
            ``BatchProcessor`` directly, and the ESCO matcher is built in its
            constructor.
``trials``  The MeSH matcher and ``TrialsNegativeSelector`` must be attached
            through the domain seams, because the ORCA orchestrator's
            ``set_selector(None)`` would otherwise clear a selector installed in
            the normal slot.

Both arms of a pair must differ in as few config fields as possible; the runner
prints the diff so an unintended second factor is visible before training starts
rather than after the numbers are quoted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DOMAINS = ("career", "trials", "go_ppi")

#: Config fields that legitimately differ between a pair of arms. Anything else
#: differing is reported as a potential confound.
EXPECTED_FACTOR_FIELDS = {
    # One entry per mechanism the career ontology can act through. The old
    # 4-factor arm moved all of these at once, which is why its +0.0226 at 10%
    # could not be attributed; each is now testable alone.
    #   negative SELECTION  -> ontology_guided_negatives (+ rank tiers)
    #   sample WEIGHTING    -> ontology_weight
    #   extra OT signal     -> use_ot_distance
    #   pathway negatives   -> use_pathway_negatives / pathway_weight
    "career": {
        "ontology_guided_negatives", "ontology_negative_rank_tiers",
        "ontology_weight", "use_ot_distance",
        "use_pathway_negatives", "pathway_weight",
    },
    "trials": {"trials_mesh_tiered_negatives", "trials_tier_sampling",
               "trials_tier_window_frac"},
    # Same shape as trials: one flag decides whether the ontology ORDERS the
    # graded pool. Pool, grade mix and label budget are identical across arms.
    # go_ppi_tier_sampling is listed too because it is a SEPARATE, deliberately
    # decomposed factor: "deterministic" additionally removes per-epoch negative
    # variety, so baseline-vs-deterministic mixes two effects. The intended
    # single-factor pair is baseline vs the stochastic arm.
    "go_ppi": {"go_ppi_go_tiered_negatives", "go_ppi_tier_sampling",
               "go_ppi_tier_window_frac"},
}


def config_diff(a: Path, b: Path) -> Dict[str, Any]:
    """Fields differing between two config files, ignoring ``_``-prefixed notes."""
    ca, cb = json.load(open(a)), json.load(open(b))
    keys = {k for k in set(ca) | set(cb) if not k.startswith("_")}
    return {k: (ca.get(k, "<absent>"), cb.get(k, "<absent>"))
            for k in sorted(keys) if ca.get(k) != cb.get(k)}


def report_factor(domain: str, baseline: Path, arm: Path) -> Dict[str, Any]:
    """Print and return the single-factor audit for a pair of configs.

    ``training_seed`` is excluded because the runner overrides it per run.
    """
    diff = {k: v for k, v in config_diff(baseline, arm).items()
            if k != "training_seed"}
    expected = EXPECTED_FACTOR_FIELDS.get(domain, set())
    unexpected = sorted(set(diff) - expected)

    print(f"\nsingle-factor audit ({domain}): {baseline.name} vs {arm.name}")
    for key, (bv, av) in diff.items():
        flag = "  " if key in expected else "!!"
        print(f"  {flag} {key}: {bv} -> {av}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} field(s) outside the intended "
              f"factor: {unexpected}. The delta cannot be attributed to the "
              f"ontology alone.")
    elif diff:
        print("  OK: differences are confined to the intended factor.")
    else:
        print("  WARNING: configs are identical — the arm is inert.")
    return {"diff": {k: list(v) for k, v in diff.items()},
            "unexpected_fields": unexpected}


def attach_trials(trainer, config) -> Dict[str, Any]:
    """Attach the MeSH matcher and train+validation graded selector.

    ``ContrastiveLearningTrainer`` uses the same ``BatchProcessor`` for training
    and validation.  A train-only selector therefore cannot resolve validation
    topics and silently falls back to the generic negative path during checkpoint
    selection.  The union is safe: pools are keyed by disjoint topic ids, so a
    train batch can never read a validation pool, while validation now receives
    its own graded candidates.
    """
    import trials_domain.record_adapter  # noqa: F401  registers "trials"
    from trials_domain.negative_selector import TrialsNegativeSelector

    batch_processor = getattr(trainer, "batch_processor", None)
    if batch_processor is None:
        raise RuntimeError("attach_trials requires a trainer exposing .batch_processor")

    matcher = None
    if bool(getattr(config, "trials_mesh_tiered_negatives", False)):
        from trials_domain.run_config import build_mesh_matcher

        matcher = build_mesh_matcher(config)
        setter = getattr(batch_processor, "set_ontology_matcher", None)
        if not callable(setter):
            raise RuntimeError(
                "BatchProcessor exposes no set_ontology_matcher; the MeSH arm "
                "cannot receive its ontology signal.")
        setter(matcher, coarse_distance_fn=matcher.branch_distance)

    selector = TrialsNegativeSelector.from_config(config, "train", matcher=matcher)
    validation_selector = TrialsNegativeSelector.from_config(
        config, "validation", matcher=matcher)
    train_pool_count = len(selector.pools)
    validation_pool_count = len(validation_selector.pools)
    overlap = set(selector.pools) & set(validation_selector.pools)
    if overlap:
        raise RuntimeError(
            f"train/validation topic leak in Trials selector: {sorted(overlap)[:5]}")
    selector.pools.update(validation_selector.pools)
    selector.views.update(validation_selector.views)

    if getattr(config, "trials_common_validation", False):
        if not selector.soft_guidance:
            raise ValueError("common Trials validation requires the V2 sampler")
        selector.common_validation_topics = set(validation_selector.pools)

    setter = getattr(batch_processor, "set_domain_negative_selector", None)
    if not callable(setter):
        raise RuntimeError(
            "BatchProcessor exposes no set_domain_negative_selector; the graded "
            "negatives cannot be supplied and training would fall back to the "
            "'No Match Available' dummy negative.")
    setter(selector)

    # Fail loudly if the tiering flag is set but no matcher reached the selector:
    # that combination trains as the baseline while reporting as the ontology arm.
    if getattr(config, "trials_mesh_tiered_negatives", False):
        if getattr(selector, "matcher", None) is None:
            raise RuntimeError(
                "trials_mesh_tiered_negatives=True but the selector has no "
                "matcher; this arm would be identical to the baseline.")

    return {
        "selector": type(selector).__name__,
        "mesh_tiered": bool(getattr(selector, "mesh_tiered", False)),
        "tier_sampling": str(getattr(selector, "tier_sampling", "uniform")),
        "tier_window_frac": float(getattr(selector, "tier_window_frac", 1.0)),
        "matcher": type(matcher).__name__ if matcher is not None else None,
        "mesh_facet": getattr(selector, "mesh_facet", "all"),
        "mesh_tier_scope": getattr(selector, "mesh_tier_scope", "both"),
        "mesh_similarity_mode": getattr(matcher, "similarity_mode", None),
        "selector_splits": ["train", "validation"],
        "train_pools": train_pool_count,
        "validation_pools": validation_pool_count,
        "pool_overlap": 0,
        "pool": selector.pool_summary(),
    }


def attach_go_ppi(trainer, config) -> Dict[str, Any]:
    """Attach the GO matcher (when the arm needs it) and the graded selector.

    Mirrors :func:`attach_trials`. The matcher is built only for the ontology arm,
    so the baseline arm pays neither the index-load cost nor any risk of the
    ontology leaking in through ORCA feature capture.
    """
    import go_ppi_domain.record_adapter  # noqa: F401  registers "go_ppi"
    from go_ppi_domain.negative_selector import GoPpiNegativeSelector

    batch_processor = getattr(trainer, "batch_processor", None)
    if batch_processor is None:
        raise RuntimeError("attach_go_ppi requires a trainer exposing .batch_processor")

    matcher = None
    if bool(getattr(config, "go_ppi_go_tiered_negatives", False)):
        from go_ppi_domain.run_config import build_go_matcher

        matcher = build_go_matcher(config)
        setter = getattr(batch_processor, "set_ontology_matcher", None)
        if not callable(setter):
            raise RuntimeError(
                "BatchProcessor exposes no set_ontology_matcher; the GO arm cannot "
                "receive its ontology signal.")
        setter(matcher, coarse_distance_fn=matcher.branch_distance)

    selector = GoPpiNegativeSelector.from_config(config, "train", matcher=matcher)
    validation_selector = GoPpiNegativeSelector.from_config(
        config, "validation", matcher=matcher)
    train_pool_count = len(selector.pools)
    validation_pool_count = len(validation_selector.pools)
    overlap = set(selector.pools) & set(validation_selector.pools)
    if overlap:
        raise RuntimeError(
            f"train/validation protein leak in GO/PPI selector: {sorted(overlap)[:5]}")
    selector.pools.update(validation_selector.pools)
    selector.views.update(validation_selector.views)

    setter = getattr(batch_processor, "set_domain_negative_selector", None)
    if not callable(setter):
        raise RuntimeError(
            "BatchProcessor exposes no set_domain_negative_selector; the graded "
            "negatives cannot be supplied and training would fall back to the "
            "'No Match Available' dummy negative.")
    setter(selector)

    # Fail loudly if the tiering flag is set but no matcher reached the selector:
    # that combination trains as the baseline while reporting as the ontology arm.
    if getattr(config, "go_ppi_go_tiered_negatives", False):
        if getattr(selector, "matcher", None) is None:
            raise RuntimeError(
                "go_ppi_go_tiered_negatives=True but the selector has no matcher; "
                "this arm would be identical to the baseline.")

    return {
        "selector": type(selector).__name__,
        "go_tiered": bool(getattr(selector, "go_tiered", False)),
        "matcher": type(matcher).__name__ if matcher is not None else None,
        "go_terms": len(matcher.index) if matcher is not None else None,
        "selector_splits": ["train", "validation"],
        "train_pools": train_pool_count,
        "validation_pools": validation_pool_count,
        "pool_overlap": 0,
        "pool": selector.pool_summary(),
    }


def attach_career(trainer, config) -> Dict[str, Any]:
    """Report (not install) the career arm's negative mechanism.

    Nothing needs attaching: ``BatchProcessor`` reads the flags itself. But the
    reachability of the ontology path is asserted here, because it was previously
    gated on ``use_pathway_negatives`` alone — so a config with
    ``ontology_guided_negatives=True`` and ``use_pathway_negatives=False`` built
    the ESCO matcher and then never consulted it, training identically to the
    baseline with no error raised.
    """
    bp = trainer.batch_processor
    guided = bool(getattr(bp, "ontology_guided_negatives", False))
    has_matcher = bp.skill_matcher is not None

    if guided and not has_matcher:
        raise RuntimeError(
            "ontology_guided_negatives=True but no skill matcher was built. "
            "Check esco_kg_path / esco_graph_path exist; without the matcher "
            "this arm is identical to the baseline.")

    return {
        "use_pathway_negatives": bool(bp.use_pathway_negatives),
        "ontology_guided_negatives": guided,
        "skill_matcher": type(bp.skill_matcher).__name__ if has_matcher else None,
        "rank_tiers": bool(getattr(config, "ontology_negative_rank_tiers", False)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--train-file", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None,
                    help="recorded in the manifest only; subsampling is done "
                         "upstream so both arms see byte-identical data")
    ap.add_argument("--compare-config", type=Path, default=None,
                    help="the paired arm's config; prints a single-factor audit")
    ap.add_argument("--validation-file", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.trainer import ContrastiveLearningTrainer

    audit = None
    if args.compare_config:
        audit = report_factor(args.domain, args.compare_config, args.config)

    config = TrainingConfig.from_json(str(args.config))
    config.training_seed = args.seed
    if args.epochs is not None:
        config.num_epochs = args.epochs
    # ``trainer.train`` takes only the dataset path; validation is read off the
    # config, so it has to be set before the trainer is constructed.
    if args.validation_file:
        config.validation_path = str(args.validation_file)
    # Single-phase. The staged ORCA schedule is a separate factor worth +0.046 on
    # the career data; leaving it on would fold it into the ontology delta.
    config.orca_enabled = False

    # Register the domain adapter BEFORE the trainer is built: the trainer's
    # DataLoader resolves config.domain_adapter during construction.
    if args.domain == "trials":
        import trials_domain.record_adapter  # noqa: F401
    elif args.domain == "go_ppi":
        import go_ppi_domain.record_adapter  # noqa: F401

    trainer = ContrastiveLearningTrainer(config=config,
                                         output_dir=str(args.output_dir))

    if args.domain == "trials":
        attach = attach_trials(trainer, config)
    elif args.domain == "go_ppi":
        attach = attach_go_ppi(trainer, config)
    else:
        attach = attach_career(trainer, config)
    logger.info("Arm wiring: %s", attach)

    if args.domain in ("trials", "go_ppi"):
        # The graded negatives never pass through the DataLoader, so they miss the
        # normal embedding preload; without this every batch would encode its
        # negatives from scratch.
        from trials_domain.label_budget_runner import (
            assert_cache_valid, preencode_pool)

        selector = trainer.batch_processor.domain_negative_selector
        attach["preencoded"] = preencode_pool(trainer, selector)
        attach["cache_check"] = assert_cache_valid(trainer)
        if args.domain == "trials" and getattr(config, "trials_soft_guidance", ""):
            from trials_domain.soft_sampling import prepare_text_scores
            attach["soft_guidance"] = config.trials_soft_guidance
            attach["common_validation_topics"] = sorted(selector.common_validation_topics)
            attach["text_mining_scores"] = prepare_text_scores(
                trainer, selector, config.trials_converted_dir)

    result = trainer.train(str(args.train_file))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lc_manifest.json").write_text(
        json.dumps({
            "domain": args.domain,
            "config": str(args.config),
            "train_file": str(args.train_file),
            "train_records": sum(1 for _ in open(args.train_file)),
            "fraction": args.fraction,
            "seed": args.seed,
            "epochs": config.num_epochs,
            "single_factor_audit": audit,
            "attach": attach,
            "train_result": _jsonable(result),
        }, indent=2), encoding="utf-8")
    logger.info("Learning-curve point complete -> %s", args.output_dir)
    return 0


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    sys.exit(main())
