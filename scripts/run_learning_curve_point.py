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

GO/PPI withholding study
------------------------
``--withholding-manifest`` switches to :func:`run_study` (design 4.1-4.4, 4.7 of
``.agents/tasks/ontology-withholding-feasibility/design_go_ppi.md``): preflight,
condition assertions, epoch callbacks, post-training checks and the extended
``lc_manifest.json``; any failure writes ``run_error.json`` and exits 2. Without
it the code path is the one above. ``--device auto|cuda|cpu`` (user decision B)
selects where the frozen encoder and the projection head run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding_constants as wc  # noqa: E402  (stdlib-only module)

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
               "go_ppi_tier_window_frac",
               # Withholding study: the GO relation-preservation loss factor.
               "go_ontology_loss_weight"},
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


def attach_go_relation_loss(trainer, config, train_file) -> Dict[str, Any]:
    """Attach the loss-only GoRelationTable when ``go_ontology_loss_weight > 0``.

    The table goes to the LOSS ENGINE only. This never calls
    ``batch_processor.set_ontology_matcher`` and never touches the selector, so
    the loss factor cannot open GO-tiered negative selection (withholding
    design 4.3, 5.4). Weight 0 attaches nothing.
    """
    weight = float(getattr(config, "go_ontology_loss_weight", 0.0))
    if weight == 0:
        return {"weight": 0.0, "table": None}
    if getattr(config, "domain_adapter", None) != "go_ppi":
        raise ValueError("go_ontology_loss_weight > 0 requires domain_adapter='go_ppi'")
    from go_ppi_domain.go_relation import GoRelationTable

    table = GoRelationTable.from_train_file(Path(config.go_index_cache), Path(train_file))
    trainer.loss_engine.set_go_relation_table(table)
    return {
        "weight": weight,
        "table_proteins": len(table.ids),
        "annotated_proteins": table.n_annotated(),
        "table_sha256": table.sha256(),
        "index_leak_pruning": table.source_meta.get("leak_pruning"),
    }


# ====================================================================== device
#: ``--device`` / ``TrainingConfig.training_device`` (remote GPU execution, user
#: decision B). ``auto`` is today's behaviour: the trainer and loss engine pick
#: CUDA when available, else CPU, for both the frozen encoder and the projection.
DEVICES = ("auto", "cuda", "cpu")


def resolve_device(requested: str) -> str:
    """``auto|cuda|cpu`` -> the device the trainer will use (``cuda`` or ``cpu``)."""
    import torch

    requested = str(requested or "auto")
    if requested not in DEVICES:
        raise ValueError(f"unknown device {requested!r}; choose from {DEVICES}")
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if requested == "auto":
        return "cuda" if available else "cpu"
    return requested


@contextmanager
def device_scope(resolved: str):
    """Build the trainer on ``resolved``.

    Every device choice in the trainer, loss engine, embedding cache and
    SentenceTransformer is ``'cuda' if torch.cuda.is_available() else 'cpu'`` at
    construction time. Forcing CPU on a CUDA machine therefore hides CUDA for the
    duration of construction only; everything built inside keeps that device. In
    every other case (CPU machine, ``cuda``, ``auto``) this is a no-op, so CPU and
    career behaviour are byte-identical.
    """
    import torch

    if resolved == "cpu" and torch.cuda.is_available():
        original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            yield
        finally:
            torch.cuda.is_available = original
    else:
        yield


def device_record(requested: str, resolved: str, trainer=None) -> Dict[str, Any]:
    """The ``lc_manifest["device"]`` block; asserts the trainer really uses ``resolved``."""
    import torch

    rec: Dict[str, Any] = {"requested": str(requested or "auto"), "resolved": resolved,
                           "cuda_available": bool(torch.cuda.is_available()),
                           "torch": torch.__version__, "cuda_device_name": None}
    if resolved == "cuda":
        rec["cuda_device_name"] = torch.cuda.get_device_name(0)
    if trainer is not None:
        used = {"trainer": torch.device(trainer.device).type,
                "loss_engine": torch.device(trainer.loss_engine.device).type}
        enc = next(iter(trainer.text_encoder.parameters()), None)
        if enc is not None:
            used["text_encoder"] = enc.device.type
        proj = next(iter(trainer.model.parameters()), None)
        if proj is not None:
            used["projection"] = proj.device.type
        wrong = {k: v for k, v in used.items() if v != resolved}
        if wrong:
            raise RuntimeError(f"device {resolved!r} requested but {wrong} differ")
        rec["components"] = used
    return rec


# =============================================================== study path
#: The four withholding-study cells (design 4.1).
STUDY_CELLS = ("baseline", "loss_only", "neg_only", "both")
LOSS_ON_CELLS = ("loss_only", "both")


def _validate_study_cli(ap: argparse.ArgumentParser, args) -> None:
    """The design 4.1 requirement/exclusion rules, as argparse errors (exit 2)."""
    study_only = {"--study-cell": args.study_cell, "--go-diagnostics": args.go_diagnostics,
                  "--pilot": args.pilot or None}
    if args.withholding_manifest is None:
        given = [k for k, v in study_only.items() if v is not None]
        if given:
            ap.error(f"{', '.join(given)} require(s) --withholding-manifest")
    else:
        if args.domain != "go_ppi":
            ap.error("--withholding-manifest requires --domain go_ppi")
        if args.study_cell is None:
            ap.error("--withholding-manifest requires --study-cell")
        if args.go_diagnostics is None:
            ap.error("--withholding-manifest requires --go-diagnostics")
        if args.validation_file is not None:
            ap.error("--withholding-manifest forbids --validation-file")
    if args.go_ppi_split_dir is not None and args.domain != "go_ppi":
        ap.error("--go-ppi-split-dir requires --domain go_ppi")


def preflight_index(config, frozen: Dict[str, Any]) -> Dict[str, Any]:
    """Design 4.2 step 2: the pruned index hash and meta, before any matcher exists.

    Uses ``GoIndex.load`` only, so a missing or wrong pickle fails here instead of
    ``build_or_load`` silently rebuilding (and overwriting) it later.
    """
    from go_ppi_domain.go_ontology import GoIndex

    path = Path(config.go_index_cache)
    if not path.is_file():
        raise RuntimeError(f"preflight 2: go_index_cache {wc.rel_to_root(path)} does not exist")
    got = wc.sha256_file(path)
    if got != frozen.get("go_index_P_sha256"):
        raise RuntimeError(f"preflight 2: go_index_cache sha256 {got} != frozen_hashes"
                           f"[go_index_P_sha256] {frozen.get('go_index_P_sha256')}")
    meta = GoIndex.load(path).meta
    if (meta.get("leak_pruning") or {}).get("with_from_human_protein") is not True:
        raise RuntimeError("preflight 2: go_index_cache is not leak-pruned "
                           "(meta.leak_pruning.with_from_human_protein is not True)")
    if meta.get("aspect") != "P":
        raise RuntimeError(f"preflight 2: go_index_cache aspect {meta.get('aspect')!r} != 'P'")
    return {"go_index_P_sha256": got, "aspect": "P", "leak_pruning": True}


def _git_head() -> Any:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=wc.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - recorded, not required
        return None


def study_preflight(args, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Design 4.2 steps 1-5 (plus the device resolution of user decision B).

    Returns the expanded ``TrainingConfig`` and the provenance the manifest needs.
    Raises ``RuntimeError`` on the first violated step.
    """
    from contrastive_learning.data_structures import TrainingConfig

    # 1. paths: keys present in the JSON file + the CLI path overrides. Under
    #    user decision B a config value may be root-anchored ("$ROOT/...") or
    #    absolute; a bare relative value (cwd-dependent) is rejected.
    bad = wc.config_path_problems(raw)
    cli = {"--go-ppi-split-dir": args.go_ppi_split_dir, "--train-file": args.train_file,
           "--output-dir": args.output_dir, "--withholding-manifest": args.withholding_manifest,
           "--go-diagnostics": args.go_diagnostics}
    bad += [f"{k}={str(v)!r}" for k, v in cli.items()
            if v is not None and not Path(v).is_absolute()]
    if bad:
        raise RuntimeError(f"preflight 1: relative path(s) {bad}; use absolute or "
                           f"'{wc.ROOT_TOKEN}/...' values")
    config = TrainingConfig.from_dict(wc.expand_config_dict(raw))
    config.training_seed = args.seed
    if args.epochs is not None:
        config.num_epochs = args.epochs
    config.orca_enabled = False
    if args.go_ppi_split_dir is not None:
        config.go_ppi_split_dir = str(args.go_ppi_split_dir)

    # 2. pruned index, before any matcher is built
    frozen_path = Path(wc.SPLITS) / "frozen_hashes.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    index = preflight_index(config, frozen)

    # 3. no validation path (variant-independent final-epoch checkpoint, design 8)
    if getattr(config, "validation_path", None) or args.validation_file:
        raise RuntimeError("preflight 3: study runs must not set validation_path")

    # 4. lambda equality (skipped only under --pilot)
    weight = float(getattr(config, "go_ontology_loss_weight", 0.0))
    lam = None
    if args.pilot:
        lam_check = "skipped (--pilot)"
    else:
        path = wc.lambda_choice_path()
        if not path.is_file():
            raise RuntimeError(f"preflight 4: {wc.rel_to_root(path)} does not exist "
                               f"(run the lambda pilot first, or pass --pilot)")
        lam = float(json.loads(path.read_text(encoding="utf-8"))["lambda"])
        expected = lam if args.study_cell in LOSS_ON_CELLS else 0.0
        if weight != expected:
            raise RuntimeError(f"preflight 4: go_ontology_loss_weight {weight} != {expected} "
                               f"for cell {args.study_cell} (lambda_choice {lam})")
        lam_check = "ok"

    # 5. code pin content hash (equality with the pin is the sweep's job)
    code_hash, code_files = wc.code_sha256()

    # device (user decision B)
    requested = args.device or getattr(config, "training_device", "auto")
    resolved = resolve_device(requested)
    return {"config": config, "frozen": frozen, "frozen_path": frozen_path,
            "index": index, "lambda": lam, "lambda_check": lam_check,
            "code": {"git_head": _git_head(), "code_sha256": code_hash,
                     "files": len(code_files)},
            "code_files": code_files, "device_requested": requested,
            "device_resolved": resolved}


def assert_post_training(trainer, config, manifest: Dict[str, Any], recorder, guard,
                         code_files=None) -> Dict[str, Any]:
    """Design 4.4: fatal (``RuntimeError``) checks after ``trainer.train`` returns."""
    import torch

    W = set(manifest.get("withheld_proteins", []))
    engine = trainer.loss_engine
    checks: Dict[str, str] = {}
    seen = set(getattr(engine, "go_relation_seen_ids", set()) or set())
    if seen & W:
        raise RuntimeError(f"post-training: L_GO saw withheld proteins {sorted(seen & W)[:5]}")
    checks["post_seen_ids_disjoint_W"] = "ok"
    missing = int(getattr(engine, "go_relation_missing_ids", 0) or 0)
    if missing:
        raise RuntimeError(f"post-training: go_relation_missing_ids == {missing} (need 0)")
    checks["post_missing_ids_zero"] = "ok"
    if recorder.none_returns:
        raise RuntimeError(f"post-training: selector returned None {recorder.none_returns} times")
    if recorder.exceptions:
        raise RuntimeError(f"post-training: selector raised {recorder.exceptions} times")
    checks["post_selector_counters_zero"] = "ok"
    n_epochs = int(config.num_epochs)
    if len(getattr(guard, "epochs", [])) != n_epochs:
        raise RuntimeError(f"post-training: StrictEpochGuard saw {len(guard.epochs)} of "
                           f"{n_epochs} epochs")
    checks["post_every_epoch_guarded"] = "ok"
    imported = wc.imported_repo_modules()
    files = set(code_files if code_files is not None else wc.code_sha256()[1])
    uncovered = sorted(set(imported) - files)
    if uncovered:
        raise RuntimeError(f"post-training: imported repo modules outside the code pin: "
                           f"{uncovered}")
    checks["post_import_coverage"] = "ok"
    out_dir = Path(trainer.output_dir)
    name = f"checkpoint_epoch_{n_epochs - 1}.pt"
    ckpt = out_dir / name
    if not ckpt.is_file():
        raise RuntimeError(f"post-training: final checkpoint {name} missing")
    epoch = torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch")
    if epoch != n_epochs - 1:
        raise RuntimeError(f"post-training: {name} records epoch {epoch} != {n_epochs - 1}")
    checks["post_final_checkpoint"] = "ok"
    emergency = sorted(p.name for p in out_dir.glob("*emergency*.pt"))
    if emergency:
        raise RuntimeError(f"post-training: emergency checkpoint(s) present {emergency}")
    checks["post_no_emergency_checkpoint"] = "ok"
    return {"checks": checks, "imported": imported, "selected_checkpoint": name,
            "selected_checkpoint_sha256": wc.sha256_file(ckpt), "seen_ids": len(seen),
            "missing_ids": missing}


def _write_run_error(output_dir: Path, stage: str, exc: BaseException,
                     ctx: Dict[str, Any]) -> None:
    import traceback

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "error": wc.scrub_root(f"{type(exc).__name__}: {exc}"),
               "traceback": wc.scrub_root("".join(traceback.format_exception(exc))),
               "cell": ctx.get("cell"), "cond_id": ctx.get("cond_id"), "seed": ctx.get("seed")}
    (output_dir / "run_error.json").write_text(json.dumps(payload, indent=2) + "\n",
                                               encoding="utf-8")


def run_study(args, audit) -> int:
    """The withholding-study path (design 4.2-4.4, 4.7). Any failure: run_error.json, exit 2."""
    ctx: Dict[str, Any] = {"cell": args.study_cell, "cond_id": None, "seed": args.seed}
    stage = "preflight"
    try:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        pre = study_preflight(args, raw)
        config = pre["config"]

        stage = "wiring"
        import go_ppi_domain.record_adapter  # noqa: F401  registers "go_ppi"
        from contrastive_learning.trainer import ContrastiveLearningTrainer
        from go_ppi_domain.withholding import assert_condition
        from go_ppi_domain.withholding_diagnostics import (
            EpochDiagnostics, SelectionRecorder, StrictEpochGuard, negative_selection_block,
            projection_param_summary)
        from trials_domain.label_budget_runner import assert_cache_valid, preencode_pool

        manifest_path = Path(args.withholding_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ctx["cond_id"] = manifest.get("cond_id")
        with device_scope(pre["device_resolved"]):
            trainer = ContrastiveLearningTrainer(config=config, output_dir=str(args.output_dir))
        device = device_record(pre["device_requested"], pre["device_resolved"], trainer)
        attach = attach_go_ppi(trainer, config)
        attach["go_relation_loss"] = attach_go_relation_loss(trainer, config, args.train_file)
        selector = trainer.batch_processor.domain_negative_selector
        recorder = SelectionRecorder(selector,
                                     unannotated_anchors=manifest["unannotated_anchors"]["ids"])
        attach["preencoded"] = preencode_pool(trainer, selector)
        attach["cache_check"] = assert_cache_valid(trainer)
        diag = EpochDiagnostics(Path(args.go_diagnostics), Path(args.train_file),
                                recorder=recorder,
                                weight=float(config.go_ontology_loss_weight))
        guard = StrictEpochGuard(cell=args.study_cell)
        trainer.epoch_end_callbacks = [recorder.on_epoch_end, diag, guard]

        stage = "condition_assertions"
        checks = assert_condition(trainer, config, manifest_path, args.train_file,
                                  args.study_cell, pre["frozen"])

        stage = "train"
        result = trainer.train(str(args.train_file))

        stage = "post_training"
        post = assert_post_training(trainer, config, manifest, recorder, guard,
                                    code_files=pre["code_files"])

        stage = "manifest"
        engine = trainer.loss_engine
        rel = attach["go_relation_loss"]
        counts = {k: checks[k] for k in ("withheld_count", "train_proteins") if k in checks}
        assertions = {k: v for k, v in checks.items()
                      if k not in ("cond_id", "withheld_count", "train_proteins")}
        assertions.update({"preflight_paths": "ok", "preflight_index": "ok",
                           "preflight_no_validation": "ok",
                           "preflight_lambda": pre["lambda_check"]})
        assertions.update(post["checks"])
        assertions["counts"] = counts
        lc = {
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
            "withholding": {
                "cond_id": manifest.get("cond_id"), "pack_id": manifest.get("pack_id"),
                "k": manifest.get("k"), "cell": args.study_cell,
                "manifest_sha256": wc.sha256_file(manifest_path),
                "frozen_hashes_sha256": wc.sha256_file(pre["frozen_path"]),
                "assertions": assertions,
                "selected_checkpoint": post["selected_checkpoint"],
                "selected_checkpoint_sha256": post["selected_checkpoint_sha256"]},
            "epoch_stats": list(guard.epochs),
            "negative_selection": negative_selection_block(recorder, selector),
            "go_relation_loss": {
                "weight": float(config.go_ontology_loss_weight),
                "table_proteins": int(rel.get("table_proteins", 0) or 0),
                "table_sha256": rel.get("table_sha256"),
                "seen_ids": post["seen_ids"], "missing_ids": post["missing_ids"],
                "skipped_batches": int(getattr(engine, "go_relation_skipped_batches", 0)),
                "applied_batches": int(getattr(engine, "go_relation_applied_batches", 0))},
            "projection_params": projection_param_summary(trainer),
            "code": dict(pre["code"], imported=post["imported"]),
            "pilot": bool(args.pilot),
            "device": device,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "lc_manifest.json").write_text(
            json.dumps(wc.relativize_json(_jsonable(lc)), indent=2), encoding="utf-8")
    except BaseException as exc:  # noqa: BLE001 - design 4.2: every failure is exit 2
        logger.error("withholding run failed at %s: %s: %s", stage, type(exc).__name__, exc)
        _write_run_error(Path(args.output_dir), stage, exc, ctx)
        return 2
    logger.info("Withholding run complete -> %s", args.output_dir)
    return 0


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
    # GO/PPI withholding study (design 4.1). Without --withholding-manifest none of
    # these change anything.
    ap.add_argument("--go-ppi-split-dir", type=Path, default=None,
                    help="overrides config.go_ppi_split_dir before the trainer is built")
    ap.add_argument("--withholding-manifest", type=Path, default=None,
                    help="enables the withholding-study path (needs --domain go_ppi, "
                         "--study-cell, --go-diagnostics; forbids --validation-file)")
    ap.add_argument("--study-cell", choices=STUDY_CELLS, default=None)
    ap.add_argument("--go-diagnostics", type=Path, default=None,
                    help="per-epoch diagnostics JSONL (design 6)")
    ap.add_argument("--pilot", action="store_true",
                    help="lambda-pilot run: skips only the lambda_choice equality check")
    ap.add_argument("--device", choices=DEVICES, default=None,
                    help="auto (default: CUDA when available) | cuda | cpu; overrides "
                         "config.training_device")
    args = ap.parse_args(argv)
    _validate_study_cli(ap, args)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.trainer import ContrastiveLearningTrainer

    audit = None
    if args.compare_config:
        audit = report_factor(args.domain, args.compare_config, args.config)

    if args.withholding_manifest is not None:
        return run_study(args, audit)

    config = TrainingConfig.from_json(str(args.config))
    config.training_seed = args.seed
    if args.epochs is not None:
        config.num_epochs = args.epochs
    # ``trainer.train`` takes only the dataset path; validation is read off the
    # config, so it has to be set before the trainer is constructed.
    if args.validation_file:
        config.validation_path = str(args.validation_file)
    if args.go_ppi_split_dir is not None:
        config.go_ppi_split_dir = str(args.go_ppi_split_dir)
    # Single-phase. The staged ORCA schedule is a separate factor worth +0.046 on
    # the career data; leaving it on would fold it into the ontology delta.
    config.orca_enabled = False

    # Register the domain adapter BEFORE the trainer is built: the trainer's
    # DataLoader resolves config.domain_adapter during construction.
    if args.domain == "trials":
        import trials_domain.record_adapter  # noqa: F401
    elif args.domain == "go_ppi":
        import go_ppi_domain.record_adapter  # noqa: F401

    # Device (user decision B): only an explicit --device or a non-"auto"
    # config.training_device changes anything; the default path is untouched.
    device_requested = args.device or getattr(config, "training_device", "auto")
    explicit_device = args.device is not None or device_requested != "auto"
    device_resolved = None
    if explicit_device:
        try:
            device_resolved = resolve_device(device_requested)
        except (RuntimeError, ValueError) as exc:
            ap.error(str(exc))
    with device_scope(device_resolved) if explicit_device else nullcontext():
        trainer = ContrastiveLearningTrainer(config=config,
                                             output_dir=str(args.output_dir))
    device = (device_record(device_requested, device_resolved, trainer)
              if explicit_device else None)

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
    lc_manifest = {
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
    }
    if device is not None:
        lc_manifest["device"] = device
    (args.output_dir / "lc_manifest.json").write_text(
        json.dumps(lc_manifest, indent=2), encoding="utf-8")
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
