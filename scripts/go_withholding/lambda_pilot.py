"""Choose the L_GO weight lambda on target-free validation (design 12.2, gate G4).

1. Pilot run: the loss_only template with lambda0 = 0.1 (``<out>/l0.1/config.json``),
   condition ``all``, seed 42, 3 epochs, via the runner subprocess with ``--pilot``
   (which skips only the lambda_choice equality check).
2. Measurement (in-process): a trainer from the same config (seed 42), the
   selector from ``attach_go_ppi`` (it already holds the target-free validation
   pools), ``checkpoint_epoch_2.pt`` loaded strictly, selector and loss engine set
   to epoch 2 (asserted). All grade-2 records of
   ``validation_targetfree/validation_positive.jsonl`` in file order, batches of
   32, last partial batch dropped; fewer than 4 batches -> exit 3. Relation table
   over the validation proteins. Per batch the design-6 gradient recipe on the
   projection parameters; r = median of lambda * |grad L_GO| / |grad L_InfoNCE|.
3. lambda* = lambda0 * 0.20 / r0, rounded in log space to {1, 2, 3, 5} x 10^n.
4. Confirmation run at lambda*: r* must lie in [0.10, 0.30]; otherwise one
   re-estimate lambda** = lambda* * 0.20 / r* (rounded) and one more
   confirmation; if that fails too: exit 3 (protocol stop, ask the user). A
   lambda equal to one already measured reuses that run (identical settings).
5. ``<out>/lambda_choice.json``, ``append_pin("G4-lambda", ...)``, then
   ``materialize_configs.py --lambda <lambda>``.

Exit codes: 0 ok; 2 failure (pilot run, inputs, materialize); 3 protocol stop.

Usage::

    "$PY" "$R/scripts/go_withholding/lambda_pilot.py" --out "$R/results/go_withholding/lambda_pilot"
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("lambda_pilot")

LAMBDA0 = 0.1
SEED = 42
EPOCHS = 3
BATCH_SIZE = 32
MIN_BATCHES = 4
TARGET_RATIO = 0.20
CONFIRM_RANGE = (0.10, 0.30)
MANTISSAS = (1, 2, 3, 5)
RUNNER = ROOT / "scripts" / "run_learning_curve_point.py"


class PilotStop(RuntimeError):
    """A protocol stop: exit 3, the user decides."""


# ------------------------------------------------------------------ pure parts
def round_lambda(x: float) -> float:
    """Nearest value, in log space, of {1, 2, 3, 5} x 10^n (ties -> the smaller)."""
    if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
        raise ValueError(f"cannot round lambda {x!r}")
    n0 = math.floor(math.log10(x))
    cands = sorted(float(f"{m}e{n}") for n in (n0 - 1, n0, n0 + 1) for m in MANTISSAS)
    return min(cands, key=lambda c: (abs(math.log(x) - math.log(c)), c))


def lambda_label(lam: float) -> str:
    return f"l{lam:g}"


def make_batches(items: Sequence[Any], batch_size: Optional[int] = None,
                 min_batches: Optional[int] = None) -> List[list]:
    """Fixed batches in order, last partial dropped; < ``min_batches`` -> PilotStop.

    Defaults are the module's ``BATCH_SIZE`` / ``MIN_BATCHES``, read at call time.
    """
    batch_size = BATCH_SIZE if batch_size is None else int(batch_size)
    min_batches = MIN_BATCHES if min_batches is None else int(min_batches)
    n = len(items) // batch_size
    if n < min_batches:
        raise PilotStop(f"too few target-free validation batches: {n} batches of "
                        f"{batch_size} from {len(items)} records (need >= {min_batches})")
    return [list(items[i * batch_size:(i + 1) * batch_size]) for i in range(n)]


def in_range(r: float) -> bool:
    return CONFIRM_RANGE[0] <= r <= CONFIRM_RANGE[1]


def choose_lambda(lambda0: float, measure: Callable[[float], Dict[str, Any]]) -> Dict[str, Any]:
    """The design 12.2 steps 3-4 state machine. ``measure(lam)`` trains (or reuses)
    a pilot run at ``lam`` and returns at least ``{"r": median ratio}``.

    Returns ``{"lambda", "r0", "r_confirm", "attempts": [...]}``; raises
    ``PilotStop`` when the ratio is not confirmed after one re-estimate.
    """
    cache: Dict[float, Dict[str, Any]] = {}
    attempts: List[Dict[str, Any]] = []

    def run(lam: float, role: str) -> float:
        if lam not in cache:
            cache[lam] = measure(lam)
        r = float(cache[lam]["r"])
        attempts.append({"role": role, "lambda": lam, "r": r})
        if not (math.isfinite(r) and r > 0):
            raise PilotStop(f"validation ratio at lambda {lam:g} is {r!r}; cannot scale")
        return r

    r0 = run(lambda0, "pilot")
    lam = round_lambda(lambda0 * TARGET_RATIO / r0)
    r = run(lam, "confirm")
    if not in_range(r):
        lam = round_lambda(lam * TARGET_RATIO / r)
        r = run(lam, "re-estimate")
        if not in_range(r):
            raise PilotStop(f"lambda not confirmed: r={r:.4g} at lambda {lam:g} outside "
                            f"{list(CONFIRM_RANGE)} after one re-estimate "
                            f"(attempts {attempts})")
    return {"lambda": lam, "r0": r0, "r_confirm": r, "attempts": attempts,
            "measurements": cache}


def set_selector_epoch(trainer, epoch: int) -> int:
    """Put the batch processor (selector curriculum) and loss engine at ``epoch``."""
    trainer.batch_processor.set_epoch(epoch)
    trainer.loss_engine.set_epoch(epoch)
    got = getattr(trainer.batch_processor, "current_epoch", None)
    if got != epoch:
        raise RuntimeError(f"batch_processor.current_epoch {got!r} != {epoch} after set_epoch")
    return epoch


# ------------------------------------------------------------- pilot run
def pilot_command(config: Path, cond_dir: Path, run_dir: Path, seed: int = SEED,
                  epochs: int = EPOCHS, device: Optional[str] = None) -> List[str]:
    """The runner argument list (never a shell string)."""
    cmd = [sys.executable, str(RUNNER), "--domain", "go_ppi", "--config", str(config),
           "--train-file", str(cond_dir / "train.jsonl"), "--go-ppi-split-dir", str(cond_dir),
           "--withholding-manifest", str(cond_dir / "withholding_manifest.json"),
           "--study-cell", "loss_only", "--go-diagnostics", str(run_dir / "go_diagnostics.jsonl"),
           "--output-dir", str(run_dir), "--seed", str(seed), "--epochs", str(epochs),
           "--pilot"]
    if device:
        cmd += ["--device", device]
    return cmd


def write_pilot_config(template: Path, run_dir: Path, lam: float) -> Path:
    cfg = json.loads(template.read_text(encoding="utf-8"))
    cfg["go_ontology_loss_weight"] = float(lam)
    cfg["_cell"] = "loss_only"
    cfg["_pilot_lambda"] = float(lam)
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "config.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def run_pilot(lam: float, template: Path, cond_dir: Path, out: Path,
              device: Optional[str]) -> Path:
    run_dir = out / lambda_label(lam)
    cfg = write_pilot_config(template, run_dir, lam)
    cmd = pilot_command(cfg, cond_dir, run_dir, device=device)
    logger.info("pilot run lambda=%g -> %s", lam, wc.rel_to_root(run_dir))
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"pilot runner exited {proc.returncode} at lambda {lam:g} "
                           f"(see {wc.rel_to_root(run_dir / 'run_error.json')})")
    return run_dir


def train_diag_ratios(run_dir: Path) -> Dict[str, Optional[float]]:
    """``ratio_weighted_go_to_infonce.median`` per epoch from the run's diagnostics."""
    out: Dict[str, Optional[float]] = {}
    path = run_dir / "go_diagnostics.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[str(row["epoch"])] = (row.get("ratio_weighted_go_to_infonce") or {}).get("median")
    return out


# ------------------------------------------------------------- measurement
def measure_validation(run_dir: Path, cond_dir: Path, validation_positive: Path, lam: float,
                       seed: int = SEED, epochs: int = EPOCHS,
                       device: Optional[str] = None) -> Dict[str, Any]:
    """Design 12.2 step 2 on the pilot run in ``run_dir`` (in-process)."""
    import torch

    import go_ppi_domain.record_adapter  # noqa: F401  registers "go_ppi"
    from contrastive_learning.trainer import ContrastiveLearningTrainer
    from go_ppi_domain.go_ontology import GoIndex
    from go_ppi_domain.go_relation import GoRelationTable
    from go_ppi_domain.record_adapter import GoPpiRecordAdapter
    from go_ppi_domain.withholding_diagnostics import (
        _grad_norm, _pinned_eval, preserved_rng, projection_params)
    from scripts.run_learning_curve_point import (
        attach_go_ppi, device_scope, resolve_device)

    config = wc.load_study_config(run_dir / "config.json")
    config.training_seed = seed
    config.num_epochs = epochs
    config.orca_enabled = False
    config.go_ppi_split_dir = str(cond_dir)
    resolved = resolve_device(device or getattr(config, "training_device", "auto"))
    with device_scope(resolved):
        trainer = ContrastiveLearningTrainer(config=config,
                                             output_dir=str(run_dir / "val_measure"))
    attach_go_ppi(trainer, config)
    e = epochs - 1
    ckpt = torch.load(run_dir / f"checkpoint_epoch_{e}.pt", map_location=trainer.device,
                      weights_only=False)
    if ckpt.get("epoch") != e:
        raise RuntimeError(f"checkpoint_epoch_{e}.pt records epoch {ckpt.get('epoch')}")
    trainer.model.load_state_dict(ckpt["model_state_dict"], strict=True)
    set_selector_epoch(trainer, e)

    adapter = GoPpiRecordAdapter(config)
    samples = []
    with open(validation_positive, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec["metadata"]["go_grade"]) != 2:
                continue
            s = adapter.build_sample(rec, i, config)
            if s is not None:
                samples.append(s)
    batches = make_batches(samples)
    proteins = set()
    for s in (x for b in batches for x in b):
        proteins.add(str(s.resume["protein_id"]))
        proteins.add(str(s.job["partner_id"]))
    table = GoRelationTable.from_index(GoIndex.load(config.go_index_cache), proteins)
    params = projection_params(trainer)
    engine = trainer.loss_engine
    ratios: List[Optional[float]] = []
    prev = engine._diagnostic_mode
    with preserved_rng(), torch.enable_grad(), _pinned_eval(trainer.model):
        engine._diagnostic_mode = True
        try:
            for batch in batches:
                triplets = trainer.batch_processor.process_batch(batch, None, None)
                if not triplets:
                    ratios.append(None)
                    continue
                emb = trainer._generate_embeddings(triplets)
                comps = engine.go_ordinal_components(triplets, emb, relation_table=table)
                gi = _grad_norm(comps["infonce"], params)
                gg = _grad_norm(comps["go_relation"], params)
                ratios.append(lam * gg / gi if (gg is not None and gi) else None)
        finally:
            engine._diagnostic_mode = prev
    vals = [r for r in ratios if r is not None]
    if not vals:
        raise RuntimeError("no batch produced a gradient ratio")
    return {"r": float(statistics.median(vals)), "ratios_val_per_batch": ratios,
            "n_records": len(samples), "n_batches": len(batches), "selector_epoch": e,
            "relation_table_proteins": len(table.ids)}


# ------------------------------------------------------------------- main
def run(out: Path, template: Path, cond_dir: Path, validation_positive: Path,
        device: Optional[str], lambda0: float = LAMBDA0) -> Dict[str, Any]:
    from scripts.go_withholding import materialize_configs

    for p in (template, cond_dir / "withholding_manifest.json", validation_positive):
        if not p.is_file():
            raise RuntimeError(f"missing input {wc.rel_to_root(p)}")
    wc.assert_writable_target(out)
    choice_path = out / "lambda_choice.json"
    if choice_path.exists():
        raise RuntimeError(f"refusing to overwrite {wc.rel_to_root(choice_path)}")
    out.mkdir(parents=True, exist_ok=True)
    runs: Dict[float, Dict[str, Any]] = {}

    def measure(lam: float) -> Dict[str, Any]:
        run_dir = run_pilot(lam, template, cond_dir, out, device)
        m = measure_validation(run_dir, cond_dir, validation_positive, lam, device=device)
        m["run_dir"] = run_dir
        m["train_diag"] = train_diag_ratios(run_dir)
        runs[lam] = m
        logger.info("lambda=%g: validation ratio median %.4g over %d batches", lam, m["r"],
                    m["n_batches"])
        return m

    result = choose_lambda(lambda0, measure)
    lam = result["lambda"]
    m = runs[lam]
    choice = {
        "lambda": lam, "lambda0": lambda0, "r0": result["r0"],
        "r_confirm": result["r_confirm"],
        "ratios_val_per_batch": {f"{k:g}": v["ratios_val_per_batch"] for k, v in runs.items()},
        "ratios_train_diag_by_epoch": {f"{k:g}": v["train_diag"] for k, v in runs.items()},
        "n_records": m["n_records"], "n_batches": m["n_batches"], "batch_size": BATCH_SIZE,
        "seed": SEED, "epochs": EPOCHS, "selector_epoch": m["selector_epoch"],
        "attempts": result["attempts"],
        "pilot_run_dirs": [wc.rel_to_root(v["run_dir"]) for v in runs.values()],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_sha256": wc.code_sha256()[0],
    }
    choice_path.write_text(json.dumps(wc.relativize_json(choice), indent=2, sort_keys=True)
                           + "\n", encoding="utf-8")
    wc.append_pin("G4-lambda", {"lambda_choice_sha256": wc.sha256_file(choice_path),
                                "lambda": lam, "r0": result["r0"],
                                "r_confirm": result["r_confirm"]})
    rc = materialize_configs.main(["--lambda", repr(float(lam))])
    if rc != 0:
        raise RuntimeError(f"materialize_configs --lambda {lam:g} exited {rc}")
    return choice


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--template", type=Path, default=None,
                    help="default: CONFIG_DIR/loss_pilot_template.json")
    ap.add_argument("--condition", type=Path, default=None, help="default: CONDITIONS/all")
    ap.add_argument("--validation-positive", type=Path, default=None,
                    help="default: VALIDATION_TF/validation_positive.jsonl")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    out = Path(args.out or (Path(wc.STUDY_RES) / "lambda_pilot"))
    template = Path(args.template or (Path(wc.CONFIG_DIR) / "loss_pilot_template.json"))
    cond = Path(args.condition or (Path(wc.CONDITIONS) / "all"))
    vpos = Path(args.validation_positive
                or (Path(wc.VALIDATION_TF) / "validation_positive.jsonl"))
    try:
        choice = run(out, template, cond, vpos, args.device)
    except PilotStop as exc:
        logger.error("PROTOCOL STOP (ask the user): %s", exc)
        return 3
    except Exception as exc:  # noqa: BLE001
        logger.error("lambda_pilot failed: %s: %s", type(exc).__name__, exc)
        return 2
    print(json.dumps({"lambda": choice["lambda"], "r0": choice["r0"],
                      "r_confirm": choice["r_confirm"], "n_batches": choice["n_batches"],
                      "wall_seconds": round(time.time() - t0, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
