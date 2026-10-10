"""Run-time guards and per-epoch diagnostics for the GO/PPI withholding study.

Design sections 4.5 (``StrictEpochGuard``), 4.6 (``SelectionRecorder``) and 6
(``EpochDiagnostics``). All three are trainer ``epoch_end_callbacks``
(``callback(trainer, epoch, epoch_stats)``, trainer hook of design 5.5), installed
by the study runner in the order recorder, diagnostics, guard.

* ``StrictEpochGuard`` raises outside the trainer's per-batch ``try``, so a
  swallowed batch failure becomes ``train()``'s emergency-checkpoint-and-re-raise
  path instead of a silently partial epoch.
* ``SelectionRecorder`` wraps the real selector the batch processor consumes and
  hashes each epoch's (anchor, positive, negatives) draws. Selector exceptions are
  counted even while paused (design 4.6: intentionally fatal at post-training).
* ``EpochDiagnostics`` measures loss components and their gradient norms on the
  projection parameters over a fixed diagnostic sample. It never accumulates
  ``.grad``, never touches the optimizer, and restores RNG state, recorder pause,
  ``_diagnostic_mode`` and the model's training mode in ``finally``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

LOSS_ON_CELLS = ("loss_only", "both")
DRIFT_TOLERANCE = 1e-5


# ------------------------------------------------------------------ helpers
@contextmanager
def preserved_rng():
    """Save and restore the ``random``, numpy and torch CPU RNG states.

    On a CUDA run (remote GPU execution, user decision B) the initialised CUDA
    generators are saved and restored too.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() and torch.cuda.is_initialized() else None)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def projection_params(trainer) -> List[torch.nn.Parameter]:
    """Trainable parameters of ``trainer.model``; none may belong to ``trainer.text_encoder``."""
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    encoder = getattr(trainer, "text_encoder", None)
    if encoder is not None and hasattr(encoder, "parameters"):
        enc_ids = {id(p) for p in encoder.parameters()}
        shared = [p for p in params if id(p) in enc_ids]
        if shared:
            raise RuntimeError(
                f"projection_params: {len(shared)} trainable model parameter(s) belong "
                f"to trainer.text_encoder")
    if not params:
        raise RuntimeError("projection_params: trainer.model has no trainable parameters")
    return params


def projection_param_summary(trainer) -> Dict[str, Any]:
    """``{"names", "numel"}`` of the projection parameters (lc_manifest, design 4.7)."""
    keep = {id(p) for p in projection_params(trainer)}
    names = [n for n, p in trainer.model.named_parameters() if id(p) in keep]
    return {"names": names, "numel": int(sum(p.numel() for p in trainer.model.parameters()
                                             if id(p) in keep))}


# ------------------------------------------------------------ epoch guard
class StrictEpochGuard:
    """Turns swallowed batch failures and a silently inactive L_GO into a fatal error."""

    def __init__(self, cell: str, min_applied_share: float = 0.9) -> None:
        self.cell = str(cell)
        self.loss_on = self.cell in LOSS_ON_CELLS
        self.min_applied_share = float(min_applied_share)
        self.epochs: List[Dict[str, Any]] = []
        self._last_applied = 0

    def __call__(self, trainer, epoch: int, epoch_stats: dict) -> None:
        failed = int(epoch_stats["failed_batches"])
        ok = int(epoch_stats["ok_batches"])
        avg = float(epoch_stats["avg_loss"])
        applied_total = int(getattr(trainer.loss_engine, "go_relation_applied_batches", 0))
        applied = applied_total - self._last_applied
        self._last_applied = applied_total
        self.epochs.append({"epoch": int(epoch), "failed_batches": failed,
                            "ok_batches": ok, "avg_loss": avg,
                            "go_relation_applied": applied})
        problems = []
        if failed > 0:
            problems.append(f"{failed} failed batch(es)")
        if ok == 0:
            problems.append("no successful batch")
        if not math.isfinite(avg):
            problems.append(f"non-finite avg_loss {avg!r}")
        if self.loss_on:
            if applied < self.min_applied_share * ok:
                problems.append(f"L_GO applied in {applied} of {ok} batches "
                                f"(< {self.min_applied_share:.2f})")
        elif applied_total != 0:
            problems.append(f"L_GO applied in {applied_total} batches in loss-off cell")
        if problems:
            raise RuntimeError(f"StrictEpochGuard[{self.cell}] epoch {epoch}: "
                               + "; ".join(problems))


# ------------------------------------------------------- selection recorder
class SelectionRecorder:
    """Wraps ``selector.select_batch_negatives`` and hashes each epoch's draws."""

    def __init__(self, selector, unannotated_anchors: Iterable[str] = ()) -> None:
        self._inner = selector.select_batch_negatives
        # Instance attribute; BatchProcessor resolves it via getattr at call time.
        selector.select_batch_negatives = self._wrapped
        self.paused = False
        self.calls = self.none_returns = self.exceptions = 0
        self._unannotated = frozenset(str(a) for a in unannotated_anchors)
        self._lines: List[str] = []
        self.by_epoch: Dict[str, str] = {}
        self.calls_by_epoch: Dict[str, int] = {}
        self._n_unannotated = 0
        self.unannotated_anchor_calls_by_epoch: Dict[str, int] = {}

    def _wrapped(self, anchor_sample, candidate_negatives, max_negatives, epoch=None):
        try:
            out = self._inner(anchor_sample, candidate_negatives, max_negatives, epoch)
        except Exception:
            self.exceptions += 1      # counted even while paused: intentional (design 4.6)
            raise
        if self.paused:
            return out
        self.calls += 1
        if out is None:
            self.none_returns += 1
            return out
        anchor = str(anchor_sample.metadata.get("resume_id")
                     or anchor_sample.resume.get("protein_id"))
        pos = str(anchor_sample.job.get("partner_id"))
        negs = ",".join(f"{v['partner_id']}:{v['grade']}" for v in out[0])
        self._lines.append(f"{epoch}\t{anchor}\t{pos}\t{negs}")
        if anchor in self._unannotated:
            self._n_unannotated += 1
        return out

    @contextmanager
    def pause(self):
        prev = self.paused
        self.paused = True
        try:
            yield self
        finally:
            self.paused = prev

    def on_epoch_end(self, trainer, epoch, epoch_stats) -> None:
        key = str(epoch)
        self.by_epoch[key] = hashlib.sha256(
            "\n".join(sorted(self._lines)).encode("utf-8")).hexdigest()
        self.calls_by_epoch[key] = len(self._lines)
        self._lines = []
        self.unannotated_anchor_calls_by_epoch[key] = self._n_unannotated
        self._n_unannotated = 0


def negative_selection_block(recorder: SelectionRecorder, selector) -> Dict[str, Any]:
    """The lc_manifest ``negative_selection`` block (design 4.7), one key per source."""
    return {
        "selection_sha256_by_epoch": dict(recorder.by_epoch),
        "calls_by_epoch": dict(recorder.calls_by_epoch),
        "unannotated_anchor_calls_by_epoch": dict(recorder.unannotated_anchor_calls_by_epoch),
        "unannotated_anchor_window": bool(selector.unannotated_anchor_window),
        "selector_none_returns": int(recorder.none_returns),
        "selector_exceptions": int(recorder.exceptions),
    }


# --------------------------------------------------------- epoch diagnostics
@contextmanager
def _pinned_eval(model):
    """Eval mode for the whole pass.

    ``trainer._generate_embeddings`` calls ``self.model.train()`` before the
    projection, which would re-enable dropout; an instance-level no-op ``train``
    shadows it for the duration, and the original mode is restored afterwards.
    """
    was_training = model.training
    model.eval()
    model.train = lambda mode=True: model
    try:
        yield
    finally:
        try:
            del model.train
        except AttributeError:
            pass
        model.train(was_training)


def _grad_norm(x: Optional[torch.Tensor], params: List[torch.nn.Parameter]) -> Optional[float]:
    if x is None:
        return None
    if not x.requires_grad:
        return 0.0
    grads = torch.autograd.grad(x, params, retain_graph=True, allow_unused=True)
    flat = [g.reshape(-1) for g in grads if g is not None]
    if not flat:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(flat)).item())


def _summary(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"median": None, "mean": None}
    return {"median": float(statistics.median(vals)), "mean": float(statistics.fmean(vals))}


def _median(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else None


class EpochDiagnostics:
    """Per-epoch loss and gradient-scale diagnostics on a fixed train sample (design 6)."""

    def __init__(self, out_path: Path, train_file: Path, recorder: SelectionRecorder,
                 n_batches: int = 8, batch_size: int = 64,
                 relation_table=None, weight: float = 0.0) -> None:
        self.out_path = Path(out_path)
        self.train_file = Path(train_file)
        self.recorder = recorder
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.table = relation_table
        self.weight = float(weight)
        self._batches: Optional[List[list]] = None
        self.rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------- setup
    def _load_batches(self, config) -> List[list]:
        """First ``n_batches * batch_size`` train lines -> TrainingSamples, fixed order."""
        from go_ppi_domain.record_adapter import GoPpiRecordAdapter

        adapter = GoPpiRecordAdapter(config)
        samples = []
        limit = self.n_batches * self.batch_size
        with open(self.train_file, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if len(samples) >= limit:
                    break
                if not line.strip():
                    continue
                s = adapter.build_sample(json.loads(line), i, config)
                if s is not None:
                    samples.append(s)
        return [samples[i:i + self.batch_size]
                for i in range(0, len(samples), self.batch_size)]

    def _ensure_table(self, config):
        if self.table is None:
            from go_ppi_domain.go_relation import GoRelationTable
            self.table = GoRelationTable.from_train_file(
                Path(config.go_index_cache), self.train_file)
        return self.table

    # ---------------------------------------------------------------- call
    def __call__(self, trainer, epoch: int, epoch_stats: dict) -> None:
        t0 = time.time()
        row: Dict[str, Any] = {
            "epoch": int(epoch), "n_batches": 0, "n_triplets": 0, "lambda": self.weight,
            "infonce_mean": None, "ranking_mean": None, "go_relation_mean": None,
            "max_component_drift": None, "drift_flag": False,
            "grad_norm_infonce": {"median": None, "mean": None},
            "grad_norm_ranking": {"median": None, "mean": None},
            "grad_norm_go_relation_unweighted": {"median": None, "mean": None},
            "ratio_weighted_go_to_infonce": {"median": None, "per_batch": []},
            "ratio_ranking_to_infonce": {"median": None},
            "grad_norm_total_preclip_eval_diag": {"median": None,
                                                  "share_above_clip_eval_diag": None},
            "relation_skipped_batches": 0, "epoch_stats": dict(epoch_stats),
            "elapsed_s": None, "error": None,
        }
        engine = trainer.loss_engine
        prev_mode = getattr(engine, "_diagnostic_mode", False)
        try:
            with preserved_rng(), self.recorder.pause(), torch.enable_grad(), \
                    _pinned_eval(trainer.model):
                engine._diagnostic_mode = True
                try:
                    self._measure(trainer, row)
                finally:
                    engine._diagnostic_mode = prev_mode
        except Exception as exc:   # non-fatal by design (secondary outcome)
            row["error"] = repr(exc)
            logger.warning("EpochDiagnostics epoch %s failed: %r", epoch, exc)
        finally:
            engine._diagnostic_mode = prev_mode
        row["elapsed_s"] = round(time.time() - t0, 3)
        if row["drift_flag"]:
            logger.error("EpochDiagnostics epoch %s: component drift %.3g > %.0e",
                         epoch, row["max_component_drift"], DRIFT_TOLERANCE)
        self.rows.append(row)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def _measure(self, trainer, row: Dict[str, Any]) -> None:
        config = trainer.config
        if self._batches is None:
            self._batches = self._load_batches(config)
        table = self._ensure_table(config)
        params = projection_params(trainer)
        clip = float(getattr(config, "gradient_clip_norm", 1.0))
        engine = trainer.loss_engine
        inf, rank, rel, drifts = [], [], [], []
        g_inf, g_rank, g_rel, g_tot, ratio_go, ratio_rank = [], [], [], [], [], []
        skipped = 0
        n_triplets = 0
        for batch in self._batches:
            triplets = trainer.batch_processor.process_batch(batch, None, None)
            if not triplets:
                continue
            emb = trainer._generate_embeddings(triplets)
            comps = engine.go_ordinal_components(triplets, emb, relation_table=table)
            ref = engine._compute_go_ordinal_loss(triplets, emb)
            drifts.append(abs(float(comps["base_total"].item()) - float(ref.item())))
            n_triplets += len(triplets)
            inf.append(float(comps["infonce"].item()))
            rank.append(float(comps["ranking"].item()))
            go = comps["go_relation"]
            if go is None:
                skipped += 1
            else:
                rel.append(float(go.item()))
            gi = _grad_norm(comps["infonce"], params)
            gr = _grad_norm(comps["ranking"], params)
            gg = _grad_norm(go, params)
            total = comps["base_total"] if go is None else comps["base_total"] + self.weight * go
            gt = _grad_norm(total, params)
            g_inf.append(gi)
            g_rank.append(gr)
            g_rel.append(gg)
            g_tot.append(gt)
            ratio_go.append((self.weight * gg / gi) if (gg is not None and gi) else None)
            ratio_rank.append((gr / gi) if gi else None)
            del comps, ref, emb, total, go
        row["n_batches"] = len(drifts)
        row["n_triplets"] = n_triplets
        row["infonce_mean"] = statistics.fmean(inf) if inf else None
        row["ranking_mean"] = statistics.fmean(rank) if rank else None
        row["go_relation_mean"] = statistics.fmean(rel) if rel else None
        row["max_component_drift"] = max(drifts) if drifts else None
        row["drift_flag"] = bool(drifts and max(drifts) > DRIFT_TOLERANCE)
        row["grad_norm_infonce"] = _summary(g_inf)
        row["grad_norm_ranking"] = _summary(g_rank)
        row["grad_norm_go_relation_unweighted"] = _summary(g_rel)
        row["ratio_weighted_go_to_infonce"] = {"median": _median(ratio_go),
                                               "per_batch": ratio_go}
        row["ratio_ranking_to_infonce"] = {"median": _median(ratio_rank)}
        tot = [v for v in g_tot if v is not None]
        row["grad_norm_total_preclip_eval_diag"] = {
            "median": _median(tot),
            "share_above_clip_eval_diag": (sum(v > clip for v in tot) / len(tot)) if tot else None}
        row["relation_skipped_batches"] = skipped
