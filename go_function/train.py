"""Trainer, run-spec schema, preflight and in-process evaluation (design §7, §12.1).

One run = one JSON spec (design §7.1 + ``mode`` and ``max_train_batches``),
executed by ``scripts/go_function/run.py --spec PATH --device auto|cuda|cpu``:

  preflight  spec schema; study.json equality; code-pin import coverage (no module
             from a protected directory); pins of labels, condition, target table,
             validation mask, features, normal forms and term text; design §5.5
             assertions 1-10 re-run on the condition files
  train      Adam + MultiStepLR, seeded torch.Generator shuffling, deterministic
             flags; masked BCE (+ lambda * EL for EL arms); epochs.jsonl per epoch
  eval       eval_inputs.npz (logits dump) -> eval.json -> scores.npz -> run.json
             -> DONE.json; eval_inputs.npz is deleted after success. Pilot runs
             (protocol §6.4) never load test labels or test features and write an
             eval.json stub {"skipped": ...} and no scores.npz

Closed world (protocol D19): W1/W2/``all`` conditions have an empty train mask.
``loss_mask`` returns None for them and no mask tensor is ever allocated, so the
per-row loss mask of a W2 condition is identical to the base data's (none); only
the erased labels differ. The only training masks are the WDGZrel/W2drop columns
over all train rows.

Any exception writes ``FAILED.json`` with ``stage`` in {preflight, train, eval}.
Every path written is repo-relative.

``_grad_norm`` is reimplemented from go_ppi_domain/withholding_diagnostics.py,
function ``_grad_norm`` (lines 218-226; source file sha256
2251327239da77d4be364954a29b19145863c56d893e809457a9754a803542f0); not imported.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from go_function import evaluate as E
from go_function.constants import (ARMS, EXIT_FATAL, EXIT_OK, EXIT_REFUSED, EXIT_STOP,
                                   REGIMES, RULES, SEEDS)
from go_function.pins import (PROTECTED, assert_writable_target, canonical_json,
                              npz_bytes, sha256_bytes, sha256_file)
from go_function.runtime import ROOT, FatalError, ProtocolStop, repo_path, repo_rel

STUDY_PATH = ROOT / "config" / "go_function" / "study.json"
CODE_PIN_PATH = ROOT / "results" / "go_function" / "code_pin.json"
RESULTS_PREFIX = "results/go_function/"
MODES = ("study", "gate", "pilot", "smoke")
FEATURES = ("esm2_650m", "interpro")
SPEC_KS = (0, 1, 5, 10, 25, "all")
PACKS = tuple(f"p{i:02d}" for i in range(1, 9))
MAX_EPOCHS = 256
GRAD_EVERY = 50
EVAL_BATCH = 512
EL_ARMS = ("V1", "V2", "V3", "V3c", "V4", "V6")
#: eval.json of a pilot run (protocol §6.4, amendment A4): no test data are read
PILOT_EVAL_SKIPPED = "protocol §6.4: pilots never read test data"
#: design §13.1: no imported module may come from these directories.
PROTECTED_IMPORTS = tuple(ROOT / p for p in ("go_ppi_domain", "scripts/go_withholding",
                                             "contrastive_learning", "trials_domain")) + PROTECTED


def _name(p: Path) -> str:
    """Repo-relative path for messages; the bare name outside the repo (tests)."""
    try:
        return repo_rel(p)
    except ValueError:
        return Path(p).name


class Refused(RuntimeError):
    """Refused because of the state of the study (exit 4), e.g. a missing pin."""
    exit_code = EXIT_REFUSED


class NonFiniteLoss(FatalError):
    pass


# ------------------------------------------------------------------ study.json (§12.1)
def load_study(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path) if path is not None else STUDY_PATH
    if not path.exists():
        raise FatalError(f"{_name(path)} is missing")
    return json.loads(path.read_text())


def set_study_field(name: str, value: Any, path: Optional[Path] = None) -> str:
    """Set a study.json field that is null (e.g. ``norm`` after G3). Equal value ->
    unchanged; a different non-null value is fatal (pinned config)."""
    path = Path(path) if path is not None else STUDY_PATH
    doc = load_study(path)
    if name not in doc:
        raise FatalError(f"study.json has no field {name!r}")
    if doc[name] is not None:
        if doc[name] != value:
            raise FatalError(f"study.json {name} is {doc[name]!r}; refusing to change it to {value!r}")
        return "unchanged"
    doc[name] = value
    assert_writable_target(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(canonical_json(doc))
    tmp.replace(path)
    return "written"


# ------------------------------------------------------------------ run spec (§7.1)
#: field -> required (True) or default value
SPEC_FIELDS: Dict[str, Any] = {
    "aspect": True, "feature": True, "variant": True, "regime": True, "rule": True,
    "retention_draw": 1, "pack": True, "k": True, "seed": True, "epochs": True,
    "lambda_el": True, "batch_size": True, "lr": True, "milestones": True, "gamma": True,
    "selection": True, "out_dir": True, "mode": True, "max_train_batches": None,
    "nf4_radius": "dgz",
}


@dataclass(frozen=True)
class RunSpec:
    aspect: str
    feature: str
    variant: str
    regime: str
    rule: str
    retention_draw: int
    pack: Optional[str]
    k: Any
    seed: int
    epochs: int
    lambda_el: Optional[float]
    batch_size: int
    lr: float
    milestones: Tuple[int, ...]
    gamma: float
    selection: str
    out_dir: str
    mode: str
    max_train_batches: Optional[int] = None
    nf4_radius: str = "dgz"

    @property
    def is_el(self) -> bool:
        return self.variant in EL_ARMS

    @property
    def condition_rule(self) -> str:
        """withholding.Spec rule: the pack-free condition is ``all``."""
        return "all" if self.pack is None else self.rule

    @property
    def condition_dir(self) -> str:
        base = f"preprocess/go_function/{self.aspect}/conditions/{self.regime}"
        if self.pack is None:
            return f"{base}/all"
        return f"{base}/{self.rule}/{self.pack}_k{self.k}" + ("_d2" if self.retention_draw == 2 else "")

    @property
    def run_name(self) -> str:
        return (f"{self.variant}__{self.regime}__{self.rule}__{self.pack or 'all'}__k{self.k}"
                + ("__d2" if self.retention_draw == 2 else "") + f"__s{self.seed}")

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["milestones"] = list(self.milestones)
        return d


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_spec(raw: Mapping[str, Any], study: Mapping[str, Any]) -> RunSpec:
    """Design §7.1 schema; any violation (or unknown field) is fatal (exit 2)."""
    def bad(msg):
        raise FatalError(f"run spec: {msg}")

    if not isinstance(raw, Mapping):
        bad("not a JSON object")
    unknown = sorted(set(raw) - set(SPEC_FIELDS))
    if unknown:
        bad(f"unknown fields {unknown}")
    missing = sorted(f for f, req in SPEC_FIELDS.items() if req is True and f not in raw)
    if missing:
        bad(f"missing fields {missing}")
    s = {f: raw.get(f, d if d is not True else None) for f, d in SPEC_FIELDS.items()}
    mode = s["mode"]
    if mode not in MODES:
        bad(f"mode must be one of {MODES}")
    if s["aspect"] not in ("bp", "mf"):
        bad("aspect must be bp or mf")
    if s["feature"] not in FEATURES:
        bad(f"feature must be one of {FEATURES}")
    if s["variant"] not in ARMS:
        bad(f"variant must be one of {ARMS}")
    if s["regime"] not in REGIMES:
        bad(f"regime must be one of {REGIMES}")
    if s["regime"] == "TPnoIPI" and s["aspect"] != "mf":
        bad("regime TPnoIPI is MF only")
    if s["rule"] not in RULES:
        bad(f"rule must be one of {RULES}")
    k = s["k"]
    if not (k == "all" if isinstance(k, str) else _is_int(k) and k in SPEC_KS):
        bad(f"k must be one of {SPEC_KS}")
    pack = s["pack"]
    if pack is not None and pack not in PACKS:
        bad(f"pack must be one of p01..p08 or null")
    if (pack is None) != (k == "all"):
        bad("pack is null exactly for k = 'all' (the 'all' condition)")
    if s["rule"] in ("WDGZrel", "W2drop") and (s["regime"] != "DGZ" or k != 0):
        bad(f"rule {s['rule']} exists only for regime DGZ at k = 0")
    if pack is None and s["rule"] not in ("W2", "W1"):
        bad("the 'all' condition is addressed with rule W2 or W1")
    draw = s["retention_draw"]
    if draw not in (1, 2) or not _is_int(draw):
        bad("retention_draw must be 1 or 2")
    if draw == 2 and not (s["rule"] == "W2" and s["regime"] == "TP" and k in (1, 5)):
        bad("retention_draw 2 exists only for rule W2, regime TP, k in {1, 5}")
    if not _is_int(s["seed"]):
        bad("seed must be an int")
    if mode == "study" and s["seed"] not in SEEDS:
        bad(f"study seeds are {SEEDS}")
    ep = s["epochs"]
    if not _is_int(ep) or not 1 <= ep <= MAX_EPOCHS:
        bad(f"epochs must be an int in 1..{MAX_EPOCHS}")
    lam = s["lambda_el"]
    is_el = s["variant"] in EL_ARMS
    if is_el:
        if not _is_num(lam) or not lam > 0:
            bad("lambda_el must be a float > 0 for EL arms")
        if mode in ("study", "gate") and float(lam) != float(study["lambda_el"]):
            bad(f"lambda_el must equal {study['lambda_el']} in {mode} mode")
    elif lam is not None:
        bad("lambda_el must be null for arms without an axiom loss")
    for f in ("batch_size", "gamma", "lr"):
        if not _is_num(s[f]) or (f == "batch_size" and not _is_int(s[f])):
            bad(f"{f} has the wrong type")
        if s[f] != study[f]:
            bad(f"{f} must equal study.json ({study[f]})")
    ms = s["milestones"]
    if not isinstance(ms, list) or not all(_is_int(m) for m in ms) or ms != list(study["milestones"]):
        bad(f"milestones must equal study.json ({study['milestones']})")
    sel = s["selection"]
    if sel not in ("final", "best_valid_loss"):
        bad("selection must be 'final' or 'best_valid_loss'")
    if sel == "best_valid_loss" and mode != "gate":
        bad("selection best_valid_loss is for gate runs only")
    if mode == "study" and sel != "final":
        bad("study runs use selection 'final'")
    if mode == "study":
        key = f"{s['aspect']}_{s['feature']}"
        star = (study.get("epochs_star") or {}).get(key)
        if star is None:
            bad(f"E* for {key} is not pinned in study.json (G7)")
        if ep != star:
            bad(f"epochs must equal the pinned E* ({star}) for {key}")
    mtb = s["max_train_batches"]
    if mtb is not None:
        if mode != "smoke":
            bad("max_train_batches is accepted only in smoke mode")
        if not _is_int(mtb) or mtb < 1:
            bad("max_train_batches must be an int >= 1")
    rad = s["nf4_radius"]
    if rad not in ("dgz", "class"):
        bad("nf4_radius must be 'dgz' or 'class'")
    if rad == "class" and (mode != "gate" or s["variant"] != "V4"):
        bad("nf4_radius 'class' is a G4 diagnostic (gate mode, V4) only (protocol D10)")
    if mode != "gate" and rad != study.get("nf4_radius", "dgz"):
        bad("nf4_radius must equal study.json")
    od = s["out_dir"]
    if not isinstance(od, str) or Path(od).is_absolute() or ".." in Path(od).parts \
            or not od.startswith(RESULTS_PREFIX) or od.rstrip("/") == RESULTS_PREFIX.rstrip("/"):
        bad(f"out_dir must be a repo-relative path under {RESULTS_PREFIX}")
    return RunSpec(aspect=s["aspect"], feature=s["feature"], variant=s["variant"],
                   regime=s["regime"], rule=s["rule"], retention_draw=draw, pack=pack, k=k,
                   seed=s["seed"], epochs=ep, lambda_el=float(lam) if lam is not None else None,
                   batch_size=s["batch_size"], lr=float(s["lr"]), milestones=tuple(ms),
                   gamma=float(s["gamma"]), selection=sel, out_dir=od.rstrip("/"), mode=mode,
                   max_train_batches=mtb, nf4_radius=rad)


def check_study_ready(study: Mapping[str, Any]) -> None:
    """The norm type must be fixed by G3 before any training run (design §6.1)."""
    if study.get("norm") not in ("layernorm", "batchnorm1d"):
        raise Refused("study.json norm is not set; run gate_checkpoints.py (G3) first")


# ------------------------------------------------------------------ code pin (§13.1)
def _under(p: Path, roots: Sequence[Path]) -> bool:
    return any(p == r.resolve() or p.is_relative_to(r.resolve()) for r in roots)


def check_import_coverage(mode: str, code_pin_path: Optional[Path] = None,
                          module_files: Optional[Sequence[str]] = None, log=None) -> Dict[str, Any]:
    """Every imported module file inside the repo (outside .venv) must be covered by
    ``code_pin.json`` and none may come from a protected directory (fatal). Without
    a code pin: refused in study mode, a warning otherwise. In study mode every file
    listed in the code pin is also hashed on disk, so an edit after the pin is caught
    whether or not the module has been imported yet (e.g. go_function.model, which
    the trainer imports after preflight)."""
    if module_files is None:
        module_files = [getattr(m, "__file__", None) for m in list(sys.modules.values())]
    venv = (ROOT / ".venv").resolve()
    inside = sorted({Path(f).resolve() for f in module_files if f
                     and Path(f).resolve().is_relative_to(ROOT)
                     and not Path(f).resolve().is_relative_to(venv)})
    bad = [repo_rel(p) for p in inside if _under(p, PROTECTED_IMPORTS)]
    if bad:
        raise FatalError(f"modules imported from protected directories: {bad}")
    path = Path(code_pin_path) if code_pin_path is not None else CODE_PIN_PATH
    if not path.exists():
        if mode == "study":
            raise Refused(f"{_name(path)} is missing (G8); study runs need the code pin")
        if log is not None:
            log.warning("no code pin (%s); import coverage not checked (mode %s)",
                        _name(path), mode)
        return {"code_pin_sha256": None, "n_modules": len(inside)}
    pin = json.loads(path.read_text())
    files = pin.get("files", {})
    uncovered, changed, missing = [], [], []
    for p in inside:
        rel = repo_rel(p)
        if rel not in files:
            uncovered.append(rel)
        elif sha256_file(p) != files[rel]:
            changed.append(rel)
    n_files_verified = 0
    if mode == "study":
        seen = set(changed)
        for rel, want in sorted(files.items()):
            p = repo_path(rel)
            n_files_verified += 1
            if not p.exists():
                missing.append(rel)
            elif rel not in seen and sha256_file(p) != want:
                changed.append(rel)
    if uncovered or changed or missing:
        raise FatalError(f"code pin: uncovered modules {uncovered}; changed files {changed}; "
                         f"missing files {missing}")
    return {"code_pin_sha256": sha256_file(path), "n_modules": len(inside),
            "n_pinned_files_verified": n_files_verified}


# ------------------------------------------------------------------ data containers
class RowStore:
    """Rows of a 0/1 CSR matrix as dense bool batches. On CUDA the matrix is
    densified once on the device (design §7.2); on CPU per batch."""

    def __init__(self, m: sp.spmatrix, device: str):
        import torch
        self.m = sp.csr_matrix(m)
        self.shape = self.m.shape
        self.dense = None
        if str(device).startswith("cuda"):
            self.dense = torch.as_tensor(self.m.toarray().astype(bool), device=device)
        self.device = device

    def rows(self, idx: np.ndarray):
        import torch
        if self.dense is not None:
            return self.dense[torch.as_tensor(idx, device=self.device)]
        return torch.as_tensor(self.m[idx].toarray().astype(bool), device=self.device)


def loss_mask(pairs: np.ndarray, shape: Tuple[int, int], device: str) -> Optional[RowStore]:
    """The train loss mask from ``train_mask.npz`` pairs (True = excluded). An empty
    pair list (W1/W2/``all``, closed world) returns None: no mask tensor exists."""
    pairs = np.asarray(pairs).reshape(-1, 2)
    if len(pairs) == 0:
        return None
    m = sp.csr_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
                      shape=shape)
    m.data[:] = 1
    return RowStore(m, device)


class FeatureStore:
    """Dense float32 rows from an ndarray (ESM) or a CSR matrix (InterPro)."""

    def __init__(self, x, device: str):
        self.x = sp.csr_matrix(x) if sp.issparse(x) else np.asarray(x, dtype=np.float32)
        self.shape = self.x.shape
        self.device = device

    def rows(self, idx):
        import torch
        blk = self.x[idx]
        blk = blk.toarray() if sp.issparse(blk) else blk
        return torch.as_tensor(np.asarray(blk, dtype=np.float32), device=self.device)


def batch_targets(labels: RowStore, mask: Optional[RowStore], rows: np.ndarray):
    """(y, masked) of one train batch; masked is None when the condition has no mask."""
    return labels.rows(rows), (None if mask is None else mask.rows(rows))


@dataclass
class TargetCols:
    id: str
    column: int
    Pa: List[int]
    siblings: List[int]
    X: List[int]
    Desc: List[int]


@dataclass
class RunData:
    """Everything a run reads, already verified by preflight."""
    x_train: Any
    x_valid: Any
    x_eval: Any
    y_train: sp.csr_matrix
    train_mask: np.ndarray
    y_valid: sp.csr_matrix
    valid_keep_cols: np.ndarray
    vocab_ids: List[str]
    y_eval: Optional[sp.csr_matrix] = None
    eval_rows: Optional[np.ndarray] = None
    nf: Any = None
    text: Optional[np.ndarray] = None
    targets: List[TargetCols] = field(default_factory=list)
    collateral: Optional[Dict[str, np.ndarray]] = None
    gate: Optional[Dict[str, Any]] = None
    hashes: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ diagnostics
def _grad_norm(x, params) -> Optional[float]:
    """L2 norm of d x / d params (None for no loss; 0 when x does not depend on them).
    Reimplemented from go_ppi_domain/withholding_diagnostics._grad_norm (see header)."""
    import torch
    if x is None:
        return None
    if not x.requires_grad:
        return 0.0
    grads = torch.autograd.grad(x, params, retain_graph=True, allow_unused=True)
    flat = [g.reshape(-1) for g in grads if g is not None]
    if not flat:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(flat)).item())


def grad_norms(model, bce, el_weighted) -> Dict[str, Optional[float]]:
    """Gradient norms of BCE and lambda*EL on the protein and the term tower, and the
    lambda*EL / BCE ratio per tower (design §7.2)."""
    prot, term = model.protein_parameters(), model.term_parameters()
    out = {"bce_protein": _grad_norm(bce, prot), "bce_term": _grad_norm(bce, term),
           "el_protein": _grad_norm(el_weighted, prot), "el_term": _grad_norm(el_weighted, term)}
    for tower in ("protein", "term"):
        b, e = out[f"bce_{tower}"], out[f"el_{tower}"]
        out[f"ratio_{tower}"] = (e / b) if (e is not None and b) else None
    return out


def configure_determinism(seed: int) -> None:
    import torch
    from go_function.runtime import seed_everything
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def validation_bce(model, x: FeatureStore, y: RowStore, keep_cols: np.ndarray,
                   batch: int = EVAL_BATCH) -> Optional[float]:
    """Target-free validation BCE: mean over every validation row and every column
    not in the validation mask (design §5.3)."""
    import torch
    import torch.nn.functional as F
    if x.shape[0] == 0 or len(keep_cols) == 0:
        return None
    cols = torch.as_tensor(np.asarray(keep_cols, dtype=np.int64), device=x.device)
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for a in range(0, x.shape[0], batch):
            idx = np.arange(a, min(a + batch, x.shape[0]))
            z = model(x.rows(idx))[:, cols]
            t = y.rows(idx)[:, cols].to(z.dtype)
            total += float(F.binary_cross_entropy_with_logits(z, t, reduction="sum"))
            count += z.numel()
    return total / count


# ------------------------------------------------------------------ training (§7.2)
def build_run_model(spec: RunSpec, study: Mapping[str, Any], data: RunData, device: str):
    from go_function.model import build_model
    model = build_model(spec.variant, data.x_train.shape[1], len(data.vocab_ids), data.nf,
                        hidden_dim=int(study["dim"]), norm=study["norm"],
                        dropout=float(study["dropout"]), margin=float(study["margin"]),
                        nf4_radius=spec.nf4_radius, text=data.text)
    return model.to(device)


def train(spec: RunSpec, study: Mapping[str, Any], data: RunData, device: str, out_dir: Path,
          log) -> Tuple[Any, Dict[str, Any]]:
    """Train; returns (model with the selected weights, summary). Writes
    ``epochs.jsonl`` line by line. A non-finite loss or gradient raises."""
    import torch
    from torch.optim.lr_scheduler import MultiStepLR
    from go_function.model import masked_bce
    configure_determinism(spec.seed)
    model = build_run_model(spec, study, data, device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    sched = MultiStepLR(opt, milestones=list(spec.milestones), gamma=spec.gamma)
    gen = torch.Generator().manual_seed(spec.seed)
    xs = FeatureStore(data.x_train, device)
    ys = RowStore(data.y_train, device)
    ms = loss_mask(data.train_mask, data.y_train.shape, device)
    xv = FeatureStore(data.x_valid, device)
    yv = RowStore(data.y_valid, device)
    n = data.y_train.shape[0]
    if xs.shape[0] != n:
        raise FatalError(f"{xs.shape[0]} feature rows for {n} label rows")
    lam = spec.lambda_el if spec.is_el else None
    n_steps = int(math.ceil(n / spec.batch_size))
    if spec.max_train_batches is not None:
        n_steps = min(n_steps, spec.max_train_batches)
    params = list(model.parameters())
    best = {"valid_bce": math.inf, "epoch": None, "state": None}
    epochs_path = out_dir / "epochs.jsonl"
    summary: Dict[str, Any] = {"n_steps_per_epoch": n_steps, "mask_tensor": ms is not None,
                               "n_masked_entries": int(len(data.train_mask))}
    with open(epochs_path, "w") as fh:
        for epoch in range(spec.epochs):
            t0 = time.time()
            model.train()
            perm = torch.randperm(n, generator=gen).numpy()
            bce_sum, el_sums, el_total_sum, gn = 0.0, {f: 0.0 for f in ("nf1", "nf2", "nf3", "nf4")}, 0.0, []
            lr = float(opt.param_groups[0]["lr"])
            for b in range(n_steps):
                rows = perm[b * spec.batch_size:(b + 1) * spec.batch_size]
                y, masked = batch_targets(ys, ms, rows)
                logits = model(xs.rows(rows))
                bce = masked_bce(logits, y, masked)
                total = bce
                el_w = None
                if lam is not None:
                    el, parts = model.el_loss()
                    el_w = lam * el
                    total = bce + el_w
                    for f, v in parts.items():
                        el_sums[f] += float(v.detach())
                    el_total_sum += float(el.detach())
                if not bool(torch.isfinite(total.detach())):
                    raise NonFiniteLoss(f"non-finite loss at epoch {epoch} batch {b}")
                if b % GRAD_EVERY == 0:
                    gn.append({"batch": b, **grad_norms(model, bce, el_w)})
                opt.zero_grad()
                total.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
                if not bool(torch.isfinite(gnorm)):
                    raise NonFiniteLoss(f"non-finite gradient at epoch {epoch} batch {b}")
                opt.step()
                bce_sum += float(bce.detach())
            vb = validation_bce(model, xv, yv, data.valid_keep_cols)
            rec = {"epoch": epoch, "lr": lr, "n_batches": n_steps,
                   "train_bce": bce_sum / n_steps,
                   "el": ({f: v / n_steps for f, v in el_sums.items()} if lam is not None else None),
                   "el_total": el_total_sum / n_steps if lam is not None else None,
                   "valid_bce_target_free": vb, "seconds": round(time.time() - t0, 3),
                   "grad_norms": gn}
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            log.info("epoch %d: train BCE %.5f%s, valid BCE %s, %.1f s", epoch, rec["train_bce"],
                     f", EL {rec['el_total']:.5f}" if lam is not None else "",
                     f"{vb:.5f}" if vb is not None else "n/a", rec["seconds"])
            if spec.selection == "best_valid_loss" and vb is not None and vb < best["valid_bce"]:
                best = {"valid_bce": vb, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            sched.step()
    if spec.selection == "best_valid_loss":
        if best["state"] is None:
            raise FatalError("best_valid_loss selection without a validation loss")
        model.load_state_dict(best["state"])
        summary.update({"selected_epoch": best["epoch"], "selected_valid_bce": best["valid_bce"]})
    else:
        summary["selected_epoch"] = spec.epochs - 1
    return model, summary


# ------------------------------------------------------------------ evaluation (§7.3, §8)
def eval_logits(model, x, device: str, batch: int = EVAL_BATCH) -> np.ndarray:
    import torch
    xs = FeatureStore(x, device)
    out = []
    model.eval()
    with torch.no_grad():
        for a in range(0, xs.shape[0], batch):
            out.append(model(xs.rows(np.arange(a, min(a + batch, xs.shape[0])))).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)


def score_columns(spec: RunSpec, targets: Sequence[TargetCols]) -> np.ndarray:
    """t, Pa(t) and siblings of every target; X(t) and Desc(t) only for V4 at k = 0."""
    cols = set()
    channels = spec.variant == "V4" and spec.k == 0
    for t in targets:
        cols.add(t.column)
        cols.update(t.Pa)
        cols.update(t.siblings)
        if channels:
            cols.update(t.X)
            cols.update(t.Desc)
    return np.asarray(sorted(cols), dtype=np.int64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-z.astype(np.float64)))).astype(np.float32)


def evaluate_run(spec: RunSpec, data: RunData, logits: np.ndarray, device: str
                 ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """eval.json content and scores.npz arrays."""
    if spec.mode == "pilot":
        return {"schema": 1, "mode": spec.mode, "run": spec.run_name,
                "skipped": PILOT_EVAL_SKIPPED}, {}
    out: Dict[str, Any] = {"schema": 1, "mode": spec.mode, "run": spec.run_name,
                           "n_eval_rows": int(logits.shape[0])}
    if spec.mode == "gate":
        g = data.gate
        if g is None:
            raise FatalError("gate run without DGZ-faithful evaluation inputs")
        probs = _sigmoid(logits)
        if len(g["src"]):
            probs = E.propagate_max(probs, g["src"], g["dst"], device=device).cpu().numpy()
        pm = E.dgz_protein_metrics(probs, g["terms"], g["test_prop"], g["go_set"], g["ic"],
                                   device=device)
        cm = E.dgz_class_auc(probs, g["terms"], g["test_prop"])
        out["dgz_faithful"] = {**pm, "auc": cm["auc"], "n_auc_terms": cm["n_terms"]}
        return out, {}
    if data.y_eval is None:
        raise FatalError("evaluation labels are missing")
    y = sp.csr_matrix(data.y_eval)
    per_t = {}
    for t in data.targets:
        col = y[:, t.column].toarray().ravel()
        per_t[t.id] = E.class_metrics(logits[:, t.column], col)
    out["targets"] = per_t
    if data.collateral is not None:
        c = data.collateral
        cols = c["cols"]
        pc = E.protein_centric(_sigmoid(logits[:, cols]), y[:, cols], c["ic"], device=device,
                               propagate=(c["src"], c["dst"]))
        out["collateral"] = {k: pc[k] for k in ("fmax", "tmax", "smin", "aupr_micro",
                                                "aupr_protein", "n_eval", "n_skipped_no_label",
                                                "n_terms")}
    cols = score_columns(spec, data.targets)
    scores = {"logits": logits[:, cols].astype(np.float32), "columns": cols,
              "column_ids": np.asarray([data.vocab_ids[c] for c in cols], dtype=str)
              if len(cols) else np.zeros(0, dtype="<U1"),
              "eval_rows": np.asarray(data.eval_rows if data.eval_rows is not None
                                      else np.arange(logits.shape[0]), dtype=np.int64)}
    return out, scores


# ------------------------------------------------------------------ run driver
def environment(device: str) -> Dict[str, Any]:
    import torch
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "device": device, "cuda": torch.version.cuda,
           "gpu": None, "platform": platform.platform()}
    if device == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def keep_checkpoint(spec: RunSpec) -> bool:
    if spec.mode in ("gate", "pilot"):
        return True
    return spec.mode == "study" and spec.pack is None and spec.seed == 13


def _scrub(text: str) -> str:
    text = text.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")
    return re.sub(r"/Users/[^\s'\"]*", "<path>", text)


def _failure(stage: str, e: BaseException) -> Dict[str, Any]:
    frames = []
    for fr in traceback.extract_tb(e.__traceback__)[-8:]:
        p = Path(fr.filename)
        try:
            name = repo_rel(p) if p.is_absolute() and p.resolve().is_relative_to(ROOT) else p.name
        except ValueError:
            name = p.name
        frames.append(f"{name}:{fr.lineno} in {fr.name}")
    return {"stage": stage, "error": type(e).__name__, "message": _scrub(str(e))[:2000],
            "traceback": frames}


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def exit_code_for(e: BaseException) -> int:
    return getattr(e, "exit_code", EXIT_FATAL) if isinstance(e, (FatalError, ProtocolStop, Refused)) \
        else EXIT_FATAL


def resources(device: str) -> Dict[str, Any]:
    """Peak memory of this process (for the G9 smoke report)."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(rss if sys.platform == "darwin" else rss * 1024)   # macOS bytes, Linux KiB
    out: Dict[str, Any] = {"max_rss_bytes": rss, "cuda_max_memory_allocated": None}
    if str(device).startswith("cuda"):
        import torch
        out["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
    return out


def _finish(spec: RunSpec, study: Mapping[str, Any], data: RunData, logits: np.ndarray,
            device: str, out_dir: Path, summary: Dict[str, Any], walls: Dict[str, float],
            t0: float, t2: float, ckpt_rel: Optional[str]) -> None:
    """eval.json -> scores.npz -> run.json -> DONE.json; eval_inputs.npz removed.
    When preflight checked the code pin, the check is repeated here, after the
    model and feature modules were imported, before run.json is written."""
    if "code" in data.info:
        data.info["code_at_finish"] = check_import_coverage(spec.mode)
    ev, scores = evaluate_run(spec, data, logits, device)
    eb = canonical_json(ev)
    _write(out_dir / "eval.json", eb)
    files = {"eval.json": sha256_bytes(eb)}
    if scores:
        sb = npz_bytes(scores)
        _write(out_dir / "scores.npz", sb)
        files["scores.npz"] = sha256_bytes(sb)
    walls["eval"] = round(time.time() - t2, 3)
    walls["total"] = round(time.time() - t0, 3)
    rj = {"schema": 1, "spec": spec.to_json(), "run_name": spec.run_name,
          "condition": spec.condition_dir, "inputs": data.hashes, "info": data.info,
          "train": summary, "outputs": files, "checkpoint": ckpt_rel,
          "environment": environment(device), "wall_seconds": walls,
          "resources": resources(device),
          "study_sha256": sha256_bytes(canonical_json(study))}
    rb = canonical_json(rj)
    _write(out_dir / "run.json", rb)
    if (out_dir / "eval_inputs.npz").exists():
        (out_dir / "eval_inputs.npz").unlink()
    _write(out_dir / "DONE.json", canonical_json({"run_json_sha256": sha256_bytes(rb),
                                                  **{f"{k}_sha256": v for k, v in files.items()}}))


def rescore(spec_raw: Mapping[str, Any], device: str, *, log,
            loader: Optional[Callable[..., RunData]] = None,
            study: Optional[Mapping[str, Any]] = None, out_dir: Optional[Path] = None) -> int:
    """Re-run only the evaluation of a run whose FAILED.json has stage eval, from
    its ``eval_inputs.npz`` (or a kept ``checkpoint.pt``). Preflight runs again;
    on success FAILED.json becomes FAILED.rescored.json and DONE.json is written."""
    study = dict(study) if study is not None else load_study()
    spec = validate_spec(spec_raw, study)
    out_dir = Path(out_dir) if out_dir is not None else repo_path(spec.out_dir)
    assert_writable_target(out_dir)
    fp = out_dir / "FAILED.json"
    failed = json.loads(fp.read_text()) if fp.exists() else None
    if (out_dir / "DONE.json").exists() or failed is None or failed.get("stage") != "eval":
        raise Refused(f"{spec.out_dir}: rescore needs FAILED.json with stage eval")
    inputs, ckpt = out_dir / "eval_inputs.npz", out_dir / "checkpoint.pt"
    if not inputs.exists() and not ckpt.exists():
        raise Refused(f"{spec.out_dir}: neither eval_inputs.npz nor checkpoint.pt is kept")
    t0 = time.time()
    check_study_ready(study)
    data = (loader or load_run_data)(spec, study, device, log)
    walls: Dict[str, float] = {"preflight": round(time.time() - t0, 3)}
    t2 = time.time()
    if spec.mode == "pilot":
        logits = None                       # pilots are never evaluated (protocol §6.4)
    elif inputs.exists():
        with np.load(inputs, allow_pickle=False) as z:
            logits = z["logits"]
    else:
        import torch
        model = build_run_model(spec, study, data, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logits = eval_logits(model, data.x_eval, device)
    n_ep = sum(1 for x in (out_dir / "epochs.jsonl").read_text().splitlines() if x.strip()) \
        if (out_dir / "epochs.jsonl").exists() else 0
    summary = {"rescored": True, "n_epochs_logged": n_ep, "selected_epoch": spec.epochs - 1}
    ckpt_rel = f"{spec.out_dir}/checkpoint.pt" if ckpt.exists() else None
    _finish(spec, study, data, logits, device, out_dir, summary, walls, t0, t2, ckpt_rel)
    fp.replace(out_dir / "FAILED.rescored.json")
    log.info("%s rescored in %.1f s", spec.run_name, walls["total"])
    return EXIT_OK


def run(spec_raw: Mapping[str, Any], device: str, *, log,
        loader: Optional[Callable[..., RunData]] = None,
        study: Optional[Mapping[str, Any]] = None, out_dir: Optional[Path] = None) -> int:
    """Execute one validated run; ``loader(spec, study, device, log)`` returns RunData
    (default: ``load_run_data``, which performs the preflight). ``out_dir`` overrides
    the spec's directory (tests only); written paths stay the spec's repo-relative ones."""
    study = dict(study) if study is not None else load_study()
    spec = validate_spec(spec_raw, study)       # schema errors: exit 2, nothing written
    out_dir = Path(out_dir) if out_dir is not None else repo_path(spec.out_dir)
    assert_writable_target(out_dir)
    if (out_dir / "DONE.json").exists():
        raise Refused(f"{spec.out_dir} already has DONE.json")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise Refused(f"{spec.out_dir} is not empty (the sweep renames failed attempts)")
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = "preflight"
    walls: Dict[str, float] = {}
    t0 = time.time()
    try:
        check_study_ready(study)
        data = (loader or load_run_data)(spec, study, device, log)
        walls["preflight"] = round(time.time() - t0, 3)
        stage = "train"
        t1 = time.time()
        model, summary = train(spec, study, data, device, out_dir, log)
        walls["train"] = round(time.time() - t1, 3)
        stage = "eval"
        t2 = time.time()
        logits = None
        if spec.mode != "pilot":            # pilots: no test forward pass (protocol §6.4)
            logits = eval_logits(model, data.x_eval, device)
            np.savez(out_dir / "eval_inputs.npz", logits=logits)
        ckpt_rel = None
        if keep_checkpoint(spec):
            import torch
            torch.save(model.state_dict(), out_dir / "checkpoint.pt")
            ckpt_rel = f"{spec.out_dir}/checkpoint.pt"
        _finish(spec, study, data, logits, device, out_dir, summary, walls, t0, t2, ckpt_rel)
        log.info("%s done in %.1f s", spec.run_name, walls["total"])
        return EXIT_OK
    except BaseException as e:  # noqa: BLE001 - every failure leaves FAILED.json
        _write(out_dir / "FAILED.json", canonical_json(_failure(stage, e)))
        log.error("stage %s failed: %s", stage, _scrub(str(e)))
        if isinstance(e, KeyboardInterrupt):
            raise
        return exit_code_for(e)


# ------------------------------------------------------------------ real-data loader + preflight (§7.4)
PRE = "preprocess/go_function"
DGZ = "dataset/go_function/dgz"


def _pins() -> Dict[str, Any]:
    from go_function.pins import PINS_PATH
    if not PINS_PATH.exists():
        raise Refused("results/go_function/protocol_pins.json is missing")
    return json.loads(PINS_PATH.read_text())["pins"]


def _pinned(pins: Mapping[str, Any], key: str, name: str):
    if key not in pins or name not in pins[key]:
        raise Refused(f"pin {key}/{name} is missing")
    return pins[key][name]


def _verified_json(rel: str, key: str, name: str, pins, hashes: Dict[str, Any]) -> Dict[str, Any]:
    b = repo_path(rel).read_bytes()
    sha = sha256_bytes(b)
    want = _pinned(pins, key, name)
    want = want["sha256"] if isinstance(want, dict) else want
    if sha != want:
        raise FatalError(f"{rel}: sha256 {sha} differs from pin {key}/{name} {want}")
    hashes[rel] = sha
    return json.loads(b)


def _verified_file(rel: str, want: str, hashes: Dict[str, Any]) -> Path:
    p = repo_path(rel)
    if not p.exists():
        raise FatalError(f"missing input {rel}")
    sha = sha256_file(p)
    if sha != want:
        raise FatalError(f"{rel}: sha256 {sha} differs from its manifest ({want})")
    hashes[rel] = sha
    return p


def eval_target_ids(spec: RunSpec, table: Mapping[str, Any], tids: Sequence[str]) -> List[str]:
    """Targets a run scores: in the MF noIPI block, targets dropped by the noIPI rule
    (protocol §6.2, ``noipi.kept`` false in the target table) are not scored."""
    if spec.regime != "TPnoIPI":
        return list(tids)
    return [t for t in tids if (table["targets"][t].get("noipi") or {}).get("kept", True)]


def load_run_data(spec: RunSpec, study: Mapping[str, Any], device: str, log) -> RunData:
    """Preflight (design §7.4) and data assembly for a run on the real study files."""
    from go_function import labels as L
    from go_function import normforms as NF
    from go_function import withholding as WH
    from go_function.ontology import DgzRule, ancestors, parents, parse_obo
    from go_function.pins import csr_sha256, load_csr, verify_inputs
    a = spec.aspect
    hashes: Dict[str, Any] = {}
    info: Dict[str, Any] = {"code": check_import_coverage(spec.mode, log=log)}
    pins = _pins()
    hashes.update(verify_inputs([f"{DGZ}/go.obo", f"{DGZ}/go.norm"]))
    if hashes[f"{DGZ}/go.norm"] != _pinned(pins, "G1-norm", "go.norm")["sha256"]:
        raise FatalError("go.norm differs from pin G1-norm")

    # labels (condition regime, TP/DGZ for assertion 9, evaluation regime)
    def labels(regime):
        man = _verified_json(f"{PRE}/{a}/labels_{regime}.json", "G1-labels",
                             f"{a}/labels_{regime}.json", pins, hashes)
        _verified_file(man["npz"], man["npz_sha256"], hashes)
        lab = L.load_labels(repo_path(man["npz"]))
        for split in L.SPLITS:
            if csr_sha256(lab[split]) != man["splits"][split]["labels_sha256"]:
                raise FatalError(f"labels_{regime} {split} differs from its manifest")
        return lab, man

    ont = parse_obo(repo_path(f"{DGZ}/go.obo"))
    anc = ancestors(ont)
    need_dgz = spec.regime == "DGZ" or spec.mode == "gate"
    dgz = DgzRule(repo_path(f"{DGZ}/go.obo")) if need_dgz else None
    labs, mans = {}, {}
    for r in sorted({spec.regime, "TP", "DGZ"} | ({"TPnoIPI"} if spec.regime == "TPnoIPI" else set())):
        labs[r], mans[r] = labels(r)
    vocab = L.build_vocab(labs[spec.regime]["vocab"], ont)
    for r in labs:
        if labs[r]["vocab"] != list(vocab.ids):
            raise FatalError(f"labels_{r} vocabulary differs from labels_{spec.regime}")
    terms_rel = f"{DGZ}/{a}/terms.pkl"
    hashes.update(verify_inputs([terms_rel]))
    if L.read_terms(repo_path(terms_rel)) != list(vocab.ids):
        raise FatalError("label vocabulary differs from terms.pkl (the model's column list)")

    # target table, condition, validation mask
    table = _verified_json(f"{PRE}/{a}/targets/target_table.json", "G6",
                           f"{a}/targets/target_table.json", pins, hashes)
    tt_sha = hashes[f"{PRE}/{a}/targets/target_table.json"]
    index = _verified_json(f"{PRE}/{a}/conditions/index.json", "G6",
                           f"{a}/conditions/index.json", pins, hashes)
    wspec = WH.Spec(a, spec.regime, spec.condition_rule, spec.pack,
                    "all" if spec.pack is None else spec.k, spec.retention_draw)
    if wspec.name not in index["conditions"]:
        raise FatalError(f"condition {wspec.name} is not in the pinned condition index")
    cdir = repo_path(spec.condition_dir)

    def cond_manifest(name: str, d: Path) -> Dict[str, Any]:
        mb = (d / "withholding_manifest.json").read_bytes()
        if sha256_bytes(mb) != index["conditions"][name]:
            raise FatalError(f"{name}: manifest differs from the pinned condition index")
        man = json.loads(mb)
        for fname, key in (("train_labels.npz", "train_labels_npz_sha256"),
                           ("train_mask.npz", "train_mask_npz_sha256")):
            _verified_file(repo_rel(d / fname), man[key], hashes)
        hashes[repo_rel(d / "withholding_manifest.json")] = sha256_bytes(mb)
        return man

    cman = cond_manifest(wspec.name, cdir)
    if cman["target_table_sha256"] != tt_sha:
        raise FatalError("condition was built from a different target table")
    vrel = f"{PRE}/{a}/conditions/{spec.regime}/validation"
    vman = _verified_json(f"{vrel}/validation_manifest.json", "G6",
                          f"{a}/conditions/{spec.regime}/validation/validation_manifest.json",
                          pins, hashes)
    _verified_file(f"{vrel}/validation_mask.npz", vman["npz_sha256"], hashes)
    if cman["validation_mask_sha256"] != vman["npz_sha256"]:
        raise FatalError("condition and validation mask disagree")
    with np.load(repo_path(f"{vrel}/validation_mask.npz"), allow_pickle=False) as z:
        vmask_cols = z["columns"].astype(np.int64)

    # assertions 1-10 on the condition files
    def regime_data(r):
        lab, man = labs[r], mans[r]
        return WH.RegimeData(regime=r, vocab=vocab, train=lab["train"], direct=lab["direct_train"],
                             extra=lab["direct_extra_train"], proteins=lab["proteins_train"],
                             valid_sha256=man["splits"]["valid"]["labels_sha256"],
                             test_sha256=man["splits"]["test"]["labels_sha256"],
                             base_manifest_sha256=hashes[f"{PRE}/{a}/labels_{r}.json"])
    rd = regime_data(spec.regime)
    prop = WH.Propagator(spec.regime, vocab, a, anc, dgz if spec.regime == "DGZ" else None)
    infos = WH.pack_target_infos(table, spec.pack) if spec.pack else []
    all_t = WH.all_target_ids(table)
    tcols = {t: int(table["targets"][t]["column"]) for t in all_t}
    P_tp = {t: WH.column_positive_rows(labs["TP"]["train"], tcols[t]) for t in all_t}
    P_dgz = {t: WH.column_positive_rows(labs["DGZ"]["train"], tcols[t]) for t in all_t}
    ci = WH.check_input_from_files(cdir, wspec, rd, prop, infos, P_tp=P_tp, P_dgz=P_dgz)
    w2_sha = None
    if spec.rule == "W2drop":
        wname = f"DGZ/W2/{spec.pack}_k0"
        wdir = repo_path(f"{PRE}/{a}/conditions/{wname}")
        cond_manifest(wname, wdir)
        w2_sha = csr_sha256(WH.load_condition_files(wdir)[0])
    done = WH.run_assertions(ci, vocab, base_labels_sha=csr_sha256(rd.train), w2_labels_sha=w2_sha,
                             cond_manifest=cman, base_manifest=mans[spec.regime])
    info["assertions_rerun"] = done
    log.info("preflight: %s assertions %s passed", wspec.name, done)
    y_train, mask = ci.labels, np.asarray(ci.mask, dtype=np.int32).reshape(-1, 2)

    # features
    lab0 = labs[spec.regime]
    if spec.feature == "interpro":
        fman = _verified_json(f"{PRE}/{a}/features_interpro.json", "G5",
                              f"{a}/features_interpro.json", pins, hashes)
        _verified_file(fman["npz"], fman["npz_sha256"], hashes)
        with np.load(repo_path(fman["npz"]), allow_pickle=False) as z:
            X = {s: load_csr(z, s) for s in L.SPLITS}
            for s in L.SPLITS:
                if [str(p) for p in z[f"proteins_{s}"]] != lab0[f"proteins_{s}"]:
                    raise FatalError(f"InterPro {s} rows differ from the label rows")
    else:
        from go_function.features import esm_view
        eman = _verified_json(f"{PRE}/esm2_650m/manifest.json", "G5", "esm2_650m/manifest.json",
                              pins, hashes)
        ep = _verified_file(f"{PRE}/esm2_650m/embeddings.npy", eman["embeddings_sha256"], hashes)
        pp = _verified_file(f"{PRE}/esm2_650m/proteins.txt", eman["proteins_sha256"], hashes)
        emb = np.load(ep, allow_pickle=False)
        names = pp.read_text().splitlines()
        X = {s: esm_view(emb, names, lab0[f"proteins_{s}"]) for s in L.SPLITS}
        del emb

    # normal forms and term text
    nf = None
    if spec.is_el:
        full = NF.load_normal_forms(repo_path(f"{DGZ}/go.norm"), dict(vocab.index))
        nf = NF.select(full, spec.variant)
    text = None
    if spec.variant in ("V5", "V6"):
        tman = _verified_json(f"{PRE}/term_text_mpnet.json", "G5", "term_text_mpnet.json",
                              pins, hashes)
        tp = _verified_file(f"{PRE}/term_text_mpnet.npy", tman["embeddings_sha256"], hashes)
        ip = _verified_file(f"{PRE}/term_ids.txt", tman["term_ids_sha256"], hashes)
        vec = np.load(tp, allow_pickle=False)
        row = {g: i for i, g in enumerate(ip.read_text().splitlines())}
        ids = list(vocab.ids) if spec.variant == "V5" else nf.class_ids()
        from go_function.constants import OWL_NOTHING
        miss = [g for g in ids if g != OWL_NOTHING and g not in row]
        if miss:
            raise FatalError(f"{len(miss)} classes have no term text; first {miss[:10]}")
        text = np.stack([vec[row[g]] if g != OWL_NOTHING else np.zeros(vec.shape[1], np.float32)
                         for g in ids]).astype(np.float32)

    # validation (target-free) and evaluation
    keep_cols = np.setdiff1d(np.arange(len(vocab)), vmask_cols)
    data = RunData(x_train=X["train"], x_valid=X["valid"], x_eval=None, y_train=y_train,
                   train_mask=mask, y_valid=lab0["valid"], valid_keep_cols=keep_cols,
                   vocab_ids=list(vocab.ids), nf=nf, text=text, hashes=hashes, info=info)
    if spec.mode == "gate":
        import pandas as pd
        from go_function.model import dgz_propagation_pairs
        rels = [f"{DGZ}/{a}/{s}_data.pkl" for s in ("train", "valid", "test")]
        hashes.update(verify_inputs(rels))
        props = {s: [list(x) for x in pd.read_pickle(repo_path(r))["prop_annotations"].tolist()]
                 for s, r in zip(("train", "valid", "test"), rels)}
        src, dst = dgz_propagation_pairs(list(vocab.ids), dgz)
        data.gate = {"terms": list(vocab.ids), "test_prop": props["test"],
                     "go_set": E.dgz_namespace_terms(dgz, a),
                     "ic": E.dgz_ic(dgz, props["train"] + props["valid"] + props["test"]),
                     "src": src, "dst": dst}
        data.x_eval = X["test"]
        return data
    if spec.mode == "pilot":
        # protocol §6.4: pilots never read test data; no evaluation labels, no
        # test features, no target columns, no collateral set (amendment A4)
        info["eval_skipped"] = PILOT_EVAL_SKIPPED
        return data
    ev = E.load_eval_labels("TPnoIPI" if spec.regime == "TPnoIPI" else "TP", a)
    data.y_eval, data.eval_rows = ev["y"], np.asarray(ev["rows"], dtype=np.int64)
    data.x_eval = X["test"][data.eval_rows]
    tids = eval_target_ids(spec, table, [i.id for i in infos] if spec.pack else all_t)
    for t in tids:
        ch = table["targets"][t]["channels"]
        col = lambda gs: sorted(vocab.index[g] for g in gs)  # noqa: E731
        data.targets.append(TargetCols(id=t, column=tcols[t], Pa=col(ch["Pa"]),
                                       siblings=col(ch["siblings"]), X=col(ch["X"]),
                                       Desc=col(ch["Desc"])))
    # collateral term set (design §8.2): TP (TPnoIPI) labels, pack D*_TP excluded
    er = "TPnoIPI" if spec.regime == "TPnoIPI" else "TP"
    tprop = WH.Propagator(er, vocab, a, anc)
    excl = sorted({int(c) for i in infos for c in tprop.dstar_cols(i.id)})
    from go_function.constants import ROOTS
    T = E.collateral_terms(vocab, labs[er]["train"], excl, ROOTS[a])
    par = parents(ont)
    pcols = [() if p is None else tuple(sorted(vocab.index[q] for q in par[p] if q in vocab.index))
             for p in vocab.primary]
    ic = E.ic_from_labels(sp.vstack([labs[er]["train"], labs[er]["valid"]]), pcols)
    src, dst = E.tp_ancestor_pairs(list(T), vocab, anc)
    data.collateral = {"cols": T, "ic": ic[T], "src": src, "dst": dst}
    info["n_collateral_terms"] = int(len(T))
    return data
