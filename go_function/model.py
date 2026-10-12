"""Bilinear multi-label model with free / EL / text term towers (design §6; gap 6).

s(p, t) = sigmoid(<f(x_p), g(t)> + b_t). The protein tower f is DGZ's encoder;
the term tower g is chosen by the arm:

  V0            free rows of Linear(hidden, |V|) (= DGZ MLP), b_t = Linear bias
  V1..V4, V3c   EL n-balls: g(t) = e_t + r_hasFunc, b_t = |rho_t| (= DGZ DGELModel)
  V5            P m_t over frozen text vectors m_t, P = Linear(768, d, bias=False),
                free per-term bias
  V6            P m_t + r_hasFunc, b_t = |rho_t|; EL over the V4 axioms with ball
                centres go_norm(P m_c); owl:Nothing gets a free centre and radius

Ported from the public DeepGOZero source, read with web_fetch:

- https://raw.githubusercontent.com/bio-ontology-research-group/deepgozero/main/deepgozero.py
  (``Residual``, ``MLPBlock``, ``DGELModel``: ``forward``, ``predict_zero``,
  ``el_loss``, ``class_dist``, ``nf1_loss`` .. ``nf4_loss``; uniform init
  +-sqrt(1/embed_dim); ``rel_embed`` with one extra row for hasFunction; margin 0.1;
  ``total_loss = loss + el_loss``; the score propagation after prediction);
- https://raw.githubusercontent.com/bio-ontology-research-group/deepgozero/main/deepgopro.py
  (``DGPROModel``, the MLP that wrote ``mlp.th``: MLPBlock + Residual + Linear +
  Sigmoid, ``nn.LayerNorm`` in its MLPBlock).

Differences from DGZ's code (all intentional, design §6):

1. DGZ's ``deepgozero.py`` writes ``nn.BatchNorm(out_features)`` (not a torch
   class); the norm is a constructor argument here (``layernorm`` | ``batchnorm1d``),
   fixed by G3 from the released checkpoint keys. The attribute keeps DGZ's name
   ``layer_norm`` so checkpoint keys map one-to-one.
2. The model returns logits; the sigmoid is applied by the loss
   (``binary_cross_entropy_with_logits``) and by the evaluator.
3. ``nf4_radius="dgz"`` keeps DGZ's ``rc = |go_rad(data[:, 1])|`` (the relation id
   used as a class index, protocol D10); ``"class"`` uses ``data[:, 0]`` (G4
   diagnostic only). The index is the arm's ``nf4_rad`` column when given: V3c
   shares one relation vector but keeps V3's per-relation radius slots
   (amendment A5), so V3 and V3c read the same radius classes.
4. An empty normal form contributes 0 (DGZ's ``th.mean`` of an empty tensor is NaN).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from go_function.constants import ARMS, OWL_NOTHING
from go_function.runtime import FatalError

NORMS = ("layernorm", "batchnorm1d")
EL_ARMS = ("V1", "V2", "V3", "V3c", "V4", "V6")
TEXT_ARMS = ("V5", "V6")
NF4_RADIUS = ("dgz", "class")
TEXT_DIM = 768
FORMS = ("nf1", "nf2", "nf3", "nf4")


class CheckpointMismatch(FatalError):
    """A DGZ checkpoint does not load strictly (fatal for G3; G2b records it)."""

    def __init__(self, missing: Sequence[str], unexpected: Sequence[str],
                 shapes: Sequence[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]):
        self.missing, self.unexpected, self.shapes = list(missing), list(unexpected), list(shapes)
        parts = []
        if self.missing:
            parts.append(f"missing keys {self.missing}")
        if self.unexpected:
            parts.append(f"unexpected keys {self.unexpected}")
        if self.shapes:
            parts.append("shape mismatches " + ", ".join(
                f"{k}: checkpoint {tuple(a)} vs model {tuple(b)}" for k, a, b in self.shapes))
        super().__init__("checkpoint mismatch: " + "; ".join(parts))


# ------------------------------------------------------------------ protein tower (§6.1)
def make_norm(norm: str, n: int) -> nn.Module:
    if norm == "layernorm":
        return nn.LayerNorm(n)
    if norm == "batchnorm1d":
        return nn.BatchNorm1d(n)
    raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class MLPBlock(nn.Module):
    """Linear -> ReLU -> norm -> Dropout (DGZ ``MLPBlock``)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 norm: str = "layernorm", dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = nn.ReLU()
        self.layer_norm = make_norm(norm, out_features)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.layer_norm(self.activation(self.linear(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# ------------------------------------------------------------------ EL losses (§6.5)
def _zero(ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=ref.dtype, device=ref.device)


def el_losses(centre: torch.Tensor, radius: torch.Tensor, rel: torch.Tensor,
              go_norm: nn.Module, forms: Mapping[str, torch.Tensor], margin: float,
              nf4_radius: str = "dgz") -> Dict[str, torch.Tensor]:
    """DGZ ``nf1_loss`` .. ``nf4_loss`` over full class tables.

    ``centre`` (n_classes x d) and ``radius`` (n_classes x 1) are the raw class
    tables (go_embed / go_rad weights, or the V6 text projection); ``rel`` is the
    relation table (rel_embed weight). Lookups ``table[idx]`` equal DGZ's
    ``nn.Embedding`` calls. ``go_norm`` is applied per call, as DGZ does."""
    if nf4_radius not in NF4_RADIUS:
        raise ValueError(f"nf4_radius must be one of {NF4_RADIUS}")

    def emb(idx):
        return go_norm(centre[idx])

    def rad(idx):
        return torch.abs(radius[idx])

    def norm(x):
        return torch.linalg.norm(x, dim=1, keepdim=True)

    out: Dict[str, torch.Tensor] = {}
    d1 = forms.get("nf1")
    if d1 is None or len(d1) == 0:
        out["nf1"] = _zero(centre)
    else:   # class_dist: ||c - d|| + rc - rd
        dist = norm(emb(d1[:, 0]) - emb(d1[:, 1])) + rad(d1[:, 0]) - rad(d1[:, 1])
        out["nf1"] = torch.mean(torch.relu(dist - margin))
    d2 = forms.get("nf2")
    if d2 is None or len(d2) == 0:
        out["nf2"] = _zero(centre)
    else:   # C and D SubClassOf E
        c, d, e = emb(d2[:, 0]), emb(d2[:, 1]), emb(d2[:, 2])
        rc, rd = rad(d2[:, 0]), rad(d2[:, 1])
        sr = rc + rd
        out["nf2"] = torch.mean(torch.relu(norm(c - d) - sr - margin)
                                + torch.relu(norm(e - c) - rc - margin)
                                + torch.relu(norm(e - d) - rd - margin))
    d3 = forms.get("nf3")
    if d3 is None or len(d3) == 0:
        out["nf3"] = _zero(centre)
    else:   # R some C SubClassOf D
        rE = rel[d3[:, 0]]
        c, d = emb(d3[:, 1]), emb(d3[:, 2])
        rc, rd = rad(d3[:, 1]), rad(d3[:, 2])
        out["nf3"] = torch.mean(torch.relu(norm(c + rE - d) + rc - rd - margin))
    d4 = forms.get("nf4")
    if d4 is None or len(d4) == 0:
        out["nf4"] = _zero(centre)
    else:   # C SubClassOf R some D
        c = emb(d4[:, 0])
        rE = rel[d4[:, 1]]
        d = emb(d4[:, 2])
        if nf4_radius == "dgz":       # DGZ: relation id read as a class index (D10)
            r4 = forms.get("nf4_rad")
            rc = rad(d4[:, 1] if r4 is None else r4)
        else:
            rc = rad(d4[:, 0])
        rd = rad(d4[:, 2])
        out["nf4"] = torch.mean(torch.relu(norm(c - (d + rE)) - (rc + rd) - margin))
    return out


# ------------------------------------------------------------------ masked BCE (§6.5)
def masked_bce(logits: torch.Tensor, y: torch.Tensor,
               masked: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean BCE over unmasked entries. ``masked`` (bool, True = excluded) None takes
    the plain path (= DGZ's mean BCE); W1/W2 conditions never build a mask."""
    y = y.to(logits.dtype)
    if masked is None:
        return F.binary_cross_entropy_with_logits(logits, y)
    keep = (~masked).to(logits.dtype)
    n = keep.sum()
    if float(n) == 0.0:
        return _zero(logits)
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * keep).sum() / n


# ------------------------------------------------------------------ the model (§6.2)
class BilinearModel(nn.Module):
    """One code path for every arm. Parameter names follow DGZ (``net``,
    ``go_embed``, ``go_rad``, ``rel_embed``, ``go_norm``) so released checkpoints
    load with an identity key map (DGZ MLP: ``net.2`` -> ``fc``)."""

    def __init__(self, arm: str, n_in: int, n_terms: int, *, hidden_dim: int = 1024,
                 norm: str = "layernorm", dropout: float = 0.1, margin: float = 0.1,
                 n_zero: int = 0, n_rels: int = 0, nf4_radius: str = "dgz",
                 text: Optional[np.ndarray] = None, nothing_index: Optional[int] = None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if nf4_radius not in NF4_RADIUS:
            raise ValueError(f"nf4_radius must be one of {NF4_RADIUS}")
        self.arm, self.n_terms, self.n_zero, self.n_rels = arm, n_terms, n_zero, n_rels
        self.hidden_dim, self.margin, self.nf4_radius = hidden_dim, margin, nf4_radius
        self.is_el = arm in EL_ARMS
        self.net = nn.Sequential(MLPBlock(n_in, hidden_dim, norm=norm, dropout=dropout),
                                 Residual(MLPBlock(hidden_dim, hidden_dim, norm=norm,
                                                   dropout=dropout)))
        k = math.sqrt(1 / hidden_dim)
        n_classes = n_terms + n_zero
        self.n_classes = n_classes
        self.has_func = n_rels                      # index of the hasFunction row
        if arm == "V0":
            self.fc = nn.Linear(hidden_dim, n_terms)
        if self.is_el:
            if arm != "V6":
                self.go_embed = nn.Embedding(n_classes, hidden_dim)
                nn.init.uniform_(self.go_embed.weight, -k, k)
            self.go_norm = nn.BatchNorm1d(hidden_dim)
            self.go_rad = nn.Embedding(n_classes, 1)
            nn.init.uniform_(self.go_rad.weight, -k, k)
            self.rel_embed = nn.Embedding(n_rels + 1, hidden_dim)
            nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.nothing_index = None
        if arm in TEXT_ARMS:
            if text is None:
                raise ValueError(f"{arm} needs frozen text vectors")
            text = np.asarray(text, dtype=np.float32)
            want = n_terms if arm == "V5" else n_classes
            if text.shape != (want, TEXT_DIM):
                raise ValueError(f"{arm}: text matrix {text.shape}, expected ({want}, {TEXT_DIM})")
            self.register_buffer("text", torch.as_tensor(text), persistent=False)
            self.proj = nn.Linear(TEXT_DIM, hidden_dim, bias=False)
            if arm == "V5":
                self.term_bias = nn.Parameter(torch.zeros(n_terms))
            if arm == "V6" and nothing_index is not None:
                if not n_terms <= nothing_index < n_classes:
                    raise ValueError("owl:Nothing must be a zero class")
                self.nothing_index = int(nothing_index)
                self.nothing_embed = nn.Embedding(1, hidden_dim)
                nn.init.uniform_(self.nothing_embed.weight, -k, k)
                self.nothing_rad = nn.Embedding(1, 1)
                nn.init.uniform_(self.nothing_rad.weight, -k, k)
        self._forms: Dict[str, torch.Tensor] = {}

    # ---------------------------------------------------------------- towers
    def protein_parameters(self) -> List[nn.Parameter]:
        return list(self.net.parameters())

    def term_parameters(self) -> List[nn.Parameter]:
        prot = {id(p) for p in self.net.parameters()}
        return [p for p in self.parameters() if id(p) not in prot]

    def set_normal_forms(self, forms: Mapping[str, np.ndarray], device=None) -> None:
        """The arm's axiom arrays (``normforms.select`` output); EL arms only."""
        if not self.is_el:
            raise ValueError(f"{self.arm} has no axiom loss")
        dev = device if device is not None else next(self.parameters()).device
        cls_cols = {"nf1": (0, 1), "nf2": (0, 1, 2), "nf3": (1, 2), "nf4": (0, 2)}
        rel_col = {"nf3": 0, "nf4": 1}
        self._forms = {}
        for f in FORMS:
            if f not in forms:
                continue
            a = np.asarray(forms[f], dtype=np.int64)
            if len(a):
                if int(a[:, list(cls_cols[f])].max()) >= self.n_classes:
                    raise ValueError(f"{f} references a class id >= {self.n_classes}")
                if f in rel_col and int(a[:, rel_col[f]].max()) >= self.n_rels:
                    raise ValueError(f"{f} references a relation id >= {self.n_rels}")
            self._forms[f] = torch.as_tensor(a, device=dev)
        r4 = forms.get("nf4_rad")
        if r4 is not None:
            r4 = np.asarray(r4, dtype=np.int64).reshape(-1)
            if len(r4) != len(self._forms.get("nf4", ())):
                raise ValueError("nf4_rad must have one entry per NF4 axiom")
            if len(r4) and int(r4.max()) >= self.n_classes:
                raise ValueError("nf4_rad references a class id beyond the class table")
            self._forms["nf4_rad"] = torch.as_tensor(r4, device=dev)

    def _tables(self, rows: Optional[slice] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(centre table, radius table) of the EL arms, raw (before go_norm)."""
        if self.arm == "V6":
            text = self.text if rows is None else self.text[rows]
            centre = self.proj(text)
            radius = self.go_rad.weight if rows is None else self.go_rad.weight[rows]
            if self.nothing_index is not None and rows is None:
                idx = torch.tensor([self.nothing_index], device=centre.device)
                centre = centre.index_copy(0, idx, self.nothing_embed.weight)
                radius = radius.index_copy(0, idx, self.nothing_rad.weight)
            return centre, radius
        centre = self.go_embed.weight if rows is None else self.go_embed.weight[rows]
        radius = self.go_rad.weight if rows is None else self.go_rad.weight[rows]
        return centre, radius

    def term_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(g, b) for the |V| columns: g (|V| x d), b (|V|)."""
        V = slice(0, self.n_terms)
        if self.arm == "V0":
            return self.fc.weight, self.fc.bias
        if self.arm == "V5":
            return self.proj(self.text), self.term_bias
        centre, radius = self._tables(V)
        hf = self.rel_embed.weight[self.has_func]
        return centre + hf, torch.abs(radius).view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (batch x |V|)."""
        h = self.net(x)
        g, b = self.term_vectors()
        return torch.matmul(h, g.T) + b

    def logits_for_classes(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """DGZ ``predict_zero`` before the sigmoid: logits for class ids ``idx``
        (vocabulary or zero classes) of an EL arm."""
        if not self.is_el:
            raise ValueError(f"{self.arm} has no class table")
        h = self.net(x)
        centre, radius = self._tables()
        hf = self.rel_embed.weight[self.has_func]
        return torch.matmul(h, (centre[idx] + hf).T) + torch.abs(radius[idx]).view(1, -1)

    def el_loss(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """(sum over forms, per form) of DGZ's EL loss over every axiom of the arm."""
        if not self.is_el:
            raise ValueError(f"{self.arm} has no axiom loss")
        centre, radius = self._tables()
        if any(a.device != centre.device for a in self._forms.values()):
            self._forms = {f: a.to(centre.device) for f, a in self._forms.items()}
        parts = el_losses(centre, radius, self.rel_embed.weight, self.go_norm, self._forms,
                          self.margin, self.nf4_radius)
        total = parts["nf1"] + parts["nf3"] + parts["nf4"] + parts["nf2"]
        return total, parts


def n_relations(rel_index: Mapping[str, int]) -> int:
    """Relation vectors of an arm (V3c maps several IRIs to one vector)."""
    return len(set(rel_index.values()))


def build_model(arm: str, n_in: int, n_terms: int, nf=None, *, hidden_dim: int = 1024,
                norm: str = "layernorm", dropout: float = 0.1, margin: float = 0.1,
                nf4_radius: str = "dgz", text: Optional[np.ndarray] = None) -> BilinearModel:
    """Model of ``arm``; ``nf`` is the arm's ``normforms.select`` output (EL arms)."""
    n_zero = n_rels = 0
    nothing = None
    if arm in EL_ARMS:
        if nf is None:
            raise ValueError(f"{arm} needs its normal forms")
        if nf.n_vocab != n_terms:
            raise ValueError("normal forms and vocabulary disagree on |V|")
        n_zero = len(nf.class_index) - nf.n_vocab
        n_rels = n_relations(nf.rel_index)
        nothing = nf.class_index.get(OWL_NOTHING)
    m = BilinearModel(arm, n_in, n_terms, hidden_dim=hidden_dim, norm=norm, dropout=dropout,
                      margin=margin, n_zero=n_zero, n_rels=n_rels, nf4_radius=nf4_radius,
                      text=text, nothing_index=nothing if arm == "V6" else None)
    if arm in EL_ARMS:
        forms = {f: getattr(nf, f) for f in FORMS}
        if getattr(nf, "nf4_rad", None) is not None:
            forms["nf4_rad"] = nf.nf4_rad
        m.set_normal_forms(forms)
    return m


# ------------------------------------------------------------------ DGZ checkpoints (§6.6)
def detect_norm(state_dict: Mapping[str, torch.Tensor]) -> str:
    """LayerNorm if the protein-tower norm has weight/bias and no running statistics;
    BatchNorm1d if it has running_mean/running_var (design §6.1, G3)."""
    pre = "net.0.layer_norm."
    keys = {k[len(pre):] for k in state_dict if k.startswith(pre)}
    if {"running_mean", "running_var"} <= keys:
        return "batchnorm1d"
    if {"weight", "bias"} <= keys and not keys & {"running_mean", "running_var",
                                                    "num_batches_tracked"}:
        return "layernorm"
    raise CheckpointMismatch([pre + "weight", pre + "bias"], [], [])


def map_dgz_keys(state_dict: Mapping[str, torch.Tensor], kind: str) -> Dict[str, torch.Tensor]:
    """DGZ key -> our key. ``mlp`` (DGPROModel): ``net.2.*`` -> ``fc.*``; ``deepgozero``
    (DGELModel): identity."""
    if kind not in ("mlp", "deepgozero"):
        raise ValueError("kind must be 'mlp' or 'deepgozero'")
    out = {}
    for k, v in state_dict.items():
        if kind == "mlp" and k.startswith("net.2."):
            k = "fc." + k[len("net.2."):]
        out[k] = v
    return out


class DgzCompatModel:
    """Loads DGZ's released ``mlp.th`` / ``deepgozero.th`` / ``deepgozero_zero.th``
    into ``BilinearModel`` (V0 / V4) with ``strict=True``."""

    @staticmethod
    def check(model: nn.Module, mapped: Mapping[str, torch.Tensor]) -> None:
        want = model.state_dict()
        missing = sorted(set(want) - set(mapped))
        unexpected = sorted(set(mapped) - set(want))
        shapes = [(k, tuple(mapped[k].shape), tuple(want[k].shape))
                  for k in sorted(set(want) & set(mapped))
                  if tuple(mapped[k].shape) != tuple(want[k].shape)]
        if missing or unexpected or shapes:
            raise CheckpointMismatch(missing, unexpected, shapes)

    @classmethod
    def load(cls, path: Path, *, kind: str, n_in: int, n_terms: int, nf=None,
             strict: bool = True, device: str = "cpu", hidden_dim: Optional[int] = None,
             state_dict: Optional[Mapping[str, torch.Tensor]] = None) -> BilinearModel:
        """``hidden_dim`` None: read from ``net.0.linear.weight`` (DGZ: 1024)."""
        if not strict:
            raise ValueError("DGZ checkpoints are loaded strictly (design §6.6)")
        sd = state_dict if state_dict is not None else torch.load(
            path, map_location="cpu", weights_only=True)
        norm = detect_norm(sd)
        if hidden_dim is None:
            w = sd.get("net.0.linear.weight")
            if w is None:
                raise CheckpointMismatch(["net.0.linear.weight"], [], [])
            hidden_dim = int(w.shape[0])
        arm = "V0" if kind == "mlp" else "V4"
        model = build_model(arm, n_in, n_terms, nf, hidden_dim=hidden_dim, norm=norm)
        mapped = map_dgz_keys(sd, kind)
        cls.check(model, mapped)
        model.load_state_dict(mapped, strict=True)
        model.norm_type = norm  # type: ignore[attr-defined]
        return model.to(device).eval()


# ------------------------------------------------------------------ DGZ score propagation
def dgz_propagation_pairs(terms: Sequence[str], dgz_rule) -> Tuple[np.ndarray, np.ndarray]:
    """(src, dst) column pairs with terms[dst] in DgzRule.closure(terms[src]),
    src != dst. With ``evaluate.propagate_max`` this is DGZ's post-prediction loop
    (pred(c) = max over columns d whose DGZ closure contains c, d = c included)."""
    index = {g: i for i, g in enumerate(terms)}
    src: List[int] = []
    dst: List[int] = []
    for j, g in enumerate(terms):
        for a in dgz_rule.closure(g):
            i = index.get(a)
            if i is not None and i != j:
                src.append(j)
                dst.append(i)
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def predict_probs(model: BilinearModel, features, device: str = "cpu",
                  batch: int = 512, idx: Optional[torch.Tensor] = None,
                  sigmoid: bool = True) -> np.ndarray:
    """sigmoid scores (float32; raw logits with ``sigmoid=False``) for every row of
    ``features`` (dense array or CSR), in eval mode, in row batches (no dense copy
    of a CSR matrix)."""
    import scipy.sparse as sp
    model.eval()
    out = []
    n = features.shape[0]
    with torch.no_grad():
        for a in range(0, n, batch):
            blk = features[a:a + batch]
            blk = blk.toarray() if sp.issparse(blk) else np.asarray(blk)
            x = torch.as_tensor(np.asarray(blk, dtype=np.float32), device=device)
            z = model(x) if idx is None else model.logits_for_classes(x, idx.to(device))
            z = torch.sigmoid(z) if sigmoid else z
            out.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)
