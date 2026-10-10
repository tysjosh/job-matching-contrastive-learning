"""Loss-only GO relation table and the pure L_GO term (withholding study, design 5.2).

``GoRelationTable`` holds a dense simGIC (biological process) matrix over a fixed
protein set. It is deliberately NOT a matcher: it exposes no
``ontology_set_similarity``, ``terms_for_gene`` or ``branch_distance``, and keeps
no reference to the source ``GoIndex``, so it cannot be handed to the batch
processor or the negative selector to open GO-tiered negative selection. It is
attached to the loss engine only.

``go_relation_loss`` is the unweighted relation-preservation term over the batch's
anchor and positive proteins (negatives excluded, so the pair set is identical
across the four study cells for a given condition and seed):

    L_GO = mean_{i<j} (cos(e_i, e_j) - simGIC(i, j))^2
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from go_ppi_domain.go_ontology import GoIndex


class GoRelationTable:
    """Dense simGIC_BP over a fixed protein set; loss-only, not a matcher."""

    def __init__(self, ids: List[str], annotated: np.ndarray, sim: np.ndarray,
                 source_meta: dict) -> None:
        self.ids = list(ids)
        self.row = {pid: i for i, pid in enumerate(self.ids)}
        self.annotated = np.asarray(annotated, dtype=bool)
        self.sim = np.asarray(sim, dtype=np.float32)
        self.source_meta = dict(source_meta)

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, pid: object) -> bool:
        return pid in self.row

    # ------------------------------------------------------------------ build
    @classmethod
    def from_train_file(cls, index_path: Path, train_file: Path) -> "GoRelationTable":
        """Table over the sorted proteins of ``train_file`` (either slot).

        The index is read with ``GoIndex.load`` only, never ``build_or_load``, so
        a missing or wrong pickle fails instead of silently rebuilding.
        """
        proteins = set()
        with open(train_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                proteins.add(str(rec["resume"]["protein_id"]))
                proteins.add(str(rec["job"]["partner_id"]))
        return cls.from_index(GoIndex.load(Path(index_path)), proteins)

    @classmethod
    def from_index(cls, index: GoIndex, proteins: Iterable[str]) -> "GoRelationTable":
        """Vectorised simGIC over the proteins' ancestor closures.

        X is the protein x term binary matrix over closures and w the IC vector:
        inter = X diag(w) X^T, s_i = sum of w over closure(i),
        union = s_i + s_j - inter, sim = clip(inter / union, 0, 1), 0 where union = 0.
        """
        from scipy import sparse

        ids = sorted(set(str(p) for p in proteins))
        closures = [index.closure_for(pid) for pid in ids]
        annotated = np.array([bool(c) for c in closures], dtype=bool)
        terms = sorted(set().union(*closures)) if closures else []
        col = {t: j for j, t in enumerate(terms)}
        w = np.array([float(index.ic.get(t, 0.0)) for t in terms], dtype=np.float64)
        rows, cols = [], []
        for i, c in enumerate(closures):
            for t in c:
                rows.append(i)
                cols.append(col[t])
        n = len(ids)
        X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                              shape=(n, len(terms)))
        Xw = X.multiply(w[None, :]).tocsr() if len(terms) else X
        inter = np.asarray((Xw @ X.T).todense(), dtype=np.float64) if n else np.zeros((0, 0))
        s = np.asarray(Xw.sum(axis=1)).ravel() if n else np.zeros(0)
        # In place to keep the peak near three n x n float64 arrays (~4.7k proteins).
        union = s[:, None] + s[None, :]
        union -= inter
        positive = union > 0
        np.divide(inter, union, out=inter, where=positive)
        inter[~positive] = 0.0
        del union, positive
        sim = np.clip(inter, 0.0, 1.0, out=inter).astype(np.float32)
        return cls(ids, annotated, sim, index.meta)

    # ---------------------------------------------------------------- access
    def submatrix(self, ids: List[str]) -> np.ndarray:
        r = np.array([self.row[i] for i in ids], dtype=np.int64)
        return self.sim[np.ix_(r, r)]

    def sha256(self) -> str:
        h = hashlib.sha256("\n".join(self.ids).encode("utf-8"))
        h.update(np.ascontiguousarray(self.sim).tobytes())
        return h.hexdigest()

    def n_annotated(self) -> int:
        return int(self.annotated.sum())


def go_relation_loss(table: GoRelationTable, triplets, embeddings: Dict[str, torch.Tensor],
                     key_fn: Callable[[dict], str], device=None
                     ) -> Tuple[Optional[torch.Tensor], List[str], int]:
    """Unweighted L_GO over the batch's anchor and positive proteins. Pure, no state.

    Returns ``(loss_or_None, ids, n_missing)`` where ``ids`` are the deduplicated
    batch proteins found in the table (annotated or not) and ``n_missing`` counts
    deduplicated proteins absent from it. ``None`` when fewer than 2 annotated
    table proteins remain.

      1. Per triplet take (anchor['protein_id'], key_fn(anchor)) and
         (positive['partner_id'], key_fn(positive)); negatives are excluded.
      2. Deduplicate by protein id keeping the first key; ids not in the table
         are counted in n_missing and skipped.
      3. Keep annotated ids; if < 2 remain return (None, ids, n_missing).
      4. E = normalize(stack(emb[key])); C = E E^T; S = table.submatrix(ids);
         return mean over the strict upper triangle of (C - S)^2.
    """
    first_key: Dict[str, str] = {}
    for t in triplets:
        for pid, slot in ((t.anchor.get("protein_id"), t.anchor),
                          (t.positive.get("partner_id"), t.positive)):
            if pid is None:
                continue
            pid = str(pid)
            if pid not in first_key:
                first_key[pid] = key_fn(slot)
    ids: List[str] = []
    n_missing = 0
    for pid in first_key:
        if pid in table.row:
            ids.append(pid)
        else:
            n_missing += 1
    used = [pid for pid in ids if table.annotated[table.row[pid]]]
    if len(used) < 2:
        return None, ids, n_missing
    E = torch.stack([embeddings[first_key[pid]] for pid in used])
    if device is not None:
        E = E.to(device)
    E = F.normalize(E, dim=-1)
    C = E @ E.T
    S = torch.from_numpy(table.submatrix(used)).to(C)
    mask = torch.triu(torch.ones_like(C, dtype=torch.bool), diagonal=1)
    return ((C - S) ** 2)[mask].mean(), ids, n_missing
