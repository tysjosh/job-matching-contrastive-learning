"""GO similarity (phi) variants for a scored split.

Every variant returns one float per record. Conventions:

* BP terms come from the record ``skill_uris`` (unpruned) or the aligned pruned
  record ``skill_uris`` (pruned). These equal ``index.direct[gene]`` of the
  matching index. The pruned index is only ever used with pruned URIs and the
  unpruned index only with unpruned URIs (``_terms`` hard-wires the pairing and
  ``data.assert_aligned`` checks the pruned records match the published ones).
* MF and CC terms come from ``index.direct[gene]`` of the matching F / C index.
* An empty side gives phi = 0.0 (the published convention); the companion
  ``miss_*`` arrays flag those records so fusion can treat "missing" != 0.
* ``simgic``, ``ancestor`` and ``exact`` go through the protected
  ``GoMatcher`` unchanged, so BP simGIC on the unpruned index is bit-identical to
  the published phi.
* ``isa`` variants use an is_a-only closure parsed here from the same OBO (the
  shared parser conflates is_a and part_of), with IC recomputed on the same
  guarded annotation corpus (``index.direct``) under that closure.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence

import numpy as np

from go_ppi_fusion import guard
from go_ppi_fusion.data import Split

ROOT = guard.ROOT
OBO = ROOT / "dataset" / "go_ppi" / "bulk" / "go-basic.obo"
GAF = ROOT / "dataset" / "go_ppi" / "bulk" / "goa_human.gaf.gz"
ALIASES = ROOT / "dataset" / "go_ppi" / "bulk" / "9606.protein.aliases.v12.0.txt.gz"
INFO = ROOT / "dataset" / "go_ppi" / "bulk" / "9606.protein.info.v12.0.txt.gz"
INDEX_PATHS = {
    ("P", "unpruned"): ROOT / "preprocess" / "go_ppi" / "go_index_P.pkl",
    ("F", "unpruned"): ROOT / "preprocess" / "go_ppi" / "go_index_F.pkl",
    ("C", "unpruned"): ROOT / "preprocess" / "go_ppi" / "go_index_C.pkl",
    ("P", "pruned"): ROOT / "preprocess" / "go_withholding" / "converted" / "go_index_P.pkl",
}
SCRATCH = Path("/tmp/goppi_fusion_best")
#: Same pruning the withholding study applied to its BP index.
LEAK_PRUNING = {"with_from_human_protein": True, "excluded_term_roots": ["GO:0005515"]}
NAMESPACE = {"P": "biological_process", "F": "molecular_function", "C": "cellular_component"}


# ------------------------------------------------------------------ indexes
def load_index(aspect: str, version: str):
    """Read-only index load. Pruned F/C are built once into /tmp (never the repo)."""
    from go_ppi_domain.go_ontology import GoIndex
    path = INDEX_PATHS.get((aspect, version))
    if path is not None:
        return GoIndex.load(path)
    if version != "pruned" or aspect not in ("F", "C"):
        raise ValueError((aspect, version))
    cache = SCRATCH / f"go_index_{aspect}_pruned.pkl"
    if cache.exists():
        return GoIndex.load(cache)
    index = build_pruned_index(aspect)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    index.save(cache)
    return index


def build_pruned_index(aspect: str):
    """``GoIndex.build`` with the withholding study's leak pruning, for any aspect."""
    from go_ppi_domain import leak_pruning as lp
    from go_ppi_domain.go_ontology import GoIndex
    acc_map = lp.human_accession_map(GAF, ALIASES, INFO)
    index = GoIndex.build(OBO, GAF, aspect, leak_pruning=dict(LEAK_PRUNING),
                          with_from_symbol_map=acc_map)
    return index


def parse_isa_parents(obo: Path = OBO) -> Dict[str, Dict[str, FrozenSet[str]]]:
    """``{namespace: {term: is_a parents}}`` for non-obsolete terms (is_a only)."""
    parents: Dict[str, set] = defaultdict(set)
    namespace: Dict[str, str] = {}
    cur: Optional[str] = None
    ns: Optional[str] = None
    obsolete = False
    in_term = False

    def flush():
        if in_term and cur and ns and not obsolete:
            namespace[cur] = ns

    with open(obo, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("["):
                flush()
                in_term = line == "[Term]"
                cur, ns, obsolete = None, None, False
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                cur = line[4:].strip()
            elif line.startswith("namespace: "):
                ns = line[11:].strip()
            elif line.startswith("is_obsolete: true"):
                obsolete = True
            elif line.startswith("is_a: ") and cur:
                parents[cur].add(line[6:].split("!")[0].strip())
        flush()
    out: Dict[str, Dict[str, FrozenSet[str]]] = defaultdict(dict)
    for t, n in namespace.items():
        out[n][t] = frozenset(p for p in parents.get(t, ()) if namespace.get(p) == n)
    return out


def isa_ancestors(parents: Dict[str, FrozenSet[str]]) -> Dict[str, FrozenSet[str]]:
    anc: Dict[str, FrozenSet[str]] = {}
    for t in parents:
        acc, stack = {t}, [t]
        while stack:
            for p in parents.get(stack.pop(), ()):
                if p not in acc:
                    acc.add(p)
                    stack.append(p)
        anc[t] = frozenset(acc)
    return anc


def corpus_ic(direct: Dict[str, FrozenSet[str]],
              ancestors: Dict[str, FrozenSet[str]]) -> Dict[str, float]:
    """IC = -log(freq / n_genes) over the propagated corpus (``GoIndex.build`` rule)."""
    freq: Dict[str, int] = defaultdict(int)
    n = 0
    for terms in direct.values():
        acc = set()
        for t in terms:
            acc |= ancestors.get(t, frozenset())
        if not acc:
            continue
        n += 1
        for t in acc:
            freq[t] += 1
    return {t: -math.log(c / n) for t, c in freq.items()}


# ------------------------------------------------------------------ similarity kernels
def _closure(terms, ancestors, cache):
    key = frozenset(terms)
    hit = cache.get(key)
    if hit is None:
        acc = set()
        for t in key:
            acc |= ancestors.get(t, frozenset())
        hit = frozenset(acc)
        cache[key] = hit
    return hit


def simgic_custom(A, B, ancestors, ic, cache) -> float:
    if not A or not B:
        return 0.0
    ca, cb = _closure(A, ancestors, cache), _closure(B, ancestors, cache)
    if not ca or not cb:
        return 0.0
    den = sum(ic.get(t, 0.0) for t in ca | cb)
    if den <= 0:
        return 0.0
    return min(1.0, max(0.0, sum(ic.get(t, 0.0) for t in ca & cb) / den))


def ic_overlap_direct(A, B, ic) -> float:
    """IC-weighted Jaccard over the direct terms (no hierarchy in the set, IC from it)."""
    fa, fb = frozenset(A), frozenset(B)
    if not fa or not fb:
        return 0.0
    den = sum(ic.get(t, 0.0) for t in fa | fb)
    return 0.0 if den <= 0 else sum(ic.get(t, 0.0) for t in fa & fb) / den


class Resnik:
    """Best-match-average Resnik (MICA = max IC over common ancestors)."""

    def __init__(self, ancestors, ic):
        self.anc = ancestors
        self.ic = ic
        self.sorted_anc: Dict[str, List[str]] = {}
        self.mica: Dict[tuple, float] = {}

    def _sorted(self, t):
        s = self.sorted_anc.get(t)
        if s is None:
            s = sorted(self.anc.get(t, ()), key=lambda x: -self.ic.get(x, 0.0))
            self.sorted_anc[t] = s
        return s

    def term(self, a, b) -> float:
        key = (a, b) if a <= b else (b, a)
        v = self.mica.get(key)
        if v is None:
            other = self.anc.get(b, frozenset())
            v = 0.0
            for t in self._sorted(a):
                if t in other:
                    v = self.ic.get(t, 0.0)
                    break
            self.mica[key] = v
        return v

    def bma(self, A, B) -> float:
        A, B = sorted(set(A)), sorted(set(B))
        if not A or not B:
            return 0.0
        M = np.array([[self.term(a, b) for b in B] for a in A])
        return float(0.5 * (M.max(1).mean() + M.max(0).mean()))


# ------------------------------------------------------------------ split-level
def _terms(split: Split, aspect: str, version: str, index) -> tuple:
    """(A_terms, B_terms) per record for one aspect and index version."""
    if aspect == "P":
        if version == "unpruned":
            return split.uris_a, split.uris_b
        if split.pruned_uris_a is None:
            raise ValueError("pruned phi requires the aligned pruned record URIs")
        return split.pruned_uris_a, split.pruned_uris_b
    d = index.direct
    return ([sorted(d.get(g, ())) for g in split.anchors],
            [sorted(d.get(g, ())) for g in split.partners])


def compute_phis(split: Split, versions: Sequence[str] = ("unpruned", "pruned"),
                 resnik: bool = True, log=print) -> Dict[str, np.ndarray]:
    """All phi variants and coverage features for one split (one array per name)."""
    import time
    from go_ppi_domain.go_ontology import GoMatcher

    guard.check_split(split.name)
    out: Dict[str, np.ndarray] = {}
    isa = None
    for version in versions:
        for aspect in ("P", "F", "C"):
            t0 = time.time()
            index = load_index(aspect, version)
            A, B = _terms(split, aspect, version, index)
            tag = f"{aspect}_{version}"
            for mode in ("simgic", "ancestor", "exact"):
                m = GoMatcher(index, similarity_mode=mode)
                vals = []
                for a, b in zip(A, B):
                    try:
                        vals.append(float(m.ontology_set_similarity(a, b)) if a and b else 0.0)
                    except Exception:
                        vals.append(0.0)
                out[f"{mode}_{tag}"] = np.asarray(vals, dtype=float)
            out[f"icdirect_{tag}"] = np.asarray(
                [ic_overlap_direct(a, b, index.ic) for a, b in zip(A, B)])
            out[f"miss_{tag}"] = np.asarray([0.0 if (a and b) else 1.0 for a, b in zip(A, B)])
            out[f"na_{tag}"] = np.asarray([len(a) for a in A], dtype=float)
            out[f"nb_{tag}"] = np.asarray([len(b) for b in B], dtype=float)
            if aspect == "P":
                if resnik:
                    r = Resnik(index.ancestors, index.ic)
                    out[f"resnik_{tag}"] = np.asarray([r.bma(a, b) for a, b in zip(A, B)])
                if isa is None:
                    isa = parse_isa_parents()
                anc = isa_ancestors(isa[NAMESPACE["P"]])
                ic = corpus_ic(index.direct, anc)
                cache: Dict = {}
                out[f"isa_simgic_{tag}"] = np.asarray(
                    [simgic_custom(a, b, anc, ic, cache) for a, b in zip(A, B)])
            log(f"  phi {tag}: {time.time() - t0:.1f}s")
        # multi-aspect combinations (missing aspects drop out of mean / max)
        sims = np.stack([out[f"simgic_{a}_{version}"] for a in "PFC"])
        miss = np.stack([out[f"miss_{a}_{version}"] for a in "PFC"]).astype(bool)
        used = np.where(miss, np.nan, sims)
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[f"simgic_mean_{version}"] = np.nan_to_num(np.nanmean(used, 0), nan=0.0)
                out[f"simgic_max_{version}"] = np.nan_to_num(np.nanmax(used, 0), nan=0.0)
        out[f"miss_all_{version}"] = miss.all(0).astype(float)
    return out


def save_phis(path: Path, phis: Dict[str, np.ndarray]) -> None:
    guard.check_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **phis)


def load_phis(path: Path) -> Dict[str, np.ndarray]:
    guard.check_path(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
