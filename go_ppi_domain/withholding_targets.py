"""Frozen target table for the GO/PPI withholding study (design section 2, gate G2).

All inputs are the post-fix data (``preprocess/go_withholding/{converted,splits}``)
and the PRUNED BP index (propagated is_a + part_of closure; ``regulates`` and
obsolete terms never enter, because ``go_ontology.parse_obo`` ignores them).

Definitions (design 2.1)::

    train partition = partition_roles(train.jsonl, train pool rows)["any"]
    T_train(t) = train-partition proteins (any role) whose closure contains t   (withheld)
    P_train(t) = T_train(t) with >= 1 positive train record                     (n(t), retention)
    P_test(t)  = test anchors with t in closure, >= 1 grade-2 and >= 1 grade-1 record
    P_val(t)   = validation proteins (any role) with t in closure
    NP_train(t; N) = U_{t' in N(t)} P_train(t') minus T_train(t)
    NP_test(t; N)  = test proteins (any role) with closure meeting N(t) and t not in closure

Pipeline: candidate rule (2.2) -> greedy non-nested selection -> S_sib strata
(2.3) -> optimal low x high assignment and greedy Jaccard acceptance with the
fixed size rule (2.4) -> neighbour sets (2.5) -> singleton packs (2.6, amended
by AM1: one target per condition, so there are no pack-mate constraints, no
slot cycling and no 24 -> 20 packing fallback; the removal cap and retention
feasibility are checked per single target) -> per-target retention order (2.7).
``build_target_table`` returns the schema-2 table and the candidate rows.

Everything is deterministic: no set iteration order reaches the output (every
emitted list is sorted or explicitly ordered), and the only randomness is
``random.Random(f"{retention_seed}|{go_id}")``.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from go_ppi_domain.go_ontology import GoIndex, parse_obo

#: Fixed parameters (design 2.2-2.7, amendment AM1); the table records them.
#: ``pack_size`` is 1, the only shape (AM1). ``pack_removal_cap`` is the k=0
#: removal cap applied to each single-target condition.
DEFAULT_PARAMS: Dict[str, Any] = {
    "min_train_proteins": 40, "min_test_support": 8, "max_removal_frac": 0.05,
    "max_protein_frac": 0.10, "min_retention_eligible": 25, "max_jaccard": 0.2,
    "strata_quantiles": [33, 67], "n_pairs": 12, "min_pairs": 10, "pack_size": 1,
    "pack_removal_cap": 0.08, "retention_seed": 20251009,
}


class TargetTableStop(RuntimeError):
    """A protocol stop (exit 3): < min_pairs matched pairs, or a selected target
    fails the single-target removal cap or retention feasibility (AM1)."""


# ------------------------------------------------------------------- graph
@dataclass(frozen=True)
class BpGraph:
    parents: Dict[str, FrozenSet[str]]
    children: Dict[str, FrozenSet[str]]
    ancestors: Dict[str, FrozenSet[str]]      # includes the term itself (GoIndex)
    name: Dict[str, str]
    depth: Dict[str, int]


def bp_graph(obo: Path, index: GoIndex) -> BpGraph:
    """Parents/children restricted to ``index.ancestors`` keys, via ``parse_obo``."""
    parents_all, _ns, _alt, names = parse_obo(Path(obo))
    keep = set(index.ancestors)
    parents = {t: frozenset(p for p in parents_all.get(t, ()) if p in keep) for t in keep}
    children: Dict[str, Set[str]] = defaultdict(set)
    for t, ps in parents.items():
        for p in ps:
            children[p].add(t)
    return BpGraph(parents=parents,
                   children={t: frozenset(children.get(t, ())) for t in keep},
                   ancestors=dict(index.ancestors),
                   name={t: names.get(t, "") for t in keep},
                   depth=dict(index.depth))


def descendants(graph: BpGraph) -> Dict[str, FrozenSet[str]]:
    """Strict descendants of every term (inverted ancestor closure)."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for t, anc in graph.ancestors.items():
        for a in anc:
            if a != t:
                out[a].add(t)
    return {t: frozenset(out.get(t, ())) for t in graph.ancestors}


def _excluded(t: str, graph: BpGraph, desc: Mapping[str, FrozenSet[str]]) -> FrozenSet[str]:
    return frozenset({t}) | graph.ancestors.get(t, frozenset()) | desc.get(t, frozenset())


def n_sib(t: str, graph: BpGraph, desc: Optional[Mapping[str, FrozenSet[str]]] = None
          ) -> FrozenSet[str]:
    """U children(p) for p in parents(t), minus {t} U ancestors(t) U descendants(t)."""
    desc = descendants(graph) if desc is None else desc
    sib: Set[str] = set()
    for p in graph.parents.get(t, ()):
        sib |= graph.children.get(p, frozenset())
    return frozenset(sib - _excluded(t, graph, desc))


def n_2(t: str, graph: BpGraph, desc: Optional[Mapping[str, FrozenSet[str]]] = None
        ) -> FrozenSet[str]:
    """Terms within <= 2 undirected hops, same exclusion as ``n_sib``."""
    desc = descendants(graph) if desc is None else desc

    def nb(x: str) -> FrozenSet[str]:
        return graph.parents.get(x, frozenset()) | graph.children.get(x, frozenset())

    one = nb(t)
    two: Set[str] = set(one)
    for x in one:
        two |= nb(x)
    return frozenset(two - _excluded(t, graph, desc))


# ------------------------------------------------------------- data roles
def _pids(rec: dict) -> Tuple[str, str]:
    m = rec.get("metadata", {})
    a = m.get("protein_id") or rec["resume"]["protein_id"]
    b = m.get("partner_id") or rec["job"]["partner_id"]
    return str(a), str(b)


def partition_roles(train_records: Iterable[dict], pool_rows: Iterable[dict]
                    ) -> Dict[str, Set[str]]:
    """``{"positive", "pool_anchor", "pool_member", "any"}`` protein sets of the train partition."""
    positive: Set[str] = set()
    for r in train_records:
        positive.update(_pids(r))
    anchors: Set[str] = set()
    members: Set[str] = set()
    for row in pool_rows:
        anchors.add(row["anchor_id"])
        members.update(row.get("weak_evidence", []))
        members.update(row.get("no_interaction", []))
    return {"positive": positive, "pool_anchor": anchors, "pool_member": members,
            "any": positive | anchors | members}


def term_proteins(index: GoIndex, proteins: Iterable[str],
                  terms: Optional[Iterable[str]] = None) -> Dict[str, Set[str]]:
    """``{t: proteins whose closure contains t}``; ``terms=None`` = every term met."""
    wanted = None if terms is None else set(terms)
    out: Dict[str, Set[str]] = defaultdict(set)
    for p in proteins:
        for t in index.closure_for(p):
            if wanted is None or t in wanted:
                out[t].add(p)
    if wanted is not None:
        for t in wanted:
            out.setdefault(t, set())
    return dict(out)


def undirected_positive_pairs(train_records: Iterable[dict]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for r in train_records:
        a, b = _pids(r)
        out.add((a, b) if a <= b else (b, a))
    return out


def degree(pairs: Iterable[Tuple[str, str]]) -> Counter:
    deg: Counter = Counter()
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    return deg


def support_S(t: str, nbr_terms: Iterable[str], P_train: Mapping[str, Set[str]],
              deg: Mapping[str, int], T_train_t: Iterable[str] = ()) -> Dict[str, int]:
    """``{"S": sum deg(p) over NP(t), "proteins": |NP(t)|}``, NP = U P_train(t') minus T_train(t)."""
    np_set: Set[str] = set()
    for u in nbr_terms:
        np_set |= P_train.get(u, set())
    np_set -= set(T_train_t)
    return {"S": int(sum(deg.get(p, 0) for p in np_set)), "proteins": len(np_set)}


def neighbour_proteins(nbr_terms: Iterable[str], P_train: Mapping[str, Set[str]],
                       T_train_t: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for u in nbr_terms:
        out |= P_train.get(u, set())
    return out - set(T_train_t)


def retention_eligible(t: str, P_train: Mapping[str, Set[str]], W: Set[str],
                       train_records: Mapping[str, Sequence[Tuple[str, str]]],
                       pools: Mapping[str, FrozenSet[str]], views: Set[str]) -> Set[str]:
    """Design 2.2 rule 5: p in P_train(t) with a positive record {p, q}, q not in W,
    whose anchor's pool row has >= 1 member not in W that is in ``views``.

    ``train_records`` is ``{protein: [(anchor, partner), ...]}`` (records touching
    that protein, either slot); ``pools`` is ``{anchor: all members}``.
    """
    out: Set[str] = set()
    for p in P_train.get(t, ()):
        for a, b in train_records.get(p, ()):
            q = b if a == p else a
            if q in W:
                continue
            if any(m not in W and m in views for m in pools.get(a, ())):
                out.add(p)
                break
    return out


def removal_frac(W: Set[str], adjacency: Mapping[str, Set[str]], n_pairs: int) -> float:
    """Share of undirected train positive pairs with an endpoint in ``W``."""
    removed: Set[Tuple[str, str]] = set()
    for p in W:
        for q in adjacency.get(p, ()):
            removed.add((p, q) if p <= q else (q, p))
    return len(removed) / n_pairs if n_pairs else 0.0


def jaccard(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# ----------------------------------------------------------------- context
@dataclass
class TargetContext:
    """Everything the candidate rule, matching and packing read."""
    index: GoIndex
    graph: BpGraph
    desc: Dict[str, FrozenSet[str]]
    roles: Dict[str, Set[str]]
    T_train: Dict[str, Set[str]]
    P_train: Dict[str, Set[str]]
    pairs: Set[Tuple[str, str]]
    adjacency: Dict[str, Set[str]]
    deg: Counter
    records_by_protein: Dict[str, List[Tuple[str, str]]]
    pool_members: Dict[str, FrozenSet[str]]
    views: Set[str]
    test_grades: Dict[str, Counter]           # test anchor -> Counter(grade)
    test_proteins: Set[str]
    val_proteins: Set[str]
    _nsib: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    _n2: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    _elig: Dict[Tuple[str, FrozenSet[str]], Set[str]] = field(default_factory=dict)

    def n_sib(self, t: str) -> FrozenSet[str]:
        if t not in self._nsib:
            self._nsib[t] = n_sib(t, self.graph, self.desc)
        return self._nsib[t]

    def n_2(self, t: str) -> FrozenSet[str]:
        if t not in self._n2:
            self._n2[t] = n_2(t, self.graph, self.desc)
        return self._n2[t]

    def P_test(self, t: str) -> Set[str]:
        return {a for a, g in self.test_grades.items()
                if g.get(2, 0) >= 1 and g.get(1, 0) >= 1 and t in self.index.closure_for(a)}

    def P_val(self, t: str) -> Set[str]:
        return {p for p in self.val_proteins if t in self.index.closure_for(p)}

    def NP_test(self, t: str, nbr: FrozenSet[str]) -> Set[str]:
        out = set()
        for p in self.test_proteins:
            c = self.index.closure_for(p)
            if t not in c and not nbr.isdisjoint(c):
                out.add(p)
        return out

    def eligible(self, t: str, W_key: FrozenSet[str], W: Set[str]) -> Set[str]:
        key = (t, W_key)
        if key not in self._elig:
            self._elig[key] = retention_eligible(t, self.P_train, W, self.records_by_protein,
                                                 self.pool_members, self.views)
        return self._elig[key]

    def W_of(self, targets: Iterable[str]) -> Set[str]:
        out: Set[str] = set()
        for t in targets:
            out |= self.T_train.get(t, set())
        return out


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_context(converted: Path, splits: Path, obo: Path) -> TargetContext:
    index = GoIndex.load(Path(converted) / "go_index_P.pkl")
    graph = bp_graph(obo, index)
    desc = descendants(graph)
    train = _read_jsonl(Path(splits) / "train.jsonl")
    pools_all = _read_jsonl(Path(splits) / "negative_pools.jsonl")
    train_pools = [r for r in pools_all if r["split"] == "train"]
    roles = partition_roles(train, train_pools)
    T_train = term_proteins(index, roles["any"])
    P_train = {t: s & roles["positive"] for t, s in T_train.items()}
    pairs = undirected_positive_pairs(train)
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    by_protein: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in train:
        a, b = _pids(r)
        by_protein[a].append((a, b))
        by_protein[b].append((a, b))
    members = {r["anchor_id"]: frozenset(r.get("weak_evidence", []))
               | frozenset(r.get("no_interaction", [])) for r in train_pools}
    views = {r["protein_id"] for r in _read_jsonl(Path(converted) / "proteins.jsonl")}
    test_grades: Dict[str, Counter] = defaultdict(Counter)
    test_proteins: Set[str] = set()
    for r in _read_jsonl(Path(splits) / "test.jsonl"):
        a, b = _pids(r)
        test_grades[a][int(r["metadata"]["go_grade"])] += 1
        test_proteins.update((a, b))
    val_proteins: Set[str] = set()
    for r in _read_jsonl(Path(splits) / "validation.jsonl"):
        val_proteins.update(_pids(r))
    for r in pools_all:
        if r["split"] == "test":
            test_proteins.add(r["anchor_id"])
            test_proteins.update(r.get("weak_evidence", []))
            test_proteins.update(r.get("no_interaction", []))
        elif r["split"] == "validation":
            val_proteins.add(r["anchor_id"])
            val_proteins.update(r.get("weak_evidence", []))
            val_proteins.update(r.get("no_interaction", []))
    return TargetContext(index=index, graph=graph, desc=desc, roles=roles, T_train=T_train,
                         P_train=P_train, pairs=pairs, adjacency=dict(adjacency),
                         deg=degree(pairs), records_by_protein=dict(by_protein),
                         pool_members=members, views=views, test_grades=dict(test_grades),
                         test_proteins=test_proteins, val_proteins=val_proteins)


# --------------------------------------------------------- candidate rule
def filter_candidates(ctx: TargetContext, params: Mapping[str, Any]) -> List[dict]:
    """One row per term with n(t) >= min_train_proteins; ``passes`` = rules 1-5.

    Rule 5 (the expensive one) is evaluated only for terms passing rules 1-4.
    """
    n_with_pos = len(ctx.roles["positive"])
    n_pairs = len(ctx.pairs)
    rows = []
    for t in sorted(ctx.P_train):
        n = len(ctx.P_train[t])
        if n < params["min_train_proteins"]:
            continue
        support = len(ctx.P_test(t))
        rf = removal_frac(ctx.T_train[t], ctx.adjacency, n_pairs)
        row = {"go_id": t, "n_train_proteins": n, "n_train_annotated": len(ctx.T_train[t]),
               "test_support": support, "removal_frac": rf,
               "protein_frac": n / n_with_pos if n_with_pos else 0.0,
               "rule1_min_train": True,
               "rule2_min_test": support >= params["min_test_support"],
               "rule3_removal": rf <= params["max_removal_frac"],
               "rule4_protein_frac": n <= params["max_protein_frac"] * n_with_pos,
               "n_retention_eligible": None, "rule5_retention": None}
        if row["rule2_min_test"] and row["rule3_removal"] and row["rule4_protein_frac"]:
            W = ctx.T_train[t]
            elig = ctx.eligible(t, frozenset([t]), W)
            row["n_retention_eligible"] = len(elig)
            row["rule5_retention"] = len(elig) >= params["min_retention_eligible"]
        row["passes"] = bool(row["rule2_min_test"] and row["rule3_removal"]
                             and row["rule4_protein_frac"] and row["rule5_retention"])
        rows.append(row)
    return rows


def greedy_non_nested(rows: Sequence[dict], P_train: Mapping[str, Set[str]],
                      ancestors: Mapping[str, FrozenSet[str]], max_jaccard: float = 0.2
                      ) -> List[dict]:
    """Rule 6: order by test support desc, GO id asc; reject nested or Jaccard > max."""
    accepted: List[dict] = []
    for row in sorted(rows, key=lambda r: (-r["test_support"], r["go_id"])):
        t = row["go_id"]
        ok = True
        for a in accepted:
            u = a["go_id"]
            if u in ancestors.get(t, ()) or t in ancestors.get(u, ()):
                ok = False
                break
            if jaccard(P_train[t], P_train[u]) > max_jaccard:
                ok = False
                break
        if ok:
            accepted.append(row)
    return accepted


def assign_strata(rows: Sequence[dict], quantiles: Sequence[float]) -> Tuple[float, float]:
    """Set ``row["stratum"]`` in {"low", "high", None}; returns the two cuts."""
    s = np.array([r["S_sib"] for r in rows], dtype=float)
    lo = float(np.percentile(s, quantiles[0])) if len(s) else 0.0
    hi = float(np.percentile(s, quantiles[1])) if len(s) else 0.0
    for r in rows:
        r["stratum"] = "low" if r["S_sib"] <= lo else ("high" if r["S_sib"] >= hi else None)
    return lo, hi


def _pair_num(pair_id: str) -> int:
    return int(pair_id[1:])


def match_strata(low: Sequence[dict], high: Sequence[dict], P_train: Mapping[str, Set[str]],
                 n_pairs: int = 12, max_jaccard: float = 0.2) -> Dict[str, List[dict]]:
    """Optimal low x high assignment, then greedy Jaccard acceptance (design 2.4).

    Cost = Euclidean distance on z-scored (log n, log test support), z over
    low U high. Pairs sorted by (cost, low id, high id). A pair is accepted while
    both its targets have Jaccard <= max with every accepted target, until
    ``n_pairs``; later pairs that pass that check form the ordered replacement
    queue. Every passing pair is numbered m01, m02, ... in this order.
    """
    from scipy.optimize import linear_sum_assignment

    if not low or not high:
        return {"accepted": [], "queue": []}
    both = list(low) + list(high)
    X = np.array([[math.log(r["n_train_proteins"]), math.log(r["test_support"])]
                  for r in both], dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    zl, zh = Z[:len(low)], Z[len(low):]
    cost = np.sqrt(((zl[:, None, :] - zh[None, :, :]) ** 2).sum(axis=2))
    ri, ci = linear_sum_assignment(cost)
    cand = sorted(((float(cost[i, j]), low[i]["go_id"], high[j]["go_id"])
                   for i, j in zip(ri, ci)))
    accepted: List[dict] = []
    queue: List[dict] = []
    acc_targets: List[str] = []
    k = 0
    for c, lo_t, hi_t in cand:
        if any(jaccard(P_train[x], P_train[y]) > max_jaccard
               for x in (lo_t, hi_t) for y in acc_targets):
            continue
        k += 1
        pair = {"pair_id": f"m{k:02d}", "low": lo_t, "high": hi_t, "cost": c}
        if len(accepted) < n_pairs:
            accepted.append(pair)
            acc_targets.extend((lo_t, hi_t))
        else:
            queue.append(pair)
    return {"accepted": accepted, "queue": queue}


def apply_size_rule(accepted: Sequence[dict], queue: Sequence[dict], n_pairs: int = 12,
                    min_pairs: int = 10) -> Tuple[List[dict], List[dict], bool]:
    """>= n_pairs -> first n_pairs; [min_pairs, n_pairs) -> exactly the first
    min_pairs (shortfall; the rest go to the queue front); < min_pairs -> stop."""
    accepted = list(accepted)
    if len(accepted) >= n_pairs:
        return accepted[:n_pairs], accepted[n_pairs:] + list(queue), False
    if len(accepted) >= min_pairs:
        return accepted[:min_pairs], accepted[min_pairs:] + list(queue), True
    raise TargetTableStop(f"only {len(accepted)} matched pairs accepted (< {min_pairs})")


# ----------------------------------------------------------------- packing
def singleton_packs(targets: Sequence[str],
                    pack_ok: Callable[[FrozenSet[str]], bool]) -> List[List[str]]:
    """Design 2.6 as amended by AM1: one target per pack (= per condition).

    There are no pack-mate constraints. Each target must still pass ``pack_ok``
    on its own (the k=0 removal cap and retention feasibility, both with
    W = T_train(t)); a failure is a protocol stop. The message holds counts only.
    """
    failed = [t for t in targets if not pack_ok(frozenset([t]))]
    if failed:
        raise TargetTableStop(f"{len(failed)} of {len(targets)} selected targets fail the "
                              "single-target removal cap or retention feasibility")
    return [[t] for t in targets]


# ------------------------------------------------------------------- table
def _smd(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def build_target_table(ctx: TargetContext, inputs: Dict[str, str],
                       params: Optional[Mapping[str, Any]] = None) -> Tuple[dict, List[dict]]:
    """The schema-2 target table and the candidate rows (design 2.2-2.8, AM1).

    ``params`` overrides ``DEFAULT_PARAMS`` (tests only; the G2 script passes
    none). ``DEFAULT_PARAMS`` is read at call time.
    """
    P = dict(DEFAULT_PARAMS)
    P.update(params or {})
    rows = filter_candidates(ctx, P)
    passing = [r for r in rows if r["passes"]]
    non_nested = greedy_non_nested(passing, ctx.P_train, ctx.graph.ancestors, P["max_jaccard"])
    nn_ids = {r["go_id"] for r in non_nested}
    for r in non_nested:
        t = r["go_id"]
        ss = support_S(t, ctx.n_sib(t), ctx.P_train, ctx.deg, ctx.T_train[t])
        s2 = support_S(t, ctx.n_2(t), ctx.P_train, ctx.deg, ctx.T_train[t])
        r.update({"S_sib": ss["S"], "S_sib_proteins": ss["proteins"],
                  "S_2": s2["S"], "S_2_proteins": s2["proteins"]})
    lo_cut, hi_cut = assign_strata(non_nested, P["strata_quantiles"])
    for r in rows:
        r["non_nested"] = r["go_id"] in nn_ids
        r.setdefault("stratum", None)
    low = [r for r in non_nested if r["stratum"] == "low"]
    high = [r for r in non_nested if r["stratum"] == "high"]
    matched = match_strata(low, high, ctx.P_train, P["n_pairs"], P["max_jaccard"])
    accepted, queue, shortfall = apply_size_rule(matched["accepted"], matched["queue"],
                                                 P["n_pairs"], P["min_pairs"])
    strata = {r["go_id"]: r["stratum"] for r in non_nested}
    by_id = {r["go_id"]: r for r in non_nested}

    def pack_ok(pk: FrozenSet[str]) -> bool:
        W = ctx.W_of(pk)
        if removal_frac(W, ctx.adjacency, len(ctx.pairs)) > P["pack_removal_cap"]:
            return False
        return all(len(ctx.eligible(t, pk, W)) >= P["min_retention_eligible"] for t in pk)

    pairs_used = sorted(accepted, key=lambda p: _pair_num(p["pair_id"]))
    pair_of = {}
    for p in pairs_used:
        pair_of[p["low"]] = p["pair_id"]
        pair_of[p["high"]] = p["pair_id"]
    order_key = {t: (_pair_num(pair_of[t]), 0 if strata[t] == "low" else 1) for t in pair_of}
    # AM1: one pack per target, in target order (pair number, low first).
    packs_sorted = singleton_packs(sorted(pair_of, key=lambda x: order_key[x]), pack_ok)
    pack_of = {}
    packs_out = []
    for i, pk in enumerate(packs_sorted, 1):
        pid = f"p{i:02d}"
        W = ctx.W_of(pk)
        key = frozenset(pk)
        elig = {t: ctx.eligible(t, key, W) for t in pk}
        packs_out.append({"pack_id": pid, "targets": list(pk),
                          "removal_frac_k0": removal_frac(W, ctx.adjacency, len(ctx.pairs)),
                          "withheld_k0": len(W),
                          "min_retention_eligible": min(len(e) for e in elig.values())})
        for t in pk:
            pack_of[t] = (pid, elig[t])

    targets_out = []
    for t in sorted(pair_of, key=lambda x: order_key[x]):
        r = by_id[t]
        pid, elig = pack_of[t]
        order = sorted(elig)
        random.Random(f"{P['retention_seed']}|{t}").shuffle(order)
        nsib, n2 = ctx.n_sib(t), ctx.n_2(t)
        p_test = ctx.P_test(t)
        targets_out.append({
            "go_id": t, "name": ctx.graph.name.get(t, ""), "depth": ctx.graph.depth.get(t),
            "stratum": strata[t], "pair_id": pair_of[t], "pack_id": pid,
            "n_train_proteins": len(ctx.P_train[t]),
            "train_proteins": sorted(ctx.P_train[t]),
            "train_annotated_proteins": sorted(ctx.T_train[t]),
            "n_train_annotated": len(ctx.T_train[t]),
            "retention_eligible": sorted(elig), "retention_order": list(order),
            "test_anchors": sorted(p_test), "test_support": len(p_test),
            "test_pos_records": sum(ctx.test_grades[a].get(2, 0) for a in p_test),
            "test_weak_records": sum(ctx.test_grades[a].get(1, 0) for a in p_test),
            "validation_proteins": sorted(ctx.P_val(t)),
            "removal_frac": r["removal_frac"],
            "S_sib": r["S_sib"], "S_sib_proteins": r["S_sib_proteins"],
            "S_2": r["S_2"], "S_2_proteins": r["S_2_proteins"],
            "N_sib": sorted(nsib), "N_2_size": len(n2),
            "nbr_train_proteins_sib": sorted(neighbour_proteins(nsib, ctx.P_train, ctx.T_train[t])),
            "nbr_test_proteins_sib": sorted(ctx.NP_test(t, nsib)),
            "nbr_train_proteins_2": sorted(neighbour_proteins(n2, ctx.P_train, ctx.T_train[t])),
            "nbr_test_proteins_2": sorted(ctx.NP_test(t, n2)),
        })
    lows = [x for x in targets_out if x["stratum"] == "low"]
    highs = [x for x in targets_out if x["stratum"] == "high"]
    n_targets, n_packs = len(targets_out), len(packs_out)
    n_conditions = 5 * n_packs + 1
    table = {
        "schema_version": 2,
        "inputs": dict(inputs),
        "params": {k: P[k] for k in DEFAULT_PARAMS},
        "train_totals": {
            "partition_proteins": len(ctx.roles["any"]),
            "proteins_with_positives": len(ctx.roles["positive"]),
            "pool_only_proteins": len(ctx.roles["any"] - ctx.roles["positive"]),
            "undirected_positive_pairs": len(ctx.pairs)},
        "strata_cuts": {"low_max_S_sib": lo_cut, "high_min_S_sib": hi_cut},
        "n_candidates": len(passing), "n_non_nested": len(non_nested),
        "n_low": len(low), "n_high": len(high),
        "n_pairs_accepted": len(matched["accepted"]),
        "shortfall": bool(shortfall),
        "n_targets": n_targets, "n_packs": n_packs, "n_conditions": n_conditions,
        "n_runs": 4 * 5 * n_conditions,
        "replacement_queue": [p["pair_id"] for p in queue],
        "targets": targets_out,
        "pairs": [{"pair_id": p["pair_id"], "low": p["low"], "high": p["high"],
                   "cost": p["cost"]} for p in pairs_used],
        "packs": packs_out,
        "balance": {
            "smd_log_n": _smd([math.log(x["n_train_proteins"]) for x in lows],
                              [math.log(x["n_train_proteins"]) for x in highs]),
            "smd_log_support": _smd([math.log(x["test_support"]) for x in lows],
                                    [math.log(x["test_support"]) for x in highs])},
    }
    return table, rows


def table_json_bytes(table: dict) -> bytes:
    """Canonical serialisation (sha256 of these bytes is the table hash)."""
    return (json.dumps(table, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
