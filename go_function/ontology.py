"""Typed GO ontology (design §1) and the DeepGOZero propagation-rule replica.

Pure Python, no torch. go_ppi_domain/go_ontology.py is neither imported nor touched.

``DgzRule`` is a port of ``Ontology(filename, with_rels=True)`` from the public
DeepGOZero source, https://raw.githubusercontent.com/bio-ontology-research-group/
deepgozero/main/utils.py (class ``Ontology``: ``load``, ``get_anchestors``,
``get_prop_terms``), read with web_fetch. The port keeps DGZ's semantics exactly:
every ``relationship:`` target is appended to ``is_a`` regardless of type and
namespace, alt ids are aliases of the same term object, obsolete primaries are
deleted after alias registration, unknown ids have an empty closure. The only
addition is a per-id closure cache, which does not change any result.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

from go_function.runtime import FatalError

TRUE_PATH = ("is_a", "part_of")
_GO_ID = re.compile(r"^GO:\d{7}$")


class OntologyError(FatalError):
    """Malformed OBO, dangling edge or cycle (fatal, exit 2)."""

    def __init__(self, message: str, line: Optional[int] = None):
        self.line = line
        super().__init__(f"{message} (line {line})" if line is not None else message)


@dataclass(frozen=True)
class Term:
    id: str
    name: str
    namespace: str
    definition: str
    is_obsolete: bool
    alt_ids: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    intersection: Tuple[Tuple[Optional[str], str], ...]


@dataclass(frozen=True, eq=False)
class Ontology:
    terms: Mapping[str, Term]
    alt_to_primary: Mapping[str, str]
    obsolete: FrozenSet[str]
    report: Mapping[str, int] = field(default_factory=dict)
    #: obsolete id (and the alt ids of an obsolete stanza) -> (name, definition) of
    #: that stanza, as written in go.obo; term text only (amendment A3)
    obsolete_text: Mapping[str, Tuple[str, str]] = field(default_factory=dict)

    def primary(self, go_id: str) -> Optional[str]:
        """alt -> primary; live -> itself; obsolete or unknown -> None."""
        if go_id in self.terms:
            return go_id
        return self.alt_to_primary.get(go_id)


# ------------------------------------------------------------------ parser
def _value(rest: str) -> str:
    """Strip a trailing ``! comment`` and ``{qualifiers}`` from a tag value."""
    rest = rest.split(" ! ", 1)[0]
    rest = re.sub(r"\s*\{[^}]*\}\s*$", "", rest)
    return rest.strip()


def _definition(rest: str) -> str:
    m = re.match(r'^"((?:[^"\\]|\\.)*)"', rest)
    return m.group(1).replace('\\"', '"') if m else ""


def parse_obo(path: Path) -> Ontology:
    """Parse ``[Term]`` stanzas with typed edges (design §1.1)."""
    raw: List[dict] = []
    cur: Optional[dict] = None
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.rstrip("\n").strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("["):
                cur = None
                if line == "[Term]":
                    cur = {"line": ln, "id": None, "name": "", "namespace": None,
                           "definition": "", "is_obsolete": False, "alt_ids": [],
                           "edges": [], "intersection": []}
                    raw.append(cur)
                continue
            if cur is None:
                continue
            tag, sep, rest = line.partition(":")
            if not sep:
                raise OntologyError(f"malformed line {line!r}", ln)
            tag, rest = tag.strip(), rest.strip()
            if tag == "id":
                cur["id"] = rest
            elif tag == "name":
                cur["name"] = rest
            elif tag == "namespace":
                cur["namespace"] = rest
            elif tag == "def":
                cur["definition"] = _definition(rest)
            elif tag == "is_obsolete":
                cur["is_obsolete"] = rest == "true"
            elif tag == "alt_id":
                cur["alt_ids"].append(_value(rest))
            elif tag == "is_a":
                cur["edges"].append(("is_a", _value(rest), ln))
            elif tag == "relationship":
                parts = _value(rest).split()
                if len(parts) != 2:
                    raise OntologyError(f"malformed relationship {rest!r}", ln)
                cur["edges"].append((parts[0], parts[1], ln))
            elif tag == "intersection_of":
                parts = _value(rest).split()
                if len(parts) == 1:
                    cur["intersection"].append((None, parts[0], ln))
                elif len(parts) == 2:
                    cur["intersection"].append((parts[0], parts[1], ln))
                else:
                    raise OntologyError(f"malformed intersection_of {rest!r}", ln)

    live_ids: Set[str] = set()
    obsolete: Set[str] = set()
    alt_to_primary: Dict[str, str] = {}
    for st in raw:
        if not st["id"]:
            raise OntologyError("[Term] stanza without id", st["line"])
        if not st["namespace"]:
            raise OntologyError(f"term {st['id']} without namespace", st["line"])
        if st["id"] in live_ids or st["id"] in obsolete:
            raise OntologyError(f"duplicate term id {st['id']}", st["line"])
        (obsolete if st["is_obsolete"] else live_ids).add(st["id"])
    for st in raw:
        for a in st["alt_ids"]:
            if st["is_obsolete"]:
                obsolete.add(a)
            else:
                if a in live_ids or a in alt_to_primary:
                    raise OntologyError(f"alt_id {a} collides with another id", st["line"])
                alt_to_primary[a] = st["id"]

    report = {"n_term_stanzas": len(raw), "n_live": len(live_ids), "n_obsolete": len(obsolete),
              "n_alt_ids": len(alt_to_primary), "edges_to_obsolete_dropped": 0,
              "edges_to_alt_mapped": 0, "intersection_to_alt_mapped": 0}

    def resolve(target: str, ln: int, kind: str) -> Optional[str]:
        if target in live_ids:
            return target
        if target in alt_to_primary:
            report[f"{kind}_to_alt_mapped"] += 1
            return alt_to_primary[target]
        if target in obsolete:
            return None
        raise OntologyError(f"dangling {kind} target {target}", ln)

    terms: Dict[str, Term] = {}
    for st in raw:
        if st["is_obsolete"]:
            continue
        edges = []
        for rel, tgt, ln in st["edges"]:
            r = resolve(tgt, ln, "edges")
            if r is None:
                report["edges_to_obsolete_dropped"] += 1
                continue
            edges.append((rel, r))
        inter = []
        for rel, tgt, ln in st["intersection"]:
            r = resolve(tgt, ln, "intersection")
            if r is None:
                raise OntologyError(f"intersection_of an obsolete term {tgt}", ln)
            inter.append((rel, r))
        terms[st["id"]] = Term(id=st["id"], name=st["name"], namespace=st["namespace"],
                               definition=st["definition"], is_obsolete=False,
                               alt_ids=tuple(st["alt_ids"]), edges=tuple(edges),
                               intersection=tuple(inter))
    obsolete_text: Dict[str, Tuple[str, str]] = {}
    for st in raw:
        if st["is_obsolete"]:
            for g in (st["id"], *st["alt_ids"]):
                obsolete_text[g] = (st["name"], st["definition"])
    return Ontology(terms=terms, alt_to_primary=alt_to_primary,
                    obsolete=frozenset(obsolete), report=report, obsolete_text=obsolete_text)


def parse_report(ont: Ontology) -> Dict[str, int]:
    return dict(ont.report)


# ------------------------------------------------------------------ closures
def parents(ont: Ontology, rels: Iterable[str] = TRUE_PATH,
            same_namespace: bool = True) -> Dict[str, FrozenSet[str]]:
    rels = frozenset(rels)
    out: Dict[str, FrozenSet[str]] = {}
    for tid, t in ont.terms.items():
        out[tid] = frozenset(p for r, p in t.edges if r in rels and
                             (not same_namespace or ont.terms[p].namespace == t.namespace))
    return out


def ancestors(ont: Ontology, rels: Iterable[str] = TRUE_PATH,
              same_namespace: bool = True) -> Dict[str, FrozenSet[str]]:
    """Reflexive closure over ``rels`` by memoised iterative DFS; a cycle raises."""
    par = parents(ont, rels, same_namespace)
    done: Dict[str, FrozenSet[str]] = {}
    for start in sorted(par):
        if start in done:
            continue
        on_stack: Set[str] = {start}
        stack: List[Tuple[str, Iterable[str]]] = [(start, iter(sorted(par[start])))]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                on_stack.discard(node)
                acc: Set[str] = {node}
                for p in par[node]:
                    acc |= done[p]
                done[node] = frozenset(acc)
                continue
            if nxt in done:
                continue
            if nxt in on_stack:
                raise OntologyError(f"cycle through {nxt} over relations {sorted(set(rels))}")
            on_stack.add(nxt)
            stack.append((nxt, iter(sorted(par[nxt]))))
    return done


def descendants(anc: Mapping[str, FrozenSet[str]]) -> Dict[str, FrozenSet[str]]:
    """Inverse of a reflexive ancestor map (reflexive)."""
    out: Dict[str, Set[str]] = {t: {t} for t in anc}
    for t, a in anc.items():
        for u in a:
            out.setdefault(u, {u}).add(t)
    return {t: frozenset(s) for t, s in out.items()}


@lru_cache(maxsize=8)
def _isa_children(ont: Ontology) -> Dict[str, FrozenSet[str]]:
    ch: Dict[str, Set[str]] = {t: set() for t in ont.terms}
    for tid, t in ont.terms.items():
        for r, p in t.edges:
            if r == "is_a":
                ch[p].add(tid)
    return {t: frozenset(s) for t, s in ch.items()}


@lru_cache(maxsize=8)
def _undirected(ont: Ontology, rels: Tuple[str, ...]) -> Dict[str, FrozenSet[str]]:
    par = parents(ont, rels, same_namespace=True)
    nb: Dict[str, Set[str]] = {t: set(ps) for t, ps in par.items()}
    for t, ps in par.items():
        for p in ps:
            nb[p].add(t)
    return {t: frozenset(s) for t, s in nb.items()}


def isa_siblings(ont: Ontology, t: str, anc: Mapping[str, FrozenSet[str]],
                 desc: Mapping[str, FrozenSet[str]]) -> FrozenSet[str]:
    """Terms sharing an is_a parent with t, minus t, anc(t) and desc(t)."""
    ch = _isa_children(ont)
    sib: Set[str] = set()
    for r, p in ont.terms[t].edges:
        if r == "is_a":
            sib |= ch[p]
    return frozenset(sib - {t} - anc[t] - desc[t])


def two_hop(ont: Ontology, t: str, anc: Mapping[str, FrozenSet[str]],
            desc: Mapping[str, FrozenSet[str]], rels: Iterable[str] = TRUE_PATH) -> FrozenSet[str]:
    """Terms within <= 2 undirected hops over ``rels``, minus t, anc(t), desc(t)."""
    nb = _undirected(ont, tuple(sorted(set(rels))))
    one = nb[t]
    two: Set[str] = set(one)
    for u in one:
        two |= nb[u]
    return frozenset(two - {t} - anc[t] - desc[t])


def logical_definition(ont: Ontology, t: str
                       ) -> Tuple[Optional[str], Tuple[Tuple[str, str], ...]]:
    """(genus, differentia) from ``intersection_of``; (None, ()) when absent."""
    genus: Optional[str] = None
    diff: List[Tuple[str, str]] = []
    for rel, cls in ont.terms[t].intersection:
        if rel is None:
            if genus is None:
                genus = cls
        else:
            diff.append((rel, cls))
    return genus, tuple(diff)


# ------------------------------------------------------------------ DGZ rule
class DgzRule:
    """Port of DGZ ``utils.Ontology(with_rels=True)`` (see module docstring)."""

    def __init__(self, obo_path: Path):
        self.ont = self._load(obo_path)
        self._cache: Dict[str, FrozenSet[str]] = {}

    @staticmethod
    def _load(filename: Path) -> Dict[str, dict]:
        # utils.Ontology.load(filename, with_rels=True), statement for statement.
        ont: Dict[str, dict] = dict()
        obj = None
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if obj is not None:
                        ont[obj["id"]] = obj
                    obj = dict()
                    obj["is_a"] = list()
                    obj["part_of"] = list()
                    obj["regulates"] = list()
                    obj["alt_ids"] = list()
                    obj["is_obsolete"] = False
                    continue
                elif line == "[Typedef]":
                    if obj is not None:
                        ont[obj["id"]] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")  # noqa: E741 (DGZ's name)
                    if l[0] == "id":
                        obj["id"] = l[1]
                    elif l[0] == "alt_id":
                        obj["alt_ids"].append(l[1])
                    elif l[0] == "namespace":
                        obj["namespace"] = l[1]
                    elif l[0] == "is_a":
                        obj["is_a"].append(l[1].split(" ! ")[0])
                    elif l[0] == "relationship":
                        it = l[1].split()
                        # add all types of relationships
                        obj["is_a"].append(it[1])
                    elif l[0] == "name":
                        obj["name"] = l[1]
                    elif l[0] == "is_obsolete" and l[1] == "true":
                        obj["is_obsolete"] = True
            if obj is not None:
                ont[obj["id"]] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]["alt_ids"]:
                ont[t_id] = ont[term_id]
            if ont[term_id]["is_obsolete"]:
                del ont[term_id]
        return ont

    def closure(self, go_id: str) -> FrozenSet[str]:
        """``get_anchestors``: BFS over ``is_a`` (all relations), reflexive on the
        string passed in."""
        hit = self._cache.get(go_id)
        if hit is not None:
            return hit
        if go_id not in self.ont:
            res: FrozenSet[str] = frozenset()
        else:
            term_set: Set[str] = set()
            q = deque()
            q.append(go_id)
            while len(q) > 0:
                t_id = q.popleft()
                if t_id not in term_set:
                    term_set.add(t_id)
                    for parent_id in self.ont[t_id]["is_a"]:
                        if parent_id in self.ont:
                            q.append(parent_id)
            res = frozenset(term_set)
        self._cache[go_id] = res
        return res

    def prop(self, go_ids: Iterable[str]) -> FrozenSet[str]:
        """``get_prop_terms``: union of closures."""
        out: Set[str] = set()
        for g in go_ids:
            out |= self.closure(g)
        return frozenset(out)

    def has_term(self, go_id: str) -> bool:
        return go_id in self.ont

    def known_ids(self) -> List[str]:
        return list(self.ont.keys())


def is_go_id(token: str) -> bool:
    return bool(_GO_ID.match(token))
