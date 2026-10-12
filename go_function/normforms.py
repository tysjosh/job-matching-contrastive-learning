"""go.norm loader and the V1..V6 axiom ladder (design §6.4).

``load_normal_forms(dgz_compat=True)`` follows DGZ's ``load_normal_forms`` in
https://raw.githubusercontent.com/bio-ontology-research-group/deepgozero/main/
deepgozero.py (read with web_fetch): ``replace('_', ':')`` per line; NF1 when both
sides have length 10; NF2 if the left side contains "and"; NF3 if the left side
contains "some"; NF4 if the right side contains "some"; class ids assigned in
order of first appearance after the vocabulary columns; relation ids in order of
first appearance. The literal ``Nothing`` is mapped to ``owl:Nothing`` and
indexed in its first-appearance position, as DGZ's parser does, so V4 indices
match DGZ checkpoints. Added fatal checks (exit 2): the unparsed-line count,
the ``owl:Nothing`` line count and position, unknown relation IRIs and non-GO
class tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from go_function.constants import OWL_NOTHING, RELATION_IRIS
from go_function.ontology import is_go_id
from go_function.runtime import FatalError

PINNED_UNPARSED = 9
PINNED_NOTHING = 30

PART_OF = "BFO:0000050"
REGULATES_FAMILY = ("RO:0002211", "RO:0002212", "RO:0002213")
LADDER_RELATIONS = {
    "V1": (),
    "V2": (PART_OF,),
    "V3": (PART_OF, *REGULATES_FAMILY),
    "V3c": (PART_OF, *REGULATES_FAMILY),
    "V4": None,  # every relation
    "V6": None,
}
LADDER_NF2 = {"V1": False, "V2": False, "V3": False, "V3c": False, "V4": True, "V6": True}


class NormFormError(FatalError):
    pass


@dataclass(frozen=True)
class NormalForms:
    nf1: np.ndarray  # (n,2) c, d                 C SubClassOf D
    nf2: np.ndarray  # (n,3) c, d, e              C and D SubClassOf E
    nf3: np.ndarray  # (n,3) rel, c, d            R some C SubClassOf D
    nf4: np.ndarray  # (n,3) c, rel, d            C SubClassOf R some D
    class_index: Dict[str, int]
    rel_index: Dict[str, int]
    unparsed: int
    n_vocab: int
    nothing_lines: int = 0
    #: line number (0-based, file order) of every row, per form; drives re-indexing
    lines: Mapping[str, np.ndarray] = field(default_factory=dict)
    arm: str = "all"
    #: (n_nf4,) class index DGZ's NF4 formula reads as rc (``go_rad[data[:, 1]]``,
    #: protocol D10). It equals ``nf4[:, 1]`` except in V3c, which shares one
    #: relation vector but keeps V3's per-relation radius slots (amendment A5), so
    #: C1 = V3 - V3c differs only in relation-vector typing. None = ``nf4[:, 1]``.
    nf4_rad: Optional[np.ndarray] = None

    def counts(self) -> Dict[str, int]:
        return {"nf1": len(self.nf1), "nf2": len(self.nf2), "nf3": len(self.nf3),
                "nf4": len(self.nf4), "unparsed": self.unparsed,
                "nothing_lines": self.nothing_lines,
                "n_classes": len(self.class_index), "n_zero_classes":
                len(self.class_index) - self.n_vocab, "n_relations": len(set(self.rel_index.values()))}

    def class_ids(self) -> List[str]:
        out = [""] * len(self.class_index)
        for c, i in self.class_index.items():
            out[i] = c
        return out


def _cls(tok: str, ln: int) -> str:
    if tok == "Nothing":
        return OWL_NOTHING
    if not is_go_id(tok):
        raise NormFormError(f"go.norm line {ln + 1}: non-GO class token {tok!r}")
    return tok


def _rel(tok: str, ln: int) -> str:
    if tok not in RELATION_IRIS:
        raise NormFormError(f"go.norm line {ln + 1}: unknown relation IRI {tok!r}")
    return tok


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.int64)


def _arr(rows: List[Tuple[int, ...]], width: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64).reshape(-1, width) if rows else _empty(width)


def load_normal_forms(path: Path, terms_dict: Mapping[str, int], dgz_compat: bool = True,
                      expected_unparsed: Optional[int] = PINNED_UNPARSED,
                      expected_nothing: Optional[int] = PINNED_NOTHING) -> NormalForms:
    """Parse go.norm (design §6.4). ``terms_dict`` maps vocabulary ids to columns
    0..n-1. ``expected_*`` default to the G1 pins; ``None`` disables a check
    (fixtures only)."""
    if not dgz_compat:
        raise ValueError("only the DGZ-compatible parse order is defined (design §6.4)")
    n_vocab = len(terms_dict)
    if sorted(terms_dict.values()) != list(range(n_vocab)):
        raise NormFormError("terms_dict values must be 0..n-1")
    zclasses: Dict[str, int] = {}
    relations: Dict[str, int] = {}

    def get_index(go_id: str) -> int:
        if go_id in terms_dict:
            return terms_dict[go_id]
        if go_id not in zclasses:
            zclasses[go_id] = n_vocab + len(zclasses)
        return zclasses[go_id]

    def get_rel_index(rel_id: str) -> int:
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]

    rows: Dict[str, List[Tuple[int, ...]]] = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
    lines: Dict[str, List[int]] = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
    unparsed = 0
    nothing_lines = 0
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip().replace("_", ":")
            has_nothing = "Nothing" in line.split()
            if line.find("SubClassOf") == -1:
                if has_nothing:
                    raise NormFormError(f"go.norm line {ln + 1}: Nothing outside an NF2 RHS")
                unparsed += 1
                continue
            left, right = line.split(" SubClassOf ")
            if len(left) == 10 and len(right) == 10:
                form = "nf1"
                c, d = _cls(left, ln), _cls(right, ln)
                row = (get_index(c), get_index(d))
            elif left.find("and") != -1:
                form = "nf2"
                go1, go2 = left.split(" and ")
                c, d, e = _cls(go1, ln), _cls(go2, ln), _cls(right, ln)
                row = (get_index(c), get_index(d), get_index(e))
            elif left.find("some") != -1:
                form = "nf3"
                rel, go1 = left.split(" some ")
                r = _rel(rel, ln)
                c, d = _cls(go1, ln), _cls(right, ln)
                row = (get_rel_index(r), get_index(c), get_index(d))
            elif right.find("some") != -1:
                form = "nf4"
                rel, go2 = right.split(" some ")
                c = _cls(left, ln)
                r = _rel(rel, ln)
                d = _cls(go2, ln)
                row = (get_index(c), get_rel_index(r), get_index(d))
            else:
                # DGZ silently skips such a line; here it can only hold a non-GO or
                # malformed class token, which is fatal (design §6.4).
                raise NormFormError(f"go.norm line {ln + 1}: SubClassOf axiom matches no "
                                    f"normal form (non-GO token?): {line!r}")
            if has_nothing:
                if not (form == "nf2" and right == "Nothing" and "Nothing" not in left.split()):
                    raise NormFormError(f"go.norm line {ln + 1}: Nothing outside an NF2 RHS")
                nothing_lines += 1
            rows[form].append(row)
            lines[form].append(ln)
    if expected_unparsed is not None and unparsed != expected_unparsed:
        raise NormFormError(f"go.norm: {unparsed} unparsed lines, pinned {expected_unparsed}")
    if expected_nothing is not None and nothing_lines != expected_nothing:
        raise NormFormError(f"go.norm: Nothing in {nothing_lines} lines, pinned {expected_nothing}")
    class_index = dict(terms_dict)
    class_index.update(zclasses)
    return NormalForms(nf1=_arr(rows["nf1"], 2), nf2=_arr(rows["nf2"], 3),
                       nf3=_arr(rows["nf3"], 3), nf4=_arr(rows["nf4"], 3),
                       class_index=class_index, rel_index=dict(relations), unparsed=unparsed,
                       n_vocab=n_vocab, nothing_lines=nothing_lines,
                       lines={k: np.asarray(v, dtype=np.int64) for k, v in lines.items()})


# ------------------------------------------------------------------ ladder
def _keep_rows(nf: NormalForms, arm: str) -> Dict[str, np.ndarray]:
    """Boolean row selections per form for ``arm``."""
    rels = LADDER_RELATIONS[arm]
    inv_rel = {i: r for r, i in nf.rel_index.items()}
    keep_rel = (lambda idx: True) if rels is None else (lambda idx: inv_rel[int(idx)] in rels)
    return {
        "nf1": np.ones(len(nf.nf1), dtype=bool),
        "nf2": np.full(len(nf.nf2), LADDER_NF2[arm], dtype=bool),
        "nf3": np.asarray([keep_rel(r) for r in nf.nf3[:, 0]], dtype=bool),
        "nf4": np.asarray([keep_rel(r) for r in nf.nf4[:, 1]], dtype=bool),
    }


#: positions of class and relation columns per form, in DGZ's get_index call order
_CLASS_COLS = {"nf1": (0, 1), "nf2": (0, 1, 2), "nf3": (1, 2), "nf4": (0, 2)}
_REL_COL = {"nf3": 0, "nf4": 1}
_CALL_ORDER = {"nf1": ("c", "c"), "nf2": ("c", "c", "c"), "nf3": ("r", "c", "c"),
               "nf4": ("c", "r", "c")}


def select(nf: NormalForms, arm: str) -> NormalForms:
    """The axiom subset of ``arm`` (V1, V2, V3, V3c, V4, V6), with zero classes and
    relations re-indexed in first-appearance (file) order within the subset.
    V3c maps part_of and the regulates family to one shared relation vector."""
    if arm not in LADDER_RELATIONS:
        raise ValueError(f"arm {arm!r} has no axioms; ladder arms are {sorted(LADDER_RELATIONS)}")
    keep = _keep_rows(nf, arm)
    inv_cls = nf.class_ids()
    inv_rel = {i: r for r, i in nf.rel_index.items()}
    n_vocab = nf.n_vocab
    events = []  # (line, form, row index)
    for form in ("nf1", "nf2", "nf3", "nf4"):
        for i in np.flatnonzero(keep[form]):
            events.append((int(nf.lines[form][i]), form, int(i)))
    events.sort()
    class_index: Dict[str, int] = {c: i for i, c in enumerate(inv_cls[:n_vocab])}
    rel_index: Dict[str, int] = {}
    untied: Dict[str, int] = {}          # per-relation ids as without sharing (= V3's)
    shared = arm == "V3c"
    out_rad: List[int] = []
    out_rows: Dict[str, List[Tuple[int, ...]]] = {k: [] for k in _CALL_ORDER}
    out_lines: Dict[str, List[int]] = {k: [] for k in _CALL_ORDER}

    def cidx(old: int) -> int:
        c = inv_cls[old]
        if c not in class_index:
            class_index[c] = len(class_index)
        return class_index[c]

    def ridx(old: int) -> int:
        r = inv_rel[old]
        if r not in untied:
            untied[r] = len(untied)
        if shared:  # every selected relation shares vector 0
            rel_index[r] = 0
            return 0
        if r not in rel_index:
            rel_index[r] = len(rel_index)
        return rel_index[r]

    for ln, form, i in events:
        old = getattr(nf, form)[i]
        new = []
        for pos, kind in enumerate(_CALL_ORDER[form]):
            new.append(ridx(int(old[pos])) if kind == "r" else cidx(int(old[pos])))
        out_rows[form].append(tuple(new))
        out_lines[form].append(ln)
        if form == "nf4":
            out_rad.append(untied[inv_rel[int(old[1])]])
    return NormalForms(nf1=_arr(out_rows["nf1"], 2), nf2=_arr(out_rows["nf2"], 3),
                       nf3=_arr(out_rows["nf3"], 3), nf4=_arr(out_rows["nf4"], 3),
                       class_index=class_index, rel_index=rel_index, unparsed=nf.unparsed,
                       n_vocab=n_vocab,
                       nothing_lines=nf.nothing_lines if LADDER_NF2[arm] else 0,
                       lines={k: np.asarray(v, dtype=np.int64) for k, v in out_lines.items()},
                       arm=arm, nf4_rad=np.asarray(out_rad, dtype=np.int64))


def _axiom_keys(nf: NormalForms) -> set:
    return {(form, int(ln)) for form, arr in nf.lines.items() for ln in arr}


def assert_ladder(nf: NormalForms) -> Dict[str, Dict[str, int]]:
    """Fatal unless V1 ⊂ V2 ⊂ V3 = V3c ⊂ V4 = V6 (as axiom sets), V1..V3c carry
    no owl:Nothing, V3c has one shared relation, and V4 reproduces the DGZ
    indices. Returns per-arm counts (pinned at G1-norm)."""
    arms = {a: select(nf, a) for a in ("V1", "V2", "V3", "V3c", "V4", "V6")}
    keys = {a: _axiom_keys(s) for a, s in arms.items()}
    chain = [("V1", "V2"), ("V2", "V3"), ("V3", "V4")]
    for lo, hi in chain:
        if not keys[lo] <= keys[hi]:
            raise NormFormError(f"ladder not nested: {lo} ⊄ {hi}")
    if keys["V3"] != keys["V3c"] or keys["V4"] != keys["V6"]:
        raise NormFormError("V3c must equal V3 and V6 must equal V4 as axiom sets")
    for a in ("V1", "V2", "V3", "V3c"):
        if OWL_NOTHING in arms[a].class_index:
            raise NormFormError(f"{a} contains owl:Nothing")
    if len(set(arms["V3c"].rel_index.values())) > 1:
        raise NormFormError("V3c must have one shared relation vector")
    if not np.array_equal(arms["V3c"].nf4_rad, arms["V3"].nf4_rad):
        raise NormFormError("V3c must keep V3's NF4 radius slots (amendment A5)")
    for a, s in arms.items():
        if a != "V3c" and not np.array_equal(s.nf4_rad, s.nf4[:, 1]):
            raise NormFormError(f"{a}: NF4 radius index differs from its relation ids")
    v4 = arms["V4"]
    if (v4.class_index != nf.class_index or v4.rel_index != nf.rel_index or
            any(not np.array_equal(getattr(v4, f), getattr(nf, f))
                for f in ("nf1", "nf2", "nf3", "nf4"))):
        raise NormFormError("V4 does not reproduce the DGZ parse order")
    return {a: s.counts() for a, s in arms.items()}
