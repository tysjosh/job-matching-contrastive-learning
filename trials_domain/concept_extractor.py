"""Deterministic MeSH concept extraction from free-text patient narratives.

The TREC topics carry **no ontology annotation** — they are raw clinical prose.
The trials side is annotated (``<mesh_term>`` under ``condition_browse`` /
``intervention_browse``), so to compute any ontology signal between a patient and
a trial we must first ground the narrative in MeSH descriptors.

Why longest-match string lookup rather than an embedding model
-------------------------------------------------------------
ORCA guarantees bit-for-bit reproducibility from ``config.training_seed`` alone
(Requirement 10.1/10.2: no unseeded or time-based source). A neural concept
extractor would put a second, separately-versioned model in the data path and
make the ontology features a function of that model's weights. Longest-match
lookup against the MeSH entry vocabulary is a pure function of the narrative and
``desc2021.gz``, so the extracted concepts are a fixed property of the dataset
and can be materialized once into the converted JSONL.

The tradeoff is recall: this finds terms that appear more or less verbatim and
misses paraphrase and abbreviation not present in the 252k-term entry vocabulary.
That is an acceptable, and honestly reportable, floor.

Noise control
-------------
The MeSH entry vocabulary contains many terms that are also ordinary English
words (``Male``, ``Female``, ``Syndrome``, ``Pain``, ``Report``, ``Persons``).
Matching them indiscriminately would attach a near-identical concept set to every
topic and destroy the signal. Three guards, in order of importance:

1. **Longest-match, non-overlapping.** Scan n-grams from
   :data:`MAX_NGRAM` words down to 1; once a span matches it is consumed, so
   ``"congenital adrenal hyperplasia"`` wins over ``"hyperplasia"``.
2. **Single-token blocklist** (:data:`GENERIC_SINGLE_TOKENS`) for unigrams that
   are common prose but valid MeSH entry terms. Multi-word matches are never
   blocked — ``"adrenal hyperplasia"`` is specific even though ``"hyperplasia"``
   alone would be dropped in isolation only if listed.
3. **Minimum length** of :data:`MIN_TERM_CHARS` characters for unigram matches,
   which removes acronym collisions like ``"a"``, ``"of"``, ``"ii"``.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from trials_domain.mesh_ontology import MeshIndex, normalize_term

logger = logging.getLogger(__name__)

#: Longest n-gram considered. MeSH preferred names run long
#: ("Diabetes Mellitus, Type 2, Susceptibility To"), but beyond ~6 words the
#: chance of a verbatim match in clinical prose is negligible and the scan cost
#: grows linearly.
MAX_NGRAM = 6

#: Minimum characters for a *unigram* match to be accepted.
MIN_TERM_CHARS = 4

#: Unigrams that are valid MeSH entry terms but function as ordinary prose in a
#: patient narrative. Matching these would attach the same handful of descriptors
#: to nearly every topic. Multi-word matches containing these words are still
#: accepted — only the bare unigram is suppressed.
GENERIC_SINGLE_TOKENS: Set[str] = {
    # demographics / person words (also the two treeless descriptors)
    "male", "female", "man", "woman", "men", "women", "boy", "girl",
    "adult", "adults", "child", "children", "infant", "patients", "patient",
    "persons", "person", "human", "humans", "age", "aged",
    # generic clinical scaffolding
    "syndrome", "disease", "diseases", "disorder", "disorders", "history",
    "symptoms", "signs", "pain", "fever", "mass", "lesion", "therapy",
    "treatment", "treatments", "therapeutics", "diagnosis", "prognosis",
    "surgery", "medicine", "drug", "drugs", "dose", "doses", "injections",
    "tablet", "tablets", "solution", "solutions", "water", "salts", "ions",
    # process / measurement words
    "time", "weight", "growth", "pressure", "temperature", "volume", "rate",
    "levels", "level", "size", "color", "light", "sound", "motion", "work",
    "risk", "safety", "quality", "control", "records", "record", "report",
    "reports", "research", "methods", "review", "role", "state", "status",
    "family", "mother", "father", "parents", "hospitals", "clinics",
    "examination", "physical examination", "evaluation", "test", "tests",
    "wounds", "injuries", "recovery", "response", "phase", "stage",
}

#: Word-boundary tokenizer that keeps intra-word hyphens and apostrophes so
#: "T-cell" and "Crohn's" survive as single tokens.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

#: De-identification markers used by the TREC topics, e.g. ``[**2148-10-1**]``.
#: Stripped before tokenization so they cannot produce spurious matches.
_DEID = re.compile(r"\[\*\*.*?\*\*\]")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens with de-identification markers removed."""
    cleaned = _DEID.sub(" ", text or "")
    return [t.lower() for t in _TOKEN.findall(cleaned)]


def _acceptable(term: str, n_words: int) -> bool:
    """Whether a matched surface form passes the noise guards."""
    if n_words > 1:
        return True
    if term in GENERIC_SINGLE_TOKENS:
        return False
    return len(term) >= MIN_TERM_CHARS


def extract_concepts(
    text: str,
    index: MeshIndex,
    max_ngram: int = MAX_NGRAM,
    restrict_branches: Optional[Sequence[str]] = None,
) -> List[str]:
    """Extract MeSH descriptor UIs from ``text`` by longest non-overlapping match.

    Args:
        text: Free-text narrative.
        index: A built :class:`MeshIndex` supplying the entry vocabulary.
        max_ngram: Longest n-gram to consider.
        restrict_branches: Optional MeSH tree prefixes to keep. Accepts bare
            branch letters (``"C"``) and deeper subtree prefixes (``"F03"``), so
            a caller can admit Mental Disorders without admitting the Behavior
            and Psychological Phenomena subtrees that share the F letter. A
            descriptor is kept when *any* of its positions matches *any* prefix.
            ``None`` keeps all.

    Returns:
        Descriptor UIs in first-occurrence order, deduplicated. Deterministic for
        a given ``text`` and ``index``.
    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    n = len(tokens)
    consumed = [False] * n
    found: List[Tuple[int, str]] = []

    keep = tuple(restrict_branches) if restrict_branches else None

    # Longest-first so specific multi-word concepts win over their head nouns.
    for size in range(min(max_ngram, n), 0, -1):
        for start in range(0, n - size + 1):
            if any(consumed[start : start + size]):
                continue
            surface = " ".join(tokens[start : start + size])
            ui = index.term_to_ui.get(normalize_term(surface))
            if ui is None:
                continue
            if not _acceptable(surface, size):
                continue
            if keep is not None and not index.in_subtree(ui, keep):
                continue
            for i in range(start, start + size):
                consumed[i] = True
            found.append((start, ui))

    # Order by position of first occurrence, deduplicating.
    out: List[str] = []
    seen: Set[str] = set()
    for _pos, ui in sorted(found, key=lambda pair: pair[0]):
        if ui not in seen:
            seen.add(ui)
            out.append(ui)
    return out


def extraction_report(
    texts: Iterable[str], index: MeshIndex, **kwargs
) -> Dict[str, float]:
    """Summary statistics for extraction over a set of texts (diagnostics only).

    Returns counts and per-text concept yield so a converter run can assert the
    extractor is actually finding concepts before the data is used for training.
    """
    counts: List[int] = []
    branch_hits: Dict[str, int] = {}
    for text in texts:
        uis = extract_concepts(text, index, **kwargs)
        counts.append(len(uis))
        for ui in uis:
            for letter in index.branches(ui):
                branch_hits[letter] = branch_hits.get(letter, 0) + 1

    if not counts:
        return {"texts": 0}
    return {
        "texts": len(counts),
        "total_concepts": sum(counts),
        "mean_per_text": sum(counts) / len(counts),
        "min_per_text": min(counts),
        "max_per_text": max(counts),
        "texts_with_zero": sum(1 for c in counts if c == 0),
        "branch_distribution": dict(
            sorted(branch_hits.items(), key=lambda kv: -kv[1])
        ),
    }
