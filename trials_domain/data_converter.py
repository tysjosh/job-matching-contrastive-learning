"""Convert the raw TREC Clinical Trials release into CDCL-ready JSONL.

Reads the local read-only dataset under ``trec-clinical-trials/`` and emits three
artifacts:

  * ``trials.jsonl``  — one record per *judged* trial, with text views, structured
    eligibility, and resolved MeSH descriptor UIs.
  * ``topics.jsonl``  — one record per patient topic, with the narrative and MeSH
    descriptors grounded by :mod:`trials_domain.concept_extractor`.
  * ``qrels.jsonl``   — the graded judgments as ``{topic, nct_id, grade}``.

Only judged trials are extracted by default. The qrels reference 48,714 distinct
NCT ids out of a 375,580-trial corpus (12.97%), so restricting to judged trials
cuts the XML parse by ~87% while losing nothing needed for the primary
judged-only negative-sampling regime. ``--include-unjudged`` additionally emits
an unjudged pool for the corpus-wide false-negative ablation.

Branch assignment
-----------------
Topic-side concepts are extracted into two sets that mirror the trials side's own
two annotation facets, so the ontology features compare like with like:

  * ``condition_uris``    <- MeSH branches C (Diseases) + F (Psychiatry)
    against the trial's ``condition_browse`` terms
  * ``intervention_uris`` <- MeSH branches D (Chemicals/Drugs) + E (Techniques)
    against the trial's ``intervention_browse`` terms

Branches A (Anatomy), B (Organisms), G (Phenomena) and the rest are deliberately
excluded from the topic side: they are either weak signal ("Spine", "Knee") or
actively harmful. ``"spinal cord conus mass"`` matches the *Conus Snail* genus in
branch B, which is exactly the kind of false concept that would corrupt the
distance geometry.

Known ontology gap
------------------
``desc2021.gz`` ships MeSH **descriptors** only. Many drug names in
``intervention_browse`` are **Supplementary Concept Records**, which live in a
separate ``supp2021.gz`` the release does not include. Measured resolution on a
4,000-trial sample: condition terms **100%**, intervention terms **83.6%**
(``Gemcitabine``, ``Fludarabine``, ``Sargramostim``, ``Liposomal doxorubicin``
and 280 other distinct strings are unresolvable). Unresolved terms are counted
and reported, never silently dropped without a tally. Adding ``supp2021.gz``
would close this gap; until then the intervention-side signal is
descriptor-backed only, which skews toward older generic agents.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from trials_domain.concept_extractor import extract_concepts
from trials_domain.mesh_ontology import MeshIndex

logger = logging.getLogger(__name__)

#: Default dataset root (read-only source).
DEFAULT_DATASET_ROOT = Path("trec-clinical-trials")

#: MeSH tree prefixes used for the topic-side *condition* concept set.
#:
#: ``C`` is Diseases. ``F03`` is Mental Disorders specifically — **not** the whole
#: ``F`` letter. F01 (Behavior and Behavior Mechanisms) and F02 (Psychological
#: Phenomena) contain descriptors whose entry terms are ordinary clinical prose,
#: and admitting them produced measurably wrong concept sets: a topic describing
#: hypogonadotropic hypogonadism resolved to ``Personal Satisfaction`` (F01.145),
#: ``Smell`` (F02.830) and ``Visual Acuity`` (F02.463) — three surface-word
#: matches and not one correct diagnosis. Restricting to F03 keeps psychiatric
#: conditions, which are legitimate trial targets, without that noise.
CONDITION_BRANCHES: Tuple[str, ...] = ("C", "F03")

#: MeSH tree prefixes used for the topic-side *intervention* concept set.
#: ``D`` is Chemicals and Drugs, ``E`` is Analytical/Diagnostic/Therapeutic
#: Techniques — the two branches the corpus's own ``intervention_browse``
#: annotations draw from.
INTERVENTION_BRANCHES: Tuple[str, ...] = ("D", "E")

#: Official qrels grades.
GRADE_NOT_RELEVANT = 0
GRADE_INELIGIBLE = 1
GRADE_ELIGIBLE = 2

#: Human-readable grade names carried into the converted records so downstream
#: code never has to remember the numeric convention.
GRADE_NAMES = {
    GRADE_NOT_RELEVANT: "not_relevant",
    GRADE_INELIGIBLE: "ineligible",
    GRADE_ELIGIBLE: "eligible",
}

_WS = re.compile(r"\s+")
_AGE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(year|month|week|day|hour|minute)", re.I)

#: Multipliers converting a parsed eligibility age unit into years.
_AGE_TO_YEARS = {
    "year": 1.0,
    "month": 1.0 / 12.0,
    "week": 1.0 / 52.0,
    "day": 1.0 / 365.0,
    "hour": 1.0 / (365.0 * 24.0),
    "minute": 1.0 / (365.0 * 24.0 * 60.0),
}


def _clean(text: Optional[str]) -> str:
    """Collapse whitespace and strip; ``""`` for ``None``."""
    if not text:
        return ""
    return _WS.sub(" ", text).strip()


def parse_age(raw: Optional[str]) -> Optional[float]:
    """Parse an eligibility age (``"14 Years"``, ``"6 Months"``) into years.

    Returns ``None`` for ``"N/A"``, empty, or unparseable values — the trials
    corpus uses ``"N/A"`` to mean "no bound", which is genuinely different from
    zero and must not be coerced to it.
    """
    text = _clean(raw)
    if not text or text.upper() == "N/A":
        return None
    match = _AGE.match(text)
    if not match:
        return None
    value, unit = match.group(1), match.group(2).lower()
    return float(value) * _AGE_TO_YEARS[unit]


@dataclass
class ConversionStats:
    """Counters for a converter run, reported at the end and easy to assert on."""

    trials_seen: int = 0
    trials_emitted: int = 0
    trials_missing: int = 0
    unjudged_emitted: int = 0
    condition_terms: int = 0
    condition_resolved: int = 0
    intervention_terms: int = 0
    intervention_resolved: int = 0
    unresolved: collections.Counter = field(default_factory=collections.Counter)
    trials_without_any_uri: int = 0
    topics_emitted: int = 0
    topics_without_condition_uri: int = 0
    topics_without_any_uri: int = 0
    #: Topic ids whose condition set came back empty, so ``branch_distance``
    #: degrades to the neutral 0.5. Recorded by id rather than counted only,
    #: because these are systematically the diagnostic-vignette topics (the
    #: narrative describes findings without naming the disease) and any
    #: per-topic result breakdown needs to know which ones they are.
    degraded_topics: List[str] = field(default_factory=list)

    def resolution_summary(self) -> Dict[str, Any]:
        """Resolution rates per facet plus the worst unresolved offenders."""

        def rate(hit: int, total: int) -> Optional[float]:
            return (hit / total) if total else None

        return {
            "condition_resolution": rate(self.condition_resolved, self.condition_terms),
            "intervention_resolution": rate(
                self.intervention_resolved, self.intervention_terms
            ),
            "distinct_unresolved": len(self.unresolved),
            "top_unresolved": self.unresolved.most_common(15),
        }


# --------------------------------------------------------------------- loading
def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """Load ``topic iteration NCT_ID grade`` into ``{topic: {nct_id: grade}}``.

    Raises:
        ValueError: On a malformed line, naming the file and line number, so a
            truncated download fails loudly rather than silently shrinking the
            judgment set.
    """
    out: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected 4 whitespace-separated "
                    f"fields, got {len(parts)}: {line!r}"
                )
            topic, _iteration, nct_id, grade = parts
            try:
                out[topic][nct_id] = int(grade)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: non-integer grade {grade!r}"
                ) from exc
    return dict(out)


def load_topics(path: Path) -> List[Tuple[str, str]]:
    """Load ``(topic_number, narrative)`` pairs from a topics XML file.

    Parsed with a regex rather than ElementTree on purpose: the 2022 file's root
    element is mislabeled ``task="2021 TREC Clinical Trials"`` upstream and the
    narratives contain raw entities. A tolerant extraction avoids coupling to
    upstream markup quirks that carry no information we need.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    found = re.findall(r'<topic\s+number="([^"]+)"\s*>(.*?)</topic>', raw, re.S)
    if not found:
        raise ValueError(f"{path}: no <topic number=...> elements found")
    topics: List[Tuple[str, str]] = []
    for number, body in found:
        # Unescape the handful of entities the topics actually use.
        text = (
            body.replace("&quot;", '"')
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&apos;", "'")
        )
        topics.append((number.strip(), _clean(text)))
    return topics


# ------------------------------------------------------------- trial XML parse
def _mesh_terms(root: ET.Element, container: str) -> List[str]:
    """MeSH term strings under ``condition_browse`` / ``intervention_browse``."""
    node = root.find(container)
    if node is None:
        return []
    return [_clean(t.text) for t in node.findall("mesh_term") if _clean(t.text)]


def parse_trial(xml_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Parse one ``NCT********.xml`` member into a flat dict.

    Returns ``None`` when the document has no ``nct_id`` (never observed in this
    snapshot, but a malformed member must not abort a 375k-file stream).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    nct_id = _clean(root.findtext("id_info/nct_id"))
    if not nct_id:
        return None

    eligibility = root.find("eligibility")

    def elig(tag: str) -> str:
        return _clean(eligibility.findtext(tag)) if eligibility is not None else ""

    interventions = [
        {
            "type": _clean(node.findtext("intervention_type")),
            "name": _clean(node.findtext("intervention_name")),
        }
        for node in root.findall("intervention")
    ]

    return {
        "nct_id": nct_id,
        "brief_title": _clean(root.findtext("brief_title")),
        "official_title": _clean(root.findtext("official_title")),
        "brief_summary": _clean(root.findtext("brief_summary/textblock")),
        "detailed_description": _clean(
            root.findtext("detailed_description/textblock")
        ),
        "conditions": [_clean(c.text) for c in root.findall("condition") if _clean(c.text)],
        "overall_status": _clean(root.findtext("overall_status")),
        "phase": _clean(root.findtext("phase")),
        "study_type": _clean(root.findtext("study_type")),
        "eligibility_criteria": elig("criteria/textblock"),
        "gender": elig("gender"),
        "minimum_age_raw": elig("minimum_age"),
        "maximum_age_raw": elig("maximum_age"),
        "minimum_age_years": parse_age(elig("minimum_age")),
        "maximum_age_years": parse_age(elig("maximum_age")),
        "healthy_volunteers": elig("healthy_volunteers"),
        "interventions": interventions,
        "condition_mesh_terms": _mesh_terms(root, "condition_browse"),
        "intervention_mesh_terms": _mesh_terms(root, "intervention_browse"),
    }


def stream_corpus(
    zip_paths: Sequence[Path],
    wanted: Optional[Set[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield parsed trial dicts from the corpus archives.

    Archives stay compressed; members are decompressed one at a time. When
    ``wanted`` is given, a member is only decompressed if its filename stem is in
    the set — the filename encodes the NCT id, so this skips ~87% of the corpus
    without paying to parse it.
    """
    for zip_path in zip_paths:
        if not zip_path.exists():
            raise FileNotFoundError(f"corpus archive not found: {zip_path}")
        logger.info("Streaming %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if not name.endswith(".xml"):
                    continue
                if wanted is not None:
                    stem = name.rsplit("/", 1)[-1][: -len(".xml")]
                    if stem not in wanted:
                        continue
                record = parse_trial(archive.read(name))
                if record is not None:
                    yield record


# ------------------------------------------------------- views and enrichment
#: Character budget for the eligibility criteria inside the encoder view.
#: Criteria blocks run to several thousand characters; sentence-transformer
#: backbones truncate at 256-512 word pieces anyway, so an unbounded criteria
#: block would push the title and conditions out of the encoder's window. The
#: budget keeps the discriminative header intact and is applied at a whitespace
#: boundary so no word is split.
CRITERIA_CHAR_BUDGET = 1200

#: Character budget for the brief summary inside the encoder view.
SUMMARY_CHAR_BUDGET = 800


def _truncate(text: str, budget: int) -> str:
    """Truncate at the last whitespace before ``budget`` characters."""
    if len(text) <= budget:
        return text
    cut = text.rfind(" ", 0, budget)
    return text[: cut if cut > 0 else budget].rstrip()


def build_trial_view(trial: Dict[str, Any]) -> str:
    """Pre-serialize a trial into the single text the encoder sees.

    Emitting a pre-built ``encoder_view`` is what lets the trials domain bypass
    the career-specific ``'resume'``/``'job'`` text branches in
    ``trainer._encode_content_to_text_embedding`` and
    ``BatchEfficientEncoder._content_to_text``: both check for ``encoder_view``
    before falling through to role/experience/title/description handling.

    Field order is deliberate — title and conditions first, since they carry the
    topical signal, then the structured eligibility bounds, then criteria prose.
    Under encoder truncation the most discriminative content survives.
    """
    parts: List[str] = []
    title = trial.get("brief_title") or trial.get("official_title") or ""
    if title:
        parts.append(f"Trial: {title}")
    if trial.get("conditions"):
        parts.append("Conditions: " + "; ".join(trial["conditions"]))
    if trial.get("interventions"):
        names = [i["name"] for i in trial["interventions"] if i.get("name")]
        if names:
            parts.append("Interventions: " + "; ".join(dict.fromkeys(names)))

    bounds: List[str] = []
    if trial.get("gender"):
        bounds.append(f"sex {trial['gender']}")
    if trial.get("minimum_age_raw") and trial["minimum_age_raw"].upper() != "N/A":
        bounds.append(f"min age {trial['minimum_age_raw']}")
    if trial.get("maximum_age_raw") and trial["maximum_age_raw"].upper() != "N/A":
        bounds.append(f"max age {trial['maximum_age_raw']}")
    if trial.get("healthy_volunteers"):
        bounds.append(f"healthy volunteers {trial['healthy_volunteers']}")
    if bounds:
        parts.append("Eligibility: " + ", ".join(bounds))

    if trial.get("brief_summary"):
        parts.append(
            "Summary: " + _truncate(trial["brief_summary"], SUMMARY_CHAR_BUDGET)
        )
    if trial.get("eligibility_criteria"):
        parts.append(
            "Criteria: " + _truncate(trial["eligibility_criteria"], CRITERIA_CHAR_BUDGET)
        )
    return "\n".join(parts)


def resolve_trial_uris(
    trial: Dict[str, Any], index: MeshIndex, stats: ConversionStats
) -> Dict[str, List[str]]:
    """Resolve a trial's MeSH term strings to descriptor UIs, tallying misses.

    The two facets stay separate: ``condition_uris`` feeds the coarse
    ``branch_distance`` (the ``d_isco`` analogue) while the union of both feeds
    the fine-grained set similarity (the ``s_esco`` analogue). Keeping them apart
    is what decorrelates those two ORCA features — computing both over the same
    input set would make them near-collinear and waste two of the five slots in
    the ReliabilityMLP's feature vector.
    """
    condition_uris: List[str] = []
    for term in trial.get("condition_mesh_terms", []):
        stats.condition_terms += 1
        ui = index.resolve(term)
        if ui:
            stats.condition_resolved += 1
            if ui not in condition_uris:
                condition_uris.append(ui)
        else:
            stats.unresolved[f"condition:{term}"] += 1

    intervention_uris: List[str] = []
    for term in trial.get("intervention_mesh_terms", []):
        stats.intervention_terms += 1
        ui = index.resolve(term)
        if ui:
            stats.intervention_resolved += 1
            if ui not in intervention_uris:
                intervention_uris.append(ui)
        else:
            stats.unresolved[f"intervention:{term}"] += 1

    return {"condition_uris": condition_uris, "intervention_uris": intervention_uris}


def build_trial_record(
    trial: Dict[str, Any], index: MeshIndex, stats: ConversionStats
) -> Dict[str, Any]:
    """Assemble the emitted ``trials.jsonl`` record for one parsed trial."""
    uris = resolve_trial_uris(trial, index, stats)
    all_uris = list(dict.fromkeys(uris["condition_uris"] + uris["intervention_uris"]))
    if not all_uris:
        stats.trials_without_any_uri += 1

    return {
        "nct_id": trial["nct_id"],
        "encoder_view": build_trial_view(trial),
        "title": trial["brief_title"] or trial["official_title"],
        "conditions": trial["conditions"],
        "overall_status": trial["overall_status"],
        "phase": trial["phase"],
        "study_type": trial["study_type"],
        # Structured eligibility — the non-ontological axis that actually
        # separates grade 1 from grade 2. MeSH similarity is close to blind to
        # this distinction, so these fields are the trials-domain analogue of the
        # career domain's structured experience-level features.
        "eligibility": {
            "gender": trial["gender"],
            "minimum_age_years": trial["minimum_age_years"],
            "maximum_age_years": trial["maximum_age_years"],
            "healthy_volunteers": trial["healthy_volunteers"],
            "criteria": trial["eligibility_criteria"],
        },
        # Ontology facets, kept separate by design (see resolve_trial_uris).
        "condition_uris": uris["condition_uris"],
        "intervention_uris": uris["intervention_uris"],
        "mesh_uris": all_uris,
        "condition_mesh_terms": trial["condition_mesh_terms"],
        "intervention_mesh_terms": trial["intervention_mesh_terms"],
    }


def build_topic_record(
    number: str,
    text: str,
    index: MeshIndex,
    stats: ConversionStats,
    topic_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the emitted ``topics.jsonl`` record for one patient topic.

    ``topic_id`` is the year-namespaced identifier (``"2022_1"``). It is passed in
    rather than derived because topic numbers restart at 1 each year: recording a
    bare ``"1"`` in the degraded-topic list would be ambiguous between 2021 and
    2022.
    """
    condition_uris = extract_concepts(
        text, index, restrict_branches=CONDITION_BRANCHES
    )
    intervention_uris = extract_concepts(
        text, index, restrict_branches=INTERVENTION_BRANCHES
    )
    all_uris = list(dict.fromkeys(condition_uris + intervention_uris))
    if not condition_uris:
        stats.topics_without_condition_uri += 1
        stats.degraded_topics.append(topic_id or number)
    if not all_uris:
        stats.topics_without_any_uri += 1

    return {
        "topic_id": number,
        # True when the coarse condition signal is unavailable for this topic.
        # ORCA's WeakTargetBuilder already drops absent signals and renormalizes
        # the remaining blend weights, so this degrades by design rather than
        # being silently filled with a fabricated concept.
        "condition_signal_present": bool(condition_uris),
        "encoder_view": text,
        "narrative": text,
        "condition_uris": condition_uris,
        "intervention_uris": intervention_uris,
        "mesh_uris": all_uris,
        "condition_terms": [index.name(u) for u in condition_uris],
        "intervention_terms": [index.name(u) for u in intervention_uris],
    }


# ------------------------------------------------------------------ orchestration
def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write ``records`` as JSONL, returning the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert(
    dataset_root: Path,
    output_dir: Path,
    mesh_cache: Optional[Path] = None,
    include_unjudged: int = 0,
    seed: int = 42,
) -> ConversionStats:
    """Convert the raw release into ``trials.jsonl`` / ``topics.jsonl`` / ``qrels.jsonl``.

    Args:
        dataset_root: The ``trec-clinical-trials`` directory.
        output_dir: Destination for the converted JSONL.
        mesh_cache: Optional pickle path for the parsed MeSH index.
        include_unjudged: If > 0, additionally emit that many unjudged trials to
            ``unjudged_pool.jsonl`` for the corpus-wide false-negative ablation.
        seed: Seed for the unjudged-pool subsample (reservoir sampling), so the
            pool is reproducible.

    Returns:
        A :class:`ConversionStats` with per-facet resolution rates.
    """
    raw = dataset_root / "raw"
    stats = ConversionStats()

    index = MeshIndex.build_or_load(raw / "ontology" / "desc2021.gz", mesh_cache)

    # ---- qrels + the judged NCT set -------------------------------------
    qrels_records: List[Dict[str, Any]] = []
    wanted: Set[str] = set()
    for year in ("2021", "2022"):
        qrels = load_qrels(raw / year / f"qrels{year}.txt")
        for topic, judgments in qrels.items():
            for nct_id, grade in judgments.items():
                wanted.add(nct_id)
                qrels_records.append(
                    {
                        "year": year,
                        # Topic ids collide across years (both start at 1), so the
                        # emitted key is namespaced. Without this, 2021 topic 1 and
                        # 2022 topic 1 would merge into one query group and their
                        # judgments would silently contradict each other.
                        "topic_id": f"{year}_{topic}",
                        "raw_topic_id": topic,
                        "nct_id": nct_id,
                        "grade": grade,
                        "grade_name": GRADE_NAMES.get(grade, "unknown"),
                    }
                )
    logger.info(
        "Loaded %d judgments over %d distinct trials",
        len(qrels_records),
        len(wanted),
    )

    # ---- topics ----------------------------------------------------------
    topic_records: List[Dict[str, Any]] = []
    for year in ("2021", "2022"):
        for number, text in load_topics(raw / year / f"topics{year}.xml"):
            topic_id = f"{year}_{number}"
            record = build_topic_record(number, text, index, stats, topic_id=topic_id)
            record["year"] = year
            record["raw_topic_id"] = number
            record["topic_id"] = topic_id
            topic_records.append(record)
    stats.topics_emitted = _write_jsonl(output_dir / "topics.jsonl", topic_records)

    # ---- trials ----------------------------------------------------------
    zip_paths = sorted(
        (raw / "corpus").glob("ClinicalTrials.2021-04-27.part*.zip")
    )
    if not zip_paths:
        raise FileNotFoundError(f"no corpus archives found under {raw / 'corpus'}")

    seen: Set[str] = set()
    trials_path = output_dir / "trials.jsonl"
    trials_path.parent.mkdir(parents=True, exist_ok=True)

    import random

    rng = random.Random(seed)
    unjudged_reservoir: List[Dict[str, Any]] = []
    unjudged_seen = 0
    # A second scan for unjudged trials would mean re-reading 1.7 GB, so the
    # reservoir is filled in the same pass when requested.
    stream_filter = None if include_unjudged else wanted

    with open(trials_path, "w", encoding="utf-8") as handle:
        for trial in stream_corpus(zip_paths, stream_filter):
            stats.trials_seen += 1
            nct_id = trial["nct_id"]

            if nct_id in wanted:
                if nct_id in seen:
                    continue
                seen.add(nct_id)
                record = build_trial_record(trial, index, stats)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.trials_emitted += 1
            elif include_unjudged:
                # Reservoir sampling: uniform subsample without holding all
                # 326k unjudged trials in memory.
                unjudged_seen += 1
                if len(unjudged_reservoir) < include_unjudged:
                    unjudged_reservoir.append(build_trial_record(trial, index, stats))
                else:
                    j = rng.randrange(unjudged_seen)
                    if j < include_unjudged:
                        unjudged_reservoir[j] = build_trial_record(trial, index, stats)

            if stats.trials_seen % 25_000 == 0:
                logger.info(
                    "  scanned %d trials, emitted %d judged",
                    stats.trials_seen,
                    stats.trials_emitted,
                )

    stats.trials_missing = len(wanted) - len(seen)

    if include_unjudged:
        stats.unjudged_emitted = _write_jsonl(
            output_dir / "unjudged_pool.jsonl", unjudged_reservoir
        )

    # ---- qrels (only judgments whose trial was actually emitted) ---------
    kept = [r for r in qrels_records if r["nct_id"] in seen]
    _write_jsonl(output_dir / "qrels.jsonl", kept)

    manifest = {
        "dataset_root": str(dataset_root),
        "mesh_source": index.source,
        "mesh_descriptors": len(index),
        "topics": stats.topics_emitted,
        "trials_emitted": stats.trials_emitted,
        "trials_missing": stats.trials_missing,
        "judgments_kept": len(kept),
        "judgments_total": len(qrels_records),
        "unjudged_emitted": stats.unjudged_emitted,
        "condition_branches": list(CONDITION_BRANCHES),
        "intervention_branches": list(INTERVENTION_BRANCHES),
        "resolution": stats.resolution_summary(),
        "trials_without_any_uri": stats.trials_without_any_uri,
        "topics_without_condition_uri": stats.topics_without_condition_uri,
        "topics_without_any_uri": stats.topics_without_any_uri,
        "degraded_topics": stats.degraded_topics,
        "seed": seed,
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("Conversion manifest: %s", json.dumps(manifest["resolution"], indent=2))
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("preprocess/trec_ct")
    )
    parser.add_argument(
        "--mesh-cache", type=Path, default=Path("embedding_cache/mesh2021_index.pkl")
    )
    parser.add_argument(
        "--include-unjudged",
        type=int,
        default=0,
        metavar="N",
        help="also emit N unjudged trials for the corpus-wide ablation "
        "(requires a full 375k-trial scan; default 0 = judged only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    stats = convert(
        args.dataset_root,
        args.output_dir,
        mesh_cache=args.mesh_cache,
        include_unjudged=args.include_unjudged,
        seed=args.seed,
    )
    print(json.dumps(stats.resolution_summary(), indent=2))
    if stats.trials_missing:
        logger.warning(
            "%d judged trials were not found in the corpus", stats.trials_missing
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
