"""Property-based test for the CVE Data_Splitter temporal ordering (Property 9).

Component under test: ``CVEDataSplitter`` (temporal strategy) in
``cve_domain/data_splitter.py``.

Property 9 — *Temporal split is monotonically ordered by (nvd_published, cve)*:
with the temporal strategy, dated records are ordered by ``nvd_published`` ascending,
tie-broken by ``cve`` lexicographic ascending, and assigned earliest→train,
next→validation, latest→test. Consequently every dated train record precedes (by that
ordering) every validation record, which precedes every test record. Records with a
missing/unparseable ``nvd_published`` are assigned to train and counted (Req 4.7).

This test uses Hypothesis (>=100 iterations) to generate CVE_View_Records with varied
``nvd_published`` values — valid ISO-8601 timestamps, deliberate ties (identical
timestamps with different ``cve`` identifiers), and missing/unparseable values — writes
them to a temp JSONL under ``tmp_path``, runs the temporal splitter, reads back the
three split files, and asserts the monotonic-ordering + undated-to-train properties.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_temporal_ordering_property.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cve_domain.data_splitter import SPLIT_NAMES, CVEDataSplitter

# Rank of each split in the temporal order earliest→train→validation→test.
_SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}

# Values that represent a missing/unparseable nvd_published (all parse to None).
_BAD_DATES = ["", "   ", "\t", "not-a-date", "2024-13-99", "99/99/9999", "N/A", None]


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@st.composite
def temporal_records(draw) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Draw a list of CVE_View_Records plus the list of undated ``cve`` ids.

    Each record gets a UNIQUE ``cve`` identifier (identifiers are unique by
    definition and also serve as the temporal tie-break key). Dated records draw
    their timestamp from a small shared pool so identical timestamps — and thus
    genuine ``cve`` tie-breaks — arise frequently. Undated records carry a
    missing/unparseable ``nvd_published``.

    We guarantee at least 12 dated records so the 80/10/10 temporal cut never
    produces an empty validation/test split (which would abort the split before
    writing artifacts, per Req 4.9) — the ordering property is what is under test
    here, not the empty-split guard.
    """
    # Pool of distinct timestamps; sampling from it forces ties.
    pool_size = draw(st.integers(min_value=1, max_value=6))
    pool = draw(
        st.lists(
            st.datetimes(
                min_value=datetime(2000, 1, 1),
                max_value=datetime(2025, 1, 1),
            ),
            min_size=pool_size,
            max_size=pool_size,
        )
    )
    pool_iso = [dt.isoformat() for dt in pool]

    n_dated = draw(st.integers(min_value=12, max_value=40))
    n_undated = draw(st.integers(min_value=0, max_value=8))

    records: List[Dict[str, Any]] = []
    undated_cves: List[str] = []
    counter = 0

    for _ in range(n_dated):
        cve = f"CVE-{counter:05d}"
        counter += 1
        records.append(
            {
                "cve": cve,
                "nvd_published": draw(st.sampled_from(pool_iso)),
                "encoder_view": f"{cve}. synthetic.",
            }
        )

    for _ in range(n_undated):
        cve = f"CVE-{counter:05d}"
        counter += 1
        undated_cves.append(cve)
        records.append(
            {
                "cve": cve,
                "nvd_published": draw(st.sampled_from(_BAD_DATES)),
                "encoder_view": f"{cve}. synthetic.",
            }
        )

    # Shuffle so input order does not coincide with temporal order.
    order = draw(st.permutations(list(range(len(records)))))
    records = [records[i] for i in order]

    return records, undated_cves


# ---------------------------------------------------------------------------
# Independent re-derivation of the temporal ordering key (the "model")
# ---------------------------------------------------------------------------
def _parse(value: Any) -> Optional[datetime]:
    """Mirror the splitter's date parsing to derive the expected ordering key."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_split_cves(out_dir: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for name in SPLIT_NAMES:
        cves: List[str] = []
        with open(out_dir / f"{name}.jsonl", "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    cves.append(json.loads(line)["cve"])
        result[name] = cves
    return result


# ---------------------------------------------------------------------------
# Property 9
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 9: Temporal split is monotonically ordered by (nvd_published, cve)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=temporal_records())
def test_temporal_split_is_monotonically_ordered(data, tmp_path):
    """Property 9 (Validates: Requirements 4.5, 4.7).

    Under the temporal strategy, dated records are ordered by ``nvd_published``
    ascending, tie-broken by ``cve`` lexicographic ascending, and assigned
    earliest→train, next→validation, latest→test — so walking the globally sorted
    dated records yields a non-decreasing split rank (train ≤ validation ≤ test).
    Undated records are all assigned to train and counted in the split report.
    """
    records, undated_cves = data

    out_dir = tmp_path / f"split_{uuid.uuid4().hex}"
    data_path = tmp_path / f"records_{uuid.uuid4().hex}.jsonl"
    _write_jsonl(records, data_path)

    splitter = CVEDataSplitter(strategy="temporal", seed=42)
    report = splitter.split_records(records, str(out_dir))

    # With >=12 dated records at 80/10/10 no split is empty, so the split succeeds.
    assert report.status == "ok", f"unexpected abort: {report.reason}"

    split_of: Dict[str, str] = {}
    for name, cves in _read_split_cves(out_dir).items():
        for cve in cves:
            split_of[cve] = name

    # Partition: every input record assigned to exactly one split, all covered.
    assert set(split_of) == {r["cve"] for r in records}
    assert len(split_of) == len(records)

    # Req 4.7: every undated record is in train, and the reassigned count matches.
    for cve in undated_cves:
        assert split_of[cve] == "train", f"undated {cve} must be in train"
    assert report.temporal_missing_date_reassigned_count == len(undated_cves)

    # Build the globally sorted dated order (nvd_published asc, cve lexicographic asc).
    dated = [
        (parsed, r["cve"])
        for r in records
        if (parsed := _parse(r.get("nvd_published"))) is not None
    ]
    dated.sort(key=lambda item: (item[0], item[1]))

    # Property 9 core: walking the sorted dated records, the split rank is
    # non-decreasing (earliest→train, next→validation, latest→test). This proves
    # every dated train record precedes every validation record, which precedes
    # every test record, by the (nvd_published, cve) ordering.
    prev_rank = -1
    for _parsed_dt, cve in dated:
        rank = _SPLIT_RANK[split_of[cve]]
        assert rank >= prev_rank, (
            f"temporal ordering violated at {cve}: split rank {rank} "
            f"followed rank {prev_rank}"
        )
        prev_rank = rank
