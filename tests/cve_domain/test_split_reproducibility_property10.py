"""Property-based tests for CVE Data_Splitter reproducibility (Property 10).

# Feature: cve-vulnerability-ranking, Property 10: Splitting is reproducible under a fixed seed

Property 10 (design.md): *For any* set of CVE_View_Records and a fixed split seed,
repeated splits produce identical per-split membership.

**Validates: Requirements 4.2, 4.6**

Requirement 4.2: WHEN a split seed is provided in the Run_Config, THE Data_Splitter
SHALL produce identical split membership across repeated runs using that seed.
Requirement 4.6: WHERE the Run_Config selects a random split strategy, THE
Data_Splitter SHALL assign CVE_View_Records to splits by seeded random sampling.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

from hypothesis import given, settings, strategies as st

# Make the repository root importable when pytest is invoked from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cve_domain.data_splitter import (  # noqa: E402
    SPLIT_NAMES,
    CVEDataSplitter,
)


# --------------------------------------------------------------------------- #
# Generators — sets of CVE_View_Records with UNIQUE cve identifiers so that
# per-split membership is well defined and comparable across runs.
# --------------------------------------------------------------------------- #

# Bands include some real-ish values plus None/"" to exercise the unbanded
# stratum path in the stratified strategy (Req 4.12).
_bands = st.sampled_from(["critical", "high", "medium", "low", None, ""])

# nvd_published: dated (ISO-8601), undated (None), and unparseable garbage so the
# temporal strategy's undated->train path is also exercised.
_published = st.one_of(
    st.none(),
    st.just("not-a-date"),
    st.from_regex(
        r"20[0-2][0-9]-[0-1][0-9]-[0-2][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]",
        fullmatch=True,
    ),
)


@st.composite
def cve_view_record_sets(draw, min_records: int = 15, max_records: int = 80):
    """Generate a list of CVE_View_Records with unique cve identifiers.

    Enough records are generated (>= 15) that the default 80/10/10 proportions
    fill every split, so the splitter reaches the ``status == "ok"`` path and
    writes the split JSONL artifacts we compare.
    """
    n = draw(st.integers(min_value=min_records, max_value=max_records))
    # Unique integer ids -> unique CVE identifiers.
    ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=1_000_000),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    records: List[Dict[str, Any]] = []
    for i in ids:
        record: Dict[str, Any] = {
            "cve": f"CVE-2024-{i:07d}",
            "encoder_view": f"CVE-2024-{i:07d}. synthetic view.",
            "nvd_published": draw(_published),
            "cve_labels": {"priority_band": draw(_bands)},
        }
        records.append(record)
    return records


def _membership(output_dir: Path) -> Dict[str, Any]:
    """Read every artifact written under ``output_dir`` for reproducibility checks.

    Captures everything that must be identical across identical-seed runs:
    * the ordered per-split cve id lists from ``split_indices.json`` (when written),
    * the raw bytes of ``split_indices.json`` and ``split_report.json``, and
    * the raw bytes of each ``{split}.jsonl`` file.

    ``split_indices.json`` and the ``{split}.jsonl`` files are only written when the
    split succeeds (``status == "ok"``); on a stopped/aborted split only
    ``split_report.json`` exists. Reproducibility must hold in either case, so this
    helper tolerates absent artifacts and represents them as ``None`` / ``b""``.
    """
    indices_path = output_dir / "split_indices.json"
    if indices_path.exists():
        indices_bytes = indices_path.read_bytes()
        splits = json.loads(indices_bytes)["splits"]
    else:
        indices_bytes = b""
        splits = None

    report_bytes = (output_dir / "split_report.json").read_bytes()

    jsonl_bytes: Dict[str, bytes] = {}
    for name in SPLIT_NAMES:
        path = output_dir / f"{name}.jsonl"
        jsonl_bytes[name] = path.read_bytes() if path.exists() else b""

    return {
        "splits": splits,
        "indices_bytes": indices_bytes,
        "report_bytes": report_bytes,
        "jsonl_bytes": jsonl_bytes,
    }


def _run_split(
    records: List[Mapping[str, Any]],
    strategy: str,
    seed: int,
    tmp_root: Path,
    tag: str,
) -> Dict[str, Any]:
    out_dir = tmp_root / tag
    splitter = CVEDataSplitter(strategy=strategy, seed=seed)
    splitter.split_records(records, str(out_dir))
    return _membership(out_dir)


# --------------------------------------------------------------------------- #
# Property 10 — repeated splits with the same seed are byte-identical.
# --------------------------------------------------------------------------- #


@settings(max_examples=120, deadline=None)
@given(
    records=cve_view_record_sets(),
    strategy=st.sampled_from(["stratified", "temporal", "random"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_split_is_reproducible_under_fixed_seed(records, strategy, seed):
    """Running the splitter twice with the same seed + input is identical (Req 4.2)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        first = _run_split(records, strategy, seed, tmp_root, "run_a")
        second = _run_split(records, strategy, seed, tmp_root, "run_b")

    # Identical ordered per-split membership.
    assert first["splits"] == second["splits"], (
        f"per-split membership differs across identical-seed runs ({strategy})"
    )

    # When the split succeeded, both runs agree on each split's membership set.
    if first["splits"] is not None:
        for name in SPLIT_NAMES:
            assert set(first["splits"][name]) == set(second["splits"][name])

    # Byte-identical artifacts: split_indices.json, split_report.json, each {split}.jsonl.
    assert first["indices_bytes"] == second["indices_bytes"], (
        "split_indices.json is not byte-identical across identical-seed runs"
    )
    assert first["report_bytes"] == second["report_bytes"], (
        "split_report.json is not byte-identical across identical-seed runs"
    )
    assert first["jsonl_bytes"] == second["jsonl_bytes"], (
        "split JSONL artifacts are not byte-identical across identical-seed runs"
    )


@settings(max_examples=120, deadline=None)
@given(
    records=cve_view_record_sets(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_random_split_reproducible_across_new_instances(records, seed):
    """A fresh splitter instance with the same seed reproduces the random split (Req 4.6)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        # Two independently constructed splitter instances, same seed.
        first = _run_split(records, "random", seed, tmp_root, "inst_a")
        second = _run_split(records, "random", seed, tmp_root, "inst_b")

    assert first["splits"] == second["splits"]
    assert first["indices_bytes"] == second["indices_bytes"]


# --------------------------------------------------------------------------- #
# Complementary check: a different seed CAN change membership for the seeded
# strategies (random / stratified). This is deterministic (not Hypothesis) with
# a dataset large enough that at least one of several seeds reshuffles it.
# --------------------------------------------------------------------------- #


def _fixed_dataset(n: int = 200) -> List[Dict[str, Any]]:
    bands = ["critical", "high", "medium", "low"]
    records: List[Dict[str, Any]] = []
    for i in range(n):
        records.append(
            {
                "cve": f"CVE-2024-{i:07d}",
                "encoder_view": f"CVE-2024-{i:07d}. synthetic view.",
                "nvd_published": f"2024-01-01T00:00:{i % 60:02d}",
                "cve_labels": {"priority_band": bands[i % len(bands)]},
            }
        )
    return records


def test_different_seed_can_change_random_membership():
    """Different seeds produce different membership for the random strategy (Req 4.6)."""
    records = _fixed_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        base = _run_split(records, "random", 42, tmp_root, "seed_42")
        # Try several alternative seeds; at least one must differ.
        differs = False
        for alt in (0, 1, 7, 99, 12345):
            other = _run_split(records, "random", alt, tmp_root, f"seed_{alt}")
            if other["splits"] != base["splits"]:
                differs = True
                break
    assert differs, "random split membership did not change for any alternative seed"


def test_different_seed_can_change_stratified_membership():
    """Different seeds produce different membership for the stratified strategy (Req 4.6)."""
    records = _fixed_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        base = _run_split(records, "stratified", 42, tmp_root, "seed_42")
        differs = False
        for alt in (0, 1, 7, 99, 12345):
            other = _run_split(records, "stratified", alt, tmp_root, f"seed_{alt}")
            if other["splits"] != base["splits"]:
                differs = True
                break
    assert differs, "stratified split membership did not change for any alternative seed"
