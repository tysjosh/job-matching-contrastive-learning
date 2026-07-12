"""CVE Data_Splitter — reproducible train/validation/test partitioning (Requirement 4).

This module partitions the converted ``CVE_View_Record`` JSONL into reproducible
train / validation / test splits, mirroring the existing career-domain
``contrastive_learning/data_splitter.py`` conventions while adding the CVE-specific
behavior required by Requirement 4:

* **stratified** (the default, Req 4.10/4.11): group records by ``priority_band``
  and allocate each band across the three splits by the configured proportions, so
  each split's band distribution matches the overall distribution within integer
  rounding. Records with a missing/empty ``priority_band`` form a dedicated
  "unbanded" stratum that is split by the same proportions and counted (Req 4.12).
* **temporal** (Req 4.5/4.7): order records by ``nvd_published`` ascending, breaking
  ties by ``cve`` lexicographic ascending, then assign earliest→train, next→
  validation, latest→test per the configured proportions. Records whose
  ``nvd_published`` is missing or unparseable are assigned to train and counted.
* **random** (Req 4.6): seeded shuffle, then proportion cut. Identical across runs
  for a fixed seed (Req 4.2).

Every input record is assigned to exactly one split and every record is assigned
(Req 4.4). Proportions default to 80/10/10 and must sum to 100% within ±0.5% and not
produce an empty split; otherwise the splitter stops before writing any split
artifact and reports the invalid configuration (Req 4.9). Outputs are
``split_indices.json``, ``train.jsonl``, ``validation.jsonl``, ``test.jsonl``, and
``split_report.json`` (per-split counts + per-split ``priority_band`` distribution,
Req 4.3/4.8).

The ``CVE_View_Record`` shape this module reads (see design "Data Models"):

* ``cve``: the identifier (used for temporal tie-break and ``split_indices.json``).
* ``nvd_published``: top-level ISO-8601 timestamp string (temporal ordering).
* ``cve_labels.priority_band``: the categorical band used for stratification.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# The three canonical split names, in the fixed order train → validation → test.
SPLIT_NAMES: Tuple[str, str, str] = ("train", "validation", "test")

# Default split proportions as percentages (Req 4.1).
DEFAULT_PROPORTIONS: Dict[str, float] = {"train": 80.0, "validation": 10.0, "test": 10.0}

# Default strategy when the Run_Config does not specify one (Req 4.10).
DEFAULT_STRATEGY = "stratified"

# Allowed absolute deviation, in percent, of the proportion sum from 100 (Req 4.1/4.9).
PROPORTION_SUM_TOLERANCE = 0.5

# Sentinel band key for records with a missing/empty priority_band (Req 4.12).
_UNBANDED = "__unbanded__"


@dataclass
class SplitReport:
    """Structured accounting of a split run (see design "SplitReport").

    Attributes:
        strategy: The split strategy actually used.
        seed: The seed used for reproducible shuffling.
        proportions: The (normalized-to-percent) train/validation/test proportions.
        per_split_counts: Number of records assigned to each split.
        per_split_band_distribution: ``priority_band`` histogram within each split.
        temporal_missing_date_reassigned_count: Records reassigned to train because
            their ``nvd_published`` was missing/unparseable (temporal only, Req 4.7).
        unbanded_count: Records placed in the dedicated unbanded stratum
            (stratified only, Req 4.12).
        status: ``"ok"`` when splits were written, otherwise a failure code
            (``"invalid_proportions"`` or ``"empty_split"``).
        reason: Human-readable explanation when ``status`` is a failure code.
    """

    strategy: str
    seed: int
    proportions: Dict[str, float]
    per_split_counts: Dict[str, int] = field(default_factory=dict)
    per_split_band_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    temporal_missing_date_reassigned_count: int = 0
    unbanded_count: int = 0
    status: str = "ok"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "proportions": self.proportions,
            "per_split_counts": self.per_split_counts,
            "per_split_band_distribution": self.per_split_band_distribution,
            "temporal_missing_date_reassigned_count": self.temporal_missing_date_reassigned_count,
            "unbanded_count": self.unbanded_count,
            "status": self.status,
            "reason": self.reason,
        }


class CVEDataSplitter:
    """Partition CVE_View_Records into reproducible train/validation/test splits.

    Args:
        strategy: ``"stratified"`` (default), ``"temporal"``, or ``"random"``.
        proportions: Percentages per split; defaults to 80/10/10. Must sum to 100
            within ±0.5 and not produce an empty split (Req 4.1/4.9).
        seed: Seed for reproducible shuffling (Req 4.2/4.6).
    """

    def __init__(
        self,
        strategy: Optional[str] = None,
        proportions: Optional[Mapping[str, float]] = None,
        seed: int = 42,
    ):
        self.strategy = (strategy or DEFAULT_STRATEGY).strip().lower()
        if self.strategy not in {"stratified", "temporal", "random"}:
            raise ValueError(
                f"Unknown split strategy: {self.strategy!r}. "
                "Expected one of 'stratified', 'temporal', 'random'."
            )
        self.proportions: Dict[str, float] = {
            name: float((proportions or DEFAULT_PROPORTIONS).get(name, DEFAULT_PROPORTIONS[name]))
            for name in SPLIT_NAMES
        }
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def split_dataset(self, data_path: str, output_dir: str = "cve_data_splits") -> SplitReport:
        """Load a CVE_View_Record JSONL file and split it (convenience wrapper)."""
        records = self._load_records(data_path)
        return self.split_records(records, output_dir)

    def split_records(self, records: Sequence[Mapping[str, Any]], output_dir: str) -> SplitReport:
        """Partition ``records`` and write all artifacts under ``output_dir``.

        Returns a :class:`SplitReport`. On an invalid configuration or an empty
        split, the splitter stops before writing any split artifact, writes only
        ``split_report.json`` with a failure status, and returns that report
        (Req 4.9).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # --- Req 4.9: validate proportions BEFORE any split assignment/writing.
        proportion_error = self._validate_proportions()
        if proportion_error is not None:
            report = SplitReport(
                strategy=self.strategy,
                seed=self.seed,
                proportions=dict(self.proportions),
                status="invalid_proportions",
                reason=proportion_error,
            )
            self._write_report_only(out_path, report)
            logger.error("Split aborted: %s", proportion_error)
            return report

        records = list(records)

        # --- Compute the split assignment fully in memory (nothing written yet).
        if self.strategy == "stratified":
            assignment, unbanded_count, reassigned = self._assign_stratified(records)
        elif self.strategy == "temporal":
            assignment, unbanded_count, reassigned = self._assign_temporal(records)
        else:  # "random"
            assignment, unbanded_count, reassigned = self._assign_random(records)

        # --- Req 4.9: empty-split guard, still before writing any split artifact.
        empty = [name for name in SPLIT_NAMES if not assignment[name]]
        if empty:
            reason = (
                f"Split(s) {empty} would be empty for {len(records)} record(s) "
                f"with proportions {self.proportions}."
            )
            report = SplitReport(
                strategy=self.strategy,
                seed=self.seed,
                proportions=dict(self.proportions),
                status="empty_split",
                reason=reason,
            )
            self._write_report_only(out_path, report)
            logger.error("Split aborted: %s", reason)
            return report

        # --- All good: write split artifacts (Req 4.3).
        self._write_splits(out_path, assignment)

        report = SplitReport(
            strategy=self.strategy,
            seed=self.seed,
            proportions=dict(self.proportions),
            per_split_counts={name: len(assignment[name]) for name in SPLIT_NAMES},
            per_split_band_distribution=self._band_distribution(assignment),
            temporal_missing_date_reassigned_count=reassigned,
            unbanded_count=unbanded_count,
            status="ok",
        )
        self._write_report_only(out_path, report)
        logger.info(
            "Split complete (%s): %s",
            self.strategy,
            {name: len(assignment[name]) for name in SPLIT_NAMES},
        )
        return report

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #
    def _assign_stratified(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Tuple[Dict[str, List[Mapping[str, Any]]], int, int]:
        """Stratified allocation by ``priority_band`` (Req 4.10/4.11/4.12).

        Groups records by band (preserving input order within a band), then cuts
        each band group by the configured proportions so each split's band
        distribution matches the overall distribution within integer rounding.
        Missing/empty bands form a dedicated unbanded stratum split the same way.
        """
        # Preserve first-seen band order for deterministic iteration.
        band_groups: "OrderedDict[str, List[Mapping[str, Any]]]" = OrderedDict()
        unbanded_count = 0
        for record in records:
            band = self._get_priority_band(record)
            if band is None:
                key = _UNBANDED
                unbanded_count += 1
            else:
                key = band
            band_groups.setdefault(key, []).append(record)

        assignment: Dict[str, List[Mapping[str, Any]]] = {name: [] for name in SPLIT_NAMES}
        for band_key, group in band_groups.items():
            # Seed per band (but deterministic) so the overall split is reproducible
            # and independent of band iteration order effects.
            rng = random.Random(f"{self.seed}:{band_key}")
            shuffled = list(group)
            rng.shuffle(shuffled)
            cuts = self._cut_by_proportions(len(shuffled))
            for name, (start, end) in cuts.items():
                assignment[name].extend(shuffled[start:end])

        return assignment, unbanded_count, 0

    def _assign_temporal(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Tuple[Dict[str, List[Mapping[str, Any]]], int, int]:
        """Temporal allocation by ``nvd_published`` (Req 4.5/4.7).

        Dated records are ordered by (nvd_published asc, cve lexicographic asc) and
        cut by the configured proportions. Records with a missing/unparseable date
        are assigned to train and counted.
        """
        dated: List[Tuple[datetime, str, Mapping[str, Any]]] = []
        undated: List[Mapping[str, Any]] = []
        for record in records:
            parsed = self._parse_published(record.get("nvd_published"))
            if parsed is None:
                undated.append(record)
            else:
                dated.append((parsed, str(record.get("cve", "")), record))

        dated.sort(key=lambda item: (item[0], item[1]))
        ordered = [record for _, _, record in dated]

        assignment: Dict[str, List[Mapping[str, Any]]] = {name: [] for name in SPLIT_NAMES}
        cuts = self._cut_by_proportions(len(ordered))
        for name, (start, end) in cuts.items():
            assignment[name].extend(ordered[start:end])

        # Undated records go to train (Req 4.7).
        assignment["train"].extend(undated)

        return assignment, 0, len(undated)

    def _assign_random(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Tuple[Dict[str, List[Mapping[str, Any]]], int, int]:
        """Seeded-random allocation (Req 4.6), reproducible under a fixed seed."""
        rng = random.Random(self.seed)
        shuffled = list(records)
        rng.shuffle(shuffled)

        assignment: Dict[str, List[Mapping[str, Any]]] = {name: [] for name in SPLIT_NAMES}
        cuts = self._cut_by_proportions(len(shuffled))
        for name, (start, end) in cuts.items():
            assignment[name].extend(shuffled[start:end])

        return assignment, 0, 0

    # ------------------------------------------------------------------ #
    # Proportion / cut helpers
    # ------------------------------------------------------------------ #
    def _validate_proportions(self) -> Optional[str]:
        """Return an error message when proportions are invalid, else ``None``.

        Invalid when any proportion is negative or the sum deviates from 100 by
        more than ±0.5 percent (Req 4.1/4.9).
        """
        for name, value in self.proportions.items():
            if value < 0:
                return f"Split proportion for {name!r} is negative: {value}."
        total = sum(self.proportions.values())
        if abs(total - 100.0) > PROPORTION_SUM_TOLERANCE:
            return (
                f"Split proportions must sum to 100% within ±{PROPORTION_SUM_TOLERANCE}%, "
                f"got {total}% ({self.proportions})."
            )
        return None

    def _cut_by_proportions(self, n: int) -> Dict[str, Tuple[int, int]]:
        """Return per-split ``(start, end)`` index ranges over ``n`` items.

        Uses cumulative rounding of the normalized proportions so the counts sum
        to exactly ``n`` and each split's share matches the proportions within
        integer rounding (Req 4.11). The final split absorbs any rounding
        remainder.
        """
        total = sum(self.proportions.values()) or 1.0
        fracs = [self.proportions[name] / total for name in SPLIT_NAMES]

        cuts: Dict[str, Tuple[int, int]] = {}
        cumulative_frac = 0.0
        prev_boundary = 0
        for i, name in enumerate(SPLIT_NAMES):
            cumulative_frac += fracs[i]
            if i == len(SPLIT_NAMES) - 1:
                boundary = n  # last split absorbs the remainder
            else:
                boundary = int(round(cumulative_frac * n))
                boundary = max(prev_boundary, min(boundary, n))
            cuts[name] = (prev_boundary, boundary)
            prev_boundary = boundary
        return cuts

    # ------------------------------------------------------------------ #
    # Record field accessors
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_priority_band(record: Mapping[str, Any]) -> Optional[str]:
        """Return the trimmed ``priority_band`` or ``None`` when missing/empty.

        Reads ``cve_labels.priority_band`` (the canonical location per the design
        Data Models), falling back to a top-level ``priority_band`` for robustness.
        """
        labels = record.get("cve_labels")
        band = None
        if isinstance(labels, Mapping):
            band = labels.get("priority_band")
        if band is None:
            band = record.get("priority_band")
        if band is None:
            return None
        band_str = str(band).strip()
        return band_str if band_str else None

    @staticmethod
    def _parse_published(value: Any) -> Optional[datetime]:
        """Parse an ``nvd_published`` value into a datetime, or ``None`` on failure.

        Handles ISO-8601 strings such as ``"2024-03-04T18:15:09.377"`` and a
        trailing ``Z`` UTC designator. Returns ``None`` for missing/unparseable
        values (Req 4.7).
        """
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed: Optional[datetime] = None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if parsed is None:
            return None
        # Normalize to a naive (tz-stripped) datetime so the temporal sort never
        # mixes offset-aware and offset-naive values (which raises TypeError).
        # Timezone-aware values (e.g. a trailing "Z") are first converted to UTC,
        # then made naive, giving a single consistent ordering key across records.
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    # ------------------------------------------------------------------ #
    # Reporting / IO
    # ------------------------------------------------------------------ #
    def _band_distribution(
        self, assignment: Mapping[str, List[Mapping[str, Any]]]
    ) -> Dict[str, Dict[str, int]]:
        """Per-split ``priority_band`` histogram (Req 4.8); unbanded → 'unbanded'."""
        distribution: Dict[str, Dict[str, int]] = {}
        for name in SPLIT_NAMES:
            counter: Counter = Counter()
            for record in assignment[name]:
                band = self._get_priority_band(record)
                counter[band if band is not None else "unbanded"] += 1
            distribution[name] = dict(counter)
        return distribution

    @staticmethod
    def _load_records(data_path: str) -> List[Dict[str, Any]]:
        """Load CVE_View_Records from a JSONL file, skipping unparseable lines."""
        records: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON at line %d: %s", line_num, exc)
        return records

    def _write_splits(
        self, out_path: Path, assignment: Mapping[str, List[Mapping[str, Any]]]
    ) -> None:
        """Write train/validation/test JSONL plus split_indices.json (Req 4.3)."""
        split_indices: Dict[str, List[Any]] = {}
        for name in SPLIT_NAMES:
            file_path = out_path / f"{name}.jsonl"
            with open(file_path, "w", encoding="utf-8") as handle:
                for record in assignment[name]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # split_indices stores per-split cve identifiers, mirroring the career
            # pipeline's per-split id lists (data_splits_v5/v7 convention).
            split_indices[name] = [record.get("cve") for record in assignment[name]]

        with open(out_path / "split_indices.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "strategy": self.strategy,
                    "seed": self.seed,
                    "proportions": self.proportions,
                    "splits": split_indices,
                },
                handle,
                indent=2,
            )

    @staticmethod
    def _write_report_only(out_path: Path, report: SplitReport) -> None:
        """Write ``split_report.json`` (Req 4.8; also the failure report for 4.9)."""
        with open(out_path / "split_report.json", "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)


def main() -> None:
    """CLI for the CVE data splitter."""
    import argparse

    parser = argparse.ArgumentParser(description="Split CVE_View_Records into train/validation/test")
    parser.add_argument("dataset", help="Path to CVE_View_Record JSONL")
    parser.add_argument(
        "--strategy",
        choices=["stratified", "temporal", "random"],
        default=DEFAULT_STRATEGY,
        help="Split strategy (default: stratified)",
    )
    parser.add_argument(
        "--proportions",
        nargs=3,
        type=float,
        default=[DEFAULT_PROPORTIONS["train"], DEFAULT_PROPORTIONS["validation"], DEFAULT_PROPORTIONS["test"]],
        metavar=("TRAIN", "VALIDATION", "TEST"),
        help="Train/validation/test proportions as percentages summing to 100",
    )
    parser.add_argument("--output-dir", default="cve_data_splits", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    splitter = CVEDataSplitter(
        strategy=args.strategy,
        proportions={
            "train": args.proportions[0],
            "validation": args.proportions[1],
            "test": args.proportions[2],
        },
        seed=args.seed,
    )
    report = splitter.split_dataset(args.dataset, args.output_dir)

    print("\n" + "=" * 60)
    print("CVE DATA SPLIT RESULTS")
    print("=" * 60)
    print(f"Status:   {report.status}")
    if report.status != "ok":
        print(f"Reason:   {report.reason}")
        return
    print(f"Strategy: {report.strategy}")
    print(f"Counts:   {report.per_split_counts}")
    for name in SPLIT_NAMES:
        print(f"  {name} band distribution: {report.per_split_band_distribution.get(name, {})}")
    if report.temporal_missing_date_reassigned_count:
        print(f"Undated → train: {report.temporal_missing_date_reassigned_count}")
    if report.unbanded_count:
        print(f"Unbanded stratum: {report.unbanded_count}")


if __name__ == "__main__":
    main()
