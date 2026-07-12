"""Stage 1 contrastive pretraining driver for the CVE domain (task 9.1).

This module is a thin **wiring layer** that drives the *existing* CDCL trainer
(``contrastive_learning/trainer.py``) for CVE Stage 1 contrastive pretraining
(Req 8.1, 8.2, 8.3, 9.1, 13.5). It does **not** reimplement training and it does
**not** touch the trainer / loss math — it only connects the CVE-domain
components that earlier tasks already built:

* ``CVEDataSplitter`` output (``train.jsonl`` / ``validation.jsonl``) is loaded as
  the anchor ``CVE_View_Records``.
* :class:`~cve_domain.positive_selector.CVEPositiveSelector` resolves each
  anchor's ontology-related positive CVE **within its split**; the selected
  positive's full ``CVE_View_Record`` is attached under ``record["positive"]`` so
  the registered :class:`~cve_domain.record_adapter.CVERecordAdapter` builds the
  two-slot (anchor view, positive view) ``TrainingSample`` — a real pair, **no
  augmentation** (Req 8.2, 13.x).
* Anchors with **no** in-split ontology sibling are excluded from Stage 1 and
  counted; no positive is fabricated (Req 13.5).
* :class:`~cve_domain.ontology_adapter.CVEOntologyAdapter` +
  :class:`~cve_domain.negative_selector.CVENegativeSelector` supply tiered
  hard/medium/easy negatives, injected into the existing
  ``batch_processor`` negative-selection point via
  ``BatchProcessor.set_cve_negative_selector`` (task 7.8). The loss math is
  untouched (Req 5.3).
* The trainer runs with ``domain_adapter="cve"`` and a **frozen** encoder
  (``freeze_text_encoder=True``, Req 9.1) and saves the best checkpoint by
  validation loss — behavior the existing trainer already implements (Req 8.3).

Data flow::

    train.jsonl ─┐                                   ┌─ CVENegativeSelector ─┐
                 ├─ CVEPositiveSelector ─ attach ────┤                       ├─► set_cve_negative_selector
    val.jsonl ───┘   "positive" per anchor           └─ CVEOntologyAdapter ──┘        (batch_processor)
                 │
                 └─► paired train/val jsonl ─► ContrastiveLearningTrainer.train() ─► best checkpoint (by val loss)

Because a single ``BatchProcessor`` is reused across the train and validation
passes, the negative selector's same-split universe (``view_lookup`` /
``present_ids``) is scoped to the **union of the loaded splits** so selected
negative ids can always be materialized into encoder views. Pooled ontology
negatives (the ~95% hot path) are unaffected by this; only the minority
same-split *random* fallback (Req 6.3/6.4) is widened from a single split to the
loaded union. A full production run can pass the complete converted record set as
``full_view_records`` to make ``present_ids`` the entire converted corpus
(Req 5.6). This module changes no trainer or loss code.

Requirements: 8.1, 8.2, 8.3, 9.1, 13.5
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from contrastive_learning.data_structures import TrainingConfig

# CVE-domain components (all built by earlier tasks).
from .ontology_adapter import CVEOntologyAdapter
from .negative_selector import CVENegativeSelector
from .positive_selector import CVEPositiveSelector

# Importing the record adapter registers the "cve" domain adapter as an
# import-time side effect (task 7.2), so the DataLoader can resolve it.
from . import record_adapter as _record_adapter  # noqa: F401  (registration side effect)

logger = logging.getLogger(__name__)

#: Key under which the selected positive ``CVE_View_Record`` is attached to an
#: anchor record for the ``CVERecordAdapter`` to read (task 7.2).
POSITIVE_KEY = "positive"

#: File names written for the paired (anchor + attached positive) split inputs
#: the trainer consumes.
PAIRED_TRAIN_FILENAME = "stage1_train_paired.jsonl"
PAIRED_VAL_FILENAME = "stage1_validation_paired.jsonl"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@dataclass
class SplitPairingReport:
    """Accounting for turning one split into paired Stage 1 training input.

    Attributes:
        input_anchors: Number of anchor ``CVE_View_Records`` read for the split.
        paired_anchors: Number of anchors that received an ontology-related
            positive and were written to the paired input.
        excluded_anchors: Number of anchors excluded because they had no in-split
            ontology sibling across all cascade levels (Req 13.5); no positive was
            fabricated.
        missing_positive_record: Number of anchors whose selected positive id was
            not present in the view lookup (defensive; expected to be 0 when the
            lookup covers the split).
    """

    input_anchors: int = 0
    paired_anchors: int = 0
    excluded_anchors: int = 0
    missing_positive_record: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "input_anchors": self.input_anchors,
            "paired_anchors": self.paired_anchors,
            "excluded_anchors": self.excluded_anchors,
            "missing_positive_record": self.missing_positive_record,
        }


@dataclass
class Stage1PreparationResult:
    """Result of preparing Stage 1 paired inputs (before the trainer runs).

    Attributes:
        paired_train_path: Path to the written paired training JSONL.
        paired_val_path: Path to the written paired validation JSONL (``None``
            when no validation split was provided).
        train_report: Pairing accounting for the training split.
        val_report: Pairing accounting for the validation split (``None`` when no
            validation split was provided).
        view_lookup: The ``cve`` -> ``CVE_View_Record`` lookup over the negative
            universe (used to materialize selected negative ids into encoder
            views inside the batch processor).
    """

    paired_train_path: Path
    paired_val_path: Optional[Path]
    train_report: SplitPairingReport
    val_report: Optional[SplitPairingReport]
    view_lookup: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_view_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load a split's ``CVE_View_Records`` from a JSONL file (one object/line)."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse {path}:{line_number} as JSON: {exc}"
                ) from exc
            if isinstance(obj, dict):
                records.append(obj)
    return records


def build_view_lookup(
    *record_sets: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Build a ``cve`` -> record lookup over one or more record sets (first-wins)."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for records in record_sets:
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            cve = rec.get("cve")
            if cve is None:
                continue
            cve = str(cve).strip()
            if cve and cve not in lookup:
                lookup[cve] = dict(rec)
    return lookup


# --------------------------------------------------------------------------- #
# Stage 1 driver
# --------------------------------------------------------------------------- #
class Stage1ContrastivePretrainer:
    """Drive the existing CDCL trainer for CVE Stage 1 contrastive pretraining.

    The driver is constructed from a :class:`TrainingConfig` (the CVE Run_Config)
    plus the ontology-pools path, then:

    1. loads the train/validation ``CVE_View_Records`` splits,
    2. selects tiered negatives and ontology-related positives per anchor,
    3. writes paired split inputs (anchor + attached positive), excluding anchors
       with no in-split sibling (Req 13.5),
    4. builds the existing trainer with ``domain_adapter="cve"`` + frozen encoder,
    5. injects the negative selector into the trainer's batch processor, and
    6. runs training, letting the trainer save the best checkpoint by validation
       loss (Req 8.3).

    Args:
        config: The CVE Run_Config. ``domain_adapter`` is forced to ``"cve"`` and
            ``freeze_text_encoder`` defaults to ``True`` (Req 9.1); view
            augmentation is disabled (no augmentation — Req 8.2).
        output_dir: Directory for paired inputs, selection reports, and trainer
            checkpoints/logs.
        denominator_pools_path: Path to ``cve_denominator_pools.jsonl`` (defaults
            to ``config.cve_denominator_pools_path``).
        cyber_kg_path: Optional Cyber_KG path (defaults to ``config.cyber_kg_path``).
    """

    def __init__(
        self,
        config: TrainingConfig,
        output_dir: str | Path,
        denominator_pools_path: Optional[str] = None,
        cyber_kg_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.denominator_pools_path = (
            denominator_pools_path or getattr(config, "cve_denominator_pools_path", None)
        )
        if not self.denominator_pools_path:
            raise ValueError(
                "A denominator pools path is required for Stage 1 (set "
                "config.cve_denominator_pools_path or pass denominator_pools_path)."
            )
        self.cyber_kg_path = cyber_kg_path or getattr(config, "cyber_kg_path", None)

        # Enforce the Stage 1 invariants on the reused trainer config. These are
        # additive config values; the trainer/loss math is unchanged.
        self.config.domain_adapter = "cve"
        # Frozen encoder by default (Req 9.1); honor an explicit unfrozen override.
        if not hasattr(self.config, "freeze_text_encoder"):
            self.config.freeze_text_encoder = True
        # No augmentation for the CVE domain (Req 8.2): anchor+positive are a real
        # pair supplied by the CVEPositiveSelector.
        self.config.use_view_augmentation = False
        # The CVE path selects negatives via the injected selector, not the career
        # pathway/global logic.
        self.config.use_pathway_negatives = False
        self.config.global_negative_sampling = False

        # Loaded lazily in prepare().
        self.ontology_adapter: Optional[CVEOntologyAdapter] = None
        self.negative_selector: Optional[CVENegativeSelector] = None
        self._view_lookup: Dict[str, Dict[str, Any]] = {}
        self._present_ids: set = set()

    # ------------------------------------------------------------------ #
    # Preparation: pairing + selection reports
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        train_records: Sequence[Mapping[str, Any]],
        validation_records: Optional[Sequence[Mapping[str, Any]]] = None,
        full_view_records: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Stage1PreparationResult:
        """Select positives/negatives and write paired Stage 1 split inputs.

        Args:
            train_records: The training split ``CVE_View_Records``.
            validation_records: The validation split ``CVE_View_Records``
                (optional but required for best-by-validation checkpointing).
            full_view_records: Optional complete converted record set. When
                provided it defines the negative *validity* universe
                (``present_ids``, Req 5.6) and the view lookup; otherwise the
                union of the loaded splits is used.

        Returns:
            A :class:`Stage1PreparationResult` with the paired input paths and
            per-split pairing reports.
        """
        train_records = list(train_records)
        validation_records = list(validation_records or [])

        # Build the negative universe (view lookup + present ids). Prefer the full
        # converted corpus when supplied (Req 5.6); else use the loaded union.
        if full_view_records is not None:
            self._view_lookup = build_view_lookup(full_view_records)
        else:
            self._view_lookup = build_view_lookup(train_records, validation_records)
        self._present_ids = set(self._view_lookup.keys())

        # Load the ontology pools once (stops before training on read/parse
        # failure — Req 5.7) and build the shared negative selector.
        self.ontology_adapter = CVEOntologyAdapter(
            self.denominator_pools_path, cyber_kg_path=self.cyber_kg_path
        )
        self.negative_selector = CVENegativeSelector.from_config(
            self.ontology_adapter, self.config
        )

        # --- Train split: negatives -> positives (exclude negatives) -> pair. ---
        train_report, paired_train_path = self._prepare_split(
            train_records,
            split_name="train",
            paired_filename=PAIRED_TRAIN_FILENAME,
        )

        # --- Validation split (optional). ---
        val_report: Optional[SplitPairingReport] = None
        paired_val_path: Optional[Path] = None
        if validation_records:
            val_report, paired_val_path = self._prepare_split(
                validation_records,
                split_name="validation",
                paired_filename=PAIRED_VAL_FILENAME,
            )

        return Stage1PreparationResult(
            paired_train_path=paired_train_path,
            paired_val_path=paired_val_path,
            train_report=train_report,
            val_report=val_report,
            view_lookup=self._view_lookup,
        )

    def _prepare_split(
        self,
        records: Sequence[Mapping[str, Any]],
        split_name: str,
        paired_filename: str,
    ) -> tuple[SplitPairingReport, Path]:
        """Select negatives + positives for one split and write its paired input."""
        assert self.negative_selector is not None  # set in prepare()

        split_out = self.output_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        # Negatives first, so the positive selector can exclude an anchor's own
        # selected negatives from its positive candidates (Req 13.4).
        negatives_by_anchor = self.negative_selector.select_for_split(
            records, present_ids=self._present_ids, output_dir=str(split_out)
        )

        # Ontology-related positive per anchor, within this split, seeded, with
        # the anchor's negatives excluded; anchors with no sibling are excluded
        # from Stage 1 (Req 13.5).
        positive_selector = CVEPositiveSelector.from_config(self.config)
        selections = positive_selector.select_for_split(
            records,
            negatives_by_anchor=negatives_by_anchor,
            output_dir=str(split_out),
        )

        paired_path = split_out / paired_filename
        report = self._write_paired_jsonl(records, selections, paired_path)
        logger.info(
            "Stage 1 %s pairing: %d/%d anchors paired, %d excluded (no sibling), "
            "%d missing positive record",
            split_name,
            report.paired_anchors,
            report.input_anchors,
            report.excluded_anchors,
            report.missing_positive_record,
        )
        return report, paired_path

    def _write_paired_jsonl(
        self,
        records: Sequence[Mapping[str, Any]],
        selections: Mapping[str, str],
        out_path: Path,
    ) -> SplitPairingReport:
        """Attach each anchor's selected positive record and write the paired JSONL.

        Anchors with no selected positive (excluded, Req 13.5) are dropped from
        the output and counted. The attached positive is the *full*
        ``CVE_View_Record`` of the selected positive cve, read from the view
        lookup, so the ``CVERecordAdapter`` can build the second view slot.
        """
        report = SplitPairingReport()
        with open(out_path, "w", encoding="utf-8") as handle:
            for rec in records:
                if not isinstance(rec, Mapping):
                    continue
                report.input_anchors += 1
                cve = str(rec.get("cve", "")).strip()
                positive_cve = selections.get(cve)
                if not positive_cve:
                    # No in-split ontology sibling -> excluded, no fabrication.
                    report.excluded_anchors += 1
                    continue
                positive_record = self._view_lookup.get(positive_cve)
                if positive_record is None:
                    # Defensive: selector drew from the split, so this is unexpected.
                    report.missing_positive_record += 1
                    continue
                paired = dict(rec)
                paired[POSITIVE_KEY] = positive_record
                handle.write(json.dumps(paired, ensure_ascii=False) + "\n")
                report.paired_anchors += 1
        return report

    # ------------------------------------------------------------------ #
    # Trainer construction + run
    # ------------------------------------------------------------------ #
    def build_trainer(self, prep: Stage1PreparationResult) -> Any:
        """Construct the existing CDCL trainer wired for CVE Stage 1.

        Sets ``config.validation_path`` to the paired validation input (so the
        trainer's best-by-validation checkpointing runs — Req 8.3) and injects
        the CVE tiered negative selector into the trainer's batch processor
        (task 7.8). Returns the constructed
        ``contrastive_learning.trainer.ContrastiveLearningTrainer``.

        The heavy ``torch`` / ``sentence-transformers`` import is deferred to this
        method so preparation and wiring can be exercised without loading the
        encoder.
        """
        # Deferred import: keeps prepare()/pairing usable without torch present.
        from contrastive_learning.trainer import ContrastiveLearningTrainer

        if prep.paired_val_path is not None:
            self.config.validation_path = str(prep.paired_val_path)

        trainer = ContrastiveLearningTrainer(
            config=self.config,
            output_dir=str(self.output_dir),
            # No ESCO graph: CVE negatives come from the injected selector.
            esco_graph_path=None,
        )

        # Route negative selection through the CVE tiered selector at the existing
        # batch-processor negative-selection point (task 7.8, Req 5.3). The loss
        # math is untouched.
        assert self.negative_selector is not None
        trainer.batch_processor.set_cve_negative_selector(
            self.negative_selector,
            view_lookup=prep.view_lookup,
            present_ids=self._present_ids,
        )
        return trainer

    def run(
        self,
        train_path: Optional[str] = None,
        validation_path: Optional[str] = None,
        full_records_path: Optional[str] = None,
    ) -> Any:
        """End-to-end Stage 1: load splits, prepare pairs, build trainer, train.

        Args:
            train_path: Path to the training ``CVE_View_Records`` JSONL (defaults
                to ``<output_dir>/../train.jsonl`` is NOT assumed; provide it or
                ``config``-derived paths).
            validation_path: Path to the validation split JSONL.
            full_records_path: Optional path to the complete converted record set
                to use as the negative validity universe (Req 5.6).

        Returns:
            The ``TrainingResults`` from the trainer (best checkpoint saved by the
            trainer under ``output_dir`` — Req 8.3).
        """
        if not train_path:
            raise ValueError("train_path is required to run Stage 1.")
        train_records = load_view_records(train_path)
        validation_records = (
            load_view_records(validation_path) if validation_path else []
        )
        full_records = (
            load_view_records(full_records_path) if full_records_path else None
        )

        prep = self.prepare(
            train_records=train_records,
            validation_records=validation_records,
            full_view_records=full_records,
        )
        trainer = self.build_trainer(prep)
        logger.info(
            "Starting CVE Stage 1 contrastive pretraining on %s (val=%s)",
            prep.paired_train_path,
            prep.paired_val_path,
        )
        return trainer.train(str(prep.paired_train_path))
