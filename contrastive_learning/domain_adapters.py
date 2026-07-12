"""Domain_Adapter_Seam for CDCL_Core (spec: cve-vulnerability-ranking).

This module defines the small, additive extension point through which the
``DataLoader`` obtains, for each raw record, an encoder view, a label, and a
grouping identifier from a configured domain adapter instead of reading
domain-specific record keys (such as ``resume``/``job``) directly.

Only the protocol and the registry live here for now. The default
``CareerDomainAdapter`` (task 2.2) and the ``CVERecordAdapter`` (task 7.2) are
implemented and registered in later tasks; the registration wiring exposed here
is what those tasks will call.

Requirements: 7.1, 7.8, 11.4
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .data_structures import TrainingConfig, TrainingSample


@runtime_checkable
class DomainAdapter(Protocol):
    """Maps one raw JSONL record to the fields the ``DataLoader`` needs.

    An adapter returns a fully-formed :class:`TrainingSample` (encoder view,
    label, and grouping identifier already placed) or ``None`` when the record
    must be skipped. The seam keeps the ``DataLoader`` agnostic to
    domain-specific record keys.
    """

    #: Stable identifier the adapter is registered under (e.g. ``"career"``).
    name: str

    def build_sample(
        self,
        record: Dict[str, Any],
        line_number: int,
        config: TrainingConfig,
    ) -> Optional[TrainingSample]:
        """Produce a :class:`TrainingSample` for ``record``.

        The returned sample carries the encoder view(s), a normalized
        ``'positive'``/``'negative'`` label, and a grouping identifier injected
        into ``metadata['resume_id']`` so grouped/ordinal batching keeps working
        unchanged. Return ``None`` when the record must be skipped.
        """
        ...

    def validate(self, sample: TrainingSample) -> bool:
        """Domain-specific validation hook for a constructed sample."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps an adapter name to its class. Populated by later tasks via
# ``register_domain_adapter`` (career -> task 2.2, cve -> task 7.2).
_REGISTRY: Dict[str, type] = {}


def register_domain_adapter(name: str, cls: type) -> None:
    """Register a domain adapter class under ``name``.

    Registering the same name again overrides the previous entry, which keeps
    the wiring simple for module-import-time registration.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Domain adapter name must be a non-empty string")
    _REGISTRY[name] = cls


def get_domain_adapter(name: str, config: TrainingConfig) -> DomainAdapter:
    """Instantiate the registered domain adapter named ``name``.

    Raises ``KeyError`` when ``name`` is not registered, listing the available
    adapter names so misconfiguration is easy to diagnose.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        registered = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown domain adapter '{name}'. Registered adapters: {registered}"
        )
    return cls(config)


# ---------------------------------------------------------------------------
# CareerDomainAdapter (default, backward compatible) - task 2.2
# ---------------------------------------------------------------------------

import hashlib
import logging

logger = logging.getLogger(__name__)


class CareerDomainAdapter:
    """Default domain adapter that reproduces the existing resume/job/label handling.

    This adapter moves the career-domain record handling out of ``DataLoader``
    (``create_training_sample`` / ``validate_sample`` and their private helpers)
    without rewriting it, so that once the seam is integrated (task 2.3) the
    loader output, injected ``resume_id`` grouping, batch composition, and
    ``DataLoaderStats`` label-conversion accounting stay byte-for-byte identical
    to the pre-seam pipeline (Req 7.2, 7.3, 12.3).

    Label-conversion counters live on ``self.stats``. The loader assigns its own
    ``DataLoaderStats`` onto the adapter during integration so the counts land on
    the same object the loader reports; when constructed standalone the adapter
    owns a fresh ``DataLoaderStats``.

    Requirements: 7.2
    """

    name = "career"

    def __init__(self, config: TrainingConfig, stats: Any = None) -> None:
        self.config = config
        if stats is None:
            # Lazy import avoids a module-load-time circular import once
            # data_loader.py imports this module at the top level (task 2.3).
            from .data_loader import DataLoaderStats
            stats = DataLoaderStats()
        self.stats = stats

    # -- grouping id -------------------------------------------------------

    @staticmethod
    def _compute_resume_id(resume: Dict[str, Any]) -> str:
        """
        Compute a deterministic resume identifier from resume content.

        Groups all (resume, job) records that share the same underlying resume
        so the ordinal loss can build genuine per-query graded comparisons.
        Uses role + first experience description + sorted skills, which is
        stable across records for the same resume.
        """
        role = str(resume.get('role', ''))
        exp = resume.get('experience', [])
        first_desc = ''
        if isinstance(exp, list) and exp:
            first = exp[0]
            if isinstance(first, dict):
                first_desc = str(first.get('description', ''))[:300]
            else:
                first_desc = str(first)[:300]
        skills = resume.get('skills', [])
        skills_str = '|'.join(sorted(map(str, skills))) if isinstance(skills, list) else str(skills)
        basis = f"{role}||{first_desc}||{skills_str}"
        return hashlib.sha256(basis.encode('utf-8')).hexdigest()[:16]

    # -- sample construction ----------------------------------------------

    def build_sample(
        self,
        record: Dict[str, Any],
        line_number: int,
        config: TrainingConfig,
    ) -> Optional[TrainingSample]:
        """
        Create a TrainingSample from a dictionary record.

        Supports flexible label formats:
        - String: 'positive'/'negative', 'pos'/'neg', 'match'/'no_match', 'yes'/'no', 'y'/'n'
        - Numeric: 1/0 (1=positive, 0=negative)
        - Boolean: true/false (true=positive, false=negative)

        Args:
            record: Dictionary containing sample data
            line_number: Line number for error reporting
            config: Active training configuration

        Returns:
            TrainingSample if valid, None if invalid
        """
        try:
            # Extract required fields
            resume = record.get('resume')
            job = record.get('job')
            label = record.get('label')

            # Generate sample_id if not provided
            sample_id = record.get('sample_id', f"sample_{line_number}")

            # Extract optional metadata
            metadata = record.get('metadata', {})

            # Basic field validation
            if not resume or not isinstance(resume, dict):
                logger.warning(
                    f"Line {line_number}: Invalid or missing resume field")
                return None

            if not job or not isinstance(job, dict):
                logger.warning(
                    f"Line {line_number}: Invalid or missing job field")
                return None

            # Inject a stable resume_id so ordinal loss can group graded jobs
            # per query. job_applicant_id is null in the v7 data, so we derive a
            # deterministic id from resume content (role + first experience text).
            if 'resume_id' not in metadata or not metadata.get('resume_id'):
                metadata = dict(metadata)  # avoid mutating the shared record dict
                metadata['resume_id'] = self._compute_resume_id(resume)

            # Normalize label format
            normalized_label = self._normalize_label(label, line_number)
            if normalized_label is None:
                return None
            label = normalized_label

            return TrainingSample(
                resume=resume,
                job=job,
                label=label,
                sample_id=sample_id,
                metadata=metadata
            )

        except Exception as e:
            logger.warning(
                f"Line {line_number}: Error creating training sample: {e}")
            return None

    # -- validation --------------------------------------------------------

    def validate(self, sample: TrainingSample) -> bool:
        """
        Validate a training sample for completeness and correctness.

        Args:
            sample: TrainingSample to validate

        Returns:
            bool: True if sample is valid, False otherwise
        """
        try:
            # Check if resume has required fields
            if not self._validate_resume(sample.resume):
                return False

            # Check if job has required fields
            if not self._validate_job(sample.job):
                return False

            # Additional validation passed in __post_init__ of TrainingSample
            return True

        except Exception as e:
            logger.warning(
                f"Sample validation error for {sample.sample_id}: {e}")
            return False

    def _validate_resume(self, resume: Dict[str, Any]) -> bool:
        """
        Validate resume data structure, supporting both basic and enhanced formats.

        Args:
            resume: Resume dictionary to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Check for essential resume fields (flexible for enhanced format)
        essential_fields = ['experience', 'skills']

        # Allow at least one essential field to be present
        has_essential_field = any(
            field in resume for field in essential_fields)
        if not has_essential_field:
            logger.debug(
                f"Resume missing essential fields: {essential_fields}")
            return False

        # Validate experience if present
        if 'experience' in resume:
            experience = resume['experience']
            # Support both string and list formats for experience
            if isinstance(experience, str):
                # String format is valid (simple text experience)
                if not experience.strip():
                    logger.debug("Resume experience string is empty")
                    return False
            elif isinstance(experience, list):
                # List of experience entries — accept any non-empty list.
                # Experience entries come in many formats (dicts with various keys,
                # strings, nested structures). The text encoder serializes them all
                # to text, so we only reject truly empty lists.
                if not experience:
                    logger.debug("Resume experience list is empty")
                    return False
            else:
                logger.debug("Resume experience must be a string or list")
                return False

        # Validate skills if present
        if 'skills' in resume:
            skills = resume['skills']
            if not isinstance(skills, list):
                logger.debug("Resume skills must be a list")
                return False

            # Support multiple skill formats
            for skill in skills:
                if isinstance(skill, dict):
                    # Accept any dict with a skill name field
                    if not any(k in skill for k in ('name', 'original_name', 'skill')):
                        logger.debug(
                            "Skill dict must have 'name', 'original_name', or 'skill'")
                        return False

        return True

    def _validate_job(self, job: Dict[str, Any]) -> bool:
        """
        Validate job data structure.

        Args:
            job: Job dictionary to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Validate title is a non-empty string
        title = job.get('title')
        if not isinstance(title, str) or not title.strip():
            logger.debug("Job title must be a non-empty string")
            return False

        # Validate description — accept string or dict with content
        description = job.get('description')
        if description is None:
            logger.debug("Job missing description field")
            return False

        if isinstance(description, str):
            if not description.strip():
                logger.debug("Job description string is empty")
                return False
        elif isinstance(description, dict):
            # Accept dict with 'original', 'text', or any non-empty string value
            has_content = False
            for key in ('original', 'text', 'description'):
                val = description.get(key, '')
                if isinstance(val, str) and val.strip():
                    has_content = True
                    break
            if not has_content:
                # Check if any string value in the dict has content
                for val in description.values():
                    if isinstance(val, str) and len(val.strip()) > 20:
                        has_content = True
                        break
            if not has_content:
                logger.debug("Job description dict has no text content")
                return False
        else:
            logger.debug(f"Job description must be a string or dict, got {type(description).__name__}")
            return False

        return True

    # -- label normalization ----------------------------------------------

    def _normalize_label(self, label: Any, line_number: int = 0) -> Optional[str]:
        """
        Normalize label to standard 'positive'/'negative' format.

        Supports multiple input formats:
        - Numeric: 1/0 (1=positive, 0=negative)
        - Boolean: true/false (true=positive, false=negative)
        - String: 'positive'/'negative', 'pos'/'neg', 'match'/'no_match', etc.

        Args:
            label: Label value in any supported format
            line_number: Line number for error reporting

        Returns:
            Normalized label string ('positive' or 'negative'), or None if invalid
        """
        if label is None:
            logger.warning(f"Line {line_number}: Missing label field")
            return None

        # Handle numeric labels (most common case for generated data)
        if label == 1 or label == '1':
            if label != 'positive':  # Only count if conversion needed
                self.stats.numeric_labels_converted += 1
            return 'positive'
        elif label == 0 or label == '0':
            if label != 'negative':  # Only count if conversion needed
                self.stats.numeric_labels_converted += 1
            return 'negative'

        # Handle boolean labels
        elif label is True or label == 'true':
            self.stats.boolean_labels_converted += 1
            return 'positive'
        elif label is False or label == 'false':
            self.stats.boolean_labels_converted += 1
            return 'negative'

        # Handle string labels (case insensitive)
        elif isinstance(label, str):
            label_lower = label.lower().strip()

            # Positive variations
            if label_lower in ['positive', 'pos', 'match', 'yes', 'y', 'true']:
                if label_lower != 'positive':  # Only count if conversion needed
                    self.stats.string_labels_converted += 1
                return 'positive'

            # Negative variations
            elif label_lower in ['negative', 'neg', 'no_match', 'no', 'n', 'false']:
                if label_lower != 'negative':  # Only count if conversion needed
                    self.stats.string_labels_converted += 1
                return 'negative'

            else:
                logger.warning(f"Line {line_number}: Invalid string label '{label}' "
                               f"(supported: positive/negative, pos/neg, match/no_match, yes/no, y/n, 1/0)")
                return None

        # Handle other numeric types
        elif isinstance(label, (int, float)):
            if label == 1:
                return 'positive'
            elif label == 0:
                return 'negative'
            else:
                logger.warning(
                    f"Line {line_number}: Invalid numeric label '{label}' (must be 1 or 0)")
                return None

        else:
            logger.warning(f"Line {line_number}: Invalid label type '{type(label).__name__}' "
                           f"(supported types: string, int, bool)")
            return None


# Register the default adapter. Career is the default path (Req 7.2, 12.5):
# when a config omits ``domain_adapter`` the loader resolves to this adapter.
register_domain_adapter("career", CareerDomainAdapter)
