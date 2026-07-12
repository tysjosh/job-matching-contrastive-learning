#!/usr/bin/env python3
"""Property-based equivalence test for the Domain_Adapter_Seam career path.

Feature: cve-vulnerability-ranking, Task 2.4 (optional test).

Property 13: The career adapter is byte-identical to the pre-seam pipeline.

This is a *model-based* test. The "model" is an independent reimplementation of
the ORIGINAL, pre-seam ``DataLoader`` career-record handling
(``create_training_sample`` / ``validate_sample`` and their helpers
``_compute_resume_id`` / ``_normalize_label`` / ``_validate_resume`` /
``_validate_job``). The "system under test" is the current, seam-integrated
``DataLoader``, which delegates through ``CareerDomainAdapter`` selected via the
``domain_adapter`` config field.

For any randomly generated career JSONL record, the test asserts that the current
pipeline and the reference model agree byte-for-byte on:

  * whether a sample is produced at all (None vs. a sample),
  * the sample_id,
  * the normalized 'positive'/'negative' label,
  * the injected ``metadata['resume_id']`` grouping id,
  * the validation outcome.

The ``domain_adapter``-absent path is exercised explicitly: a config from which
the ``domain_adapter`` attribute has been deleted must still resolve to the
default career adapter (the seam uses ``getattr(config, "domain_adapter",
"career")``), and must produce identical results to a config that names
``"career"`` and to the reference model.

Validates: Requirements 7.2, 7.3, 7.7, 12.2, 12.3, 12.5
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Make the repo root importable so ``contrastive_learning`` resolves regardless
# of the directory pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contrastive_learning.data_loader import DataLoader  # noqa: E402
from contrastive_learning.data_structures import TrainingConfig, TrainingSample  # noqa: E402


# ---------------------------------------------------------------------------
# Reference model: an independent reimplementation of the PRE-SEAM career logic.
# ---------------------------------------------------------------------------
#
# The values below mirror the semantics of the original DataLoader methods.
# They are written independently of the CareerDomainAdapter so that the test
# genuinely compares two implementations rather than a function against itself.


def ref_compute_resume_id(resume: Dict[str, Any]) -> str:
    """Reference reimplementation of the pre-seam ``_compute_resume_id``."""
    role = str(resume.get("role", ""))
    exp = resume.get("experience", [])
    first_desc = ""
    if isinstance(exp, list) and exp:
        first = exp[0]
        if isinstance(first, dict):
            first_desc = str(first.get("description", ""))[:300]
        else:
            first_desc = str(first)[:300]
    skills = resume.get("skills", [])
    if isinstance(skills, list):
        skills_str = "|".join(sorted(map(str, skills)))
    else:
        skills_str = str(skills)
    basis = f"{role}||{first_desc}||{skills_str}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def ref_normalize_label(label: Any) -> Optional[str]:
    """Reference reimplementation of the pre-seam ``_normalize_label``."""
    if label is None:
        return None

    # Numeric 1/0 including their string forms.
    if label == 1 or label == "1":
        return "positive"
    if label == 0 or label == "0":
        return "negative"

    # Booleans and their lowercase string spellings.
    if label is True or label == "true":
        return "positive"
    if label is False or label == "false":
        return "negative"

    if isinstance(label, str):
        label_lower = label.lower().strip()
        if label_lower in ["positive", "pos", "match", "yes", "y", "true"]:
            return "positive"
        if label_lower in ["negative", "neg", "no_match", "no", "n", "false"]:
            return "negative"
        return None

    if isinstance(label, (int, float)):
        if label == 1:
            return "positive"
        if label == 0:
            return "negative"
        return None

    return None


def ref_validate_resume(resume: Dict[str, Any]) -> bool:
    """Reference reimplementation of the pre-seam ``_validate_resume``."""
    essential_fields = ["experience", "skills"]
    if not any(field in resume for field in essential_fields):
        return False

    if "experience" in resume:
        experience = resume["experience"]
        if isinstance(experience, str):
            if not experience.strip():
                return False
        elif isinstance(experience, list):
            if not experience:
                return False
        else:
            return False

    if "skills" in resume:
        skills = resume["skills"]
        if not isinstance(skills, list):
            return False
        for skill in skills:
            if isinstance(skill, dict):
                if not any(k in skill for k in ("name", "original_name", "skill")):
                    return False

    return True


def ref_validate_job(job: Dict[str, Any]) -> bool:
    """Reference reimplementation of the pre-seam ``_validate_job``."""
    title = job.get("title")
    if not isinstance(title, str) or not title.strip():
        return False

    description = job.get("description")
    if description is None:
        return False

    if isinstance(description, str):
        if not description.strip():
            return False
    elif isinstance(description, dict):
        has_content = False
        for key in ("original", "text", "description"):
            val = description.get(key, "")
            if isinstance(val, str) and val.strip():
                has_content = True
                break
        if not has_content:
            for val in description.values():
                if isinstance(val, str) and len(val.strip()) > 20:
                    has_content = True
                    break
        if not has_content:
            return False
    else:
        return False

    return True


def ref_build_sample(
    record: Dict[str, Any], line_number: int
) -> Optional[Tuple[str, str, str]]:
    """Reference reimplementation of the pre-seam ``create_training_sample``.

    Returns ``(sample_id, normalized_label, resume_id)`` or ``None`` when the
    record would be skipped. Also enforces the ``TrainingSample`` invariants
    that ``__post_init__`` checks (label domain, dict types, non-empty id).
    """
    try:
        resume = record.get("resume")
        job = record.get("job")
        label = record.get("label")

        sample_id = record.get("sample_id", f"sample_{line_number}")
        metadata = record.get("metadata", {})

        if not resume or not isinstance(resume, dict):
            return None
        if not job or not isinstance(job, dict):
            return None

        if "resume_id" not in metadata or not metadata.get("resume_id"):
            metadata = dict(metadata)
            metadata["resume_id"] = ref_compute_resume_id(resume)

        normalized_label = ref_normalize_label(label)
        if normalized_label is None:
            return None

        # TrainingSample.__post_init__ invariants.
        if normalized_label not in ("positive", "negative"):
            return None
        if not sample_id:
            return None

        return (sample_id, normalized_label, metadata["resume_id"])
    except Exception:
        return None


def ref_outcome(record: Dict[str, Any], line_number: int) -> Dict[str, Any]:
    """Full reference outcome: creation + validation."""
    built = ref_build_sample(record, line_number)
    if built is None:
        return {"created": False, "sample_id": None, "label": None,
                "resume_id": None, "valid": None}
    sample_id, label, resume_id = built
    valid = ref_validate_resume(record["resume"]) and ref_validate_job(record["job"])
    return {"created": True, "sample_id": sample_id, "label": label,
            "resume_id": resume_id, "valid": valid}


# ---------------------------------------------------------------------------
# System-under-test driver: the current seam-integrated DataLoader.
# ---------------------------------------------------------------------------


def sut_outcome(loader: DataLoader, record: Dict[str, Any],
                line_number: int) -> Dict[str, Any]:
    """Outcome produced by the current DataLoader through the seam."""
    sample: Optional[TrainingSample] = loader.create_training_sample(record, line_number)
    if sample is None:
        return {"created": False, "sample_id": None, "label": None,
                "resume_id": None, "valid": None}
    valid = loader.validate_sample(sample)
    return {"created": True, "sample_id": sample.sample_id, "label": sample.label,
            "resume_id": sample.metadata.get("resume_id"), "valid": valid}


def _make_career_loader() -> DataLoader:
    """DataLoader whose config explicitly names the career adapter."""
    config = TrainingConfig(batch_size=32, loss_type="infonce",
                            shuffle_data=False, training_phase="supervised")
    # default domain_adapter is already "career"; be explicit for clarity.
    config.domain_adapter = "career"
    return DataLoader(config)


class _AbsentAdapterConfig:
    """A config object that genuinely has NO ``domain_adapter`` attribute.

    A real ``TrainingConfig`` always carries ``domain_adapter`` (dataclass
    default ``"career"``), so deleting the instance attribute only unmasks the
    class default. To truly exercise the seam's
    ``getattr(config, "domain_adapter", "career")`` fallback — the behavior an
    older, pre-seam career config JSON (which never mentions the field) relies
    on — we use a minimal object that omits the attribute entirely while
    supplying only what ``DataLoader.__init__`` and the career path read.
    """

    def __init__(self) -> None:
        self.batch_size = 32
        self.shuffle_data = False
        self.loss_type = "infonce"
        self.training_phase = "supervised"
        # deliberately no ``domain_adapter`` and no ``group_by_resume``


def _make_absent_field_loader() -> DataLoader:
    """DataLoader whose config has NO ``domain_adapter`` attribute at all.

    Exercises Req 12.5 / 7.7: the seam falls back to the default career adapter
    via ``getattr(config, "domain_adapter", "career")``.
    """
    config = _AbsentAdapterConfig()
    assert not hasattr(config, "domain_adapter")
    return DataLoader(config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hypothesis strategies for random career JSONL records.
# ---------------------------------------------------------------------------

_text = st.text(min_size=0, max_size=40)

# skills: list of strings and/or dicts (valid and invalid dict shapes), or a
# non-list to exercise the "skills must be a list" rejection path.
_skill_dict = st.one_of(
    st.fixed_dictionaries({"name": _text}),
    st.fixed_dictionaries({"original_name": _text}),
    st.fixed_dictionaries({"skill": _text}),
    st.fixed_dictionaries({"other": _text}),  # invalid: no recognized key
)
_skills = st.one_of(
    st.lists(st.one_of(_text, _skill_dict), max_size=4),
    st.text(max_size=10),  # non-list -> invalid
    st.integers(),         # non-list -> invalid
)

_experience_entry = st.one_of(
    st.fixed_dictionaries({"description": _text}),
    st.fixed_dictionaries({"title": _text}),
    _text,
)
_experience = st.one_of(
    st.lists(_experience_entry, max_size=3),  # possibly empty -> invalid
    _text,                                    # string form
    st.integers(),                            # wrong type -> invalid
)


@st.composite
def resumes(draw):
    d: Dict[str, Any] = {}
    if draw(st.booleans()):
        d["role"] = draw(_text)
    if draw(st.booleans()):
        d["experience_level"] = draw(_text)
    # Independently include experience / skills so records land on both sides
    # of the "at least one essential field" rule.
    if draw(st.booleans()):
        d["experience"] = draw(_experience)
    if draw(st.booleans()):
        d["skills"] = draw(_skills)
    return d


_job_description = st.one_of(
    _text,
    st.none(),
    st.fixed_dictionaries({"original": _text}),
    st.fixed_dictionaries({"text": _text}),
    st.fixed_dictionaries({"filler": st.text(max_size=60)}),  # only-long-value path
    st.fixed_dictionaries({"empty": st.just("")}),
)


@st.composite
def jobs(draw):
    d: Dict[str, Any] = {}
    if draw(st.booleans()):
        d["title"] = draw(st.one_of(_text, st.none(), st.integers()))
    if draw(st.booleans()):
        d["description"] = draw(_job_description)
    return d


# Labels: cover numeric, string-numeric, boolean, boolean-strings, textual
# spellings (valid and invalid), floats, None, and unsupported types.
_labels = st.one_of(
    st.sampled_from([
        1, 0, "1", "0", True, False, "true", "false", "True", "False",
        "positive", "negative", "POSITIVE", "Negative", "pos", "neg",
        "match", "no_match", "yes", "no", "y", "n", "  positive  ",
        "maybe", "unknown", "", "2", 2, -1, 1.0, 0.0, 3.5, None,
    ]),
    _text,
)


@st.composite
def career_records(draw):
    rec: Dict[str, Any] = {}
    # resume / job may be present-and-dict, present-but-not-dict, or absent.
    resume_choice = draw(st.integers(min_value=0, max_value=3))
    if resume_choice == 0:
        rec["resume"] = draw(resumes())
    elif resume_choice == 1:
        rec["resume"] = draw(st.one_of(st.none(), st.text(max_size=5),
                                       st.integers(), st.just({})))
    # choice 2/3 -> omit the key entirely

    job_choice = draw(st.integers(min_value=0, max_value=3))
    if job_choice == 0:
        rec["job"] = draw(jobs())
    elif job_choice == 1:
        rec["job"] = draw(st.one_of(st.none(), st.text(max_size=5),
                                    st.integers(), st.just({})))

    if draw(st.booleans()):
        rec["label"] = draw(_labels)

    if draw(st.booleans()):
        rec["sample_id"] = draw(st.one_of(_text, st.integers()))

    # metadata: absent, empty, with/without a resume_id (present or falsy).
    meta_choice = draw(st.integers(min_value=0, max_value=4))
    if meta_choice == 1:
        rec["metadata"] = {}
    elif meta_choice == 2:
        rec["metadata"] = {"resume_id": draw(_text.filter(lambda s: s != ""))}
    elif meta_choice == 3:
        rec["metadata"] = {"resume_id": ""}  # falsy -> should be re-injected
    elif meta_choice == 4:
        rec["metadata"] = {"original_label": draw(_text)}
    return rec


# ---------------------------------------------------------------------------
# The property.
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 13: The career adapter is byte-identical to the pre-seam pipeline
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(record=career_records(), line_number=st.integers(min_value=0, max_value=10_000))
def test_property_13_career_adapter_byte_identical(record: Dict[str, Any], line_number: int):
    """The seam (career adapter) matches the pre-seam reference model exactly,
    on both the explicit ``domain_adapter="career"`` path and the
    ``domain_adapter``-absent default path."""
    expected = ref_outcome(record, line_number)

    # Fresh loaders per example so shared DataLoaderStats never leaks across
    # examples and the record dict a loader may copy is never mutated.
    career_loader = _make_career_loader()
    absent_loader = _make_absent_field_loader()

    # Both loaders must resolve to the career adapter.
    assert career_loader._adapter.name == "career"
    assert absent_loader._adapter.name == "career"

    got_career = sut_outcome(career_loader, dict(record), line_number)
    got_absent = sut_outcome(absent_loader, dict(record), line_number)

    assert got_career == expected, (
        f"career-adapter path diverged from pre-seam model\n"
        f"record={record!r}\nexpected={expected}\ngot={got_career}"
    )
    assert got_absent == expected, (
        f"domain_adapter-absent path diverged from pre-seam model\n"
        f"record={record!r}\nexpected={expected}\ngot={got_absent}"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
