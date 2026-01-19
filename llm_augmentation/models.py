"""
Data Models for LLM Career-Aware Data Augmentation System

This module defines all dataclasses used throughout the LLM augmentation system,
including source records, augmented views, and configuration structures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


# =============================================================================
# Source Record Models
# =============================================================================

@dataclass
class JobDescription:
    """Job description data structure."""
    original: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobDescription":
        """Create JobDescription from dictionary."""
        if isinstance(data, str):
            return cls(original=data)
        return cls(original=data.get("original", ""))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"original": self.original}


@dataclass
class JobData:
    """Job posting data structure."""
    title: str
    description: JobDescription
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobData":
        """Create JobData from dictionary."""
        description = data.get("description", {})
        if isinstance(description, str):
            description = JobDescription(original=description)
        else:
            description = JobDescription.from_dict(description)
        return cls(
            title=data.get("title", ""),
            description=description
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description.to_dict()
        }


@dataclass
class ResumeData:
    """Resume data structure."""
    experience: str
    original_text: str
    responsibilities: List[str]
    role: str
    skills: List[Dict[str, Any]]
    education: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumeData":
        """Create ResumeData from dictionary."""
        return cls(
            experience=data.get("experience", ""),
            original_text=data.get("original_text", ""),
            responsibilities=data.get("responsibilities", []),
            role=data.get("role", ""),
            skills=data.get("skills", []),
            education=data.get("education")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "experience": self.experience,
            "original_text": self.original_text,
            "responsibilities": self.responsibilities,
            "role": self.role,
            "skills": self.skills
        }
        if self.education is not None:
            result["education"] = self.education
        return result


@dataclass
class SourceRecord:
    """Original resume-job pair from processed data."""
    job_applicant_id: str
    resume: ResumeData
    job: JobData
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceRecord":
        """Create SourceRecord from dictionary."""
        return cls(
            job_applicant_id=data.get("job_applicant_id", ""),
            resume=ResumeData.from_dict(data.get("resume", {})),
            job=JobData.from_dict(data.get("job", {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_applicant_id": self.job_applicant_id,
            "resume": self.resume.to_dict(),
            "job": self.job.to_dict()
        }


# =============================================================================
# Augmentation Output Models
# =============================================================================

@dataclass
class AugmentationMeta:
    """Metadata about the augmentation transformation."""
    view_type: str  # "aspirational" or "foundational"
    target_level: str  # "senior", "lead", "junior", "entry"
    transformation_method: str  # "llm" or "rule_based_fallback"
    semantic_similarity: float
    technical_term_preservation: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "view_type": self.view_type,
            "target_level": self.target_level,
            "transformation_method": self.transformation_method,
            "semantic_similarity": self.semantic_similarity,
            "technical_term_preservation": self.technical_term_preservation,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AugmentationMeta":
        """Create AugmentationMeta from dictionary."""
        return cls(
            view_type=data.get("view_type", ""),
            target_level=data.get("target_level", ""),
            transformation_method=data.get("transformation_method", ""),
            semantic_similarity=data.get("semantic_similarity", 0.0),
            technical_term_preservation=data.get("technical_term_preservation", 0.0),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z")
        )


@dataclass
class AugmentedView:
    """A transformed version of a SourceRecord representing a different career level."""
    job_applicant_id: str
    resume: ResumeData
    job: JobData
    augmentation_meta: AugmentationMeta
    
    @classmethod
    def from_source_record(
        cls,
        source: SourceRecord,
        transformed_resume: ResumeData,
        transformed_job: JobData,
        meta: AugmentationMeta
    ) -> "AugmentedView":
        """Create AugmentedView from a source record and transformations."""
        return cls(
            job_applicant_id=source.job_applicant_id,
            resume=transformed_resume,
            job=transformed_job,
            augmentation_meta=meta
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_applicant_id": self.job_applicant_id,
            "resume": self.resume.to_dict(),
            "job": self.job.to_dict(),
            "_augmentation_meta": self.augmentation_meta.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AugmentedView":
        """Create AugmentedView from dictionary."""
        return cls(
            job_applicant_id=data.get("job_applicant_id", ""),
            resume=ResumeData.from_dict(data.get("resume", {})),
            job=JobData.from_dict(data.get("job", {})),
            augmentation_meta=AugmentationMeta.from_dict(
                data.get("_augmentation_meta", {})
            )
        )


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class LLMProviderConfig:
    """Configuration for the LLM provider."""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key_env: str = "OPENAI_API_KEY"
    provider_type: str = "openai"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMProviderConfig":
        """Create LLMProviderConfig from dictionary."""
        return cls(
            model_name=data.get("model_name", "gpt-4o-mini"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1024),
            api_key_env=data.get("api_key_env", "OPENAI_API_KEY"),
            provider_type=data.get("provider_type", "openai")
        )


@dataclass
class TransformationConfig:
    """Configuration for transformation behavior."""
    upward_target_level: str = "senior"
    downward_target_level: str = "junior"
    skills_mask_ratio_min: float = 0.2
    skills_mask_ratio_max: float = 0.4
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationConfig":
        """Create TransformationConfig from dictionary."""
        return cls(
            upward_target_level=data.get("upward_target_level", "senior"),
            downward_target_level=data.get("downward_target_level", "junior"),
            skills_mask_ratio_min=data.get("skills_mask_ratio_min", 0.2),
            skills_mask_ratio_max=data.get("skills_mask_ratio_max", 0.4)
        )


@dataclass
class ValidationConfig:
    """Configuration for validation thresholds."""
    min_semantic_similarity: float = 0.5
    max_semantic_similarity: float = 0.95
    min_technical_preservation: float = 0.95
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        """Create ValidationConfig from dictionary."""
        return cls(
            min_semantic_similarity=data.get("min_semantic_similarity", 0.5),
            max_semantic_similarity=data.get("max_semantic_similarity", 0.95),
            min_technical_preservation=data.get("min_technical_preservation", 0.95)
        )


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    enabled: bool = True
    max_retries: int = 3
    use_rule_based: bool = True
    retry_delay_base: float = 1.0
    retry_delay_multiplier: float = 2.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FallbackConfig":
        """Create FallbackConfig from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            max_retries=data.get("max_retries", 3),
            use_rule_based=data.get("use_rule_based", True),
            retry_delay_base=data.get("retry_delay_base", 1.0),
            retry_delay_multiplier=data.get("retry_delay_multiplier", 2.0)
        )


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 10
    save_interval: int = 100
    progress_logging_interval: int = 10
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchConfig":
        """Create BatchConfig from dictionary."""
        return cls(
            batch_size=data.get("batch_size", 10),
            save_interval=data.get("save_interval", 100),
            progress_logging_interval=data.get("progress_logging_interval", 10)
        )


@dataclass
class PathsConfig:
    """Configuration for file paths."""
    cs_skills_path: str = "dataset/cs_skills.json"
    esco_domains_path: str = "esco_it_career_domains_refined.json"
    input_file: str = "preprocess/processed_combined_enriched_data.jsonl"
    output_file: str = "llm_augmentation/augmented_data.jsonl"
    summary_file: str = "llm_augmentation/augmentation_summary.json"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathsConfig":
        """Create PathsConfig from dictionary."""
        return cls(
            cs_skills_path=data.get("cs_skills_path", "dataset/cs_skills.json"),
            esco_domains_path=data.get("esco_domains_path", "esco_it_career_domains_refined.json"),
            input_file=data.get("input_file", "preprocess/processed_combined_enriched_data.jsonl"),
            output_file=data.get("output_file", "llm_augmentation/augmented_data.jsonl"),
            summary_file=data.get("summary_file", "llm_augmentation/augmentation_summary.json")
        )


@dataclass
class PromptsConfig:
    """Configuration for LLM prompts."""
    upward_experience: str = ""
    downward_experience: str = ""
    upward_responsibilities: str = ""
    downward_responsibilities: str = ""
    upward_role: str = ""
    downward_role: str = ""
    upward_job_description: str = ""
    downward_job_description: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptsConfig":
        """Create PromptsConfig from dictionary."""
        return cls(
            upward_experience=data.get("upward_experience", ""),
            downward_experience=data.get("downward_experience", ""),
            upward_responsibilities=data.get("upward_responsibilities", ""),
            downward_responsibilities=data.get("downward_responsibilities", ""),
            upward_role=data.get("upward_role", ""),
            downward_role=data.get("downward_role", ""),
            upward_job_description=data.get("upward_job_description", ""),
            downward_job_description=data.get("downward_job_description", "")
        )


@dataclass
class LLMAugmentationConfig:
    """Main configuration for the LLM augmentation system."""
    llm_provider: LLMProviderConfig
    transformation: TransformationConfig
    validation: ValidationConfig
    fallback: FallbackConfig
    batch_processing: BatchConfig
    paths: PathsConfig
    prompts: PromptsConfig
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMAugmentationConfig":
        """Create LLMAugmentationConfig from dictionary."""
        return cls(
            llm_provider=LLMProviderConfig.from_dict(data.get("llm_provider", {})),
            transformation=TransformationConfig.from_dict(data.get("transformation", {})),
            validation=ValidationConfig.from_dict(data.get("validation", {})),
            fallback=FallbackConfig.from_dict(data.get("fallback", {})),
            batch_processing=BatchConfig.from_dict(data.get("batch_processing", {})),
            paths=PathsConfig.from_dict(data.get("paths", {})),
            prompts=PromptsConfig.from_dict(data.get("prompts", {}))
        )
    
    @classmethod
    def load_from_file(cls, config_path: str) -> "LLMAugmentationConfig":
        """Load configuration from a JSON file."""
        import json
        with open(config_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Validation Result Models
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a transformation validation."""
    is_valid: bool
    semantic_similarity: float
    technical_preservation: float
    career_level_valid: bool
    ontology_aligned: bool = True
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "is_valid": self.is_valid,
            "semantic_similarity": self.semantic_similarity,
            "technical_preservation": self.technical_preservation,
            "career_level_valid": self.career_level_valid,
            "ontology_aligned": self.ontology_aligned
        }
        if self.rejection_reason:
            result["rejection_reason"] = self.rejection_reason
        return result


# =============================================================================
# Statistics Models
# =============================================================================

@dataclass
class AugmentationStats:
    """Statistics about the augmentation process."""
    total_records_processed: int = 0
    total_views_generated: int = 0
    aspirational_views: int = 0
    foundational_views: int = 0
    llm_transformations: int = 0
    fallback_transformations: int = 0
    validation_failures: int = 0
    average_semantic_similarity: float = 0.0
    average_technical_preservation: float = 0.0
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_records_processed": self.total_records_processed,
            "total_views_generated": self.total_views_generated,
            "aspirational_views": self.aspirational_views,
            "foundational_views": self.foundational_views,
            "llm_transformations": self.llm_transformations,
            "fallback_transformations": self.fallback_transformations,
            "validation_failures": self.validation_failures,
            "average_semantic_similarity": self.average_semantic_similarity,
            "average_technical_preservation": self.average_technical_preservation,
            "processing_time_seconds": self.processing_time_seconds,
            "errors": self.errors
        }
