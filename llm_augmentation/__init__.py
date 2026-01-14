"""
LLM-Based Career-Aware Data Augmentation System

This module implements a "Career Time Machine" approach that uses Large Language Models
to transform resume-job pairs into career progression views, generating 2x more training
data while maintaining semantic coherence and career realism.

Unlike the rule-based augmentation in the `augmentation` folder, this system leverages
LLM capabilities for more natural and contextually appropriate transformations.

The system generates:
- Aspirational View: Senior-level perspective with leadership, impact, and strategic language
- Foundational View: Junior-level perspective with learning, support, and task-focused language

Components:
- LLMAugmentationOrchestrator: Main orchestrator coordinating the augmentation pipeline
- UpwardLLMTransformer: Transforms content to senior-level perspective
- DownwardLLMTransformer: Transforms content to junior-level perspective
- JobLLMTransformer: Transforms job descriptions to match career levels
- TechnicalTermProtector: Preserves technical terms during LLM transformation
- SemanticCoherenceValidator: Validates transformation quality and semantic coherence
- RuleBasedFallback: Falls back to rule-based transformation when LLM fails
"""

__version__ = "0.1.0"

from .models import (
    SourceRecord,
    ResumeData,
    JobData,
    JobDescription,
    AugmentedView,
    AugmentationMeta,
    LLMAugmentationConfig,
    LLMProviderConfig,
    TransformationConfig,
    ValidationConfig,
    FallbackConfig,
    BatchConfig,
    PathsConfig,
    PromptsConfig,
    ValidationResult,
    AugmentationStats,
)

from .llm_client import (
    LLMClient,
    LLMClientError,
    LLMResponse,
    OpenAIClient,
    AnthropicClient,
    MockLLMClient,
    create_llm_client,
)

from .technical_term_protector import (
    TechnicalTermProtector,
    TermMapping,
)

from .semantic_validator import (
    SemanticCoherenceValidator,
    ValidationResult as SemanticValidationResult,
)

from .upward_llm_transformer import (
    UpwardLLMTransformer,
    TransformationResult,
)

from .downward_llm_transformer import (
    DownwardLLMTransformer,
)

__all__ = [
    "__version__",
    # Data Models
    "SourceRecord",
    "ResumeData",
    "JobData",
    "JobDescription",
    "AugmentedView",
    "AugmentationMeta",
    # Configuration Models
    "LLMAugmentationConfig",
    "LLMProviderConfig",
    "TransformationConfig",
    "ValidationConfig",
    "FallbackConfig",
    "BatchConfig",
    "PathsConfig",
    "PromptsConfig",
    # Validation and Stats
    "ValidationResult",
    "AugmentationStats",
    # LLM Client
    "LLMClient",
    "LLMClientError",
    "LLMResponse",
    "OpenAIClient",
    "AnthropicClient",
    "MockLLMClient",
    "create_llm_client",
    # Technical Term Protection
    "TechnicalTermProtector",
    "TermMapping",
    # Semantic Validation
    "SemanticCoherenceValidator",
    "SemanticValidationResult",
    # Upward LLM Transformer
    "UpwardLLMTransformer",
    "TransformationResult",
    # Downward LLM Transformer
    "DownwardLLMTransformer",
]
