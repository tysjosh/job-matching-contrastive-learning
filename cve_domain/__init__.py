"""CVE domain package.

CVE-specific logic for adapting the existing CDCL contrastive-learning pipeline
to the cybersecurity vulnerability (CVE) priority-ranking domain. All modules in
this package are additive and do not modify the career-domain CDCL core.
"""

from .data_converter import (
    MAX_VIEW_CHARS,
    DEFAULT_MAX_CPES,
    ProfileTextView,
    build_profile_text_view,
    build_profile_text_view_report,
)
from .ontology_adapter import (
    CVEOntologyAdapter,
    CVEOntologyLoadError,
    DenominatorPool,
)
from .data_splitter import (
    CVEDataSplitter,
    SplitReport,
    SPLIT_NAMES,
    DEFAULT_PROPORTIONS,
    DEFAULT_STRATEGY,
)
from .negative_selector import (
    CVENegativeSelector,
    NegativeSelectionReport,
    TIER_NAMES,
    DEFAULT_TIER_RATIOS,
)
from .positive_selector import (
    CVEPositiveSelector,
    PositiveSelectionReport,
    CASCADE_LEVELS,
)
# Importing record_adapter registers the "cve" domain adapter as a side effect.
from .record_adapter import (
    CVERecordAdapter,
    ontology_overlap_signal,
    DEFAULT_ONTOLOGY_SIGNAL_WEIGHTS,
    CVE_NEUTRAL_ONTOLOGY_SIGNAL,
)
from .stage1 import (
    Stage1ContrastivePretrainer,
    Stage1PreparationResult,
    SplitPairingReport,
    load_view_records,
    build_view_lookup,
)
from .supervised_heads import (
    CVESupervisedHeads,
    SupervisedHead,
    PRIORITY_SCORE_HEAD,
    PRIORITY_BAND_HEAD,
    IN_KEV_HEAD,
    RANSOMWARE_HEAD,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_DROPOUT,
)
from .stage2 import (
    CVEStage2Trainer,
    HeadConfiguration,
    Stage2Result,
    detect_head_configuration,
    ALL_HEADS,
    STAGE2_BEST_CHECKPOINT,
    STAGE2_REPORT_FILENAME,
)
from .run_config import (
    CVERunConfig,
    RunManifest,
    CVERunConfigError,
    CVERunConfigMissingFieldError,
    CVERunConfigReadError,
    load_run_config,
    CVE_DOMAIN_ADAPTER,
    RUN_MANIFEST_FILENAME,
)

__all__ = [
    "MAX_VIEW_CHARS",
    "DEFAULT_MAX_CPES",
    "ProfileTextView",
    "build_profile_text_view",
    "build_profile_text_view_report",
    "CVEOntologyAdapter",
    "CVEOntologyLoadError",
    "DenominatorPool",
    "CVEDataSplitter",
    "SplitReport",
    "SPLIT_NAMES",
    "DEFAULT_PROPORTIONS",
    "DEFAULT_STRATEGY",
    "CVENegativeSelector",
    "NegativeSelectionReport",
    "TIER_NAMES",
    "DEFAULT_TIER_RATIOS",
    "CVEPositiveSelector",
    "PositiveSelectionReport",
    "CASCADE_LEVELS",
    "CVERecordAdapter",
    "ontology_overlap_signal",
    "DEFAULT_ONTOLOGY_SIGNAL_WEIGHTS",
    "CVE_NEUTRAL_ONTOLOGY_SIGNAL",
    "Stage1ContrastivePretrainer",
    "Stage1PreparationResult",
    "SplitPairingReport",
    "load_view_records",
    "build_view_lookup",
    "CVESupervisedHeads",
    "SupervisedHead",
    "PRIORITY_SCORE_HEAD",
    "PRIORITY_BAND_HEAD",
    "IN_KEV_HEAD",
    "RANSOMWARE_HEAD",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_DROPOUT",
    "CVEStage2Trainer",
    "HeadConfiguration",
    "Stage2Result",
    "detect_head_configuration",
    "ALL_HEADS",
    "STAGE2_BEST_CHECKPOINT",
    "STAGE2_REPORT_FILENAME",
    "CVERunConfig",
    "RunManifest",
    "CVERunConfigError",
    "CVERunConfigMissingFieldError",
    "CVERunConfigReadError",
    "load_run_config",
    "CVE_DOMAIN_ADAPTER",
    "RUN_MANIFEST_FILENAME",
]
