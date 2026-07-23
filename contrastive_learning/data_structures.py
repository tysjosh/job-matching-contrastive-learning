"""
Core data structures for contrastive learning training system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class TrainingSample:
    """Represents a single training sample with resume-job pair and label."""
    resume: Dict[str, Any]
    job: Dict[str, Any]
    label: str  # 'positive' or 'negative'
    sample_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the training sample after initialization."""
        if self.label not in ['positive', 'negative']:
            raise ValueError(
                f"Label must be 'positive' or 'negative', got: {self.label}")

        if not isinstance(self.resume, dict):
            raise ValueError("Resume must be a dictionary")

        if not isinstance(self.job, dict):
            raise ValueError("Job must be a dictionary")

        if not self.sample_id:
            raise ValueError("Sample ID cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'resume': self.resume,
            'job': self.job,
            'label': self.label,
            'sample_id': self.sample_id,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSample':
        """Create TrainingSample from dictionary."""
        return cls(
            resume=data['resume'],
            job=data['job'],
            label=data['label'],
            sample_id=data['sample_id'],
            metadata=data.get('metadata', {})
        )


@dataclass
class ContrastiveTriplet:
    """Represents a contrastive learning triplet with anchor, positive, and negatives."""
    anchor: Dict[str, Any]  # Resume
    positive: Dict[str, Any]  # Matching job
    negatives: List[Dict[str, Any]]  # Non-matching jobs
    career_distances: List[float]  # Distance scores for pathway weighting
    view_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the contrastive triplet after initialization."""
        if not isinstance(self.anchor, dict):
            raise ValueError("Anchor must be a dictionary")

        if not isinstance(self.positive, dict):
            raise ValueError("Positive must be a dictionary")

        if not isinstance(self.negatives, list) or not self.negatives:
            raise ValueError("Negatives must be a non-empty list")

        if len(self.career_distances) != len(self.negatives):
            raise ValueError("Career distances must match number of negatives")

        for distance in self.career_distances:
            if not isinstance(distance, (int, float)) or distance < 0:
                raise ValueError(
                    "Career distances must be non-negative numbers")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'anchor': self.anchor,
            'positive': self.positive,
            'negatives': self.negatives,
            'career_distances': self.career_distances,
            'view_metadata': self.view_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContrastiveTriplet':
        """Create ContrastiveTriplet from dictionary."""
        return cls(
            anchor=data['anchor'],
            positive=data['positive'],
            negatives=data['negatives'],
            career_distances=data['career_distances'],
            view_metadata=data.get('view_metadata', {})
        )


@dataclass
class TrainingConfig:
    """Configuration parameters for contrastive learning training."""
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 10
    temperature: float = 0.1
    # Deprecated: mixed sampling is not used; select one strategy via use_pathway_negatives.
    # Kept for backward compatibility with existing configs.
    negative_sampling_ratio: float = 0.7
    pathway_weight: float = 2.0
    use_pathway_negatives: bool = True
    use_view_augmentation: bool = True
    checkpoint_frequency: int = 1000  # batches
    log_frequency: int = 100  # batches
    shuffle_data: bool = True
    # Text encoder configuration
    text_encoder_model: str = 'all-MiniLM-L6-v2'  # SentenceTransformer model name
    text_encoder_device: Optional[str] = None  # Auto-detect if None
    
    # Embedding cache configuration
    embedding_cache_size: int = 10000  # Maximum number of embeddings to cache
    enable_embedding_preload: bool = True  # Whether to preload embeddings before training
    clear_cache_between_epochs: bool = True  # Clear cache between epochs to prevent memory leaks
    # On-disk location of the preloaded text-embedding cache. Defaults to the
    # historical shared path so career behavior is unchanged; per-dataset runs
    # (e.g. the CVE domain) point this under their own isolated output dir so
    # datasets never share a cache file (domain isolation).
    embedding_cache_path: str = "embedding_cache/text_embeddings.pt"
    
    # View augmentation specific settings
    max_resume_views: int = 5  # Maximum number of resume views to generate
    max_job_views: int = 10  # Maximum number of job views to generate
    # Use original data if augmentation fails
    fallback_on_augmentation_failure: bool = True
    # Distance threshold settings for pathway-aware negatives
    hard_negative_max_distance: float = 2.0  # Maximum distance for hard negatives
    # Maximum distance for medium negatives
    medium_negative_max_distance: float = 4.0
    # Maximum negatives sampled per anchor
    max_negatives_per_anchor: int = 20
    # ESCO graph configuration
    # Path to ESCO graph file (.gexf format) — used for career graph / pathway negatives
    esco_graph_path: Optional[str] = None
    # Path to full ESCO knowledge graph (.gexf) — used for skill-level ontology matching
    # Falls back to esco_graph_path if not set
    esco_kg_path: Optional[str] = None
    
    # Global negative sampling configuration
    global_negative_sampling: bool = False  # Enable global negative sampling
    global_negative_pool_size: int = 1000   # Max jobs to keep in memory
    
    # Research-grade model configuration
    freeze_text_encoder: bool = True        # Freeze SentenceTransformer to avoid catastrophic forgetting
    projection_dim: int = 128              # Smaller projection to reduce overfitting (was 256)
    projection_dropout: float = 0.1        # Dropout for regularization
    weight_decay: float = 0.0              # L2 regularization for optimizer (0.001 recommended)
    
    # NEW: Structured features configuration
    use_structured_features: bool = False   # Enable explicit level encoding and structured features
    structured_feature_dim: int = 32        # Dimension of encoded structured features
    
    # NEW: 2-Phase Training Configuration
    training_phase: str = "supervised"  # "self_supervised" | "supervised" | "fine_tuning"
    
    # NEW: Self-supervised training settings
    use_augmentation_labels_only: bool = False  # Use only augmentation-generated positive pairs
    augmentation_positive_ratio: float = 1.0    # Percentage of augmented positives to use (0.0-1.0)
    
    # NEW: Fine-tuning configuration
    pretrained_model_path: Optional[str] = None  # Path to pre-trained contrastive model
    freeze_contrastive_layers: bool = True       # Freeze contrastive encoder during fine-tuning
    classification_dropout: float = 0.1          # Dropout for classification head
    
    # NEW: Enhanced augmentation configuration
    augmentation_config_path: Optional[str] = None  # Path to augmentation configuration file
    augmentation_quality_profile: str = "balanced"  # Quality profile: fast, balanced, high_quality
    enhanced_augmentation_validation: bool = True   # Enable enhanced validation
    augmentation_diversity_monitoring: bool = True  # Enable diversity monitoring
    augmentation_metadata_sync: bool = True         # Enable metadata synchronization
    
    # NEW: Augmentation quality gates (optional detailed configuration)
    augmentation_quality_gates: Optional[Dict[str, float]] = None
    augmentation_similarity_thresholds: Optional[Dict[str, float]] = None
    augmentation_fallback_config: Optional[Dict[str, Any]] = None
    
    # Validation configuration
    validation_path: Optional[str] = None  # Path to validation dataset (JSONL format)
    validate_every_n_epochs: int = 1       # Run validation every N epochs

    # Ontology-aware loss weighting (uses precomputed ESCO enrichment scores)
    ontology_weight: float = 0.0           # 0.0 = disabled, 0.3 = moderate, 0.5 = strong
    ot_distance_scale: float = 10.0        # Normalization scale for OT distance
    use_ot_distance: bool = True           # Include OT distance in ontology weight (false = only ontology_similarity)
    # Decouple skill-level ontology NEGATIVE selection from the sample-level
    # ontology WEIGHT. Historically the OntologySkillMatcher was built only when
    # ontology_weight > 0, so a "no sample weighting" run silently lost skill-level
    # ontology negatives too. Set True to build the matcher (→ skill-level ontology
    # negatives) regardless of ontology_weight, enabling a clean ontology-guided
    # ordinal ablation. Default False keeps existing career/CVE runs byte-identical.
    ontology_guided_negatives: bool = False

    # Phase 1 loss function selection
    loss_type: str = "infonce"             # "infonce" (standard), "wasserstein" (graduated), "hybrid" (infonce + ws2), or "ordinal" (OCL)
    ws2_weight: float = 0.3               # Weight for WS2 component in hybrid loss (0.0-1.0)

    # Ordinal contrastive loss (OCL) configuration
    ordinal_alpha: float = 0.5            # φ-guided margin scale: m₁(φ) = α·(1 − φ)
    ordinal_lambda1: float = 1.0          # Weight for L₂ (good_fit vs potential_fit margin)
    ordinal_lambda2: float = 1.0          # Weight for L₃ (potential_fit vs no_fit margin)
    ordinal_m2: float = 0.3              # Fixed margin for L₃ (potential_fit above no_fit)
    ordinal_fixed_m1: bool = False        # If True, L₂ uses fixed margin (ordinal_m2) instead of φ-guided m₁=α·(1−φ)
    ordinal_curriculum_switch: float = 0.3 # Fraction of epochs before enabling L₂ + L₃ (0.3 = 30%)

    # Resume-grouped batching: keep same-resume records in one batch so the
    # query-anchored ordinal loss sees graded siblings.
    #   None  -> auto (enabled when loss_type == "ordinal")
    #   True  -> force on;  False -> force off (e.g., ordinal ablation)
    group_by_resume: Optional[bool] = None

    # Encoder sequence-length cap used ONLY when fine-tuning the encoder
    # (freeze_text_encoder=False). Attention activations scale with seq_len^2
    # and are retained for backprop, so long resume texts cause GPU OOM.
    unfrozen_max_seq_length: int = 256

    # General encoder seq-length cap applied in BOTH frozen and unfrozen modes
    # when set (overrides unfrozen_max_seq_length). Used for the seq-length
    # control experiment (e.g., frozen encoder at 256 to isolate truncation).
    encoder_max_seq_length: Optional[int] = None

    # Enhanced φ configuration
    phi_essential_weight: float = 1.0      # Weight for essential skills in φ denominator
    phi_optional_weight: float = 0.5       # Weight for optional skills in φ denominator
    phi_use_weighted: bool = False          # If True, weight essential > optional in φ computation
    esco_relations_path: Optional[str] = None  # Path to occupationSkillRelations_en.csv

    # Adaptive margin annealing: λ₁(t) = λ₁ · (1 − anneal_rate · t/T)
    margin_anneal: bool = False            # If True, decay λ₁ over training
    margin_anneal_rate: float = 1.0        # 1.0 = full decay to 0; 0.5 = decay to 50%

    # Confidence-gated margins: only apply φ-derived L₂ when φ < threshold
    phi_gate_threshold: float = 1.0        # 1.0 = no gating (always apply); 0.25 = only low-φ tuples

    # ISCO group distance for negative selection and loss weighting
    use_isco_negatives: bool = False        # Use ISCO group distance in negative bucketing
    isco_weight: float = 0.4               # Weight for ISCO distance in combined signal (0=skill only, 1=ISCO only)
    isco_loss_weight: bool = False          # Include ISCO proximity in loss weighting
    isco_only_weight: bool = False          # Use ONLY ISCO proximity for loss weighting (skip skill signals)
    esco_occupations_path: Optional[str] = None  # Path to occupations_en.csv

    # Skill reuse level downweighting in similarity computation
    use_reuse_weighting: bool = False       # Downweight transversal skills in similarity
    esco_skills_path: Optional[str] = None  # Path to skills_en.csv
    reuse_weight_transversal: float = 0.3   # Weight for transversal skills
    reuse_weight_cross_sector: float = 0.6  # Weight for cross-sector skills
    reuse_weight_sector_specific: float = 1.0  # Weight for sector-specific skills

    # Negative selection curriculum control
    negative_curriculum: bool = True       # True = shift hard/medium/easy ratios over epochs; False = fixed ratios
    negative_scheduler: str = "fixed"      # "fixed" | "linear_easy_to_hard" | "adaptive_val_dgp" | "performance_gated_triplet"
    negative_hard_ratio: float = 0.33      # Fixed hard ratio when negative_scheduler="fixed"
    negative_medium_ratio: float = 0.34    # Fixed medium ratio when negative_scheduler="fixed"
    negative_easy_ratio: float = 0.33      # Fixed easy ratio when negative_scheduler="fixed"

    # Phase 2 class imbalance handling
    pos_class_weight: float = 0.0          # 0.0 = disabled, 2.5 = recommended for 28% positive ratio

    # Reproducibility
    training_seed: int = 42                 # Random seed for reproducibility (overridable via CLI --seed)

    # ConFit-inspired improvements
    use_symmetric_loss: bool = False        # If True, compute L_R + L_J (both directions)
    use_in_batch_negatives: bool = False    # If True, use in-batch negatives instead of global pool
    use_rejection_hard_negatives: bool = False  # If True, sample hard negatives from explicit rejections
    rejection_hard_neg_count: int = 4       # Number of rejection-based hard negatives per anchor
    positive_only_batches: bool = False     # If True, only load positive samples into batches (ConFit-style)
    global_resume_pool_size: int = 1000     # Max resumes to keep in memory for symmetric loss reverse direction

    # ------------------------------------------------------------------
    # CVE domain / multi-domain support (additive — defaults preserve the
    # existing career-domain behavior byte-for-byte). See spec
    # cve-vulnerability-ranking. `max_negatives_per_anchor` and
    # `freeze_text_encoder` already exist above with career-safe defaults and
    # are reused for the CVE Run_Config.
    # ------------------------------------------------------------------
    # Active Domain_Adapter selected via the Domain_Adapter_Seam. "career"
    # reproduces today's resume/job/label handling exactly.
    domain_adapter: str = "career"
    # Data split strategy for the CVE Data_Splitter: "stratified" (default,
    # matches data_splits_v7), "temporal", or "random".
    split_strategy: str = "stratified"
    # Seed for reproducible splits and seeded negative/positive selection.
    split_seed: int = 42
    # Train/validation/test split proportions (percent, sum ~100). Default
    # 80/10/10 mirrors the existing career pipeline.
    split_proportions: Dict[str, float] = field(
        default_factory=lambda: {"train": 80, "validation": 10, "test": 10})
    # Per-tier negative ratios for the CVE Negative_Selector; normalized to
    # sum to 1 at selection time.
    negative_tier_ratios: Dict[str, float] = field(
        default_factory=lambda: {"hard": 0.34, "medium": 0.33, "easy": 0.33})
    # CVE data source paths (None keeps the career domain unaffected).
    cve_csv_path: Optional[str] = None
    cve_profiles_path: Optional[str] = None
    cve_denominator_pools_path: Optional[str] = None
    cyber_kg_path: Optional[str] = None
    # CVE sample-level loss weighting (only active when ontology_weight > 0):
    # by default the per-sample weight uses ONLY the label-completeness quality
    # tier (an independent data-quality signal). Setting this True additionally
    # modulates the weight by the anchor↔positive ontology overlap — an opt-in
    # ABLATION, since that overlap is the same signal that selected the positive
    # (self-referential), so it is off by default.
    cve_ontology_overlap_weighting: bool = False
    # CVE Stage 2 classification heads: weight the losses by class frequency
    # (sqrt-inverse-freq CE for priority_band, capped pos_weight BCE for
    # in_kev/ransomware) to counter majority-class collapse. On by default.
    cve_class_balanced_heads: bool = True
    # Upper bound on the binary pos_weight. Raw neg/pos reaches ~1000x on the
    # full data and destabilizes training; capping keeps up-weighting useful.
    cve_pos_weight_cap: float = 10.0
    # CVE Stage 2 classification heads: use focal loss (imbalance-robust) instead
    # of weighted CE / pos_weight BCE. Focal down-weights easy majority examples
    # by (1-p_t)^gamma, which fixes the majority-class collapse at the extreme
    # full-data skews (band 91.6%, in_kev ~1:212, ransomware ~1:1092) without the
    # training instability of huge pos_weights. On by default; set False to ablate.
    cve_focal_loss: bool = True
    cve_focal_gamma: float = 2.0
    # CVE Stage 1 positive-pair signal (which CVEs are pulled together by the
    # contrastive objective). The default "ontology" pairs anchors that share a
    # CWE/CPE/vendor — a signal that is roughly orthogonal to priority, so it
    # degrades priority-band separation vs. the frozen baseline (observed:
    # separation_ratio 0.62 -> 0.35). "priority_band" pairs anchors that share
    # the same priority_band label (supervised-contrastive / SupCon style),
    # realigning Stage 1 with the downstream ranking/classification target.
    # "priority_band_and_ontology" prefers a same-band positive that also shares
    # an ontology token, falling back to same-band-only, then excluding.
    cve_positive_signal: str = "ontology"
    # CVE Stage 1 negatives: bias selection toward CVEs whose priority_band
    # differs from the anchor's (within each ontology tier and the random
    # fallback). Mirrors supervised-contrastive practice (negatives should be
    # other classes) and complements cve_positive_signal="priority_band". Off by
    # default (unchanged ontology-tiered behavior). Requires the Stage 1 driver's
    # band lookup; a no-op otherwise.
    cve_negative_cross_band: bool = False
    # CVE Stage 1 ORDINAL contrastive: the priority_band ordinal order, lowest→
    # highest severity. Used only when loss_type="ordinal" on the CVE domain to
    # rank each candidate's band relative to the anchor's (band-proximity graded
    # relevance: same band → good, adjacent → potential, distant → no). Bands not
    # in this list get no rank and are treated as the no_fit floor. Career runs
    # never read this (their levels come from good_fit/potential_fit/no_fit).
    cve_band_order: List[str] = field(
        default_factory=lambda: ["watch", "low", "medium", "high", "critical"]
    )
    # CVE ordinal Stage 1 negative BAND QUOTAS (primary fix for band skew). When
    # on, negatives are drawn by band-distance to the anchor (adjacent bands →
    # graded level 1, distant → level 0) from the whole split's band index rather
    # than the ontology pools, so every query gets genuinely graded candidates
    # even though ~91% of CVEs share the majority band. Same-band candidates are
    # used only as last-resort fill. Requires the Stage 1 driver's band lookup +
    # cve_band_order; a no-op otherwise. Off by default (ontology-tiered negatives).
    cve_negative_band_quota: bool = False
    # Share of the negative budget targeted at ADJACENT bands (distance 1); the
    # remainder targets distant bands (distance >= 2). 0.5 = an even split.
    cve_negative_band_quota_adjacent_ratio: float = 0.5
    # CVE Stage 2 decision calibration: after training, fit the classification
    # decision rules on the VALIDATION split instead of using fixed rules that
    # collapse under class imbalance (binary heads thresholded at 0.5; band by
    # plain argmax). Binary heads get the probability threshold that maximizes
    # validation macro-F1; the band head gets a -log(prior) logit adjustment with
    # a strength chosen on validation. Applied at predict time on the test split
    # (no leakage). Off by default (unchanged 0.5 / argmax behavior).
    cve_calibrate_thresholds: bool = False

    # ------------------------------------------------------------------
    # ORCA training mode (additive — every field defaults to the OFF /
    # OSCAR-equivalent value so the existing career InfoNCE/ordinal path
    # stays byte-identical). ORCA activates only when ``orca_enabled`` is
    # True; all ORCA code lives under the top-level ``orca/`` package and is
    # gated through the loss-engine factory. See spec ``orca`` design B.9.
    # ------------------------------------------------------------------
    orca_enabled: bool = False              # master gate; False => byte-identical career path
    orca_variant: str = "denominator"       # one of the six variant switches (see design B.9)
    orca_r_min: float = 0.05                # reliability floor in the denominator
    orca_omega: float = 0.5                 # ISCO vs ESCO mix in d_ont
    orca_beta: float = 1.0                  # ontology distance sharpness (r_ont)
    orca_gamma_enc: float = 5.0             # encoder similarity sharpness (r_enc)
    orca_lambda_ont: float = 0.5            # weak-target weights (renormalized if a signal missing)
    orca_lambda_enc: float = 0.5
    orca_lambda_hist: float = 0.0
    orca_use_history: bool = False          # True only for temporal/repeated-interaction data
    orca_eta_rel: float = 1.0               # weight on reliability BCE loss
    orca_use_alignment: bool = False        # ORCA-Full only
    orca_lambda_align: float = 0.1
    # d_esco,d_isco,d_ot,s_esco,s_isco (=5) plus the coverage-feature width; the
    # scalar-only layout uses 5. Callers with coverage features set this to
    # ``5 + coverage_dim``.
    orca_feature_dim: int = 5
    orca_use_ontology_features: bool = True  # False => ORCA-NoOntology feature layout
    orca_adaptive_sampling: bool = True      # False => ORCA-Denominator (OSCAR bucket sampling)
    orca_sampling_epsilon: float = 0.1
    orca_gamma_s: float = 0.5               # curriculum: 0.5 early -> 1.0/2.0 later
    orca_random_mix: float = 0.5            # curriculum: high early -> low later
    orca_warmup_epochs: int = 3
    orca_reliability_epochs: int = 3
    orca_joint_epochs: int = 10
    # Fall back to live cached embeddings if the warmup snapshot is unavailable
    # in Phase 3/4; otherwise a missing snapshot raises a configuration error.
    orca_allow_live_warmup_fallback: bool = False
    # When True, ORCA per-negative ontology-feature capture also computes the
    # (expensive Sinkhorn) optimal-transport distance ``d_ot`` for each negative.
    # Default False keeps negative processing cheap; ``d_ot`` is then left at a
    # neutral 0.0 in the captured feature vector.
    orca_capture_ot_distance: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")

        if not 0 <= self.negative_sampling_ratio <= 1:
            raise ValueError("Negative sampling ratio must be between 0 and 1")

        if self.pathway_weight < 0:
            raise ValueError("Pathway weight must be non-negative")

        if self.checkpoint_frequency <= 0:
            raise ValueError("Checkpoint frequency must be positive")

        if self.log_frequency <= 0:
            raise ValueError("Log frequency must be positive")

        if self.max_resume_views <= 0:
            raise ValueError("Max resume views must be positive")

        if self.max_job_views <= 0:
            raise ValueError("Max job views must be positive")

        if not self.text_encoder_model or not isinstance(self.text_encoder_model, str):
            raise ValueError("Text encoder model must be a non-empty string")

        if self.hard_negative_max_distance <= 0:
            raise ValueError("Hard negative max distance must be positive")

        if self.medium_negative_max_distance <= self.hard_negative_max_distance:
            raise ValueError(
                "Medium negative max distance must be greater than hard negative max distance")

        if self.max_negatives_per_anchor <= 0:
            raise ValueError("Max negatives per anchor must be positive")
        
        # NEW: Validate 2-phase training configuration
        self._validate_training_phase_config()

    def _validate_training_phase_config(self):
        """Validate 2-phase training specific configuration parameters."""
        # Validate training_phase
        valid_phases = ["self_supervised", "supervised", "fine_tuning"]
        if self.training_phase not in valid_phases:
            raise ValueError(
                f"training_phase must be one of {valid_phases}, got: {self.training_phase}")
        
        # Validate augmentation_positive_ratio
        if not 0.0 <= self.augmentation_positive_ratio <= 1.0:
            raise ValueError(
                f"augmentation_positive_ratio must be between 0.0 and 1.0, got: {self.augmentation_positive_ratio}")
        
        # Validate classification_dropout
        if not 0.0 <= self.classification_dropout <= 1.0:
            raise ValueError(
                f"classification_dropout must be between 0.0 and 1.0, got: {self.classification_dropout}")
        
        # Phase-specific validation
        if self.training_phase == "fine_tuning":
            if self.pretrained_model_path is None:
                raise ValueError(
                    "pretrained_model_path is required when training_phase is 'fine_tuning'")
            
            if not isinstance(self.pretrained_model_path, str) or not self.pretrained_model_path.strip():
                raise ValueError(
                    "pretrained_model_path must be a non-empty string when training_phase is 'fine_tuning'")
        
        if self.training_phase == "self_supervised":
            if self.use_augmentation_labels_only and self.augmentation_positive_ratio == 0.0:
                raise ValueError(
                    "augmentation_positive_ratio cannot be 0.0 when use_augmentation_labels_only is True")
        
        # Backward compatibility check - ensure existing configurations work
        if self.training_phase == "supervised":
            # In supervised mode, these parameters should not conflict with existing behavior
            if self.use_augmentation_labels_only:
                # This could potentially break existing workflows, so warn but allow
                pass
            
            if self.pretrained_model_path is not None:
                # This is fine - user might want to load a pre-trained model for regular training
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary, ignoring unknown keys."""
        import dataclasses as _dc
        valid_keys = {f.name for f in _dc.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install PyYAML")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {file_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, file_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {file_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install PyYAML")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingResults:
    """Results and metrics from contrastive learning training."""
    final_loss: float
    epoch_losses: List[float]
    training_time: float
    total_batches: int
    total_samples: int
    checkpoint_paths: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_losses: List[float] = field(default_factory=list)  # Validation loss per epoch

    def __post_init__(self):
        """Validate training results."""
        if self.final_loss < 0:
            raise ValueError("Final loss cannot be negative")

        if any(loss < 0 for loss in self.epoch_losses):
            raise ValueError("Epoch losses cannot be negative")

        if self.training_time < 0:
            raise ValueError("Training time cannot be negative")

        if self.total_batches < 0:
            raise ValueError("Total batches cannot be negative")

        if self.total_samples < 0:
            raise ValueError("Total samples cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'final_loss': self.final_loss,
            'epoch_losses': self.epoch_losses,
            'validation_losses': self.validation_losses,
            'training_time': self.training_time,
            'total_batches': self.total_batches,
            'total_samples': self.total_samples,
            'checkpoint_paths': self.checkpoint_paths,
            'metrics': self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingResults':
        """Create TrainingResults from dictionary."""
        return cls(
            final_loss=data['final_loss'],
            epoch_losses=data['epoch_losses'],
            training_time=data['training_time'],
            total_batches=data['total_batches'],
            total_samples=data['total_samples'],
            checkpoint_paths=data['checkpoint_paths'],
            metrics=data.get('metrics', {}),
            validation_losses=data.get('validation_losses', [])
        )

    def save_json(self, file_path: str) -> None:
        """Save results to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, file_path: str) -> 'TrainingResults':
        """Load results from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)
