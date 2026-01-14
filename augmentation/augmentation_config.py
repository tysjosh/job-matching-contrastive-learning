"""
Augmentation Configuration Management

This module provides configuration management for the enhanced augmentation system,
including quality profiles, validation thresholds, and integration parameters.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationThresholds:
    """Validation threshold configuration"""
    upward_min: float = 0.5
    upward_max: float = 0.8
    downward_min: float = 0.4
    downward_max: float = 0.85


@dataclass
class QualityGates:
    """Quality gate configuration"""
    min_transformation_quality: float = 0.4
    min_metadata_consistency: float = 0.7
    min_technical_preservation: float = 0.8


@dataclass
class DiversityRequirements:
    """Diversity requirement configuration"""
    min_pairwise_distance: float = 0.3
    max_collapse_risk: float = 0.2
    min_diversity_threshold: float = 0.3


@dataclass
class TransformationParameters:
    """Transformation parameter configuration"""
    upward_impact_phrase_variety: int = 10
    upward_leadership_context_probability: float = 0.7
    downward_learning_context_variety: int = 8
    downward_support_language_probability: float = 0.8
    technical_term_preservation: bool = True


@dataclass
class FallbackConfig:
    """Fallback handling configuration"""
    enabled: bool = True
    max_attempts: int = 3
    progressive_relaxation: bool = True
    exclude_on_failure: bool = True
    diversity_tracking: bool = True


@dataclass
class QualityProfile:
    """Quality vs speed profile configuration"""
    name: str
    enhanced_validation: bool
    validation_thresholds: ValidationThresholds
    quality_gates: QualityGates
    diversity_requirements: DiversityRequirements
    transformation_parameters: TransformationParameters
    fallback_config: FallbackConfig
    metadata_sync_enabled: bool = True
    diversity_monitoring: bool = True


@dataclass
class AugmentationConfig:
    """Complete augmentation system configuration"""
    enhanced_validation_enabled: bool = True
    metadata_synchronization_enabled: bool = True
    
    # Core configuration components
    validation_thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    quality_gates: QualityGates = field(default_factory=QualityGates)
    diversity_requirements: DiversityRequirements = field(default_factory=DiversityRequirements)
    transformation_parameters: TransformationParameters = field(default_factory=TransformationParameters)
    fallback_config: FallbackConfig = field(default_factory=FallbackConfig)
    
    # Quality profiles
    quality_profiles: Dict[str, QualityProfile] = field(default_factory=dict)
    active_profile: str = "balanced"
    
    # Reporting configuration
    detailed_quality_metrics: bool = True
    batch_level_validation: bool = True
    embedding_diversity_monitoring: bool = True
    transformation_statistics: bool = True
    failure_analysis: bool = True

    @classmethod
    def from_file(cls, config_path: str) -> 'AugmentationConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
        
        except FileNotFoundError:
            logger.warning(f"Augmentation config file not found: {config_path}. Using defaults.")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in augmentation config: {e}. Using defaults.")
            return cls()
        except Exception as e:
            logger.error(f"Error loading augmentation config: {e}. Using defaults.")
            return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AugmentationConfig':
        """Create configuration from dictionary"""
        
        # Extract enhanced validation settings
        enhanced_validation = config_dict.get('enhanced_validation', {})
        
        # Create validation thresholds
        similarity_thresholds = enhanced_validation.get('similarity_thresholds', {})
        upward_thresholds = similarity_thresholds.get('upward', {})
        downward_thresholds = similarity_thresholds.get('downward', {})
        
        validation_thresholds = ValidationThresholds(
            upward_min=upward_thresholds.get('min', 0.5),
            upward_max=upward_thresholds.get('max', 0.8),
            downward_min=downward_thresholds.get('min', 0.4),
            downward_max=downward_thresholds.get('max', 0.85)
        )
        
        # Create quality gates
        quality_gates_dict = enhanced_validation.get('quality_gates', {})
        quality_gates = QualityGates(
            min_transformation_quality=quality_gates_dict.get('min_transformation_quality', 0.4),
            min_metadata_consistency=quality_gates_dict.get('min_metadata_consistency', 0.7),
            min_technical_preservation=quality_gates_dict.get('min_technical_preservation', 0.8)
        )
        
        # Create diversity requirements
        diversity_dict = enhanced_validation.get('diversity_requirements', {})
        diversity_requirements = DiversityRequirements(
            min_pairwise_distance=diversity_dict.get('min_pairwise_distance', 0.3),
            max_collapse_risk=diversity_dict.get('max_collapse_risk', 0.2),
            min_diversity_threshold=diversity_dict.get('min_diversity_threshold', 0.3)
        )
        
        # Create transformation parameters
        transform_params = config_dict.get('transformation_parameters', {})
        upward_params = transform_params.get('upward_transformer', {})
        downward_params = transform_params.get('downward_transformer', {})
        
        transformation_parameters = TransformationParameters(
            upward_impact_phrase_variety=upward_params.get('impact_phrase_variety', 10),
            upward_leadership_context_probability=upward_params.get('leadership_context_probability', 0.7),
            downward_learning_context_variety=downward_params.get('learning_context_variety', 8),
            downward_support_language_probability=downward_params.get('support_language_probability', 0.8),
            technical_term_preservation=upward_params.get('technical_term_preservation', True)
        )
        
        # Create fallback configuration
        fallback_dict = config_dict.get('fallback_handling', {})
        fallback_config = FallbackConfig(
            enabled=fallback_dict.get('enabled', True),
            max_attempts=fallback_dict.get('max_attempts', 3),
            progressive_relaxation=fallback_dict.get('progressive_relaxation', True),
            exclude_on_failure=fallback_dict.get('exclude_on_failure', True),
            diversity_tracking=fallback_dict.get('diversity_tracking', True)
        )
        
        # Create quality profiles
        quality_profiles = {}
        profiles_dict = config_dict.get('quality_vs_speed_profiles', {})
        
        for profile_name, profile_config in profiles_dict.items():
            # Create profile-specific configurations
            profile_validation_thresholds = validation_thresholds
            profile_quality_gates = quality_gates
            profile_diversity_requirements = diversity_requirements
            
            # Adjust thresholds based on profile
            if profile_name == 'fast':
                profile_quality_gates = QualityGates(
                    min_transformation_quality=0.2,
                    min_metadata_consistency=0.5,
                    min_technical_preservation=0.6
                )
                profile_diversity_requirements = DiversityRequirements(
                    min_pairwise_distance=0.2,
                    max_collapse_risk=0.3,
                    min_diversity_threshold=0.2
                )
            elif profile_name == 'high_quality':
                profile_quality_gates = QualityGates(
                    min_transformation_quality=0.6,
                    min_metadata_consistency=0.8,
                    min_technical_preservation=0.9
                )
            
            quality_profiles[profile_name] = QualityProfile(
                name=profile_name,
                enhanced_validation=profile_config.get('enhanced_validation', True),
                validation_thresholds=profile_validation_thresholds,
                quality_gates=profile_quality_gates,
                diversity_requirements=profile_diversity_requirements,
                transformation_parameters=transformation_parameters,
                fallback_config=fallback_config,
                metadata_sync_enabled=profile_config.get('basic_metadata_sync', True) or profile_config.get('comprehensive_metadata_sync', True),
                diversity_monitoring=profile_config.get('diversity_monitoring', True)
            )
        
        # Extract reporting configuration
        reporting_dict = config_dict.get('reporting', {})
        
        # Extract metadata synchronization settings
        metadata_sync = config_dict.get('metadata_synchronization', {})
        
        return cls(
            enhanced_validation_enabled=enhanced_validation.get('enabled', True),
            metadata_synchronization_enabled=metadata_sync.get('enabled', True),
            validation_thresholds=validation_thresholds,
            quality_gates=quality_gates,
            diversity_requirements=diversity_requirements,
            transformation_parameters=transformation_parameters,
            fallback_config=fallback_config,
            quality_profiles=quality_profiles,
            active_profile='balanced',  # Default profile
            detailed_quality_metrics=reporting_dict.get('detailed_quality_metrics', True),
            batch_level_validation=reporting_dict.get('batch_level_validation', True),
            embedding_diversity_monitoring=reporting_dict.get('embedding_diversity_monitoring', True),
            transformation_statistics=reporting_dict.get('transformation_statistics', True),
            failure_analysis=reporting_dict.get('failure_analysis', True)
        )

    def get_active_profile(self) -> QualityProfile:
        """Get the currently active quality profile"""
        if self.active_profile in self.quality_profiles:
            return self.quality_profiles[self.active_profile]
        
        # Fallback to balanced profile or create default
        if 'balanced' in self.quality_profiles:
            return self.quality_profiles['balanced']
        
        # Create default profile
        return QualityProfile(
            name='default',
            enhanced_validation=self.enhanced_validation_enabled,
            validation_thresholds=self.validation_thresholds,
            quality_gates=self.quality_gates,
            diversity_requirements=self.diversity_requirements,
            transformation_parameters=self.transformation_parameters,
            fallback_config=self.fallback_config,
            metadata_sync_enabled=self.metadata_synchronization_enabled,
            diversity_monitoring=True
        )

    def set_active_profile(self, profile_name: str) -> bool:
        """Set the active quality profile"""
        if profile_name in self.quality_profiles:
            self.active_profile = profile_name
            logger.info(f"Switched to augmentation quality profile: {profile_name}")
            return True
        else:
            logger.warning(f"Quality profile '{profile_name}' not found. Available profiles: {list(self.quality_profiles.keys())}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'enhanced_validation_enabled': self.enhanced_validation_enabled,
            'metadata_synchronization_enabled': self.metadata_synchronization_enabled,
            'validation_thresholds': {
                'upward_min': self.validation_thresholds.upward_min,
                'upward_max': self.validation_thresholds.upward_max,
                'downward_min': self.validation_thresholds.downward_min,
                'downward_max': self.validation_thresholds.downward_max
            },
            'quality_gates': {
                'min_transformation_quality': self.quality_gates.min_transformation_quality,
                'min_metadata_consistency': self.quality_gates.min_metadata_consistency,
                'min_technical_preservation': self.quality_gates.min_technical_preservation
            },
            'diversity_requirements': {
                'min_pairwise_distance': self.diversity_requirements.min_pairwise_distance,
                'max_collapse_risk': self.diversity_requirements.max_collapse_risk,
                'min_diversity_threshold': self.diversity_requirements.min_diversity_threshold
            },
            'active_profile': self.active_profile,
            'quality_profiles': {name: profile.name for name, profile in self.quality_profiles.items()},
            'reporting': {
                'detailed_quality_metrics': self.detailed_quality_metrics,
                'batch_level_validation': self.batch_level_validation,
                'embedding_diversity_monitoring': self.embedding_diversity_monitoring,
                'transformation_statistics': self.transformation_statistics,
                'failure_analysis': self.failure_analysis
            }
        }

    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Augmentation configuration saved to: {config_path}")


def load_augmentation_config(config_path: Optional[str] = None, 
                           quality_profile: Optional[str] = None) -> AugmentationConfig:
    """
    Load augmentation configuration with optional profile selection.
    
    Args:
        config_path: Path to augmentation configuration file
        quality_profile: Quality profile to activate
        
    Returns:
        AugmentationConfig: Loaded configuration
    """
    if config_path is None:
        config_path = "config/augmentation_config.json"
    
    config = AugmentationConfig.from_file(config_path)
    
    if quality_profile:
        config.set_active_profile(quality_profile)
    
    return config


def create_default_augmentation_config() -> AugmentationConfig:
    """Create a default augmentation configuration"""
    return AugmentationConfig()


def create_fast_augmentation_config() -> AugmentationConfig:
    """Create a fast/minimal augmentation configuration"""
    config = AugmentationConfig()
    config.enhanced_validation_enabled = False
    config.quality_gates = QualityGates(
        min_transformation_quality=0.2,
        min_metadata_consistency=0.5,
        min_technical_preservation=0.6
    )
    config.diversity_requirements = DiversityRequirements(
        min_pairwise_distance=0.2,
        max_collapse_risk=0.3,
        min_diversity_threshold=0.2
    )
    config.fallback_config = FallbackConfig(
        enabled=True,
        max_attempts=1,
        progressive_relaxation=False,
        exclude_on_failure=True,
        diversity_tracking=False
    )
    return config


def create_high_quality_augmentation_config() -> AugmentationConfig:
    """Create a high-quality augmentation configuration"""
    config = AugmentationConfig()
    config.enhanced_validation_enabled = True
    config.quality_gates = QualityGates(
        min_transformation_quality=0.6,
        min_metadata_consistency=0.8,
        min_technical_preservation=0.9
    )
    config.diversity_requirements = DiversityRequirements(
        min_pairwise_distance=0.4,
        max_collapse_risk=0.1,
        min_diversity_threshold=0.4
    )
    config.fallback_config = FallbackConfig(
        enabled=True,
        max_attempts=5,
        progressive_relaxation=True,
        exclude_on_failure=True,
        diversity_tracking=True
    )
    return config