"""
Pipeline Integration Utilities for Enhanced Augmentation System

This module provides utilities to integrate the enhanced augmentation system
with the training pipeline, including configuration loading and parameter
application.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from augmentation.augmentation_config import AugmentationConfig, load_augmentation_config
from augmentation.dataset_augmentation_orchestrator import DatasetAugmentationOrchestrator
from augmentation.enhanced_semantic_validator import EnhancedSemanticValidator
from augmentation.metadata_synchronizer import MetadataSynchronizer

logger = logging.getLogger(__name__)


class AugmentationPipelineIntegrator:
    """
    Integrates enhanced augmentation system with training pipeline.
    
    Handles configuration loading, parameter application, and component
    initialization for seamless integration with existing training workflows.
    """
    
    def __init__(self, 
                 training_config: Dict[str, Any],
                 pipeline_config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline integrator.
        
        Args:
            training_config: Training configuration dictionary
            pipeline_config: Pipeline configuration dictionary (optional)
        """
        self.training_config = training_config
        self.pipeline_config = pipeline_config or {}
        
        # Load augmentation configuration
        self.augmentation_config = self._load_augmentation_config()
        
        # Initialize components
        self.enhanced_validator = None
        self.metadata_synchronizer = None
        self.orchestrator = None
        
    def _load_augmentation_config(self) -> AugmentationConfig:
        """Load augmentation configuration from training config"""
        
        # Try to get augmentation config path from training config
        augmentation_config_path = self.training_config.get('augmentation_config_path')
        
        # Fallback to pipeline config
        if not augmentation_config_path and self.pipeline_config:
            augmentation_config_path = self.pipeline_config.get('augmentation_config_path')
        
        # Get quality profile
        quality_profile = (
            self.training_config.get('augmentation_quality_profile') or
            self.pipeline_config.get('augmentation_quality_profile') or
            'balanced'
        )
        
        # Load configuration
        if augmentation_config_path and Path(augmentation_config_path).exists():
            logger.info(f"Loading augmentation config from: {augmentation_config_path}")
            config = load_augmentation_config(augmentation_config_path, quality_profile)
        else:
            logger.info("Using default augmentation configuration")
            config = load_augmentation_config(quality_profile=quality_profile)
        
        logger.info(f"Active augmentation quality profile: {config.active_profile}")
        return config
    
    def get_enhanced_validation_parameters(self) -> Dict[str, Any]:
        """Get enhanced validation parameters for component initialization"""
        
        active_profile = self.augmentation_config.get_active_profile()
        
        return {
            'upward_min_threshold': active_profile.validation_thresholds.upward_min,
            'upward_max_threshold': active_profile.validation_thresholds.upward_max,
            'downward_min_threshold': active_profile.validation_thresholds.downward_min,
            'downward_max_threshold': active_profile.validation_thresholds.downward_max,
            'min_transformation_quality': active_profile.quality_gates.min_transformation_quality,
            'min_metadata_consistency': active_profile.quality_gates.min_metadata_consistency,
            'min_technical_preservation': active_profile.quality_gates.min_technical_preservation,
            'min_diversity_threshold': active_profile.diversity_requirements.min_diversity_threshold,
            'max_collapse_risk': active_profile.diversity_requirements.max_collapse_risk
        }
    
    def get_orchestrator_parameters(self) -> Dict[str, Any]:
        """Get parameters for dataset augmentation orchestrator initialization"""
        
        active_profile = self.augmentation_config.get_active_profile()
        
        return {
            'enable_enhanced_validation': active_profile.enhanced_validation,
            'augmentation_config': self.augmentation_config,
            'quality_profile': active_profile.name
        }
    
    def initialize_enhanced_validator(self) -> Optional[EnhancedSemanticValidator]:
        """Initialize enhanced semantic validator with current configuration"""
        
        if not self.augmentation_config.enhanced_validation_enabled:
            logger.info("Enhanced validation disabled")
            return None
        
        validation_params = self.get_enhanced_validation_parameters()
        
        try:
            self.enhanced_validator = EnhancedSemanticValidator(**validation_params)
            logger.info("Enhanced semantic validator initialized successfully")
            return self.enhanced_validator
        except Exception as e:
            logger.error(f"Failed to initialize enhanced validator: {e}")
            return None
    
    def initialize_metadata_synchronizer(self) -> Optional[MetadataSynchronizer]:
        """Initialize metadata synchronizer if enabled"""
        
        if not self.augmentation_config.metadata_synchronization_enabled:
            logger.info("Metadata synchronization disabled")
            return None
        
        try:
            self.metadata_synchronizer = MetadataSynchronizer()
            logger.info("Metadata synchronizer initialized successfully")
            return self.metadata_synchronizer
        except Exception as e:
            logger.error(f"Failed to initialize metadata synchronizer: {e}")
            return None
    
    def initialize_orchestrator(self, 
                              esco_skills_hierarchy: Dict,
                              career_graph: Any,
                              lambda1: float = 0.3,
                              lambda2: float = 0.2) -> Optional[DatasetAugmentationOrchestrator]:
        """Initialize dataset augmentation orchestrator with enhanced configuration"""
        
        orchestrator_params = self.get_orchestrator_parameters()
        
        try:
            self.orchestrator = DatasetAugmentationOrchestrator(
                esco_skills_hierarchy=esco_skills_hierarchy,
                career_graph=career_graph,
                lambda1=lambda1,
                lambda2=lambda2,
                **orchestrator_params
            )
            logger.info("Dataset augmentation orchestrator initialized successfully")
            return self.orchestrator
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return None
    
    def apply_training_config_overrides(self) -> Dict[str, Any]:
        """Apply augmentation-related overrides to training configuration"""
        
        overrides = {}
        active_profile = self.augmentation_config.get_active_profile()
        
        # Apply enhanced validation settings
        if 'enhanced_augmentation_validation' in self.training_config:
            overrides['enhanced_augmentation_validation'] = active_profile.enhanced_validation
        
        # Apply diversity monitoring settings
        if 'augmentation_diversity_monitoring' in self.training_config:
            overrides['augmentation_diversity_monitoring'] = active_profile.diversity_monitoring
        
        # Apply metadata sync settings
        if 'augmentation_metadata_sync' in self.training_config:
            overrides['augmentation_metadata_sync'] = active_profile.metadata_sync_enabled
        
        # Apply quality gate overrides if present in training config
        if any(key.startswith('augmentation_quality_gates') for key in self.training_config.keys()):
            quality_gates = active_profile.quality_gates
            overrides.update({
                'augmentation_quality_gates': {
                    'min_transformation_quality': quality_gates.min_transformation_quality,
                    'min_metadata_consistency': quality_gates.min_metadata_consistency,
                    'min_technical_preservation': quality_gates.min_technical_preservation,
                    'min_diversity_threshold': active_profile.diversity_requirements.min_diversity_threshold,
                    'max_collapse_risk': active_profile.diversity_requirements.max_collapse_risk
                }
            })
        
        # Apply similarity threshold overrides if present in training config
        if any(key.startswith('augmentation_similarity_thresholds') for key in self.training_config.keys()):
            thresholds = active_profile.validation_thresholds
            overrides.update({
                'augmentation_similarity_thresholds': {
                    'upward_min': thresholds.upward_min,
                    'upward_max': thresholds.upward_max,
                    'downward_min': thresholds.downward_min,
                    'downward_max': thresholds.downward_max
                }
            })
        
        # Apply fallback configuration overrides if present in training config
        if any(key.startswith('augmentation_fallback_config') for key in self.training_config.keys()):
            fallback_config = active_profile.fallback_config
            overrides.update({
                'augmentation_fallback_config': {
                    'max_attempts': fallback_config.max_attempts,
                    'progressive_relaxation': fallback_config.progressive_relaxation,
                    'exclude_on_failure': fallback_config.exclude_on_failure
                }
            })
        
        if overrides:
            logger.info(f"Applied {len(overrides)} augmentation configuration overrides")
            for key, value in overrides.items():
                logger.debug(f"  {key}: {value}")
        
        return overrides
    
    def get_quality_profile_info(self) -> Dict[str, Any]:
        """Get information about the active quality profile"""
        
        active_profile = self.augmentation_config.get_active_profile()
        
        return {
            'profile_name': active_profile.name,
            'enhanced_validation': active_profile.enhanced_validation,
            'metadata_sync_enabled': active_profile.metadata_sync_enabled,
            'diversity_monitoring': active_profile.diversity_monitoring,
            'quality_gates': {
                'min_transformation_quality': active_profile.quality_gates.min_transformation_quality,
                'min_metadata_consistency': active_profile.quality_gates.min_metadata_consistency,
                'min_technical_preservation': active_profile.quality_gates.min_technical_preservation
            },
            'validation_thresholds': {
                'upward_min': active_profile.validation_thresholds.upward_min,
                'upward_max': active_profile.validation_thresholds.upward_max,
                'downward_min': active_profile.validation_thresholds.downward_min,
                'downward_max': active_profile.validation_thresholds.downward_max
            },
            'fallback_config': {
                'max_attempts': active_profile.fallback_config.max_attempts,
                'progressive_relaxation': active_profile.fallback_config.progressive_relaxation,
                'exclude_on_failure': active_profile.fallback_config.exclude_on_failure
            }
        }
    
    def validate_configuration(self) -> Tuple[bool, list]:
        """Validate the augmentation configuration"""
        
        errors = []
        
        # Check if augmentation config is properly loaded
        if not self.augmentation_config:
            errors.append("Failed to load augmentation configuration")
            return False, errors
        
        # Check if active profile exists
        try:
            active_profile = self.augmentation_config.get_active_profile()
            if not active_profile:
                errors.append("No active quality profile found")
        except Exception as e:
            errors.append(f"Error accessing active profile: {e}")
        
        # Validate training config compatibility
        if self.training_config.get('use_view_augmentation', False):
            if not active_profile.enhanced_validation and self.training_config.get('enhanced_augmentation_validation', False):
                errors.append("Enhanced validation requested but disabled in quality profile")
        
        # Check for conflicting settings
        if (self.training_config.get('augmentation_diversity_monitoring', False) and 
            not active_profile.diversity_monitoring):
            errors.append("Diversity monitoring requested but disabled in quality profile")
        
        return len(errors) == 0, errors


def create_pipeline_integrator(training_config: Dict[str, Any], 
                             pipeline_config: Optional[Dict[str, Any]] = None) -> AugmentationPipelineIntegrator:
    """
    Create and configure a pipeline integrator.
    
    Args:
        training_config: Training configuration dictionary
        pipeline_config: Pipeline configuration dictionary (optional)
        
    Returns:
        AugmentationPipelineIntegrator: Configured integrator instance
    """
    return AugmentationPipelineIntegrator(training_config, pipeline_config)


def apply_augmentation_config_to_training(training_config: Dict[str, Any],
                                        pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Apply augmentation configuration overrides to training configuration.
    
    Args:
        training_config: Training configuration dictionary
        pipeline_config: Pipeline configuration dictionary (optional)
        
    Returns:
        Dict[str, Any]: Updated training configuration with augmentation overrides
    """
    integrator = create_pipeline_integrator(training_config, pipeline_config)
    
    # Validate configuration
    is_valid, errors = integrator.validate_configuration()
    if not is_valid:
        logger.warning(f"Augmentation configuration validation failed: {errors}")
    
    # Apply overrides
    overrides = integrator.apply_training_config_overrides()
    
    # Update training config
    updated_config = training_config.copy()
    updated_config.update(overrides)
    
    return updated_config


def log_augmentation_configuration(training_config: Dict[str, Any],
                                 pipeline_config: Optional[Dict[str, Any]] = None):
    """
    Log augmentation configuration information for debugging.
    
    Args:
        training_config: Training configuration dictionary
        pipeline_config: Pipeline configuration dictionary (optional)
    """
    try:
        integrator = create_pipeline_integrator(training_config, pipeline_config)
        profile_info = integrator.get_quality_profile_info()
        
        logger.info("=== Augmentation Configuration ===")
        logger.info(f"Quality Profile: {profile_info['profile_name']}")
        logger.info(f"Enhanced Validation: {profile_info['enhanced_validation']}")
        logger.info(f"Metadata Sync: {profile_info['metadata_sync_enabled']}")
        logger.info(f"Diversity Monitoring: {profile_info['diversity_monitoring']}")
        
        logger.info("Quality Gates:")
        for gate, value in profile_info['quality_gates'].items():
            logger.info(f"  {gate}: {value}")
        
        logger.info("Validation Thresholds:")
        for threshold, value in profile_info['validation_thresholds'].items():
            logger.info(f"  {threshold}: {value}")
        
        logger.info("=== End Augmentation Configuration ===")
        
    except Exception as e:
        logger.error(f"Failed to log augmentation configuration: {e}")