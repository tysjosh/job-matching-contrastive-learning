#!/usr/bin/env python3
"""
Training mode detection system for two-phase training integration
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .pipeline_config import PipelineConfig, TwoPhaseTrainingConfig


class TrainingMode(Enum):
    """Enumeration of supported training modes"""
    SINGLE_PHASE = "single_phase"
    TWO_PHASE = "two_phase"


class TrainingModeDetector:
    """
    Detects and validates training mode configuration for pipeline execution.
    
    Analyzes configuration to determine if two-phase training should be used,
    validates configuration completeness, and provides fallback to single-phase
    mode for backward compatibility.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the training mode detector.
        
        Args:
            logger: Optional logger instance for detailed logging
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def detect_training_mode(self, config: PipelineConfig) -> TrainingMode:
        """
        Detect the appropriate training mode based on configuration.
        
        Args:
            config: Pipeline configuration to analyze
            
        Returns:
            TrainingMode: Detected training mode (SINGLE_PHASE or TWO_PHASE)
        """
        self.logger.info("Analyzing configuration for training mode detection")
        
        # Check if two-phase training is explicitly enabled
        if not config.two_phase_training.enabled:
            self.logger.info("Two-phase training is disabled, using single-phase mode")
            return TrainingMode.SINGLE_PHASE
        
        # Check if required two-phase configuration is present
        phase1_path = config.get_effective_phase1_config_path()
        phase2_path = config.get_effective_phase2_config_path()
        
        if not phase1_path or not phase2_path:
            self.logger.warning(
                "Two-phase training enabled but missing config paths, "
                "falling back to single-phase mode"
            )
            return TrainingMode.SINGLE_PHASE
        
        # Validate that config files exist
        if not Path(phase1_path).exists():
            self.logger.warning(
                f"Phase 1 config file not found: {phase1_path}, "
                "falling back to single-phase mode"
            )
            return TrainingMode.SINGLE_PHASE
        
        if not Path(phase2_path).exists():
            self.logger.warning(
                f"Phase 2 config file not found: {phase2_path}, "
                "falling back to single-phase mode"
            )
            return TrainingMode.SINGLE_PHASE
        
        self.logger.info("Two-phase training configuration detected and validated")
        return TrainingMode.TWO_PHASE
    
    def validate_two_phase_configuration(self, config: PipelineConfig) -> Tuple[bool, List[str]]:
        """
        Validate completeness of two-phase training configuration.
        
        Args:
            config: Pipeline configuration to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic configuration validation
        config_errors = config.validate_configuration()
        errors.extend(config_errors)
        
        # Two-phase specific validation
        if config.two_phase_training.enabled:
            # Validate phase configuration compatibility
            compatibility_errors = config.validate_phase_compatibility()
            errors.extend(compatibility_errors)
            
            # Additional validation for two-phase parameters
            two_phase_errors = self._validate_two_phase_parameters(config.two_phase_training)
            errors.extend(two_phase_errors)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info("Two-phase configuration validation passed")
        else:
            self.logger.error(f"Two-phase configuration validation failed with {len(errors)} errors")
            for error in errors:
                self.logger.error(f"  - {error}")
        
        return is_valid, errors
    
    def _validate_two_phase_parameters(self, two_phase_config: TwoPhaseTrainingConfig) -> List[str]:
        """
        Validate two-phase specific parameters.
        
        Args:
            two_phase_config: Two-phase training configuration
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate data strategies
        valid_phase1_strategies = ["augmentation_only", "all_data"]
        if two_phase_config.phase1_data_strategy not in valid_phase1_strategies:
            errors.append(
                f"Invalid phase1_data_strategy: {two_phase_config.phase1_data_strategy}. "
                f"Must be one of {valid_phase1_strategies}"
            )
        
        valid_phase2_strategies = ["labeled_only", "all_data"]
        if two_phase_config.phase2_data_strategy not in valid_phase2_strategies:
            errors.append(
                f"Invalid phase2_data_strategy: {two_phase_config.phase2_data_strategy}. "
                f"Must be one of {valid_phase2_strategies}"
            )
        
        # Validate checkpoint save strategy
        valid_checkpoint_strategies = ["best_only", "all_epochs", "last_only"]
        if two_phase_config.checkpoint_save_strategy not in valid_checkpoint_strategies:
            errors.append(
                f"Invalid checkpoint_save_strategy: {two_phase_config.checkpoint_save_strategy}. "
                f"Must be one of {valid_checkpoint_strategies}"
            )
        
        # Validate output directories are different
        if two_phase_config.phase1_output_dir == two_phase_config.phase2_output_dir:
            errors.append("Phase 1 and Phase 2 output directories must be different")
        
        return errors
    
    def get_fallback_reason(self, config: PipelineConfig) -> Optional[str]:
        """
        Get the reason why two-phase training would fall back to single-phase.
        
        Args:
            config: Pipeline configuration to analyze
            
        Returns:
            Optional[str]: Reason for fallback, or None if two-phase is viable
        """
        if not config.two_phase_training.enabled:
            return "Two-phase training is disabled in configuration"
        
        phase1_path = config.get_effective_phase1_config_path()
        phase2_path = config.get_effective_phase2_config_path()
        
        if not phase1_path:
            return "Phase 1 configuration path is not specified"
        
        if not phase2_path:
            return "Phase 2 configuration path is not specified"
        
        if not Path(phase1_path).exists():
            return f"Phase 1 configuration file not found: {phase1_path}"
        
        if not Path(phase2_path).exists():
            return f"Phase 2 configuration file not found: {phase2_path}"
        
        # Check for validation errors
        is_valid, errors = self.validate_two_phase_configuration(config)
        if not is_valid:
            return f"Configuration validation failed: {'; '.join(errors[:3])}"
        
        return None
    
    def analyze_configuration_completeness(self, config: PipelineConfig) -> Dict[str, Any]:
        """
        Analyze the completeness of configuration for both training modes.
        
        Args:
            config: Pipeline configuration to analyze
            
        Returns:
            Dict[str, Any]: Analysis results including mode, validation status, and recommendations
        """
        detected_mode = self.detect_training_mode(config)
        fallback_reason = self.get_fallback_reason(config)
        
        analysis = {
            "detected_mode": detected_mode.value,
            "two_phase_enabled": config.two_phase_training.enabled,
            "fallback_reason": fallback_reason,
            "configuration_complete": fallback_reason is None,
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        if detected_mode == TrainingMode.SINGLE_PHASE and config.two_phase_training.enabled:
            analysis["recommendations"].append(
                "Two-phase training is enabled but falling back to single-phase. "
                f"Reason: {fallback_reason}"
            )
        
        if detected_mode == TrainingMode.TWO_PHASE:
            # Validate configuration completeness
            is_valid, errors = self.validate_two_phase_configuration(config)
            analysis["validation_errors"] = errors
            
            if not is_valid:
                analysis["recommendations"].append(
                    "Two-phase configuration has validation errors that should be addressed"
                )
        
        # Check for backward compatibility
        if not config.two_phase_training.enabled:
            single_phase_config_exists = Path(config.training_config_path).exists()
            analysis["single_phase_config_available"] = single_phase_config_exists
            
            if not single_phase_config_exists:
                analysis["recommendations"].append(
                    f"Single-phase training config not found: {config.training_config_path}"
                )
        
        self.logger.info(f"Configuration analysis complete: mode={detected_mode.value}")
        
        return analysis
    
    def load_phase_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load and validate a phase-specific configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict[str, Any]: Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file has invalid JSON
            ValueError: If configuration is invalid
        """
        config_path_obj = Path(config_path)
        
        if not config_path_obj.exists():
            error_msg = f"Configuration file not found: {config_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(config_path_obj, 'r') as f:
                config_data = json.load(f)
            
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            return config_data
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file {config_path}: {e}"
            self.logger.error(error_msg)
            raise json.JSONDecodeError(error_msg, e.doc, e.pos)
        
        except Exception as e:
            error_msg = f"Error loading configuration file {config_path}: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def validate_phase_configuration(self, config_data: Dict[str, Any], phase_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a phase-specific configuration for required fields and structure.
        
        Args:
            config_data: Configuration data to validate
            phase_name: Name of the phase (for error messages)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Required top-level sections
        required_sections = ["model", "training", "data"]
        for section in required_sections:
            if section not in config_data:
                errors.append(f"{phase_name} configuration missing required section: {section}")
        
        # Validate model configuration
        if "model" in config_data:
            model_errors = self._validate_model_config(config_data["model"], phase_name)
            errors.extend(model_errors)
        
        # Validate training configuration
        if "training" in config_data:
            training_errors = self._validate_training_config(config_data["training"], phase_name)
            errors.extend(training_errors)
        
        # Validate data configuration
        if "data" in config_data:
            data_errors = self._validate_data_config(config_data["data"], phase_name)
            errors.extend(data_errors)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info(f"{phase_name} configuration validation passed")
        else:
            self.logger.error(f"{phase_name} configuration validation failed with {len(errors)} errors")
            for error in errors:
                self.logger.error(f"  - {error}")
        
        return is_valid, errors
    
    def _validate_model_config(self, model_config: Dict[str, Any], phase_name: str) -> List[str]:
        """
        Validate model configuration section.
        
        Args:
            model_config: Model configuration dictionary
            phase_name: Name of the phase (for error messages)
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Required model fields
        required_fields = ["embedding_dim", "encoder_name"]
        for field in required_fields:
            if field not in model_config:
                errors.append(f"{phase_name} model configuration missing required field: {field}")
        
        # Validate embedding dimension
        if "embedding_dim" in model_config:
            embedding_dim = model_config["embedding_dim"]
            if not isinstance(embedding_dim, int) or embedding_dim <= 0:
                errors.append(f"{phase_name} embedding_dim must be a positive integer")
        
        # Validate encoder name
        if "encoder_name" in model_config:
            encoder_name = model_config["encoder_name"]
            if not isinstance(encoder_name, str) or not encoder_name.strip():
                errors.append(f"{phase_name} encoder_name must be a non-empty string")
        
        return errors
    
    def _validate_training_config(self, training_config: Dict[str, Any], phase_name: str) -> List[str]:
        """
        Validate training configuration section.
        
        Args:
            training_config: Training configuration dictionary
            phase_name: Name of the phase (for error messages)
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Required training fields
        required_fields = ["epochs", "batch_size", "learning_rate"]
        for field in required_fields:
            if field not in training_config:
                errors.append(f"{phase_name} training configuration missing required field: {field}")
        
        # Validate epochs
        if "epochs" in training_config:
            epochs = training_config["epochs"]
            if not isinstance(epochs, int) or epochs <= 0:
                errors.append(f"{phase_name} epochs must be a positive integer")
        
        # Validate batch size
        if "batch_size" in training_config:
            batch_size = training_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append(f"{phase_name} batch_size must be a positive integer")
        
        # Validate learning rate
        if "learning_rate" in training_config:
            learning_rate = training_config["learning_rate"]
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                errors.append(f"{phase_name} learning_rate must be a positive number")
        
        return errors
    
    def _validate_data_config(self, data_config: Dict[str, Any], phase_name: str) -> List[str]:
        """
        Validate data configuration section.
        
        Args:
            data_config: Data configuration dictionary
            phase_name: Name of the phase (for error messages)
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Required data fields
        required_fields = ["input_format"]
        for field in required_fields:
            if field not in data_config:
                errors.append(f"{phase_name} data configuration missing required field: {field}")
        
        # Validate input format
        if "input_format" in data_config:
            input_format = data_config["input_format"]
            valid_formats = ["json", "csv", "parquet", "text"]
            if input_format not in valid_formats:
                errors.append(
                    f"{phase_name} input_format must be one of {valid_formats}, got: {input_format}"
                )
        
        return errors
    
    def validate_cross_phase_compatibility(self, phase1_config: Dict[str, Any], 
                                         phase2_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate compatibility between Phase 1 and Phase 2 configurations.
        
        Args:
            phase1_config: Phase 1 configuration dictionary
            phase2_config: Phase 2 configuration dictionary
            
        Returns:
            Tuple[bool, List[str]]: (is_compatible, list_of_compatibility_errors)
        """
        errors = []
        
        # Check model architecture compatibility
        if "model" in phase1_config and "model" in phase2_config:
            model_errors = self._validate_model_compatibility(
                phase1_config["model"], phase2_config["model"]
            )
            errors.extend(model_errors)
        
        # Check data format compatibility
        if "data" in phase1_config and "data" in phase2_config:
            data_errors = self._validate_data_compatibility(
                phase1_config["data"], phase2_config["data"]
            )
            errors.extend(data_errors)
        
        is_compatible = len(errors) == 0
        
        if is_compatible:
            self.logger.info("Cross-phase compatibility validation passed")
        else:
            self.logger.error(f"Cross-phase compatibility validation failed with {len(errors)} errors")
            for error in errors:
                self.logger.error(f"  - {error}")
        
        return is_compatible, errors
    
    def _validate_model_compatibility(self, phase1_model: Dict[str, Any], 
                                    phase2_model: Dict[str, Any]) -> List[str]:
        """
        Validate model compatibility between phases.
        
        Args:
            phase1_model: Phase 1 model configuration
            phase2_model: Phase 2 model configuration
            
        Returns:
            List[str]: List of compatibility errors
        """
        errors = []
        
        # Check embedding dimension compatibility
        if "embedding_dim" in phase1_model and "embedding_dim" in phase2_model:
            if phase1_model["embedding_dim"] != phase2_model["embedding_dim"]:
                errors.append(
                    f"Embedding dimensions must match between phases: "
                    f"Phase 1 = {phase1_model['embedding_dim']}, "
                    f"Phase 2 = {phase2_model['embedding_dim']}"
                )
        
        # Check encoder compatibility (should be same for optimal transfer learning)
        if "encoder_name" in phase1_model and "encoder_name" in phase2_model:
            if phase1_model["encoder_name"] != phase2_model["encoder_name"]:
                errors.append(
                    f"Encoder names should match for optimal transfer learning: "
                    f"Phase 1 = {phase1_model['encoder_name']}, "
                    f"Phase 2 = {phase2_model['encoder_name']}"
                )
        
        return errors
    
    def _validate_data_compatibility(self, phase1_data: Dict[str, Any], 
                                   phase2_data: Dict[str, Any]) -> List[str]:
        """
        Validate data compatibility between phases.
        
        Args:
            phase1_data: Phase 1 data configuration
            phase2_data: Phase 2 data configuration
            
        Returns:
            List[str]: List of compatibility errors
        """
        errors = []
        
        # Check input format compatibility
        if "input_format" in phase1_data and "input_format" in phase2_data:
            if phase1_data["input_format"] != phase2_data["input_format"]:
                errors.append(
                    f"Input formats must match between phases: "
                    f"Phase 1 = {phase1_data['input_format']}, "
                    f"Phase 2 = {phase2_data['input_format']}"
                )
        
        return errors
    
    def load_and_validate_phase_configs(self, config: PipelineConfig) -> Tuple[bool, Dict[str, Any]]:
        """
        Load and validate both phase configurations with cross-phase compatibility check.
        
        Args:
            config: Pipeline configuration containing phase config paths
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_valid, validation_results)
        """
        validation_results = {
            "phase1_loaded": False,
            "phase2_loaded": False,
            "phase1_valid": False,
            "phase2_valid": False,
            "cross_phase_compatible": False,
            "errors": [],
            "phase1_config": None,
            "phase2_config": None
        }
        
        phase1_path = config.get_effective_phase1_config_path()
        phase2_path = config.get_effective_phase2_config_path()
        
        if not phase1_path or not phase2_path:
            validation_results["errors"].append("Phase configuration paths not specified")
            return False, validation_results
        
        try:
            # Load Phase 1 configuration
            phase1_config = self.load_phase_configuration(phase1_path)
            validation_results["phase1_loaded"] = True
            validation_results["phase1_config"] = phase1_config
            
            # Validate Phase 1 configuration
            phase1_valid, phase1_errors = self.validate_phase_configuration(phase1_config, "Phase 1")
            validation_results["phase1_valid"] = phase1_valid
            validation_results["errors"].extend(phase1_errors)
            
        except Exception as e:
            validation_results["errors"].append(f"Failed to load/validate Phase 1 config: {e}")
        
        try:
            # Load Phase 2 configuration
            phase2_config = self.load_phase_configuration(phase2_path)
            validation_results["phase2_loaded"] = True
            validation_results["phase2_config"] = phase2_config
            
            # Validate Phase 2 configuration
            phase2_valid, phase2_errors = self.validate_phase_configuration(phase2_config, "Phase 2")
            validation_results["phase2_valid"] = phase2_valid
            validation_results["errors"].extend(phase2_errors)
            
        except Exception as e:
            validation_results["errors"].append(f"Failed to load/validate Phase 2 config: {e}")
        
        # Cross-phase compatibility validation
        if (validation_results["phase1_loaded"] and validation_results["phase2_loaded"] and
            validation_results["phase1_valid"] and validation_results["phase2_valid"]):
            
            compatible, compatibility_errors = self.validate_cross_phase_compatibility(
                validation_results["phase1_config"], validation_results["phase2_config"]
            )
            validation_results["cross_phase_compatible"] = compatible
            validation_results["errors"].extend(compatibility_errors)
        
        # Overall validation status
        overall_valid = (
            validation_results["phase1_valid"] and 
            validation_results["phase2_valid"] and 
            validation_results["cross_phase_compatible"]
        )
        
        return overall_valid, validation_results