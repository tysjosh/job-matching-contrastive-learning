#!/usr/bin/env python3
"""
Complete ML Pipeline Orchestrator for Train/Validate/Test workflow
"""

import time
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

import torch
from torch.utils.data import DataLoader

from .pipeline_config import PipelineConfig, PipelineResults
from .data_splitter import DataSplitter, SplitConfig
from .evaluator import ContrastiveEvaluator, EvaluationConfig
from .data_structures import TrainingConfig
from .trainer import ContrastiveLearningTrainer
from .training_mode_detector import TrainingModeDetector, TrainingMode
from .training_strategy import TrainingStrategy, SinglePhaseStrategy, TwoPhaseStrategy, TrainingStrategyResult

# Import augmentation integration utilities
try:
    from augmentation.pipeline_integration import (
        create_pipeline_integrator, 
        apply_augmentation_config_to_training,
        log_augmentation_configuration
    )
    AUGMENTATION_INTEGRATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_INTEGRATION_AVAILABLE = False
    logger.warning("Augmentation integration not available - enhanced augmentation features disabled")

logger = logging.getLogger(__name__)


class MLPipeline:
    """Complete ML pipeline orchestrator"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = None
        self.experiment_name = config.experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Setup output directories
        self.output_dir = Path(config.output_base_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        logger.info(f"🚀 Initializing ML Pipeline: {self.experiment_name}")
        logger.info(f"📁 Output directory: {self.output_dir}")

        # Initialize training mode detection and strategy selection
        self.mode_detector = TrainingModeDetector(logger)
        self.detected_mode = self.mode_detector.detect_training_mode(config)
        self.training_strategy = self._select_training_strategy()

        # Enhanced logging for training mode and configuration validation
        logger.info(f"🎯 Training mode detected: {self.detected_mode.value}")
        
        if self.detected_mode == TrainingMode.TWO_PHASE:
            logger.info("🔄 Two-phase training mode enabled")
            # Validate two-phase configuration
            is_valid, errors = self.mode_detector.validate_two_phase_configuration(config)
            if not is_valid:
                logger.error("❌ Two-phase configuration validation failed:")
                for error in errors:
                    logger.error(f"   - {error}")
                raise ValueError(f"Two-phase configuration validation failed: {errors}")
            else:
                logger.info("✅ Two-phase configuration validation passed")
        else:
            logger.info("📈 Single-phase training mode (backward compatible)")
            fallback_reason = self.mode_detector.get_fallback_reason(config)
            if fallback_reason:
                logger.info(f"💡 Fallback reason: {fallback_reason}")

        logger.info(f"🛠️  Training strategy: {self.training_strategy.__class__.__name__}")

    def _select_training_strategy(self) -> TrainingStrategy:
        """
        Select the appropriate training strategy based on detected training mode.
        
        Returns:
            TrainingStrategy: Configured strategy instance for the detected mode
        """
        if self.detected_mode == TrainingMode.TWO_PHASE:
            logger.info("Initializing TwoPhaseStrategy")
            return TwoPhaseStrategy(self.config, self.output_dir)
        else:
            logger.info("Initializing SinglePhaseStrategy")
            return SinglePhaseStrategy(self.config, self.output_dir)

    def run_complete_pipeline(self, dataset_path: str,
                              train_data: Optional[str] = None,
                              val_data: Optional[str] = None,
                              test_data: Optional[str] = None) -> PipelineResults:
        """Execute complete train/validate/test pipeline"""

        self.start_time = time.time()
        start_datetime = datetime.now()

        logger.info("="*80)
        logger.info("🚀 STARTING COMPLETE ML PIPELINE")
        logger.info("="*80)

        try:
            # Initialize results
            results = PipelineResults(
                experiment_name=self.experiment_name,
                start_time=start_datetime.isoformat(),
                end_time="",
                total_duration=0.0
            )

            # Phase 1: Data Splitting
            data_splits = self._handle_data_splitting(
                dataset_path, train_data, val_data, test_data)
            results.data_splits = data_splits['splits']
            results.split_statistics = data_splits['statistics']

            # Phase 2: Training with Validation using Strategy Pattern
            if not self.config.skip_training:
                training_results = self._execute_training_strategy(data_splits['splits'])
                
                # Populate results with strategy output (maintaining backward compatibility)
                results.final_model_path = training_results.final_model_path
                results.best_model_path = training_results.best_model_path
                results.training_history = training_results.training_history
                results.validation_metrics = training_results.validation_metrics
                results.validation_history = training_results.validation_history
                results.checkpoint_paths = training_results.checkpoint_paths
                
                # Set training mode and add two-phase specific results if available
                results.training_mode = training_results.training_mode
                if training_results.training_mode == "two_phase":
                    results.phase1_results = training_results.phase1_results
                    results.phase2_results = training_results.phase2_results
                    results.phase_transition_metrics = training_results.phase_transition_metrics
                    results.comparative_analysis = training_results.comparative_analysis

            # Phase 3: Final Testing
            if not self.config.skip_testing:
                test_results = self._run_final_testing(
                    data_splits['splits']['test'],
                    results.best_model_path or results.final_model_path
                )
                results.test_metrics = test_results['metrics']
                results.test_detailed_metrics = test_results['detailed_metrics']
                results.test_predictions_path = test_results['predictions_path']
                results.test_embeddings_path = test_results['embeddings_path']
                results.visualization_paths.extend(
                    test_results['visualization_paths'])

            # Phase 4: Comprehensive Reporting
            report_paths = self._generate_comprehensive_report(results)
            results.report_paths = report_paths

            # Finalize results
            end_time = time.time()
            results.end_time = datetime.now().isoformat()
            results.total_duration = end_time - self.start_time
            results.config_used = self.config.to_dict()

            # Save final results
            results_path = self.output_dir / "pipeline_results.json"
            results.save_to_file(str(results_path))

            logger.info("="*80)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(
                f"⏱️  Total duration: {results.total_duration:.2f} seconds")
            logger.info(f"📊 Results saved to: {results_path}")
            logger.info("="*80)

            return results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _handle_data_splitting(self, dataset_path: str, train_data: Optional[str],
                               val_data: Optional[str], test_data: Optional[str]) -> Dict[str, Any]:
        """Handle data splitting phase"""

        if self.config.skip_data_splitting and all([train_data, val_data, test_data]):
            logger.info("📊 Using provided data splits")
            return {
                'splits': {
                    'train': train_data,
                    'validation': val_data,
                    'test': test_data
                },
                'statistics': {'message': 'Using pre-split data'}
            }

        logger.info("📊 Phase 1: Data Splitting")

        # Create split configuration
        split_config = SplitConfig(
            strategy=self.config.data_splitting.strategy,
            ratios=self.config.data_splitting.ratios,
            seed=self.config.data_splitting.seed,
            min_samples_per_split=self.config.data_splitting.min_samples_per_split,
            validate_splits=self.config.data_splitting.validate_splits
        )

        # Create output directory for splits
        splits_dir = self.output_dir / "data_splits"

        # Run data splitting
        splitter = DataSplitter(split_config)
        split_results = splitter.split_dataset(dataset_path, str(splits_dir))

        logger.info(f"✅ Data splitting complete:")
        for split_name, count in split_results.statistics['splits'].items():
            logger.info(
                f"   {split_name}: {count['count']:,} samples ({count['percentage']:.1f}%)")

        if split_results.validation_report and split_results.validation_report['overall_status'] == 'failed':
            logger.warning(
                "⚠️  Data split validation failed - check logs for details")

        return {
            'splits': split_results.splits,
            'statistics': split_results.statistics,
            'validation_report': split_results.validation_report
        }

    def _execute_training_strategy(self, data_splits: Dict[str, str]) -> 'TrainingStrategyResult':
        """
        Execute training using the selected strategy pattern.
        
        Args:
            data_splits: Dictionary containing data split paths
            
        Returns:
            TrainingStrategyResult: Results from the executed training strategy
        """
        logger.info("🏋️ Phase 2: Training with Strategy Pattern")
        logger.info(f"Using strategy: {self.training_strategy.__class__.__name__}")
        
        # Validate data splits for the selected strategy
        validation_errors = self.training_strategy.validate_data_splits(data_splits)
        if validation_errors:
            logger.error("❌ Data splits validation failed:")
            for error in validation_errors:
                logger.error(f"   - {error}")
            raise ValueError(f"Data splits validation failed: {validation_errors}")
        
        # Execute training strategy
        try:
            strategy_results = self.training_strategy.execute_training(data_splits)
            
            # Log strategy completion
            logger.info("✅ Training strategy completed successfully")
            logger.info(f"⏱️  Training time: {strategy_results.total_training_time:.2f} seconds")
            logger.info(f"📁 Final model: {strategy_results.final_model_path}")
            
            if strategy_results.has_errors():
                logger.warning(f"⚠️  Training completed with {len(strategy_results.errors)} errors")
                for error in strategy_results.errors:
                    logger.warning(f"   - {error}")
            
            if strategy_results.has_warnings():
                logger.info(f"💡 Training completed with {len(strategy_results.warnings)} warnings")
                for warning in strategy_results.warnings:
                    logger.info(f"   - {warning}")
            
            return strategy_results
            
        except Exception as e:
            logger.error(f"❌ Training strategy execution failed: {e}")
            raise

    def _run_training_with_validation(self, train_data_path: str, val_data_path: str) -> Dict[str, Any]:
        """Run training with validation loop"""

        logger.info("🏋️ Phase 2: Training with Validation")

        # Load training configuration
        training_config = self._load_training_config()

        # Apply overrides
        for key, value in self.config.training_overrides.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
                logger.info(f"   Override: {key} = {value}")

        # Setup trainer
        training_output_dir = self.output_dir / "training"
        training_output_dir.mkdir(exist_ok=True)

        trainer = ContrastiveLearningTrainer(
            training_config, output_dir=str(training_output_dir))

        # Load validation data for evaluation
        val_loader = self._create_data_loader(
            val_data_path, training_config.batch_size)

        # Setup validation evaluator
        eval_config = EvaluationConfig(
            metrics=self.config.validation.metrics,
            save_embeddings=False,  # Don't save embeddings during validation
            save_predictions=self.config.validation.save_predictions,
            generate_visualizations=self.config.validation.generate_plots,
            batch_size=self.config.validation.batch_size or training_config.batch_size,
            device=self.config.device
        )
        evaluator = ContrastiveEvaluator(
            eval_config, text_encoder=trainer.text_encoder)

        # Enhanced training loop with validation
        training_results = self._enhanced_training_loop(
            trainer, train_data_path, val_loader, evaluator, training_output_dir
        )

        return training_results

    def _enhanced_training_loop(self, trainer, train_data_path: str, val_loader: DataLoader,
                                evaluator: ContrastiveEvaluator, output_dir: Path) -> Dict[str, Any]:
        """Enhanced training loop with validation and early stopping"""

        # Initialize tracking variables
        best_metric = float(
            '-inf') if self.config.early_stopping.mode == 'max' else float('inf')
        patience_counter = 0
        training_history = []
        validation_history = []
        checkpoint_paths = []

        logger.info(f"🎯 Starting training with early stopping:")
        logger.info(f"   Metric: {self.config.early_stopping.metric}")
        logger.info(f"   Patience: {self.config.early_stopping.patience}")
        logger.info(f"   Mode: {self.config.early_stopping.mode}")

        # Start training
        epoch = 0
        training_complete = False

        try:
            while epoch < trainer.config.num_epochs and not training_complete:
                epoch_start_time = time.time()

                logger.info(f"📈 Epoch {epoch + 1}/{trainer.config.num_epochs}")

                # Training phase
                train_metrics = trainer.train_epoch(train_data_path, epoch)

                # Validation phase
                val_metrics = None
                if self._should_validate(epoch):
                    logger.info("🔍 Running validation...")
                    val_output_dir = output_dir / \
                        f"validation_epoch_{epoch + 1}"
                    val_results = evaluator.evaluate_model(
                        trainer.model, val_loader, str(val_output_dir))
                    val_metrics = val_results.metrics

                    validation_history.append({
                        'epoch': epoch + 1,
                        'metrics': val_metrics,
                        'timestamp': datetime.now().isoformat()
                    })

                # Record training history
                epoch_duration = time.time() - epoch_start_time
                epoch_record = {
                    'epoch': epoch + 1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'duration': epoch_duration,
                    'timestamp': datetime.now().isoformat()
                }
                training_history.append(epoch_record)

                # Checkpointing
                if self._should_save_checkpoint(epoch):
                    checkpoint_path = self._save_checkpoint(
                        trainer, epoch, train_metrics, val_metrics, output_dir)
                    checkpoint_paths.append(checkpoint_path)

                # Early stopping logic
                if val_metrics and self.config.early_stopping.enabled:
                    current_metric = val_metrics.get(
                        self.config.early_stopping.metric)

                    if current_metric is not None:
                        improved = self._check_improvement(
                            current_metric, best_metric)

                        if improved:
                            best_metric = current_metric
                            patience_counter = 0

                            # Save best model
                            best_model_path = self._save_best_checkpoint(
                                trainer, epoch, val_metrics, output_dir)
                            logger.info(
                                f"🏆 New best model saved: {self.config.early_stopping.metric} = {current_metric:.4f}")
                        else:
                            patience_counter += 1
                            logger.info(
                                f"⏳ Patience: {patience_counter}/{self.config.early_stopping.patience}")

                            if patience_counter >= self.config.early_stopping.patience:
                                logger.info(
                                    f"🛑 Early stopping triggered after {epoch + 1} epochs")
                                training_complete = True

                epoch += 1

            # Save final model
            final_model_path = self._save_final_checkpoint(
                trainer, epoch - 1, output_dir)

        except KeyboardInterrupt:
            logger.info("⏹️  Training interrupted by user")
            final_model_path = self._save_checkpoint(
                trainer, epoch, {}, {}, output_dir)

        # Find best model path
        best_model_path = None
        for checkpoint_path in checkpoint_paths:
            if 'best' in checkpoint_path:
                best_model_path = checkpoint_path
                break

        return {
            'final_model_path': final_model_path,
            'best_model_path': best_model_path,
            'training_history': training_history,
            'validation_history': validation_history,
            'validation_metrics': validation_history[-1]['metrics'] if validation_history else None,
            'checkpoint_paths': checkpoint_paths
        }

    def _run_final_testing(self, test_data_path: str, model_path: str) -> Dict[str, Any]:
        """Run final testing phase"""

        logger.info("🧪 Phase 3: Final Testing")

        # Setup test output directory
        test_output_dir = self.output_dir / "testing"
        test_output_dir.mkdir(exist_ok=True)

        # Load model
        model = self._load_model(model_path)

        # Create test data loader
        test_loader = self._create_data_loader(test_data_path,
                                               self.config.testing.batch_size or 32)

        # Setup evaluator for testing
        from sentence_transformers import SentenceTransformer
        training_config = self._load_training_config()
        test_text_encoder = SentenceTransformer(
            training_config.text_encoder_model)
        test_text_encoder.to(self.config.device)

        eval_config = EvaluationConfig(
            metrics=["all"],  # Use all metrics for final testing
            save_embeddings=self.config.testing.save_embeddings,
            save_predictions=self.config.testing.save_predictions,
            generate_visualizations=self.config.testing.generate_visualizations,
            batch_size=self.config.testing.batch_size or 32,
            device=self.config.device
        )

        evaluator = ContrastiveEvaluator(
            eval_config, text_encoder=test_text_encoder)

        # Run evaluation
        logger.info("🔬 Running comprehensive test evaluation...")
        test_results = evaluator.evaluate_model(
            model, test_loader, str(test_output_dir))

        # Log key metrics
        logger.info("📊 Test Results:")
        for metric_name, value in test_results.metrics.items():
            if isinstance(value, float):
                logger.info(f"   {metric_name}: {value:.4f}")

        return {
            'metrics': test_results.metrics,
            'detailed_metrics': test_results.detailed_metrics,
            'predictions_path': str(test_output_dir / "predictions.json") if test_results.predictions else None,
            'embeddings_path': str(test_output_dir / "embeddings.npy") if test_results.embeddings is not None else None,
            'visualization_paths': test_results.visualization_paths or []
        }

    def _generate_comprehensive_report(self, results: PipelineResults) -> List[str]:
        """Generate comprehensive pipeline report with two-phase training support"""

        logger.info("📋 Phase 4: Generating Enhanced Reports")
        logger.info(f"Training mode: {results.training_mode}")

        report_dir = self.output_dir / "reports"
        report_dir.mkdir(exist_ok=True)

        report_paths = []

        # Generate JSON report
        if "json" in self.config.reporting.formats:
            json_path = report_dir / "pipeline_report.json"
            self._generate_enhanced_json_report(results, json_path)
            report_paths.append(str(json_path))

        # Generate HTML report
        if "html" in self.config.reporting.formats:
            html_path = report_dir / "pipeline_report.html"
            self._generate_enhanced_html_report(results, html_path)
            report_paths.append(str(html_path))

        # Generate two-phase specific reports if applicable
        if results.training_mode == "two_phase":
            # Generate comparative analysis report
            if results.comparative_analysis:
                comparative_path = report_dir / "two_phase_comparative_analysis.json"
                self._generate_comparative_analysis_report(results, comparative_path)
                report_paths.append(str(comparative_path))
            
            # Generate phase breakdown report
            phase_breakdown_path = report_dir / "phase_breakdown_report.json"
            self._generate_phase_breakdown_report(results, phase_breakdown_path)
            report_paths.append(str(phase_breakdown_path))

        logger.info(f"📄 Enhanced reports generated: {len(report_paths)} files")
        if results.training_mode == "two_phase":
            logger.info("📊 Two-phase specific reports included")

        return report_paths

    def _load_training_config(self) -> TrainingConfig:
        """Load training configuration with augmentation integration"""
        config_path = Path(self.config.training_config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Training config not found: {config_path}")

        # Load base training configuration
        training_config = TrainingConfig.from_json(str(config_path))
        
        # Apply augmentation configuration if available
        if AUGMENTATION_INTEGRATION_AVAILABLE:
            try:
                # Convert to dict for processing
                config_dict = training_config.to_dict()
                
                # Add pipeline config information
                pipeline_config_dict = self.config.to_dict()
                
                # Apply augmentation configuration overrides
                updated_config_dict = apply_augmentation_config_to_training(
                    config_dict, pipeline_config_dict
                )
                
                # Log augmentation configuration
                log_augmentation_configuration(config_dict, pipeline_config_dict)
                
                # Create updated training config
                training_config = TrainingConfig.from_dict(updated_config_dict)
                
                logger.info("✅ Augmentation configuration applied to training config")
                
            except Exception as e:
                logger.warning(f"Failed to apply augmentation configuration: {e}")
                logger.info("Continuing with base training configuration")
        
        return training_config

    def _create_data_loader(self, data_path: str, batch_size: int):
        """Create data loader for given dataset"""
        from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader
        import json

        logger.info(f"Creating data loader for: {data_path}")

        class JSONLDataset(Dataset):
            def __init__(self, file_path):
                self.samples = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                sample = json.loads(line)
                                # Convert to expected format
                                processed_sample = {
                                    'resume': sample.get('resume_text', ''),
                                    'job': sample.get('job_text', ''),
                                    'label': sample.get('label', 1)  # Use actual label if available, default to 1
                                }
                                self.samples.append(processed_sample)
                            except json.JSONDecodeError:
                                continue

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        try:
            dataset = JSONLDataset(data_path)
            if len(dataset) == 0:
                logger.warning(f"No valid samples found in {data_path}")
                return None

            data_loader = PyTorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # Don't shuffle validation data
                num_workers=0   # Use single thread for compatibility
            )

            logger.info(f"Created data loader with {len(dataset)} samples")
            return data_loader

        except Exception as e:
            logger.error(f"Failed to create data loader: {e}")
            return None

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from: {model_path}")

        try:
            # Check if model path exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None

            checkpoint = torch.load(
                model_path, map_location=self.config.device)

            # Extract model from checkpoint
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    # Create new model instance and load state dict
                    from .trainer import ContrastiveLearningTrainer
                    # Get model architecture from trainer
                    training_config = self._load_training_config()
                    temp_trainer = ContrastiveLearningTrainer(training_config)
                    model = temp_trainer.model
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    logger.error("No model found in checkpoint")
                    return None
            else:
                # Assume checkpoint is the model itself
                model = checkpoint

            # Move model to correct device
            model = model.to(self.config.device)
            logger.info(f"✅ Model loaded successfully from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def _should_validate(self, epoch: int) -> bool:
        """Check if validation should be run for this epoch"""
        if self.config.validation.frequency == "every_epoch":
            return True
        elif self.config.validation.frequency == "every_n_epochs":
            return (epoch + 1) % self.config.validation.frequency_value == 0
        return False

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved for this epoch"""
        return (epoch + 1) % self.config.checkpointing.save_frequency == 0

    def _save_checkpoint(self, trainer, epoch: int, train_metrics: Dict, val_metrics: Dict,
                         output_dir: Path) -> str:
        """Save model checkpoint"""
        # Save checkpoint using trainer's method
        combined_metrics = {**train_metrics, **val_metrics}
        checkpoint_path = trainer.save_checkpoint(epoch, combined_metrics)

        logger.info(f"💾 Checkpoint saved at epoch {epoch + 1}")
        return checkpoint_path

    def _save_best_checkpoint(self, trainer, epoch: int, val_metrics: Dict, output_dir: Path) -> str:
        """Save best model checkpoint"""
        checkpoint_path = trainer.save_checkpoint(epoch, val_metrics)
        return checkpoint_path

    def _save_final_checkpoint(self, trainer, epoch: int, output_dir: Path) -> str:
        """Save final model checkpoint"""
        final_metrics = {"epoch": epoch, "final": True}
        checkpoint_path = trainer.save_checkpoint(epoch, final_metrics)
        return checkpoint_path

    def _check_improvement(self, current_metric: float, best_metric: float) -> bool:
        """Check if current metric is an improvement over best metric"""
        if self.config.early_stopping.mode == 'max':
            return current_metric > best_metric + self.config.early_stopping.min_delta
        else:
            return current_metric < best_metric - self.config.early_stopping.min_delta

    def _generate_enhanced_json_report(self, results: PipelineResults, output_path: Path):
        """Generate enhanced JSON report with two-phase training support"""
        report_data = {
            "pipeline_summary": {
                "experiment_name": results.experiment_name,
                "duration": results.total_duration,
                "training_mode": results.training_mode,
                "status": "completed"
            },
            "data_splits": results.split_statistics,
            "training_summary": {
                "final_model": results.final_model_path,
                "best_model": results.best_model_path,
                "epochs_completed": len(results.training_history) if results.training_history else 0,
                "training_mode": results.training_mode
            },
            "test_metrics": results.test_metrics,
            "configuration": results.config_used
        }

        # Add two-phase specific information
        if results.training_mode == "two_phase":
            report_data["two_phase_training"] = {
                "phase1_summary": {
                    "final_loss": results.phase1_results.final_loss if results.phase1_results else None,
                    "total_samples": results.phase1_results.total_samples if results.phase1_results else None,
                    "training_time": results.phase1_results.training_time if results.phase1_results else None
                },
                "phase2_summary": {
                    "final_accuracy": results.phase2_results.get('final_accuracy') if results.phase2_results else None,
                    "best_accuracy": results.phase2_results.get('best_accuracy') if results.phase2_results else None,
                    "training_time": results.phase2_results.get('training_time') if results.phase2_results else None
                },
                "phase_transition": results.phase_transition_metrics,
                "comparative_analysis": results.comparative_analysis
            }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"📄 Enhanced JSON report saved: {output_path}")

    def _generate_json_report(self, results: PipelineResults, output_path: Path):
        """Legacy JSON report method for backward compatibility"""
        return self._generate_enhanced_json_report(results, output_path)

    def _generate_enhanced_html_report(self, results: PipelineResults, output_path: Path):
        """Generate enhanced HTML report with two-phase training support"""
        
        # Generate two-phase specific content
        two_phase_content = ""
        if results.training_mode == "two_phase":
            phase1_loss = results.phase1_results.final_loss if results.phase1_results else "N/A"
            phase1_samples = results.phase1_results.total_samples if results.phase1_results else "N/A"
            phase2_accuracy = results.phase2_results.get('final_accuracy', 'N/A') if results.phase2_results else "N/A"
            phase2_best_accuracy = results.phase2_results.get('best_accuracy', 'N/A') if results.phase2_results else "N/A"
            
            two_phase_content = f"""
            <div class="section two-phase">
                <h3>🔄 Two-Phase Training Results</h3>
                <div class="phase-section">
                    <h4>📈 Phase 1: Self-Supervised Pre-training</h4>
                    <div class="metric">Final Loss: <span class="value">{phase1_loss}</span></div>
                    <div class="metric">Total Samples: <span class="value">{phase1_samples}</span></div>
                </div>
                <div class="phase-section">
                    <h4>🎯 Phase 2: Supervised Fine-tuning</h4>
                    <div class="metric">Final Accuracy: <span class="value success">{phase2_accuracy}</span></div>
                    <div class="metric">Best Accuracy: <span class="value success">{phase2_best_accuracy}</span></div>
                </div>
            </div>
            """

        # Enhanced HTML report template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pipeline Report - {results.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
                .header {{ background-color: #e9ecef; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .section {{ margin: 20px 0; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .two-phase {{ border-left: 4px solid #28a745; }}
                .single-phase {{ border-left: 4px solid #007cba; }}
                .phase-section {{ margin: 15px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
                .metric {{ margin: 8px 0; font-size: 14px; }}
                .value {{ font-weight: bold; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                .training-mode {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
                .mode-two-phase {{ background-color: #d4edda; color: #155724; }}
                .mode-single-phase {{ background-color: #d1ecf1; color: #0c5460; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 ML Pipeline Report</h1>
                <h2>{results.experiment_name}</h2>
                <p><strong>Training Mode:</strong> 
                   <span class="training-mode mode-{results.training_mode.replace('_', '-')}">{results.training_mode.replace('_', ' ').title()}</span>
                </p>
                <p><strong>Duration:</strong> {results.total_duration:.2f} seconds</p>
                <p><strong>Completed:</strong> {results.end_time}</p>
            </div>
            
            <div class="section">
                <h3>📊 Data Splits</h3>
                <p><strong>Strategy:</strong> {self.config.data_splitting.strategy}</p>
                <div class="metric">Training Mode: <span class="value">{results.training_mode}</span></div>
            </div>
            
            <div class="section {results.training_mode.replace('_', '-')}">
                <h3>🏋️ Training Results</h3>
                <div class="metric">Final Model: <span class="value">{results.final_model_path or 'N/A'}</span></div>
                <div class="metric">Best Model: <span class="value">{results.best_model_path or 'N/A'}</span></div>
                <div class="metric">Training Epochs: <span class="value">{len(results.training_history) if results.training_history else 0}</span></div>
            </div>
            
            {two_phase_content}
            
            <div class="section">
                <h3>🧪 Test Results</h3>
                <div class="metric">Test Metrics Available: <span class="value {'success' if results.test_metrics else 'warning'}">{bool(results.test_metrics)}</span></div>
                {self._generate_test_metrics_html(results.test_metrics) if results.test_metrics else '<p>No test metrics available</p>'}
            </div>
            
            <div class="section">
                <h3>📁 Generated Artifacts</h3>
                <div class="metric">Visualization Files: <span class="value">{len(results.visualization_paths)}</span></div>
                <div class="metric">Report Files: <span class="value">{len(results.report_paths)}</span></div>
                <div class="metric">Checkpoint Files: <span class="value">{len(results.checkpoint_paths)}</span></div>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"📄 Enhanced HTML report saved: {output_path}")

    def _generate_html_report(self, results: PipelineResults, output_path: Path):
        """Legacy HTML report method for backward compatibility"""
        return self._generate_enhanced_html_report(results, output_path)

    def _setup_logging(self):
        """Setup logging for the pipeline"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / \
            f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        logger.info(f"📝 Logging to: {log_file}")

    def _generate_test_metrics_html(self, test_metrics: Dict[str, Any]) -> str:
        """Generate HTML content for test metrics"""
        if not test_metrics:
            return "<p>No test metrics available</p>"
        
        html_content = ""
        for metric_name, value in test_metrics.items():
            if isinstance(value, float):
                html_content += f'<div class="metric">{metric_name}: <span class="value">{value:.4f}</span></div>'
            else:
                html_content += f'<div class="metric">{metric_name}: <span class="value">{value}</span></div>'
        
        return html_content

    def _generate_comparative_analysis_report(self, results: PipelineResults, output_path: Path):
        """Generate comparative analysis report for two-phase training"""
        if not results.comparative_analysis:
            logger.warning("No comparative analysis data available")
            return

        comparative_data = {
            "experiment_name": results.experiment_name,
            "training_mode": results.training_mode,
            "comparative_analysis": results.comparative_analysis,
            "phase1_summary": {
                "final_loss": results.phase1_results.final_loss if results.phase1_results else None,
                "total_samples": results.phase1_results.total_samples if results.phase1_results else None,
                "training_time": results.phase1_results.training_time if results.phase1_results else None
            },
            "phase2_summary": {
                "final_accuracy": results.phase2_results.get('final_accuracy') if results.phase2_results else None,
                "best_accuracy": results.phase2_results.get('best_accuracy') if results.phase2_results else None,
                "training_time": results.phase2_results.get('training_time') if results.phase2_results else None
            },
            "phase_transition_metrics": results.phase_transition_metrics
        }

        with open(output_path, 'w') as f:
            json.dump(comparative_data, f, indent=2)

        logger.info(f"📊 Comparative analysis report saved: {output_path}")

    def _generate_phase_breakdown_report(self, results: PipelineResults, output_path: Path):
        """Generate detailed phase breakdown report for two-phase training"""
        phase_breakdown = {
            "experiment_name": results.experiment_name,
            "training_mode": results.training_mode,
            "total_duration": results.total_duration,
            "phase_breakdown": {
                "phase1": {
                    "type": "self_supervised_pretraining",
                    "results": results.phase1_results.to_dict() if results.phase1_results and hasattr(results.phase1_results, 'to_dict') else results.phase1_results,
                    "output_model": results.phase1_results.checkpoint_paths[-1] if results.phase1_results and results.phase1_results.checkpoint_paths else None
                },
                "phase2": {
                    "type": "supervised_finetuning", 
                    "results": results.phase2_results,
                    "input_model": results.phase1_results.checkpoint_paths[-1] if results.phase1_results and results.phase1_results.checkpoint_paths else None,
                    "output_model": results.final_model_path
                },
                "transition": results.phase_transition_metrics
            },
            "model_paths": {
                "pretrained_model": results.phase1_results.checkpoint_paths[-1] if results.phase1_results and results.phase1_results.checkpoint_paths else None,
                "final_model": results.final_model_path,
                "best_model": results.best_model_path
            }
        }

        with open(output_path, 'w') as f:
            json.dump(phase_breakdown, f, indent=2)

        logger.info(f"📋 Phase breakdown report saved: {output_path}")
