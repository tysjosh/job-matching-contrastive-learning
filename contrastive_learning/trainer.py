"""
ContrastiveLearningTrainer - Main orchestrator for contrastive learning training.

This module implements the main training orchestrator that coordinates all components
of the contrastive learning system including data loading, batch processing, loss
computation, and model updates with checkpoint management and progress logging.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
import signal

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .data_structures import TrainingConfig, TrainingResults, TrainingSample, ContrastiveTriplet
from .data_loader import DataLoader
from .batch_processor import BatchProcessor
from .loss_engine import ContrastiveLossEngine
from .embedding_cache import EmbeddingCache, BatchEfficientEncoder
from .logging_utils import (
    BatchMetrics, EpochMetrics, MemoryMonitor,
    timed_operation, setup_training_logger
)
logger = logging.getLogger(__name__)
# Import diagnostic integration
try:
    from diagnostic.training_integration import create_training_integration, DiagnosticTriggerConfig
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    logger.warning("Diagnostic integration not available")


class TrainingInterruptedException(Exception):
    """Exception raised when training is interrupted by user signal."""
    pass


class ContrastiveLearningTrainer:
    """
    Main orchestrator for contrastive learning training.

    Coordinates all components including data loading, batch processing, loss computation,
    and model updates. Provides checkpoint management, progress logging, and training
    interruption/resumption support.
    """

    def __init__(self, config: TrainingConfig, model: Optional[nn.Module] = None,
                 output_dir: str = "training_output", esco_graph_path: Optional[str] = None):
        """
        Initialize the ContrastiveLearningTrainer.

        Args:
            config: Training configuration
            model: Optional PyTorch model (if None, creates a dummy model for testing)
            output_dir: Directory for saving checkpoints and logs
            esco_graph_path: Optional path to ESCO graph file (overrides config.esco_graph_path)

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ContrastiveLearningTrainer. "
                "Install with: pip install torch"
            )

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize structured logging
        self.structured_logger = setup_training_logger(
            name="contrastive_training",
            log_dir=self.output_dir / "logs",
            log_level=logging.INFO
        )
        self.memory_monitor = MemoryMonitor()

        # Initialize components
        self.data_loader = DataLoader(config)

        # Determine ESCO graph path (parameter overrides config)
        final_esco_path = esco_graph_path or config.esco_graph_path

        # Initialize BatchProcessor with proper ESCO graph path
        self.batch_processor = BatchProcessor(
            config, esco_graph_path=final_esco_path)
        self.loss_engine = ContrastiveLossEngine(config)

        # Initialize semantic text encoder from config
        self.text_encoder = SentenceTransformer(config.text_encoder_model)

        # Set device for BOTH text encoder and main model consistently
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Move text encoder to device with comprehensive device setting
        self.text_encoder.to(self.device)

        # Ensure ALL components of SentenceTransformer are on the correct device
        for module in self.text_encoder.modules():
            if hasattr(module, 'to'):
                module.to(self.device)

        # Set the device for the underlying transformers model
        if hasattr(self.text_encoder, '_modules'):
            for name, module in self.text_encoder._modules.items():
                if module is not None and hasattr(module, 'to'):
                    module.to(self.device)

        # Model and optimizer setup
        # Get the text encoder's embedding dimension dynamically
        text_encoder_dim = self.text_encoder.get_sentence_embedding_dimension()
        self.model = model or self._create_dual_encoder_model(
            input_dim=text_encoder_dim)

        # RESEARCH-GRADE SETUP: Freeze SentenceTransformer to avoid catastrophic forgetting
        if getattr(config, 'freeze_text_encoder', True):
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
            # Only train the minimal projection head (prevents overfitting)
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
            
            logger.info("✅ SentenceTransformer FROZEN - avoiding catastrophic forgetting")
            logger.info(f"📊 Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            logger.info(f"📊 Frozen parameters: {sum(p.numel() for p in self.text_encoder.parameters()):,}")
        else:
            # Legacy mode: train both models (not recommended for research)
            import itertools
            all_parameters = itertools.chain(
                self.model.parameters(),
                self.text_encoder.parameters()
            )
            self.optimizer = optim.Adam(all_parameters, lr=config.learning_rate)
            logger.warning("⚠️ Training both models - risk of catastrophic forgetting!")
        
        self.model.to(self.device)

        # Initialize efficient embedding cache
        cache_size = getattr(config, 'embedding_cache_size', 10000)  # Default 10k embeddings
        self.embedding_cache = EmbeddingCache(
            max_cache_size=cache_size,
            device=self.device,
            enable_stats=True
        )
        
        # Initialize batch-efficient encoder
        self.batch_encoder = BatchEfficientEncoder(
            text_encoder=self.text_encoder,
            model=self.model,
            device=self.device
        )

        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches_processed = 0
        self.training_interrupted = False
        self.best_loss = float('inf')
        
        # Global negative sampling state
        self.global_job_pool = None

        # Metrics tracking
        self.epoch_losses = []
        self.batch_losses = []
        self.training_metrics = {
            'total_samples_processed': 0,
            'total_triplets_created': 0,
            'checkpoint_saves': 0,
            'training_start_time': None,
            'training_end_time': None
        }

        # Setup signal handlers for graceful interruption
        self._setup_signal_handlers()

        # Initialize diagnostic integration if available
        self.diagnostic_integration = None
        if DIAGNOSTICS_AVAILABLE:
            try:
                trigger_config = DiagnosticTriggerConfig(
                    loss_stagnation_steps=100,
                    gradient_explosion_threshold=10.0,
                    periodic_check_frequency=500,
                    quick_check_frequency=50,
                    enable_async_diagnostics=True,
                    max_diagnostic_overhead_ms=50.0
                )

                self.diagnostic_integration = create_training_integration(
                    trigger_config=trigger_config,
                    enable_async=True
                )

                # Add alert callback for handling training issues
                self.diagnostic_integration.add_alert_callback(
                    self._handle_diagnostic_alert)

                self.structured_logger.logger.info(
                    "Diagnostic integration enabled")
            except Exception as e:
                self.structured_logger.logger.warning(
                    f"Failed to initialize diagnostics: {e}")
                self.diagnostic_integration = None

        # Log initialization
        self.structured_logger.logger.info(
            f"ContrastiveLearningTrainer initialized with config: {config}")
        self.structured_logger.logger.info(
            f"Output directory: {self.output_dir}")
        self.structured_logger.logger.info(f"Device: {self.device}")
        self.structured_logger.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Log initial memory usage
        memory_info = self.memory_monitor.get_memory_usage()
        self.structured_logger.log_memory_usage("initialization",
                                                memory_info["system_memory_mb"] or 0,
                                                memory_info["gpu_memory_mb"])

    def train(self, dataset_path: Union[str, Path]) -> TrainingResults:
        """
        Run the complete training process.

        Args:
            dataset_path: Path to training dataset (JSONL format)

        Returns:
            TrainingResults containing training metrics and checkpoint paths

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            TrainingInterruptedException: If training is interrupted
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Start training session with comprehensive logging
        session_id = self.structured_logger.start_training_session({
            "config": asdict(self.config),
            "dataset_path": str(dataset_path),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device)
        })

        self.training_metrics['training_start_time'] = time.time()

        try:
            # Save initial configuration
            self._save_config()
            
            # Load global job pool if enabled
            if self.config.global_negative_sampling:
                logger.info("Loading global job pool for consistent negative sampling...")
                self.global_job_pool = self.data_loader.load_global_job_pool(
                    dataset_path, max_jobs=self.config.global_negative_pool_size
                )
                logger.info(f"Global job pool loaded with {len(self.global_job_pool)} jobs")
            
            # Preload embeddings if enabled
            if self.config.enable_embedding_preload:
                logger.info("Preloading embeddings for efficient training...")
                self.preload_dataset_embeddings(dataset_path)
                
                # Log cache stats after preloading
                cache_stats = self.embedding_cache.get_cache_stats()
                logger.info(f"Preloading complete - Cache size: {cache_stats['cache_size']}, "
                           f"Memory usage: {cache_stats['memory_usage_mb']:.1f} MB")

            # Run training epochs
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if self.training_interrupted:
                    self.structured_logger.log_training_interruption("User signal", {
                        "epoch": epoch,
                        "total_batches_processed": self.total_batches_processed
                    })
                    raise TrainingInterruptedException(
                        "Training interrupted by user")

                self.current_epoch = epoch

                with timed_operation(self.structured_logger, f"epoch_{epoch}", {"epoch": epoch}):
                    epoch_loss = self._train_epoch(dataset_path)
                    self.epoch_losses.append(epoch_loss)

                # Save checkpoint if needed
                if self._should_save_checkpoint():
                    checkpoint_path = self._save_checkpoint(epoch, epoch_loss)
                    self.structured_logger.log_checkpoint_save(epoch, checkpoint_path, {
                        "loss": epoch_loss,
                        "best_loss": self.best_loss,
                        "total_batches": self.total_batches_processed
                    })

            # Training completed successfully
            self.training_metrics['training_end_time'] = time.time()
            results = self._create_training_results()

            # Save final results and end session
            self._save_training_results(results)
            
            # Log final embedding cache statistics
            cache_stats = self.embedding_cache.get_cache_stats()
            logger.info(f"Final embedding cache stats - "
                       f"Size: {cache_stats['cache_size']}, "
                       f"Hit rate: {cache_stats['hit_rate']:.2%}, "
                       f"Total encodings: {cache_stats['total_encodings']}, "
                       f"Time saved: {cache_stats['time_saved_seconds']:.1f}s, "
                       f"Memory usage: {cache_stats['memory_usage_mb']:.1f}MB")
            
            # Cleanup global job pool to free memory
            self._cleanup_resources()
            
            self.structured_logger.end_training_session({
                "final_loss": results.final_loss,
                "best_loss": self.best_loss,
                "total_samples": results.total_samples,
                "total_batches": results.total_batches,
                "cache_stats": cache_stats
            })

            self.structured_logger.logger.info(
                "Training completed successfully")
            return results

        except TrainingInterruptedException:
            self.structured_logger.logger.info(
                "Training interrupted, saving checkpoint...")
            checkpoint_path = self._save_checkpoint(self.current_epoch,
                                                    self.epoch_losses[-1] if self.epoch_losses else float('inf'))
            self.structured_logger.log_checkpoint_save(self.current_epoch, checkpoint_path, {
                "interruption": True,
                "total_batches": self.total_batches_processed
            })

            # End session with interruption
            self.structured_logger.end_training_session({
                "interrupted": True,
                "final_loss": self.epoch_losses[-1] if self.epoch_losses else None
            })
            raise
        except Exception as e:
            self.structured_logger.logger.error(
                f"Training failed with error: {e}")
            # Save emergency checkpoint
            try:
                checkpoint_path = self._save_checkpoint(
                    self.current_epoch, float('inf'), emergency=True)
                self.structured_logger.log_checkpoint_save(self.current_epoch, checkpoint_path, {
                    "emergency": True,
                    "error": str(e)
                })
            except Exception as checkpoint_error:
                self.structured_logger.logger.error(
                    f"Failed to save emergency checkpoint: {checkpoint_error}")

            # End session with error
            self.structured_logger.end_training_session({
                "error": str(e),
                "final_loss": None
            })
            raise

    def train_epoch(self, dataset_path: Union[str, Path], epoch: int = None) -> Dict[str, Any]:
        """
        Public interface for training a single epoch (for pipeline integration).

        Args:
            dataset_path: Path to training dataset
            epoch: Optional epoch number (will use current_epoch if not provided)

        Returns:
            Dictionary containing training metrics for the epoch
        """
        if epoch is not None:
            self.current_epoch = epoch

        dataset_path = Path(dataset_path)
        avg_loss = self._train_epoch(dataset_path)

        # Return metrics in format expected by pipeline
        return {
            'average_loss': avg_loss,
            'epoch': self.current_epoch,
            'samples_processed': self.training_metrics.get('total_samples_processed', 0),
            'total_batches': self.total_batches_processed
        }

    def _train_epoch(self, dataset_path: Path) -> float:
        """
        Train for one epoch.

        Args:
            dataset_path: Path to training dataset

        Returns:
            Average loss for the epoch
        """
        self.structured_logger.log_epoch_start(
            self.current_epoch, self.config.num_epochs)

        # Update loss engine with current epoch for curriculum learning
        self.loss_engine.set_epoch(self.current_epoch)

        epoch_start_time = time.time()
        epoch_losses = []
        epoch_samples = 0
        epoch_triplets = 0
        failed_batches = 0
        memory_peak = 0
        gpu_memory_peak = 0

        self.model.train()

        try:
            # Process batches
            for batch_idx, batch in enumerate(self.data_loader.load_batches(dataset_path)):
                if self.training_interrupted:
                    break

                self.current_batch = batch_idx

                # Log batch start
                self.structured_logger.log_batch_start(
                    batch_idx, self.current_epoch, len(batch))

                # Monitor memory at batch start
                memory_info = self.memory_monitor.get_memory_usage()
                if memory_info["system_memory_mb"]:
                    memory_peak = max(
                        memory_peak, memory_info["system_memory_mb"])
                if memory_info["gpu_memory_mb"]:
                    gpu_memory_peak = max(
                        gpu_memory_peak, memory_info["gpu_memory_mb"])

                try:
                    batch_loss, batch_stats = self._train_batch(batch)

                    if batch_loss is not None:
                        epoch_losses.append(batch_loss)
                        epoch_samples += batch_stats['samples_processed']
                        epoch_triplets += batch_stats['triplets_created']

                        # Create batch metrics
                        batch_metrics = BatchMetrics(
                            batch_id=batch_idx,
                            epoch=self.current_epoch,
                            samples_processed=batch_stats['samples_processed'],
                            triplets_created=batch_stats['triplets_created'],
                            view_combinations=batch_stats.get(
                                'view_combinations', 0),
                            processing_time=batch_stats['processing_time'],
                            loss=batch_loss,
                            memory_usage_mb=memory_info["system_memory_mb"],
                            gpu_memory_mb=memory_info["gpu_memory_mb"]
                        )

                        # Log batch completion
                        self.structured_logger.log_batch_completion(
                            batch_metrics)

                        # Log batch progress at specified frequency
                        if (batch_idx + 1) % self.config.log_frequency == 0:
                            self._log_batch_progress(
                                batch_idx + 1, batch_loss, batch_stats)
                    else:
                        failed_batches += 1

                except Exception as batch_error:
                    failed_batches += 1
                    self.structured_logger.log_batch_error(
                        batch_idx, self.current_epoch, batch_error,
                        {"batch_size": len(
                            batch), "total_batches_processed": self.total_batches_processed}
                    )

                self.total_batches_processed += 1

        except Exception as e:
            self.structured_logger.logger.error(
                f"Error during epoch {self.current_epoch}: {e}")
            raise

        # Calculate epoch metrics
        avg_epoch_loss = sum(epoch_losses) / \
            len(epoch_losses) if epoch_losses else float('inf')
        min_epoch_loss = min(epoch_losses) if epoch_losses else float('inf')
        max_epoch_loss = max(epoch_losses) if epoch_losses else float('inf')
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = epoch_time / \
            max(1, len(epoch_losses) + failed_batches)

        # Update training metrics
        self.training_metrics['total_samples_processed'] += epoch_samples
        self.training_metrics['total_triplets_created'] += epoch_triplets

        # Create epoch metrics
        epoch_metrics = EpochMetrics(
            epoch=self.current_epoch,
            total_batches=len(epoch_losses) + failed_batches,
            total_samples=epoch_samples,
            total_triplets=epoch_triplets,
            avg_loss=avg_epoch_loss,
            min_loss=min_epoch_loss,
            max_loss=max_epoch_loss,
            epoch_time=epoch_time,
            avg_batch_time=avg_batch_time,
            memory_peak_mb=memory_peak if memory_peak > 0 else None,
            gpu_memory_peak_mb=gpu_memory_peak if gpu_memory_peak > 0 else None,
            failed_batches=failed_batches
        )

        # Log epoch completion
        self.structured_logger.log_epoch_completion(epoch_metrics)

        # Clear embedding cache between epochs to prevent memory leak (configurable)
        clear_cache = getattr(self.config, 'clear_cache_between_epochs', True)
        if clear_cache and hasattr(self, 'embedding_cache'):
            cache_stats_before = self.embedding_cache.get_cache_stats()
            memory_freed = cache_stats_before.get('memory_usage_mb', 0)
            self.embedding_cache.clear_cache()
            logger.info(f"Cleared embedding cache after epoch {self.current_epoch}. "
                       f"Freed {memory_freed:.1f}MB, had {cache_stats_before.get('cache_size', 0)} embeddings")

        # Diagnostic integration - epoch end
        if self.diagnostic_integration:
            try:
                validation_metrics = {
                    'avg_loss': avg_epoch_loss,
                    'min_loss': min_epoch_loss,
                    'max_loss': max_epoch_loss,
                    'samples_processed': epoch_samples,
                    'triplets_created': epoch_triplets,
                    'failed_batches': failed_batches,
                    'epoch_time': epoch_time
                }

                diagnostic_report = self.diagnostic_integration.on_epoch_end(
                    epoch=self.current_epoch,
                    model=self.model,
                    validation_metrics=validation_metrics
                )

                if diagnostic_report:
                    self.structured_logger.logger.info(
                        f"Epoch {self.current_epoch} diagnostic report generated")

            except Exception as diagnostic_error:
                self.structured_logger.logger.warning(
                    f"Epoch diagnostic error: {diagnostic_error}")

        return avg_epoch_loss

    def _train_batch(self, batch: List[TrainingSample]) -> tuple[Optional[float], Dict[str, Any]]:
        """
        Train on a single batch.

        Args:
            batch: List of training samples

        Returns:
            Tuple of (batch_loss, batch_statistics)
        """
        batch_stats = {
            'samples_processed': len(batch),
            'triplets_created': 0,
            'view_combinations': 0,
            'processing_time': 0
        }

        batch_start_time = time.time()

        try:
            # Process batch to create triplets (with optional global job pool)
            triplets = self.batch_processor.process_batch(batch, self.global_job_pool)
            batch_stats['triplets_created'] = len(triplets)

            if not triplets:
                logger.warning("No valid triplets created from batch")
                return None, batch_stats

            # Generate embeddings for triplets
            embeddings = self._generate_embeddings(triplets)

            if not embeddings:
                logger.warning("No embeddings generated for batch")
                return None, batch_stats

            # Compute loss
            loss = self.loss_engine.compute_loss(triplets, embeddings)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
                logger.warning(f"Triplets count: {len(triplets)}, Embeddings count: {len(embeddings)}")
                # Log some embedding statistics for debugging
                if embeddings:
                    sample_emb = next(iter(embeddings.values()))
                    logger.warning(f"Sample embedding stats - mean: {sample_emb.mean().item():.4f}, "
                                 f"std: {sample_emb.std().item():.4f}, "
                                 f"min: {sample_emb.min().item():.4f}, "
                                 f"max: {sample_emb.max().item():.4f}")
                return None, batch_stats

            # Store loss value before backward pass
            loss_value = loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            try:
                loss.backward()
            except RuntimeError as e:
                if "Trying to backward through the graph a second time" in str(e):
                    logger.warning("Computational graph reuse detected, skipping batch")
                    return None, batch_stats
                else:
                    raise e

            # Gradient clipping for stability (configurable)
            grad_clip_norm = getattr(self.config, 'gradient_clip_norm', 1.0)
            
            # Log gradient norm before clipping for debugging
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=grad_clip_norm)
            
            if total_norm > grad_clip_norm * 2:  # Log if gradients are unusually large
                logger.warning(f"Large gradient norm detected: {total_norm:.4f} (clipped to {grad_clip_norm})")

            self.optimizer.step()

            # Diagnostic integration - monitor training step
            if self.diagnostic_integration:
                try:
                    # Prepare batch data for diagnostics (use detached embeddings)
                    diagnostic_batch_data = {
                        'triplets': triplets,
                        'embeddings': {k: v.detach() for k, v in embeddings.items()},
                        'samples_processed': batch_stats['samples_processed'],
                        'triplets_created': batch_stats['triplets_created'],
                        'batch_size': len(batch)
                    }

                    # Run diagnostic monitoring with detached loss value
                    diagnostic_report = self.diagnostic_integration.on_training_step(
                        step=self.total_batches_processed,
                        loss=loss_value,
                        model=self.model,
                        batch_data=diagnostic_batch_data,
                        optimizer=self.optimizer
                    )

                    # Handle critical diagnostic issues
                    if diagnostic_report and diagnostic_report.severity_level == "critical":
                        self.structured_logger.logger.critical(
                            f"Critical diagnostic issues at step {self.total_batches_processed}: {diagnostic_report.summary}"
                        )

                except Exception as diagnostic_error:
                    self.structured_logger.logger.warning(
                        f"Diagnostic integration error: {diagnostic_error}")

            batch_stats['processing_time'] = time.time() - batch_start_time

            return loss_value, batch_stats

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            batch_stats['processing_time'] = time.time() - batch_start_time
            return None, batch_stats

    def _generate_embeddings(self, triplets: List[ContrastiveTriplet]) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for all content in triplets using efficient caching.

        This method uses the EmbeddingCache to avoid redundant encoding of the same
        content across batches, providing significant speedup.

        Args:
            triplets: List of contrastive triplets

        Returns:
            Dictionary mapping content keys to final model embeddings
        """
        try:
            # Ensure all models are on the correct device
            self.model.to(self.device)
            self.text_encoder.to(self.device)

            # Collect all unique content items for this batch
            content_items = []
            content_key_mapping = {}

            for triplet in triplets:
                # Add anchor (resume)
                anchor_key = self.embedding_cache.get_content_key(triplet.anchor)
                if anchor_key not in content_key_mapping:
                    content_items.append((triplet.anchor, 'resume'))
                    content_key_mapping[anchor_key] = len(content_items) - 1

                # Add positive (job)
                positive_key = self.embedding_cache.get_content_key(triplet.positive)
                if positive_key not in content_key_mapping:
                    content_items.append((triplet.positive, 'job'))
                    content_key_mapping[positive_key] = len(content_items) - 1

                # Add negatives (jobs)
                for negative in triplet.negatives:
                    negative_key = self.embedding_cache.get_content_key(negative)
                    if negative_key not in content_key_mapping:
                        content_items.append((negative, 'job'))
                        content_key_mapping[negative_key] = len(content_items) - 1

            # Get embeddings using the efficient cache
            embeddings = self.embedding_cache.get_embeddings_batch(
                content_items, 
                self.batch_encoder.encode_batch
            )

            # Log cache statistics periodically
            if self.total_batches_processed % 50 == 0:
                cache_stats = self.embedding_cache.get_cache_stats()
                logger.info(f"Cache stats - Size: {cache_stats['cache_size']}, "
                           f"Hit rate: {cache_stats['hit_rate']:.2%}, "
                           f"Time saved: {cache_stats['time_saved_seconds']:.1f}s, "
                           f"Memory: {cache_stats['memory_usage_mb']:.1f}MB")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings with cache: {e}")
            # Fallback to original method if cache fails
            return self._generate_embeddings_fallback(triplets)

    def _encode_content_to_text_embedding(self, content: Dict[str, Any], content_type: str) -> torch.Tensor:
        """
        Convert structured resume/job data to text embeddings using frozen SentenceTransformer.

        Research-grade approach:
        1. Use frozen SentenceTransformer (no catastrophic forgetting)
        2. Focus on contrastive learning innovations (global sampling, career-aware negatives)
        3. Minimal projection for task adaptation

        Args:
            content: Content dictionary with rich preprocessing data
            content_type: 'resume' or 'job'

        Returns:
            SentenceTransformer embedding tensor (from frozen encoder)
        """
        if content_type == 'resume':
            # Extract rich text from preprocessed resume data
            text_parts = []

            # 1. HIGHEST PRIORITY: Experience text (complete resume content)
            if 'experience' in content:
                experience = content['experience']
                if isinstance(experience, list) and len(experience) > 0:
                    # Extract description from first experience entry (most comprehensive)
                    if isinstance(experience[0], dict):
                        experience_text = experience[0].get('description', '')
                        if experience_text:
                            text_parts.append(f"Profile: {experience_text}")
                    else:
                        text_parts.append(f"Profile: {str(experience[0])}")
                elif isinstance(experience, str):
                    text_parts.append(f"Profile: {experience}")

            # 2. Role and experience level (career positioning)
            role = content.get('role', '')
            exp_level = content.get('experience_level', '')
            if role and exp_level:
                text_parts.append(f"Position: {exp_level} {role}")

            # 3. Structured skills by category (prioritize most common categories)
            if 'skills' in content:
                skills_by_category = {}
                for skill in content['skills']:
                    if isinstance(skill, dict):
                        category = skill.get('category', 'Other')
                        skill_name = skill.get('name', '')
                        level = skill.get('level', '')

                        if category not in skills_by_category:
                            skills_by_category[category] = []
                        skills_by_category[category].append(
                            f"{skill_name} ({level})" if level else skill_name)

                # Prioritize most frequent categories from data analysis
                priority_categories = [
                    'Programming Languages', 'Cloud Platforms  Devops Tools', 'Project Management Methodologies']

                for category in priority_categories:
                    if category in skills_by_category:
                        text_parts.append(
                            f"{category}: {', '.join(skills_by_category[category])}")

                # Add remaining categories
                for category, skill_list in skills_by_category.items():
                    if category not in priority_categories:
                        text_parts.append(
                            f"{category}: {', '.join(skill_list)}")

            # 4. Keywords for semantic enrichment (limit to top 10)
            if 'keywords' in content:
                text_parts.append(
                    f"Keywords: {', '.join(content['keywords'][:10])}")

            full_text = " [SEP] ".join(text_parts)

        elif content_type == 'job':
            # Similar semantic extraction for jobs
            text_parts = []

            # Job title and basic info
            title = content.get('title', content.get('jobtitle', ''))
            if title:
                text_parts.append(f"Position: {title}")

            # Extract skills from job - handle both string and dict formats
            skills = content.get('skills', [])
            if skills:
                if isinstance(skills, str):
                    text_parts.append(f"Required Skills: {skills}")
                elif isinstance(skills, list):
                    # Handle list of skills that could be strings or dictionaries
                    skill_strings = []
                    for skill in skills:
                        if isinstance(skill, dict):
                            # Extract skill name and level from dictionary
                            skill_name = skill.get('name', skill.get('skill'))
                            if skill_name:
                                skill_level = skill.get('level', '')
                                if skill_level:
                                    skill_strings.append(
                                        f"{skill_name} ({skill_level})")
                                else:
                                    skill_strings.append(skill_name)
                        elif isinstance(skill, str):
                            skill_strings.append(skill)

                    if skill_strings:
                        text_parts.append(
                            f"Required Skills: {', '.join(skill_strings)}")

            # Job description
            description = content.get(
                'description', content.get('jobdescription', ''))
            if description:
                # Handle dict format (with 'original' field) or string format
                if isinstance(description, dict):
                    desc_text = description.get('original', '')
                elif isinstance(description, str):
                    desc_text = description
                else:
                    desc_text = str(description)

                # Truncate long descriptions
                if desc_text:
                    desc_text = desc_text[:500] + \
                        "..." if len(desc_text) > 500 else desc_text
                    text_parts.append(f"Description: {desc_text}")

            full_text = " [SEP] ".join(text_parts)

        else:
            # Fallback for unknown content type
            full_text = str(content)

        # Use frozen SentenceTransformer for semantic encoding (research-grade approach)
        try:
            # Always use no_grad for frozen encoder (saves memory and prevents gradients)
            with torch.no_grad():
                embedding = self.text_encoder.encode(
                    full_text,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                    normalize_embeddings=False  # Let the projection head handle normalization
                )

            # Clone the tensor to make it a regular tensor that can be used in autograd
            # This fixes the "Inference tensors cannot be saved for backward" error
            return embedding.clone().detach().requires_grad_(False)

        except Exception as e:
            logger.error(f"Error encoding content with frozen SentenceTransformer: {e}")
            # Fallback to simple encoding
            fallback_text = str(content)[:100]
            try:
                with torch.no_grad():
                    embedding = self.text_encoder.encode(
                        fallback_text,
                        convert_to_tensor=True,
                        device=self.device,
                        show_progress_bar=False,
                        normalize_embeddings=False
                    )
                # Clone the fallback tensor as well
                embedding = embedding.clone().detach().requires_grad_(False)
            except Exception as fallback_error:
                logger.error(f"Fallback encoding also failed: {fallback_error}")
                # Create a zero tensor on the correct device as last resort
                embedding_dim = self.text_encoder.get_sentence_embedding_dimension()
                embedding = torch.zeros(embedding_dim, device=self.device)
            return embedding

    def _encode_content(self, content: Dict[str, Any], content_type: str) -> torch.Tensor:
        """
        Legacy method for backward compatibility.

        This method provides the old single-stage embedding process for cases
        where the full two-stage process is not needed.

        Args:
            content: Content dictionary with rich preprocessing data
            content_type: 'resume' or 'job'

        Returns:
            Final embedding tensor (same as _encode_content_to_text_embedding for now)
        """
        return self._encode_content_to_text_embedding(content, content_type)

    def _get_content_key(self, content: Dict[str, Any]) -> str:
        """Generate a consistent key for content (legacy method)."""
        # Delegate to the embedding cache for consistency
        return self.embedding_cache.get_content_key(content)
    
    def _generate_embeddings_fallback(self, triplets: List[ContrastiveTriplet]) -> Dict[str, torch.Tensor]:
        """
        Fallback embedding generation method (original implementation).
        Used when the cache-based method fails.
        """
        embeddings = {}

        try:
            # Ensure all models are on the correct device
            self.model.to(self.device)
            self.text_encoder.to(self.device)

            # Collect all unique content
            all_content = {}

            for triplet in triplets:
                # Add anchor (resume)
                anchor_key = self._get_content_key(triplet.anchor)
                all_content[anchor_key] = ('resume', triplet.anchor)

                # Add positive (job)
                positive_key = self._get_content_key(triplet.positive)
                all_content[positive_key] = ('job', triplet.positive)

                # Add negatives (jobs)
                for negative in triplet.negatives:
                    negative_key = self._get_content_key(negative)
                    all_content[negative_key] = ('job', negative)

            # Batch process content for efficiency
            content_keys = list(all_content.keys())
            content_texts = []

            for content_key in content_keys:
                content_type, content_data = all_content[content_key]
                text_embedding = self._encode_content_to_text_embedding(
                    content_data, content_type)
                content_texts.append(text_embedding)

            if content_texts:
                # Stack all text embeddings
                text_embeddings_tensor = torch.stack(content_texts)
                text_embeddings_tensor = text_embeddings_tensor.to(self.device)

                # Pass through the contrastive model for final embeddings
                self.model.train()
                final_embeddings = self.model(text_embeddings_tensor)
                final_embeddings = final_embeddings.to(self.device)

                # Map back to content keys
                for i, content_key in enumerate(content_keys):
                    embeddings[content_key] = final_embeddings[i]

            return embeddings

        except Exception as e:
            logger.error(f"Error in fallback embedding generation: {e}")
            return {}

    def _create_dual_encoder_model(self, input_dim: int = 768) -> nn.Module:
        """
        Create a research-grade contrastive model that eliminates double encoding problems.

        This implementation:
        1. Uses minimal projection to avoid overfitting on small datasets
        2. Eliminates complex residual connections that don't help
        3. Focuses on the contrastive learning innovations (global sampling, career-aware negatives)
        4. Provides stable training with proper normalization

        Args:
            input_dim: Input embedding dimension from the frozen SentenceTransformer
        """
        class CareerAwareContrastiveModel(nn.Module):
            def __init__(self, input_dim: int = 384, projection_dim: int = 128, dropout: float = 0.1):
                """
                Minimal contrastive model for career-aware resume-job matching.

                Args:
                    input_dim: Input embedding dimension (from frozen SentenceTransformer)
                    projection_dim: Output embedding dimension (smaller to reduce overfitting)
                    dropout: Dropout rate for regularization
                """
                super().__init__()

                # Minimal projection head - focus on contrastive learning innovations
                # No complex encoder or residual connections needed
                self.projection_head = nn.Sequential(
                    nn.Linear(input_dim, projection_dim * 2),  # Expand slightly
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(projection_dim * 2, projection_dim)  # Project to final size
                )

                # Initialize weights for stable training
                self._initialize_weights()

            def _initialize_weights(self):
                """Initialize model weights using Xavier initialization."""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass through the minimal contrastive model.

                Args:
                    x: Input embeddings from frozen SentenceTransformer

                Returns:
                    Normalized embeddings for contrastive learning
                """
                # Simple projection (no complex encoding needed)
                projected = self.projection_head(x)

                # Return raw projections - let the loss function handle normalization if needed
                return projected

            def get_embedding_dim(self) -> int:
                """Get the output embedding dimension."""
                # Get output dimension from the last Linear layer
                for layer in reversed(self.projection_head):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
                return 128  # Default projection_dim

        # Use config values for research-grade setup
        projection_dim = getattr(self.config, 'projection_dim', 128)
        dropout = getattr(self.config, 'projection_dropout', 0.1)
        
        return CareerAwareContrastiveModel(
            input_dim=input_dim, 
            projection_dim=projection_dim, 
            dropout=dropout
        )
    
    def get_embedding_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self.embedding_cache.get_cache_stats()
    
    def clear_embedding_cache(self):
        """Clear the embedding cache (useful between epochs or for memory management)."""
        self.embedding_cache.clear_cache()
        logger.info("Embedding cache cleared")
    
    def preload_dataset_embeddings(self, dataset_path: Union[str, Path], batch_size: int = 64):
        """
        Preload embeddings for the entire dataset to maximize cache efficiency.
        
        Args:
            dataset_path: Path to training dataset
            batch_size: Batch size for preloading
        """
        logger.info(f"Preloading embeddings for dataset: {dataset_path}")
        
        try:
            # Collect all unique content from the dataset
            all_content_items = set()
            
            for batch in self.data_loader.load_batches(dataset_path):
                for sample in batch:
                    # Add resume
                    resume_key = self.embedding_cache.get_content_key(sample.resume)
                    all_content_items.add((resume_key, sample.resume, 'resume'))
                    
                    # Add job
                    job_key = self.embedding_cache.get_content_key(sample.job)
                    all_content_items.add((job_key, sample.job, 'job'))
            
            # Convert to list and remove keys (only need content and type)
            content_items = [(content, content_type) for _, content, content_type in all_content_items]
            
            # Preload embeddings
            self.embedding_cache.preload_embeddings(
                content_items, 
                self.batch_encoder.encode_batch,
                batch_size=batch_size
            )
            
            # Log final stats
            stats = self.embedding_cache.get_cache_stats()
            logger.info(f"Preloading complete. Cache size: {stats['cache_size']}, "
                       f"Memory usage: {stats['memory_usage_mb']:.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to preload embeddings: {e}")
            logger.info("Continuing with on-demand embedding generation")

    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> str:
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Training metrics dictionary

        Returns:
            Path to saved checkpoint
        """
        return self._save_checkpoint(epoch, metrics.get('loss', float('inf')))

    def _save_checkpoint(self, epoch: int, loss: float, emergency: bool = False) -> str:
        """
        Internal method to save checkpoint.

        Args:
            epoch: Current epoch number
            loss: Current loss value
            emergency: Whether this is an emergency checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}"
        if emergency:
            checkpoint_name += "_emergency"
        checkpoint_name += ".pt"

        checkpoint_path = self.output_dir / checkpoint_name

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': asdict(self.config),
            'training_metrics': self.training_metrics.copy(),
            'epoch_losses': self.epoch_losses.copy(),
            'total_batches_processed': self.total_batches_processed,
            'best_loss': self.best_loss
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.training_metrics['checkpoint_saves'] += 1

            # Update best loss if this is better
            if loss < self.best_loss:
                self.best_loss = loss
                best_checkpoint_path = self.output_dir / "best_checkpoint.pt"
                torch.save(checkpoint_data, best_checkpoint_path)

            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load training checkpoint and resume training state.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing checkpoint metadata

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            checkpoint_data = torch.load(
                checkpoint_path, map_location=self.device)

            # Restore model and optimizer state
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.optimizer.load_state_dict(
                checkpoint_data['optimizer_state_dict'])

            # Restore training state
            self.current_epoch = checkpoint_data['epoch']
            self.total_batches_processed = checkpoint_data.get(
                'total_batches_processed', 0)
            self.best_loss = checkpoint_data.get('best_loss', float('inf'))
            self.epoch_losses = checkpoint_data.get('epoch_losses', [])
            self.training_metrics.update(
                checkpoint_data.get('training_metrics', {}))

            logger.info(
                f"Checkpoint loaded successfully. Resuming from epoch {self.current_epoch}")

            return {
                'epoch': checkpoint_data['epoch'],
                'loss': checkpoint_data['loss'],
                'total_batches': self.total_batches_processed,
                'config': checkpoint_data.get('config', {})
            }

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _should_save_checkpoint(self) -> bool:
        """Determine if a checkpoint should be saved."""
        return (self.total_batches_processed % self.config.checkpoint_frequency == 0 or
                self.current_epoch == self.config.num_epochs - 1)

    def _log_batch_progress(self, batch_num: int, loss: float, stats: Dict[str, Any]) -> None:
        """Log batch training progress."""
        # Get current memory usage for progress logging
        memory_info = self.memory_monitor.get_memory_usage()

        progress_msg = (f"Epoch {self.current_epoch + 1}, Batch {batch_num}: "
                        f"loss={loss:.6f}, samples={stats['samples_processed']}, "
                        f"triplets={stats['triplets_created']}, "
                        f"time={stats['processing_time']:.3f}s")

        if memory_info["system_memory_mb"]:
            progress_msg += f", mem={memory_info['system_memory_mb']:.1f}MB"
        if memory_info["gpu_memory_mb"]:
            progress_msg += f", gpu_mem={memory_info['gpu_memory_mb']:.1f}MB"

        self.structured_logger.logger.info(progress_msg)

    def _log_epoch_completion(self, epoch: int, loss: float) -> None:
        """Log epoch completion."""
        self.structured_logger.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} completed: "
                                           f"avg_loss={loss:.6f}, best_loss={self.best_loss:.6f}")

    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

    def _create_training_results(self) -> TrainingResults:
        """Create TrainingResults object from current training state."""
        training_time = (self.training_metrics.get('training_end_time', time.time()) -
                         self.training_metrics.get('training_start_time', 0))

        # Find all checkpoint files
        checkpoint_paths = []
        for checkpoint_file in self.output_dir.glob("checkpoint_*.pt"):
            checkpoint_paths.append(str(checkpoint_file))

        return TrainingResults(
            final_loss=self.epoch_losses[-1] if self.epoch_losses else float(
                'inf'),
            epoch_losses=self.epoch_losses.copy(),
            training_time=training_time,
            total_batches=self.total_batches_processed,
            total_samples=self.training_metrics['total_samples_processed'],
            checkpoint_paths=checkpoint_paths,
            metrics=self.training_metrics.copy()
        )

    def _save_training_results(self, results: TrainingResults) -> None:
        """Save training results to file."""
        results_path = self.output_dir / "training_results.json"
        results.save_json(str(results_path))
        logger.info(f"Training results saved to {results_path}")

    def _cleanup_resources(self) -> None:
        """
        Cleanup memory resources after training.
        
        Frees global job pool and embedding cache to release memory.
        """
        import gc
        
        memory_freed = 0
        
        # Cleanup global job pool
        if self.global_job_pool is not None:
            pool_size = len(self.global_job_pool)
            # Estimate memory (rough: ~1KB per job on average)
            estimated_memory = pool_size * 1024 / (1024 * 1024)  # MB
            memory_freed += estimated_memory
            
            self.global_job_pool = None
            logger.info(f"Global job pool cleared: {pool_size} jobs (~{estimated_memory:.1f} MB)")
        
        # Cleanup embedding cache
        if hasattr(self, 'embedding_cache') and self.embedding_cache is not None:
            cache_stats = self.embedding_cache.get_cache_stats()
            cache_memory = cache_stats.get('memory_usage_mb', 0)
            memory_freed += cache_memory
            
            self.embedding_cache.clear_cache()
            logger.info(f"Embedding cache cleared: ~{cache_memory:.1f} MB")
        
        # Force garbage collection
        gc.collect()
        
        if memory_freed > 0:
            logger.info(f"Total memory freed: ~{memory_freed:.1f} MB")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful training interruption."""
        def signal_handler(signum, frame):
            logger.info(
                f"Received signal {signum}, setting interruption flag...")
            self.training_interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status and metrics.

        Returns:
            Dictionary containing current training status
        """
        status = {
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'total_batches_processed': self.total_batches_processed,
            'best_loss': self.best_loss,
            'training_interrupted': self.training_interrupted,
            'epoch_losses': self.epoch_losses.copy(),
            'training_metrics': self.training_metrics.copy()
        }

        # Add comprehensive logging summary
        if hasattr(self, 'structured_logger'):
            status['logging_summary'] = self.structured_logger.get_training_summary()

        return status

    def generate_training_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive training summary with performance stats.

        Returns:
            Dictionary containing detailed training summary
        """
        if not hasattr(self, 'structured_logger'):
            return self.get_training_status()

        # Get logging summary
        logging_summary = self.structured_logger.get_training_summary()

        # Calculate additional performance metrics
        total_training_time = (
            self.training_metrics.get('training_end_time', time.time()) -
            self.training_metrics.get('training_start_time', 0)
        )

        samples_per_second = (
            logging_summary['total_samples'] / total_training_time
            if total_training_time > 0 else 0
        )

        batches_per_second = (
            logging_summary['total_batches'] / total_training_time
            if total_training_time > 0 else 0
        )

        # Compile comprehensive summary
        summary = {
            'training_overview': {
                'total_epochs': len(self.epoch_losses),
                'total_batches': self.total_batches_processed,
                'total_samples': logging_summary['total_samples'],
                'total_triplets': logging_summary['total_triplets'],
                'training_time_seconds': total_training_time,
                'final_loss': self.epoch_losses[-1] if self.epoch_losses else None,
                'best_loss': self.best_loss,
                'training_completed': not self.training_interrupted
            },
            'performance_metrics': {
                'avg_batch_loss': logging_summary['avg_batch_loss'],
                'avg_batch_time_seconds': logging_summary['avg_batch_time'],
                'samples_per_second': samples_per_second,
                'batches_per_second': batches_per_second,
                'failed_batches': logging_summary['failed_batches'],
                'success_rate': (
                    (logging_summary['total_batches'] - logging_summary['failed_batches']) /
                    max(1, logging_summary['total_batches'])
                )
            },
            'system_metrics': {
                'checkpoint_saves': self.training_metrics.get('checkpoint_saves', 0),
                'interruptions': logging_summary.get('interruptions', 0),
                'errors': logging_summary['error_count']
            },
            'configuration': asdict(self.config),
            'logging_summary': logging_summary
        }

        return summary

    def _handle_diagnostic_alert(self, alert_type: str, alert_data: dict) -> None:
        """Handle diagnostic alerts during training."""
        self.structured_logger.logger.warning(
            f"Diagnostic alert: {alert_type}")

        if alert_type == "loss_explosion":
            self.structured_logger.logger.warning(
                "Loss explosion detected - reducing learning rate by 50%")
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= 0.5
                self.structured_logger.logger.info(
                    f"Learning rate reduced from {old_lr} to {param_group['lr']}")

        elif alert_type == "gradient_explosion":
            self.structured_logger.logger.warning(
                "Gradient explosion detected - enabling gradient clipping")
            # Enable gradient clipping for subsequent batches
            if hasattr(self.config, 'max_grad_norm'):
                self.config.max_grad_norm = 1.0

        elif alert_type == "loss_stagnation":
            self.structured_logger.logger.warning(
                "Loss stagnation detected - consider adjusting hyperparameters")
            # Could implement learning rate scheduling here

        elif alert_type == "embedding_collapse":
            self.structured_logger.logger.critical(
                "Embedding collapse detected - training may need restart")
            # Save emergency checkpoint
            emergency_path = self.output_dir / \
                f"emergency_checkpoint_step_{self.total_batches_processed}.pt"
            self.save_checkpoint(emergency_path)
            self.structured_logger.logger.info(
                f"Emergency checkpoint saved: {emergency_path}")

    def close(self) -> None:
        """Close trainer and cleanup resources."""
        # Cleanup global job pool and other memory resources
        self._cleanup_resources()
        
        # Cleanup diagnostic integration
        if self.diagnostic_integration:
            try:
                self.diagnostic_integration.stop_async_diagnostics()

                # Get and log performance stats
                stats = self.diagnostic_integration.get_performance_stats()
                if not stats.get('no_data'):
                    self.structured_logger.logger.info(
                        f"Diagnostic performance: {stats['average_overhead_ms']:.2f}ms avg overhead, "
                        f"{stats['total_steps_monitored']} steps monitored"
                    )

            except Exception as e:
                self.structured_logger.logger.warning(
                    f"Error cleaning up diagnostics: {e}")

        if hasattr(self, 'structured_logger'):
            # Generate final summary before closing
            try:
                final_summary = self.generate_training_summary()
                summary_file = self.output_dir / "final_training_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(final_summary, f, indent=2, default=str)
                self.structured_logger.logger.info(
                    f"Final training summary saved: {summary_file}")
            except Exception as e:
                self.structured_logger.logger.error(
                    f"Failed to save final summary: {e}")

            # Close structured logger
            self.structured_logger.close()

    def resume_training(self, dataset_path: Union[str, Path],
                        checkpoint_path: Union[str, Path]) -> TrainingResults:
        """
        Resume training from a checkpoint.

        Args:
            dataset_path: Path to training dataset
            checkpoint_path: Path to checkpoint file

        Returns:
            TrainingResults from resumed training
        """
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)

        # Reset interruption flag
        self.training_interrupted = False

        # Continue training
        logger.info(f"Resuming training from epoch {self.current_epoch}")
        return self.train(dataset_path)
