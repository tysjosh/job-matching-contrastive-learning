"""
Comprehensive logging and monitoring utilities for contrastive learning training.

This module provides structured logging, metrics collection, and monitoring
functionality for the contrastive learning training system.
"""

import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading
from contextlib import contextmanager


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""
    batch_id: int
    epoch: int
    samples_processed: int
    triplets_created: int
    view_combinations: int
    processing_time: float
    loss: Optional[float]
    memory_usage_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class EpochMetrics:
    """Metrics for a complete epoch."""
    epoch: int
    total_batches: int
    total_samples: int
    total_triplets: int
    avg_loss: float
    min_loss: float
    max_loss: float
    epoch_time: float
    avg_batch_time: float
    memory_peak_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None
    failed_batches: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class TrainingSession:
    """Complete training session metrics."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_epochs: int = 0
    total_batches: int = 0
    total_samples: int = 0
    total_triplets: int = 0
    final_loss: Optional[float] = None
    best_loss: float = float('inf')
    checkpoint_saves: int = 0
    interruptions: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class StructuredLogger:
    """
    Structured logger for contrastive learning training.
    
    Provides structured logging with JSON output, metrics collection,
    and performance monitoring capabilities.
    """
    
    def __init__(self, name: str, log_dir: Union[str, Path], 
                 log_level: int = logging.INFO):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_loggers(log_level)
        
        # Metrics storage
        self.batch_metrics: List[BatchMetrics] = []
        self.epoch_metrics: List[EpochMetrics] = []
        self.current_session: Optional[TrainingSession] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Memory monitoring
        self._memory_monitor = MemoryMonitor()
    
    def _setup_loggers(self, log_level: int) -> None:
        """Setup structured loggers with file and console handlers."""
        # Main logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()  # Clear existing handlers
        
        # Console handler with structured format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON handler for structured metrics
        self.metrics_file = self.log_dir / f"{self.name}_metrics.jsonl"
        self.metrics_handler = open(self.metrics_file, 'a')
        
        # Error log file
        error_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def start_training_session(self, config: Dict[str, Any]) -> str:
        """
        Start a new training session.
        
        Args:
            config: Training configuration
            
        Returns:
            Session ID
        """
        session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock:
            self.current_session = TrainingSession(
                session_id=session_id,
                start_time=time.time()
            )
        
        # Log session start
        self.logger.info(f"Starting training session: {session_id}")
        self._log_structured_event("training_session_start", {
            "session_id": session_id,
            "config": config,
            "timestamp": time.time()
        })
        
        return session_id
    
    def end_training_session(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        End the current training session.
        
        Args:
            final_metrics: Optional final training metrics
        """
        if not self.current_session:
            self.logger.warning("No active training session to end")
            return
        
        with self._lock:
            self.current_session.end_time = time.time()
            if final_metrics:
                self.current_session.final_loss = final_metrics.get('final_loss')
                self.current_session.best_loss = final_metrics.get('best_loss', self.current_session.best_loss)
        
        # Log session end
        session_duration = self.current_session.end_time - self.current_session.start_time
        self.logger.info(f"Training session ended: {self.current_session.session_id}")
        self.logger.info(f"Session duration: {session_duration:.2f}s")
        
        self._log_structured_event("training_session_end", {
            "session": self.current_session.to_dict(),
            "duration_seconds": session_duration,
            "timestamp": time.time()
        })
        
        # Save session summary
        self._save_session_summary()
    
    def log_batch_start(self, batch_id: int, epoch: int, batch_size: int) -> None:
        """
        Log the start of batch processing.
        
        Args:
            batch_id: Batch identifier
            epoch: Current epoch
            batch_size: Number of samples in batch
        """
        self.logger.debug(f"Starting batch {batch_id} (epoch {epoch}) with {batch_size} samples")
        
        self._log_structured_event("batch_start", {
            "batch_id": batch_id,
            "epoch": epoch,
            "batch_size": batch_size,
            "timestamp": time.time()
        })
    
    def log_batch_completion(self, metrics: BatchMetrics) -> None:
        """
        Log batch completion with metrics.
        
        Args:
            metrics: Batch processing metrics
        """
        with self._lock:
            self.batch_metrics.append(metrics)
            if self.current_session:
                self.current_session.total_batches += 1
                self.current_session.total_samples += metrics.samples_processed
                self.current_session.total_triplets += metrics.triplets_created
        
        # Log batch completion
        loss_str = f"{metrics.loss:.6f}" if metrics.loss is not None else "N/A"
        self.logger.info(
            f"Batch {metrics.batch_id} completed: "
            f"loss={loss_str}, "
            f"samples={metrics.samples_processed}, "
            f"triplets={metrics.triplets_created}, "
            f"time={metrics.processing_time:.3f}s"
        )
        
        self._log_structured_event("batch_completion", metrics.to_dict())
    
    def log_batch_error(self, batch_id: int, epoch: int, error: Exception, 
                       context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log batch processing error with context.
        
        Args:
            batch_id: Batch identifier
            epoch: Current epoch
            error: Exception that occurred
            context: Additional context information
        """
        error_info = {
            "batch_id": batch_id,
            "epoch": epoch,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Add to session errors
        with self._lock:
            if self.current_session:
                self.current_session.errors.append(error_info)
        
        self.logger.error(f"Batch {batch_id} failed: {error}")
        self._log_structured_event("batch_error", error_info)
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """
        Log the start of an epoch.
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        self.logger.info(f"Starting epoch {epoch + 1}/{total_epochs}")
        
        self._log_structured_event("epoch_start", {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "timestamp": time.time()
        })
    
    def log_epoch_completion(self, metrics: EpochMetrics) -> None:
        """
        Log epoch completion with metrics.
        
        Args:
            metrics: Epoch processing metrics
        """
        with self._lock:
            self.epoch_metrics.append(metrics)
            if self.current_session:
                self.current_session.total_epochs += 1
                if metrics.min_loss < self.current_session.best_loss:
                    self.current_session.best_loss = metrics.min_loss
        
        # Log epoch completion
        self.logger.info(
            f"Epoch {metrics.epoch + 1} completed: "
            f"avg_loss={metrics.avg_loss:.6f}, "
            f"batches={metrics.total_batches}, "
            f"samples={metrics.total_samples}, "
            f"time={metrics.epoch_time:.2f}s"
        )
        
        self._log_structured_event("epoch_completion", metrics.to_dict())
    
    def log_checkpoint_save(self, epoch: int, checkpoint_path: str, 
                           metrics: Dict[str, Any]) -> None:
        """
        Log checkpoint save event.
        
        Args:
            epoch: Current epoch
            checkpoint_path: Path to saved checkpoint
            metrics: Current training metrics
        """
        with self._lock:
            if self.current_session:
                self.current_session.checkpoint_saves += 1
        
        self.logger.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
        
        self._log_structured_event("checkpoint_save", {
            "epoch": epoch,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
            "timestamp": time.time()
        })
    
    def log_training_interruption(self, reason: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log training interruption.
        
        Args:
            reason: Reason for interruption
            context: Additional context
        """
        with self._lock:
            if self.current_session:
                self.current_session.interruptions += 1
        
        self.logger.warning(f"Training interrupted: {reason}")
        
        self._log_structured_event("training_interruption", {
            "reason": reason,
            "context": context or {},
            "timestamp": time.time()
        })
    
    def log_memory_usage(self, stage: str, memory_mb: float, gpu_memory_mb: Optional[float] = None) -> None:
        """
        Log memory usage at different stages.
        
        Args:
            stage: Training stage (e.g., 'batch_start', 'batch_end')
            memory_mb: System memory usage in MB
            gpu_memory_mb: GPU memory usage in MB (if available)
        """
        self.logger.debug(f"Memory usage at {stage}: {memory_mb:.1f}MB RAM" + 
                         (f", {gpu_memory_mb:.1f}MB GPU" if gpu_memory_mb else ""))
        
        self._log_structured_event("memory_usage", {
            "stage": stage,
            "memory_mb": memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
            "timestamp": time.time()
        })
    
    def _log_structured_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log structured event to JSON file.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        
        try:
            json_line = json.dumps(event, default=str)
            self.metrics_handler.write(json_line + '\n')
            self.metrics_handler.flush()
        except Exception as e:
            self.logger.error(f"Failed to write structured event: {e}")
    
    def _save_session_summary(self) -> None:
        """Save training session summary."""
        if not self.current_session:
            return
        
        summary_file = self.log_dir / f"session_summary_{self.current_session.session_id}.json"
        
        summary = {
            "session": self.current_session.to_dict(),
            "batch_metrics_count": len(self.batch_metrics),
            "epoch_metrics_count": len(self.epoch_metrics),
            "avg_batch_time": sum(m.processing_time for m in self.batch_metrics) / len(self.batch_metrics) if self.batch_metrics else 0,
            "avg_epoch_time": sum(m.epoch_time for m in self.epoch_metrics) / len(self.epoch_metrics) if self.epoch_metrics else 0
        }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"Session summary saved: {summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save session summary: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary containing training summary
        """
        with self._lock:
            valid_losses = [m.loss for m in self.batch_metrics if m.loss is not None]
            return {
                "session": self.current_session.to_dict() if self.current_session else None,
                "total_batches": len(self.batch_metrics),
                "total_epochs": len(self.epoch_metrics),
                "avg_batch_loss": sum(valid_losses) / len(valid_losses) if valid_losses else 0,
                "avg_batch_time": sum(m.processing_time for m in self.batch_metrics) / len(self.batch_metrics) if self.batch_metrics else 0,
                "total_samples": sum(m.samples_processed for m in self.batch_metrics),
                "total_triplets": sum(m.triplets_created for m in self.batch_metrics),
                "failed_batches": len([m for m in self.batch_metrics if m.loss is None]),
                "error_count": len(self.current_session.errors) if self.current_session else 0
            }
    
    def close(self) -> None:
        """Close logger and cleanup resources."""
        if hasattr(self, 'metrics_handler'):
            self.metrics_handler.close()
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class MemoryMonitor:
    """Monitor system and GPU memory usage."""
    
    def __init__(self):
        """Initialize memory monitor."""
        self._torch_available = False
        try:
            import torch
            self._torch_available = torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False
    
    def get_memory_usage(self) -> Dict[str, Optional[float]]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage in MB
        """
        memory_info = {
            "system_memory_mb": None,
            "gpu_memory_mb": None
        }
        
        # System memory
        if self._psutil_available:
            try:
                import psutil
                process = psutil.Process()
                memory_info["system_memory_mb"] = process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
        
        # GPU memory
        if self._torch_available:
            try:
                import torch
                if torch.cuda.is_available():
                    memory_info["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            except Exception:
                pass
        
        return memory_info


@contextmanager
def timed_operation(logger: StructuredLogger, operation_name: str, 
                   context: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing operations.
    
    Args:
        logger: Structured logger instance
        operation_name: Name of the operation
        context: Additional context information
    """
    start_time = time.time()
    logger.logger.debug(f"Starting {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.logger.debug(f"Completed {operation_name} in {duration:.3f}s")
        
        logger._log_structured_event("timed_operation", {
            "operation": operation_name,
            "duration_seconds": duration,
            "success": True,
            "context": context or {}
        })
        
    except Exception as e:
        duration = time.time() - start_time
        logger.logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}")
        
        logger._log_structured_event("timed_operation", {
            "operation": operation_name,
            "duration_seconds": duration,
            "success": False,
            "error": str(e),
            "context": context or {}
        })
        raise


def setup_training_logger(name: str, log_dir: Union[str, Path], 
                         log_level: int = logging.INFO) -> StructuredLogger:
    """
    Setup structured logger for training.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(name, log_dir, log_level)