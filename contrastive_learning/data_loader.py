"""
DataLoader for efficient batch processing of contrastive learning training data.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Iterator, Union, Optional, TextIO
from dataclasses import dataclass

from .data_structures import TrainingSample, TrainingConfig


logger = logging.getLogger(__name__)


@dataclass
class DataLoaderStats:
    """Statistics about data loading process."""
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    total_batches: int = 0
    skipped_records: int = 0
    # NEW: Self-supervised training statistics
    augmented_samples_found: int = 0
    augmented_samples_used: int = 0
    original_samples_found: int = 0
    original_samples_used: int = 0
    # NEW: Label conversion statistics
    numeric_labels_converted: int = 0
    string_labels_converted: int = 0
    boolean_labels_converted: int = 0


class DataLoader:
    """
    Efficient data loader for contrastive learning training samples.

    Supports streaming JSONL files with configurable batch sizes and memory-efficient
    processing for large datasets.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize DataLoader with training configuration.

        Args:
            config: Training configuration containing batch_size and other parameters
        """
        self.config = config
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle_data
        self.stats = DataLoaderStats()

        # Validate batch size is within acceptable range
        if not 32 <= self.batch_size <= 512:
            logger.warning(
                f"Batch size {self.batch_size} is outside recommended range [32, 512]")

    def load_batches(self, file_path: Union[str, Path]) -> Iterator[List[TrainingSample]]:
        """
        Load training samples in batches from a JSONL file.

        Args:
            file_path: Path to JSONL file containing training samples

        Yields:
            List[TrainingSample]: Batches of validated training samples

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Training data file not found: {file_path}")

        logger.info(f"Loading training data from {file_path}")
        logger.info(f"Training phase: {self.config.training_phase}")
        logger.info(f"Batch size: {self.batch_size}, Shuffle: {self.shuffle}")

        # Reset stats for new loading session
        self.stats = DataLoaderStats()

        with open(path, 'r', encoding='utf-8') as file:
            if self.config.training_phase == "self_supervised":
                yield from self._load_self_supervised_batches(file)
            else:
                yield from self._stream_batches(file)

        # Log final statistics
        self._log_loading_stats()

    def load_batches_from_stream(self, stream: TextIO) -> Iterator[List[TrainingSample]]:
        """
        Load training samples in batches from a text stream.

        Args:
            stream: Text stream containing JSONL data

        Yields:
            List[TrainingSample]: Batches of validated training samples
        """
        logger.info("Loading training data from stream")
        logger.info(f"Batch size: {self.batch_size}, Shuffle: {self.shuffle}")

        # Reset stats for new loading session
        self.stats = DataLoaderStats()

        yield from self._stream_batches(stream)

        # Log final statistics
        self._log_loading_stats()

    def _load_self_supervised_batches(self, stream: TextIO) -> Iterator[List[TrainingSample]]:
        """
        Load batches for self-supervised training, filtering by augmentation metadata.

        Args:
            stream: Text stream to read from

        Yields:
            List[TrainingSample]: Batches of filtered training samples for self-supervised learning
        """
        logger.info("Loading data in self-supervised mode")
        logger.info(
            f"Use augmentation labels only: {self.config.use_augmentation_labels_only}")
        logger.info(
            f"Augmentation positive ratio: {self.config.augmentation_positive_ratio}")

        current_batch = []
        line_number = 0

        for line in stream:
            line_number += 1
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            try:
                # Parse JSON record
                record = json.loads(line)
                self.stats.total_samples += 1

                # Create and validate training sample
                sample = self.create_training_sample(record, line_number)
                if not sample or not self.validate_sample(sample):
                    self.stats.invalid_samples += 1
                    continue

                # Track sample types for statistics
                is_augmented = self._is_augmented_sample(sample)
                if is_augmented:
                    self.stats.augmented_samples_found += 1
                else:
                    self.stats.original_samples_found += 1

                # Apply self-supervised filtering logic
                if self._should_include_in_self_supervised(sample):
                    # Handle augmented samples
                    if is_augmented:
                        # Apply augmentation_positive_ratio filtering
                        if self._should_use_augmented_sample(self.stats.augmented_samples_found):
                            current_batch.append(sample)
                            self.stats.augmented_samples_used += 1
                            self.stats.valid_samples += 1
                        else:
                            # Skip this augmented sample due to ratio control
                            continue
                    else:
                        # Handle original samples - include if not using augmentation labels only
                        if not self.config.use_augmentation_labels_only:
                            current_batch.append(sample)
                            self.stats.original_samples_used += 1
                            self.stats.valid_samples += 1
                        else:
                            # Skip original samples when using augmentation labels only
                            continue
                else:
                    self.stats.invalid_samples += 1
                    continue

                # Yield batch when it reaches the configured size
                if len(current_batch) >= self.batch_size:
                    batch = self._prepare_batch(current_batch)
                    self.stats.total_batches += 1
                    yield batch
                    current_batch = []

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_number}: {e}")
                self.stats.skipped_records += 1
                continue
            except Exception as e:
                logger.warning(f"Error processing line {line_number}: {e}")
                self.stats.skipped_records += 1
                continue

        # Yield remaining samples as final batch if any
        if current_batch:
            batch = self._prepare_batch(current_batch)
            self.stats.total_batches += 1
            yield batch

        # Log self-supervised specific statistics
        logger.info(f"Self-supervised filtering results:")
        logger.info(
            f"  Original samples found: {self.stats.original_samples_found}")
        logger.info(
            f"  Original samples used: {self.stats.original_samples_used}")
        logger.info(
            f"  Augmented samples found: {self.stats.augmented_samples_found}")
        logger.info(
            f"  Augmented samples used: {self.stats.augmented_samples_used}")

        if self.stats.augmented_samples_found > 0:
            actual_ratio = self.stats.augmented_samples_used / \
                self.stats.augmented_samples_found
            logger.info(
                f"  Actual augmentation usage ratio: {actual_ratio:.3f}")
            logger.info(
                f"  Configured augmentation ratio: {self.config.augmentation_positive_ratio}")

        total_found = self.stats.original_samples_found + \
            self.stats.augmented_samples_found
        if total_found > 0:
            augmented_percentage = (
                self.stats.augmented_samples_found / total_found) * 100
            logger.info(
                f"  Dataset composition: {augmented_percentage:.1f}% augmented, {100-augmented_percentage:.1f}% original")

    def _stream_batches(self, stream: TextIO) -> Iterator[List[TrainingSample]]:
        """
        Stream batches from a text stream with memory-efficient processing.

        Args:
            stream: Text stream to read from

        Yields:
            List[TrainingSample]: Batches of validated training samples
        """
        current_batch = []
        line_number = 0

        for line in stream:
            line_number += 1
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            try:
                # Parse JSON record
                record = json.loads(line)
                self.stats.total_samples += 1

                # Create and validate training sample
                sample = self.create_training_sample(record, line_number)
                if sample and self.validate_sample(sample):
                    current_batch.append(sample)
                    self.stats.valid_samples += 1
                else:
                    self.stats.invalid_samples += 1
                    continue

                # Yield batch when it reaches the configured size
                if len(current_batch) >= self.batch_size:
                    batch = self._prepare_batch(current_batch)
                    self.stats.total_batches += 1
                    yield batch
                    current_batch = []

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_number}: {e}")
                self.stats.skipped_records += 1
                continue
            except Exception as e:
                logger.warning(f"Error processing line {line_number}: {e}")
                self.stats.skipped_records += 1
                continue

        # Yield remaining samples as final batch if any
        if current_batch:
            batch = self._prepare_batch(current_batch)
            self.stats.total_batches += 1
            yield batch

    def create_training_sample(self, record: Dict[str, Any], line_number: int = 0) -> Optional[TrainingSample]:
        """
        Create a TrainingSample from a dictionary record.

        Supports flexible label formats:
        - String: 'positive'/'negative', 'pos'/'neg', 'match'/'no_match', 'yes'/'no', 'y'/'n'
        - Numeric: 1/0 (1=positive, 0=negative)
        - Boolean: true/false (true=positive, false=negative)

        Args:
            record: Dictionary containing sample data
            line_number: Line number for error reporting

        Returns:
            TrainingSample if valid, None if invalid
        """
        try:
            # Extract required fields
            resume = record.get('resume')
            job = record.get('job')
            label = record.get('label')

            # Generate sample_id if not provided
            sample_id = record.get('sample_id', f"sample_{line_number}")

            # Extract optional metadata
            metadata = record.get('metadata', {})

            # Basic field validation
            if not resume or not isinstance(resume, dict):
                logger.warning(
                    f"Line {line_number}: Invalid or missing resume field")
                return None

            if not job or not isinstance(job, dict):
                logger.warning(
                    f"Line {line_number}: Invalid or missing job field")
                return None

            # Normalize label format
            normalized_label = self._normalize_label(label, line_number)
            if normalized_label is None:
                return None
            label = normalized_label

            return TrainingSample(
                resume=resume,
                job=job,
                label=label,
                sample_id=sample_id,
                metadata=metadata
            )

        except Exception as e:
            logger.warning(
                f"Line {line_number}: Error creating training sample: {e}")
            return None

    def validate_sample(self, sample: TrainingSample) -> bool:
        """
        Validate a training sample for completeness and correctness.

        Args:
            sample: TrainingSample to validate

        Returns:
            bool: True if sample is valid, False otherwise
        """
        try:
            # Check if resume has required fields
            if not self._validate_resume(sample.resume):
                return False

            # Check if job has required fields
            if not self._validate_job(sample.job):
                return False

            # Additional validation passed in __post_init__ of TrainingSample
            return True

        except Exception as e:
            logger.warning(
                f"Sample validation error for {sample.sample_id}: {e}")
            return False

    def _validate_resume(self, resume: Dict[str, Any]) -> bool:
        """
        Validate resume data structure, supporting both basic and enhanced formats.

        Args:
            resume: Resume dictionary to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Check for essential resume fields (flexible for enhanced format)
        essential_fields = ['experience', 'skills']

        # Allow at least one essential field to be present
        has_essential_field = any(
            field in resume for field in essential_fields)
        if not has_essential_field:
            logger.debug(
                f"Resume missing essential fields: {essential_fields}")
            return False

        # Validate experience if present
        if 'experience' in resume:
            experience = resume['experience']
            # Support both string and list formats for experience
            if isinstance(experience, str):
                # String format is valid (simple text experience)
                if not experience.strip():
                    logger.debug("Resume experience string is empty")
                    return False
            elif isinstance(experience, list):
                # List format with enhanced experience objects
                for exp in experience:
                    if isinstance(exp, dict) and 'responsibilities' in exp:
                        responsibilities = exp['responsibilities']
                        # Validate enhanced responsibilities format
                        if not isinstance(responsibilities, dict):
                            logger.debug(
                                "Experience responsibilities must be a dict")
                            return False

                        # Check for required responsibility fields - be flexible about format
                        has_content = False

                        # Check technical_terms (list format)
                        if 'technical_terms' in responsibilities and isinstance(responsibilities['technical_terms'], list):
                            # Any technical_terms list is acceptable (empty or not) as it's parsed from content
                            has_content = True

                        # Check achievements/impact (string format)
                        for field in ['achievements', 'impact']:
                            if field in responsibilities:
                                value = responsibilities[field]
                                if isinstance(value, str) and value.strip():
                                    has_content = True
                                elif isinstance(value, list) and value:  # non-empty list
                                    has_content = True

                        # If no valid content in any expected fields, check if we should accept anyway
                        # Since this is generated training data, we can be more lenient
                        has_any_fields = any(field in responsibilities for field in [
                                             'technical_terms', 'achievements', 'impact', 'action_verbs'])
                        if not has_content and has_any_fields:
                            # Accept samples that at least have the proper responsibilities structure
                            # even if content is empty (this is valid for our use case)
                            has_content = True

                        if not has_content:
                            logger.debug(
                                "No meaningful content found in responsibilities")
                            return False
            else:
                logger.debug("Resume experience must be a string or list")
                return False

        # Validate skills if present
        if 'skills' in resume:
            skills = resume['skills']
            if not isinstance(skills, list):
                logger.debug("Resume skills must be a list")
                return False

            # Support enhanced skills format with categories and levels
            for skill in skills:
                if isinstance(skill, dict):
                    # Enhanced skill format validation
                    if 'name' not in skill and 'original_name' not in skill:
                        logger.debug(
                            "Enhanced skill must have 'name' or 'original_name'")
                        return False

        return True

    def _validate_job(self, job: Dict[str, Any]) -> bool:
        """
        Validate job data structure.

        Args:
            job: Job dictionary to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Check for essential job fields
        required_fields = ['title', 'description']  # Minimal required fields

        for field in required_fields:
            if field not in job:
                logger.debug(f"Job missing required field: {field}")
                return False

        # Validate title is a non-empty string
        if not isinstance(job.get('title'), str) or not job.get('title').strip():
            logger.debug("Job title must be a non-empty string")
            return False

        # Validate description is a dictionary with a non-empty 'original' field
        description = job.get('description')
        if not isinstance(description, dict) or not description.get('original', '').strip():
            logger.debug(
                "Job description must be a dictionary with a non-empty 'original' field")
            return False

        return True

    def _should_include_in_self_supervised(self, sample: TrainingSample) -> bool:
        """
        Determine if a sample should be included in self-supervised training.

        Args:
            sample: TrainingSample to evaluate

        Returns:
            bool: True if sample should be included, False otherwise
        """
        # If using augmentation labels only, only include augmented samples
        if self.config.use_augmentation_labels_only:
            return self._is_augmented_sample(sample)

        # Otherwise, include all valid samples (both original and augmented)
        return True

    def _is_augmented_sample(self, sample: TrainingSample) -> bool:
        """
        Check if a sample is augmentation-generated based on metadata.

        Args:
            sample: TrainingSample to check

        Returns:
            bool: True if sample is augmentation-generated, False otherwise
        """
        augmentation_type = sample.metadata.get(
            'augmentation_type', 'Original')

        # Handle None augmentation_type (treat as Original)
        if augmentation_type is None:
            augmentation_type = 'Original'

        # Define augmentation types that should be considered as augmented positive pairs
        augmented_types = {
            # Aspirational resume ↔ senior job
            'Aspirational Match (Tup/Tup)',
            # Foundational resume ↔ junior job
            'Foundational Match (Tdown/Tdown)',
            'Aspirational Match',               # Generic aspirational match
            'Foundational Match',               # Generic foundational match
            'Hard_Negative',                    # Hard negative augmentation
            'Augmented'                         # Generic augmented label
        }

        # Consider samples as augmented if they match known augmentation patterns
        # This provides compatibility with various augmentation formats
        is_augmented = (
            augmentation_type in augmented_types or
            augmentation_type.startswith('Aspirational') or
            augmentation_type.startswith('Foundational') or
            augmentation_type.startswith('Hard_Negative') or
            (augmentation_type != 'Original' and 'Match' in augmentation_type)
        )

        return is_augmented

    def _normalize_label(self, label: Any, line_number: int = 0) -> Optional[str]:
        """
        Normalize label to standard 'positive'/'negative' format.

        Supports multiple input formats:
        - Numeric: 1/0 (1=positive, 0=negative)
        - Boolean: true/false (true=positive, false=negative)
        - String: 'positive'/'negative', 'pos'/'neg', 'match'/'no_match', etc.

        Args:
            label: Label value in any supported format
            line_number: Line number for error reporting

        Returns:
            Normalized label string ('positive' or 'negative'), or None if invalid
        """
        if label is None:
            logger.warning(f"Line {line_number}: Missing label field")
            return None

        # Handle numeric labels (most common case for generated data)
        if label == 1 or label == '1':
            if label != 'positive':  # Only count if conversion needed
                self.stats.numeric_labels_converted += 1
            return 'positive'
        elif label == 0 or label == '0':
            if label != 'negative':  # Only count if conversion needed
                self.stats.numeric_labels_converted += 1
            return 'negative'

        # Handle boolean labels
        elif label is True or label == 'true':
            self.stats.boolean_labels_converted += 1
            return 'positive'
        elif label is False or label == 'false':
            self.stats.boolean_labels_converted += 1
            return 'negative'

        # Handle string labels (case insensitive)
        elif isinstance(label, str):
            label_lower = label.lower().strip()

            # Positive variations
            if label_lower in ['positive', 'pos', 'match', 'yes', 'y', 'true']:
                if label_lower != 'positive':  # Only count if conversion needed
                    self.stats.string_labels_converted += 1
                return 'positive'

            # Negative variations
            elif label_lower in ['negative', 'neg', 'no_match', 'no', 'n', 'false']:
                if label_lower != 'negative':  # Only count if conversion needed
                    self.stats.string_labels_converted += 1
                return 'negative'

            else:
                logger.warning(f"Line {line_number}: Invalid string label '{label}' "
                               f"(supported: positive/negative, pos/neg, match/no_match, yes/no, y/n, 1/0)")
                return None

        # Handle other numeric types
        elif isinstance(label, (int, float)):
            if label == 1:
                return 'positive'
            elif label == 0:
                return 'negative'
            else:
                logger.warning(
                    f"Line {line_number}: Invalid numeric label '{label}' (must be 1 or 0)")
                return None

        else:
            logger.warning(f"Line {line_number}: Invalid label type '{type(label).__name__}' "
                           f"(supported types: string, int, bool)")
            return None

    def _should_use_augmented_sample(self, augmented_count: int) -> bool:
        """
        Determine if an augmented sample should be used based on augmentation_positive_ratio.

        Uses deterministic sampling to ensure consistent behavior across runs while
        respecting the configured ratio of augmented positives to use.

        Args:
            augmented_count: Current count of augmented samples encountered (1-indexed)

        Returns:
            bool: True if sample should be used, False otherwise
        """
        # If ratio is 1.0, use all augmented samples
        if self.config.augmentation_positive_ratio >= 1.0:
            return True

        # If ratio is 0.0, use no augmented samples
        if self.config.augmentation_positive_ratio <= 0.0:
            return False

        # Use deterministic sampling based on ratio with proper precision handling
        # For ratios > 0.5, we need to include more samples than we skip
        # For ratios < 0.5, we skip more samples than we include

        ratio = self.config.augmentation_positive_ratio

        # Handle common ratios with exact logic for precision
        if abs(ratio - 0.5) < 0.01:
            # 50%: use every 2nd sample (1, 3, 5, ...)
            return augmented_count % 2 == 1
        elif abs(ratio - 1/3) < 0.01 or abs(ratio - 0.33) < 0.01:
            # 33%: use every 3rd sample (1, 4, 7, ...)
            return augmented_count % 3 == 1
        elif abs(ratio - 0.25) < 0.01:
            # 25%: use every 4th sample (1, 5, 9, ...)
            return augmented_count % 4 == 1
        elif abs(ratio - 2/3) < 0.01 or abs(ratio - 0.67) < 0.01:
            # 67%: skip every 3rd sample (use 1, 2, 4, 5, 7, 8, ...)
            return augmented_count % 3 != 0
        elif abs(ratio - 0.75) < 0.01:
            # 75%: skip every 4th sample (use 1, 2, 3, 5, 6, 7, ...)
            return augmented_count % 4 != 0
        elif abs(ratio - 0.8) < 0.01:
            # 80%: skip every 5th sample
            return augmented_count % 5 != 0
        elif abs(ratio - 0.9) < 0.01:
            # 90%: skip every 10th sample
            return augmented_count % 10 != 0
        elif ratio > 0.5:
            # General case for ratios > 0.5: calculate skip interval
            # For ratio 0.7, we want to use 70%, so skip 30% → skip every ~3.33 samples
            skip_ratio = 1.0 - ratio
            if skip_ratio > 0:
                skip_interval = max(2, round(1.0 / skip_ratio))
                return augmented_count % skip_interval != 0
            return True
        else:
            # General case for ratios < 0.5: calculate use interval
            # For ratio 0.2, we want to use 20% → use every 5th sample
            use_interval = max(1, round(1.0 / ratio))
            return augmented_count % use_interval == 1

    def validate_self_supervised_compatibility(self, file_path: Union[str, Path], sample_size: int = 100) -> Dict[str, Any]:
        """
        Validate that a dataset is compatible with self-supervised training.

        Args:
            file_path: Path to JSONL file to validate
            sample_size: Number of samples to check for validation

        Returns:
            Dict containing validation results and recommendations
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Training data file not found: {file_path}")

        logger.info(
            f"Validating self-supervised compatibility for {file_path}")

        total_checked = 0
        augmented_found = 0
        original_found = 0
        missing_metadata = 0
        augmentation_types = set()

        with open(path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                if total_checked >= sample_size:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    sample = self.create_training_sample(record, line_num)

                    if sample and self.validate_sample(sample):
                        total_checked += 1

                        # Check metadata presence
                        if 'augmentation_type' not in sample.metadata:
                            missing_metadata += 1
                        else:
                            augmentation_type = sample.metadata['augmentation_type']
                            augmentation_types.add(augmentation_type)

                            if self._is_augmented_sample(sample):
                                augmented_found += 1
                            else:
                                original_found += 1

                except (json.JSONDecodeError, Exception):
                    continue

        # Generate validation report
        validation_results = {
            'total_samples_checked': total_checked,
            'augmented_samples': augmented_found,
            'original_samples': original_found,
            'missing_metadata': missing_metadata,
            'augmentation_types_found': list(augmentation_types),
            'is_compatible': True,
            'warnings': [],
            'recommendations': []
        }

        # Check compatibility and generate warnings/recommendations
        if total_checked == 0:
            validation_results['is_compatible'] = False
            validation_results['warnings'].append(
                "No valid samples found in dataset")

        if missing_metadata > 0:
            validation_results['warnings'].append(
                f"{missing_metadata}/{total_checked} samples missing augmentation_type metadata")

        if augmented_found == 0:
            validation_results['warnings'].append(
                "No augmented samples found - self-supervised training may not be effective")
            validation_results['recommendations'].append(
                "Consider running data augmentation before self-supervised training")

        augmented_ratio = augmented_found / total_checked if total_checked > 0 else 0
        if augmented_ratio < 0.1:
            validation_results['warnings'].append(
                f"Low augmentation ratio ({augmented_ratio:.2%}) - limited self-supervised signal")

        if self.config.use_augmentation_labels_only and augmented_found == 0:
            validation_results['is_compatible'] = False
            validation_results['warnings'].append(
                "use_augmentation_labels_only=True but no augmented samples found")

        logger.info(f"Validation complete: {validation_results}")
        return validation_results

    def _prepare_batch(self, samples: List[TrainingSample]) -> List[TrainingSample]:
        """
        Prepare a batch of samples, applying shuffling if configured.

        Args:
            samples: List of training samples

        Returns:
            List[TrainingSample]: Prepared batch
        """
        if self.shuffle:
            # Create a copy to avoid modifying the original list
            batch = samples.copy()
            random.shuffle(batch)
            return batch
        else:
            return samples

    def _log_loading_stats(self) -> None:
        """Log statistics about the data loading process."""
        logger.info("Data loading completed:")
        logger.info(f"  Total samples processed: {self.stats.total_samples}")
        logger.info(f"  Valid samples: {self.stats.valid_samples}")
        logger.info(f"  Invalid samples: {self.stats.invalid_samples}")
        logger.info(f"  Skipped records: {self.stats.skipped_records}")
        logger.info(f"  Total batches created: {self.stats.total_batches}")

        # Log label conversion statistics
        total_conversions = (self.stats.numeric_labels_converted +
                             self.stats.string_labels_converted +
                             self.stats.boolean_labels_converted)
        if total_conversions > 0:
            logger.info(f"  Label conversions performed: {total_conversions}")
            logger.info(
                f"    Numeric labels (1/0): {self.stats.numeric_labels_converted}")
            logger.info(
                f"    String labels (pos/neg, etc.): {self.stats.string_labels_converted}")
            logger.info(
                f"    Boolean labels (true/false): {self.stats.boolean_labels_converted}")

        if self.stats.total_samples > 0:
            valid_rate = (self.stats.valid_samples /
                          self.stats.total_samples) * 100
            logger.info(f"  Validation rate: {valid_rate:.1f}%")

    def get_stats(self) -> DataLoaderStats:
        """
        Get current loading statistics.

        Returns:
            DataLoaderStats: Current statistics
        """
        return self.stats

    def load_global_job_pool(self, file_path: Union[str, Path], max_jobs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load all unique jobs from dataset for global negative sampling.

        Args:
            file_path: Path to JSONL file containing training samples
            max_jobs: Maximum number of jobs to load (for memory management)

        Returns:
            List of unique job dictionaries

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Training data file not found: {file_path}")

        logger.info(f"Loading global job pool from {file_path}")
        if max_jobs:
            logger.info(f"Limited to {max_jobs} jobs for memory management")

        jobs = []
        seen_job_ids = set()

        with open(path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    job = record.get('job')

                    if job and isinstance(job, dict):
                        # Create a unique identifier for deduplication
                        job_id = job.get('job_id', job.get(
                            'title', f'job_{line_number}'))

                        if job_id not in seen_job_ids:
                            jobs.append(job)
                            seen_job_ids.add(job_id)

                            # Stop if we've reached the maximum
                            if max_jobs and len(jobs) >= max_jobs:
                                logger.info(
                                    f"Reached maximum job pool size: {max_jobs}")
                                break

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.warning(
                        f"Error processing line {line_number} for job pool: {e}")
                    continue

        logger.info(f"Loaded {len(jobs)} unique jobs into global pool")
        return jobs

    def estimate_memory_usage(self, file_path: Union[str, Path], sample_size: int = 1000) -> Dict[str, Any]:
        """
        Estimate memory usage for loading the dataset.

        Args:
            file_path: Path to the dataset file
            sample_size: Number of samples to use for estimation

        Returns:
            Dict containing memory usage estimates
        """
        import sys

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Sample a few records to estimate memory usage
        sample_memory = 0
        sample_count = 0

        with open(path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                if line_num >= sample_size:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    sample = self.create_training_sample(record, line_num)
                    if sample and self.validate_sample(sample):
                        sample_memory += sys.getsizeof(sample.to_dict())
                        sample_count += 1
                except:
                    continue

        if sample_count == 0:
            return {"error": "No valid samples found for estimation"}

        # Estimate total file size and memory requirements
        file_size = path.stat().st_size
        avg_sample_memory = sample_memory / sample_count

        # Rough estimation of total samples based on file size
        avg_line_size = file_size / sum(1 for _ in open(path, 'r'))
        estimated_total_samples = file_size / avg_line_size

        # Memory estimates
        estimated_total_memory = estimated_total_samples * avg_sample_memory
        batch_memory = self.batch_size * avg_sample_memory

        return {
            "file_size_mb": file_size / (1024 * 1024),
            "estimated_total_samples": int(estimated_total_samples),
            "avg_sample_memory_bytes": int(avg_sample_memory),
            "estimated_total_memory_mb": estimated_total_memory / (1024 * 1024),
            "batch_memory_mb": batch_memory / (1024 * 1024),
            "sample_count_used": sample_count
        }

    def estimate_job_pool_memory(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate memory usage of a job pool.

        Args:
            jobs: List of job dictionaries

        Returns:
            Dict containing memory usage estimates
        """
        import sys

        if not jobs:
            return {"total_jobs": 0, "memory_mb": 0.0}

        # Sample a subset for estimation if pool is large
        sample_size = min(100, len(jobs))
        sample_jobs = jobs[:sample_size]

        total_memory = sum(sys.getsizeof(json.dumps(job))
                           for job in sample_jobs)
        avg_job_memory = total_memory / sample_size
        estimated_total_memory = avg_job_memory * len(jobs)

        return {
            "total_jobs": len(jobs),
            "memory_mb": estimated_total_memory / (1024 * 1024),
            "avg_job_bytes": int(avg_job_memory),
            "sample_size": sample_size
        }


class GlobalJobPoolManager:
    """
    Manages global job pool with memory-efficient operations and cleanup support.

    Provides:
    - Lazy loading of job pool
    - Memory usage tracking
    - Explicit cleanup mechanism
    - Pagination support for very large datasets
    """

    def __init__(self, data_loader: DataLoader, max_pool_size: int = 10000):
        """
        Initialize the global job pool manager.

        Args:
            data_loader: DataLoader instance for loading jobs
            max_pool_size: Maximum number of jobs to keep in memory
        """
        self.data_loader = data_loader
        self.max_pool_size = max_pool_size
        self._job_pool: Optional[List[Dict[str, Any]]] = None
        self._pool_stats: Dict[str, Any] = {}
        self._is_loaded = False

    def load(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load the global job pool from a dataset file.

        Args:
            file_path: Path to the dataset file

        Returns:
            List of job dictionaries
        """
        if self._is_loaded and self._job_pool is not None:
            logger.info(
                f"Reusing existing job pool with {len(self._job_pool)} jobs")
            return self._job_pool

        self._job_pool = self.data_loader.load_global_job_pool(
            file_path, max_jobs=self.max_pool_size
        )
        self._is_loaded = True

        # Track memory usage
        self._pool_stats = self.data_loader.estimate_job_pool_memory(
            self._job_pool)
        logger.info(f"Job pool loaded: {self._pool_stats['total_jobs']} jobs, "
                    f"~{self._pool_stats['memory_mb']:.1f} MB")

        return self._job_pool

    def get_pool(self) -> Optional[List[Dict[str, Any]]]:
        """Get the current job pool (None if not loaded)."""
        return self._job_pool

    def get_stats(self) -> Dict[str, Any]:
        """Get job pool statistics."""
        return {
            "is_loaded": self._is_loaded,
            "pool_size": len(self._job_pool) if self._job_pool else 0,
            **self._pool_stats
        }

    def clear(self) -> None:
        """
        Clear the job pool from memory.

        Call this after training to free memory.
        """
        if self._job_pool is not None:
            pool_size = len(self._job_pool)
            memory_freed = self._pool_stats.get('memory_mb', 0)

            self._job_pool = None
            self._pool_stats = {}
            self._is_loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            logger.info(
                f"Job pool cleared: freed {pool_size} jobs (~{memory_freed:.1f} MB)")

    def __del__(self):
        """Cleanup on deletion."""
        self.clear()
