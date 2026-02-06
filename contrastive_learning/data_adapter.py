"""
Data adapter to convert the augmented dataset format to the expected contrastive learning format.
"""

import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional
from dataclasses import dataclass

from .data_structures import TrainingSample


logger = logging.getLogger(__name__)


@dataclass
class DataAdapterConfig:
    """Configuration for data adaptation process."""
    augmented_data_path: str
    output_path: str
    min_job_description_length: int = 50
    min_resume_experience_length: int = 20
    max_samples_per_original: int = 10  # Maximum views per original record
    include_original_only: bool = False  # Whether to include only original views
    include_augmented_only: bool = False  # Whether to include only augmented views
    balance_labels: bool = False  # Whether to balance positive/negative samples


class DataAdapter:
    """
    Adapter to convert the augmented dataset format to contrastive learning format.

    Converts augmented_combined_data.jsonl into the expected TrainingSample format
    for contrastive learning training.
    """

    def __init__(self, config: DataAdapterConfig):
        """
        Initialize DataAdapter with configuration.

        Args:
            config: Data adaptation configuration
        """
        self.config = config
        self.augmented_data = None

    def load_data(self) -> None:
        """Load augmented data file into memory for processing."""
        logger.info("Loading augmented data file...")

        # Load augmented training data
        logger.info(
            f"Loading augmented data from {self.config.augmented_data_path}")
        self.augmented_data = []

        with open(self.config.augmented_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        self.augmented_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue

        logger.info(f"Loaded {len(self.augmented_data)} augmented samples")

        # Log view type distribution
        view_types = {}
        for sample in self.augmented_data:
            view_type = sample.get('view_type', 'unknown')
            view_types[view_type] = view_types.get(view_type, 0) + 1

        logger.info("View type distribution:")
        for view_type, count in view_types.items():
            logger.info(f"  {view_type}: {count}")

        logger.info("Data loading completed successfully")

    def convert_to_training_samples(self) -> Iterator[TrainingSample]:
        """
        Convert the loaded augmented data to TrainingSample format.

        Yields:
            TrainingSample: Converted training samples
        """
        if not self.augmented_data:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Converting augmented data to training samples...")

        successful_conversions = 0
        failed_conversions = 0
        original_counts = {}

        # Filter data based on configuration
        filtered_data = self._filter_data()
        logger.info(
            f"Filtered to {len(filtered_data)} samples based on configuration")

        for idx, augmented_sample in enumerate(filtered_data):
            try:
                # Fix: use correct field name from augmented data
                original_id = augmented_sample.get('original_record_id', augmented_sample.get('original_id'))

                # Track samples per original record
                if original_id in original_counts:
                    if original_counts[original_id] >= self.config.max_samples_per_original:
                        continue
                    original_counts[original_id] += 1
                else:
                    original_counts[original_id] = 1

                # Convert to TrainingSample format
                training_sample = self._create_training_sample(
                    augmented_sample, idx)

                if training_sample and self._validate_training_sample(training_sample):
                    successful_conversions += 1
                    yield training_sample
                else:
                    failed_conversions += 1

            except Exception as e:
                logger.warning(f"Error processing augmented sample {idx}: {e}")
                failed_conversions += 1
                continue

        logger.info(
            f"Conversion completed: {successful_conversions} successful, {failed_conversions} failed")
        logger.info(
            f"Processed samples from {len(original_counts)} original records")

    def _filter_data(self) -> List[Dict[str, Any]]:
        """
        Filter augmented data based on configuration settings.

        Returns:
            List of filtered augmented samples
        """
        filtered_data = []

        for sample in self.augmented_data:
            view_type = sample.get('view_type', '')

            # Apply filtering based on configuration
            if self.config.include_original_only and view_type != 'original':
                continue

            if self.config.include_augmented_only and view_type == 'original':
                continue

            filtered_data.append(sample)

        return filtered_data

    def _create_training_sample(
        self,
        augmented_sample: Dict[str, Any],
        sample_idx: int
    ) -> Optional[TrainingSample]:
        """
        Create a TrainingSample from augmented sample data.

        Args:
            augmented_sample: Augmented sample data
            sample_idx: Sample index for ID generation

        Returns:
            TrainingSample if successful, None if conversion fails
        """
        try:
            # Create structured resume data with proper format
            resume = {
                'role': augmented_sample.get('resume', {}).get('role', ''),
                'experience': [{
                    'description': augmented_sample.get('resume', {}).get('experience', ''),
                    'responsibilities': {
                        'action_verbs': [],
                        'technical_terms': []
                    }
                }] if augmented_sample.get('resume', {}).get('experience') else [],
                'experience_level': augmented_sample.get('resume', {}).get('experience_level', ''),
                'skills': augmented_sample.get('resume', {}).get('skills', []),
                'keywords': augmented_sample.get('resume', {}).get('keywords', []),
            }
            job = augmented_sample.get('job', {})

            # Convert label to expected format
            label = augmented_sample.get('label', 0)
            label = 'positive' if label == 1 else 'negative'

            # Generate unique sample ID
            original_record_id = augmented_sample.get('original_record_id', 'unknown')
            augmentation_type = augmented_sample.get('augmentation_type', 'unknown')
            
            # Create a unique identifier based on original_record_id and sample index
            # This ensures each sample has a unique ID even when job_applicant_id equals original_record_id
            sample_id = f"sample_{original_record_id}_{sample_idx}"

            # Create comprehensive metadata using correct field names
            # FIX: Extract job_applicant_id from metadata if not at top level
            job_applicant_id = (augmented_sample.get('job_applicant_id') or 
                               augmented_sample.get('metadata', {}).get('job_applicant_id'))
            
            metadata = {
                'original_record_id': augmented_sample.get('original_record_id'),
                'job_applicant_id': job_applicant_id,
                'augmentation_type': augmented_sample.get('augmentation_type'),
                'label': augmented_sample.get('label'),
                'original_label': label
            }

            return TrainingSample(
                resume=resume,
                job=job,
                label=label,
                sample_id=sample_id,
                metadata=metadata
            )

        except Exception as e:
            logger.warning(f"Error creating training sample {sample_idx}: {e}")
            return None

    def _validate_training_sample(self, sample: TrainingSample) -> bool:
        """
        Validate that a training sample meets quality criteria.

        Args:
            sample: Training sample to validate

        Returns:
            bool: True if sample is valid
        """
        try:
            # Check job description length
            job_desc = sample.job.get('description', {})
            if isinstance(job_desc, dict):
                job_desc_text = job_desc.get('original', '')
            else:
                job_desc_text = str(job_desc)

            if len(job_desc_text) < self.config.min_job_description_length:
                return False

            # Check resume experience length
            resume_exp = sample.resume.get('experience', '')
            if len(str(resume_exp)) < self.config.min_resume_experience_length:
                return False

            # Check for essential fields
            if not sample.job.get('title', '').strip():
                return False

            # Check if resume has meaningful content
            resume_skills = sample.resume.get('skills', [])
            if not resume_skills and not sample.resume.get('experience', ''):
                return False

            return True

        except Exception as e:
            logger.debug(
                f"Validation error for sample {sample.sample_id}: {e}")
            return False

    def convert_and_save(self) -> None:
        """
        Convert data and save to output file.
        """
        logger.info(
            f"Starting data conversion and saving to {self.config.output_path}")

        # Ensure output directory exists
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_samples = 0
        positive_samples = 0
        negative_samples = 0

        samples_to_write = []

        # First pass: collect all samples
        for sample in self.convert_to_training_samples():
            samples_to_write.append(sample)
            if sample.label == 'positive':
                positive_samples += 1
            else:
                negative_samples += 1

        # Balance labels if configured
        if self.config.balance_labels:
            samples_to_write = self._balance_labels(samples_to_write)
            logger.info(f"Balanced dataset: {len(samples_to_write)} samples")

        # Write samples to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples_to_write:
                sample_dict = sample.to_dict()
                
                # FIX: Add job_applicant_id at top level for BatchProcessor
                if 'job_applicant_id' in sample.metadata:
                    sample_dict['job_applicant_id'] = sample.metadata['job_applicant_id']
                
                f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')
                total_samples += 1

                if total_samples % 1000 == 0:
                    logger.info(f"Processed {total_samples} samples...")

        logger.info(
            f"Data conversion completed. Saved {total_samples} samples to {self.config.output_path}")
        logger.info(f"Final distribution - Positive: {sum(1 for s in samples_to_write if s.label == 'positive')}, "
                    f"Negative: {sum(1 for s in samples_to_write if s.label == 'negative')}")

    def _balance_labels(self, samples: List[TrainingSample]) -> List[TrainingSample]:
        """
        Balance positive and negative samples.

        Args:
            samples: List of training samples

        Returns:
            List of balanced samples
        """
        positive_samples = [s for s in samples if s.label == 'positive']
        negative_samples = [s for s in samples if s.label == 'negative']

        logger.info(
            f"Before balancing - Positive: {len(positive_samples)}, Negative: {len(negative_samples)}")

        # Use the smaller count as the target
        min_count = min(len(positive_samples), len(negative_samples))

        if min_count == 0:
            logger.warning(
                "Cannot balance labels - one class has zero samples")
            return samples

        # Randomly sample to balance
        import random
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)

        balanced_samples = positive_samples[:min_count] + \
            negative_samples[:min_count]
        random.shuffle(balanced_samples)

        logger.info(
            f"After balancing - Total: {len(balanced_samples)} samples ({min_count} each class)")

        return balanced_samples

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.

        Returns:
            Dict containing data statistics
        """
        if not self.augmented_data:
            return {"error": "Data not loaded"}

        # Analyze augmented data
        positive_samples = sum(
            1 for sample in self.augmented_data if sample.get('label') == 1)
        negative_samples = len(self.augmented_data) - positive_samples

        # Count view types
        view_type_counts = {}
        augmentation_field_counts = {}

        for sample in self.augmented_data:
            view_type = sample.get('view_type', 'unknown')
            view_type_counts[view_type] = view_type_counts.get(
                view_type, 0) + 1

            aug_field = sample.get('augmentation_field', 'none')
            augmentation_field_counts[aug_field] = augmentation_field_counts.get(
                aug_field, 0) + 1

        # Count unique original records using correct field name
        unique_originals = len(set(sample.get('original_record_id')
                               for sample in self.augmented_data))

        return {
            'total_samples': len(self.augmented_data),
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'positive_ratio': positive_samples / len(self.augmented_data) if self.augmented_data else 0,
            'unique_original_records': unique_originals,
            'view_type_distribution': view_type_counts,
            'augmentation_field_distribution': augmentation_field_counts,
            'avg_views_per_original': len(self.augmented_data) / unique_originals if unique_originals > 0 else 0
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert augmented data to training format')
    parser.add_argument('--input-file', required=True,
                        help='Path to augmented data file')
    parser.add_argument('--output-file', required=True,
                        help='Path to output training data file')
    parser.add_argument('--balance-labels', action='store_true',
                        help='Balance positive/negative samples')

    args = parser.parse_args()

    # Configure for augmented data
    config = DataAdapterConfig(
        augmented_data_path=args.input_file,
        output_path=args.output_file,
        balance_labels=args.balance_labels
    )

    adapter = DataAdapter(config)
    adapter.load_data()
    adapter.convert_and_save()

    # Print statistics
    stats = adapter.get_data_statistics()
    print(f"Conversion completed. Statistics: {stats}")
