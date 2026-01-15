"""
BatchProcessor for contrastive triplet generation.

This module implements the BatchProcessor class that converts batches of training samples
into contrastive triplets for training. It supports configurable negative sampling ratios
and in-batch negative sampling for computational efficiency.
"""

import random
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging

from .data_structures import TrainingSample, ContrastiveTriplet, TrainingConfig

if TYPE_CHECKING:
    from .career_graph import CareerGraph

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes batches of training samples to create contrastive triplets.

    The BatchProcessor converts each positive training sample into a contrastive triplet
    where the resume serves as the anchor, the matched job as the positive, and other
    jobs from the same batch serve as negatives. This in-batch negative sampling
    strategy provides computational efficiency while maintaining training effectiveness.
    """

    def __init__(self, config: TrainingConfig, career_graph: Optional['CareerGraph'] = None,
                 esco_graph_path: Optional[str] = None):
        """
        Initialize the BatchProcessor with training configuration.

        Args:
            config: TrainingConfig containing negative sampling parameters
            career_graph: Optional CareerGraph for pathway-aware negative sampling
            esco_graph_path: Path to ESCO graph file (required if career_graph is None and use_pathway_negatives=True)
        """
        self.config = config
        self.negative_sampling_ratio = config.negative_sampling_ratio
        self.use_pathway_negatives = config.use_pathway_negatives
        self.max_negatives_per_anchor = config.max_negatives_per_anchor

        if not self.use_pathway_negatives and self.config.pathway_weight != 0:
            logger.info("Pathway negatives disabled; setting pathway_weight to 0.0.")
            self.config.pathway_weight = 0.0

        # Initialize or create CareerGraph with ESCO graph
        if career_graph is not None:
            self.career_graph = career_graph
        elif self.use_pathway_negatives:
            if not esco_graph_path:
                raise ValueError(
                    "esco_graph_path is required when use_pathway_negatives=True and no CareerGraph is provided")
            # Create CareerGraph with ESCO graph path
            self.career_graph = self.create_career_graph(
                config, esco_graph_path)
        else:
            self.career_graph = None

        # Set random seed for reproducible negative sampling
        random.seed(42)

        logger.info(f"BatchProcessor initialized with pathway_negatives={'enabled' if self.career_graph else 'disabled'}, "
                    f"hard_distance_threshold={config.hard_negative_max_distance}, "
                    f"medium_distance_threshold={config.medium_negative_max_distance}")

    def process_batch(self, batch: List[TrainingSample], global_job_pool: Optional[List[Dict[str, Any]]] = None) -> List[ContrastiveTriplet]:
        """
        Process a batch of training samples to create contrastive triplets.

        Args:
            batch: List of TrainingSample objects
            global_job_pool: Optional global pool of jobs for negative sampling

        Returns:
            List of ContrastiveTriplet objects

        Raises:
            ValueError: If batch is empty or contains insufficient positive samples
        """
        if not batch:
            raise ValueError("Batch cannot be empty")

        # Filter positive samples (these will become anchors)
        positive_samples = [
            sample for sample in batch if sample.label == 'positive']

        if not positive_samples:
            raise ValueError("Batch must contain at least one positive sample")

        logger.debug(
            f"Processing batch with {len(batch)} samples, {len(positive_samples)} positive")

        triplets = []

        # Create triplets for each positive sample
        for anchor_sample in positive_samples:
            try:
                triplet = self._create_triplet(anchor_sample, batch, global_job_pool)
                triplets.append(triplet)
            except Exception as e:
                logger.warning(
                    f"Failed to create triplet for sample {anchor_sample.sample_id}: {e}")
                continue

        logger.debug(f"Created {len(triplets)} triplets from batch")
        return triplets

    def _create_triplet(self, anchor_sample: TrainingSample, batch: List[TrainingSample], global_job_pool: Optional[List[Dict[str, Any]]] = None) -> ContrastiveTriplet:
        """
        Create a contrastive triplet for a single anchor sample.

        Args:
            anchor_sample: The positive sample to use as anchor
            batch: Full batch of samples for negative selection
            global_job_pool: Optional global pool of jobs for negative sampling

        Returns:
            ContrastiveTriplet with anchor, positive, and negatives
        """
        # Use resume as anchor and matched job as positive
        anchor = anchor_sample.resume
        positive = anchor_sample.job

        # Select negative jobs from the batch or global pool
        negatives, career_distances = self._select_negatives(
            anchor_sample, batch, global_job_pool)

        # Determine sampling strategy for metadata
        sampling_strategy = 'global' if global_job_pool else 'in_batch'

        # Create view metadata (placeholder for future view augmentation integration)
        view_metadata = {
            'anchor_id': anchor_sample.sample_id,
            'positive_job_applicant_id': anchor_sample.metadata.get('job_applicant_id', 'unknown'),
            'negative_count': len(negatives),
            'sampling_strategy': sampling_strategy
        }

        return ContrastiveTriplet(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            career_distances=career_distances,
            view_metadata=view_metadata
        )

    def _select_negatives(self, anchor_sample: TrainingSample, batch: List[TrainingSample], global_job_pool: Optional[List[Dict[str, Any]]] = None) -> tuple[List[Dict[str, Any]], List[float]]:
        """
        Select negative jobs using either global pool or batch-based sampling.

        This method supports both global and in-batch negative sampling:
        - If global_job_pool is provided: Uses global negative sampling for consistent training
        - If global_job_pool is None: Falls back to in-batch sampling (original behavior)
        
        Also respects the use_pathway_negatives configuration setting:
        - If True: Uses CareerGraph for intelligent career-aware negative selection
        - If False: Uses simple random sampling from available candidates

        Args:
            anchor_sample: The anchor sample (positive)
            batch: Full batch of samples (used for fallback)
            global_job_pool: Optional global pool of jobs for negative sampling

        Returns:
            Tuple of (negative_jobs, career_distances)
        """
        # Choose negative candidate source: global pool or batch
        if global_job_pool:
            # Use global negative sampling for consistent training
            candidate_negatives = []
            anchor_job_id = anchor_sample.job.get('job_id', anchor_sample.job.get('title', ''))
            
            for job in global_job_pool:
                job_id = job.get('job_id', job.get('title', ''))
                # Skip the same job as the positive
                if job_id != anchor_job_id:
                    candidate_negatives.append(job)
            
            logger.debug(f"Using global negative sampling with {len(candidate_negatives)} candidates")
        else:
            # Fall back to in-batch negative sampling (original behavior)
            candidate_negatives = []
            anchor_job_applicant_id = anchor_sample.metadata.get('job_applicant_id')

            for sample in batch:
                job = sample.job
                sample_job_applicant_id = sample.metadata.get('job_applicant_id')

                # Skip only samples with same job_applicant_id that are positive labels
                if sample_job_applicant_id == anchor_job_applicant_id and sample.label == 'positive':
                    continue

                candidate_negatives.append(job)
            
            logger.debug(f"Using in-batch negative sampling with {len(candidate_negatives)} candidates")

        if not candidate_negatives:
            # If no other jobs available, create a dummy negative
            logger.warning(
                f"No candidate negatives found for sample {anchor_sample.sample_id}")
            dummy_negative = {
                'job_id': 'dummy_negative',
                'title': 'No Match Available',
                'description': 'Placeholder negative sample',
                'level': 'unknown'
            }
            return [dummy_negative], [self.config.medium_negative_max_distance]

        # Limit number of negatives to prevent memory issues
        max_negatives = min(len(candidate_negatives), self.max_negatives_per_anchor)

        # Check if pathway-aware negative selection is enabled
        if not self.use_pathway_negatives:
            # Use simple random negative sampling
            import random
            selected_negatives = random.sample(
                candidate_negatives, min(max_negatives, len(candidate_negatives)))
            # Return zeros for career distances since pathway analysis is disabled
            career_distances = [0.0] * len(selected_negatives)

            logger.debug(
                f"Selected {len(selected_negatives)} random negatives (pathway negatives disabled)")
            return selected_negatives, career_distances

        # Use CareerGraph for pathway-aware selection
        if not self.career_graph:
            raise ValueError("CareerGraph is required for pathway-aware negative selection. "
                             "Provide either a CareerGraph instance or esco_graph_path parameter, "
                             "or set use_pathway_negatives=False to disable this feature.")

        try:
            logger.debug(
                f"Using CareerGraph for pathway-aware negative sampling from {len(candidate_negatives)} candidates")

            # Use CareerGraph's intelligent pathway-aware selection
            selected_negatives = self.career_graph.select_pathway_negatives(
                anchor_sample.job, candidate_negatives, max_negatives
            )

            # Compute career distances for the selected negatives
            career_distances = []
            for negative_job in selected_negatives:
                distance = self.career_graph.compute_career_distance(
                    anchor_sample.job, negative_job
                )
                career_distances.append(distance)

            # Log detailed statistics about the career-aware selection
            if career_distances:
                avg_distance = sum(career_distances) / len(career_distances)
                min_distance = min(career_distances)
                max_distance = max(career_distances)

                # Count negatives by category using updated thresholds
                hard_count = sum(1 for d in career_distances if d <=
                                 self.config.hard_negative_max_distance)
                medium_count = sum(1 for d in career_distances
                                   if self.config.hard_negative_max_distance < d <= self.config.medium_negative_max_distance)
                easy_count = sum(1 for d in career_distances if d >
                                 self.config.medium_negative_max_distance)

                logger.info(
                    f"CareerGraph selected {len(selected_negatives)} negatives: "
                    f"avg_distance={avg_distance:.2f}, range=[{min_distance:.1f}, {max_distance:.1f}], "
                    f"hard={hard_count}, medium={medium_count}, easy={easy_count}"
                )

            return selected_negatives, career_distances

        except Exception as e:
            logger.error(f"CareerGraph pathway-aware sampling failed: {e}")
            # Check if it's due to missing URIs and provide helpful error message
            if "Missing URIs" in str(e):
                raise RuntimeError(
                    f"Failed to select pathway-aware negatives: Jobs must have ESCO URIs for career distance calculation. "
                    f"Ensure job data includes 'uri', 'job_uri', or 'occupation_uri' fields.") from e
            else:
                raise RuntimeError(
                    f"Failed to select pathway-aware negatives: {e}") from e

    @staticmethod
    def create_career_graph(config: TrainingConfig, esco_graph_path: str) -> 'CareerGraph':
        """
        Create a CareerGraph instance using configuration parameters and ESCO graph.

        This is a shared factory method that can be used by any component needing
        a CareerGraph instance with consistent configuration.

        Args:
            config: TrainingConfig with distance thresholds
            esco_graph_path: Path to the ESCO graph file

        Returns:
            CareerGraph instance configured with training parameters
        """
        from .career_graph import CareerGraph

        return CareerGraph(
            esco_graph_path=esco_graph_path,
            hard_negative_max_distance=config.hard_negative_max_distance,
            medium_negative_max_distance=config.medium_negative_max_distance,
            use_distance_cache=True
        )

    def get_batch_statistics(self, batch: List[TrainingSample]) -> Dict[str, Any]:
        """
        Get statistics about a batch for monitoring and debugging.

        Args:
            batch: List of TrainingSample objects

        Returns:
            Dictionary with batch statistics
        """
        if not batch:
            return {'total_samples': 0, 'positive_samples': 0, 'negative_samples': 0}

        positive_count = sum(
            1 for sample in batch if sample.label == 'positive')
        negative_count = len(batch) - positive_count

        # Get unique job count for negative sampling potential
        unique_jobs = set()
        for sample in batch:
            job_id = sample.job.get(
                'job_id', sample.job.get('title', 'unknown'))
            unique_jobs.add(job_id)

        return {
            'total_samples': len(batch),
            'positive_samples': positive_count,
            'negative_samples': negative_count,
            'unique_jobs': len(unique_jobs),
            'avg_negatives_per_positive': max(0, len(unique_jobs) - 1) if positive_count > 0 else 0
        }
