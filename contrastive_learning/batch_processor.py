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

        # Initialize OntologySkillMatcher for skill-level negative selection
        self.skill_matcher = None
        use_ontology = getattr(config, 'ontology_weight', 0.0) > 0.0
        if use_ontology and esco_graph_path:
            try:
                from .ontology_skill_matcher import OntologySkillMatcher
                # Use the full ESCO KG for skill matching (not the career graph)
                # Fall back to esco_graph_path if no separate KG path configured
                kg_path = getattr(config, 'esco_kg_path', None) or esco_graph_path
                self.skill_matcher = OntologySkillMatcher(kg_path)
                logger.info("OntologySkillMatcher enabled for skill-level negative selection")
            except Exception as e:
                logger.warning(f"Failed to initialize OntologySkillMatcher: {e}")

        # Set random seed for reproducible negative sampling
        random.seed(42)

        # Curriculum learning state for ontology negative selection
        self.current_epoch = 0
        self.total_epochs = config.num_epochs

        logger.info(f"BatchProcessor initialized with pathway_negatives={'enabled' if self.career_graph else 'disabled'}, "
                    f"skill_matcher={'enabled' if self.skill_matcher else 'disabled'}, "
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

        # Create view metadata including ontology scores for loss weighting
        view_metadata = {
            'anchor_id': anchor_sample.sample_id,
            'positive_job_applicant_id': anchor_sample.metadata.get('job_applicant_id', 'unknown'),
            'negative_count': len(negatives),
            'sampling_strategy': sampling_strategy,
            'ontology_similarity': anchor_sample.metadata.get('ontology_similarity'),
            'ot_distance': anchor_sample.metadata.get('ot_distance'),
            'quality_tier': anchor_sample.metadata.get('quality_tier'),
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
            selected_negatives = random.sample(
                candidate_negatives, min(max_negatives, len(candidate_negatives)))
            # Return zeros for career distances since pathway analysis is disabled
            career_distances = [0.0] * len(selected_negatives)

            logger.debug(
                f"Selected {len(selected_negatives)} random negatives (pathway negatives disabled)")
            return selected_negatives, career_distances

        # ── Skill-level ontology selection (preferred when available) ──
        if self.skill_matcher:
            try:
                resume_uris = anchor_sample.resume.get('skill_uris', [])
                if resume_uris:
                    return self._select_ontology_negatives(
                        resume_uris, candidate_negatives, max_negatives)
            except Exception as e:
                logger.warning(f"Ontology negative selection failed: {e}")

        # ── Random fallback (no skill URIs or ontology selection failed) ──
        # Samples without skill URIs get random negatives. The ontology weight
        # in the loss engine already downweights these samples (tier C → 0.75x).
        selected_negatives = random.sample(
            candidate_negatives, min(max_negatives, len(candidate_negatives)))
        career_distances = [0.0] * len(selected_negatives)

        logger.debug(
            f"Selected {len(selected_negatives)} random negatives (no skill URIs available)")
        return selected_negatives, career_distances

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
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for curriculum learning in negative selection."""
        self.current_epoch = epoch

    def _select_ontology_negatives(
        self,
        resume_skill_uris: List[str],
        candidate_negatives: List[Dict[str, Any]],
        max_negatives: int,
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """
        Select negatives based on skill-level ontology distance to the resume.

        Uses curriculum learning to shift hard/medium/easy ratios over epochs:
        - Early training: mostly easy negatives (model learns basic distinctions)
        - Late training: mostly hard negatives (model learns fine-grained distinctions)

        Args:
            resume_skill_uris: Skill URIs from the anchor resume
            candidate_negatives: Candidate negative jobs
            max_negatives: Number of negatives to select

        Returns:
            Tuple of (selected_negatives, ontology_distances)
        """
        # Compute ontology distance for each candidate
        scored = []
        for job in candidate_negatives:
            job_uris = job.get('skill_uris', [])
            if job_uris and resume_skill_uris:
                sim = self.skill_matcher.ontology_set_similarity(resume_skill_uris, job_uris)
                distance = 1.0 - sim  # 0 = identical skills, 1 = no overlap
            else:
                distance = 0.5  # neutral when no URIs
            scored.append((job, distance))

        # Bucket by ontology distance
        hard = [(j, d) for j, d in scored if d <= 0.3]       # very similar skills
        medium = [(j, d) for j, d in scored if 0.3 < d <= 0.6]
        easy = [(j, d) for j, d in scored if d > 0.6]        # very different skills

        # Curriculum learning: shift ratios from easy-heavy to hard-heavy
        # epoch_ratio: 0.0 at start → 1.0 at end
        epoch_ratio = self.current_epoch / max(1, self.total_epochs)

        # Early: 20% hard, 30% medium, 50% easy
        # Late:  60% hard, 30% medium, 10% easy
        hard_ratio = 0.2 + 0.4 * epoch_ratio    # 0.2 → 0.6
        easy_ratio = 0.5 - 0.4 * epoch_ratio    # 0.5 → 0.1
        medium_ratio = 1.0 - hard_ratio - easy_ratio  # stays ~0.3

        hard_count = int(max_negatives * hard_ratio)
        medium_count = int(max_negatives * medium_ratio)
        easy_count = max_negatives - hard_count - medium_count

        selected = []
        if hard and hard_count > 0:
            selected.extend(random.sample(hard, min(hard_count, len(hard))))
        if medium and medium_count > 0:
            selected.extend(random.sample(medium, min(medium_count, len(medium))))
        if easy and easy_count > 0:
            selected.extend(random.sample(easy, min(easy_count, len(easy))))

        # Fill remaining from any bucket
        if len(selected) < max_negatives:
            used = set(id(j) for j, _ in selected)
            remaining = [(j, d) for j, d in scored if id(j) not in used]
            random.shuffle(remaining)
            selected.extend(remaining[:max_negatives - len(selected)])

        selected = selected[:max_negatives]

        # Convert ontology distance to 0-10 scale for career_distances field
        negatives = [j for j, _ in selected]
        distances = [d * 10.0 for _, d in selected]

        logger.debug(
            f"Ontology negatives (epoch {self.current_epoch}): "
            f"ratios=hard:{hard_ratio:.0%}/med:{medium_ratio:.0%}/easy:{easy_ratio:.0%}, "
            f"selected=hard:{min(hard_count, len(hard))}/med:{min(medium_count, len(medium))}/easy:{min(easy_count, len(easy))}"
        )

        return negatives, distances


