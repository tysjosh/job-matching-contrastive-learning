"""
Contrastive loss engine for pathway-aware contrastive learning.

This module implements the ContrastiveLossEngine that computes contrastive loss
with temperature scaling, pathway weighting, and support for multiple view combinations.
"""

import torch
import logging
from typing import List, Dict, Any, Optional

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .data_structures import ContrastiveTriplet, TrainingConfig

logger = logging.getLogger(__name__)


class ContrastiveLossEngine:
    """
    Computes pathway-weighted contrastive loss for resume-job matching.

    This engine implements contrastive loss with temperature scaling and pathway-aware
    weighting that gives higher importance to career-relevant distinctions. It supports
    multiple view combinations and includes numerical stability checks.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the contrastive loss engine.

        Args:
            config: Training configuration containing temperature, pathway_weight, etc.

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ContrastiveLossEngine. "
                "Install with: pip install torch"
            )

        self.config = config
        self.temperature = config.temperature
        self.pathway_weight = config.pathway_weight
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Numerical stability parameters
        self.eps = 1e-8
        self.max_exp = 50.0  # Prevent overflow in exp
        self.gradient_clip_value = 1.0
        
        # Curriculum learning parameters
        self.current_epoch = 0
        self.total_epochs = config.num_epochs

        logger.info(f"Initialized ContrastiveLossEngine with temperature={self.temperature}, "
                    f"pathway_weight={self.pathway_weight}, device={self.device}")

    def compute_loss(self, triplets: List[ContrastiveTriplet],
                     embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss for a batch of triplets.

        Args:
            triplets: List of ContrastiveTriplet objects
            embeddings: Dictionary mapping content to embeddings
                       Keys should be string representations of resume/job content

        Returns:
            torch.Tensor: Computed contrastive loss

        Raises:
            ValueError: If triplets is empty or embeddings are missing
        """
        if not triplets:
            raise ValueError("Cannot compute loss for empty triplets list")

        if not embeddings:
            raise ValueError("Embeddings dictionary cannot be empty")

        losses = []
        
        for triplet in triplets:
            try:
                triplet_loss = self._compute_triplet_loss(triplet, embeddings)
                if triplet_loss is not None:
                    losses.append(triplet_loss)
            except Exception as e:
                logger.warning(f"Failed to compute loss for triplet: {e}")
                continue

        if not losses:
            logger.warning("No valid triplets found for loss computation")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Average loss across valid triplets
        if len(losses) == 1:
            return losses[0]
        else:
            return torch.stack(losses).mean()

    def _compute_triplet_loss(self, triplet: ContrastiveTriplet,
                              embeddings: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute contrastive loss for a single triplet.

        Args:
            triplet: ContrastiveTriplet containing anchor, positive, and negatives
            embeddings: Dictionary mapping content to embeddings

        Returns:
            torch.Tensor: Loss for this triplet, or None if computation fails
        """
        try:
            # Get embeddings for anchor (resume), positive (job), and negatives
            anchor_key = self._get_content_key(triplet.anchor)
            positive_key = self._get_content_key(triplet.positive)

            if anchor_key not in embeddings or positive_key not in embeddings:
                logger.warning("Missing embeddings for anchor or positive")
                return None

            anchor_emb = embeddings[anchor_key]
            positive_emb = embeddings[positive_key]

            # Get negative embeddings
            negative_embs = []
            negative_distances = []

            for i, negative in enumerate(triplet.negatives):
                negative_key = self._get_content_key(negative)
                if negative_key in embeddings:
                    negative_embs.append(embeddings[negative_key])
                    negative_distances.append(triplet.career_distances[i])

            if not negative_embs:
                logger.warning("No valid negative embeddings found")
                return None

            # Compute pathway-weighted contrastive loss
            return self._pathway_weighted_loss(
                anchor_emb, positive_emb, negative_embs, negative_distances
            )

        except Exception as e:
            logger.error(f"Error computing triplet loss: {e}")
            return None

    def _pathway_weighted_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                               negatives: List[torch.Tensor],
                               career_distances: List[float]) -> torch.Tensor:
        """
        Compute pathway-weighted contrastive loss.

        This implements InfoNCE loss with pathway weighting that gives higher
        importance to negatives that are career-relevant (closer in career space).

        Args:
            anchor: Anchor embedding (resume)
            positive: Positive embedding (matching job)
            negatives: List of negative embeddings (non-matching jobs)
            career_distances: Career distances for pathway weighting

        Returns:
            torch.Tensor: Pathway-weighted contrastive loss
        """
        # Ensure all tensors are on the same device
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negatives = [neg.to(self.device) for neg in negatives]

        # FIXED: Model now returns normalized embeddings, no need to normalize here
        # This provides consistent scale and stable gradients across all batches
        # anchor, positive, and negatives are already L2-normalized from the model

        # Compute similarities (cosine similarity via dot product on normalized vectors)
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # Compute negative similarities with pathway weighting
        neg_sims = []
        pathway_weights = []

        for neg_emb, distance in zip(negatives, career_distances):
            neg_sim = torch.sum(anchor * neg_emb, dim=-1) / self.temperature
            neg_sims.append(neg_sim)

            # Higher career distance means less similar career path
            # Apply higher weight to closer career paths (harder negatives)
            pathway_weight = self._compute_pathway_weight(distance)
            pathway_weights.append(pathway_weight)

        # Clamp similarities to prevent overflow
        pos_sim = torch.clamp(pos_sim, max=self.max_exp)
        neg_sims = [torch.clamp(sim, max=self.max_exp) for sim in neg_sims]

        # Compute weighted negative log-likelihood
        pos_exp = torch.exp(pos_sim)

        # Apply pathway weights to negative exponentials with better numerical stability
        weighted_neg_exps = []
        for neg_sim, weight in zip(neg_sims, pathway_weights):
            # Clamp weight to prevent extreme values
            weight = torch.clamp(torch.tensor(weight), min=0.1, max=5.0)
            weighted_neg_exp = weight * torch.exp(neg_sim)
            weighted_neg_exps.append(weighted_neg_exp)

        # Sum all exponentials (positive + weighted negatives)
        denominator = pos_exp + torch.sum(torch.stack(weighted_neg_exps))

        # Add epsilon for numerical stability and clamp to prevent overflow
        denominator = torch.clamp(denominator, min=self.eps, max=1e10)

        # Compute negative log-likelihood with additional stability checks
        ratio = pos_exp / denominator
        ratio = torch.clamp(ratio, min=self.eps, max=1.0)  # Ensure valid probability
        loss = -torch.log(ratio)

        # Check for NaN or infinite values
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                "NaN or infinite loss detected, returning zero loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

    def _compute_pathway_weight(self, career_distance: float) -> float:
        """
        Compute pathway weight based on career distance with curriculum learning.
        
        Uses curriculum learning to gradually increase focus on hard negatives:
        - Early training (epoch 0): All negatives weighted equally (weight=1.0)
        - Late training (final epoch): Hard negatives weighted more (up to 1.0 + pathway_weight)
        
        This prevents early collapse while still learning career structure later.

        Args:
            career_distance: Career distance between positive and negative job

        Returns:
            float: Pathway weight (higher for closer career paths, increases over time)
        """
        # If pathway_weight is 0, disable weighting entirely
        if self.pathway_weight == 0.0:
            return 1.0
        
        # Compute curriculum progress (0.0 at start, 1.0 at end)
        epoch_ratio = self.current_epoch / max(1, self.total_epochs)
        
        # Normalize distance to [0, 1] range
        # 0 = adjacent (same career path), 1 = distant (different domains)
        normalized_distance = min(career_distance, 10.0) / 10.0
        
        # Curriculum strength: gradually increase from 0 to 1
        curriculum_strength = epoch_ratio
        
        # Base weight (always 1.0 for all negatives)
        base_weight = 1.0
        
        # Hard negative bonus: increases over training
        # - Adjacent careers (d=0): get full bonus
        # - Distant careers (d=10): get no bonus
        hard_negative_bonus = curriculum_strength * self.pathway_weight * (1.0 - normalized_distance)
        
        # Final weight: starts at 1.0 for all, gradually increases for hard negatives
        weight = base_weight + hard_negative_bonus
        
        return weight
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for curriculum learning.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch

    def compute_view_combinations_loss(self, triplets: List[ContrastiveTriplet],
                                       resume_embeddings: Dict[str, torch.Tensor],
                                       job_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss across multiple view combinations.

        This method handles cases where we have multiple views of resumes and jobs,
        computing loss across all valid combinations as specified in Requirement 4.2.

        Args:
            triplets: List of ContrastiveTriplet objects with view metadata
            resume_embeddings: Embeddings for different resume views
            job_embeddings: Embeddings for different job views

        Returns:
            torch.Tensor: Combined loss across all view combinations
        """
        if not triplets:
            raise ValueError("Cannot compute loss for empty triplets list")

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_combinations = 0

        for triplet in triplets:
            # Get view combinations from metadata
            view_combinations = triplet.view_metadata.get(
                'view_combinations', [])

            if not view_combinations:
                # Fallback to original embeddings
                all_embeddings = {**resume_embeddings, **job_embeddings}
                triplet_loss = self._compute_triplet_loss(
                    triplet, all_embeddings)
                if triplet_loss is not None:
                    total_loss = total_loss + triplet_loss
                    total_combinations += 1
            else:
                # Compute loss for each view combination
                for combo in view_combinations:
                    try:
                        combo_loss = self._compute_view_combination_loss(
                            triplet, combo, resume_embeddings, job_embeddings
                        )
                        if combo_loss is not None:
                            total_loss = total_loss + combo_loss
                            total_combinations += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute loss for view combination: {e}")
                        continue

        if total_combinations == 0:
            logger.warning(
                "No valid view combinations found for loss computation")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Average loss across all combinations
        avg_loss = total_loss / total_combinations

        return avg_loss

    def _compute_view_combination_loss(self, triplet: ContrastiveTriplet,
                                       view_combination: Dict[str, Any],
                                       resume_embeddings: Dict[str, torch.Tensor],
                                       job_embeddings: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute loss for a specific view combination.

        Args:
            triplet: ContrastiveTriplet object
            view_combination: Dictionary specifying which views to use
            resume_embeddings: Resume view embeddings
            job_embeddings: Job view embeddings

        Returns:
            torch.Tensor: Loss for this view combination, or None if computation fails
        """
        try:
            # Extract view keys from combination
            resume_view_key = view_combination.get('resume_view_key')
            positive_view_key = view_combination.get('positive_view_key')
            negative_view_keys = view_combination.get('negative_view_keys', [])

            # Get embeddings
            if resume_view_key not in resume_embeddings:
                return None
            if positive_view_key not in job_embeddings:
                return None

            anchor_emb = resume_embeddings[resume_view_key]
            positive_emb = job_embeddings[positive_view_key]

            # Get negative embeddings
            negative_embs = []
            negative_distances = []

            for i, neg_key in enumerate(negative_view_keys):
                if neg_key in job_embeddings and i < len(triplet.career_distances):
                    negative_embs.append(job_embeddings[neg_key])
                    negative_distances.append(triplet.career_distances[i])

            if not negative_embs:
                return None

            return self._pathway_weighted_loss(
                anchor_emb, positive_emb, negative_embs, negative_distances
            )

        except Exception as e:
            logger.error(f"Error computing view combination loss: {e}")
            return None

    def _get_content_key(self, content: Dict[str, Any]) -> str:
        """
        Generate a consistent key for content to look up embeddings.
        
        Uses the same key generation method as EmbeddingCache for consistency.

        Args:
            content: Resume or job content dictionary

        Returns:
            str: Key for embedding lookup
        """
        # Use the same key generation as EmbeddingCache for consistency
        import hashlib
        import json
        try:
            # Create a normalized representation for consistent hashing
            normalized_content = self._normalize_content_for_hashing(content)
            content_str = json.dumps(normalized_content, sort_keys=True, default=str)
            
            # Use SHA-256 for better collision resistance than hash()
            return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:16]
        except (TypeError, ValueError) as e:
            logger.warning(f"Error creating content key: {e}")
            # Fallback to string representation
            content_str = str(sorted(content.items()))
            return hashlib.md5(content_str.encode('utf-8')).hexdigest()[:16]
    
    def _normalize_content_for_hashing(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize content for consistent hashing by removing non-essential fields.
        
        Args:
            content: Original content dictionary
            
        Returns:
            Dict: Normalized content for hashing
        """
        # Create a copy and remove metadata that doesn't affect embeddings
        normalized = {}
        
        # Essential fields that affect text encoding
        essential_fields = {
            'experience', 'role', 'experience_level', 'skills', 'keywords',
            'title', 'jobtitle', 'description', 'jobdescription'
        }
        
        for key, value in content.items():
            if key in essential_fields:
                if isinstance(value, (dict, list)):
                    # Recursively normalize nested structures
                    normalized[key] = self._normalize_nested_structure(value)
                else:
                    normalized[key] = value
        
        return normalized
    
    def _normalize_nested_structure(self, obj: Any) -> Any:
        """Recursively normalize nested dictionaries and lists."""
        if isinstance(obj, dict):
            return {k: self._normalize_nested_structure(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_nested_structure(item) for item in obj]
        else:
            return obj

    def validate_gradients(self, loss: torch.Tensor, parameters: List[torch.Tensor] = None) -> bool:
        """
        Validate that gradients are well-behaved (no NaN or infinite values).

        Args:
            loss: Loss tensor to validate
            parameters: List of parameters to check gradients for (optional)

        Returns:
            bool: True if gradients are valid, False otherwise
        """
        if not loss.requires_grad:
            return True

        try:
            # For testing purposes, just check if the loss tensor itself is valid
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Invalid loss value detected")
                return False

            # If parameters are provided, check their gradients
            if parameters:
                # Compute gradients
                loss.backward(retain_graph=True)

                for param in parameters:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.warning(
                                "Invalid gradients detected in parameters")
                            return False

            return True

        except Exception as e:
            logger.error(f"Error validating gradients: {e}")
            return False

    def get_loss_components(self, triplets: List[ContrastiveTriplet],
                            embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Get detailed loss components for analysis and debugging.

        Args:
            triplets: List of ContrastiveTriplet objects
            embeddings: Dictionary mapping content to embeddings

        Returns:
            Dict[str, float]: Dictionary containing loss components and statistics
        """
        components = {
            'total_loss': 0.0,
            'avg_positive_similarity': 0.0,
            'avg_negative_similarity': 0.0,
            'avg_pathway_weight': 0.0,
            'valid_triplets': 0,
            'failed_triplets': 0
        }

        total_pos_sim = 0.0
        total_neg_sim = 0.0
        total_pathway_weight = 0.0
        total_negatives = 0

        for triplet in triplets:
            try:
                # Get embeddings
                anchor_key = self._get_content_key(triplet.anchor)
                positive_key = self._get_content_key(triplet.positive)

                if anchor_key not in embeddings or positive_key not in embeddings:
                    components['failed_triplets'] += 1
                    continue

                # Embeddings are already normalized by the model
                anchor_emb = embeddings[anchor_key]
                positive_emb = embeddings[positive_key]

                # Compute positive similarity
                pos_sim = torch.sum(anchor_emb * positive_emb, dim=-1).item()
                total_pos_sim += pos_sim

                # Compute negative similarities and pathway weights
                for i, negative in enumerate(triplet.negatives):
                    negative_key = self._get_content_key(negative)
                    if negative_key in embeddings:
                        # Embeddings are already normalized by the model
                        neg_emb = embeddings[negative_key]
                        neg_sim = torch.sum(
                            anchor_emb * neg_emb, dim=-1).item()
                        total_neg_sim += neg_sim

                        pathway_weight = self._compute_pathway_weight(
                            triplet.career_distances[i]
                        )
                        total_pathway_weight += pathway_weight
                        total_negatives += 1

                components['valid_triplets'] += 1

            except Exception as e:
                logger.warning(f"Error analyzing triplet: {e}")
                components['failed_triplets'] += 1

        # Compute averages
        if components['valid_triplets'] > 0:
            components['avg_positive_similarity'] = total_pos_sim / \
                components['valid_triplets']

        if total_negatives > 0:
            components['avg_negative_similarity'] = total_neg_sim / \
                total_negatives
            components['avg_pathway_weight'] = total_pathway_weight / \
                total_negatives

        # Compute total loss
        total_loss = self.compute_loss(triplets, embeddings)
        components['total_loss'] = total_loss.item(
        ) if total_loss.requires_grad else 0.0

        return components

    def clip_gradients(self, parameters: List[torch.Tensor]) -> None:
        """
        Apply gradient clipping to model parameters.

        Args:
            parameters: List of model parameters to clip gradients for
        """
        if parameters:
            torch.nn.utils.clip_grad_value_(
                parameters, self.gradient_clip_value)
