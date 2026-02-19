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
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Numerical stability parameters
        self.eps = 1e-8
        self.max_exp = 50.0  # Prevent overflow in exp
        self.gradient_clip_value = 1.0

        # Sample-level weighting: quality tier + ontology coverage
        self.ontology_weight = getattr(config, 'ontology_weight', 0.3)
        self.ot_distance_scale = getattr(config, 'ot_distance_scale', 10.0)

        logger.info(f"Initialized ContrastiveLossEngine with temperature={self.temperature}, "
                    f"ontology_weight={self.ontology_weight}, device={self.device}")

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

        Applies ontology-aware positive weighting when scores are available
        in view_metadata (from precomputed ESCO enrichment).

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

            for i, negative in enumerate(triplet.negatives):
                negative_key = self._get_content_key(negative)
                if negative_key in embeddings:
                    negative_embs.append(embeddings[negative_key])

            if not negative_embs:
                logger.warning("No valid negative embeddings found")
                return None

            # Compute plain InfoNCE loss (no per-negative weighting)
            loss = self._infonce_loss(anchor_emb, positive_emb, negative_embs)

            if loss is None:
                return None

            # Apply sample-level ontology weight (label confidence + data quality)
            ont_weight = self._compute_ontology_weight(triplet.view_metadata)
            loss = loss * ont_weight

            return loss

        except Exception as e:
            logger.error(f"Error computing triplet loss: {e}")
            return None

    def _compute_ontology_weight(self, view_metadata: Dict[str, Any]) -> float:
        """
        Compute a sample-level weight based on precomputed ontology scores.

        Uses ontology_similarity and ot_distance from ESCO enrichment to
        upweight samples where the ontology confirms the label signal,
        and downweight samples with weak/missing ontology evidence.

        Args:
            view_metadata: Triplet metadata containing ontology scores

        Returns:
            float: Multiplicative weight for this sample's loss (0.5 to 1.5)
        """
        if self.ontology_weight == 0.0:
            return 1.0

        ont_sim = view_metadata.get('ontology_similarity')
        ot_dist = view_metadata.get('ot_distance')
        tier = view_metadata.get('quality_tier', 'F')

        # Base weight from quality tier
        tier_weights = {'A': 1.0, 'B': 0.9, 'C': 0.75, 'D': 0.6, 'F': 0.5}
        base = tier_weights.get(tier, 0.5)

        if ont_sim is None and ot_dist is None:
            return base

        # Ontology signal: blend similarity and normalized OT distance
        ont_signal = 0.0
        count = 0
        if ont_sim is not None:
            ont_signal += ont_sim  # 0-1, higher = more similar
            count += 1
        if ot_dist is not None:
            # Normalize OT distance to 0-1 (inverted: lower distance = higher signal)
            normalized_ot = max(0.0, 1.0 - ot_dist / self.ot_distance_scale)
            ont_signal += normalized_ot
            count += 1

        if count > 0:
            ont_signal /= count  # average of available signals, 0-1

        # Final weight: base ± ontology adjustment
        weight = base * (1.0 + self.ontology_weight * (2.0 * ont_signal - 1.0))

        # Clamp to reasonable range
        return max(0.5, min(1.5, weight))

    def _infonce_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                      negatives: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute standard InfoNCE contrastive loss.

        Negative difficulty is handled by the selection strategy (ontology-based
        bucketing with curriculum learning), not by per-negative weighting.

        Args:
            anchor: Anchor embedding (resume), L2-normalized
            positive: Positive embedding (matching job), L2-normalized
            negatives: List of negative embeddings (non-matching jobs), L2-normalized

        Returns:
            torch.Tensor: InfoNCE loss
        """
        # Ensure all tensors are on the same device
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negatives = [neg.to(self.device) for neg in negatives]

        # Compute similarities (dot product on normalized vectors = cosine similarity)
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        neg_sims = []
        for neg_emb in negatives:
            neg_sim = torch.sum(anchor * neg_emb, dim=-1) / self.temperature
            neg_sims.append(neg_sim)

        # Clamp similarities to prevent overflow
        pos_sim = torch.clamp(pos_sim, max=self.max_exp)
        neg_sims = [torch.clamp(sim, max=self.max_exp) for sim in neg_sims]

        # Standard InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg_i))))
        pos_exp = torch.exp(pos_sim)
        neg_exps = torch.stack([torch.exp(s) for s in neg_sims])
        denominator = pos_exp + torch.sum(neg_exps)

        # Numerical stability
        denominator = torch.clamp(denominator, min=self.eps, max=1e10)
        ratio = pos_exp / denominator
        ratio = torch.clamp(ratio, min=self.eps, max=1.0)
        loss = -torch.log(ratio)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN or infinite loss detected, returning zero loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

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

            for i, neg_key in enumerate(negative_view_keys):
                if neg_key in job_embeddings and i < len(triplet.career_distances):
                    negative_embs.append(job_embeddings[neg_key])

            if not negative_embs:
                return None

            return self._infonce_loss(
                anchor_emb, positive_emb, negative_embs
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
            'avg_ontology_weight': 0.0,
            'valid_triplets': 0,
            'failed_triplets': 0
        }

        total_pos_sim = 0.0
        total_neg_sim = 0.0
        total_ontology_weight = 0.0
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

                # Track ontology weight
                ontology_weight = self._compute_ontology_weight(triplet.view_metadata)
                total_ontology_weight += ontology_weight

                # Compute negative similarities
                for i, negative in enumerate(triplet.negatives):
                    negative_key = self._get_content_key(negative)
                    if negative_key in embeddings:
                        neg_emb = embeddings[negative_key]
                        neg_sim = torch.sum(
                            anchor_emb * neg_emb, dim=-1).item()
                        total_neg_sim += neg_sim
                        total_negatives += 1

                components['valid_triplets'] += 1

            except Exception as e:
                logger.warning(f"Error analyzing triplet: {e}")
                components['failed_triplets'] += 1

        # Compute averages
        if components['valid_triplets'] > 0:
            components['avg_positive_similarity'] = total_pos_sim / \
                components['valid_triplets']
            components['avg_ontology_weight'] = total_ontology_weight / \
                components['valid_triplets']

        if total_negatives > 0:
            components['avg_negative_similarity'] = total_neg_sim / \
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
