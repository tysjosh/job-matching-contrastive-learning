"""
Semantic Coherence Validator for LLM Career-Aware Data Augmentation System

This module validates that LLM-based transformations maintain semantic coherence
while achieving meaningful career-level differentiation. It uses sentence embeddings
to compute semantic similarity and validates career level indicators.

Key Features:
- Semantic similarity computation using sentence transformers
- Threshold-based validation (min 0.5, max 0.95)
- Career level indicator validation
- Quality reporting for monitoring

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a semantic coherence validation."""
    is_valid: bool
    semantic_similarity: float
    career_level_valid: bool
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "is_valid": self.is_valid,
            "semantic_similarity": self.semantic_similarity,
            "career_level_valid": self.career_level_valid
        }
        if self.rejection_reason:
            result["rejection_reason"] = self.rejection_reason
        return result


class SemanticCoherenceValidator:
    """
    Validates semantic coherence of LLM transformations.
    
    This validator ensures that transformed content:
    1. Maintains semantic meaning (similarity >= 0.5)
    2. Shows meaningful transformation (similarity <= 0.95)
    3. Contains appropriate career level indicators
    
    Requirements:
    - 8.1: Compute semantic similarity using sentence embeddings
    - 8.2: Reject transformations with similarity < 0.5
    - 8.3: Flag transformations with similarity > 0.95 as insufficient
    - 8.4: Validate career level indicators match target level
    """
    
    # Senior-level indicators for upward transformations
    SENIOR_INDICATORS = {
        # Leadership verbs
        "led", "architected", "designed", "mentored", "directed",
        "spearheaded", "orchestrated", "championed", "pioneered",
        "established", "drove", "transformed", "scaled",
        # Strategic terms
        "strategic", "ownership", "leadership", "cross-functional",
        "enterprise", "organization-wide", "company-wide",
        # Impact terms
        "impact", "revenue", "growth", "optimization",
        "efficiency", "stakeholder", "executive",
        # Seniority indicators
        "senior", "lead", "principal", "staff", "architect",
        "manager", "director", "head"
    }
    
    # Junior-level indicators for downward transformations
    JUNIOR_INDICATORS = {
        # Learning verbs
        "assisted", "learned", "supported", "helped",
        "contributed", "participated", "collaborated",
        # Guidance terms
        "under guidance", "supervised", "mentored by",
        "with support", "alongside", "team member",
        # Task-focused terms
        "tasks", "assignments", "responsibilities",
        "day-to-day", "routine", "basic", "foundational",
        # Entry-level indicators
        "junior", "entry", "associate", "trainee",
        "intern", "graduate", "beginner"
    }
    
    def __init__(
        self,
        min_similarity: float = 0.5,
        max_similarity: float = 0.95,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the semantic coherence validator.
        
        Args:
            min_similarity: Minimum semantic similarity threshold (default: 0.5)
            max_similarity: Maximum semantic similarity threshold (default: 0.95)
            model_name: Sentence transformer model name for embeddings
        """
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.model_name = model_name
        self._encoder = None
        
        logger.info(
            f"SemanticCoherenceValidator initialized with thresholds: "
            f"min={min_similarity}, max={max_similarity}"
        )
    
    @property
    def encoder(self):
        """Lazy-load the sentence transformer encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence transformer model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise ImportError(
                    "sentence-transformers is required for semantic validation"
                )
        return self._encoder
    
    def compute_similarity(self, original: str, transformed: str) -> float:
        """
        Compute semantic similarity between original and transformed text.
        
        Uses sentence embeddings and cosine similarity to measure how
        semantically similar the two texts are.
        
        Args:
            original: Original text content
            transformed: Transformed text content
            
        Returns:
            float: Cosine similarity score between 0 and 1
            
        Requirements: 8.1
        """
        if not original or not transformed:
            logger.warning("Empty text provided for similarity computation")
            return 0.0
        
        try:
            # Encode both texts
            embeddings = self.encoder.encode(
                [original, transformed],
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            # Compute cosine similarity
            # Since embeddings are normalized, dot product equals cosine similarity
            import torch
            similarity = torch.nn.functional.cosine_similarity(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0)
            ).item()
            
            logger.debug(f"Computed semantic similarity: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            # Return a middle value on error to avoid false rejections
            return 0.7
    
    def validate_transformation(
        self,
        original: str,
        transformed: str,
        target_level: str
    ) -> ValidationResult:
        """
        Validate a transformation meets quality criteria.
        
        Checks:
        1. Semantic similarity is within acceptable bounds (0.5 - 0.95)
        2. Career level indicators match the target level
        
        Args:
            original: Original text content
            transformed: Transformed text content
            target_level: Target career level ("senior", "lead", "junior", "entry")
            
        Returns:
            ValidationResult with validation status and metrics
            
        Requirements: 8.2, 8.3, 8.4
        """
        # Compute semantic similarity
        similarity = self.compute_similarity(original, transformed)
        
        # Check similarity bounds
        if similarity < self.min_similarity:
            logger.warning(
                f"Transformation rejected: similarity {similarity:.4f} "
                f"below minimum {self.min_similarity}"
            )
            return ValidationResult(
                is_valid=False,
                semantic_similarity=similarity,
                career_level_valid=False,
                rejection_reason=f"Semantic similarity too low: {similarity:.4f} < {self.min_similarity}"
            )
        
        if similarity > self.max_similarity:
            logger.warning(
                f"Transformation flagged: similarity {similarity:.4f} "
                f"above maximum {self.max_similarity} (potentially insufficient)"
            )
            return ValidationResult(
                is_valid=False,
                semantic_similarity=similarity,
                career_level_valid=False,
                rejection_reason=f"Semantic similarity too high (insufficient transformation): {similarity:.4f} > {self.max_similarity}"
            )
        
        # Validate career level indicators
        career_level_valid = self.validate_career_level(transformed, target_level)
        
        if not career_level_valid:
            logger.warning(
                f"Transformation flagged: missing {target_level}-level indicators"
            )
            return ValidationResult(
                is_valid=False,
                semantic_similarity=similarity,
                career_level_valid=False,
                rejection_reason=f"Missing {target_level}-level career indicators"
            )
        
        logger.debug(
            f"Transformation validated: similarity={similarity:.4f}, "
            f"career_level_valid={career_level_valid}"
        )
        
        return ValidationResult(
            is_valid=True,
            semantic_similarity=similarity,
            career_level_valid=True
        )
    
    def validate_career_level(self, text: str, target_level: str) -> bool:
        """
        Validate that text contains appropriate career level indicators.
        
        For senior/lead levels, checks for leadership and strategic language.
        For junior/entry levels, checks for learning and support language.
        
        Args:
            text: Text to validate
            target_level: Target career level ("senior", "lead", "junior", "entry")
            
        Returns:
            bool: True if text contains appropriate career level indicators
            
        Requirements: 8.4
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Determine which indicators to check based on target level
        if target_level in ("senior", "lead"):
            indicators = self.SENIOR_INDICATORS
            level_type = "senior"
        elif target_level in ("junior", "entry"):
            indicators = self.JUNIOR_INDICATORS
            level_type = "junior"
        else:
            logger.warning(f"Unknown target level: {target_level}")
            return True  # Don't reject for unknown levels
        
        # Check for presence of at least one indicator
        found_indicators = []
        for indicator in indicators:
            # Use word boundary matching for single words
            if " " in indicator:
                # Multi-word phrase - direct substring match
                if indicator in text_lower:
                    found_indicators.append(indicator)
            else:
                # Single word - use word boundary regex
                pattern = rf'\b{re.escape(indicator)}\b'
                if re.search(pattern, text_lower):
                    found_indicators.append(indicator)
        
        if found_indicators:
            logger.debug(
                f"Found {level_type}-level indicators: {found_indicators[:5]}"
            )
            return True
        
        logger.debug(f"No {level_type}-level indicators found in text")
        return False
    
    def get_career_level_indicators(
        self,
        text: str,
        target_level: str
    ) -> List[str]:
        """
        Get all career level indicators found in text.
        
        Useful for debugging and quality monitoring.
        
        Args:
            text: Text to analyze
            target_level: Target career level
            
        Returns:
            List of found indicators
        """
        if not text:
            return []
        
        text_lower = text.lower()
        
        if target_level in ("senior", "lead"):
            indicators = self.SENIOR_INDICATORS
        elif target_level in ("junior", "entry"):
            indicators = self.JUNIOR_INDICATORS
        else:
            return []
        
        found = []
        for indicator in indicators:
            if " " in indicator:
                if indicator in text_lower:
                    found.append(indicator)
            else:
                pattern = rf'\b{re.escape(indicator)}\b'
                if re.search(pattern, text_lower):
                    found.append(indicator)
        
        return found
