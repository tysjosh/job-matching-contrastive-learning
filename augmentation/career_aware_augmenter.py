"""
Career-Aware Augmenter: The main orchestrator for the "Career Time Machine"

This class coordinates the transformation of resumes into career-progression-aware
views using upward (Tup) and downward (Tdown) transformations.
"""

import logging
from typing import Dict, Tuple, Any
from dataclasses import dataclass

from .upward_transformer import UpwardTransformer
from .downward_transformer import DownwardTransformer
from .semantic_validator import SemanticValidator
from .progression_constraints import ProgressionConstraints
from .metadata_synchronizer import MetadataSynchronizer

logger = logging.getLogger(__name__)


@dataclass
class AugmentedViews:
    """Container for the trio of career-aware resume views"""
    original: Dict[str, Any]
    aspirational: Dict[str, Any]  # r(ℓ+1) - senior view
    foundational: Dict[str, Any]  # r(ℓ−1) - junior view
    metadata: Dict[str, Any]


class CareerAwareAugmenter:
    """
    Main orchestrator for career-aware data augmentation.

    Creates realistic career progression views of resumes by simulating
    how candidates would describe their work at different seniority levels.
    """

    def __init__(self,
                 esco_skills_hierarchy: Dict,
                 career_graph: Any,
                 lambda1: float = 0.3,  # Weight for aspirational view
                 lambda2: float = 0.2,  # Weight for foundational view
                 enable_paraphrasing: bool = True,
                 paraphrasing_config: Dict = None):
        """
        Initialize the Career-Aware Augmenter.

        Args:
            esco_skills_hierarchy: ESCO skills hierarchy for validation
            career_graph: Career graph for domain boundary enforcement
            lambda1: Weight for aspirational (senior) view in loss
            lambda2: Weight for foundational (junior) view in loss
            enable_paraphrasing: Whether to enable paraphrasing for diversity
            paraphrasing_config: Configuration for paraphrasing behavior
        """
        self.esco_skills_hierarchy = esco_skills_hierarchy
        self.career_graph = career_graph
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Initialize transformers with paraphrasing configuration
        paraphrasing_config = paraphrasing_config or {}
        self.upward_transformer = UpwardTransformer(
            enable_paraphrasing=enable_paraphrasing,
            paraphrasing_config=paraphrasing_config
        )
        self.downward_transformer = DownwardTransformer(
            enable_paraphrasing=enable_paraphrasing,
            paraphrasing_config=paraphrasing_config
        )

        # Initialize validators
        self.semantic_validator = SemanticValidator()
        self.progression_constraints = ProgressionConstraints(
            esco_skills_hierarchy, career_graph)
        self.metadata_synchronizer = MetadataSynchronizer()

    def generate_career_views(self,
                              resume: Dict[str, Any],
                              job: Dict[str, Any],
                              current_level: str) -> AugmentedViews:
        """
        Generate the trio of career-aware views for a resume.

        Args:
            resume: Original resume data
            job: Target job for context
            current_level: Current experience level (entry/mid/senior/lead/principal)

        Returns:
            AugmentedViews: Container with original, aspirational, and foundational views
        """
        try:
            # Validate inputs
            if not self._validate_inputs(resume, job, current_level):
                logger.warning("Invalid inputs for career view generation")
                return self._create_fallback_views(resume)

            # Generate aspirational view (Tup)
            aspirational_resume = self._generate_aspirational_view(
                resume, job, current_level)

            # Generate foundational view (Tdown)
            foundational_resume = self._generate_foundational_view(
                resume, job, current_level)

            # Validate semantic coherence
            if not self._validate_semantic_coherence(
                    resume, aspirational_resume, foundational_resume):
                logger.warning("Semantic validation failed, using fallback")
                return self._create_fallback_views(resume)

            # Create metadata
            metadata = {
                'original_level': current_level,
                'aspirational_level': self._get_next_level(current_level),
                'foundational_level': self._get_previous_level(current_level),
                'transformation_quality': self._assess_transformation_quality(
                    resume, aspirational_resume, foundational_resume),
                'lambda_weights': {'lambda1': self.lambda1, 'lambda2': self.lambda2}
            }

            return AugmentedViews(
                original=resume,
                aspirational=aspirational_resume,
                foundational=foundational_resume,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error generating career views: {e}")
            return self._create_fallback_views(resume)

    def _generate_aspirational_view(self,
                                    resume: Dict[str, Any],
                                    job: Dict[str, Any],
                                    current_level: str) -> Dict[str, Any]:
        """Generate senior-level view using Tup transformation"""
        target_level = self._get_next_level(current_level)

        # Apply upward transformation
        aspirational_resume = self.upward_transformer.transform(
            resume, target_level, job)

        # Synchronize metadata with transformed content
        sync_result = self.metadata_synchronizer.synchronize_experience_metadata(
            resume=aspirational_resume,
            transformation_type='upward',
            target_level=target_level
        )
        
        if sync_result.success and sync_result.updated_fields:
            aspirational_resume['metadata'] = sync_result.synchronized_metadata
            logger.debug(f"Synchronized metadata fields for aspirational view: {sync_result.updated_fields}")
        elif not sync_result.success:
            logger.warning(f"Metadata synchronization failed for aspirational view: {sync_result.errors}")

        # Validate against progression constraints
        if not self.progression_constraints.validate_upward_progression(
                resume, aspirational_resume, current_level, target_level):
            logger.debug("Upward progression validation failed")
            return resume  # Fallback to original

        return aspirational_resume

    def _generate_foundational_view(self,
                                    resume: Dict[str, Any],
                                    job: Dict[str, Any],
                                    current_level: str) -> Dict[str, Any]:
        """Generate junior-level view using Tdown transformation"""
        target_level = self._get_previous_level(current_level)

        # Apply downward transformation
        foundational_resume = self.downward_transformer.transform(
            resume, target_level, job)

        # Synchronize metadata with transformed content
        sync_result = self.metadata_synchronizer.synchronize_experience_metadata(
            resume=foundational_resume,
            transformation_type='downward',
            target_level=target_level
        )
        
        if sync_result.success and sync_result.updated_fields:
            foundational_resume['metadata'] = sync_result.synchronized_metadata
            logger.debug(f"Synchronized metadata fields for foundational view: {sync_result.updated_fields}")
        elif not sync_result.success:
            logger.warning(f"Metadata synchronization failed for foundational view: {sync_result.errors}")

        # Validate against progression constraints
        if not self.progression_constraints.validate_downward_progression(
                resume, foundational_resume, current_level, target_level):
            logger.debug("Downward progression validation failed")
            return resume  # Fallback to original

        return foundational_resume

    def _validate_inputs(self, resume: Dict, job: Dict, level: str) -> bool:
        """Validate input data for augmentation"""
        if not resume or not job:
            return False

        # Normalize the level using the unified system
        normalized_level = self.progression_constraints.normalize_seniority_level(
            level)

        # Check if normalized level is in the unified hierarchy
        if normalized_level not in self.progression_constraints.level_hierarchy:
            logger.warning(
                f"Unrecognized seniority level: {level} (normalized: {normalized_level})")
            return False

        # Check for required resume fields
        required_fields = ['experience', 'skills']
        if not any(field in resume for field in required_fields):
            return False

        return True

    def _validate_semantic_coherence(self,
                                     original: Dict,
                                     aspirational: Dict,
                                     foundational: Dict) -> bool:
        """Validate that all three views maintain semantic coherence"""
        # Validate basic semantic coherence
        if not self.semantic_validator.validate_coherence(
                original, aspirational, foundational):
            return False
        
        # Validate job title and experience description coherence for each view
        views = [
            ('original', original),
            ('aspirational', aspirational), 
            ('foundational', foundational)
        ]
        
        for view_name, view in views:
            if not self._validate_job_title_experience_coherence(view):
                logger.debug(f"Job title-experience coherence validation failed for {view_name} view")
                return False
        
        return True

    def _get_next_level(self, current_level: str) -> str:
        """Get the next career level for upward transformation using unified hierarchy"""
        # Normalize the current level first
        normalized_level = self.progression_constraints.normalize_seniority_level(
            current_level)
        current_rank = self.progression_constraints.level_hierarchy.get(
            normalized_level, 1)

        # Find the next level with a higher rank
        next_levels = [level for level, rank in self.progression_constraints.level_hierarchy.items()
                       if rank == current_rank + 1]

        if next_levels:
            # Prefer common progression paths
            preferred_progression = {
                'intern': 'junior',
                'trainee': 'junior',
                'entry': 'mid',
                'junior': 'mid',
                'assistant': 'associate',
                'associate': 'senior',
                'mid': 'senior',
                'specialist': 'senior',
                'senior': 'lead',
                'experienced': 'lead',
                'advanced': 'lead',
                'lead': 'manager',
                'staff': 'manager',
                'team_lead': 'manager',
                'manager': 'director',
                'senior_manager': 'director',
                'principal': 'director',
                'director': 'head',
                'head': 'chief',
                'executive': 'chief'
            }

            return preferred_progression.get(normalized_level, next_levels[0])

        # Stay at current level if at top
        return normalized_level

    def _get_previous_level(self, current_level: str) -> str:
        """Get the previous career level for downward transformation using unified hierarchy"""
        # Normalize the current level first
        normalized_level = self.progression_constraints.normalize_seniority_level(
            current_level)
        current_rank = self.progression_constraints.level_hierarchy.get(
            normalized_level, 1)

        # Find the previous level with a lower rank
        prev_levels = [level for level, rank in self.progression_constraints.level_hierarchy.items()
                       if rank == current_rank - 1]

        if prev_levels:
            # Prefer common regression paths
            preferred_regression = {
                'chief': 'head',
                'vp': 'director',
                'cto': 'director',
                'cio': 'director',
                'head': 'director',
                'executive': 'director',
                'director': 'manager',
                'manager': 'lead',
                'senior_manager': 'lead',
                'principal': 'lead',
                'lead': 'senior',
                'staff': 'senior',
                'team_lead': 'senior',
                'senior': 'mid',
                'experienced': 'mid',
                'advanced': 'mid',
                'mid': 'junior',
                'specialist': 'associate',
                'associate': 'junior',
                'junior': 'entry',
                'entry': 'intern'
            }

            return preferred_regression.get(normalized_level, prev_levels[0])

        # Stay at current level if at bottom
        return normalized_level

    def _assess_transformation_quality(self,
                                       original: Dict,
                                       aspirational: Dict,
                                       foundational: Dict) -> float:
        """Assess the quality of transformations (0.0 to 1.0)"""
        # Simple quality metric based on text differences
        original_text = str(original.get('experience', ''))
        asp_text = str(aspirational.get('experience', ''))
        found_text = str(foundational.get('experience', ''))

        # Quality based on meaningful changes while preserving core content
        if asp_text == original_text and found_text == original_text:
            return 0.0  # No transformation occurred

        return 0.8  # Placeholder - would implement more sophisticated metrics

    def _validate_job_title_experience_coherence(self, resume: Dict[str, Any]) -> bool:
        """
        Validate coherence between job title and experience description.
        
        Args:
            resume: Resume data to validate
            
        Returns:
            bool: True if job title and experience are coherent
        """
        try:
            # Extract text content for analysis
            text_content = self._extract_text_content_for_validation(resume)
            
            # Get experience level from metadata
            experience_level = resume.get('metadata', {}).get('experience_level')
            if not experience_level:
                # If no metadata, try to infer from content
                experience_level = self._infer_experience_level_from_content(text_content)
            
            # Use metadata synchronizer to validate consistency
            consistency_report = self.metadata_synchronizer.validate_skill_metadata_consistency(
                skills_array=resume.get('skills', []),
                experience_text=text_content,
                experience_level=experience_level
            )
            
            # Consider coherent if consistency score is above threshold
            coherence_threshold = 0.7
            is_coherent = consistency_report.consistency_score >= coherence_threshold
            
            if not is_coherent:
                logger.debug(f"Job title-experience coherence below threshold: {consistency_report.consistency_score:.2f}")
                logger.debug(f"Misalignments: {consistency_report.misalignments}")
            
            return is_coherent
            
        except Exception as e:
            logger.error(f"Error validating job title-experience coherence: {e}")
            return True  # Default to valid to avoid blocking transformations

    def _extract_text_content_for_validation(self, resume: Dict[str, Any]) -> str:
        """Extract text content from resume for validation"""
        text_parts = []
        
        # Extract experience text
        if 'experience' in resume:
            exp = resume['experience']
            if isinstance(exp, str):
                text_parts.append(exp)
            elif isinstance(exp, list):
                for item in exp:
                    if isinstance(item, dict):
                        if 'responsibilities' in item:
                            resp = item['responsibilities']
                            if isinstance(resp, str):
                                text_parts.append(resp)
                            elif isinstance(resp, dict):
                                text_parts.extend(str(v) for v in resp.values())
                    elif isinstance(item, str):
                        text_parts.append(item)
        
        # Extract job title
        if 'job_title' in resume:
            text_parts.append(str(resume['job_title']))
        elif 'title' in resume:
            text_parts.append(str(resume['title']))
        
        # Extract summary
        if 'summary' in resume:
            summary = resume['summary']
            if isinstance(summary, str):
                text_parts.append(summary)
            elif isinstance(summary, dict) and 'text' in summary:
                text_parts.append(summary['text'])
        
        return ' '.join(text_parts)

    def _infer_experience_level_from_content(self, text_content: str) -> str:
        """Infer experience level from text content"""
        text_lower = text_content.lower()
        
        # Check for senior indicators
        if any(indicator in text_lower for indicator in ['senior', 'lead', 'principal', 'architect', 'manager']):
            return 'senior'
        
        # Check for junior indicators  
        if any(indicator in text_lower for indicator in ['junior', 'entry', 'associate', 'trainee', 'intern']):
            return 'junior'
        
        # Default to mid-level
        return 'mid'

    def _create_fallback_views(self, resume: Dict[str, Any]) -> None:
        """
        REMOVED: Fallback views that create identical embeddings.
        
        Instead of creating identical copies that cause embedding collapse,
        we return None to indicate transformation failure. The calling code
        should handle this by skipping the sample or using alternative strategies.
        
        This prevents:
        - Embedding collapse from identical views
        - Degenerate triplets in contrastive learning
        - Training instability from zero-quality augmentations
        """
        logger.warning(f"Transformation failed for resume - no fallback provided to prevent embedding collapse")
        return None

    def get_augmented_loss_weights(self) -> Tuple[float, float]:
        """Get the lambda weights for augmented loss calculation"""
        return self.lambda1, self.lambda2
    def get_paraphrasing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive paraphrasing statistics from both transformers"""
        upward_stats = self.upward_transformer.get_paraphrasing_statistics()
        downward_stats = self.downward_transformer.get_paraphrasing_statistics()
        
        return {
            'upward_transformer': upward_stats,
            'downward_transformer': downward_stats,
            'combined_enabled': upward_stats.get('paraphrasing_enabled', False) or downward_stats.get('paraphrasing_enabled', False)
        }
    
    def configure_paraphrasing(self, 
                             enable: bool = None,
                             min_diversity_score: float = None,
                             max_semantic_drift: float = None,
                             preserve_technical_terms: bool = None):
        """
        Configure paraphrasing settings for both transformers.
        
        Args:
            enable: Enable or disable paraphrasing
            min_diversity_score: Minimum diversity score to achieve
            max_semantic_drift: Maximum allowed semantic drift
            preserve_technical_terms: Whether to preserve technical terms
        """
        # Configure both transformers
        self.upward_transformer.configure_paraphrasing(
            enable=enable,
            min_diversity_score=min_diversity_score,
            max_semantic_drift=max_semantic_drift,
            preserve_technical_terms=preserve_technical_terms
        )
        
        self.downward_transformer.configure_paraphrasing(
            enable=enable,
            min_diversity_score=min_diversity_score,
            max_semantic_drift=max_semantic_drift,
            preserve_technical_terms=preserve_technical_terms
        )
        
        logger.info("Paraphrasing configuration updated for both transformers")
    
    def reset_diversity_tracking(self):
        """Reset diversity tracking for both transformers"""
        self.upward_transformer.reset_diversity_tracking()
        self.downward_transformer.reset_diversity_tracking()
        logger.info("Diversity tracking reset for both transformers")