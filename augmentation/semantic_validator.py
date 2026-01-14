"""
Semantic Validator: Ensures career-aware transformations maintain meaning

This validator implements the "Semantic Validity Preservation" guardrails
to ensure transformations remain realistic and coherent.
"""

import logging
import re
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class SemanticValidator:
    """
    Validates that career-aware transformations maintain semantic coherence
    while adding appropriate career progression context.
    """

    def __init__(self,
                 # Minimum 40% similarity to prevent semantic drift
                 min_similarity_threshold: float = 0.4,
                 # Maximum 85% similarity to prevent embedding collapse
                 max_similarity_threshold: float = 0.85,
                 # Higher threshold for upward transformations
                 upward_min_threshold: float = 0.5,
                 # Stricter max for upward transformations
                 upward_max_threshold: float = 0.8,
                 max_length_ratio: float = 2.0,             # Reduced from 3.0 for tighter control
                 min_transformation_quality: float = 0.3):  # Minimum quality threshold
        """
        Initialize enhanced semantic validator with embedding collapse prevention.

        Args:
            min_similarity_threshold: Minimum similarity to prevent semantic drift
            max_similarity_threshold: Maximum similarity to prevent embedding collapse
            upward_min_threshold: Minimum similarity for upward transformations
            upward_max_threshold: Maximum similarity for upward transformations  
            max_length_ratio: Maximum allowed length increase ratio
            min_transformation_quality: Minimum transformation quality required
        """
        self.min_similarity_threshold = min_similarity_threshold
        self.max_similarity_threshold = max_similarity_threshold
        self.upward_min_threshold = upward_min_threshold
        self.upward_max_threshold = upward_max_threshold
        self.max_length_ratio = max_length_ratio
        self.min_transformation_quality = min_transformation_quality
        self._load_validation_rules()

        # Initialize text encoder for embedding validation
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            self.text_encoder = None
            logger.warning(
                "SentenceTransformer not available - embedding validation disabled")

    def _load_validation_rules(self):
        """Load semantic validation rules"""

        self.core_concepts = self._load_core_concepts_from_cs_database()

        # Forbidden transformations that would break semantic meaning
        self.forbidden_transformations = [
            # Can't change core technical skills
            ('python', 'java'),
            ('frontend', 'backend'),
            ('mobile', 'web'),
            ('data science', 'cybersecurity'),
            # Can't change fundamental job functions
            ('developer', 'manager'),
            ('engineer', 'designer'),
            ('analyst', 'developer')
        ]

        # Required preservation patterns (more lenient for augmentation)
        self.preservation_patterns = [
            # Time periods - allow transformation (e.g., "3 years" -> "5+ years")
            r'\b\d+\s*(years?|months?)\b',
            r'\b\d+%\b',  # Percentages
            r'\$\d+[kmb]?\b',  # Money amounts
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\.(com|org|net)\b'  # URLs/domains
        ]

        # Flexible patterns that can be modified during transformation
        self.flexible_patterns = [
            r'\b\d+\s*(years?|months?)\b',  # Years of experience can change
        ]

    def _load_core_concepts_from_cs_database(self) -> Dict[str, List[str]]:
        """Load core concepts from CS skills database for validation"""
        import os
        import json

        try:
            # Load CS skills database
            dataset_dir = 'dataset'
            cs_skills_file = os.path.join(dataset_dir, 'cs_skills.json')

            with open(cs_skills_file, 'r', encoding='utf-8') as f:
                cs_skills = json.load(f)

            # Build core concepts from CS skills database
            core_concepts = {}

            # Programming languages (lowercase for matching)
            if 'programming_languages' in cs_skills:
                core_concepts['programming_languages'] = [
                    lang.lower() for lang in cs_skills['programming_languages'] if lang
                ]

            # Technologies (frameworks + cloud platforms + tools)
            technologies = []
            tech_categories = ['frameworks', 'web_development_frameworks', 'cloud_platforms',
                               'devops_tools', 'containerization', 'databases']
            for category in tech_categories:
                if category in cs_skills:
                    for item in cs_skills[category]:
                        if item:
                            # Normalize cloud platform names
                            if 'Amazon Web Services (AWS)' in item:
                                technologies.append('aws')
                            elif 'Google Cloud Platform (GCP)' in item:
                                technologies.append('gcp')
                            elif 'Microsoft Azure' in item:
                                technologies.append('azure')
                            else:
                                technologies.append(item.lower())

            core_concepts['technologies'] = technologies

            # Methodologies (add common ones from soft skills and methodologies)
            methodologies = ['agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd',
                             'microservices', 'rest', 'graphql', 'api']
            if 'soft_skills' in cs_skills:
                for skill in cs_skills['soft_skills']:
                    if skill and any(term in skill.lower() for term in ['agile', 'scrum', 'methodology']):
                        methodologies.append(skill.lower())

            core_concepts['methodologies'] = methodologies

            # Domains (common specialization areas)
            domains = ['web development', 'mobile development', 'data science',
                       'machine learning', 'cybersecurity', 'cloud computing',
                       'backend development', 'frontend development', 'devops']

            core_concepts['domains'] = domains

            return core_concepts

        except Exception as e:
            # Fallback to basic hardcoded concepts if database loading fails
            return {
                'programming_languages': [
                    'python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust',
                    'typescript', 'php', 'ruby', 'swift', 'kotlin'
                ],
                'technologies': [
                    'react', 'angular', 'vue', 'node', 'django', 'flask',
                    'spring', 'express', 'docker', 'kubernetes', 'aws', 'azure'
                ],
                'methodologies': [
                    'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd',
                    'microservices', 'rest', 'graphql', 'api'
                ],
                'domains': [
                    'web development', 'mobile development', 'data science',
                    'machine learning', 'cybersecurity', 'cloud computing'
                ]
            }

    def _get_technical_terms_set(self) -> set:
        """Get technical terms from core concepts for preservation checking"""
        technical_terms = set()

        # Add programming languages
        if 'programming_languages' in self.core_concepts:
            technical_terms.update(self.core_concepts['programming_languages'])

        # Add selected technologies (focus on acronyms and key terms)
        if 'technologies' in self.core_concepts:
            for tech in self.core_concepts['technologies']:
                # Include important acronyms and technologies
                if len(tech) <= 5 or tech in ['sql', 'api', 'aws', 'gcp', 'azure',
                                              'html', 'css', 'xml', 'json', 'rest', 'crud',
                                              'docker', 'kubernetes', 'react', 'vue', 'node']:
                    technical_terms.add(tech.lower())

        # Add common technical acronyms
        common_acronyms = {'sql', 'api', 'aws', 'gcp',
                           'html', 'css', 'xml', 'json', 'rest', 'crud'}
        technical_terms.update(common_acronyms)

        return technical_terms

    def validate_coherence(self,
                           original: Dict[str, Any],
                           aspirational: Dict[str, Any],
                           foundational: Dict[str, Any]) -> bool:
        """
        Validate semantic coherence across all three career views.

        Args:
            original: Original resume
            aspirational: Senior-level view
            foundational: Junior-level view

        Returns:
            bool: True if all views maintain semantic coherence
        """
        try:
            # Validate original vs aspirational
            if not self._validate_transformation_pair(original, aspirational, 'upward'):
                logger.debug("Aspirational view validation failed")
                return False

            # Validate original vs foundational
            if not self._validate_transformation_pair(original, foundational, 'downward'):
                logger.debug("Foundational view validation failed")
                return False

            # Validate core concept preservation across all views
            if not self._validate_core_concept_preservation(
                    original, aspirational, foundational):
                logger.debug("Core concept preservation validation failed")
                return False

            return True

        except Exception as e:
            logger.error(f"Coherence validation error: {e}")
            return False

    def _validate_transformation_pair(self,
                                      original: Dict,
                                      transformed: Dict,
                                      direction: str) -> bool:
        """Validate a single transformation pair"""

        # Focus similarity check on experience text only (most important for validation)
        original_exp = original.get('experience', '')
        transformed_exp = transformed.get('experience', '')

        # Use different thresholds based on transformation direction
        threshold = self.upward_min_threshold if direction == 'upward' else self.min_similarity_threshold

        # Check similarity threshold on experience text
        similarity = self._calculate_similarity(original_exp, transformed_exp)

        logger.debug(
            f"{direction.capitalize()} similarity: {similarity:.3f} (threshold: {threshold:.3f})")

        if similarity < threshold:
            logger.debug(
                f"Similarity too low: {similarity:.3f} < {threshold:.3f}")
            return False

        # Extract full text content for other validations
        original_text = self._extract_text_content(original)
        transformed_text = self._extract_text_content(transformed)

        # Check length ratio
        length_ratio = len(transformed_text) / max(len(original_text), 1)
        if length_ratio > self.max_length_ratio:
            logger.debug(f"Length ratio too high: {length_ratio:.3f}")
            return False

        # Check for forbidden transformations
        if self._contains_forbidden_transformation(original_text, transformed_text):
            logger.debug("Contains forbidden transformation")
            return False

        # Validate preservation patterns
        if not self._validate_preservation_patterns(original_text, transformed_text):
            logger.debug("Preservation pattern validation failed")
            return False

        return True

    def _validate_core_concept_preservation(self,
                                            original: Dict,
                                            aspirational: Dict,
                                            foundational: Dict) -> bool:
        """Ensure core technical concepts are preserved across all views"""

        # Extract core concepts from original
        original_concepts = self._extract_core_concepts(original)

        # Check preservation in aspirational view
        aspirational_concepts = self._extract_core_concepts(aspirational)
        if not self._concepts_preserved(original_concepts, aspirational_concepts):
            return False

        # Check preservation in foundational view
        foundational_concepts = self._extract_core_concepts(foundational)
        if not self._concepts_preserved(original_concepts, foundational_concepts):
            return False

        return True

    def _extract_text_content(self, resume: Dict[str, Any]) -> str:
        """Extract all text content from resume for comparison"""
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
                                text_parts.extend(str(v)
                                                  for v in resp.values())
                    elif isinstance(item, str):
                        text_parts.append(item)

        # Extract skills text (only skill names, not full dictionaries)
        if 'skills' in resume:
            skills = resume['skills']
            if isinstance(skills, list):
                for skill in skills:
                    if isinstance(skill, dict) and 'name' in skill:
                        text_parts.append(skill['name'])
                    elif isinstance(skill, str):
                        text_parts.append(skill)

        # Extract summary text
        if 'summary' in resume:
            summary = resume['summary']
            if isinstance(summary, str):
                text_parts.append(summary)
            elif isinstance(summary, dict) and 'text' in summary:
                text_parts.append(summary['text'])

        return ' '.join(text_parts)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts with improved tokenization"""
        import re

        def tokenize(text):
            # Remove punctuation, handle contractions, and split into words
            # This handles cases like "javascript," -> "javascript"
            # Replace punctuation with spaces
            text = re.sub(r"[^\w\s]", " ", text.lower())
            text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace
            words = set(text.split())
            return words

        words1 = tokenize(text1)
        words2 = tokenize(text2)

        # Debug logging for tokenization
        logger.debug(f"Original words ({len(words1)}): {sorted(list(words1))}")
        logger.debug(
            f"Transformed words ({len(words2)}): {sorted(list(words2))}")

        # Calculate overlap metrics
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        logger.debug(
            f"Word overlap ({len(intersection)}/{len(words1)}): {sorted(list(intersection))}")

        if union == 0:
            return 1.0  # Both texts are empty

        # Jaccard similarity (intersection over union)
        jaccard_similarity = len(intersection) / len(union)

        # Overlap coefficient (intersection over minimum) - more forgiving for augmentations
        if len(words1) == 0 and len(words2) == 0:
            overlap_coefficient = 1.0
        elif len(words1) == 0 or len(words2) == 0:
            overlap_coefficient = 0.0
        else:
            overlap_coefficient = len(intersection) / \
                min(len(words1), len(words2))

        # Word preservation ratio (how much of original is preserved)
        preservation_ratio = len(intersection) / \
            len(words1) if len(words1) > 0 else 1.0

        # Sequence similarity for word order (with better tokenization)
        tokens1 = re.sub(r"[^\w\s]", " ", text1.lower()).split()
        tokens2 = re.sub(r"[^\w\s]", " ", text2.lower()).split()

        matcher = SequenceMatcher(None, tokens1, tokens2)
        sequence_similarity = matcher.ratio()

        # Combine metrics with emphasis on preservation for semantic validation
        # Higher weight on preservation ratio and overlap coefficient for augmentation
        combined_similarity = (0.3 * jaccard_similarity +
                               0.4 * overlap_coefficient +
                               0.2 * preservation_ratio +
                               0.1 * sequence_similarity)

        logger.debug(f"Similarity metrics - Jaccard: {jaccard_similarity:.3f}, "
                     f"Overlap: {overlap_coefficient:.3f}, "
                     f"Preservation: {preservation_ratio:.3f}, "
                     f"Sequence: {sequence_similarity:.3f}, "
                     f"Combined: {combined_similarity:.3f}")

        return combined_similarity

    def _contains_forbidden_transformation(self, original: str, transformed: str) -> bool:
        """Check if transformation contains forbidden changes"""
        original_lower = original.lower()
        transformed_lower = transformed.lower()

        for forbidden_from, forbidden_to in self.forbidden_transformations:
            # Check if we're changing core technical concepts inappropriately
            if (forbidden_from in original_lower and
                forbidden_to in transformed_lower and
                    forbidden_from not in transformed_lower):
                return True

        return False

    def _validate_preservation_patterns(self, original: str, transformed: str) -> bool:
        """Validate that important patterns are preserved (with flexibility for augmentation)"""
        for pattern in self.preservation_patterns:
            original_matches = set(re.findall(
                pattern, original, re.IGNORECASE))
            transformed_matches = set(re.findall(
                pattern, transformed, re.IGNORECASE))

            # Skip validation for flexible patterns in augmentation contexts
            if pattern in self.flexible_patterns:
                # For flexible patterns like years of experience, allow transformation
                # Just check that if there was a time period, there's still some time period
                if original_matches and not transformed_matches:
                    logger.debug(
                        f"Flexible pattern completely removed, checking for similar patterns")
                    # Check for variations like "5+" instead of "5"
                    variation_pattern = r'\b\d+\+?\s*(years?|months?)\b'
                    variation_matches = set(re.findall(
                        variation_pattern, transformed, re.IGNORECASE))
                    if not variation_matches:
                        logger.debug(
                            f"No time period found after transformation")
                        return False
                # Otherwise allow transformation (e.g., "3 years" -> "5+ years")
                continue

            # Special handling for technical terms (case-insensitive comparison)
            if pattern == r'\b[A-Z]{2,}\b':
                # For technical terms, compare case-insensitively
                original_lower = {match.lower() for match in original_matches}
                transformed_lower = {match.lower()
                                     for match in transformed_matches}

                # Only check that most technical terms are preserved (case-insensitive, allow some loss)
                technical_terms = self._get_technical_terms_set()
                original_technical = original_lower.intersection(
                    technical_terms)
                transformed_technical = transformed_lower.intersection(
                    technical_terms)

                # More lenient preservation - allow up to 80% of technical terms to be lost during transformation
                if original_technical:
                    preserved_ratio = len(transformed_technical.intersection(
                        original_technical)) / len(original_technical)
                    if preserved_ratio < 0.2:  # Only require 20% preservation for flexible augmentation
                        logger.debug(
                            f"Technical terms preservation too low: {preserved_ratio:.3f}, "
                            f"original: {original_technical}, "
                            f"preserved: {transformed_technical.intersection(original_technical)}")
                        return False
            else:
                # For other patterns (percentages, money amounts, URLs), be more lenient
                # Allow some loss but not complete loss if there were many matches
                if len(original_matches) > 2:  # If there were many matches originally
                    # Allow up to 50% loss
                    preserved_ratio = len(transformed_matches.intersection(
                        original_matches)) / len(original_matches)
                    if preserved_ratio < 0.5:
                        logger.debug(
                            f"Pattern preservation too low for {pattern}: {preserved_ratio:.3f}")
                        return False
                elif original_matches and not transformed_matches.intersection(original_matches):
                    # If there were only a few matches, require at least one to be preserved
                    logger.debug(
                        f"No preservation for critical pattern: {pattern}")
                    return False

        return True

    def _extract_core_concepts(self, resume: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract core technical concepts from resume"""
        text = self._extract_text_content(resume).lower()
        concepts = {}

        for concept_type, concept_list in self.core_concepts.items():
            found_concepts = []
            for concept in concept_list:
                if concept in text:
                    found_concepts.append(concept)
            concepts[concept_type] = found_concepts

        return concepts

    def _concepts_preserved(self,
                            original_concepts: Dict[str, List[str]],
                            transformed_concepts: Dict[str, List[str]]) -> bool:
        """Check if core concepts are preserved in transformation with more lenient thresholds"""
        for concept_type, original_list in original_concepts.items():
            transformed_list = transformed_concepts.get(concept_type, [])

            # Different preservation requirements for different concept types
            if concept_type == 'programming_languages':
                # Programming languages should be mostly preserved (70%)
                required_ratio = 0.7
            elif concept_type == 'technologies':
                # Technologies can be more flexible (30% - very lenient for augmentation)
                # For downward transformations, technologies might be removed to simplify
                required_ratio = 0.3
            elif concept_type == 'domains':
                # Domains should be mostly preserved (60%)
                required_ratio = 0.6
            else:
                # Default ratio (30% - very lenient)
                required_ratio = 0.3

            if original_list:
                preserved_count = len(
                    set(original_list) & set(transformed_list))
                preserved_ratio = preserved_count / len(original_list)

                logger.debug(f"Concept preservation for {concept_type}: "
                             f"{preserved_count}/{len(original_list)} = {preserved_ratio:.3f} "
                             f"(required: {required_ratio:.3f})")

                if preserved_ratio < required_ratio:
                    missing_concepts = set(
                        original_list) - set(transformed_list)
                    logger.debug(f"Missing {concept_type}: {missing_concepts}")

                    # For technologies, be very forgiving - allow complete removal in downward transformations
                    if concept_type == 'technologies':
                        # Few technologies originally
                        if preserved_count == 0 and len(original_list) <= 3:
                            logger.debug(
                                f"Technologies: Allowing complete removal for simplification")
                            continue
                        elif preserved_count > 0:  # At least one preserved
                            logger.debug(
                                f"Technologies: Allowing partial preservation ({preserved_count} preserved)")
                            continue

                    return False

        return True

    def validate_single_transformation(self,
                                       original: Dict[str, Any],
                                       transformed: Dict[str, Any],
                                       transformation_type: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a single transformation and return validation report.

        Args:
            original: Original resume
            transformed: Transformed resume
            transformation_type: 'upward' or 'downward'

        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            'transformation_type': transformation_type,
            'similarity_score': 0.0,
            'length_ratio': 0.0,
            'concepts_preserved': False,
            'patterns_preserved': False,
            'forbidden_changes': False,
            'overall_valid': False
        }

        try:
            original_text = self._extract_text_content(original)
            transformed_text = self._extract_text_content(transformed)

            # Calculate metrics
            report['similarity_score'] = self._calculate_similarity(
                original_text, transformed_text)
            report['length_ratio'] = len(
                transformed_text) / max(len(original_text), 1)
            report['concepts_preserved'] = self._concepts_preserved(
                self._extract_core_concepts(original),
                self._extract_core_concepts(transformed)
            )
            report['patterns_preserved'] = self._validate_preservation_patterns(
                original_text, transformed_text)
            report['forbidden_changes'] = self._contains_forbidden_transformation(
                original_text, transformed_text)

            # Use appropriate threshold based on transformation type
            threshold = self.upward_min_threshold if transformation_type == 'upward' else self.min_similarity_threshold

            # Overall validation
            report['overall_valid'] = (
                report['similarity_score'] >= threshold and
                report['length_ratio'] <= self.max_length_ratio and
                report['concepts_preserved'] and
                report['patterns_preserved'] and
                not report['forbidden_changes']
            )

            return report['overall_valid'], report

        except Exception as e:
            logger.error(f"Validation error: {e}")
            report['error'] = str(e)
            return False, report

    def validate_embedding_distance(self,
                                    original: Dict[str, Any],
                                    transformed: Dict[str, Any],
                                    transformation_type: str) -> Tuple[bool, float]:
        """
        Validate embedding distance to prevent collapse and ensure meaningful differences.

        Args:
            original: Original resume
            transformed: Transformed resume  
            transformation_type: 'upward' or 'downward'

        Returns:
            Tuple of (is_valid, similarity_score)
        """
        if not self.text_encoder:
            logger.warning(
                "Text encoder not available - skipping embedding validation")
            return True, 0.5  # Default to valid if encoder unavailable

        try:
            # Extract text content
            original_text = self._extract_text_content(original)
            transformed_text = self._extract_text_content(transformed)

            # Generate embeddings
            original_emb = self.text_encoder.encode(
                original_text, convert_to_tensor=True)
            transformed_emb = self.text_encoder.encode(
                transformed_text, convert_to_tensor=True)

            # Calculate cosine similarity
            import torch.nn.functional as F
            cosine_sim = F.cosine_similarity(original_emb.unsqueeze(0),
                                             transformed_emb.unsqueeze(0)).item()

            # Use transformation-specific thresholds
            if transformation_type == 'upward':
                min_threshold = self.upward_min_threshold
                max_threshold = self.upward_max_threshold
            else:
                min_threshold = self.min_similarity_threshold
                max_threshold = self.max_similarity_threshold

            # Validate embedding distance
            is_valid = min_threshold <= cosine_sim <= max_threshold

            if not is_valid:
                if cosine_sim > max_threshold:
                    logger.debug(
                        f"Embedding collapse risk: similarity {cosine_sim:.3f} > {max_threshold}")
                else:
                    logger.debug(
                        f"Semantic drift risk: similarity {cosine_sim:.3f} < {min_threshold}")

            return is_valid, cosine_sim

        except Exception as e:
            logger.error(f"Embedding distance validation error: {e}")
            return False, 0.0

    def validate_transformation_quality(self,
                                        original: Dict[str, Any],
                                        transformed: Dict[str, Any],
                                        transformation_type: str,
                                        transformation_quality: float) -> bool:
        """
        Comprehensive validation including quality gates and embedding distance.

        Args:
            original: Original resume
            transformed: Transformed resume
            transformation_type: 'upward' or 'downward'
            transformation_quality: Quality score from transformer

        Returns:
            bool: True if transformation meets all quality requirements
        """
        # Quality gate: Reject low-quality transformations
        if transformation_quality < self.min_transformation_quality:
            logger.debug(
                f"Transformation quality too low: {transformation_quality:.3f} < {self.min_transformation_quality}")
            return False

        # Semantic validation
        is_semantically_valid = self._validate_transformation_pair(
            original, transformed, transformation_type
        )
        if not is_semantically_valid:
            logger.debug("Semantic validation failed")
            return False

        # Embedding distance validation
        is_embedding_valid, embedding_similarity = self.validate_embedding_distance(
            original, transformed, transformation_type
        )
        if not is_embedding_valid:
            logger.debug(
                f"Embedding distance validation failed: {embedding_similarity:.3f}")
            return False

        logger.debug(f"Transformation validation passed - Quality: {transformation_quality:.3f}, "
                     f"Embedding similarity: {embedding_similarity:.3f}")
        return True
