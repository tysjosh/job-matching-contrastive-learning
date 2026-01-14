"""
Enhanced Semantic Validator: Advanced validation with embedding monitoring and metadata consistency

This enhanced validator implements comprehensive quality gates including:
- Context-aware similarity thresholds
- Embedding distance validation to prevent collapse
- Technical term preservation with single-character handling
- Metadata consistency validation
- Quality reporting and recommendations
"""

import logging
import re
import numpy as np
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Comprehensive validation result with detailed metrics and recommendations"""
    is_valid: bool
    semantic_score: float
    embedding_similarity: float
    metadata_consistency: float
    technical_preservation: float
    quality_gates_passed: List[str]
    quality_gates_failed: List[str]
    recommendations: List[str]
    failure_reasons: List[str]
    transformation_type: str
    overall_quality_score: float


@dataclass
class DiversityReport:
    """Report on embedding diversity across transformations"""
    min_pairwise_distance: float
    mean_pairwise_distance: float
    diversity_index: float
    collapse_risk_score: float
    recommendations: List[str]
    is_diverse: bool


@dataclass
class PreservationReport:
    """Report on technical term preservation"""
    preserved_terms: Set[str]
    corrupted_terms: Set[str]
    preservation_ratio: float
    single_char_terms_handled: int
    recommendations: List[str]
    is_preserved: bool


@dataclass
class ConsistencyReport:
    """Report on metadata consistency"""
    experience_level_aligned: bool
    skill_proficiency_aligned: bool
    job_title_coherent: bool
    consistency_score: float
    misalignments: List[str]
    recommendations: List[str]


class EnhancedSemanticValidator:
    """
    Enhanced semantic validator with comprehensive quality gates and monitoring.
    
    Key enhancements over base validator:
    - Context-aware similarity thresholds
    - Embedding distance validation
    - Technical term preservation with single-character handling
    - Metadata consistency validation
    - Quality reporting and recommendations
    """

    def __init__(self,
                 # Context-aware thresholds
                 upward_min_threshold: float = 0.5,
                 upward_max_threshold: float = 0.8,
                 downward_min_threshold: float = 0.4,
                 downward_max_threshold: float = 0.85,
                 
                 # Quality gates
                 min_transformation_quality: float = 0.4,
                 min_metadata_consistency: float = 0.7,
                 min_technical_preservation: float = 0.8,
                 
                 # Diversity requirements
                 min_diversity_threshold: float = 0.3,
                 max_collapse_risk: float = 0.2,
                 
                 # Length constraints
                 max_length_ratio: float = 2.0):
        """
        Initialize enhanced semantic validator.
        
        Args:
            upward_min_threshold: Minimum similarity for upward transformations
            upward_max_threshold: Maximum similarity for upward transformations
            downward_min_threshold: Minimum similarity for downward transformations
            downward_max_threshold: Maximum similarity for downward transformations
            min_transformation_quality: Minimum overall transformation quality
            min_metadata_consistency: Minimum metadata consistency score
            min_technical_preservation: Minimum technical term preservation ratio
            min_diversity_threshold: Minimum embedding diversity threshold
            max_collapse_risk: Maximum acceptable collapse risk score
            max_length_ratio: Maximum allowed length increase ratio
        """
        self.upward_min_threshold = upward_min_threshold
        self.upward_max_threshold = upward_max_threshold
        self.downward_min_threshold = downward_min_threshold
        self.downward_max_threshold = downward_max_threshold
        
        self.min_transformation_quality = min_transformation_quality
        self.min_metadata_consistency = min_metadata_consistency
        self.min_technical_preservation = min_technical_preservation
        
        self.min_diversity_threshold = min_diversity_threshold
        self.max_collapse_risk = max_collapse_risk
        self.max_length_ratio = max_length_ratio
        
        # Initialize components
        self._load_validation_rules()
        self._initialize_text_encoder()
        
        # Experience level mappings for metadata validation
        self.experience_level_mappings = {
            'junior': ['entry', 'junior', 'associate', 'beginner', '0-2 years'],
            'mid': ['mid', 'intermediate', 'experienced', '2-5 years', '3-7 years'],
            'senior': ['senior', 'lead', 'principal', 'expert', '5+ years', '7+ years'],
            'executive': ['director', 'vp', 'cto', 'head', 'chief', '10+ years']
        }
        
        # Skill proficiency mappings
        self.skill_proficiency_mappings = {
            'beginner': ['basic', 'beginner', 'learning', 'familiar'],
            'intermediate': ['intermediate', 'proficient', 'experienced', 'solid'],
            'advanced': ['advanced', 'expert', 'mastery', 'deep', 'extensive'],
            'expert': ['expert', 'mastery', 'thought leader', 'architect', 'guru']
        }

    def _initialize_text_encoder(self):
        """Initialize text encoder for embedding validation"""
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Text encoder initialized successfully")
        except ImportError:
            self.text_encoder = None
            logger.warning("SentenceTransformer not available - embedding validation disabled")

    def _load_validation_rules(self):
        """Load enhanced validation rules with technical term handling"""
        # Load core concepts from CS database
        self.core_concepts = self._load_core_concepts_from_cs_database()
        
        # Enhanced technical terms with single-character handling
        self.single_char_technical_terms = {
            'c', 'r', 'go', 'f#', 'c#', 'c++', 'sql', 'xml', 'css', 'php'
        }
        
        # Technical term patterns for preservation
        self.technical_term_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\.(js|py|java|cpp|cs|php|rb|go|rs)\b',  # File extensions
            r'\b\d+\.\d+\b',  # Version numbers
            r'\b[a-zA-Z]+\d+\b',  # Alphanumeric terms (e.g., Python3, HTML5)
        ]
        
        # Forbidden transformations that break semantic meaning
        self.forbidden_transformations = [
            ('python', 'java'),
            ('frontend', 'backend'),
            ('mobile', 'web'),
            ('data science', 'cybersecurity'),
            ('developer', 'manager'),
            ('engineer', 'designer'),
            ('analyst', 'developer')
        ]
        
        # Metadata consistency patterns
        self.experience_indicators = {
            'junior': ['junior', 'entry', 'associate', 'beginner', 'new', 'starting'],
            'senior': ['senior', 'lead', 'principal', 'expert', 'advanced', 'seasoned']
        }

    def _load_core_concepts_from_cs_database(self) -> Dict[str, List[str]]:
        """Load core concepts from CS skills database"""
        import os
        import json

        try:
            dataset_dir = 'dataset'
            cs_skills_file = os.path.join(dataset_dir, 'cs_skills.json')

            with open(cs_skills_file, 'r', encoding='utf-8') as f:
                cs_skills = json.load(f)

            core_concepts = {}

            # Programming languages
            if 'programming_languages' in cs_skills:
                core_concepts['programming_languages'] = [
                    lang.lower() for lang in cs_skills['programming_languages'] if lang
                ]

            # Technologies
            technologies = []
            tech_categories = ['frameworks', 'web_development_frameworks', 'cloud_platforms',
                               'devops_tools', 'containerization', 'databases']
            for category in tech_categories:
                if category in cs_skills:
                    for item in cs_skills[category]:
                        if item:
                            if 'Amazon Web Services (AWS)' in item:
                                technologies.append('aws')
                            elif 'Google Cloud Platform (GCP)' in item:
                                technologies.append('gcp')
                            elif 'Microsoft Azure' in item:
                                technologies.append('azure')
                            else:
                                technologies.append(item.lower())

            core_concepts['technologies'] = technologies

            # Methodologies
            methodologies = ['agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd',
                             'microservices', 'rest', 'graphql', 'api']
            core_concepts['methodologies'] = methodologies

            # Domains
            domains = ['web development', 'mobile development', 'data science',
                       'machine learning', 'cybersecurity', 'cloud computing',
                       'backend development', 'frontend development', 'devops']
            core_concepts['domains'] = domains

            return core_concepts

        except Exception as e:
            logger.warning(f"Failed to load CS skills database: {e}")
            # Fallback to basic concepts
            return {
                'programming_languages': [
                    'python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust',
                    'typescript', 'php', 'ruby', 'swift', 'kotlin', 'c', 'r'
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

    def validate_transformation_with_metadata(self,
                                              original: Dict[str, Any],
                                              transformed: Dict[str, Any],
                                              transformation_type: str,
                                              experience_level_mapping: Optional[Dict] = None) -> ValidationResult:
        """
        Comprehensive validation with metadata consistency checking.
        
        Args:
            original: Original resume/job data
            transformed: Transformed resume/job data
            transformation_type: 'upward' or 'downward'
            experience_level_mapping: Optional mapping for experience levels
            
        Returns:
            ValidationResult with comprehensive metrics and recommendations
        """
        quality_gates_passed = []
        quality_gates_failed = []
        recommendations = []
        failure_reasons = []
        
        try:
            # Extract text content
            original_text = self._extract_text_content(original)
            transformed_text = self._extract_text_content(transformed)
            
            # 1. Semantic similarity validation
            semantic_score = self._calculate_enhanced_similarity(original_text, transformed_text)
            min_threshold, max_threshold = self._get_context_aware_thresholds(transformation_type)
            
            if min_threshold <= semantic_score <= max_threshold:
                quality_gates_passed.append("semantic_similarity")
            else:
                quality_gates_failed.append("semantic_similarity")
                if semantic_score < min_threshold:
                    failure_reasons.append(f"Semantic similarity too low: {semantic_score:.3f} < {min_threshold}")
                    recommendations.append("Increase content preservation in transformation")
                else:
                    failure_reasons.append(f"Semantic similarity too high (collapse risk): {semantic_score:.3f} > {max_threshold}")
                    recommendations.append("Increase transformation diversity")
            
            # 2. Embedding distance validation
            embedding_similarity = 0.0
            if self.text_encoder:
                is_embedding_valid, embedding_similarity = self._validate_embedding_distance(
                    original_text, transformed_text, transformation_type
                )
                if is_embedding_valid:
                    quality_gates_passed.append("embedding_distance")
                else:
                    quality_gates_failed.append("embedding_distance")
                    failure_reasons.append(f"Embedding distance validation failed: {embedding_similarity:.3f}")
                    recommendations.append("Adjust transformation to achieve better embedding separation")
            
            # 3. Technical term preservation
            preservation_report = self.validate_technical_term_preservation(
                original_text, transformed_text, self._get_technical_terms_set()
            )
            
            if preservation_report.is_preserved:
                quality_gates_passed.append("technical_preservation")
            else:
                quality_gates_failed.append("technical_preservation")
                failure_reasons.extend([f"Technical term corruption: {term}" for term in preservation_report.corrupted_terms])
                recommendations.extend(preservation_report.recommendations)
            
            # 4. Metadata consistency validation
            consistency_report = self._validate_metadata_consistency(
                original, transformed, transformation_type, experience_level_mapping
            )
            
            if consistency_report.consistency_score >= self.min_metadata_consistency:
                quality_gates_passed.append("metadata_consistency")
            else:
                quality_gates_failed.append("metadata_consistency")
                failure_reasons.extend(consistency_report.misalignments)
                recommendations.extend(consistency_report.recommendations)
            
            # 5. Length ratio validation
            length_ratio = len(transformed_text) / max(len(original_text), 1)
            if length_ratio <= self.max_length_ratio:
                quality_gates_passed.append("length_ratio")
            else:
                quality_gates_failed.append("length_ratio")
                failure_reasons.append(f"Length ratio too high: {length_ratio:.2f} > {self.max_length_ratio}")
                recommendations.append("Reduce transformation verbosity")
            
            # 6. Forbidden transformation check
            if not self._contains_forbidden_transformation(original_text, transformed_text):
                quality_gates_passed.append("forbidden_transformations")
            else:
                quality_gates_failed.append("forbidden_transformations")
                failure_reasons.append("Contains forbidden semantic transformation")
                recommendations.append("Avoid changing core technical concepts")
            
            # Calculate overall quality score
            overall_quality_score = self._calculate_overall_quality_score(
                semantic_score, embedding_similarity, preservation_report.preservation_ratio,
                consistency_report.consistency_score, len(quality_gates_passed), len(quality_gates_failed)
            )
            
            # Determine overall validity
            is_valid = (
                len(quality_gates_failed) == 0 and
                overall_quality_score >= self.min_transformation_quality
            )
            
            return ValidationResult(
                is_valid=is_valid,
                semantic_score=semantic_score,
                embedding_similarity=embedding_similarity,
                metadata_consistency=consistency_report.consistency_score,
                technical_preservation=preservation_report.preservation_ratio,
                quality_gates_passed=quality_gates_passed,
                quality_gates_failed=quality_gates_failed,
                recommendations=recommendations,
                failure_reasons=failure_reasons,
                transformation_type=transformation_type,
                overall_quality_score=overall_quality_score
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                semantic_score=0.0,
                embedding_similarity=0.0,
                metadata_consistency=0.0,
                technical_preservation=0.0,
                quality_gates_passed=[],
                quality_gates_failed=["validation_error"],
                recommendations=["Fix validation system error"],
                failure_reasons=[f"Validation system error: {str(e)}"],
                transformation_type=transformation_type,
                overall_quality_score=0.0
            )

    def validate_embedding_diversity(self,
                                     transformations: List[Dict[str, Any]],
                                     min_diversity_threshold: float = None) -> DiversityReport:
        """
        Validate embedding diversity across multiple transformations to prevent collapse.
        
        Args:
            transformations: List of transformation dictionaries with 'original' and 'transformed' keys
            min_diversity_threshold: Override default diversity threshold
            
        Returns:
            DiversityReport with diversity metrics and recommendations
        """
        if min_diversity_threshold is None:
            min_diversity_threshold = self.min_diversity_threshold
            
        recommendations = []
        
        if not self.text_encoder:
            logger.warning("Text encoder not available - diversity validation disabled")
            return DiversityReport(
                min_pairwise_distance=0.0,
                mean_pairwise_distance=0.0,
                diversity_index=0.0,
                collapse_risk_score=1.0,
                recommendations=["Enable text encoder for diversity validation"],
                is_diverse=False
            )
        
        try:
            # Extract embeddings for all transformations
            embeddings = []
            for transformation in transformations:
                if 'transformed' in transformation:
                    text = self._extract_text_content(transformation['transformed'])
                    embedding = self.text_encoder.encode(text, convert_to_tensor=True)
                    embeddings.append(embedding.cpu().numpy())
            
            if len(embeddings) < 2:
                return DiversityReport(
                    min_pairwise_distance=1.0,
                    mean_pairwise_distance=1.0,
                    diversity_index=1.0,
                    collapse_risk_score=0.0,
                    recommendations=[],
                    is_diverse=True
                )
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Cosine distance = 1 - cosine similarity
                    cosine_sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    distance = 1 - cosine_sim
                    distances.append(distance)
            
            # Calculate diversity metrics
            min_distance = min(distances)
            mean_distance = np.mean(distances)
            
            # Shannon diversity equivalent for embeddings
            # Higher diversity index means more diverse embeddings
            diversity_index = mean_distance
            
            # Collapse risk score (0 = no risk, 1 = high risk)
            collapse_risk_score = max(0, (min_diversity_threshold - min_distance) / min_diversity_threshold)
            
            # Generate recommendations
            if min_distance < min_diversity_threshold:
                recommendations.append(f"Increase transformation diversity - minimum distance {min_distance:.3f} below threshold {min_diversity_threshold}")
            
            if collapse_risk_score > self.max_collapse_risk:
                recommendations.append("High embedding collapse risk detected - review transformation strategies")
            
            if mean_distance < 0.2:
                recommendations.append("Low mean embedding distance - transformations may be too similar")
            
            is_diverse = (
                min_distance >= min_diversity_threshold and
                collapse_risk_score <= self.max_collapse_risk
            )
            
            return DiversityReport(
                min_pairwise_distance=min_distance,
                mean_pairwise_distance=mean_distance,
                diversity_index=diversity_index,
                collapse_risk_score=collapse_risk_score,
                recommendations=recommendations,
                is_diverse=is_diverse
            )
            
        except Exception as e:
            logger.error(f"Diversity validation error: {e}")
            return DiversityReport(
                min_pairwise_distance=0.0,
                mean_pairwise_distance=0.0,
                diversity_index=0.0,
                collapse_risk_score=1.0,
                recommendations=[f"Diversity validation error: {str(e)}"],
                is_diverse=False
            )

    def validate_technical_term_preservation(self,
                                             original_text: str,
                                             transformed_text: str,
                                             technical_terms: Set[str]) -> PreservationReport:
        """
        Validate technical term preservation with special handling for single-character terms.
        
        Args:
            original_text: Original text content
            transformed_text: Transformed text content
            technical_terms: Set of technical terms to check for preservation
            
        Returns:
            PreservationReport with preservation metrics and recommendations
        """
        preserved_terms = set()
        corrupted_terms = set()
        single_char_handled = 0
        recommendations = []
        
        try:
            # Normalize texts for comparison
            original_lower = original_text.lower()
            transformed_lower = transformed_text.lower()
            
            # Find technical terms in original text
            original_terms_found = set()
            for term in technical_terms:
                if self._find_technical_term(term, original_lower):
                    original_terms_found.add(term)
            
            # Check preservation in transformed text
            for term in original_terms_found:
                if self._find_technical_term(term, transformed_lower):
                    preserved_terms.add(term)
                    
                    # Special handling for single-character terms
                    if len(term) == 1 and term in self.single_char_technical_terms:
                        single_char_handled += 1
                else:
                    corrupted_terms.add(term)
                    
                    # Check for common corruption patterns
                    corruption_type = self._detect_corruption_type(term, original_text, transformed_text)
                    if corruption_type:
                        recommendations.append(f"Fix {corruption_type} corruption for term '{term}'")
            
            # Calculate preservation ratio
            preservation_ratio = len(preserved_terms) / max(len(original_terms_found), 1)
            
            # Generate recommendations
            if preservation_ratio < self.min_technical_preservation:
                recommendations.append(f"Technical term preservation too low: {preservation_ratio:.2f}")
                recommendations.append("Implement better technical term protection during transformation")
            
            if corrupted_terms:
                recommendations.append(f"Preserve technical terms: {', '.join(sorted(corrupted_terms))}")
            
            is_preserved = preservation_ratio >= self.min_technical_preservation
            
            return PreservationReport(
                preserved_terms=preserved_terms,
                corrupted_terms=corrupted_terms,
                preservation_ratio=preservation_ratio,
                single_char_terms_handled=single_char_handled,
                recommendations=recommendations,
                is_preserved=is_preserved
            )
            
        except Exception as e:
            logger.error(f"Technical term preservation validation error: {e}")
            return PreservationReport(
                preserved_terms=set(),
                corrupted_terms=technical_terms,
                preservation_ratio=0.0,
                single_char_terms_handled=0,
                recommendations=[f"Technical term validation error: {str(e)}"],
                is_preserved=False
            )

    def _find_technical_term(self, term: str, text: str) -> bool:
        """
        Find technical term in text with special handling for single-character terms.
        
        Args:
            term: Technical term to find
            text: Text to search in
            
        Returns:
            bool: True if term is found with proper context
        """
        if len(term) == 1 and term in self.single_char_technical_terms:
            # Special handling for single-character technical terms like 'C', 'R'
            # Look for word boundaries and avoid false positives
            patterns = [
                rf'\b{re.escape(term)}\b',  # Standalone
                rf'\b{re.escape(term)}\+\+',  # C++
                rf'\b{re.escape(term)}#',  # C#, F#
                rf'{re.escape(term)}\s+programming',  # "C programming"
                rf'{re.escape(term)}\s+language',  # "C language"
            ]
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            return False
        elif '+' in term or '#' in term:
            # Special handling for terms with special characters like C++, C#
            # Use literal matching for these terms
            return term.lower() in text.lower()
        else:
            # Regular term matching with word boundaries
            pattern = rf'\b{re.escape(term)}\b'
            return bool(re.search(pattern, text, re.IGNORECASE))

    def _detect_corruption_type(self, term: str, original: str, transformed: str) -> Optional[str]:
        """Detect the type of corruption that occurred to a technical term"""
        # Check for punctuation corruption
        if f"{term}," in original.lower() and f"{term}," not in transformed.lower():
            return "punctuation"
        
        # Check for case corruption
        if term.upper() in original and term.upper() not in transformed:
            return "case"
        
        # Check for word boundary corruption
        if f" {term} " in original.lower() and f" {term} " not in transformed.lower():
            return "word_boundary"
        
        return "unknown"

    def _get_context_aware_thresholds(self, transformation_type: str) -> Tuple[float, float]:
        """Get similarity thresholds based on transformation type"""
        if transformation_type == 'upward':
            return self.upward_min_threshold, self.upward_max_threshold
        else:  # downward
            return self.downward_min_threshold, self.downward_max_threshold

    def _calculate_enhanced_similarity(self, text1: str, text2: str) -> float:
        """Calculate enhanced semantic similarity with improved tokenization"""
        def tokenize(text):
            # Enhanced tokenization that preserves technical terms
            text = re.sub(r"[^\w\s\+\#]", " ", text.lower())  # Preserve + and # for C++, C#
            text = re.sub(r"\s+", " ", text.strip())
            words = set(text.split())
            return words

        words1 = tokenize(text1)
        words2 = tokenize(text2)

        if not words1 and not words2:
            return 1.0

        # Calculate multiple similarity metrics
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 1.0

        # Overlap coefficient
        overlap = len(intersection) / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 1.0

        # Preservation ratio
        preservation = len(intersection) / len(words1) if words1 else 1.0

        # Sequence similarity
        tokens1 = re.sub(r"[^\w\s\+\#]", " ", text1.lower()).split()
        tokens2 = re.sub(r"[^\w\s\+\#]", " ", text2.lower()).split()
        matcher = SequenceMatcher(None, tokens1, tokens2)
        sequence = matcher.ratio()

        # Weighted combination emphasizing preservation and overlap
        combined = (0.3 * jaccard + 0.4 * overlap + 0.2 * preservation + 0.1 * sequence)

        return combined

    def _validate_embedding_distance(self, 
                                     original_text: str, 
                                     transformed_text: str,
                                     transformation_type: str) -> Tuple[bool, float]:
        """Validate embedding distance to prevent collapse"""
        if not self.text_encoder:
            return True, 0.5  # Default valid if encoder unavailable

        try:
            # Generate embeddings
            original_emb = self.text_encoder.encode(original_text, convert_to_tensor=True)
            transformed_emb = self.text_encoder.encode(transformed_text, convert_to_tensor=True)

            # Calculate cosine similarity
            import torch.nn.functional as F
            cosine_sim = F.cosine_similarity(original_emb.unsqueeze(0), 
                                           transformed_emb.unsqueeze(0)).item()

            # Get thresholds
            min_threshold, max_threshold = self._get_context_aware_thresholds(transformation_type)

            # Validate distance
            is_valid = min_threshold <= cosine_sim <= max_threshold

            return is_valid, cosine_sim

        except Exception as e:
            logger.error(f"Embedding distance validation error: {e}")
            return False, 0.0

    def _validate_metadata_consistency(self,
                                       original: Dict[str, Any],
                                       transformed: Dict[str, Any],
                                       transformation_type: str,
                                       experience_level_mapping: Optional[Dict] = None) -> ConsistencyReport:
        """Validate metadata consistency between transformed content and experience levels"""
        misalignments = []
        recommendations = []
        
        try:
            # Extract metadata
            original_exp_level = self._extract_experience_level(original)
            transformed_exp_level = self._extract_experience_level(transformed)
            
            # Extract skill proficiency indicators
            original_skills = self._extract_skill_proficiency_indicators(original)
            transformed_skills = self._extract_skill_proficiency_indicators(transformed)
            
            # Extract job title indicators
            original_title = self._extract_job_title_level(original)
            transformed_title = self._extract_job_title_level(transformed)
            
            # 1. Experience level alignment
            experience_aligned = self._check_experience_level_alignment(
                original_exp_level, transformed_exp_level, transformation_type
            )
            if not experience_aligned:
                misalignments.append(f"Experience level misalignment: {original_exp_level} -> {transformed_exp_level}")
                recommendations.append("Ensure experience level metadata matches transformation direction")
            
            # 2. Skill proficiency alignment
            skill_aligned = self._check_skill_proficiency_alignment(
                original_skills, transformed_skills, transformation_type
            )
            if not skill_aligned:
                misalignments.append("Skill proficiency levels don't match transformation direction")
                recommendations.append("Update skill proficiency indicators to match experience level changes")
            
            # 3. Job title coherence
            title_coherent = self._check_job_title_coherence(
                original_title, transformed_title, transformation_type
            )
            if not title_coherent:
                misalignments.append("Job title doesn't align with experience level transformation")
                recommendations.append("Ensure job title reflects appropriate seniority level")
            
            # Calculate consistency score
            consistency_score = sum([experience_aligned, skill_aligned, title_coherent]) / 3.0
            
            return ConsistencyReport(
                experience_level_aligned=experience_aligned,
                skill_proficiency_aligned=skill_aligned,
                job_title_coherent=title_coherent,
                consistency_score=consistency_score,
                misalignments=misalignments,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Metadata consistency validation error: {e}")
            return ConsistencyReport(
                experience_level_aligned=False,
                skill_proficiency_aligned=False,
                job_title_coherent=False,
                consistency_score=0.0,
                misalignments=[f"Metadata validation error: {str(e)}"],
                recommendations=["Fix metadata validation system"]
            )

    def _extract_experience_level(self, data: Dict[str, Any]) -> str:
        """Extract experience level from resume/job data"""
        # Check explicit experience_level field
        if 'experience_level' in data:
            return str(data['experience_level']).lower()
        
        # Check metadata
        if 'metadata' in data and 'experience_level' in data['metadata']:
            return str(data['metadata']['experience_level']).lower()
        
        # Infer from text content
        text = self._extract_text_content(data).lower()
        
        # Check for explicit indicators
        for level, indicators in self.experience_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    return level
        
        return 'unknown'

    def _extract_skill_proficiency_indicators(self, data: Dict[str, Any]) -> List[str]:
        """Extract skill proficiency indicators from data"""
        indicators = []
        text = self._extract_text_content(data).lower()
        
        # Look for proficiency keywords
        proficiency_patterns = [
            r'\b(expert|mastery|advanced|proficient|experienced|beginner|basic)\s+(?:in|with|at)\b',
            r'\b(deep|extensive|solid|basic)\s+(?:knowledge|experience|understanding)\b',
            r'\b\d+\+?\s*years?\s+(?:of\s+)?experience\b'
        ]
        
        for pattern in proficiency_patterns:
            matches = re.findall(pattern, text)
            indicators.extend(matches)
        
        return indicators

    def _extract_job_title_level(self, data: Dict[str, Any]) -> str:
        """Extract seniority level from job title"""
        # Check job_title field
        title = ""
        if 'job_title' in data:
            title = str(data['job_title']).lower()
        elif 'title' in data:
            title = str(data['title']).lower()
        
        # Determine seniority level
        if any(word in title for word in ['senior', 'lead', 'principal', 'architect', 'staff']):
            return 'senior'
        elif any(word in title for word in ['junior', 'entry', 'associate', 'intern']):
            return 'junior'
        else:
            return 'mid'

    def _check_experience_level_alignment(self, original: str, transformed: str, transformation_type: str) -> bool:
        """Check if experience level change aligns with transformation type"""
        if original == 'unknown' or transformed == 'unknown':
            return True  # Can't validate unknown levels
        
        level_hierarchy = ['junior', 'mid', 'senior', 'executive']
        
        try:
            orig_idx = level_hierarchy.index(original)
            trans_idx = level_hierarchy.index(transformed)
            
            if transformation_type == 'upward':
                return trans_idx >= orig_idx  # Should stay same or increase
            else:  # downward
                return trans_idx <= orig_idx  # Should stay same or decrease
        except ValueError:
            return True  # Unknown levels, assume valid

    def _check_skill_proficiency_alignment(self, original: List[str], transformed: List[str], transformation_type: str) -> bool:
        """Check if skill proficiency changes align with transformation type"""
        if not original and not transformed:
            return True
        
        # Simple heuristic: count advanced vs basic terms
        advanced_terms = ['expert', 'mastery', 'advanced', 'deep', 'extensive']
        basic_terms = ['beginner', 'basic', 'learning', 'familiar']
        
        orig_advanced = sum(1 for term in original if any(adv in term for adv in advanced_terms))
        orig_basic = sum(1 for term in original if any(bas in term for bas in basic_terms))
        
        trans_advanced = sum(1 for term in transformed if any(adv in term for adv in advanced_terms))
        trans_basic = sum(1 for term in transformed if any(bas in term for bas in basic_terms))
        
        if transformation_type == 'upward':
            # Should have more advanced terms or same
            return trans_advanced >= orig_advanced
        else:  # downward
            # Should have more basic terms or same advanced
            return trans_basic >= orig_basic or trans_advanced <= orig_advanced

    def _check_job_title_coherence(self, original: str, transformed: str, transformation_type: str) -> bool:
        """Check if job title seniority aligns with transformation"""
        if original == transformed:
            return True  # Same title is always coherent
        
        level_hierarchy = ['junior', 'mid', 'senior']
        
        try:
            orig_idx = level_hierarchy.index(original)
            trans_idx = level_hierarchy.index(transformed)
            
            if transformation_type == 'upward':
                return trans_idx >= orig_idx
            else:  # downward
                return trans_idx <= orig_idx
        except ValueError:
            return True  # Unknown levels, assume coherent

    def _extract_text_content(self, data: Dict[str, Any]) -> str:
        """Extract all text content from resume/job data"""
        text_parts = []

        # Extract experience text
        if 'experience' in data:
            exp = data['experience']
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

        # Extract skills text
        if 'skills' in data:
            skills = data['skills']
            if isinstance(skills, list):
                for skill in skills:
                    if isinstance(skill, dict) and 'name' in skill:
                        text_parts.append(skill['name'])
                    elif isinstance(skill, str):
                        text_parts.append(skill)

        # Extract summary text
        if 'summary' in data:
            summary = data['summary']
            if isinstance(summary, str):
                text_parts.append(summary)
            elif isinstance(summary, dict) and 'text' in summary:
                text_parts.append(summary['text'])

        # Extract job title
        if 'job_title' in data:
            text_parts.append(str(data['job_title']))
        elif 'title' in data:
            text_parts.append(str(data['title']))

        return ' '.join(text_parts)

    def _get_technical_terms_set(self) -> Set[str]:
        """Get comprehensive set of technical terms for preservation"""
        technical_terms = set()

        # Add from core concepts
        for concept_list in self.core_concepts.values():
            technical_terms.update(concept_list)

        # Add single-character technical terms
        technical_terms.update(self.single_char_technical_terms)

        # Add common technical acronyms
        common_acronyms = {
            'sql', 'api', 'aws', 'gcp', 'html', 'css', 'xml', 'json', 
            'rest', 'crud', 'http', 'https', 'tcp', 'udp', 'ssh', 'ftp'
        }
        technical_terms.update(common_acronyms)

        return technical_terms

    def _contains_forbidden_transformation(self, original: str, transformed: str) -> bool:
        """Check if transformation contains forbidden semantic changes"""
        original_lower = original.lower()
        transformed_lower = transformed.lower()

        for forbidden_from, forbidden_to in self.forbidden_transformations:
            if (forbidden_from in original_lower and
                forbidden_to in transformed_lower and
                forbidden_from not in transformed_lower):
                return True

        return False

    def _calculate_overall_quality_score(self,
                                         semantic_score: float,
                                         embedding_similarity: float,
                                         technical_preservation: float,
                                         metadata_consistency: float,
                                         gates_passed: int,
                                         gates_failed: int) -> float:
        """Calculate overall transformation quality score"""
        # Weighted combination of quality metrics
        quality_score = (
            0.3 * semantic_score +
            0.2 * embedding_similarity +
            0.2 * technical_preservation +
            0.2 * metadata_consistency +
            0.1 * (gates_passed / max(gates_passed + gates_failed, 1))
        )
        
        return min(1.0, max(0.0, quality_score))