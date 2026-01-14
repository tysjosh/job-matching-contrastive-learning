"""
Career-Aware Paraphraser: Intelligent paraphrasing for professional content

This module provides specialized paraphrasing capabilities that maintain semantic coherence
while increasing linguistic diversity in career-related text transformations.
"""

import logging
import re
import random
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CareerLevel(Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    EXECUTIVE = "executive"


@dataclass
class ParaphrasingResult:
    """Result of paraphrasing operation with quality metrics"""
    original_text: str
    paraphrased_text: str
    diversity_score: float
    technical_terms_preserved: int
    semantic_similarity: float
    success: bool
    fallback_used: bool = False


class CareerAwareParaphraser:
    """
    Specialized paraphraser for career content that maintains professional context
    while increasing linguistic diversity to prevent embedding collapse.
    """

    def __init__(self, 
                 preserve_technical_terms: bool = True,
                 min_diversity_score: float = 0.3,
                 max_semantic_drift: float = 0.8):
        """
        Initialize career-aware paraphraser.
        
        Args:
            preserve_technical_terms: Whether to preserve technical terms
            min_diversity_score: Minimum diversity score to achieve
            max_semantic_drift: Maximum allowed semantic drift
        """
        self.preserve_technical_terms = preserve_technical_terms
        self.min_diversity_score = min_diversity_score
        self.max_semantic_drift = max_semantic_drift
        
        # Load paraphrasing rules and patterns
        self._load_paraphrasing_patterns()
        self._load_technical_terms()
        self._load_career_level_vocabularies()
        
        # Initialize diversity tracking
        self.used_patterns = set()
        self.pattern_usage_count = {}

    def _load_paraphrasing_patterns(self):
        """Load paraphrasing patterns for different text types"""
        
        # Action verb transformations (maintaining professional tone)
        self.action_verb_patterns = {
            # Leadership and management
            'led': ['spearheaded', 'directed', 'guided', 'orchestrated', 'championed'],
            'managed': ['oversaw', 'supervised', 'coordinated', 'administered', 'steered'],
            'developed': ['created', 'built', 'designed', 'engineered', 'crafted'],
            'implemented': ['deployed', 'executed', 'established', 'launched', 'rolled out'],
            'improved': ['enhanced', 'optimized', 'refined', 'strengthened', 'upgraded'],
            'collaborated': ['partnered', 'worked closely with', 'teamed up with', 'coordinated with'],
            
            # Technical actions
            'coded': ['programmed', 'developed software for', 'wrote code for', 'implemented'],
            'designed': ['architected', 'planned', 'conceptualized', 'structured', 'modeled'],
            'tested': ['validated', 'verified', 'quality assured', 'debugged', 'evaluated'],
            'deployed': ['released', 'launched', 'rolled out', 'implemented', 'delivered'],
            
            # Analysis and problem-solving
            'analyzed': ['examined', 'evaluated', 'assessed', 'investigated', 'studied'],
            'solved': ['resolved', 'addressed', 'tackled', 'fixed', 'remediated'],
            'researched': ['investigated', 'explored', 'studied', 'examined', 'analyzed'],
        }
        
        # Sentence structure patterns
        self.sentence_patterns = {
            # Achievement patterns
            'achieved_X_by_Y': [
                'Delivered {achievement} through {method}',
                'Accomplished {achievement} by {method}',
                'Realized {achievement} via {method}',
                'Attained {achievement} using {method}'
            ],
            'responsible_for_X': [
                'Accountable for {responsibility}',
                'Oversaw {responsibility}',
                'Managed {responsibility}',
                'Led {responsibility}'
            ],
            'worked_on_X': [
                'Contributed to {project}',
                'Participated in {project}',
                'Engaged in {project}',
                'Involved in {project}'
            ]
        }
        
        # Professional context phrases
        self.context_phrases = {
            'team_context': [
                'cross-functional team', 'collaborative environment', 'team setting',
                'multi-disciplinary group', 'project team', 'development team'
            ],
            'project_context': [
                'strategic initiative', 'key project', 'critical program',
                'major undertaking', 'important effort', 'significant project'
            ],
            'impact_context': [
                'resulting in', 'leading to', 'which contributed to',
                'thereby achieving', 'ultimately delivering', 'consequently producing'
            ]
        }

    def _load_technical_terms(self):
        """Load technical terms that should be preserved during paraphrasing"""
        
        # Programming languages (case-sensitive preservation)
        self.programming_languages = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'r',
            'go', 'rust', 'swift', 'kotlin', 'scala', 'php', 'ruby', 'perl',
            'matlab', 'sql', 'html', 'css', 'xml', 'json'
        }
        
        # Frameworks and technologies
        self.frameworks_technologies = {
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'spring', 'hibernate', 'tensorflow', 'pytorch', 'scikit-learn',
            'pandas', 'numpy', 'docker', 'kubernetes', 'jenkins', 'git',
            'aws', 'azure', 'gcp', 'mongodb', 'postgresql', 'mysql', 'redis'
        }
        
        # Methodologies and practices
        self.methodologies = {
            'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd',
            'microservices', 'rest', 'graphql', 'api', 'mvc', 'mvp', 'solid'
        }
        
        # Combined technical terms set
        self.technical_terms = (
            self.programming_languages | 
            self.frameworks_technologies | 
            self.methodologies
        )
        
        # Technical term patterns for regex matching
        self.technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms (API, REST, etc.)
            r'\b\w+\.(js|py|java|cpp|cs|php|rb|go|rs)\b',  # File extensions
            r'\b\d+\.\d+\b',  # Version numbers
            r'\b[a-zA-Z]+\d+\b',  # Alphanumeric terms (HTML5, Python3)
        ]

    def _load_career_level_vocabularies(self):
        """Load vocabulary appropriate for different career levels"""
        
        self.career_vocabularies = {
            CareerLevel.ENTRY: {
                'action_modifiers': ['assisted in', 'contributed to', 'participated in', 'supported'],
                'responsibility_level': ['basic', 'fundamental', 'core', 'essential'],
                'learning_indicators': ['learned', 'gained experience in', 'developed skills in', 'familiarized with'],
                'scope_indicators': ['individual tasks', 'specific components', 'assigned projects', 'guided work']
            },
            CareerLevel.MID: {
                'action_modifiers': ['managed', 'coordinated', 'executed', 'delivered'],
                'responsibility_level': ['significant', 'important', 'key', 'critical'],
                'ownership_indicators': ['owned', 'responsible for', 'accountable for', 'drove'],
                'scope_indicators': ['project components', 'team deliverables', 'system modules', 'feature sets']
            },
            CareerLevel.SENIOR: {
                'action_modifiers': ['led', 'spearheaded', 'architected', 'strategized'],
                'responsibility_level': ['strategic', 'enterprise-level', 'organization-wide', 'mission-critical'],
                'leadership_indicators': ['mentored', 'guided teams', 'established standards', 'drove adoption'],
                'scope_indicators': ['cross-functional initiatives', 'organizational systems', 'strategic programs', 'enterprise solutions']
            },
            CareerLevel.EXECUTIVE: {
                'action_modifiers': ['orchestrated', 'championed', 'transformed', 'revolutionized'],
                'responsibility_level': ['transformational', 'visionary', 'industry-leading', 'groundbreaking'],
                'leadership_indicators': ['built organizations', 'established culture', 'drove transformation', 'shaped strategy'],
                'scope_indicators': ['organizational transformation', 'market expansion', 'strategic vision', 'industry innovation']
            }
        }

    def paraphrase_experience_text(self, 
                                 text: str, 
                                 career_level: str,
                                 preserve_technical: bool = None,
                                 diversity_target: float = None) -> ParaphrasingResult:
        """
        Paraphrase experience text while maintaining career-appropriate language.
        
        Args:
            text: Original experience text
            career_level: Target career level (entry, mid, senior, executive)
            preserve_technical: Override for technical term preservation
            diversity_target: Target diversity score
            
        Returns:
            ParaphrasingResult with paraphrased text and quality metrics
        """
        if preserve_technical is None:
            preserve_technical = self.preserve_technical_terms
        if diversity_target is None:
            diversity_target = self.min_diversity_score
            
        try:
            # Step 1: Extract and preserve technical terms
            technical_terms = self._extract_technical_terms(text) if preserve_technical else set()
            
            # Step 2: Apply career-level appropriate paraphrasing
            level_enum = self._parse_career_level(career_level)
            paraphrased_text = self._apply_career_aware_paraphrasing(text, level_enum)
            
            # Step 3: Restore technical terms
            if preserve_technical and technical_terms:
                paraphrased_text = self._restore_technical_terms(paraphrased_text, technical_terms, text)
            
            # Step 4: Calculate quality metrics
            diversity_score = self._calculate_diversity_score(text, paraphrased_text)
            semantic_similarity = self._calculate_semantic_similarity(text, paraphrased_text)
            
            # Step 5: Validate result
            success = (
                diversity_score >= diversity_target and
                semantic_similarity <= self.max_semantic_drift and
                len(paraphrased_text.strip()) > 0
            )
            
            # Step 6: Fallback if quality is insufficient
            if not success and diversity_score < diversity_target:
                logger.debug(f"Applying fallback paraphrasing - diversity too low: {diversity_score:.3f}")
                paraphrased_text = self._apply_fallback_paraphrasing(text, level_enum)
                diversity_score = self._calculate_diversity_score(text, paraphrased_text)
                success = diversity_score >= (diversity_target * 0.7)  # Relaxed threshold
            
            return ParaphrasingResult(
                original_text=text,
                paraphrased_text=paraphrased_text,
                diversity_score=diversity_score,
                technical_terms_preserved=len(technical_terms),
                semantic_similarity=semantic_similarity,
                success=success,
                fallback_used=not success
            )
            
        except Exception as e:
            logger.error(f"Paraphrasing failed: {e}")
            return ParaphrasingResult(
                original_text=text,
                paraphrased_text=text,  # Return original on failure
                diversity_score=0.0,
                technical_terms_preserved=0,
                semantic_similarity=1.0,
                success=False
            )

    def _extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms from text for preservation"""
        technical_terms = set()
        text_lower = text.lower()
        
        # Extract known technical terms
        for term in self.technical_terms:
            if term in text_lower:
                # Find the actual case-sensitive version in the original text
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                matches = pattern.findall(text)
                technical_terms.update(matches)
        
        # Extract terms matching technical patterns
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text)
            technical_terms.update(matches)
        
        return technical_terms

    def _parse_career_level(self, career_level: str) -> CareerLevel:
        """Parse career level string to enum"""
        level_mapping = {
            'entry': CareerLevel.ENTRY,
            'junior': CareerLevel.ENTRY,
            'mid': CareerLevel.MID,
            'intermediate': CareerLevel.MID,
            'senior': CareerLevel.SENIOR,
            'lead': CareerLevel.SENIOR,
            'principal': CareerLevel.SENIOR,
            'executive': CareerLevel.EXECUTIVE,
            'director': CareerLevel.EXECUTIVE,
            'vp': CareerLevel.EXECUTIVE
        }
        return level_mapping.get(career_level.lower(), CareerLevel.MID)

    def _apply_career_aware_paraphrasing(self, text: str, career_level: CareerLevel) -> str:
        """Apply career-level appropriate paraphrasing"""
        
        # Split text into sentences for better processing
        sentences = self._split_into_sentences(text)
        paraphrased_sentences = []
        
        for sentence in sentences:
            paraphrased_sentence = self._paraphrase_sentence(sentence, career_level)
            paraphrased_sentences.append(paraphrased_sentence)
        
        return ' '.join(paraphrased_sentences)

    def _paraphrase_sentence(self, sentence: str, career_level: CareerLevel) -> str:
        """Paraphrase a single sentence with career-appropriate vocabulary"""
        
        paraphrased = sentence
        
        # Step 1: Replace action verbs with career-appropriate alternatives
        paraphrased = self._replace_action_verbs(paraphrased, career_level)
        
        # Step 2: Enhance with career-level vocabulary
        paraphrased = self._enhance_with_career_vocabulary(paraphrased, career_level)
        
        # Step 3: Apply sentence structure variations
        paraphrased = self._apply_sentence_structure_variations(paraphrased)
        
        # Step 4: Add contextual phrases if appropriate
        paraphrased = self._add_contextual_phrases(paraphrased, career_level)
        
        return paraphrased

    def _replace_action_verbs(self, text: str, career_level: CareerLevel) -> str:
        """Replace action verbs with career-appropriate alternatives"""
        
        for original_verb, alternatives in self.action_verb_patterns.items():
            # Create pattern to match verb in different forms
            pattern = rf'\b{re.escape(original_verb)}(?:ed|ing|s)?\b'
            
            def replace_verb(match):
                matched_verb = match.group(0)
                
                # Select alternative based on career level and diversity
                alternative = self._select_diverse_alternative(alternatives, original_verb)
                
                # Maintain verb tense/form
                if matched_verb.endswith('ed'):
                    return alternative + 'ed' if not alternative.endswith('ed') else alternative
                elif matched_verb.endswith('ing'):
                    return alternative + 'ing' if not alternative.endswith('ing') else alternative
                elif matched_verb.endswith('s'):
                    return alternative + 's' if not alternative.endswith('s') else alternative
                else:
                    return alternative
            
            text = re.sub(pattern, replace_verb, text, flags=re.IGNORECASE)
        
        return text

    def _enhance_with_career_vocabulary(self, text: str, career_level: CareerLevel) -> str:
        """Enhance text with career-level appropriate vocabulary"""
        
        if career_level not in self.career_vocabularies:
            return text
        
        vocab = self.career_vocabularies[career_level]
        
        # Add responsibility level indicators
        if 'responsibility_level' in vocab:
            # Look for opportunities to add level indicators
            responsibility_patterns = [
                r'\b(project|task|initiative|work|responsibility)\b'
            ]
            
            for pattern in responsibility_patterns:
                def add_level_indicator(match):
                    matched_word = match.group(0)
                    level_indicator = random.choice(vocab['responsibility_level'])
                    return f"{level_indicator} {matched_word}"
                
                # Apply sparingly to avoid over-modification
                if random.random() < 0.3:  # 30% chance
                    text = re.sub(pattern, add_level_indicator, text, count=1, flags=re.IGNORECASE)
        
        return text

    def _apply_sentence_structure_variations(self, text: str) -> str:
        """Apply sentence structure variations for diversity"""
        
        # Pattern: "Responsible for X" -> variations
        responsible_pattern = r'Responsible for (.+?)(?:\.|$)'
        match = re.search(responsible_pattern, text, re.IGNORECASE)
        if match:
            responsibility = match.group(1)
            alternatives = self.sentence_patterns['responsible_for_X']
            new_structure = random.choice(alternatives).format(responsibility=responsibility)
            text = re.sub(responsible_pattern, new_structure, text, flags=re.IGNORECASE)
        
        # Pattern: "Worked on X" -> variations
        worked_pattern = r'Worked on (.+?)(?:\.|$)'
        match = re.search(worked_pattern, text, re.IGNORECASE)
        if match:
            project = match.group(1)
            alternatives = self.sentence_patterns['worked_on_X']
            new_structure = random.choice(alternatives).format(project=project)
            text = re.sub(worked_pattern, new_structure, text, flags=re.IGNORECASE)
        
        return text

    def _add_contextual_phrases(self, text: str, career_level: CareerLevel) -> str:
        """Add contextual phrases appropriate for career level"""
        
        # Add team context for collaborative work
        if 'team' in text.lower() and random.random() < 0.4:
            team_contexts = self.context_phrases['team_context']
            context = random.choice(team_contexts)
            text = re.sub(r'\bteam\b', context, text, count=1, flags=re.IGNORECASE)
        
        # Add project context for senior levels
        if career_level in [CareerLevel.SENIOR, CareerLevel.EXECUTIVE]:
            if 'project' in text.lower() and random.random() < 0.3:
                project_contexts = self.context_phrases['project_context']
                context = random.choice(project_contexts)
                text = re.sub(r'\bproject\b', context, text, count=1, flags=re.IGNORECASE)
        
        return text

    def _select_diverse_alternative(self, alternatives: List[str], original_key: str) -> str:
        """Select alternative that maximizes diversity"""
        
        # Track usage to promote diversity
        if original_key not in self.pattern_usage_count:
            self.pattern_usage_count[original_key] = {}
        
        usage_count = self.pattern_usage_count[original_key]
        
        # Prefer less-used alternatives
        available_alternatives = [alt for alt in alternatives if usage_count.get(alt, 0) < 2]
        
        if not available_alternatives:
            # Reset counts if all alternatives have been used
            self.pattern_usage_count[original_key] = {}
            available_alternatives = alternatives
        
        selected = random.choice(available_alternatives)
        usage_count[selected] = usage_count.get(selected, 0) + 1
        
        return selected

    def _apply_fallback_paraphrasing(self, text: str, career_level: CareerLevel) -> str:
        """Apply fallback paraphrasing when primary method fails"""
        
        # Simple word-level replacements for basic diversity
        fallback_replacements = {
            'developed': 'created',
            'implemented': 'deployed',
            'managed': 'oversaw',
            'worked': 'collaborated',
            'created': 'built',
            'designed': 'architected',
            'improved': 'enhanced',
            'analyzed': 'evaluated'
        }
        
        paraphrased = text
        for original, replacement in fallback_replacements.items():
            if original in paraphrased.lower():
                paraphrased = re.sub(rf'\b{original}\b', replacement, paraphrased, count=1, flags=re.IGNORECASE)
                break  # Apply only one replacement to avoid over-modification
        
        return paraphrased

    def _restore_technical_terms(self, paraphrased_text: str, technical_terms: Set[str], original_text: str) -> str:
        """Restore technical terms that may have been modified during paraphrasing"""
        
        restored_text = paraphrased_text
        
        for term in technical_terms:
            # Check if term was corrupted in paraphrasing
            if term.lower() not in paraphrased_text.lower():
                # Find similar corrupted versions and restore
                corrupted_patterns = [
                    rf'\b{re.escape(term[:-1])}\w*\b',  # Partial matches
                    rf'\b\w*{re.escape(term[1:])}\b'    # Suffix matches
                ]
                
                for pattern in corrupted_patterns:
                    matches = re.findall(pattern, paraphrased_text, re.IGNORECASE)
                    for match in matches:
                        if self._is_likely_corrupted_technical_term(match, term):
                            restored_text = restored_text.replace(match, term, 1)
                            break
        
        return restored_text

    def _is_likely_corrupted_technical_term(self, candidate: str, original_term: str) -> bool:
        """Check if candidate is likely a corrupted version of the technical term"""
        
        # Simple heuristic: check character overlap
        overlap = len(set(candidate.lower()) & set(original_term.lower()))
        overlap_ratio = overlap / len(original_term)
        
        return overlap_ratio > 0.7 and len(candidate) <= len(original_term) + 2

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for processing"""
        
        # Simple sentence splitting (can be enhanced with more sophisticated NLP)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_diversity_score(self, original: str, paraphrased: str) -> float:
        """Calculate diversity score between original and paraphrased text"""
        
        # Tokenize both texts
        original_words = set(re.findall(r'\b\w+\b', original.lower()))
        paraphrased_words = set(re.findall(r'\b\w+\b', paraphrased.lower()))
        
        if not original_words:
            return 0.0
        
        # Calculate word-level diversity (Jaccard distance)
        intersection = original_words & paraphrased_words
        union = original_words | paraphrased_words
        
        jaccard_similarity = len(intersection) / len(union) if union else 1.0
        diversity_score = 1.0 - jaccard_similarity
        
        return diversity_score

    def _calculate_semantic_similarity(self, original: str, paraphrased: str) -> float:
        """Calculate semantic similarity between texts"""
        
        # Simple similarity based on word overlap (can be enhanced with embeddings)
        original_words = set(re.findall(r'\b\w+\b', original.lower()))
        paraphrased_words = set(re.findall(r'\b\w+\b', paraphrased.lower()))
        
        if not original_words and not paraphrased_words:
            return 1.0
        if not original_words or not paraphrased_words:
            return 0.0
        
        intersection = original_words & paraphrased_words
        overlap_coefficient = len(intersection) / min(len(original_words), len(paraphrased_words))
        
        return overlap_coefficient

    def get_paraphrasing_statistics(self) -> Dict[str, any]:
        """Get statistics about paraphrasing usage"""
        
        total_patterns_used = sum(len(usage) for usage in self.pattern_usage_count.values())
        unique_patterns = len(self.pattern_usage_count)
        
        return {
            'total_patterns_used': total_patterns_used,
            'unique_patterns': unique_patterns,
            'pattern_usage_distribution': dict(self.pattern_usage_count),
            'technical_terms_loaded': len(self.technical_terms),
            'action_verb_patterns': len(self.action_verb_patterns)
        }

    def reset_diversity_tracking(self):
        """Reset diversity tracking for new augmentation batch"""
        self.used_patterns.clear()
        self.pattern_usage_count.clear()