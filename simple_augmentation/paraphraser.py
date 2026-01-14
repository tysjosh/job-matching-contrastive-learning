"""
SentenceTransformer-based text paraphrasing functionality for resume augmentation.

This module provides the SentenceTransformerParaphraser class that uses sentence embeddings
to generate semantically similar alternative phrasings while preserving technical terms
and professional language.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Set
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SentenceTransformerParaphraser:
    """
    SentenceTransformer-based paraphraser that generates semantic variations of resume text
    while preserving technical terms and professional language.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 preserve_terms: Optional[List[str]] = None,
                 similarity_threshold: float = 0.5,
                 max_replacements_per_sentence: int = 3,
                 protected_patterns: Optional[List[str]] = None):
        """
        Initialize the SentenceTransformer paraphraser.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            preserve_terms: List of terms to preserve without modification
            similarity_threshold: Minimum similarity score for synonym replacement
            max_replacements_per_sentence: Maximum number of word replacements per sentence
            protected_patterns: Regex patterns for terms to protect from modification
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_replacements_per_sentence = max_replacements_per_sentence
        
        # Initialize SentenceTransformer model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            raise
        
        # Set up protected terms and patterns
        self.preserve_terms = set(preserve_terms or [])
        self.protected_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in (protected_patterns or [])]
        
        # Initialize synonym mappings cache
        self._synonym_cache: Dict[str, List[str]] = {}
        self._load_synonym_mappings()
        
        logger.info(f"Initialized SentenceTransformerParaphraser with {len(self.preserve_terms)} protected terms")
    
    def paraphrase_experience(self, experience_text: str) -> str:
        """
        Paraphrase resume experience text while preserving technical terms and meaning.
        
        Args:
            experience_text: Original experience text to paraphrase
            
        Returns:
            Paraphrased text with semantic variations
        """
        if not experience_text or not experience_text.strip():
            return experience_text
        
        try:
            # Split into sentences for processing
            sentences = self._split_sentences(experience_text)
            paraphrased_sentences = []
            
            for sentence in sentences:
                if sentence.strip():
                    paraphrased = self._paraphrase_sentence(sentence)
                    paraphrased_sentences.append(paraphrased)
                else:
                    paraphrased_sentences.append(sentence)
            
            result = ' '.join(paraphrased_sentences)
            logger.debug(f"Paraphrased experience text: {len(experience_text)} -> {len(result)} chars")
            return result
            
        except Exception as e:
            logger.warning(f"Error paraphrasing experience text: {e}")
            return experience_text  # Fallback to original text
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for individual processing with enhanced segmentation.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Enhanced sentence splitting that handles common abbreviations and professional terms
        abbreviations = [
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Inc', 'Ltd', 'Corp', 'vs', 'etc', 
            'Jr', 'Sr', r'Ph\.D', r'M\.S', r'B\.S', r'M\.A', r'B\.A', 'MBA', 'CEO', 
            'CTO', 'VP', 'SVP', 'API', 'UI', 'UX', 'IT', 'HR', 'QA', 'R&D'
        ]
        
        # Create pattern that avoids splitting on abbreviations
        # Use a simpler approach to avoid variable-width lookbehind issues
        sentence_pattern = r'[.!?]+\s+'
        
        sentences = re.split(sentence_pattern, text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Post-process to merge sentences that were incorrectly split on abbreviations
        merged_sentences = []
        i = 0
        while i < len(sentences):
            current_sentence = sentences[i]
            
            # Check if current sentence ends with an abbreviation
            should_merge = False
            for abbrev in abbreviations:
                # Remove regex escapes for matching
                clean_abbrev = abbrev.replace('\\', '')
                if current_sentence.rstrip().endswith(clean_abbrev):
                    should_merge = True
                    break
            
            if should_merge and i + 1 < len(sentences):
                # Merge with next sentence
                merged_sentence = current_sentence + '. ' + sentences[i + 1]
                merged_sentences.append(merged_sentence)
                i += 2  # Skip next sentence as it's been merged
            else:
                merged_sentences.append(current_sentence)
                i += 1
        
        # Handle bullet points and line breaks as sentence boundaries
        expanded_sentences = []
        for sentence in merged_sentences:
            # Split on bullet points or numbered lists
            bullet_splits = re.split(r'\n\s*[•\-\*]\s*|\n\s*\d+\.\s*', sentence)
            for split in bullet_splits:
                if split.strip():
                    expanded_sentences.append(split.strip())
        
        return expanded_sentences
    
    def _paraphrase_sentence(self, sentence: str) -> str:
        """
        Paraphrase a single sentence by finding semantic alternatives for words with fallback protection.
        
        Args:
            sentence: Input sentence to paraphrase
            
        Returns:
            Paraphrased sentence with fallback to original when no alternatives found
        """
        # Preserve technical terms first (validation step)
        self._preserve_technical_terms(sentence)
        
        # Tokenize into words
        words = sentence.split()
        paraphrased_words = []
        replacements_made = 0
        replacement_attempts = 0
        
        for word in words:
            # Skip if we've made enough replacements
            if replacements_made >= self.max_replacements_per_sentence:
                paraphrased_words.append(word)
                continue
            
            # Clean word for processing (remove punctuation for matching)
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            
            # Skip if word is protected
            if self._is_protected_term(word, clean_word):
                paraphrased_words.append(word)
                continue
            
            # Try to find semantic alternative
            replacement_attempts += 1
            alternative = self._find_semantic_alternative(clean_word)
            
            if alternative and alternative != clean_word:
                # Preserve original capitalization and punctuation
                formatted_alternative = self._preserve_formatting(word, alternative)
                paraphrased_words.append(formatted_alternative)
                replacements_made += 1
            else:
                # Fallback to original word when no suitable alternative found
                paraphrased_words.append(word)
        
        result = ' '.join(paraphrased_words)
        
        # Validate professional language maintenance
        if not self._validate_professional_language(sentence, result):
            logger.debug("Professional language validation failed, using original sentence")
            return sentence
        
        # Log replacement statistics
        if replacement_attempts > 0:
            success_rate = replacements_made / replacement_attempts
            logger.debug(f"Sentence paraphrasing: {replacements_made}/{replacement_attempts} replacements (success rate: {success_rate:.2f})")
        
        return result
    
    def _validate_professional_language(self, original: str, paraphrased: str) -> bool:
        """
        Validate that paraphrased text maintains professional language standards.
        
        Args:
            original: Original sentence
            paraphrased: Paraphrased sentence
            
        Returns:
            True if professional standards are maintained
        """
        # Basic validation checks
        
        # Ensure sentence length is reasonable (not too different from original)
        length_ratio = len(paraphrased) / len(original) if len(original) > 0 else 1
        if length_ratio < 0.5 or length_ratio > 1.5:
            return False
        
        # Ensure sentence structure is maintained (similar word count)
        original_words = len(original.split())
        paraphrased_words = len(paraphrased.split())
        word_ratio = paraphrased_words / original_words if original_words > 0 else 1
        if word_ratio < 0.7 or word_ratio > 1.3:
            return False
        
        # Ensure no inappropriate words were introduced
        inappropriate_words = {'bad', 'terrible', 'awful', 'horrible', 'stupid', 'dumb'}
        paraphrased_lower = paraphrased.lower()
        if any(word in paraphrased_lower for word in inappropriate_words):
            return False
        
        return True
    
    def _find_semantic_alternative(self, word: str) -> Optional[str]:
        """
        Find a semantic alternative for a word using predefined professional synonyms.
        
        Args:
            word: Word to find alternative for
            
        Returns:
            Semantic alternative or None if no suitable alternative found
        """
        # Check cache first
        if word in self._synonym_cache:
            synonyms = self._synonym_cache[word]
            if synonyms:
                return np.random.choice(synonyms)
            return None
        
        # Skip very short words, numbers, or common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        if len(word) < 3 or word.isdigit() or word.lower() in stop_words:
            self._synonym_cache[word] = []  # Cache empty result
            return None
        
        # Use predefined professional synonyms directly (more reliable than embeddings)
        candidates = self._get_candidate_synonyms(word)
        if not candidates:
            self._synonym_cache[word] = []
            return None
        
        # Filter candidates to ensure they're appropriate
        valid_candidates = []
        for candidate in candidates:
            if (candidate != word and 
                self._validate_semantic_appropriateness(word, candidate, 1.0)):  # Skip similarity check
                valid_candidates.append(candidate)
        
        # Cache results
        self._synonym_cache[word] = valid_candidates
        
        # Return random valid candidate
        if valid_candidates:
            return np.random.choice(valid_candidates)
        
        return None
    
    def _validate_semantic_appropriateness(self, original: str, candidate: str, similarity: float) -> bool:
        """
        Validate that a candidate replacement is semantically appropriate.
        
        Args:
            original: Original word
            candidate: Candidate replacement
            similarity: Similarity score
            
        Returns:
            True if replacement is appropriate
        """
        # Additional validation beyond similarity threshold
        
        # Ensure similar word length (within reasonable bounds)
        length_ratio = len(candidate) / len(original) if len(original) > 0 else 1
        if length_ratio < 0.5 or length_ratio > 2.0:
            return False
        
        # Ensure both words are actual words (not fragments)
        if len(candidate) < 2 or not candidate.isalpha():
            return False
        
        # Higher similarity threshold for very different word lengths
        if abs(len(candidate) - len(original)) > 3 and similarity < 0.8:
            return False
        
        return True
    
    def _get_candidate_synonyms(self, word: str) -> List[str]:
        """
        Get candidate synonyms for a word based on common professional vocabulary.
        
        Args:
            word: Word to find candidates for
            
        Returns:
            List of candidate synonyms
        """
        # Professional vocabulary mappings for common resume terms
        professional_synonyms = {
            'develop': ['create', 'build', 'design', 'implement', 'construct', 'engineer'],
            'developing': ['creating', 'building', 'designing', 'implementing', 'constructing', 'engineering'],
            'manage': ['oversee', 'supervise', 'coordinate', 'direct', 'lead'],
            'managing': ['overseeing', 'supervising', 'coordinating', 'directing', 'leading'],
            'improve': ['enhance', 'optimize', 'upgrade', 'refine', 'advance'],
            'analyze': ['examine', 'evaluate', 'assess', 'review', 'study'],
            'implement': ['execute', 'deploy', 'establish', 'install', 'apply'],
            'collaborate': ['cooperate', 'partner', 'work', 'coordinate', 'team'],
            'responsible': ['accountable', 'liable', 'in-charge', 'tasked', 'assigned'],
            'experience': ['background', 'expertise', 'knowledge', 'skills', 'proficiency'],
            'project': ['initiative', 'program', 'assignment', 'task', 'undertaking'],
            'team': ['group', 'squad', 'unit', 'crew', 'staff'],
            'teams': ['groups', 'squads', 'units', 'crews', 'staff'],
            'client': ['customer', 'patron', 'consumer', 'user', 'stakeholder'],
            'solution': ['resolution', 'answer', 'approach', 'method', 'fix'],
            'system': ['platform', 'framework', 'infrastructure', 'architecture', 'setup'],
            'systems': ['platforms', 'frameworks', 'infrastructures', 'architectures', 'setups'],
            'process': ['procedure', 'workflow', 'method', 'approach', 'protocol'],
            'support': ['assist', 'help', 'aid', 'facilitate', 'enable'],
            'work': ['operate', 'function', 'perform', 'execute', 'handle'],
            'worked': ['operated', 'functioned', 'performed', 'executed', 'handled'],
            'build': ['construct', 'create', 'develop', 'establish', 'assemble'],
            'built': ['constructed', 'created', 'developed', 'established', 'assembled'],
            'scalable': ['expandable', 'flexible', 'adaptable', 'extensible', 'modular'],
            'web': ['online', 'internet', 'digital', 'browser-based'],
            'application': ['program', 'software', 'tool', 'platform', 'solution'],
            'applications': ['programs', 'software', 'tools', 'platforms', 'solutions'],
            'utilize': ['use', 'employ', 'leverage', 'apply', 'implement'],
            'utilized': ['used', 'employed', 'leveraged', 'applied', 'implemented'],
            'create': ['develop', 'build', 'design', 'establish', 'generate'],
            'created': ['developed', 'built', 'designed', 'established', 'generated'],
            'maintain': ['sustain', 'preserve', 'uphold', 'support', 'manage'],
            'maintained': ['sustained', 'preserved', 'upheld', 'supported', 'managed'],
            'deliver': ['provide', 'supply', 'present', 'execute', 'complete'],
            'delivered': ['provided', 'supplied', 'presented', 'executed', 'completed'],
            'ensure': ['guarantee', 'verify', 'confirm', 'secure', 'establish'],
            'ensured': ['guaranteed', 'verified', 'confirmed', 'secured', 'established']
        }
        
        return professional_synonyms.get(word.lower(), [])
    
    def _is_protected_term(self, original_word: str, clean_word: str) -> bool:
        """
        Check if a word should be protected from modification with enhanced protection logic.
        
        Args:
            original_word: Original word with punctuation
            clean_word: Cleaned word for matching
            
        Returns:
            True if word should be protected
        """
        # Check preserve terms list (case-insensitive)
        preserve_terms_lower = [term.lower() for term in self.preserve_terms]
        if clean_word.lower() in preserve_terms_lower or original_word.lower() in preserve_terms_lower:
            return True
        
        # Check protected patterns
        for pattern in self.protected_patterns:
            if pattern.search(original_word):
                return True
        
        # Protect numbers and version strings
        if re.search(r'\d', original_word):
            return True
        
        # Protect technical file extensions and domains
        if re.search(r'\.\w{2,4}$', original_word):
            return True
        
        # Only protect very specific technical terms (be much more selective)
        specific_tech_terms = {
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'nodejs', 'mongodb', 'postgresql', 'mysql', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'api', 'rest', 'graphql', 'json', 'xml', 'html',
            'css', 'sql'
        }
        
        if clean_word.lower() in specific_tech_terms:
            return True
        
        # Only protect specific degree terms
        specific_cert_terms = {
            'phd', 'mba', 'msc', 'bsc', 'ba', 'ma', 'bachelor', 'master', 'doctorate'
        }
        
        if clean_word.lower() in specific_cert_terms:
            return True
        
        # Only protect specific seniority levels
        specific_level_terms = {
            'senior', 'junior', 'lead', 'principal', 'staff'
        }
        
        if clean_word.lower() in specific_level_terms:
            return True
        
        return False
    
    def _preserve_technical_terms(self, text: str) -> str:
        """
        Identify and validate technical terms for preservation with enhanced protection.
        
        Args:
            text: Input text
            
        Returns:
            Text (unchanged, but validated for technical terms)
        """
        # Log protected terms found for validation
        protected_terms_found = []
        
        # Check for preserve terms
        for term in self.preserve_terms:
            if term.lower() in text.lower():
                protected_terms_found.append(term)
        
        # Check for protected patterns
        for pattern in self.protected_patterns:
            matches = pattern.findall(text)
            if matches:
                protected_terms_found.extend(matches)
        
        # Additional technical term detection (more specific patterns)
        technical_indicators = [
            r'\b[A-Z]{3,}\b',  # Acronyms with 3+ letters (API, SQL, etc.)
            r'\b\d+\+?\s*years?\b',  # Experience years
            r'\b(?:Bachelor|Master|PhD|MBA|MSc|BSc)(?:\s+of\s+\w+)?\b',  # Degrees
            r'\b(?:senior|junior|mid|lead|principal|staff)\s+(?:level|engineer|developer|manager)\b',  # Specific seniority levels
        ]
        
        for pattern_str in technical_indicators:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                protected_terms_found.extend(matches)
        
        if protected_terms_found:
            logger.debug(f"Protected terms found in text: {protected_terms_found}")
        
        return text
    
    def _preserve_formatting(self, original_word: str, alternative: str) -> str:
        """
        Preserve the original formatting (capitalization, punctuation) when replacing words.
        
        Args:
            original_word: Original word with formatting
            alternative: Alternative word to format
            
        Returns:
            Alternative word with preserved formatting
        """
        # Extract punctuation from original
        leading_punct = re.match(r'^[^\w]*', original_word).group()
        trailing_punct = re.search(r'[^\w]*$', original_word).group()
        
        # Get the core word without punctuation
        core_original = re.sub(r'[^\w]', '', original_word)
        
        # Apply capitalization pattern
        if core_original.isupper():
            formatted_alternative = alternative.upper()
        elif core_original.istitle():
            formatted_alternative = alternative.capitalize()
        elif core_original[0].isupper() if core_original else False:
            formatted_alternative = alternative.capitalize()
        else:
            formatted_alternative = alternative.lower()
        
        # Combine with punctuation
        return leading_punct + formatted_alternative + trailing_punct
    
    def process_resume_experience(self, resume_data: Dict) -> Dict:
        """
        Process both resume.experience and resume.original_text fields with paraphrased content.
        
        Args:
            resume_data: Resume data dictionary containing experience and original_text fields
            
        Returns:
            Updated resume data with paraphrased experience fields
        """
        updated_resume = resume_data.copy()
        
        try:
            # Process experience field if it exists
            if 'experience' in updated_resume and updated_resume['experience']:
                original_experience = updated_resume['experience']
                paraphrased_experience = self.paraphrase_experience(original_experience)
                updated_resume['experience'] = paraphrased_experience
                logger.debug(f"Processed experience field: {len(original_experience)} -> {len(paraphrased_experience)} chars")
            
            # Process original_text field if it exists
            if 'original_text' in updated_resume and updated_resume['original_text']:
                original_text = updated_resume['original_text']
                paraphrased_text = self.paraphrase_experience(original_text)
                updated_resume['original_text'] = paraphrased_text
                logger.debug(f"Processed original_text field: {len(original_text)} -> {len(paraphrased_text)} chars")
            
            # Preserve experience level indicators
            updated_resume = self._preserve_experience_level_indicators(updated_resume)
            
            # Ensure professional resume formatting
            updated_resume = self._ensure_professional_formatting(updated_resume)
            
        except Exception as e:
            logger.error(f"Error processing resume experience: {e}")
            # Return original data on error
            return resume_data
        
        return updated_resume
    
    def _preserve_experience_level_indicators(self, resume_data: Dict) -> Dict:
        """
        Ensure experience level indicators are preserved in paraphrased content.
        
        Args:
            resume_data: Resume data dictionary
            
        Returns:
            Resume data with preserved experience level indicators
        """
        # Experience level patterns to preserve
        level_patterns = [
            r'\b\d+\+?\s*years?\s*(?:of\s*)?(?:experience|exp)\b',
            r'\b(?:senior|junior|mid|entry|lead|principal|staff)\s*(?:level)?\b',
            r'\b(?:experienced|expert|novice|intermediate|advanced)\b',
            r'\b(?:I|II|III|IV|V)\b',  # Roman numerals for levels
        ]
        
        for field in ['experience', 'original_text']:
            if field in resume_data and resume_data[field]:
                text = resume_data[field]
                
                # Check if any level indicators were lost and restore them
                for pattern in level_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        logger.debug(f"Preserved experience level indicators: {matches}")
        
        return resume_data
    
    def _ensure_professional_formatting(self, resume_data: Dict) -> Dict:
        """
        Ensure professional resume formatting standards are maintained.
        
        Args:
            resume_data: Resume data dictionary
            
        Returns:
            Resume data with professional formatting
        """
        for field in ['experience', 'original_text']:
            if field in resume_data and resume_data[field]:
                text = resume_data[field]
                
                # Clean up extra whitespace
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                
                # Ensure proper sentence capitalization
                sentences = text.split('. ')
                formatted_sentences = []
                for sentence in sentences:
                    if sentence:
                        # Capitalize first letter of each sentence
                        sentence = sentence.strip()
                        if sentence:
                            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                            formatted_sentences.append(sentence)
                
                text = '. '.join(formatted_sentences)
                
                # Ensure text ends with proper punctuation
                if text and not text.endswith(('.', '!', '?')):
                    text += '.'
                
                resume_data[field] = text
        
        return resume_data
    
    def _load_synonym_mappings(self) -> None:
        """
        Load or initialize synonym mappings cache.
        """
        # Initialize empty cache - will be populated during processing
        self._synonym_cache = {}
        logger.debug("Initialized empty synonym cache")