"""
Technical Term Protector for LLM Career-Aware Data Augmentation System

This module provides functionality to protect technical terms (programming languages,
frameworks, tools, etc.) during LLM transformation by using placeholder substitution.
This ensures that technical terms are not corrupted or altered by the LLM.

Requirements: 7.1, 7.2, 7.3, 7.5
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class TermMapping:
    """Mapping between original terms and their placeholders."""
    original_to_placeholder: Dict[str, str] = field(default_factory=dict)
    placeholder_to_original: Dict[str, str] = field(default_factory=dict)


class TechnicalTermProtector:
    """
    Protects technical terms during LLM transformation.
    
    This class identifies technical terms from the CS skills database,
    replaces them with placeholders before LLM processing, and restores
    them afterward to ensure technical accuracy is preserved.
    
    Attributes:
        technical_terms: Set of all technical terms loaded from the database
        case_sensitive_terms: Dict mapping lowercase terms to their canonical casing
    """
    
    # Placeholder format: __TECH_TERM_{index}__
    PLACEHOLDER_PREFIX = "__TECH_TERM_"
    PLACEHOLDER_SUFFIX = "__"
    
    def __init__(self, cs_skills_path: str = "dataset/cs_skills.json"):
        """
        Initialize with CS skills database.
        
        Args:
            cs_skills_path: Path to the CS skills JSON file
        """
        self.cs_skills_path = cs_skills_path
        self.technical_terms: Set[str] = set()
        self.case_sensitive_terms: Dict[str, str] = {}  # lowercase -> canonical
        self._load_technical_terms()
    
    def _load_technical_terms(self) -> None:
        """Load technical terms from the CS skills database."""
        path = Path(self.cs_skills_path)
        if not path.exists():
            raise FileNotFoundError(f"CS skills file not found: {self.cs_skills_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            skills_data = json.load(f)
        
        # Collect all terms from all categories
        for category, terms in skills_data.items():
            if isinstance(terms, list):
                for term in terms:
                    if isinstance(term, str) and term.strip():
                        canonical_term = term.strip()
                        self.technical_terms.add(canonical_term)
                        # Store case-sensitive mapping
                        self.case_sensitive_terms[canonical_term.lower()] = canonical_term
    
    def _create_placeholder(self, index: int) -> str:
        """Create a placeholder string for a given index."""
        return f"{self.PLACEHOLDER_PREFIX}{index}{self.PLACEHOLDER_SUFFIX}"
    
    def _is_placeholder(self, text: str) -> bool:
        """Check if a string is a placeholder."""
        return (
            text.startswith(self.PLACEHOLDER_PREFIX) and 
            text.endswith(self.PLACEHOLDER_SUFFIX)
        )
    
    def _find_term_in_text(self, text: str, term: str) -> List[Tuple[int, int, str]]:
        """
        Find all occurrences of a term in text, respecting word boundaries.
        
        Returns list of (start, end, matched_text) tuples.
        """
        matches = []
        # Escape special regex characters in the term
        escaped_term = re.escape(term)
        # Use word boundaries to match whole words only
        pattern = rf'\b{escaped_term}\b'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((match.start(), match.end(), match.group()))
        
        return matches
    
    def protect_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace technical terms with placeholders.
        
        This method identifies technical terms in the input text and replaces
        them with unique placeholders. The original terms are stored in a
        mapping that can be used to restore them later.
        
        Args:
            text: Input text containing technical terms
            
        Returns:
            Tuple of (protected_text, term_mapping) where term_mapping maps
            placeholders to original terms
        """
        if not text:
            return text, {}
        
        # Find all term occurrences with their positions
        # Sort terms by length (longest first) to handle overlapping terms
        sorted_terms = sorted(self.technical_terms, key=len, reverse=True)
        
        # Track all matches: (start, end, original_text, term)
        all_matches: List[Tuple[int, int, str, str]] = []
        
        for term in sorted_terms:
            matches = self._find_term_in_text(text, term)
            for start, end, matched_text in matches:
                all_matches.append((start, end, matched_text, term))
        
        # Remove overlapping matches (keep longer/earlier matches)
        all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        non_overlapping: List[Tuple[int, int, str, str]] = []
        last_end = -1
        
        for start, end, matched_text, term in all_matches:
            if start >= last_end:
                non_overlapping.append((start, end, matched_text, term))
                last_end = end
        
        # Build protected text and mapping
        term_mapping: Dict[str, str] = {}
        protected_parts: List[str] = []
        last_pos = 0
        placeholder_index = 0
        
        for start, end, matched_text, term in non_overlapping:
            # Add text before this match
            protected_parts.append(text[last_pos:start])
            
            # Create placeholder and store mapping
            placeholder = self._create_placeholder(placeholder_index)
            term_mapping[placeholder] = matched_text  # Preserve original casing
            protected_parts.append(placeholder)
            
            placeholder_index += 1
            last_pos = end
        
        # Add remaining text
        protected_parts.append(text[last_pos:])
        
        protected_text = "".join(protected_parts)
        return protected_text, term_mapping
    
    def restore_terms(self, text: str, term_mapping: Dict[str, str]) -> str:
        """
        Restore original technical terms from placeholders.
        
        Args:
            text: Text containing placeholders
            term_mapping: Mapping from placeholders to original terms
            
        Returns:
            Text with placeholders replaced by original terms
        """
        if not text or not term_mapping:
            return text
        
        restored_text = text
        for placeholder, original_term in term_mapping.items():
            restored_text = restored_text.replace(placeholder, original_term)
        
        return restored_text
    
    def get_preservation_rate(self, original: str, transformed: str) -> float:
        """
        Calculate percentage of technical terms preserved.
        
        This method compares the technical terms in the original text with
        those in the transformed text to determine how many were preserved.
        
        Args:
            original: Original text before transformation
            transformed: Text after transformation
            
        Returns:
            Preservation rate as a float between 0.0 and 1.0
        """
        if not original:
            return 1.0  # No terms to preserve
        
        # Find terms in original text
        original_terms = self._extract_terms(original)
        
        if not original_terms:
            return 1.0  # No terms to preserve
        
        # Find terms in transformed text
        transformed_terms = self._extract_terms(transformed)
        
        # Count preserved terms (case-insensitive comparison)
        original_lower = {t.lower() for t in original_terms}
        transformed_lower = {t.lower() for t in transformed_terms}
        
        preserved_count = len(original_lower & transformed_lower)
        total_count = len(original_lower)
        
        return preserved_count / total_count if total_count > 0 else 1.0
    
    def _extract_terms(self, text: str) -> Set[str]:
        """
        Extract all technical terms found in the text.
        
        Args:
            text: Text to search for technical terms
            
        Returns:
            Set of technical terms found in the text
        """
        if not text:
            return set()
        
        found_terms: Set[str] = set()
        
        for term in self.technical_terms:
            matches = self._find_term_in_text(text, term)
            for _, _, matched_text in matches:
                found_terms.add(matched_text)
        
        return found_terms
    
    def get_canonical_term(self, term: str) -> str:
        """
        Get the canonical (properly cased) version of a term.
        
        Args:
            term: Term to look up (case-insensitive)
            
        Returns:
            Canonical version of the term, or original if not found
        """
        return self.case_sensitive_terms.get(term.lower(), term)
    
    def is_technical_term(self, term: str) -> bool:
        """
        Check if a term is a known technical term.
        
        Args:
            term: Term to check (case-insensitive)
            
        Returns:
            True if the term is a known technical term
        """
        return term.lower() in self.case_sensitive_terms
