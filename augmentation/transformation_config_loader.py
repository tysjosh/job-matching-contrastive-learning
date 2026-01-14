"""
Transformation Configuration Loader

Centralized loader for all transformation-related configuration files.
Replaces hardcoded static data with external YAML configurations for scalability.
"""

import logging
import os
import random
from typing import Dict, List, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import yaml, provide fallback message if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. Install with: pip install pyyaml")


class TransformationConfigLoader:
    """
    Centralized configuration loader for transformation rules and mappings.
    
    Loads configuration from YAML files instead of hardcoded dictionaries,
    enabling easier maintenance and scalability.
    """
    
    # Default config directory relative to project root
    DEFAULT_CONFIG_DIR = "config"
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Base directory for configuration files.
                       Defaults to 'config' in project root.
        """
        self.config_dir = config_dir or self._find_config_dir()
        self._cache: Dict[str, Any] = {}
        self._phrase_usage_history: Dict[str, int] = {}
        self._max_phrase_reuse = 3
        
        if not YAML_AVAILABLE:
            logger.error("PyYAML is required for configuration loading")
        
        logger.info(f"TransformationConfigLoader initialized with config_dir: {self.config_dir}")
    
    def _find_config_dir(self) -> str:
        """Find the config directory relative to this file or project root."""
        # Try relative to this file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Check parent directory (project root)
        config_path = os.path.join(parent_dir, self.DEFAULT_CONFIG_DIR)
        if os.path.exists(config_path):
            return config_path
        
        # Check current working directory
        cwd_config = os.path.join(os.getcwd(), self.DEFAULT_CONFIG_DIR)
        if os.path.exists(cwd_config):
            return cwd_config
        
        # Default to parent/config
        return config_path
    
    def _load_yaml_file(self, relative_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            relative_path: Path relative to config_dir
            
        Returns:
            Parsed YAML content as dictionary
        """
        if not YAML_AVAILABLE:
            logger.error("Cannot load YAML: PyYAML not installed")
            return {}
        
        # Check cache first
        if relative_path in self._cache:
            return self._cache[relative_path]
        
        full_path = os.path.join(self.config_dir, relative_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                self._cache[relative_path] = data
                logger.debug(f"Loaded config: {relative_path}")
                return data
        except FileNotFoundError:
            logger.warning(f"Config file not found: {full_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {full_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config {full_path}: {e}")
            return {}
    
    def clear_cache(self) -> None:
        """Clear the configuration cache to force reload."""
        self._cache.clear()
        self._phrase_usage_history.clear()
        logger.debug("Configuration cache cleared")
    
    # ==================== Verb Transformations ====================
    
    def load_verb_upgrades(self) -> Dict[str, List[str]]:
        """Load verb upgrade transformations (junior -> senior)."""
        config = self._load_yaml_file("transformation_rules/verb_transformations.yaml")
        return config.get('upgrades', {})
    
    def load_verb_downgrades(self) -> Dict[str, List[str]]:
        """Load verb downgrade transformations (senior -> junior)."""
        config = self._load_yaml_file("transformation_rules/verb_transformations.yaml")
        return config.get('downgrades', {})
    
    # ==================== Scope Transformations ====================
    
    def load_scope_amplifiers(self) -> Dict[str, List[str]]:
        """Load scope amplifier transformations (small -> large)."""
        config = self._load_yaml_file("transformation_rules/scope_transformations.yaml")
        return config.get('amplifiers', {})
    
    def load_scope_reducers(self) -> Dict[str, List[str]]:
        """Load scope reducer transformations (large -> small)."""
        config = self._load_yaml_file("transformation_rules/scope_transformations.yaml")
        return config.get('reducers', {})
    
    # ==================== Title Transformations ====================
    
    def load_title_upgrades(self) -> Dict[str, List[str]]:
        """Load title upgrade transformations."""
        config = self._load_yaml_file("transformation_rules/title_transformations.yaml")
        return config.get('upgrades', {})
    
    def load_title_downgrades(self) -> Dict[str, List[str]]:
        """Load title downgrade transformations."""
        config = self._load_yaml_file("transformation_rules/title_transformations.yaml")
        return config.get('downgrades', {})
    
    # ==================== Impact Phrases ====================
    
    def load_impact_phrases(self, domain: str, level: str) -> List[str]:
        """
        Load impact phrases for a specific domain and level.
        
        Args:
            domain: Career domain (e.g., 'software_development', 'data_science')
            level: Experience level (e.g., 'senior', 'lead', 'principal')
            
        Returns:
            List of impact phrases for the domain/level combination
        """
        config = self._load_yaml_file("transformation_rules/impact_phrases.yaml")
        
        domain_phrases = config.get(domain, {})
        if isinstance(domain_phrases, dict):
            phrases = domain_phrases.get(level, [])
            if phrases:
                return phrases
        
        # Fallback to generic phrases
        return config.get('fallback', [
            'resulting in improved system performance',
            'leading to enhanced code quality',
            'driving team productivity improvements'
        ])
    
    def get_random_impact_phrase(self, domain: str, level: str) -> str:
        """
        Get a random impact phrase with variety tracking.
        
        Args:
            domain: Career domain
            level: Experience level
            
        Returns:
            A randomly selected impact phrase
        """
        phrases = self.load_impact_phrases(domain, level)
        return self._get_varied_phrase(phrases, f"impact_{domain}_{level}")
    
    def load_all_impact_phrases(self) -> Dict[str, Dict[str, List[str]]]:
        """Load all impact phrases organized by domain and level."""
        return self._load_yaml_file("transformation_rules/impact_phrases.yaml")
    
    # ==================== Learning Phrases ====================
    
    def load_learning_phrases_by_domain(self, domain: str, level: str) -> List[str]:
        """
        Load learning phrases for a specific domain and level.
        
        Args:
            domain: Career domain
            level: Experience level ('entry' or 'junior')
            
        Returns:
            List of learning phrases
        """
        config = self._load_yaml_file("transformation_rules/learning_phrases.yaml")
        
        by_domain = config.get('by_domain', {})
        domain_phrases = by_domain.get(domain, {})
        
        if isinstance(domain_phrases, dict):
            phrases = domain_phrases.get(level, [])
            if phrases:
                return phrases
        
        # Fallback
        return config.get('fallback', [
            'under senior guidance',
            'as part of the team',
            'while learning best practices'
        ])
    
    def load_support_phrases_by_context(self, context: str) -> List[str]:
        """
        Load support phrases for a specific context.
        
        Args:
            context: Context type (e.g., 'technical_implementation', 'problem_solving')
            
        Returns:
            List of support phrases
        """
        config = self._load_yaml_file("transformation_rules/learning_phrases.yaml")
        by_context = config.get('by_context', {})
        return by_context.get(context, by_context.get('general', []))
    
    def load_learning_enhancement_phrases(self, level: str, phrase_type: str) -> List[str]:
        """
        Load learning enhancement phrases.
        
        Args:
            level: 'entry_level' or 'junior_level'
            phrase_type: 'technical', 'analytical', or 'collaborative'
            
        Returns:
            List of enhancement phrases
        """
        config = self._load_yaml_file("transformation_rules/learning_phrases.yaml")
        enhancement = config.get('enhancement', {})
        level_phrases = enhancement.get(level, {})
        return level_phrases.get(phrase_type, [])
    
    def load_task_focused_additions(self) -> List[str]:
        """Load task-focused addition phrases."""
        config = self._load_yaml_file("transformation_rules/learning_phrases.yaml")
        return config.get('task_focused', [])
    
    def get_random_learning_phrase(self, domain: str, level: str) -> str:
        """Get a random learning phrase with variety tracking."""
        phrases = self.load_learning_phrases_by_domain(domain, level)
        return self._get_varied_phrase(phrases, f"learning_{domain}_{level}")
    
    # ==================== Experience Level Mappings ====================
    
    def load_experience_level_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load experience level mappings for metadata synchronization."""
        config = self._load_yaml_file("metadata_mappings/experience_levels.yaml")
        return config.get('levels', {})
    
    def load_title_to_level_mapping(self) -> Dict[str, List[str]]:
        """Load title to experience level mapping."""
        config = self._load_yaml_file("metadata_mappings/experience_levels.yaml")
        return config.get('title_to_level', {})
    
    def load_level_proficiencies(self) -> Dict[str, List[str]]:
        """Load expected proficiencies for each level."""
        config = self._load_yaml_file("metadata_mappings/experience_levels.yaml")
        return config.get('level_proficiencies', {})
    
    # ==================== Skill Proficiency Mappings ====================
    
    def load_skill_proficiency_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load skill proficiency mappings."""
        config = self._load_yaml_file("metadata_mappings/skill_proficiency.yaml")
        return config.get('proficiency_levels', {})
    
    # ==================== Transformation Patterns ====================
    
    def load_transformation_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for metadata extraction."""
        config = self._load_yaml_file("metadata_mappings/transformation_patterns.yaml")
        patterns = {}
        
        for key in ['years_experience', 'skill_proficiency', 'seniority_indicators']:
            patterns[key] = config.get(key, [])
        
        return patterns
    
    def load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain classification keywords."""
        config = self._load_yaml_file("metadata_mappings/transformation_patterns.yaml")
        return config.get('domain_keywords', {})
    
    # ==================== Professional Synonyms ====================
    
    def load_professional_synonyms(self) -> Dict[str, List[str]]:
        """Load professional synonym mappings for paraphrasing."""
        config = self._load_yaml_file("synonyms/professional_synonyms.yaml")
        return config.get('mappings', {})
    
    def load_protected_terms(self) -> Dict[str, List[str]]:
        """Load protected terms that should not be replaced."""
        config = self._load_yaml_file("synonyms/professional_synonyms.yaml")
        return config.get('protected_terms', {})
    
    def load_stop_words(self) -> List[str]:
        """Load stop words to skip during paraphrasing."""
        config = self._load_yaml_file("synonyms/professional_synonyms.yaml")
        return config.get('stop_words', [])
    
    def load_inappropriate_words(self) -> List[str]:
        """Load inappropriate words to avoid."""
        config = self._load_yaml_file("synonyms/professional_synonyms.yaml")
        return config.get('inappropriate_words', [])
    
    def load_abbreviations(self) -> List[str]:
        """Load abbreviations to preserve."""
        config = self._load_yaml_file("synonyms/professional_synonyms.yaml")
        return config.get('abbreviations', [])
    
    # ==================== Utility Methods ====================
    
    def _get_varied_phrase(self, phrases: List[str], category: str) -> str:
        """
        Get a phrase with variety tracking to avoid repetition.
        
        Args:
            phrases: List of available phrases
            category: Category key for tracking usage
            
        Returns:
            A phrase that hasn't been overused
        """
        if not phrases:
            return ""
        
        # Filter out overused phrases
        available = []
        for phrase in phrases:
            key = f"{category}:{phrase}"
            usage_count = self._phrase_usage_history.get(key, 0)
            if usage_count < self._max_phrase_reuse:
                available.append(phrase)
        
        # Reset if all phrases are overused
        if not available:
            for phrase in phrases:
                key = f"{category}:{phrase}"
                self._phrase_usage_history[key] = 0
            available = phrases
        
        # Select and track
        selected = random.choice(available)
        key = f"{category}:{selected}"
        self._phrase_usage_history[key] = self._phrase_usage_history.get(key, 0) + 1
        
        return selected
    
    def get_ownership_phrases(self) -> List[str]:
        """Get ownership phrases for upward transformation."""
        # These are simple enough to keep as defaults, but could be moved to config
        return [
            'took ownership of',
            'led the initiative to',
            'drove the development of',
            'spearheaded the effort to',
            'championed the implementation of',
            'orchestrated the delivery of',
            'directed the creation of',
            'oversaw the development of'
        ]
    
    def get_strategic_additions(self) -> List[str]:
        """Get strategic thinking additions for upward transformation."""
        return [
            'with focus on scalability',
            'considering long-term maintainability',
            'aligned with business objectives',
            'following industry best practices',
            'ensuring enterprise-grade quality',
            'with emphasis on performance optimization',
            'incorporating security best practices',
            'designed for future extensibility'
        ]


# Singleton instance for easy access
_config_loader_instance: Optional[TransformationConfigLoader] = None


def get_config_loader(config_dir: Optional[str] = None) -> TransformationConfigLoader:
    """
    Get the singleton configuration loader instance.
    
    Args:
        config_dir: Optional config directory override
        
    Returns:
        TransformationConfigLoader instance
    """
    global _config_loader_instance
    
    if _config_loader_instance is None or config_dir is not None:
        _config_loader_instance = TransformationConfigLoader(config_dir)
    
    return _config_loader_instance


def reset_config_loader() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _config_loader_instance
    _config_loader_instance = None
