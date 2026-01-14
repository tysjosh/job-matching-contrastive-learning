"""
Job Description Transformer: Applies Tup/Tdown transformations to job descriptions

This module transforms job descriptions to create senior-level and junior-level variants,
enabling the generation of career-progression-aware training pairs.
"""

import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


class JobTransformer:
    """
    Transforms job descriptions for career-aware augmentation.
    
    Creates senior-level (Tup) and junior-level (Tdown) variants of job descriptions
    to pair with transformed resumes for comprehensive career progression training.
    """
    
    def __init__(self):
        """Initialize job transformer with transformation rules"""
        self._load_transformation_rules()
    
    def _load_transformation_rules(self):
        """Load transformation rules for job descriptions"""
        
        # Title transformations for upward progression
        self.title_upgrades = {
            'analyst': ['senior analyst', 'lead analyst', 'principal analyst'],
            'developer': ['senior developer', 'lead developer', 'principal engineer'],
            'engineer': ['senior engineer', 'staff engineer', 'principal engineer'],
            'designer': ['senior designer', 'lead designer', 'design director'],
            'manager': ['senior manager', 'director', 'vice president'],
            'specialist': ['senior specialist', 'lead specialist', 'principal specialist'],
            'researcher': ['senior researcher', 'lead researcher', 'principal scientist'],
            'architect': ['senior architect', 'principal architect', 'chief architect']
        }
        
        # Title transformations for downward progression
        self.title_downgrades = {
            'senior': ['', 'junior', 'associate'],
            'lead': ['', 'senior', 'associate'],
            'principal': ['senior', 'lead', ''],
            'director': ['manager', 'senior manager', 'lead'],
            'manager': ['associate', 'coordinator', 'specialist'],
            'architect': ['engineer', 'developer', 'analyst']
        }
        
        # Responsibility transformations for upward progression
        self.responsibility_upgrades = {
            'develop': ['architect', 'design', 'lead development of'],
            'create': ['design', 'architect', 'drive creation of'],
            'work with': ['lead', 'manage', 'coordinate'],
            'implement': ['design and implement', 'architect', 'drive implementation'],
            'support': ['lead', 'manage', 'oversee'],
            'assist': ['coordinate', 'manage', 'lead'],
            'handle': ['own', 'drive', 'manage'],
            'maintain': ['optimize', 'enhance', 'strategically manage'],
            'monitor': ['oversee', 'manage', 'strategically monitor'],
            'analyze': ['drive analysis', 'lead analytical efforts', 'strategically analyze']
        }
        
        # Responsibility transformations for downward progression  
        self.responsibility_downgrades = {
            'architect': ['develop', 'build', 'work on'],
            'design': ['create', 'build', 'work on'],
            'lead': ['assist with', 'support', 'contribute to'],
            'manage': ['assist with', 'support', 'help coordinate'],
            'own': ['work on', 'contribute to', 'assist with'],
            'drive': ['support', 'assist with', 'contribute to'],
            'oversee': ['assist with', 'support', 'work under supervision'],
            'strategically': ['', 'tactically', 'operationally'],
            'coordinate': ['assist with coordination', 'support', 'help with']
        }
        
        # Scope transformations for upward progression
        self.scope_upgrades = {
            'project': ['program', 'initiative', 'strategic program'],
            'team': ['organization', 'department', 'cross-functional teams'],
            'feature': ['product', 'platform', 'system'],
            'component': ['system', 'architecture', 'platform'],
            'task': ['project', 'initiative', 'program'],
            'report': ['strategic analysis', 'business intelligence', 'executive briefing']
        }
        
        # Scope transformations for downward progression
        self.scope_downgrades = {
            'program': ['project', 'initiative', 'task'],
            'system': ['component', 'feature', 'module'],
            'platform': ['application', 'tool', 'component'],
            'organization': ['team', 'group', 'department'],
            'strategic': ['operational', 'tactical', ''],
            'enterprise': ['departmental', 'team-level', 'local']
        }
        
        # Impact phrases for senior roles
        self.senior_impact_phrases = [
            "driving business growth",
            "delivering strategic value",
            "optimizing organizational performance",
            "leading digital transformation",
            "maximizing ROI",
            "ensuring competitive advantage"
        ]
        
        # Learning phrases for junior roles
        self.junior_learning_phrases = [
            "gaining experience in",
            "developing skills in", 
            "learning best practices for",
            "building foundation in",
            "growing expertise in",
            "expanding knowledge of"
        ]
    
    def transform_job_upward(self, job: Dict[str, Any], target_level: str) -> Dict[str, Any]:
        """
        Transform job description to senior-level requirements.
        
        Args:
            job: Original job data
            target_level: Target seniority level
            
        Returns:
            Dict: Transformed job with senior-level requirements
        """
        transformed_job = job.copy()
        
        try:
            # Transform job title
            if 'title' in job:
                transformed_job['title'] = self._elevate_job_title(job['title'], target_level)
            
            # Transform job description
            if 'description' in job:
                if isinstance(job['description'], dict) and 'original' in job['description']:
                    original_desc = job['description']['original']
                    transformed_desc = self._transform_description_upward(original_desc, target_level)
                    transformed_job['description'] = job['description'].copy()
                    transformed_job['description']['original'] = transformed_desc
                elif isinstance(job['description'], str):
                    transformed_job['description'] = self._transform_description_upward(
                        job['description'], target_level)
            
            # Update experience level
            transformed_job['experience_level'] = target_level
            
            # Add transformation metadata
            transformed_job['_transformation_meta'] = {
                'type': 'job_upward',
                'target_level': target_level,
                'original_title': job.get('title', ''),
                'transformation_applied': True
            }
            
            return transformed_job
            
        except Exception as e:
            logger.error(f"Job upward transformation failed: {e}")
            return job
    
    def transform_job_downward(self, job: Dict[str, Any], target_level: str) -> Dict[str, Any]:
        """
        Transform job description to junior-level requirements.
        
        Args:
            job: Original job data
            target_level: Target junior level
            
        Returns:
            Dict: Transformed job with junior-level requirements
        """
        transformed_job = job.copy()
        
        try:
            # Transform job title
            if 'title' in job:
                transformed_job['title'] = self._reduce_job_title(job['title'], target_level)
            
            # Transform job description
            if 'description' in job:
                if isinstance(job['description'], dict) and 'original' in job['description']:
                    original_desc = job['description']['original']
                    transformed_desc = self._transform_description_downward(original_desc, target_level)
                    transformed_job['description'] = job['description'].copy()
                    transformed_job['description']['original'] = transformed_desc
                elif isinstance(job['description'], str):
                    transformed_job['description'] = self._transform_description_downward(
                        job['description'], target_level)
            
            # Update experience level
            transformed_job['experience_level'] = target_level
            
            # Add transformation metadata
            transformed_job['_transformation_meta'] = {
                'type': 'job_downward',
                'target_level': target_level,
                'original_title': job.get('title', ''),
                'transformation_applied': True
            }
            
            return transformed_job
            
        except Exception as e:
            logger.error(f"Job downward transformation failed: {e}")
            return job
    
    def _elevate_job_title(self, title: str, target_level: str) -> str:
        """Elevate job title to reflect seniority"""
        title_lower = title.lower().strip()
        
        # Don't modify if already has seniority prefix
        if any(prefix in title_lower for prefix in ['senior', 'lead', 'principal', 'chief', 'director']):
            return title
        
        # Apply title upgrades
        for base_title, upgrades in self.title_upgrades.items():
            if base_title in title_lower:
                if target_level == 'senior':
                    return upgrades[0]
                elif target_level == 'lead':
                    return upgrades[1] if len(upgrades) > 1 else upgrades[0]
                elif target_level == 'principal':
                    return upgrades[2] if len(upgrades) > 2 else upgrades[-1]
        
        # Fallback: add appropriate prefix
        if target_level == 'senior':
            return f"Senior {title}"
        elif target_level == 'lead':
            return f"Lead {title}"
        elif target_level == 'principal':
            return f"Principal {title}"
        
        return title
    
    def _reduce_job_title(self, title: str, target_level: str) -> str:
        """Reduce job title to reflect junior level"""
        title_lower = title.lower()
        
        # Remove senior prefixes
        for senior_term, junior_options in self.title_downgrades.items():
            if senior_term in title_lower:
                title = re.sub(rf'\b{senior_term}\s+', '', title, flags=re.IGNORECASE)
        
        # Add junior prefixes
        if target_level == 'entry' and 'junior' not in title_lower and 'intern' not in title_lower:
            return f"Junior {title}"
        elif target_level == 'mid' and all(prefix not in title_lower for prefix in ['senior', 'lead', 'principal']):
            return title  # Mid-level often has no prefix
        
        return title
    
    def _transform_description_upward(self, description: str, target_level: str) -> str:
        """Transform job description to senior-level requirements"""
        transformed_desc = description
        
        # Apply responsibility upgrades
        for junior_resp, senior_resps in self.responsibility_upgrades.items():
            pattern = r'\b' + re.escape(junior_resp) + r'\b'
            if re.search(pattern, transformed_desc, re.IGNORECASE):
                senior_resp = senior_resps[0]  # Use first option
                transformed_desc = re.sub(
                    pattern, senior_resp, transformed_desc, flags=re.IGNORECASE)
        
        # Apply scope upgrades
        for small_scope, large_scopes in self.scope_upgrades.items():
            pattern = r'\b' + re.escape(small_scope) + r'\b'
            if re.search(pattern, transformed_desc, re.IGNORECASE):
                large_scope = large_scopes[0]  # Use first option
                transformed_desc = re.sub(
                    pattern, large_scope, transformed_desc, flags=re.IGNORECASE)
        
        # Add strategic impact for senior+ levels
        if target_level in ['lead', 'principal']:
            impact_phrase = self.senior_impact_phrases[
                hash(description) % len(self.senior_impact_phrases)]
            if not any(phrase in transformed_desc.lower() for phrase in 
                      ['strategic', 'driving', 'leading', 'optimizing']):
                transformed_desc += f" This role is essential for {impact_phrase}."
        
        return transformed_desc
    
    def _transform_description_downward(self, description: str, target_level: str) -> str:
        """Transform job description to junior-level requirements"""
        transformed_desc = description
        
        # Apply responsibility downgrades
        for senior_resp, junior_resps in self.responsibility_downgrades.items():
            pattern = r'\b' + re.escape(senior_resp) + r'\b'
            if re.search(pattern, transformed_desc, re.IGNORECASE):
                junior_resp = junior_resps[0]  # Use first option
                transformed_desc = re.sub(
                    pattern, junior_resp, transformed_desc, flags=re.IGNORECASE)
        
        # Apply scope downgrades
        for large_scope, small_scopes in self.scope_downgrades.items():
            pattern = r'\b' + re.escape(large_scope) + r'\b'
            if re.search(pattern, transformed_desc, re.IGNORECASE):
                small_scope = small_scopes[0]  # Use first option
                transformed_desc = re.sub(
                    pattern, small_scope, transformed_desc, flags=re.IGNORECASE)
        
        # Add learning context for entry-level
        if target_level == 'entry':
            learning_phrase = self.junior_learning_phrases[
                hash(description) % len(self.junior_learning_phrases)]
            if not any(phrase in transformed_desc.lower() for phrase in 
                      ['learning', 'developing', 'gaining', 'growing']):
                transformed_desc += f" This role offers opportunities for {learning_phrase} industry best practices."
        
        return transformed_desc
