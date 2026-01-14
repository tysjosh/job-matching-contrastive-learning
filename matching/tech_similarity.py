"""
Tech-aware similarity computation for enhanced career matching
Specialized for CS/IT/Tech roles with tech stack alignment scoring
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import logging

from .similarity import CareerAwareSimilarityComputer
from .tech_utils import TechSkillMatch, TechStackSummary

@dataclass
class TechAlignmentScore:
    overall_score: float
    skill_overlap_score: float
    stack_compatibility_score: float
    experience_alignment_score: float
    specialization_match: bool
    details: Dict[str, float]

class TechAwareSimilarityComputer(CareerAwareSimilarityComputer):
    """Enhanced similarity computation for tech role matching"""
    
    def __init__(self, career_graph, config: dict = None):
        super().__init__(career_graph)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Tech-specific configuration
        self.tech_skill_weights = self._define_tech_skill_weights()
        self.stack_compatibility_matrix = self._define_stack_compatibility()
        self.specialization_bonuses = self._define_specialization_bonuses()
        
        # Performance optimization - cache frequently used computations
        self.alignment_cache = {}
        
    def _define_tech_skill_weights(self) -> Dict[str, float]:
        """Define importance weights for different tech skill categories"""
        return {
            'programming_languages': 0.3,      # Core programming skills
            'web_frameworks': 0.25,            # Framework experience
            'databases': 0.15,                 # Data storage knowledge
            'cloud_platforms': 0.15,           # Cloud and infrastructure
            'devops_tools': 0.1,               # DevOps and CI/CD
            'data_science': 0.2,               # ML/AI and data skills
            'mobile_development': 0.15,        # Mobile development
            'testing_tools': 0.08,             # Testing frameworks
            'methodologies': 0.07              # Development methodologies
        }
    
    def _define_stack_compatibility(self) -> Dict[str, Dict[str, float]]:
        """Define compatibility scores between different technology stacks"""
        return {
            # Frontend frameworks compatibility
            'React': {'Angular': 0.7, 'Vue.js': 0.8, 'JavaScript': 1.0, 'TypeScript': 0.9},
            'Angular': {'React': 0.7, 'Vue.js': 0.6, 'TypeScript': 1.0, 'JavaScript': 0.8},
            'Vue.js': {'React': 0.8, 'Angular': 0.6, 'JavaScript': 1.0, 'TypeScript': 0.8},
            
            # Backend frameworks compatibility  
            'Django': {'Flask': 0.9, 'Python': 1.0, 'FastAPI': 0.8},
            'Flask': {'Django': 0.9, 'Python': 1.0, 'FastAPI': 0.9},
            'Spring': {'Java': 1.0, 'Kotlin': 0.8, 'Scala': 0.6},
            'Express.js': {'Node.js': 1.0, 'JavaScript': 1.0, 'TypeScript': 0.9},
            
            # Database compatibility
            'PostgreSQL': {'MySQL': 0.8, 'SQL': 1.0, 'MongoDB': 0.3},
            'MySQL': {'PostgreSQL': 0.8, 'SQL': 1.0, 'MongoDB': 0.3},
            'MongoDB': {'Redis': 0.6, 'NoSQL': 1.0, 'PostgreSQL': 0.3},
            
            # Cloud platforms compatibility
            'Amazon Web Services': {'Microsoft Azure': 0.7, 'Google Cloud Platform': 0.7, 'Docker': 0.8},
            'Microsoft Azure': {'Amazon Web Services': 0.7, 'Google Cloud Platform': 0.8, 'Docker': 0.8},
            'Google Cloud Platform': {'Amazon Web Services': 0.7, 'Microsoft Azure': 0.8, 'Docker': 0.8},
            
            # Programming languages compatibility
            'Python': {'Java': 0.6, 'JavaScript': 0.5, 'Go': 0.7, 'R': 0.8},
            'Java': {'Python': 0.6, 'Kotlin': 0.9, 'Scala': 0.8, 'C#': 0.7},
            'JavaScript': {'TypeScript': 0.95, 'Python': 0.5, 'Java': 0.4},
            'TypeScript': {'JavaScript': 0.95, 'C#': 0.6, 'Java': 0.5}
        }
    
    def _define_specialization_bonuses(self) -> Dict[str, float]:
        """Define bonus scores for matching specializations"""
        return {
            'exact_match': 0.15,           # Same specialization (e.g., Frontend -> Frontend)
            'complementary_match': 0.1,    # Complementary specializations (e.g., Frontend -> Full-Stack)
            'adjacent_match': 0.05,        # Related specializations (e.g., Backend -> DevOps)
            'mismatch_penalty': -0.05      # Different specializations (e.g., Frontend -> Data Science)
        }
    
    def compute_tech_enhanced_similarity(self,
                                       resume_embedding: np.ndarray,
                                       job_embedding: np.ndarray,
                                       resume_metadata: dict,
                                       job_metadata: dict) -> float:
        """Enhanced similarity with tech stack alignment and specialization matching"""
        
        # Base career-aware similarity from parent class
        base_similarity = super().compute_similarity(
            resume_embedding, job_embedding, resume_metadata, job_metadata
        )
        
        # Tech-specific enhancements
        tech_alignment = self._calculate_comprehensive_tech_alignment(
            resume_metadata, job_metadata
        )
        
        # Apply tech alignment bonus (weighted to not dominate base similarity)
        tech_bonus = tech_alignment.overall_score * 0.15  # Max 15% bonus
        
        # Specialization matching bonus
        specialization_bonus = self._calculate_specialization_bonus(
            resume_metadata.get('specialization', 'Software Developer'),
            job_metadata.get('specialization', 'Software Developer')
        )
        
        # Final enhanced similarity
        enhanced_similarity = base_similarity + tech_bonus + specialization_bonus
        
        # Ensure similarity stays in valid range [0, 1]
        return min(max(enhanced_similarity, 0.0), 1.0)
    
    def _calculate_comprehensive_tech_alignment(self, 
                                              resume_metadata: dict, 
                                              job_metadata: dict) -> TechAlignmentScore:
        """Calculate comprehensive tech alignment score between resume and job"""
        
        # Extract tech skills from metadata
        resume_skills = resume_metadata.get('skills', {})
        job_required_skills = job_metadata.get('required_skills', {})
        job_preferred_skills = job_metadata.get('preferred_skills', {})
        
        # Calculate different alignment components
        skill_overlap = self._calculate_skill_overlap_score(
            resume_skills, job_required_skills, job_preferred_skills
        )
        
        stack_compatibility = self._calculate_stack_compatibility_score(
            resume_skills, job_required_skills
        )
        
        experience_alignment = self._calculate_experience_alignment_score(
            resume_metadata, job_metadata
        )
        
        specialization_match = self._check_specialization_compatibility(
            resume_metadata.get('specialization', ''),
            job_metadata.get('specialization', '')
        )
        
        # Combine scores with weights
        overall_score = (
            skill_overlap * 0.4 +
            stack_compatibility * 0.3 +
            experience_alignment * 0.3
        )
        
        return TechAlignmentScore(
            overall_score=overall_score,
            skill_overlap_score=skill_overlap,
            stack_compatibility_score=stack_compatibility,
            experience_alignment_score=experience_alignment,
            specialization_match=specialization_match,
            details={
                'skill_categories_matched': self._count_matching_categories(resume_skills, job_required_skills),
                'high_priority_skills_matched': self._count_high_priority_matches(resume_skills, job_required_skills),
                'stack_diversity_score': self._calculate_stack_diversity(resume_skills)
            }
        )
    
    def _calculate_skill_overlap_score(self, 
                                     resume_skills: Dict[str, List], 
                                     job_required: Dict[str, List], 
                                     job_preferred: Dict[str, List]) -> float:
        """Calculate skill overlap score with category weighting"""
        
        total_weighted_score = 0.0
        total_possible_weight = 0.0
        
        for category, weight in self.tech_skill_weights.items():
            resume_category_skills = self._extract_skill_names(resume_skills.get(category, []))
            required_category_skills = self._extract_skill_names(job_required.get(category, []))
            preferred_category_skills = self._extract_skill_names(job_preferred.get(category, []))
            
            if not required_category_skills and not preferred_category_skills:
                continue
                
            total_possible_weight += weight
            
            # Required skills matching (higher importance)
            if required_category_skills:
                required_overlap = len(resume_category_skills.intersection(required_category_skills))
                required_coverage = required_overlap / len(required_category_skills)
                category_score = required_coverage * 1.0  # Full weight for required skills
            else:
                category_score = 0.0
            
            # Preferred skills matching (bonus points)
            if preferred_category_skills:
                preferred_overlap = len(resume_category_skills.intersection(preferred_category_skills))
                preferred_coverage = preferred_overlap / len(preferred_category_skills)
                category_score += preferred_coverage * 0.3  # Bonus for preferred skills
            
            # Apply stack compatibility bonuses
            compatibility_bonus = self._calculate_category_compatibility_bonus(
                resume_category_skills, required_category_skills
            )
            category_score += compatibility_bonus
            
            total_weighted_score += min(category_score, 1.0) * weight
        
        return total_weighted_score / total_possible_weight if total_possible_weight > 0 else 0.0
    
    def _calculate_stack_compatibility_score(self, 
                                           resume_skills: Dict[str, List], 
                                           job_required: Dict[str, List]) -> float:
        """Calculate technology stack compatibility score"""
        
        compatibility_scores = []
        
        # Get all resume skills as a flat set
        all_resume_skills = set()
        for skill_list in resume_skills.values():
            all_resume_skills.update(self._extract_skill_names(skill_list))
        
        # Get all required job skills as a flat set
        all_required_skills = set()
        for skill_list in job_required.values():
            all_required_skills.update(self._extract_skill_names(skill_list))
        
        # Calculate compatibility for each required skill
        for required_skill in all_required_skills:
            max_compatibility = 0.0
            
            # Check direct match first
            if required_skill in all_resume_skills:
                max_compatibility = 1.0
            else:
                # Check compatibility with resume skills
                for resume_skill in all_resume_skills:
                    compatibility = self._get_skill_compatibility(resume_skill, required_skill)
                    max_compatibility = max(max_compatibility, compatibility)
            
            compatibility_scores.append(max_compatibility)
        
        return np.mean(compatibility_scores) if compatibility_scores else 0.0
    
    def _calculate_experience_alignment_score(self, 
                                            resume_metadata: dict, 
                                            job_metadata: dict) -> float:
        """Calculate experience level alignment score"""
        
        # Extract experience indicators
        resume_experience = self._extract_experience_indicators(resume_metadata)
        job_requirements = self._extract_job_experience_requirements(job_metadata)
        
        alignment_scores = []
        
        # Years of experience alignment
        resume_years = resume_experience.get('total_years', 0)
        required_years = job_requirements.get('min_years', 0)
        
        if required_years > 0:
            if resume_years >= required_years:
                years_score = min(1.0, resume_years / required_years)
            else:
                years_score = resume_years / required_years * 0.7  # Penalty for insufficient experience
            alignment_scores.append(years_score)
        
        # Technology-specific experience alignment
        for tech, required_exp in job_requirements.get('tech_experience', {}).items():
            resume_tech_exp = resume_experience.get('tech_specific', {}).get(tech, 0)
            if required_exp > 0:
                tech_score = min(1.0, resume_tech_exp / required_exp)
                alignment_scores.append(tech_score)
        
        # Project complexity alignment
        resume_complexity = resume_experience.get('complexity_score', 0)
        required_complexity = job_requirements.get('complexity_threshold', 0)
        
        if required_complexity > 0:
            complexity_score = min(1.0, resume_complexity / required_complexity)
            alignment_scores.append(complexity_score)
        
        return np.mean(alignment_scores) if alignment_scores else 0.5
    
    def _extract_skill_names(self, skill_list: List) -> set:
        """Extract skill names from skill objects or strings"""
        if not skill_list:
            return set()
        
        skill_names = set()
        for skill in skill_list:
            if isinstance(skill, TechSkillMatch):
                skill_names.add(skill.skill.lower())
            elif isinstance(skill, str):
                skill_names.add(skill.lower())
            elif isinstance(skill, dict) and 'skill' in skill:
                skill_names.add(skill['skill'].lower())
        
        return skill_names
    
    def _get_skill_compatibility(self, skill1: str, skill2: str) -> float:
        """Get compatibility score between two skills"""
        skill1_lower = skill1.lower()
        skill2_lower = skill2.lower()
        
        # Direct match
        if skill1_lower == skill2_lower:
            return 1.0
        
        # Check compatibility matrix
        compatibility_matrix = self.stack_compatibility_matrix
        
        if skill1 in compatibility_matrix:
            return compatibility_matrix[skill1].get(skill2, 0.0)
        
        if skill2 in compatibility_matrix:
            return compatibility_matrix[skill2].get(skill1, 0.0)
        
        # Default compatibility for unknown skills
        return 0.0
    
    def _calculate_category_compatibility_bonus(self, 
                                              resume_skills: set, 
                                              required_skills: set) -> float:
        """Calculate bonus for technology stack compatibility within a category"""
        if not resume_skills or not required_skills:
            return 0.0
        
        compatibility_scores = []
        for req_skill in required_skills:
            max_compatibility = max(
                (self._get_skill_compatibility(res_skill, req_skill) 
                 for res_skill in resume_skills),
                default=0.0
            )
            compatibility_scores.append(max_compatibility)
        
        avg_compatibility = np.mean(compatibility_scores)
        return max(0.0, (avg_compatibility - 0.5) * 0.2)  # Bonus only above 50% compatibility
    
    def _calculate_specialization_bonus(self, 
                                      resume_specialization: str, 
                                      job_specialization: str) -> float:
        """Calculate bonus for specialization matching"""
        if not resume_specialization or not job_specialization:
            return 0.0
        
        resume_spec = resume_specialization.lower()
        job_spec = job_specialization.lower()
        
        # Exact match
        if resume_spec == job_spec:
            return self.specialization_bonuses['exact_match']
        
        # Define specialization relationships
        specialization_relationships = {
            'frontend developer': ['full-stack developer', 'ui/ux developer'],
            'backend developer': ['full-stack developer', 'devops engineer', 'api developer'],
            'full-stack developer': ['frontend developer', 'backend developer', 'web developer'],
            'data scientist': ['machine learning engineer', 'ai engineer', 'data analyst'],
            'devops engineer': ['backend developer', 'cloud engineer', 'infrastructure engineer'],
            'mobile developer': ['full-stack developer', 'app developer'],
        }
        
        # Complementary match
        if (job_spec in specialization_relationships.get(resume_spec, []) or
            resume_spec in specialization_relationships.get(job_spec, [])):
            return self.specialization_bonuses['complementary_match']
        
        # Adjacent match (check for partial matches)
        if any(word in job_spec for word in resume_spec.split()) or \
           any(word in resume_spec for word in job_spec.split()):
            return self.specialization_bonuses['adjacent_match']
        
        # Different specializations
        return self.specialization_bonuses['mismatch_penalty']
    
    def _check_specialization_compatibility(self, resume_spec: str, job_spec: str) -> bool:
        """Check if specializations are compatible"""
        if not resume_spec or not job_spec:
            return True  # Neutral if unknown
        
        bonus = self._calculate_specialization_bonus(resume_spec, job_spec)
        return bonus >= 0
    
    def _extract_experience_indicators(self, metadata: dict) -> Dict[str, any]:
        """Extract experience indicators from resume metadata"""
        text = metadata.get('text', '')
        
        # Extract years of experience
        years_match = re.search(r'(\d+)[\s\-]*(?:to|\+)?\s*years', text.lower())
        total_years = int(years_match.group(1)) if years_match else 0
        
        # Extract technology-specific experience
        tech_experience = {}
        tech_patterns = {
            'python': r'python.*?(\d+)[\s\-]*years?',
            'java': r'java(?!script).*?(\d+)[\s\-]*years?',
            'javascript': r'javascript.*?(\d+)[\s\-]*years?',
            'react': r'react.*?(\d+)[\s\-]*years?',
            'aws': r'aws.*?(\d+)[\s\-]*years?'
        }
        
        for tech, pattern in tech_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                tech_experience[tech] = int(match.group(1))
        
        # Calculate complexity score based on skills and experience
        skills = metadata.get('skills', {})
        complexity_score = sum(len(skill_list) for skill_list in skills.values())
        
        return {
            'total_years': total_years,
            'tech_specific': tech_experience,
            'complexity_score': complexity_score
        }
    
    def _extract_job_experience_requirements(self, metadata: dict) -> Dict[str, any]:
        """Extract experience requirements from job metadata"""
        text = metadata.get('text', '')
        
        # Extract minimum years requirement
        years_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'minimum\s+(\d+)\s+years?',
            r'at\s+least\s+(\d+)\s+years?'
        ]
        
        min_years = 0
        for pattern in years_patterns:
            match = re.search(pattern, text.lower())
            if match:
                min_years = int(match.group(1))
                break
        
        # Extract complexity threshold based on job requirements
        complexity_indicators = ['architecture', 'lead', 'senior', 'principal', 'staff']
        complexity_threshold = sum(1 for indicator in complexity_indicators if indicator in text.lower())
        
        return {
            'min_years': min_years,
            'tech_experience': {},  # Could be expanded to parse specific tech requirements
            'complexity_threshold': complexity_threshold
        }
    
    def _count_matching_categories(self, resume_skills: dict, job_required: dict) -> int:
        """Count how many skill categories have matches"""
        matching_categories = 0
        
        for category in job_required.keys():
            if category in resume_skills and resume_skills[category] and job_required[category]:
                resume_names = self._extract_skill_names(resume_skills[category])
                required_names = self._extract_skill_names(job_required[category])
                
                if resume_names.intersection(required_names):
                    matching_categories += 1
        
        return matching_categories
    
    def _count_high_priority_matches(self, resume_skills: dict, job_required: dict) -> int:
        """Count matches in high-priority categories"""
        high_priority_categories = ['programming_languages', 'web_frameworks', 'databases']
        high_priority_matches = 0
        
        for category in high_priority_categories:
            if category in job_required and category in resume_skills:
                resume_names = self._extract_skill_names(resume_skills[category])
                required_names = self._extract_skill_names(job_required[category])
                
                matches = len(resume_names.intersection(required_names))
                high_priority_matches += matches
        
        return high_priority_matches
    
    def _calculate_stack_diversity(self, resume_skills: dict) -> float:
        """Calculate diversity score of candidate's tech stack"""
        total_categories = len([cat for cat, skills in resume_skills.items() if skills])
        max_categories = len(self.tech_skill_weights)
        
        return total_categories / max_categories if max_categories > 0 else 0.0

    @lru_cache(maxsize=1000)
    def get_cached_similarity(self, resume_hash: str, job_hash: str, 
                            resume_embedding_hash: str, job_embedding_hash: str) -> Optional[float]:
        """Get cached similarity score if available"""
        cache_key = f"{resume_hash}_{job_hash}_{resume_embedding_hash}_{job_embedding_hash}"
        return self.alignment_cache.get(cache_key)
    
    def cache_similarity(self, resume_hash: str, job_hash: str,
                        resume_embedding_hash: str, job_embedding_hash: str, 
                        similarity: float):
        """Cache similarity score for future use"""
        cache_key = f"{resume_hash}_{job_hash}_{resume_embedding_hash}_{job_embedding_hash}"
        self.alignment_cache[cache_key] = similarity
        
        # Prevent cache from growing too large
        if len(self.alignment_cache) > 10000:
            # Remove oldest entries
            keys_to_remove = list(self.alignment_cache.keys())[:1000]
            for key in keys_to_remove:
                del self.alignment_cache[key]
