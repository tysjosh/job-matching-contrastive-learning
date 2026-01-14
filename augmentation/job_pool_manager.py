"""
Job Pool Manager: Manages job selection for negative generation in contrastive learning.

This module provides ESCO-enhanced job selection for creating diverse negative samples:
- Cross-domain jobs using ESCO domain classification
- Skill-different jobs using ESCO occupation-skill relationships
- Random jobs for easy negatives
- ESCO occupation-based intelligent selection
"""

import json
import random
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class JobPoolManager:
    """
    ESCO-enhanced job pool manager for negative generation in contrastive learning.

    Leverages ESCO domain classification, occupation-skill relationships, and career graph
    data to provide intelligent job selection for creating diverse negative samples.
    """

    def __init__(self, dataset_path: str,
                 esco_config_file: Optional[str] = None,
                 esco_csv_path: str = "dataset/esco/",
                 esco_skills_hierarchy: Optional[Dict] = None,
                 career_graph: Optional[Any] = None):
        """
        Initialize ESCO-enhanced job pool manager.

        Args:
            dataset_path: Path to the dataset JSONL file
            esco_config_file: Path to ESCO career domains config file
            esco_csv_path: Path to ESCO CSV files directory
            esco_skills_hierarchy: ESCO skills hierarchy dictionary
            career_graph: Career graph for domain boundary enforcement
        """
        self.dataset_path = dataset_path
        self.esco_config_file = esco_config_file or 'esco_it_career_domains_refined.json'
        self.esco_csv_path = esco_csv_path
        self.esco_skills_hierarchy = esco_skills_hierarchy
        self.career_graph = career_graph

        # Initialize ESCO components (import here to avoid circular imports)
        self._initialize_esco_components()

        # Job organization structures
        self.all_jobs = []
        self.jobs_by_domain = {}
        self.jobs_by_occupation = {}
        self.jobs_by_skills = {}
        self.job_occupations = {}  # job -> detected occupation mapping

        # Load and organize all jobs
        self._load_all_jobs()
        self._organize_jobs_with_esco()

    def _initialize_esco_components(self):
        """Initialize ESCO data loaders and domain classifier"""
        try:
            # Import here to avoid circular imports
            from augmentation.progression_constraints import ESCODomainLoader, ESCODataLoader, ESCOSkillNormalizer

            # Initialize ESCO domain loader
            self.esco_domain_loader = ESCODomainLoader(self.esco_config_file)

            # Initialize ESCO data loader for occupation-skill relationships
            self.esco_data_loader = ESCODataLoader(self.esco_csv_path)

            # Initialize skill normalizer
            self.skill_normalizer = ESCOSkillNormalizer(
                esco_skills_hierarchy=self.esco_skills_hierarchy,
                esco_csv_path=self.esco_csv_path
            )

            # Load ESCO data
            self.career_domains = self.esco_domain_loader.load_career_domains()
            self.occupation_skill_relations = self.esco_data_loader.load_occupation_skill_relations()

            logger.info(f"ESCO components initialized: {len(self.career_domains)} domains, "
                        f"{len(self.occupation_skill_relations)} occupation-skill relations")

        except Exception as e:
            logger.warning(
                f"Failed to initialize ESCO components: {e}. Falling back to basic mode.")
            self.esco_domain_loader = None
            self.esco_data_loader = None
            self.skill_normalizer = None
            self.career_domains = {}
            self.occupation_skill_relations = {}

    def _load_all_jobs(self):
        """Load all unique jobs from the dataset"""
        seen_jobs = set()

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line.strip())
                        job = record.get('job', {})

                        # Create a unique identifier for the job
                        job_title = job.get('title', '').lower().strip()
                        job_desc = str(job.get('description', {}).get(
                            'original', ''))[:100]
                        job_key = f"{job_title}_{hash(job_desc)}"

                        if job_key not in seen_jobs and job_title:
                            seen_jobs.add(job_key)
                            self.all_jobs.append(job)

            logger.info(
                f"Loaded {len(self.all_jobs)} unique jobs from dataset")

        except Exception as e:
            logger.error(f"Failed to load jobs from {self.dataset_path}: {e}")
            # Fallback: create some default jobs
            self._create_fallback_jobs()

    def _create_fallback_jobs(self):
        """Create fallback jobs if dataset loading fails"""
        self.all_jobs = [
            {
                "title": "Software Engineer",
                "description": {"original": "Develop software applications"},
                "experience_level": "mid"
            },
            {
                "title": "Marketing Manager",
                "description": {"original": "Manage marketing campaigns"},
                "experience_level": "senior"
            },
            {
                "title": "Construction Worker",
                "description": {"original": "Build construction projects"},
                "experience_level": "entry"
            }
        ]
        logger.warning("Using fallback jobs due to loading error")

    def _organize_jobs_with_esco(self):
        """Organize jobs using ESCO domain classification and occupation detection"""
        total_jobs = len(self.all_jobs)
        logger.info(
            f"Organizing {total_jobs} jobs with ESCO classification...")

        for idx, job in enumerate(self.all_jobs, 1):
            # Progress logging every 100 jobs
            if idx % 100 == 0:
                logger.info(
                    f"Organized {idx}/{total_jobs} jobs ({idx/total_jobs*100:.1f}%)...")

            # ESCO-based domain classification
            domain = self._extract_job_domain_esco(job)
            if domain not in self.jobs_by_domain:
                self.jobs_by_domain[domain] = []
            self.jobs_by_domain[domain].append(job)

            # ESCO-based occupation detection
            occupation = self._extract_job_occupation_esco(job)
            if occupation:
                self.job_occupations[id(job)] = occupation
                if occupation not in self.jobs_by_occupation:
                    self.jobs_by_occupation[occupation] = []
                self.jobs_by_occupation[occupation].append(job)

            # ESCO-enhanced skill extraction
            skills = self._extract_job_skills_esco(job)
            for skill in skills:
                if skill not in self.jobs_by_skills:
                    self.jobs_by_skills[skill] = []
                self.jobs_by_skills[skill].append(job)

        logger.info(f"Jobs organized: {len(self.jobs_by_domain)} domains, "
                    f"{len(self.jobs_by_occupation)} occupations, "
                    f"{len(self.jobs_by_skills)} unique skills")

        # Log cache efficiency
        if hasattr(self, '_occupation_cache'):
            logger.info(
                f"Occupation cache size: {len(self._occupation_cache)} unique job titles")

    def _extract_job_domain_esco(self, job: Dict) -> str:
        """Extract domain using ESCO domain classification"""

        # Combine job title and description for domain analysis
        job_text = ""
        if 'title' in job:
            job_text += job['title'] + " "

        if 'description' in job:
            desc = job['description']
            if isinstance(desc, dict) and 'original' in desc:
                job_text += desc['original']
            elif isinstance(desc, str):
                job_text += desc

        # Use ESCO domain classifier
        domain = self.esco_domain_loader.get_domain_for_text(job_text)

        return domain

    def _extract_job_occupation_esco(self, job: Dict) -> Optional[str]:
        """Extract occupation using ESCO occupation matching (optimized with caching)"""
        if not self.career_graph or 'occupations' not in self.career_graph:
            return None

        # Extract job text for occupation matching
        job_title = job.get('title', '').lower().strip()

        # OPTIMIZATION 1: Cache check - if we've seen this title before
        if not hasattr(self, '_occupation_cache'):
            self._occupation_cache = {}

        if job_title in self._occupation_cache:
            return self._occupation_cache[job_title]

        # Build full job text only if needed
        job_text = job_title + " "
        if 'description' in job:
            desc = job['description']
            if isinstance(desc, dict) and 'original' in desc:
                job_text += desc['original']
            elif isinstance(desc, str):
                job_text += desc

        job_text = job_text.lower().strip()
        if not job_text:
            return None

        # OPTIMIZATION 2: Pre-compute job words once
        job_words = set(job_text.split())

        # Enhanced occupation matching (similar to _extract_occupation_from_resume)
        best_match = None
        best_score = 0
        exact_match_threshold = 50  # OPTIMIZATION 3: Early exit if we find a strong match

        for occupation_title in self.career_graph['occupations'].keys():
            occupation_lower = occupation_title.lower()

            # Strategy 1: Exact phrase matching (highest weight)
            if occupation_lower in job_text:
                score = len(occupation_lower) * 3
                if score > best_score:
                    best_score = score
                    best_match = occupation_title
                    # OPTIMIZATION 3: Early exit for strong exact matches
                    if best_score >= exact_match_threshold:
                        break
                    continue

            # Strategy 2: Word-level matching (only if no exact match yet)
            if best_score < exact_match_threshold:
                occupation_words = set(occupation_lower.split())
                common_words = occupation_words & job_words

                if common_words:
                    # Weight by number of matching words and their length
                    score = sum(len(word) for word in common_words)
                    if score > best_score:
                        best_score = score
                        best_match = occupation_title

        # Cache the result
        self._occupation_cache[job_title] = best_match if best_score > 0 else None

        return best_match if best_score > 0 else None

    def _extract_job_skills_esco(self, job: Dict) -> List[str]:
        """Extract and normalize skills using ESCO skill normalizer"""
        skills = set()

        # Get skills from job structure
        if 'skills' in job and isinstance(job['skills'], list):
            for skill in job['skills']:
                if isinstance(skill, dict) and 'name' in skill:
                    normalized_skill = self._normalize_skill(skill['name'])
                    skills.add(normalized_skill)
                elif isinstance(skill, str):
                    normalized_skill = self._normalize_skill(skill)
                    skills.add(normalized_skill)

        # Extract from description keywords
        description = job.get('description', {})
        if isinstance(description, dict):
            keywords = description.get('keywords', [])
            for kw in keywords:
                if isinstance(kw, str):
                    normalized_skill = self._normalize_skill(kw)
                    skills.add(normalized_skill)

        return list(skills)

    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill using ESCO skill normalizer"""
        if self.skill_normalizer:
            return self.skill_normalizer.normalize_skill(skill)
        else:
            return skill.lower().strip()

    def select_cross_domain_job(self, original_job: Dict) -> Dict:
        """
        Select a job from a completely different ESCO domain.

        Args:
            original_job: The original job to find a cross-domain match for

        Returns:
            Dict: Job from a different ESCO domain
        """
        original_domain = self._extract_job_domain_esco(original_job)

        # Get all domains except the original
        other_domains = [domain for domain in self.jobs_by_domain.keys()
                         if domain != original_domain and self.jobs_by_domain[domain]]

        if other_domains:
            # Select random domain and random job from that domain
            selected_domain = random.choice(other_domains)
            return random.choice(self.jobs_by_domain[selected_domain])
        else:
            # Fallback: select any random job
            return self.select_random_job()

    def select_skill_mismatch_job(self, original_job: Dict) -> Dict:
        """
        Select a job with significantly different ESCO-normalized skill requirements.

        Args:
            original_job: The original job to find a skill mismatch for

        Returns:
            Dict: Job with different ESCO skills
        """
        original_skills = set(self._extract_job_skills_esco(original_job))

        best_mismatch = None
        min_overlap = float('inf')

        # Find job with minimal ESCO skill overlap
        for job in self.all_jobs:
            job_skills = set(self._extract_job_skills_esco(job))

            if job_skills:  # Only consider jobs with identifiable skills
                overlap = len(original_skills.intersection(job_skills))
                overlap_ratio = overlap / \
                    len(job_skills) if job_skills else 1.0

                if overlap_ratio < min_overlap:
                    min_overlap = overlap_ratio
                    best_mismatch = job

        return best_mismatch if best_mismatch else self.select_random_job()

    def select_occupation_mismatch_job(self, original_job: Dict) -> Dict:
        """
        Select a job from a different ESCO occupation category.

        Args:
            original_job: The original job to find an occupation mismatch for

        Returns:
            Dict: Job from a different occupation
        """
        original_occupation = self._extract_job_occupation_esco(original_job)

        if not original_occupation or not self.jobs_by_occupation:
            # Fallback to skill mismatch if no occupation detected
            return self.select_skill_mismatch_job(original_job)

        # Get all occupations except the original
        other_occupations = [occ for occ in self.jobs_by_occupation.keys()
                             if occ != original_occupation and self.jobs_by_occupation[occ]]

        if other_occupations:
            # Select random occupation and random job from that occupation
            selected_occupation = random.choice(other_occupations)
            return random.choice(self.jobs_by_occupation[selected_occupation])
        else:
            # Fallback: select cross-domain job
            return self.select_cross_domain_job(original_job)

    def select_random_job(self) -> Dict:
        """
        Select a completely random job from the dataset.

        Returns:
            Dict: Random job
        """
        if self.all_jobs:
            return random.choice(self.all_jobs)
        else:
            # Fallback job if no jobs loaded
            return {
                "title": "Generic Job",
                "description": {"original": "Generic job description"},
                "experience_level": "mid"
            }

    def select_random_job_excluding(self, exclude_job: Dict) -> Dict:
        """
        Select a random job that's different from the excluded job.

        Args:
            exclude_job: Job to exclude from selection

        Returns:
            Dict: Random job different from excluded job
        """
        exclude_title = exclude_job.get('title', '').lower().strip()

        # Filter out jobs with same title
        available_jobs = [job for job in self.all_jobs
                          if job.get('title', '').lower().strip() != exclude_title]

        if available_jobs:
            return random.choice(available_jobs)
        else:
            # If all jobs are same title (unlikely), return any job
            return self.select_random_job()

    def find_matching_job(self, occupation: str, level: str) -> Optional[Dict]:
        """
        Find a job matching a given occupation and experience level.

        Args:
            occupation: The ESCO occupation to match.
            level: The experience level to match (e.g., 'mid', 'senior').

        Returns:
            A matching job dictionary, or None if no match is found.
        """
        if occupation not in self.jobs_by_occupation:
            return None

        candidate_jobs = self.jobs_by_occupation[occupation]

        # Filter by experience level
        level_matched_jobs = [
            job for job in candidate_jobs
            if job.get('experience_level') == level
        ]

        if level_matched_jobs:
            return random.choice(level_matched_jobs)

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the ESCO-enhanced job pool"""
        stats = {
            'total_jobs': len(self.all_jobs),
            'domains': {domain: len(jobs) for domain, jobs in self.jobs_by_domain.items()},
            'occupations': {occ: len(jobs) for occ, jobs in self.jobs_by_occupation.items()},
            'total_skills': len(self.jobs_by_skills),
            'total_occupations_detected': len(self.jobs_by_occupation),
            'top_skills': sorted(self.jobs_by_skills.keys(),
                                 key=lambda x: len(self.jobs_by_skills[x]),
                                 reverse=True)[:10],
            'top_occupations': sorted(self.jobs_by_occupation.keys(),
                                      key=lambda x: len(
                                          self.jobs_by_occupation[x]),
                                      reverse=True)[:10] if self.jobs_by_occupation else [],
            'esco_components_loaded': {
                'domain_loader': self.esco_domain_loader is not None,
                'data_loader': self.esco_data_loader is not None,
                'skill_normalizer': self.skill_normalizer is not None,
                'career_domains_count': len(self.career_domains),
                'occupation_skill_relations_count': len(self.occupation_skill_relations)
            }
        }

        # Add coverage statistics
        jobs_with_occupation = sum(1 for job in self.all_jobs
                                   if id(job) in self.job_occupations)
        stats['coverage'] = {
            'jobs_with_detected_occupation': jobs_with_occupation,
            'occupation_detection_rate': jobs_with_occupation / len(self.all_jobs) if self.all_jobs else 0,
            'average_skills_per_job': sum(len(self._extract_job_skills_esco(job)) for job in self.all_jobs) / len(self.all_jobs) if self.all_jobs else 0
        }

        return stats
