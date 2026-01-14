"""
Downward Transformer (Tdown): Creates junior-level views of resumes

This transformer reduces resume descriptions to reflect how a more junior
professional would describe the same work, emphasizing learning, support, and task execution.
"""

import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Import config loader - required dependency
from augmentation.transformation_config_loader import get_config_loader

# Import paraphraser
from augmentation.career_aware_paraphraser import CareerAwareParaphraser, ParaphrasingResult


class DownwardTransformer:
    """
    Transforms resume content to reflect junior-level perspective.

    Applies Tdown transformation by adding:
    - Support and assistance language
    - Learning and development context
    - Task-focused descriptions
    - Guidance and mentorship indicators
    """

    def __init__(self, metadata_synchronizer=None, enable_paraphrasing: bool = True, 
                 paraphrasing_config: Optional[Dict] = None, config_dir: Optional[str] = None):
        """
        Initialize the downward transformer.

        Args:
            metadata_synchronizer: Optional metadata synchronizer for consistency validation
            enable_paraphrasing: Whether to enable paraphrasing for diversity
            paraphrasing_config: Configuration for paraphrasing behavior
            config_dir: Optional custom config directory path
        
        Raises:
            RuntimeError: If configuration files cannot be loaded
        """
        # Initialize config loader
        self.config_loader = get_config_loader(config_dir)
        
        self._load_transformation_rules()

        # Import metadata synchronizer if not provided
        if metadata_synchronizer is None:
            try:
                from .metadata_synchronizer import MetadataSynchronizer
                self.metadata_synchronizer = MetadataSynchronizer()
            except ImportError:
                logger.warning(
                    "MetadataSynchronizer not available, metadata sync disabled")
                self.metadata_synchronizer = None
        else:
            self.metadata_synchronizer = metadata_synchronizer

        # Initialize paraphrasing
        self.enable_paraphrasing = enable_paraphrasing
        if self.enable_paraphrasing:
            paraphrasing_config = paraphrasing_config or {}
            self.paraphraser = CareerAwareParaphraser(
                preserve_technical_terms=paraphrasing_config.get(
                    'preserve_technical_terms', True),
                min_diversity_score=paraphrasing_config.get(
                    'min_diversity_score', 0.3),
                max_semantic_drift=paraphrasing_config.get(
                    'max_semantic_drift', 0.8)
            )
            logger.info(
                "Downward transformer initialized with paraphrasing enabled")
        else:
            self.paraphraser = None
            logger.info(
                "Downward transformer initialized without paraphrasing")

    def _load_transformation_rules(self):
        """Load transformation rules from config files"""
        
        self.action_verb_downgrades = self.config_loader.load_verb_downgrades()
        self.scope_reducers = self.config_loader.load_scope_reducers()
        self.support_phrases_by_context = self._load_support_phrases_from_config()
        self.learning_enhancement_phrases = self._load_learning_enhancement_from_config()
        self.learning_phrases_by_domain = self._load_learning_phrases_from_config()
        self.task_focused_additions = self.config_loader.load_task_focused_additions()
        self.title_downgrades = self.config_loader.load_title_downgrades()
        
        # Validate required data was loaded
        if not self.action_verb_downgrades:
            raise RuntimeError("Failed to load verb downgrades from config. Check config/transformation_rules/verb_transformations.yaml")
        if not self.scope_reducers:
            raise RuntimeError("Failed to load scope reducers from config. Check config/transformation_rules/scope_transformations.yaml")
        
        # Variety tracking to prevent repetitive patterns
        self._phrase_usage_history = {}
        self._max_phrase_reuse = 3
        self._support_phrase_usage = {}
        self._learning_phrase_usage = {}
        
        logger.info("Loaded transformation rules from config files")
    
    def _load_support_phrases_from_config(self) -> Dict[str, List[str]]:
        """Load support phrases by context from config"""
        result = {}
        for context in ['technical_implementation', 'problem_solving', 'project_management',
                       'research_analysis', 'team_collaboration', 'general']:
            result[context] = self.config_loader.load_support_phrases_by_context(context)
        return result
    
    def _load_learning_enhancement_from_config(self) -> Dict[str, Dict[str, List[str]]]:
        """Load learning enhancement phrases from config"""
        result = {}
        for level in ['entry_level', 'junior_level']:
            result[level] = {}
            for phrase_type in ['technical', 'analytical', 'collaborative']:
                result[level][phrase_type] = self.config_loader.load_learning_enhancement_phrases(level, phrase_type)
        return result
    
    def _load_learning_phrases_from_config(self) -> Dict[str, Dict[str, List[str]]]:
        """Load learning phrases by domain from config"""
        result = {}
        for domain in ['software_development', 'data_science', 'cybersecurity', 
                      'systems_administration', 'product_management']:
            result[domain] = {}
            for level in ['entry', 'junior']:
                result[domain][level] = self.config_loader.load_learning_phrases_by_domain(domain, level)
        return result

    def _get_technical_terms_from_cs_database(self) -> List[str]:
        """Load technical terms from CS skills database for preservation during transformation"""
        import os
        import json

        try:
            # Load CS skills database
            dataset_dir = 'dataset'
            cs_skills_file = os.path.join(dataset_dir, 'cs_skills.json')

            with open(cs_skills_file, 'r', encoding='utf-8') as f:
                cs_skills = json.load(f)

            # Extract technical terms that should be preserved (case-sensitive)
            technical_terms = []

            # Programming languages (uppercase common acronyms)
            programming_languages = cs_skills.get('programming_languages', [])
            for lang in programming_languages:
                # Add uppercase versions of common acronyms
                if lang.lower() in ['sql', 'html', 'css', 'xml', 'json', 'api']:
                    technical_terms.append(lang.upper())
                # Add proper case versions
                technical_terms.append(lang)

            # Cloud platforms and services
            cloud_platforms = cs_skills.get('cloud_platforms', [])
            for platform in cloud_platforms:
                # Handle AWS, GCP, Azure specially
                if 'AWS' in platform:
                    technical_terms.append('AWS')
                elif 'GCP' in platform or 'Google Cloud' in platform:
                    technical_terms.append('GCP')
                elif 'Azure' in platform:
                    technical_terms.append('Azure')
                technical_terms.append(platform)

            # DevOps tools
            devops_categories = [
                'devops_tools', 'cloud_platforms__devops_tools', 'containerization']
            for category in devops_categories:
                if category in cs_skills:
                    for tool in cs_skills[category]:
                        technical_terms.append(tool)

            # Add common technical acronyms
            common_acronyms = ['REST', 'CRUD', 'HTTP',
                               'HTTPS', 'TCP', 'UDP', 'SSH', 'FTP']
            technical_terms.extend(common_acronyms)

            # Remove duplicates while preserving case
            unique_terms = []
            seen_lower = set()
            for term in technical_terms:
                if term and term.lower() not in seen_lower:
                    unique_terms.append(term)
                    seen_lower.add(term.lower())

            # Sort by length (longest first) to prevent shorter terms from breaking longer words
            unique_terms.sort(key=len, reverse=True)
            
            # Filter out single-character terms that could cause regex issues
            unique_terms = [term for term in unique_terms if len(term) > 1 or term in ['C#', 'C++']]

            return unique_terms

        except Exception as e:
            # Fallback to minimal list if database loading fails
            logger.warning(f"Failed to load CS skills database: {e}")
            return ['SQL', 'API', 'AWS', 'GCP', 'HTML', 'CSS', 'XML', 'JSON', 'REST', 'CRUD', 'HTTP', 'HTTPS']

    def transform(self,
                  resume: Dict[str, Any],
                  target_level: str,
                  job_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply downward transformation to create junior-level view.

        Args:
            resume: Original resume data
            target_level: Target junior seniority level
            job_context: Job context for domain-aware transformation

        Returns:
            Dict: Transformed resume with junior-level perspective or None if transformation fails
        """
        # Create transformation checkpoint for rollback
        transformation_checkpoint = {
            'original_resume': resume.copy(),
            'transformation_steps': [],
            'failed_at_step': None
        }

        try:
            transformed_resume = resume.copy()

            # Step 1: Transform experience descriptions
            if 'experience' in resume:
                try:
                    transformed_resume['experience'] = self._transform_experience(
                        resume['experience'], target_level, job_context, resume)
                    transformation_checkpoint['transformation_steps'].append(
                        'experience')
                except Exception as e:
                    logger.warning(f"Experience transformation failed: {e}")
                    transformation_checkpoint['failed_at_step'] = 'experience'
                    return self._handle_transformation_failure(transformation_checkpoint, e)

            # Step 2: Transform skills with comprehensive proficiency handling
            if 'skills' in resume:
                try:
                    transformed_resume['skills'] = self._transform_skills_comprehensive(
                        resume['skills'], target_level, transformed_resume)
                    transformation_checkpoint['transformation_steps'].append(
                        'skills')
                except Exception as e:
                    logger.warning(f"Skills transformation failed: {e}")
                    transformation_checkpoint['failed_at_step'] = 'skills'
                    return self._handle_transformation_failure(transformation_checkpoint, e)

            # Step 3: Transform role/title if present
            if 'role' in resume:
                try:
                    transformed_resume['role'] = self._reduce_job_title(
                        resume['role'], target_level)
                    transformation_checkpoint['transformation_steps'].append(
                        'role')
                except Exception as e:
                    logger.warning(f"Role transformation failed: {e}")
                    transformation_checkpoint['failed_at_step'] = 'role'
                    return self._handle_transformation_failure(transformation_checkpoint, e)

            # Step 4: Transform summary if present
            if 'summary' in resume:
                try:
                    transformed_resume['summary'] = self._transform_summary(
                        resume['summary'], target_level, resume, job_context)
                    transformation_checkpoint['transformation_steps'].append(
                        'summary')
                except Exception as e:
                    logger.warning(f"Summary transformation failed: {e}")
                    transformation_checkpoint['failed_at_step'] = 'summary'
                    return self._handle_transformation_failure(transformation_checkpoint, e)

            # Step 5: Transform responsibilities if present
            if 'responsibilities' in resume:
                try:
                    transformed_resume['responsibilities'] = self._transform_responsibilities_list(
                        resume['responsibilities'], target_level)
                    transformation_checkpoint['transformation_steps'].append(
                        'responsibilities')
                except Exception as e:
                    logger.warning(
                        f"Responsibilities transformation failed: {e}")
                    transformation_checkpoint['failed_at_step'] = 'responsibilities'
                    return self._handle_transformation_failure(transformation_checkpoint, e)

            # Step 6: Update experience level metadata
            transformed_resume['experience_level'] = target_level

            # Step 7: Synchronize metadata for consistency
            if self.metadata_synchronizer:
                try:
                    sync_result = self.metadata_synchronizer.synchronize_experience_metadata(
                        transformed_resume, 'downward', target_level)

                    if sync_result.success:
                        # Update metadata with synchronized values
                        if 'metadata' not in transformed_resume:
                            transformed_resume['metadata'] = {}
                        transformed_resume['metadata'].update(
                            sync_result.synchronized_metadata)
                        transformation_checkpoint['transformation_steps'].append(
                            'metadata_sync')

                        # Log warnings if consistency issues found
                        if sync_result.warnings:
                            logger.warning(
                                f"Metadata sync warnings: {sync_result.warnings}")
                    else:
                        logger.warning(
                            f"Metadata synchronization failed: {sync_result.errors}")
                        # Continue without failing the entire transformation

                except Exception as e:
                    logger.warning(f"Metadata synchronization error: {e}")
                    # Continue without failing the entire transformation

            # Step 8: Add transformation metadata
            transformed_resume['_transformation_meta'] = {
                'type': 'downward',
                'target_level': target_level,
                'applied_rules': self._get_applied_rules(),
                'transformation_steps': transformation_checkpoint['transformation_steps'],
                'metadata_synchronized': self.metadata_synchronizer is not None
            }

            # Step 9: Validate transformation completeness
            if not self._validate_transformation_completeness(transformed_resume, target_level):
                logger.warning("Transformation completeness validation failed")
                return self._handle_transformation_failure(
                    transformation_checkpoint,
                    Exception(
                        "Transformation incomplete - failed completeness validation")
                )

            return transformed_resume

        except Exception as e:
            logger.error(f"Downward transformation failed: {e}")
            transformation_checkpoint['failed_at_step'] = 'unknown'
            return self._handle_transformation_failure(transformation_checkpoint, e)

    def _transform_experience(self,
                              experience: Any,
                              target_level: str,
                              job_context: Dict,
                              resume: Dict[str, Any] = None) -> Any:
        """Transform experience descriptions to junior level"""
        if isinstance(experience, str):
            return self._transform_experience_text(experience, target_level, job_context, resume)
        elif isinstance(experience, list):
            return [self._transform_experience_item(item, target_level, job_context, resume)
                    for item in experience]
        else:
            return experience

    def _transform_experience_text(self,
                                   text: str,
                                   target_level: str,
                                   job_context: Dict,
                                   resume: Dict[str, Any] = None) -> str:
        """Transform experience text to junior-level language with optional paraphrasing"""

        # Step 1: Apply traditional career-level transformation
        transformed_text = self._apply_career_level_transformation(
            text, target_level, job_context, resume)

        # Step 2: Apply paraphrasing for diversity if enabled
        if self.enable_paraphrasing and self.paraphraser:
            paraphrasing_result = self.paraphraser.paraphrase_experience_text(
                text=transformed_text,
                career_level=target_level,
                preserve_technical=True,
                diversity_target=0.3
            )

            if paraphrasing_result.success:
                logger.debug(f"Paraphrasing successful - diversity: {paraphrasing_result.diversity_score:.3f}, "
                             f"technical terms preserved: {paraphrasing_result.technical_terms_preserved}")
                return paraphrasing_result.paraphrased_text
            else:
                logger.debug(
                    f"Paraphrasing failed or insufficient diversity - using traditional transformation")
                return transformed_text
        else:
            return transformed_text

    def _apply_career_level_transformation(self,
                                           text: str,
                                           target_level: str,
                                           job_context: Dict,
                                           resume: Dict[str, Any] = None) -> str:
        """Apply traditional career-level transformation (original logic)"""
        transformed_text = text

        # Preserve technical terms from CS skills database (case-sensitive)
        import re  # Import re at the top of the function to avoid scoping issues
        technical_terms = self._get_technical_terms_from_cs_database()
        preserved_terms = {}

        # Store technical terms with placeholders (using word boundaries for single letters)
        for i, term in enumerate(technical_terms):
            if term in transformed_text:
                placeholder = f"__TECH_TERM_{i}__"
                preserved_terms[placeholder] = term

                # For single-letter terms like "C", use word boundaries to avoid corrupting other words
                if len(term) == 1 and term.isalpha():
                    # Only replace if it's a standalone word
                    pattern = r'\b' + re.escape(term) + r'\b'
                    if re.search(pattern, transformed_text):
                        transformed_text = re.sub(
                            pattern, placeholder, transformed_text)
                else:
                    # For multi-character terms, use simple replacement (already sorted by length)
                    transformed_text = transformed_text.replace(
                        term, placeholder)

        # Apply action verb downgrades
        for senior_verb, junior_verbs in self.action_verb_downgrades.items():
            pattern = r'\b' + re.escape(senior_verb) + r'\b'
            if re.search(pattern, transformed_text, re.IGNORECASE):
                junior_verb = junior_verbs[0]  # Use first option
                transformed_text = re.sub(
                    pattern, junior_verb, transformed_text, flags=re.IGNORECASE)

        # Apply scope reducers
        for large_scope, small_scopes in self.scope_reducers.items():
            pattern = r'\b' + re.escape(large_scope) + r'\b'
            if re.search(pattern, transformed_text, re.IGNORECASE):
                small_scope = small_scopes[0]  # Use first option
                transformed_text = re.sub(
                    pattern, small_scope, transformed_text, flags=re.IGNORECASE)

        # Add responsibility-aware adjustments for junior level
        if resume and 'responsibilities' in resume:
            transformed_text = self._adjust_with_responsibilities(
                transformed_text, resume['responsibilities'], target_level)

        # Modify proficiency language more subtly - for entry and junior levels
        if target_level in ['entry', 'junior']:
            if 'Proficient in' in transformed_text:
                transformed_text = transformed_text.replace(
                    'Proficient in', 'Familiar with', 1)
            elif 'Expert in' in transformed_text:
                transformed_text = transformed_text.replace(
                    'Expert in', 'Familiar with', 1)
            elif 'senior-level experience' in transformed_text:
                transformed_text = transformed_text.replace(
                    'senior-level experience', 'entry-level experience', 1)
            elif 'mid-level experience' in transformed_text:
                transformed_text = transformed_text.replace(
                    'mid-level experience', 'entry-level experience', 1)
        elif target_level == 'mid':
            # For mid-level, make minimal changes
            if 'Expert in' in transformed_text:
                transformed_text = transformed_text.replace(
                    'Expert in', 'Proficient in', 1)
            elif 'senior-level experience' in transformed_text:
                transformed_text = transformed_text.replace(
                    'senior-level experience', 'mid-level experience', 1)

        # Enhance with context-aware support language for junior levels
        if target_level in ['entry', 'junior']:
            transformed_text = self._enhance_with_support_language(
                transformed_text, target_level)

        # Restore technical terms
        for placeholder, term in preserved_terms.items():
            transformed_text = transformed_text.replace(placeholder, term)

        return transformed_text

    def _reduce_job_title(self, title: str, target_level: str) -> str:
        """Reduce job title to reflect junior level"""
        title_lower = title.lower()

        # Remove senior prefixes
        title_downgrades = {
            'senior': ['', 'junior', 'associate'],
            'lead': ['', 'senior', 'associate'],
            'principal': ['senior', 'lead', ''],
            'director': ['manager', 'senior manager', 'lead'],
            'manager': ['associate', 'coordinator', 'specialist']
        }

        for senior_term, junior_options in title_downgrades.items():
            if senior_term in title_lower:
                title = re.sub(rf'\b{senior_term}\s+', '',
                               title, flags=re.IGNORECASE)

        # Add junior prefixes
        if target_level == 'entry' and 'junior' not in title_lower and 'intern' not in title_lower:
            return f"Junior {title}"
        elif target_level == 'entry' and 'associate' not in title_lower:
            return f"Associate {title}"

        return title

    def _transform_summary(self, summary: str, target_level: str, resume: Dict[str, Any] = None, job_context: Dict[str, Any] = None) -> str:
        """Transform summary to reflect junior-level perspective"""
        return self._transform_experience_text(summary, target_level, job_context or {}, resume)

    def _detect_domain_from_context(self, resume: Dict[str, Any], job_context: Dict[str, Any]) -> str:
        """
        Detect the most appropriate domain based on resume content and job context.

        Args:
            resume: Resume data for context analysis
            job_context: Job context information

        Returns:
            Detected domain string
        """
        # Try to get domain from job context first
        if job_context and 'domain' in job_context:
            return job_context['domain']

        # Extract text for domain analysis
        analysis_text = ""

        # Add job context text if available
        if job_context:
            if 'title' in job_context:
                analysis_text += f" {job_context['title']}"
            if 'description' in job_context:
                analysis_text += f" {job_context['description']}"
            if 'requirements' in job_context:
                analysis_text += f" {job_context['requirements']}"

        # Add resume text for additional context
        if resume:
            if 'role' in resume:
                analysis_text += f" {resume['role']}"
            if 'experience' in resume:
                if isinstance(resume['experience'], str):
                    analysis_text += f" {resume['experience']}"
                elif isinstance(resume['experience'], list):
                    for exp in resume['experience']:
                        if isinstance(exp, str):
                            analysis_text += f" {exp}"
                        elif isinstance(exp, dict) and 'description' in exp:
                            analysis_text += f" {exp['description']}"
            if 'skills' in resume:
                if isinstance(resume['skills'], list):
                    for skill in resume['skills']:
                        if isinstance(skill, str):
                            analysis_text += f" {skill}"
                        elif isinstance(skill, dict) and 'name' in skill:
                            analysis_text += f" {skill['name']}"

        # Use ESCO domain detection if available
        try:
            from .progression_constraints import ESCODomainLoader
            esco_loader = ESCODomainLoader()
            detected_domain = esco_loader.get_domain_for_text(analysis_text)
            if detected_domain:
                return detected_domain
        except Exception as e:
            logger.warning(f"ESCO domain detection failed: {e}")

        # Fallback to keyword-based detection
        return self._fallback_domain_detection(analysis_text)

    def _fallback_domain_detection(self, text: str) -> str:
        """
        Fallback domain detection using keyword matching.

        Args:
            text: Text to analyze for domain keywords

        Returns:
            Detected domain string
        """
        text_lower = text.lower()

        # Domain keyword mappings
        domain_keywords = {
            'software_development': [
                'developer', 'engineer', 'programming', 'coding', 'software', 'web', 'mobile',
                'frontend', 'backend', 'fullstack', 'javascript', 'python', 'java', 'react',
                'angular', 'node', 'api', 'database', 'sql', 'nosql', 'git', 'agile', 'scrum'
            ],
            'data_science': [
                'data scientist', 'machine learning', 'ml', 'ai', 'artificial intelligence',
                'analytics', 'statistics', 'python', 'r', 'tensorflow', 'pytorch', 'pandas',
                'numpy', 'visualization', 'modeling', 'algorithm', 'big data', 'hadoop', 'spark'
            ],
            'cybersecurity': [
                'security', 'cybersecurity', 'infosec', 'penetration testing', 'vulnerability',
                'firewall', 'encryption', 'compliance', 'risk assessment', 'incident response',
                'threat', 'malware', 'forensics', 'siem', 'soc', 'cissp', 'ceh', 'oscp'
            ],
            'systems_administration': [
                'system administrator', 'sysadmin', 'devops', 'infrastructure', 'server',
                'linux', 'windows', 'unix', 'cloud', 'aws', 'azure', 'gcp', 'docker',
                'kubernetes', 'ansible', 'terraform', 'monitoring', 'networking', 'virtualization'
            ],
            'product_management': [
                'product manager', 'product owner', 'roadmap', 'stakeholder', 'requirements',
                'user story', 'market research', 'competitive analysis', 'go-to-market',
                'product strategy', 'user experience', 'ux', 'ui', 'metrics', 'kpi', 'analytics'
            ]
        }

        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score

        # Return domain with highest score, default to software_development
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return 'software_development'  # Default fallback

    def _get_learning_phrases_for_domain(self, domain: str, target_level: str) -> List[str]:
        """Get learning phrases for specific domain and level"""
        domain_phrases = self.learning_phrases_by_domain.get(domain, {})
        level_phrases = domain_phrases.get(target_level, [])

        # Fallback to generic phrases if domain-specific not available
        if not level_phrases:
            fallback_phrases = [
                'under senior guidance',
                'as part of the team',
                'while learning best practices',
                'following established procedures',
                'with mentorship support',
                'as a team member',
                'under supervision',
                'through hands-on experience',
                'with continuous learning',
                'following team standards'
            ]
            return fallback_phrases

        return level_phrases

    def _select_diverse_learning_phrase(self, phrases: List[str], context_key: str) -> str:
        """
        Select a learning phrase with diversity tracking to prevent repetitive patterns.

        Args:
            phrases: List of available learning phrases
            context_key: Unique key for tracking usage (e.g., combination of domain, level, content hash)

        Returns:
            Selected learning phrase
        """
        if not phrases:
            return "under guidance"

        # Initialize usage tracking for this context if not exists
        if context_key not in self._phrase_usage_history:
            self._phrase_usage_history[context_key] = {}

        usage_history = self._phrase_usage_history[context_key]

        # Find phrases that haven't been overused
        available_phrases = []
        for phrase in phrases:
            usage_count = usage_history.get(phrase, 0)
            if usage_count < self._max_phrase_reuse:
                available_phrases.append(phrase)

        # If all phrases are overused, reset usage history and use all phrases
        if not available_phrases:
            self._phrase_usage_history[context_key] = {}
            available_phrases = phrases

        # Select phrase based on content hash for consistency within same content
        # but with variety across different content
        phrase_index = hash(context_key) % len(available_phrases)
        selected_phrase = available_phrases[phrase_index]

        # Update usage history
        usage_history[selected_phrase] = usage_history.get(
            selected_phrase, 0) + 1

        return selected_phrase



    def _transform_experience_item(self,
                                   item: Dict,
                                   target_level: str,
                                   job_context: Dict,
                                   resume: Dict[str, Any] = None) -> Dict:
        """Transform individual experience item"""
        if not isinstance(item, dict):
            return item

        transformed_item = item.copy()

        # Transform description fields
        description_fields = ['description', 'summary', 'details']
        for field in description_fields:
            if field in item:
                if isinstance(item[field], str):
                    transformed_item[field] = self._transform_experience_text(
                        item[field], target_level, job_context, resume)
                elif isinstance(item[field], list):
                    transformed_item[field] = [
                        self._transform_experience_text(
                            desc, target_level, job_context, resume)
                        if isinstance(desc, str) else desc
                        for desc in item[field]
                    ]

        # Transform responsibilities if present
        if 'responsibilities' in item:
            responsibilities = item['responsibilities']
            if isinstance(responsibilities, dict):
                transformed_item['responsibilities'] = self._transform_responsibilities(
                    responsibilities, target_level, job_context)
            elif isinstance(responsibilities, str):
                transformed_item['responsibilities'] = self._transform_experience_text(
                    responsibilities, target_level, job_context, resume)

        # Transform job title to reflect junior level
        if 'title' in item:
            transformed_item['title'] = self._reduce_job_title(
                item['title'], target_level)

        return transformed_item

    def _transform_responsibilities(self,
                                    responsibilities: Dict,
                                    target_level: str,
                                    job_context: Dict) -> Dict:
        """Transform responsibilities dictionary"""
        transformed_resp = responsibilities.copy()

        # Transform technical terms to more junior language
        if 'technical_terms' in responsibilities:
            transformed_resp['technical_terms'] = self._reduce_technical_terms(
                responsibilities['technical_terms'], target_level)

        # Reduce achievements to task completion
        if 'achievements' in responsibilities:
            transformed_resp['achievements'] = self._reduce_achievements(
                responsibilities['achievements'], target_level)

        # Reduce impact to individual contribution
        if 'impact' in responsibilities:
            transformed_resp['impact'] = self._reduce_impact(
                responsibilities['impact'], target_level)

        return transformed_resp

    def _transform_skills_comprehensive(self, skills: List, target_level: str, transformed_resume: Dict[str, Any]) -> List:
        """
        Comprehensive skills transformation with proficiency field handling and metadata alignment.

        Args:
            skills: Original skills list
            target_level: Target experience level
            transformed_resume: Resume being transformed (for context)

        Returns:
            Transformed skills list with consistent proficiency levels
        """
        if not isinstance(skills, list):
            return skills

        transformed_skills = []

        for skill in skills:
            if isinstance(skill, str):
                # Transform string skills with learning context
                transformed_skill = self._transform_string_skill(
                    skill, target_level)
                transformed_skills.append(transformed_skill)

            elif isinstance(skill, dict):
                # Transform skill objects with comprehensive proficiency handling
                transformed_skill = self._transform_skill_object(
                    skill, target_level)
                transformed_skills.append(transformed_skill)

            else:
                # Keep other types as-is
                transformed_skills.append(skill)

        # Validate skill array consistency with experience level
        if self.metadata_synchronizer:
            try:
                # Extract experience text for consistency validation
                experience_text = self._extract_experience_text(
                    transformed_resume)
                consistency_report = self.metadata_synchronizer.validate_skill_metadata_consistency(
                    transformed_skills, experience_text, target_level
                )

                # Log consistency issues
                if consistency_report.consistency_score < 0.8:
                    logger.warning(
                        f"Skills consistency issues: {consistency_report.misalignments}")

            except Exception as e:
                logger.warning(f"Skills consistency validation failed: {e}")

        return transformed_skills

    def _transform_string_skill(self, skill: str, target_level: str) -> str:
        """Transform string skill with appropriate learning context"""
        if target_level == 'entry':
            return f"Learning {skill}"
        elif target_level == 'junior':
            return f"Developing skills in {skill}"
        elif target_level == 'mid':
            return f"Working with {skill}"
        else:
            return skill

    def _transform_skill_object(self, skill: Dict[str, Any], target_level: str) -> Dict[str, Any]:
        """
        Transform skill object with comprehensive proficiency field handling.

        Handles both 'level' and 'proficiency' fields consistently.
        """
        transformed_skill = skill.copy()

        # Handle 'level' field (legacy)
        if 'level' in skill:
            transformed_skill['level'] = self._reduce_skill_level(
                skill['level'])

        # Handle 'proficiency' field (primary)
        if 'proficiency' in skill:
            transformed_skill['proficiency'] = self._reduce_skill_proficiency(
                skill['proficiency'], target_level)

        # If neither field exists, add proficiency based on target level
        if 'proficiency' not in skill and 'level' not in skill:
            transformed_skill['proficiency'] = self._get_default_proficiency_for_level(
                target_level)

        # Ensure consistency between level and proficiency if both exist
        if 'level' in transformed_skill and 'proficiency' in transformed_skill:
            # Align level with proficiency (proficiency takes precedence)
            aligned_level = self._align_level_with_proficiency(
                transformed_skill['proficiency'])
            if aligned_level:
                transformed_skill['level'] = aligned_level

        # Add learning context to skill name if appropriate
        if 'name' in skill and target_level in ['entry', 'junior']:
            original_name = skill['name']
            if not any(indicator in original_name.lower() for indicator in ['learning', 'developing', 'basic']):
                if target_level == 'entry':
                    transformed_skill['name'] = f"Basic {original_name}"
                elif target_level == 'junior':
                    transformed_skill['name'] = f"Developing {original_name}"

        return transformed_skill

    def _get_default_proficiency_for_level(self, target_level: str) -> str:
        """Get default proficiency level for target experience level"""
        level_to_proficiency = {
            'entry': 'beginner',
            'junior': 'beginner',
            'mid': 'intermediate',
            'senior': 'advanced',
            'lead': 'expert',
            'principal': 'expert'
        }
        return level_to_proficiency.get(target_level.lower(), 'intermediate')

    def _align_level_with_proficiency(self, proficiency: str) -> Optional[str]:
        """Align level field with proficiency field for consistency"""
        proficiency_to_level = {
            'beginner': 'entry',
            'intermediate': 'mid',
            'advanced': 'senior',
            'expert': 'lead'
        }
        return proficiency_to_level.get(proficiency.lower())

    def _extract_experience_text(self, resume: Dict[str, Any]) -> str:
        """Extract experience text from resume for validation"""
        text_parts = []

        if 'experience' in resume:
            exp = resume['experience']
            if isinstance(exp, str):
                text_parts.append(exp)
            elif isinstance(exp, list):
                for item in exp:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        for field in ['description', 'summary', 'responsibilities']:
                            if field in item and isinstance(item[field], str):
                                text_parts.append(item[field])

        if 'summary' in resume and isinstance(resume['summary'], str):
            text_parts.append(resume['summary'])

        return ' '.join(text_parts)

    def _handle_transformation_failure(self, checkpoint: Dict[str, Any], error: Exception) -> Optional[Dict[str, Any]]:
        """
        Handle transformation failure with rollback mechanism.

        Args:
            checkpoint: Transformation checkpoint with original data and steps
            error: Exception that caused the failure

        Returns:
            None to indicate transformation failure (prevents embedding collapse)
        """
        failed_step = checkpoint.get('failed_at_step', 'unknown')
        completed_steps = checkpoint.get('transformation_steps', [])

        logger.error(
            f"Transformation failed at step '{failed_step}' after completing {completed_steps}: {error}")

        # Log rollback information for debugging
        logger.info(
            f"Rolling back transformation - completed steps: {completed_steps}")

        # Return None instead of original resume to prevent embedding collapse
        # This follows the enhanced fallback strategy of failing fast rather than
        # returning potentially corrupted data
        return None

    def _validate_transformation_completeness(self, transformed_resume: Dict[str, Any], target_level: str) -> bool:
        """
        Validate that transformation is complete and consistent.

        Args:
            transformed_resume: Transformed resume data
            target_level: Target experience level

        Returns:
            True if transformation is complete and consistent
        """
        try:
            # Check that experience level was updated
            if transformed_resume.get('experience_level') != target_level:
                logger.warning(
                    f"Experience level not updated: expected {target_level}, got {transformed_resume.get('experience_level')}")
                return False

            # Check that skills were transformed if they exist
            if 'skills' in transformed_resume:
                skills = transformed_resume['skills']
                if isinstance(skills, list) and skills:
                    # Verify at least some skills have appropriate proficiency for target level
                    expected_proficiencies = self._get_expected_proficiencies_for_level(
                        target_level)
                    skill_proficiencies = []

                    for skill in skills:
                        if isinstance(skill, dict) and 'proficiency' in skill:
                            skill_proficiencies.append(
                                skill['proficiency'].lower())

                    if skill_proficiencies:
                        # At least 50% of skills should have appropriate proficiency
                        appropriate_count = sum(
                            1 for prof in skill_proficiencies if prof in expected_proficiencies)
                        if appropriate_count / len(skill_proficiencies) < 0.5:
                            logger.warning(
                                f"Skills proficiency not appropriately downgraded for {target_level}")
                            return False

            # Check that transformation metadata was added
            if '_transformation_meta' not in transformed_resume:
                logger.warning("Transformation metadata missing")
                return False

            meta = transformed_resume['_transformation_meta']
            if meta.get('type') != 'downward' or meta.get('target_level') != target_level:
                logger.warning("Transformation metadata inconsistent")
                return False

            return True

        except Exception as e:
            logger.error(f"Transformation completeness validation error: {e}")
            return False

    def _get_expected_proficiencies_for_level(self, target_level: str) -> List[str]:
        """Get expected proficiency levels for target experience level"""
        level_proficiencies = {
            'entry': ['beginner'],
            'junior': ['beginner', 'intermediate'],
            'mid': ['intermediate', 'advanced'],
            'senior': ['advanced', 'expert'],
            'lead': ['expert'],
            'principal': ['expert']
        }
        return level_proficiencies.get(target_level.lower(), ['intermediate'])



    def _reduce_technical_terms(self, terms: List[str], target_level: str) -> List[str]:
        """Reduce technical terms to more basic language"""
        reduced_terms = []
        for term in terms:
            # Simplify architectural terms for junior levels
            if target_level == 'entry':
                if 'architecture' in term.lower():
                    reduced_terms.append(term.replace(
                        'architecture', 'development'))
                elif 'system design' in term.lower():
                    reduced_terms.append(
                        term.replace('system design', 'coding'))
                else:
                    reduced_terms.append(term)
            else:
                reduced_terms.append(term)

        return reduced_terms

    def _reduce_achievements(self, achievements: List[str], target_level: str) -> List[str]:
        """Reduce achievements to task completion language"""
        reduced = []
        for achievement in achievements:
            # Focus on task completion rather than business impact
            if target_level == 'entry':
                # Add learning context
                reduced.append(f"Successfully completed {achievement.lower()}")
            else:
                reduced.append(achievement)

        return reduced

    def _reduce_impact(self, impact: List[str], target_level: str) -> List[str]:
        """Reduce impact descriptions to individual contribution"""
        reduced = []
        for impact_item in impact:
            if target_level == 'entry':
                # Focus on individual learning and contribution
                reduced.append(f"Contributed to {impact_item.lower()}")
            elif target_level == 'mid':
                # Focus on team-level contribution
                reduced.append(f"Helped achieve {impact_item.lower()}")
            else:
                reduced.append(impact_item)

        return reduced

    def _reduce_skill_level(self, current_level: str) -> str:
        """Reduce skill proficiency level"""
        level_regression = {
            'principal': 'lead',
            'lead': 'senior',
            'senior': 'mid',
            'mid': 'junior',
            'junior': 'entry',
            'entry': 'entry'  # Stay at bottom
        }
        return level_regression.get(current_level.lower(), current_level)

    def _reduce_skill_proficiency(self, current_proficiency: str, target_level: str) -> str:
        """Reduce skill proficiency based on target experience level"""
        # Map experience levels to appropriate skill proficiency levels
        level_to_proficiency_mapping = {
            'entry': 'beginner',
            'junior': 'beginner',
            'mid': 'intermediate',
            'senior': 'advanced',
            'lead': 'expert',
            'principal': 'expert'
        }

        # Get target proficiency based on experience level
        target_proficiency = level_to_proficiency_mapping.get(
            target_level.lower())

        if not target_proficiency:
            return current_proficiency

        # Proficiency progression hierarchy
        proficiency_hierarchy = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3,
            'expert': 4
        }

        current_rank = proficiency_hierarchy.get(
            current_proficiency.lower(), 2)
        target_rank = proficiency_hierarchy.get(target_proficiency.lower(), 2)

        # Only reduce, never elevate in downward transformation
        if target_rank < current_rank:
            return target_proficiency
        else:
            return current_proficiency

    def _add_learning_context(self, summary: str, target_level: str, resume: Dict[str, Any] = None, job_context: Dict[str, Any] = None) -> str:
        """Add learning and development context to summary with domain awareness"""
        if target_level not in ['entry', 'junior', 'mid']:
            return summary

        # Detect domain from context
        domain = self._detect_domain_from_context(
            resume or {}, job_context or {})

        # Get domain-specific learning phrases
        learning_phrases = self._get_learning_phrases_for_domain(
            domain, target_level)

        if learning_phrases:
            # Create context key for diversity tracking
            context_key = f"{domain}_{target_level}_summary_{hash(summary) % 1000}"

            # Select diverse learning phrase
            learning_phrase = self._select_diverse_learning_phrase(
                learning_phrases, context_key)
            return f"{summary} {learning_phrase}"

        # Fallback to original logic if domain-specific phrases not available
        if target_level == 'entry':
            # Add entry-level learning language
            learning_additions = [
                'eager to learn and grow',
                'seeking to develop technical skills',
                'motivated to contribute to team success',
                'focused on building foundational expertise'
            ]
            addition = learning_additions[hash(
                summary) % len(learning_additions)]
            return f"{summary} {addition}"
        elif target_level in ['junior', 'mid']:
            # Add mid-level development context
            development_additions = [
                'developing expertise in',
                'growing technical capabilities',
                'building on foundational skills',
                'expanding knowledge through practice'
            ]
            addition = development_additions[hash(
                summary) % len(development_additions)]
            return f"{summary} {addition}"

        return summary

    def _detect_content_context(self, text: str) -> str:
        """
        Detect the primary context of the text content for appropriate support language selection.

        Args:
            text: Text content to analyze

        Returns:
            Context category for support phrase selection
        """
        text_lower = text.lower()

        # Technical implementation context
        technical_keywords = [
            'develop', 'code', 'program', 'implement', 'build', 'create', 'design',
            'software', 'application', 'system', 'api', 'database', 'algorithm'
        ]

        # Problem solving context
        problem_keywords = [
            'debug', 'troubleshoot', 'fix', 'resolve', 'solve', 'investigate',
            'diagnose', 'optimize', 'improve', 'enhance', 'refactor'
        ]

        # Project management context
        project_keywords = [
            'manage', 'coordinate', 'plan', 'organize', 'schedule', 'deliver',
            'execute', 'oversee', 'lead', 'direct', 'supervise'
        ]

        # Research/analysis context
        research_keywords = [
            'analyze', 'research', 'study', 'examine', 'evaluate', 'assess',
            'investigate', 'review', 'survey', 'explore', 'data', 'metrics'
        ]

        # Team collaboration context
        team_keywords = [
            'collaborate', 'team', 'group', 'peer', 'colleague', 'partner',
            'coordinate', 'communicate', 'meeting', 'discussion'
        ]

        # Score each context
        context_scores = {
            'technical_implementation': sum(1 for keyword in technical_keywords if keyword in text_lower),
            'problem_solving': sum(1 for keyword in problem_keywords if keyword in text_lower),
            'project_management': sum(1 for keyword in project_keywords if keyword in text_lower),
            'research_analysis': sum(1 for keyword in research_keywords if keyword in text_lower),
            'team_collaboration': sum(1 for keyword in team_keywords if keyword in text_lower)
        }

        # Return context with highest score, default to general
        if any(score > 0 for score in context_scores.values()):
            return max(context_scores, key=context_scores.get)

        return 'general'

    def _select_context_aware_support_phrase(self, text: str, target_level: str) -> str:
        """
        Select appropriate support phrase based on content context and target level.

        Args:
            text: Original text content for context analysis
            target_level: Target experience level

        Returns:
            Context-appropriate support phrase
        """
        # Detect content context
        context = self._detect_content_context(text)

        # Get available phrases for this context
        available_phrases = self.support_phrases_by_context.get(
            context, self.support_phrases_by_context['general'])

        # Create tracking key for variety
        tracking_key = f"{context}_{target_level}"

        # Initialize usage tracking if needed
        if tracking_key not in self._support_phrase_usage:
            self._support_phrase_usage[tracking_key] = {}

        usage_history = self._support_phrase_usage[tracking_key]

        # Find phrases that haven't been overused
        underused_phrases = []
        for phrase in available_phrases:
            usage_count = usage_history.get(phrase, 0)
            if usage_count < self._max_phrase_reuse:
                underused_phrases.append(phrase)

        # Reset if all phrases are overused
        if not underused_phrases:
            self._support_phrase_usage[tracking_key] = {}
            underused_phrases = available_phrases

        # Select phrase based on content hash for consistency
        phrase_index = hash(text + target_level) % len(underused_phrases)
        selected_phrase = underused_phrases[phrase_index]

        # Update usage tracking
        usage_history[selected_phrase] = usage_history.get(
            selected_phrase, 0) + 1

        return selected_phrase

    def _select_learning_enhancement_phrase(self, text: str, target_level: str, domain: str) -> str:
        """
        Select appropriate learning enhancement phrase based on content and domain.

        Args:
            text: Original text content
            target_level: Target experience level (entry/junior)
            domain: Domain context (technical/analytical/collaborative)

        Returns:
            Appropriate learning enhancement phrase
        """
        if target_level not in ['entry', 'junior']:
            return ""

        # Map target level to phrase category
        level_key = f"{target_level}_level"

        # Get domain-specific phrases
        level_phrases = self.learning_enhancement_phrases.get(level_key, {})
        domain_phrases = level_phrases.get(
            domain, level_phrases.get('technical', []))

        if not domain_phrases:
            return ""

        # Create tracking key
        tracking_key = f"{level_key}_{domain}"

        # Initialize usage tracking
        if tracking_key not in self._learning_phrase_usage:
            self._learning_phrase_usage[tracking_key] = {}

        usage_history = self._learning_phrase_usage[tracking_key]

        # Find underused phrases
        underused_phrases = []
        for phrase in domain_phrases:
            usage_count = usage_history.get(phrase, 0)
            if usage_count < self._max_phrase_reuse:
                underused_phrases.append(phrase)

        # Reset if all overused
        if not underused_phrases:
            self._learning_phrase_usage[tracking_key] = {}
            underused_phrases = domain_phrases

        # Select based on content hash
        phrase_index = hash(text + domain) % len(underused_phrases)
        selected_phrase = underused_phrases[phrase_index]

        # Update usage
        usage_history[selected_phrase] = usage_history.get(
            selected_phrase, 0) + 1

        return selected_phrase

    def _enhance_with_support_language(self, text: str, target_level: str) -> str:
        """
        Enhance text with context-aware support language for junior levels.

        Args:
            text: Original text to enhance
            target_level: Target experience level

        Returns:
            Enhanced text with appropriate support language
        """
        if target_level not in ['entry', 'junior']:
            return text

        # Check if text already has support language
        existing_support_indicators = [
            'assisted', 'helped', 'supported', 'contributed', 'participated',
            'collaborated', 'learned', 'mentored', 'guided', 'trained'
        ]

        if any(indicator in text.lower() for indicator in existing_support_indicators):
            # Text already has support language, just add learning enhancement
            domain = self._detect_learning_domain(text)
            learning_phrase = self._select_learning_enhancement_phrase(
                text, target_level, domain)
            if learning_phrase:
                return f"{text.rstrip('.')} {learning_phrase}."
            return text

        # Transform text to include support language
        enhanced_text = text

        # Replace leadership verbs with support language
        leadership_patterns = [
            (r'\b(led|managed|directed|oversaw)\b',
             lambda m: self._select_context_aware_support_phrase(text, target_level)),
            (r'\b(developed|created|built|designed)\b',
             lambda m: self._select_context_aware_support_phrase(text, target_level)),
            (r'\b(implemented|delivered|executed)\b',
             lambda m: self._select_context_aware_support_phrase(text, target_level))
        ]

        for pattern, replacement_func in leadership_patterns:
            if re.search(pattern, enhanced_text, re.IGNORECASE):
                replacement = replacement_func(None)
                enhanced_text = re.sub(
                    pattern, replacement, enhanced_text, count=1, flags=re.IGNORECASE)
                break

        # Add learning enhancement
        domain = self._detect_learning_domain(enhanced_text)
        learning_phrase = self._select_learning_enhancement_phrase(
            enhanced_text, target_level, domain)
        if learning_phrase:
            enhanced_text = f"{enhanced_text.rstrip('.')} {learning_phrase}."

        return enhanced_text

    def _detect_learning_domain(self, text: str) -> str:
        """
        Detect the learning domain for appropriate enhancement phrase selection.

        Args:
            text: Text to analyze

        Returns:
            Learning domain (technical/analytical/collaborative)
        """
        text_lower = text.lower()

        # Technical domain indicators
        technical_indicators = [
            'code', 'program', 'develop', 'software', 'system', 'application',
            'database', 'api', 'algorithm', 'framework', 'library', 'tool'
        ]

        # Analytical domain indicators
        analytical_indicators = [
            'analyze', 'data', 'research', 'study', 'metrics', 'statistics',
            'report', 'insight', 'trend', 'pattern', 'evaluation', 'assessment'
        ]

        # Collaborative domain indicators
        collaborative_indicators = [
            'team', 'collaborate', 'coordinate', 'communicate', 'meeting',
            'discussion', 'presentation', 'stakeholder', 'client', 'partner'
        ]

        # Score domains
        technical_score = sum(
            1 for indicator in technical_indicators if indicator in text_lower)
        analytical_score = sum(
            1 for indicator in analytical_indicators if indicator in text_lower)
        collaborative_score = sum(
            1 for indicator in collaborative_indicators if indicator in text_lower)

        # Return highest scoring domain
        scores = {
            'technical': technical_score,
            'analytical': analytical_score,
            'collaborative': collaborative_score
        }

        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)

        return 'technical'  # Default to technical

    def _get_applied_rules(self) -> List[str]:
        """Get list of transformation rules that were applied"""
        return [
            'action_verb_downgrades',
            'scope_reducers',
            'context_aware_support_language',
            'learning_enhancement_phrases',
            'comprehensive_skill_proficiency_reduction',
            'metadata_synchronization'
        ]

    def _transform_responsibilities_list(self, responsibilities: List[str], target_level: str) -> List[str]:
        """Transform responsibilities to reflect junior-level language with enhanced support phrases"""
        if not responsibilities:
            return responsibilities

        transformed_responsibilities = []

        # Enhanced junior-level responsibility mappings with context awareness
        responsibility_downgrades = {
            'driving strategic outcomes and business results': 'contributing to team goals and deliverables',
            'leading organizational change and adaptation': 'adapting to team changes and new processes',
            'orchestrating complex initiatives and programs': 'supporting project tasks and assignments',
            'delivering enterprise-scale solutions': 'completing assigned development tasks',
            'architecting solutions for complex business challenges': 'solving coding problems and bug fixes',
            'leading cross-functional teams and stakeholders': 'collaborating with team members and peers',
            'establishing and optimizing organizational processes': 'following established team processes',
            'designing and architecting scalable solutions': 'implementing features and components',
            'overseeing operational excellence and strategy': 'supporting daily operations and maintenance',
            'architecting and optimizing enterprise systems': 'maintaining and updating system components'
        }

        for responsibility in responsibilities:
            transformed = responsibility.lower().strip()

            # Apply responsibility downgrades
            downgraded = False
            for senior_resp, junior_resp in responsibility_downgrades.items():
                if senior_resp in transformed:
                    # Enhance with context-aware support language
                    enhanced_resp = self._enhance_with_support_language(
                        junior_resp, target_level)
                    transformed_responsibilities.append(enhanced_resp)
                    downgraded = True
                    break

            if not downgraded:
                # Apply context-aware transformation for unmapped responsibilities
                if target_level in ['entry', 'junior']:
                    # Use context-aware support phrases
                    support_phrase = self._select_context_aware_support_phrase(
                        responsibility, target_level)

                    if 'delivering' in transformed:
                        adjusted = f"{support_phrase} {responsibility} under supervision"
                    elif 'managing' in transformed:
                        adjusted = f"{support_phrase} {responsibility} as part of team"
                    elif 'leading' in transformed:
                        adjusted = f"participating in {responsibility} with team support"
                    elif 'developing' in transformed or 'creating' in transformed:
                        adjusted = f"{support_phrase} {responsibility}"
                    else:
                        adjusted = f"learning {responsibility} with mentorship"

                    # Add learning enhancement
                    enhanced_adjusted = self._enhance_with_support_language(
                        adjusted, target_level)
                    transformed_responsibilities.append(enhanced_adjusted)

                elif target_level == 'mid':
                    if 'delivering' in transformed:
                        adjusted = f"contributing to {responsibility} with team coordination"
                    elif 'managing' in transformed:
                        adjusted = f"supporting {responsibility} with growing independence"
                    else:
                        adjusted = f"developing skills in {responsibility}"

                    # Add moderate learning enhancement for mid-level
                    enhanced_adjusted = self._enhance_with_support_language(
                        adjusted, target_level)
                    transformed_responsibilities.append(enhanced_adjusted)
                else:
                    transformed_responsibilities.append(responsibility)

        return transformed_responsibilities

    def _adjust_with_responsibilities(self, text: str, responsibilities: List[str], target_level: str) -> str:
        """Adjust text with responsibility-aware language for junior levels"""
        if not responsibilities:
            return text

        adjusted_text = text

        # Check if responsibilities suggest support/learning language
        support_indicators = ['helping', 'supporting',
                              'assisting', 'contributing', 'learning']
        has_support_context = any(indicator in ' '.join(responsibilities).lower()
                                  for indicator in support_indicators)

        if has_support_context and target_level in ['entry', 'junior']:
            # Enhance with support/learning language if not already present
            support_words = ['assisted', 'helped', 'supported',
                             'contributed', 'learned', 'participated']
            if not any(word in adjusted_text.lower() for word in support_words):
                # Transform leadership language to support language
                import re
                sentences = re.split(r'(?<=[.!?])\s+', adjusted_text)
                if sentences and len(sentences) > 0:
                    first_sentence = sentences[0].strip()
                    if not any(word in first_sentence.lower() for word in support_words):
                        # Transform first sentence to include support context
                        if first_sentence.lower().startswith(('led', 'managed', 'drove', 'spearheaded')):
                            first_word = first_sentence.split()[0]
                            if first_word.lower() == 'led':
                                enhanced_first = first_sentence.replace(
                                    first_word, 'Assisted in', 1)
                            elif first_word.lower() == 'managed':
                                enhanced_first = first_sentence.replace(
                                    first_word, 'Helped manage', 1)
                            elif first_word.lower() in ['drove', 'spearheaded']:
                                enhanced_first = first_sentence.replace(
                                    first_word, 'Participated in', 1)
                            else:
                                enhanced_first = f"Supported {first_sentence.lower()}"
                            sentences[0] = enhanced_first
                            adjusted_text = ' '.join(sentences)

        return adjusted_text

    def get_paraphrasing_statistics(self) -> Dict[str, Any]:
        """Get statistics about paraphrasing usage"""
        if not self.enable_paraphrasing or not self.paraphraser:
            return {
                'paraphrasing_enabled': False,
                'message': 'Paraphrasing is disabled'
            }

        paraphrasing_stats = self.paraphraser.get_paraphrasing_statistics()

        return {
            'paraphrasing_enabled': True,
            'paraphrasing_stats': paraphrasing_stats,
            'support_phrase_usage': len(self._support_phrase_usage),
            'learning_phrase_usage': len(self._learning_phrase_usage),
            'max_phrase_reuse': self._max_phrase_reuse
        }

    def reset_diversity_tracking(self):
        """Reset diversity tracking for new augmentation batch"""
        # Reset traditional diversity tracking
        self._support_phrase_usage.clear()
        self._learning_phrase_usage.clear()

        # Reset paraphrasing diversity tracking
        if self.enable_paraphrasing and self.paraphraser:
            self.paraphraser.reset_diversity_tracking()

    def configure_paraphrasing(self,
                               enable: bool = None,
                               min_diversity_score: float = None,
                               max_semantic_drift: float = None,
                               preserve_technical_terms: bool = None):
        """
        Configure paraphrasing settings dynamically.

        Args:
            enable: Enable or disable paraphrasing
            min_diversity_score: Minimum diversity score to achieve
            max_semantic_drift: Maximum allowed semantic drift
            preserve_technical_terms: Whether to preserve technical terms
        """
        if enable is not None:
            self.enable_paraphrasing = enable

            if enable and not self.paraphraser:
                # Initialize paraphraser if enabling for the first time
                self.paraphraser = CareerAwareParaphraser()
                logger.info("Paraphrasing enabled and initialized")
            elif not enable:
                logger.info("Paraphrasing disabled")

        if self.paraphraser and enable is not False:
            # Update paraphraser configuration
            if min_diversity_score is not None:
                self.paraphraser.min_diversity_score = min_diversity_score
            if max_semantic_drift is not None:
                self.paraphraser.max_semantic_drift = max_semantic_drift
            if preserve_technical_terms is not None:
                self.paraphraser.preserve_technical_terms = preserve_technical_terms

            logger.info(f"Paraphrasing configuration updated")
