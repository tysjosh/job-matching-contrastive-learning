"""
Upward Transformer (Tup): Creates senior-level views of resumes

This transformer elevates resume descriptions to reflect how a more senior
professional would describe the same work, adding ownership, impact, and scale.
"""

from augmentation.career_aware_paraphraser import CareerAwareParaphraser, ParaphrasingResult
from augmentation.transformation_config_loader import get_config_loader
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Import config loader - required dependency

# Import paraphraser


class UpwardTransformer:
    """
    Transforms resume content to reflect senior-level perspective.

    Applies Tup transformation by adding:
    - Ownership and leadership language
    - Business impact and metrics
    - Strategic and architectural thinking
    - Scale and scope amplification
    """

    def __init__(self, enable_paraphrasing: bool = True, paraphrasing_config: Optional[Dict] = None,
                 config_dir: Optional[str] = None):
        """
        Initialize the upward transformer.

        Args:
            enable_paraphrasing: Whether to enable paraphrasing for diversity
            paraphrasing_config: Configuration for paraphrasing behavior
            config_dir: Optional custom config directory path

        Raises:
            RuntimeError: If configuration files cannot be loaded
        """
        # Initialize config loader
        self.config_loader = get_config_loader(config_dir)

        self._load_transformation_rules()

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
                "Upward transformer initialized with paraphrasing enabled")
        else:
            self.paraphraser = None
            logger.info("Upward transformer initialized without paraphrasing")

    def _load_transformation_rules(self):
        """Load transformation rules from config files"""

        self.action_verb_upgrades = self.config_loader.load_verb_upgrades()
        self.scope_amplifiers = self.config_loader.load_scope_amplifiers()
        self.impact_phrases_by_domain = self.config_loader.load_all_impact_phrases()
        self.ownership_phrases = self.config_loader.get_ownership_phrases()
        self.strategic_additions = self.config_loader.get_strategic_additions()
        self.title_upgrades = self.config_loader.load_title_upgrades()

        # Validate required data was loaded
        if not self.action_verb_upgrades:
            raise RuntimeError(
                "Failed to load verb upgrades from config. Check config/transformation_rules/verb_transformations.yaml")
        if not self.scope_amplifiers:
            raise RuntimeError(
                "Failed to load scope amplifiers from config. Check config/transformation_rules/scope_transformations.yaml")

        # Variety tracking to prevent repetitive patterns
        self._phrase_usage_history = {}
        self._max_phrase_reuse = 3

        logger.info("Loaded transformation rules from config files")

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
            # This fixes the issue where "C" would corrupt "Coursera" → "__TECH_TERM_X__oursera"
            unique_terms.sort(key=len, reverse=True)

            # Filter out single-character terms that could cause regex issues
            # Single chars like "C", "R" are too ambiguous and cause false matches
            unique_terms = [term for term in unique_terms if len(term) > 1 or term in [
                'C#', 'C++']]

            return unique_terms

        except Exception as e:
            # Fallback to hardcoded list if database loading fails
            return ['SQL', 'API', 'AWS', 'GCP', 'HTML', 'CSS', 'XML', 'JSON', 'REST', 'CRUD', 'HTTP', 'HTTPS']

    def transform(self,
                  resume: Dict[str, Any],
                  target_level: str,
                  job_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply upward transformation to create senior-level view.

        Args:
            resume: Original resume data
            target_level: Target seniority level
            job_context: Job context for domain-aware transformation

        Returns:
            Dict: Transformed resume with senior-level perspective
        """
        # Create a deep copy for transformation
        transformed_resume = self._deep_copy_resume(resume)
        original_resume = self._deep_copy_resume(resume)  # Backup for rollback

        transformation_steps = []  # Track transformation steps for rollback

        try:
            # Step 1: Transform experience descriptions
            if 'experience' in resume:
                old_experience = transformed_resume.get('experience')
                transformed_resume['experience'] = self._transform_experience(
                    resume['experience'], target_level, job_context, resume)
                transformation_steps.append(('experience', old_experience))

            # Step 2: Transform skills with progression awareness
            if 'skills' in resume:
                old_skills = transformed_resume.get('skills')
                transformed_resume['skills'] = self._transform_skills(
                    resume['skills'], target_level)
                transformation_steps.append(('skills', old_skills))

            # Step 3: Transform role/title if present
            if 'role' in resume:
                old_role = transformed_resume.get('role')
                transformed_resume['role'] = self._elevate_job_title(
                    resume['role'], target_level)
                transformation_steps.append(('role', old_role))

            # Step 4: Transform summary if present
            if 'summary' in resume:
                old_summary = transformed_resume.get('summary')
                transformed_resume['summary'] = self._transform_summary(
                    resume['summary'], target_level, resume)
                transformation_steps.append(('summary', old_summary))

            # Step 5: Transform responsibilities if present
            if 'responsibilities' in resume:
                old_responsibilities = transformed_resume.get(
                    'responsibilities')
                transformed_resume['responsibilities'] = self._transform_responsibilities(
                    resume['responsibilities'], target_level)
                transformation_steps.append(
                    ('responsibilities', old_responsibilities))

            # Step 6: Update experience level metadata
            old_experience_level = transformed_resume.get('experience_level')
            transformed_resume['experience_level'] = target_level
            transformation_steps.append(
                ('experience_level', old_experience_level))

            # Step 7: Synchronize metadata with transformed content
            sync_result = self._synchronize_metadata(
                transformed_resume, target_level)
            if not sync_result['success']:
                logger.warning(
                    f"Metadata synchronization failed: {sync_result['errors']}")
                # Continue with transformation but log the issue
            else:
                transformation_steps.append(
                    ('metadata_sync', sync_result['original_metadata']))

            # Step 8: Validate transformation completeness
            validation_result = self._validate_transformation_completeness(
                original_resume, transformed_resume, target_level)

            if not validation_result['is_complete']:
                logger.warning(
                    f"Transformation incomplete: {validation_result['missing_elements']}")
                # Attempt to fix incomplete transformation
                if not self._fix_incomplete_transformation(transformed_resume, validation_result, target_level):
                    # If fix fails, rollback
                    logger.error(
                        "Failed to fix incomplete transformation, rolling back")
                    return self._rollback_transformation(original_resume, transformation_steps)

            # Step 9: Add transformation metadata
            transformed_resume['_transformation_meta'] = {
                'type': 'upward',
                'target_level': target_level,
                'applied_rules': self._get_applied_rules(),
                'transformation_quality': validation_result.get('quality_score', 0.8),
                'metadata_synchronized': sync_result['success']
            }

            return transformed_resume

        except Exception as e:
            logger.error(f"Upward transformation failed: {e}")
            # Rollback on any failure
            return self._rollback_transformation(original_resume, transformation_steps)

    def _transform_experience(self,
                              experience: Any,
                              target_level: str,
                              job_context: Dict,
                              resume: Dict[str, Any] = None) -> Any:
        """Transform experience descriptions to senior level"""
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
        """Transform experience text to senior-level language with optional paraphrasing"""

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

        # Apply action verb upgrades
        for junior_verb, senior_verbs in self.action_verb_upgrades.items():
            pattern = r'\b' + re.escape(junior_verb) + r'\b'
            if re.search(pattern, transformed_text, re.IGNORECASE):
                senior_verb = senior_verbs[0]  # Use first option
                transformed_text = re.sub(
                    pattern, senior_verb, transformed_text, flags=re.IGNORECASE)

        # Apply scope amplifiers
        for small_scope, large_scopes in self.scope_amplifiers.items():
            pattern = r'\b' + re.escape(small_scope) + r'\b'
            if re.search(pattern, transformed_text, re.IGNORECASE):
                large_scope = large_scopes[0]  # Use first option
                transformed_text = re.sub(
                    pattern, large_scope, transformed_text, flags=re.IGNORECASE)

        # Enhance proficiency language
        if 'Proficient in' in transformed_text:
            transformed_text = transformed_text.replace(
                'Proficient in', 'Expert in', 1)
        elif 'Familiar with' in transformed_text:
            transformed_text = transformed_text.replace(
                'Familiar with', 'Proficient in', 1)

        # Add responsibility-aware enhancements
        if resume and 'responsibilities' in resume:
            transformed_text = self._enhance_with_responsibilities(
                transformed_text, resume['responsibilities'], target_level)

        # Add impact context if not already present
        if not any(phrase in transformed_text.lower() for phrase in
                   ['resulting', 'leading to', 'improving', 'enhancing', 'driving']):
            # Detect domain from context
            domain = self._detect_domain_from_context(
                resume or {}, job_context)

            # Get impact phrases from domain-specific collection
            impact_phrases = self._get_impact_phrases_for_domain(
                domain, target_level)

            if impact_phrases:
                # Create context key for diversity tracking
                context_key = f"{domain}_{target_level}_{hash(text) % 1000}"

                # Select diverse impact phrase
                impact_phrase = self._select_diverse_impact_phrase(
                    impact_phrases, context_key)
                transformed_text = f"{transformed_text.rstrip('.')}, {impact_phrase}"

        # Add strategic context for senior+ levels
        if target_level in ['lead', 'principal']:
            strategic_addition = self.strategic_additions[
                hash(text) % len(self.strategic_additions)]
            transformed_text = f"{transformed_text.rstrip('.')} {strategic_addition}"

        # Restore technical terms
        for placeholder, term in preserved_terms.items():
            transformed_text = transformed_text.replace(placeholder, term)

        return transformed_text

    def _elevate_job_title(self, title: str, target_level: str) -> str:
        """Elevate job title to reflect seniority"""
        if not title:
            return title

        title_lower = title.lower()

        # Check if title already has seniority markers
        seniority_markers = ['senior', 'lead', 'principal', 'chief', 'head',
                             'director', 'vp', 'vice president', 'staff', 'expert']
        has_seniority = any(
            marker in title_lower for marker in seniority_markers)

        # If already has appropriate level, return as-is
        if target_level == 'senior' and any(marker in title_lower for marker in ['senior', 'lead']):
            return title
        elif target_level == 'lead' and any(marker in title_lower for marker in ['lead', 'principal', 'staff']):
            return title
        elif target_level == 'principal' and any(marker in title_lower for marker in ['principal', 'chief', 'director']):
            return title

        # Apply title upgrades based on common patterns
        title_upgrades = {
            'analyst': ['senior analyst', 'lead analyst', 'principal analyst'],
            'developer': ['senior developer', 'lead developer', 'principal engineer'],
            'engineer': ['senior engineer', 'staff engineer', 'principal engineer'],
            'designer': ['senior designer', 'lead designer', 'design director'],
            'manager': ['senior manager', 'director', 'vice president'],
            'scientist': ['senior scientist', 'lead scientist', 'principal scientist'],
            'administrator': ['senior administrator', 'lead administrator', 'principal administrator'],
            'architect': ['senior architect', 'lead architect', 'principal architect'],
            'consultant': ['senior consultant', 'lead consultant', 'principal consultant'],
            'specialist': ['senior specialist', 'lead specialist', 'principal specialist'],
            'coordinator': ['senior coordinator', 'lead coordinator', 'manager'],
            'technician': ['senior technician', 'lead technician', 'principal technician']
        }

        # Try pattern matching first
        for base_title, upgrades in title_upgrades.items():
            if base_title in title_lower:
                if target_level == 'senior':
                    return upgrades[0]
                elif target_level == 'lead':
                    return upgrades[1] if len(upgrades) > 1 else upgrades[0]
                elif target_level == 'principal':
                    return upgrades[2] if len(upgrades) > 2 else upgrades[-1]

        # Generic fallback: preserve capitalization and add prefix
        # Handle titles that start with articles or adjectives
        words = title.split()

        # Determine the prefix based on target level
        prefix = None
        if target_level == 'senior':
            prefix = 'Senior'
        elif target_level == 'lead':
            prefix = 'Lead'
        elif target_level == 'principal':
            prefix = 'Principal'

        if prefix and not has_seniority:
            # For multi-word titles, intelligently insert the prefix
            # e.g., "Database Administrator" -> "Senior Database Administrator"
            # e.g., "Construction Manager" -> "Senior Construction Manager"
            if words[0].lower() in ['the', 'a', 'an']:
                # "The Manager" -> "The Senior Manager"
                return f"{words[0]} {prefix} {' '.join(words[1:])}"
            else:
                # Most common case: prepend prefix
                return f"{prefix} {title}"

        return title

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

        # Transform title if present
        if 'title' in item:
            transformed_item['title'] = self._elevate_job_title(
                item['title'], target_level)

        return transformed_item

    def _transform_responsibilities(self, responsibilities: List[str], target_level: str) -> List[str]:
        """Transform responsibilities to reflect senior-level language"""
        if not responsibilities:
            return responsibilities

        transformed_responsibilities = []

        # Senior-level responsibility mappings
        responsibility_upgrades = {
            'delivering results': 'driving strategic outcomes and business results',
            'adapting to dynamic environments': 'leading organizational change and adaptation',
            'managing tasks': 'orchestrating complex initiatives and programs',
            'completing projects': 'delivering enterprise-scale solutions',
            'solving problems': 'architecting solutions for complex business challenges',
            'working with teams': 'leading cross-functional teams and stakeholders',
            'following processes': 'establishing and optimizing organizational processes',
            'implementing solutions': 'designing and architecting scalable solutions',
            'supporting operations': 'overseeing operational excellence and strategy',
            'maintaining systems': 'architecting and optimizing enterprise systems'
        }

        for responsibility in responsibilities:
            transformed = responsibility.lower().strip()

            # Apply responsibility upgrades
            for basic_resp, senior_resp in responsibility_upgrades.items():
                if basic_resp in transformed:
                    transformed_responsibilities.append(senior_resp)
                    break
            else:
                # If no direct mapping, enhance with senior-level language
                if target_level in ['lead', 'principal']:
                    if 'delivering' in transformed:
                        enhanced = f"strategically {responsibility} with measurable business impact"
                    elif 'managing' in transformed:
                        enhanced = f"leading and {responsibility} across enterprise scale"
                    elif 'developing' in transformed:
                        enhanced = f"architecting and {responsibility} for scalable solutions"
                    else:
                        enhanced = f"driving {responsibility} with strategic oversight"
                    transformed_responsibilities.append(enhanced)
                elif target_level == 'senior':
                    if 'delivering' in transformed:
                        enhanced = f"effectively {responsibility} with quality focus"
                    elif 'managing' in transformed:
                        enhanced = f"proficiently {responsibility} with team coordination"
                    else:
                        enhanced = f"skillfully {responsibility} with best practices"
                    transformed_responsibilities.append(enhanced)
                else:
                    transformed_responsibilities.append(responsibility)

        return transformed_responsibilities

    def _enhance_with_responsibilities(self, text: str, responsibilities: List[str], target_level: str) -> str:
        """Enhance text with responsibility-aware language based on extracted responsibilities"""
        if not responsibilities:
            return text

        # Map common responsibility patterns to text enhancements
        enhancement_mapping = {
            'delivering results': ['achieving', 'accomplishing', 'executing'],
            'adapting to dynamic environments': ['flexible', 'agile', 'responsive'],
            'managing tasks': ['coordinating', 'organizing', 'overseeing'],
            'problem solving': ['troubleshooting', 'optimizing', 'resolving'],
            'team collaboration': ['leading', 'mentoring', 'facilitating'],
            'process improvement': ['streamlining', 'enhancing', 'optimizing']
        }

        enhanced_text = text

        # Check if any responsibilities suggest leadership/ownership language
        leadership_indicators = ['delivering',
                                 'managing', 'leading', 'overseeing', 'driving']
        has_leadership = any(indicator in ' '.join(responsibilities).lower()
                             for indicator in leadership_indicators)

        if has_leadership and target_level in ['senior', 'lead', 'principal']:
            # Enhance with ownership language if not already present
            ownership_words = ['led', 'owned',
                               'drove', 'spearheaded', 'orchestrated']
            if not any(word in enhanced_text.lower() for word in ownership_words):
                # Add ownership context to the beginning of sentences
                import re
                sentences = re.split(r'(?<=[.!?])\s+', enhanced_text)
                if sentences and len(sentences) > 0:
                    first_sentence = sentences[0].strip()
                    if not any(word in first_sentence.lower() for word in ownership_words):
                        # Transform first sentence to include ownership
                        if first_sentence.lower().startswith(('developed', 'created', 'built', 'implemented')):
                            first_word = first_sentence.split()[0]
                            if first_word.lower() in ['developed', 'created', 'built']:
                                enhanced_first = first_sentence.replace(
                                    first_word, 'Led development of', 1)
                            elif first_word.lower() == 'implemented':
                                enhanced_first = first_sentence.replace(
                                    first_word, 'Spearheaded implementation of', 1)
                            else:
                                enhanced_first = f"Owned {first_sentence.lower()}"
                            sentences[0] = enhanced_first
                            enhanced_text = ' '.join(sentences)

        return enhanced_text

    def _transform_skills(self, skills: List, target_level: str) -> List:
        """Transform skills list to reflect senior-level expertise"""
        if not isinstance(skills, list):
            return skills

        transformed_skills = []
        for skill in skills:
            if isinstance(skill, str):
                # Add proficiency indicators for senior levels
                if target_level in ['lead', 'principal']:
                    transformed_skills.append(f"Expert in {skill}")
                elif target_level == 'senior':
                    transformed_skills.append(f"Advanced {skill}")
                else:
                    transformed_skills.append(skill)
            elif isinstance(skill, dict):
                # Transform skill objects
                transformed_skill = skill.copy()

                # Handle both 'level' and 'proficiency' fields
                if 'level' in skill:
                    transformed_skill['level'] = self._elevate_skill_level(
                        skill['level'])

                if 'proficiency' in skill:
                    transformed_skill['proficiency'] = self._elevate_skill_proficiency(
                        skill['proficiency'], target_level)

                transformed_skills.append(transformed_skill)
            else:
                transformed_skills.append(skill)

        return transformed_skills

    def _transform_summary(self, summary: Any, target_level: str) -> Any:
        """Transform summary to senior-level language"""
        if isinstance(summary, str):
            return self._add_leadership_context(summary, target_level)
        elif isinstance(summary, dict):
            transformed_summary = summary.copy()
            if 'text' in summary:
                transformed_summary['text'] = self._add_leadership_context(
                    summary['text'], target_level)
            return transformed_summary
        else:
            return summary

    def _elevate_technical_terms(self, terms: List[str], target_level: str) -> List[str]:
        """Elevate technical terms to more senior language"""
        elevated_terms = []
        for term in terms:
            # Add architectural/design context for senior levels
            if target_level in ['lead', 'principal']:
                if 'development' in term.lower():
                    elevated_terms.append(term.replace(
                        'development', 'architecture'))
                elif 'coding' in term.lower():
                    elevated_terms.append(
                        term.replace('coding', 'system design'))
                else:
                    elevated_terms.append(term)
            else:
                elevated_terms.append(term)

        return elevated_terms

    def _enhance_achievements(self, achievements: List[str], target_level: str) -> List[str]:
        """Enhance achievements with business impact language"""
        enhanced = []
        for achievement in achievements:
            # Add business context and metrics
            if target_level in ['senior', 'lead', 'principal']:
                if not any(word in achievement.lower() for word in
                           ['improved', 'increased', 'reduced', 'enhanced']):
                    enhanced.append(
                        f"{achievement}, improving system efficiency")
                else:
                    enhanced.append(achievement)
            else:
                enhanced.append(achievement)

        return enhanced

    def _amplify_impact(self, impact: List[str], target_level: str) -> List[str]:
        """Amplify impact descriptions for senior levels"""
        amplified = []
        for impact_item in impact:
            if target_level in ['lead', 'principal']:
                # Add strategic and organizational impact
                amplified.append(f"{impact_item} across multiple teams")
            elif target_level == 'senior':
                # Add team and project-level impact
                amplified.append(f"{impact_item} for the development team")
            else:
                amplified.append(impact_item)

        return amplified

    def _elevate_skill_level(self, current_level: str) -> str:
        """Elevate skill proficiency level"""
        level_progression = {
            'entry': 'junior',
            'junior': 'mid',
            'mid': 'senior',
            'senior': 'lead',
            'lead': 'principal',
            'principal': 'principal'  # Stay at top
        }
        return level_progression.get(current_level.lower(), current_level)

    def _elevate_skill_proficiency(self, current_proficiency: str, target_level: str) -> str:
        """Elevate skill proficiency based on target experience level"""
        # Map experience levels to appropriate skill proficiency levels
        level_to_proficiency_mapping = {
            'entry': 'beginner',
            'junior': 'intermediate',
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

        # Only elevate, never downgrade in upward transformation
        if target_rank > current_rank:
            return target_proficiency
        else:
            return current_proficiency

    def _add_leadership_context(self, summary: str, target_level: str) -> str:
        """Add leadership and strategic context to summary"""
        if target_level in ['lead', 'principal']:
            # Add leadership language
            leadership_additions = [
                'with proven leadership experience',
                'experienced in team management and mentoring',
                'skilled in strategic planning and execution',
                'focused on driving technical excellence'
            ]
            addition = leadership_additions[hash(
                summary) % len(leadership_additions)]
            return f"{summary} {addition}"
        elif target_level == 'senior':
            # Add senior-level context
            senior_additions = [
                'with deep technical expertise',
                'experienced in complex problem solving',
                'skilled in system design and architecture',
                'focused on delivering high-quality solutions'
            ]
            addition = senior_additions[hash(summary) % len(senior_additions)]
            return f"{summary} {addition}"

        return summary

    def _get_applied_rules(self) -> List[str]:
        """Get list of transformation rules that were applied"""
        return [
            'action_verb_upgrades',
            'scope_amplifiers',
            'impact_enhancement',
            'leadership_context'
        ]

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

    def _get_impact_phrases_for_domain(self, domain: str, target_level: str) -> List[str]:
        """Get impact phrases for specific domain and level"""
        domain_phrases = self.impact_phrases_by_domain.get(domain, {})
        level_phrases = domain_phrases.get(target_level, [])

        # Fallback to generic phrases if domain-specific not available
        if not level_phrases:
            fallback_phrases = [
                'resulting in improved system performance',
                'leading to enhanced code quality',
                'driving technical excellence',
                'improving development efficiency',
                'enhancing team productivity'
            ]
            return fallback_phrases

        return level_phrases

    def _select_diverse_impact_phrase(self, phrases: List[str], context_key: str) -> str:
        """
        Select an impact phrase with diversity tracking to prevent repetitive patterns.

        Args:
            phrases: List of available impact phrases
            context_key: Unique key for tracking usage (e.g., combination of domain, level, content hash)

        Returns:
            Selected impact phrase
        """
        if not phrases:
            return "resulting in improved outcomes"

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

    def _deep_copy_resume(self, resume: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of resume data"""
        import copy
        return copy.deepcopy(resume)

    def _synchronize_metadata(self, resume: Dict[str, Any], target_level: str) -> Dict[str, Any]:
        """Synchronize metadata with transformed content using MetadataSynchronizer"""
        try:
            # Import here to avoid circular imports
            from .metadata_synchronizer import MetadataSynchronizer

            synchronizer = MetadataSynchronizer()

            # Store original metadata for rollback
            original_metadata = resume.get('metadata', {}).copy()

            # Synchronize experience metadata
            sync_result = synchronizer.synchronize_experience_metadata(
                resume=resume,
                transformation_type='upward',
                target_level=target_level
            )

            if sync_result.success:
                # Update resume with synchronized metadata
                if 'metadata' not in resume:
                    resume['metadata'] = {}
                resume['metadata'].update(sync_result.synchronized_metadata)

                return {
                    'success': True,
                    'updated_fields': sync_result.updated_fields,
                    'original_metadata': original_metadata,
                    'warnings': sync_result.warnings
                }
            else:
                return {
                    'success': False,
                    'errors': sync_result.errors,
                    'original_metadata': original_metadata
                }

        except Exception as e:
            logger.error(f"Metadata synchronization error: {e}")
            return {
                'success': False,
                'errors': [f"Synchronization failed: {str(e)}"],
                'original_metadata': resume.get('metadata', {})
            }

    def _validate_transformation_completeness(self,
                                              original: Dict[str, Any],
                                              transformed: Dict[str, Any],
                                              target_level: str) -> Dict[str, Any]:
        """Validate that transformation is complete and consistent"""
        missing_elements = []
        quality_score = 1.0

        # Check if all skill objects have been properly transformed
        if 'skills' in transformed and isinstance(transformed['skills'], list):
            for i, skill in enumerate(transformed['skills']):
                if isinstance(skill, dict):
                    # Check if proficiency was updated for skill objects
                    if 'proficiency' in skill:
                        proficiency = skill['proficiency'].lower()
                        expected_proficiencies = self._get_expected_proficiencies(
                            target_level)
                        if proficiency not in expected_proficiencies:
                            missing_elements.append(
                                f"skills[{i}].proficiency not elevated to target level")
                            quality_score -= 0.1

        # Check if experience level metadata is consistent
        if transformed.get('experience_level') != target_level:
            missing_elements.append("experience_level metadata not updated")
            quality_score -= 0.2

        # Check if job title reflects target level (if present)
        if 'role' in transformed:
            title_level = self._extract_title_level(transformed['role'])
            if not self._is_title_appropriate_for_level(title_level, target_level):
                missing_elements.append(
                    "job title not elevated to match target level")
                quality_score -= 0.1

        # Check if experience text contains appropriate seniority language
        if 'experience' in transformed:
            exp_text = str(transformed['experience']).lower()
            senior_indicators = ['senior', 'lead',
                                 'architect', 'expert', 'advanced']
            if target_level in ['senior', 'lead', 'principal'] and not any(indicator in exp_text for indicator in senior_indicators):
                missing_elements.append(
                    "experience text lacks senior-level language")
                quality_score -= 0.1

        return {
            'is_complete': len(missing_elements) == 0,
            'missing_elements': missing_elements,
            'quality_score': max(0.0, quality_score)
        }

    def _fix_incomplete_transformation(self,
                                       resume: Dict[str, Any],
                                       validation_result: Dict[str, Any],
                                       target_level: str) -> bool:
        """Attempt to fix incomplete transformation"""
        try:
            missing_elements = validation_result['missing_elements']

            for element in missing_elements:
                if 'proficiency not elevated' in element:
                    # Fix skill proficiency issues
                    self._fix_skill_proficiency_issues(resume, target_level)
                elif 'experience_level metadata' in element:
                    # Fix experience level metadata
                    resume['experience_level'] = target_level
                elif 'job title not elevated' in element:
                    # Fix job title issues
                    if 'role' in resume:
                        resume['role'] = self._elevate_job_title(
                            resume['role'], target_level)

            return True

        except Exception as e:
            logger.error(f"Failed to fix incomplete transformation: {e}")
            return False

    def _fix_skill_proficiency_issues(self, resume: Dict[str, Any], target_level: str):
        """Fix skill proficiency issues in skill arrays"""
        if 'skills' in resume and isinstance(resume['skills'], list):
            for skill in resume['skills']:
                if isinstance(skill, dict) and 'proficiency' in skill:
                    skill['proficiency'] = self._elevate_skill_proficiency(
                        skill['proficiency'], target_level)

    def _rollback_transformation(self,
                                 original_resume: Dict[str, Any],
                                 transformation_steps: List[tuple]) -> Dict[str, Any]:
        """Rollback transformation to original state"""
        logger.warning("Rolling back transformation due to failure")

        # Return original resume with failure metadata
        rollback_resume = self._deep_copy_resume(original_resume)
        rollback_resume['_transformation_meta'] = {
            'type': 'upward',
            'status': 'failed',
            'rollback_applied': True,
            'failed_steps': len(transformation_steps)
        }

        return rollback_resume

    def _get_expected_proficiencies(self, target_level: str) -> List[str]:
        """Get expected proficiency levels for target experience level"""
        level_to_proficiencies = {
            'entry': ['beginner', 'intermediate'],
            'junior': ['beginner', 'intermediate'],
            'mid': ['intermediate', 'advanced'],
            'senior': ['advanced', 'expert'],
            'lead': ['expert'],
            'principal': ['expert']
        }
        return level_to_proficiencies.get(target_level.lower(), ['intermediate', 'advanced'])

    def _extract_title_level(self, title: str) -> str:
        """Extract seniority level from job title"""
        title_lower = title.lower()
        if any(word in title_lower for word in ['senior', 'lead', 'principal', 'architect', 'staff']):
            return 'senior'
        elif any(word in title_lower for word in ['junior', 'entry', 'associate', 'intern']):
            return 'junior'
        else:
            return 'mid'

    def _is_title_appropriate_for_level(self, title_level: str, target_level: str) -> bool:
        """Check if title level is appropriate for target experience level"""
        level_hierarchy = {
            'junior': 1,
            'mid': 2,
            'senior': 3,
            'lead': 4,
            'principal': 5
        }

        title_rank = level_hierarchy.get(title_level, 2)
        target_rank = level_hierarchy.get(target_level, 2)

        # Title should be at or above target level for upward transformation
        return title_rank >= target_rank

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
            'phrase_usage_history': len(self._phrase_usage_history),
            'max_phrase_reuse': self._max_phrase_reuse
        }

    def reset_diversity_tracking(self):
        """Reset diversity tracking for new augmentation batch"""
        # Reset traditional diversity tracking
        self._phrase_usage_history.clear()

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
