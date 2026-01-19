"""
Downward LLM Transformer for Career-Aware Data Augmentation System

This module transforms resume content to junior-level perspective using LLM prompts.
It adds learning context, support language, and task-focused descriptions
while preserving technical terms and maintaining semantic coherence.

Requirements: 2.2, 3.2, 4.2, 5.3, 5.4
"""

import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient, LLMClientError
from .models import TransformationConfig
from .technical_term_protector import TechnicalTermProtector
from .esco_context import ESCOContextBuilder, ESCOContext

logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """Result of a transformation operation."""
    success: bool
    transformed_text: str
    original_text: str
    term_preservation_rate: float = 1.0
    error_message: Optional[str] = None


class DownwardLLMTransformer:
    """
    Transforms resume content to junior-level perspective using LLM prompts.
    
    This transformer reduces resume descriptions to reflect how a more junior
    professional would describe the same work, adding learning context,
    support language, and task-focused descriptions.
    
    Attributes:
        llm_client: LLM client for generating transformations
        config: Transformation configuration
        term_protector: Technical term protector for preserving technical terms
    """

    # Default prompts for junior-level transformation
    DEFAULT_EXPERIENCE_PROMPT = """Transform the following work experience description to reflect a junior-level professional who is learning and growing in their role.

Guidelines:
- Add language demonstrating learning, support, and collaboration with senior team members
- Include phrases showing guidance received and skills being developed
- Preserve ALL technical terms exactly as written (programming languages, frameworks, tools)
- Maintain the core technical content and achievements but frame them as learning experiences
- Use action verbs like: assisted, supported, contributed, learned, helped, participated, collaborated

Original experience:
{text}

Context (role: {role}, domain: {domain})
{esco_context}

Transformed junior-level experience (respond with ONLY the transformed text, no explanations):"""

    DEFAULT_RESPONSIBILITIES_PROMPT = """Transform these responsibilities to reflect junior-level support and learning.

Guidelines:
- Add learning and support aspects
- Include collaboration with senior team members
- Preserve the core duties and technical content
- Use support language: "assisted", "supported", "helped", "contributed to", "participated in"
- Maintain approximately the same number of responsibilities

Original responsibilities:
{responsibilities}
{esco_context}

Transformed junior-level responsibilities (respond with ONLY a numbered list, one per line):"""

    DEFAULT_ROLE_PROMPT = """Reduce this job title to a junior-level position while keeping it in the same job family/domain.

Guidelines:
- Remove seniority prefixes (Senior, Lead, Principal, Staff)
- Add appropriate junior prefix (Junior, Associate) if needed
- Keep the core job function/domain intact
- Do not change the fundamental role type

Original title: {title}

Junior-level title (respond with ONLY the new title):"""

    DEFAULT_SKILLS_PROMPT = """Adjust this skills list for a junior-level professional.

Guidelines:
- Remove or mask 20-40% of advanced/senior-level skills
- Downgrade proficiency levels to reflect developing expertise
- Preserve core technical skills but adjust proficiency
- Remove skills like: System Architecture, Technical Leadership, Team Management

Original skills:
{skills}

Domain context: {domain}
{esco_context}

Adjusted junior-level skills (respond with a JSON array of skill objects with 'name' and 'proficiency' keys):"""

    # Advanced skills that should be masked for junior-level
    ADVANCED_SKILLS_TO_MASK = [
        "system architecture",
        "technical leadership",
        "team management",
        "strategic planning",
        "stakeholder management",
        "enterprise architecture",
        "solution architecture",
        "technical mentoring",
        "team mentoring",
        "cross-functional collaboration",
        "roadmap planning",
        "budget management",
        "vendor management",
        "executive communication",
        "organizational design",
        "performance management",
        "capacity planning",
        "disaster recovery planning",
    ]

    # Proficiency downgrade mapping
    PROFICIENCY_DOWNGRADES = {
        "expert": "intermediate",
        "advanced": "intermediate",
        "proficient": "beginner",
        "intermediate": "beginner",
        "beginner": "beginner",
        "basic": "basic",
    }

    def __init__(
        self,
        llm_client: LLMClient,
        config: TransformationConfig,
        term_protector: Optional[TechnicalTermProtector] = None,
        prompts: Optional[Dict[str, str]] = None,
        esco_context_builder: Optional[ESCOContextBuilder] = None,
        cs_skills_path: Optional[str] = None,
        esco_domains_path: Optional[str] = None
    ):
        """
        Initialize the downward LLM transformer.
        
        Args:
            llm_client: LLM client for generating transformations
            config: Transformation configuration
            term_protector: Optional technical term protector (created if not provided)
            prompts: Optional custom prompts for transformations
        """
        self.llm_client = llm_client
        self.config = config
        self.term_protector = term_protector or TechnicalTermProtector(
            cs_skills_path=cs_skills_path or "dataset/cs_skills.json"
        )
        self.esco_context_builder = esco_context_builder or ESCOContextBuilder(
            esco_domains_path=esco_domains_path or "esco_it_career_domains_refined.json",
            cs_skills_path=cs_skills_path or "dataset/cs_skills.json"
        )
        
        # Set up prompts (use custom or defaults)
        self.prompts = {
            "experience": prompts.get("downward_experience", self.DEFAULT_EXPERIENCE_PROMPT) if prompts else self.DEFAULT_EXPERIENCE_PROMPT,
            "responsibilities": prompts.get("downward_responsibilities", self.DEFAULT_RESPONSIBILITIES_PROMPT) if prompts else self.DEFAULT_RESPONSIBILITIES_PROMPT,
            "role": prompts.get("downward_role", self.DEFAULT_ROLE_PROMPT) if prompts else self.DEFAULT_ROLE_PROMPT,
            "skills": prompts.get("downward_skills", self.DEFAULT_SKILLS_PROMPT) if prompts else self.DEFAULT_SKILLS_PROMPT,
        }
        
        logger.info(
            f"DownwardLLMTransformer initialized with target level: {config.downward_target_level}"
        )

    def _format_esco_context(self, context: Optional[ESCOContext]) -> str:
        """Format ESCO context for prompt injection."""
        if context is None:
            return ""
        block = context.to_prompt_block()
        return f"\n{block}" if block else ""

    def _extract_json_array(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract and parse a JSON array from text."""
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    def _validate_skill_objects(self, skills: List[Dict[str, Any]]) -> bool:
        """Validate skill objects structure."""
        if not skills:
            return False
        for skill in skills:
            if not isinstance(skill, dict):
                return False
            if "name" not in skill:
                return False
            if "proficiency" not in skill and "level" not in skill:
                return False
        return True

    def _generate_with_retry(
        self,
        prompt: str,
        parse_fn,
        max_attempts: int = 3
    ):
        """Generate with retries and parsing."""
        for _ in range(max_attempts):
            response = self.llm_client.generate(prompt)
            parsed = parse_fn(response.text.strip())
            if parsed is not None:
                return parsed
            prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON. Do not add commentary or code fences."
            )
        return None

    def transform_experience(
        self,
        experience: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TransformationResult:
        """
        Transform experience text to junior-level language.
        
        Uses LLM to add learning context, support language, and
        task-focused descriptions while preserving technical terms.
        
        Args:
            experience: Original experience text
            context: Optional context dict with 'role' and 'domain' keys
            
        Returns:
            TransformationResult with transformed text and metadata
            
        Requirements: 2.2
        """
        if not experience or not experience.strip():
            return TransformationResult(
                success=True,
                transformed_text=experience,
                original_text=experience,
                term_preservation_rate=1.0
            )
        
        context = context or {}
        role = context.get("role", "Professional")
        domain = context.get("domain", "Technology")
        esco_context = context.get("esco_context")
        
        try:
            # Step 1: Protect technical terms
            protected_text, term_mapping = self.term_protector.protect_terms(experience)
            
            # Step 2: Build prompt
            prompt = self.prompts["experience"].format(
                text=protected_text,
                role=role,
                domain=domain,
                esco_context=self._format_esco_context(esco_context)
            )
            
            # Step 3: Call LLM
            response = self.llm_client.generate(prompt)
            transformed_protected = response.text.strip()
            
            # Step 4: Restore technical terms
            transformed_text = self.term_protector.restore_terms(
                transformed_protected, term_mapping
            )
            
            # Step 5: Calculate preservation rate
            preservation_rate = self.term_protector.get_preservation_rate(
                experience, transformed_text
            )
            
            logger.debug(
                f"Experience transformation complete. "
                f"Preservation rate: {preservation_rate:.2%}"
            )
            
            return TransformationResult(
                success=True,
                transformed_text=transformed_text,
                original_text=experience,
                term_preservation_rate=preservation_rate
            )
            
        except LLMClientError as e:
            logger.error(f"LLM error during experience transformation: {e}")
            return TransformationResult(
                success=False,
                transformed_text=experience,
                original_text=experience,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during experience transformation: {e}")
            return TransformationResult(
                success=False,
                transformed_text=experience,
                original_text=experience,
                error_message=str(e)
            )

    def transform_responsibilities(
        self,
        responsibilities: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], TransformationResult]:
        """
        Transform responsibilities to support/learning language.
        
        Uses LLM to add learning and support aspects while
        preserving the core duties and maintaining responsibility count
        within 20% of original.
        
        Args:
            responsibilities: List of original responsibility strings
            
        Returns:
            Tuple of (transformed_responsibilities, TransformationResult)
            
        Requirements: 3.2
        """
        if not responsibilities:
            return [], TransformationResult(
                success=True,
                transformed_text="",
                original_text="",
                term_preservation_rate=1.0
            )
        
        original_count = len(responsibilities)
        original_text = "\n".join(f"- {r}" for r in responsibilities)
        context = context or {}
        esco_context = context.get("esco_context")
        
        try:
            # Step 1: Protect technical terms in all responsibilities
            protected_responsibilities = []
            all_term_mappings: Dict[str, str] = {}
            placeholder_counter = 0
            
            for resp in responsibilities:
                protected, mapping = self.term_protector.protect_terms(resp)
                # Remap to unique placeholders
                for old_placeholder, term in mapping.items():
                    new_placeholder = f"__TECH_TERM_{placeholder_counter}__"
                    protected = protected.replace(old_placeholder, new_placeholder)
                    all_term_mappings[new_placeholder] = term
                    placeholder_counter += 1
                protected_responsibilities.append(protected)
            
            protected_text = "\n".join(f"- {r}" for r in protected_responsibilities)
            
            # Step 2: Build prompt
            prompt = self.prompts["responsibilities"].format(
                responsibilities=protected_text,
                esco_context=self._format_esco_context(esco_context)
            )

            def parse_fn(text: str) -> Optional[List[str]]:
                parsed = self._parse_responsibilities_response(text)
                return parsed if parsed else None

            transformed_list = self._generate_with_retry(prompt, parse_fn)
            if transformed_list is None:
                return responsibilities, TransformationResult(
                    success=False,
                    transformed_text=original_text,
                    original_text=original_text,
                    error_message="Failed to parse responsibilities output"
                )
            
            # Step 5: Restore technical terms
            restored_list = []
            for resp in transformed_list:
                restored = self.term_protector.restore_terms(resp, all_term_mappings)
                restored_list.append(restored)
            
            # Step 6: Validate count constraint (within 20% of original)
            min_count = int(original_count * 0.8)
            max_count = int(original_count * 1.2) + 1  # +1 for rounding
            
            if len(restored_list) < min_count:
                logger.warning(
                    f"Transformed responsibilities count ({len(restored_list)}) "
                    f"below minimum ({min_count}). Padding with originals."
                )
                # Pad with transformed versions of remaining originals
                while len(restored_list) < min_count and len(restored_list) < original_count:
                    idx = len(restored_list)
                    if idx < len(responsibilities):
                        restored_list.append(self._quick_transform_responsibility(responsibilities[idx]))
            
            if len(restored_list) > max_count:
                logger.warning(
                    f"Transformed responsibilities count ({len(restored_list)}) "
                    f"above maximum ({max_count}). Truncating."
                )
                restored_list = restored_list[:max_count]
            
            # Step 7: Calculate preservation rate
            combined_original = " ".join(responsibilities)
            combined_transformed = " ".join(restored_list)
            preservation_rate = self.term_protector.get_preservation_rate(
                combined_original, combined_transformed
            )
            
            logger.debug(
                f"Responsibilities transformation complete. "
                f"Original: {original_count}, Transformed: {len(restored_list)}, "
                f"Preservation rate: {preservation_rate:.2%}"
            )
            
            return restored_list, TransformationResult(
                success=True,
                transformed_text="\n".join(restored_list),
                original_text=original_text,
                term_preservation_rate=preservation_rate
            )
            
        except LLMClientError as e:
            logger.error(f"LLM error during responsibilities transformation: {e}")
            return responsibilities, TransformationResult(
                success=False,
                transformed_text=original_text,
                original_text=original_text,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during responsibilities transformation: {e}")
            return responsibilities, TransformationResult(
                success=False,
                transformed_text=original_text,
                original_text=original_text,
                error_message=str(e)
            )

    def _parse_responsibilities_response(self, response_text: str) -> List[str]:
        """
        Parse LLM response into a list of responsibilities.
        
        Handles various formats: numbered lists, bullet points, plain lines.
        """
        lines = response_text.strip().split("\n")
        responsibilities = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove common list prefixes
            # Numbered: "1.", "1)", "1:"
            line = re.sub(r"^\d+[\.\)\:]\s*", "", line)
            # Bullets: "-", "*", "•"
            line = re.sub(r"^[\-\*\•]\s*", "", line)
            
            if line:
                responsibilities.append(line)
        
        return responsibilities

    def _quick_transform_responsibility(self, responsibility: str) -> str:
        """
        Quick rule-based transformation for a single responsibility.
        
        Used as fallback when LLM output needs padding.
        """
        # Simple support language additions
        support_prefixes = [
            "Assisted with ",
            "Supported ",
            "Helped with ",
            "Contributed to ",
        ]
        
        # Check if already has support language
        lower_resp = responsibility.lower()
        if any(prefix.lower() in lower_resp for prefix in support_prefixes):
            return responsibility
        
        # Add support prefix
        prefix = support_prefixes[hash(responsibility) % len(support_prefixes)]
        
        # Lowercase first letter if needed
        if responsibility and responsibility[0].isupper():
            transformed = prefix + responsibility[0].lower() + responsibility[1:]
        else:
            transformed = prefix + responsibility
        
        return transformed

    def transform_role(self, role: str) -> TransformationResult:
        """
        Reduce job title to junior level.
        
        Uses LLM to transform the title while keeping it in the same
        job family/domain.
        
        Args:
            role: Original job title
            
        Returns:
            TransformationResult with reduced title
            
        Requirements: 4.2
        """
        if not role or not role.strip():
            return TransformationResult(
                success=True,
                transformed_text=role,
                original_text=role,
                term_preservation_rate=1.0
            )
        
        try:
            # Step 1: Check if already junior-level
            if self._is_already_junior(role):
                logger.debug(f"Role '{role}' is already junior-level")
                return TransformationResult(
                    success=True,
                    transformed_text=role,
                    original_text=role,
                    term_preservation_rate=1.0
                )
            
            # Step 2: Try rule-based reduction first (faster, more predictable)
            rule_based_result = self._rule_based_role_reduction(role)
            if rule_based_result:
                return TransformationResult(
                    success=True,
                    transformed_text=rule_based_result,
                    original_text=role,
                    term_preservation_rate=1.0
                )
            
            # Step 3: Fall back to LLM for complex titles
            prompt = self.prompts["role"].format(title=role)
            response = self.llm_client.generate(prompt, max_tokens=50)
            transformed_role = response.text.strip()
            
            # Clean up response (remove quotes, extra text)
            transformed_role = transformed_role.strip('"\'')
            # Take only first line if multiple
            transformed_role = transformed_role.split("\n")[0].strip()
            
            # Validate the transformation preserved the domain
            if not self._validate_role_domain(role, transformed_role):
                logger.warning(
                    f"LLM role transformation changed domain. "
                    f"Original: '{role}', Transformed: '{transformed_role}'. "
                    f"Using rule-based fallback."
                )
                transformed_role = self._remove_seniority_prefix(role)
            
            logger.debug(f"Role transformation: '{role}' -> '{transformed_role}'")
            
            return TransformationResult(
                success=True,
                transformed_text=transformed_role,
                original_text=role,
                term_preservation_rate=1.0
            )
            
        except LLMClientError as e:
            logger.error(f"LLM error during role transformation: {e}")
            # Fallback to simple prefix removal
            return TransformationResult(
                success=False,
                transformed_text=self._remove_seniority_prefix(role),
                original_text=role,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during role transformation: {e}")
            return TransformationResult(
                success=False,
                transformed_text=self._remove_seniority_prefix(role),
                original_text=role,
                error_message=str(e)
            )

    def _is_already_junior(self, role: str) -> bool:
        """Check if a role is already junior-level."""
        junior_indicators = [
            "junior", "associate", "entry", "intern", "trainee",
            "graduate", "apprentice"
        ]
        role_lower = role.lower()
        return any(indicator in role_lower for indicator in junior_indicators)

    def _remove_seniority_prefix(self, role: str) -> str:
        """Remove seniority prefixes from a role title."""
        seniority_prefixes = [
            "senior ", "lead ", "principal ", "staff ", "chief ",
            "head ", "director ", "vp ", "vice president "
        ]
        
        result = role
        for prefix in seniority_prefixes:
            if result.lower().startswith(prefix):
                result = result[len(prefix):]
                break
        
        return result.strip()

    def _rule_based_role_reduction(self, role: str) -> Optional[str]:
        """
        Apply rule-based role reduction for common patterns.
        
        Returns reduced role or None if no rule matches.
        """
        role_lower = role.lower().strip()
        
        # Common role reduction patterns
        reduction_rules = {
            "senior developer": "Junior Developer",
            "senior software developer": "Junior Software Developer",
            "senior software engineer": "Junior Software Engineer",
            "senior engineer": "Junior Engineer",
            "lead developer": "Developer",
            "lead software engineer": "Software Engineer",
            "lead engineer": "Engineer",
            "principal engineer": "Software Engineer",
            "principal software engineer": "Software Engineer",
            "staff engineer": "Software Engineer",
            "staff software engineer": "Software Engineer",
            "senior analyst": "Junior Analyst",
            "senior data analyst": "Junior Data Analyst",
            "senior business analyst": "Junior Business Analyst",
            "lead analyst": "Analyst",
            "senior designer": "Junior Designer",
            "senior ux designer": "Junior UX Designer",
            "senior ui designer": "Junior UI Designer",
            "lead designer": "Designer",
            "senior product designer": "Junior Product Designer",
            "senior scientist": "Junior Scientist",
            "senior data scientist": "Junior Data Scientist",
            "lead data scientist": "Data Scientist",
            "senior consultant": "Junior Consultant",
            "senior administrator": "Junior Administrator",
            "senior system administrator": "Junior System Administrator",
            "senior devops engineer": "Junior DevOps Engineer",
            "lead devops engineer": "DevOps Engineer",
            "senior qa engineer": "Junior QA Engineer",
            "senior test engineer": "Junior Test Engineer",
            "senior frontend developer": "Junior Frontend Developer",
            "senior backend developer": "Junior Backend Developer",
            "senior full stack developer": "Junior Full Stack Developer",
            "senior mobile developer": "Junior Mobile Developer",
            "senior web developer": "Junior Web Developer",
            "tech lead": "Developer",
            "technical lead": "Developer",
            "engineering manager": "Software Engineer",
            "development manager": "Developer",
        }
        
        # Check for exact match
        if role_lower in reduction_rules:
            return reduction_rules[role_lower]
        
        # Check for partial matches with senior prefix
        if role_lower.startswith("senior "):
            base_role = role[7:]  # Remove "Senior "
            return f"Junior {base_role}"
        
        if role_lower.startswith("lead "):
            base_role = role[5:]  # Remove "Lead "
            return base_role
        
        if role_lower.startswith("principal "):
            base_role = role[10:]  # Remove "Principal "
            return base_role
        
        if role_lower.startswith("staff "):
            base_role = role[6:]  # Remove "Staff "
            return base_role
        
        return None

    def _validate_role_domain(self, original: str, transformed: str) -> bool:
        """
        Validate that the transformed role preserves the original domain.
        
        Checks that key domain words are preserved.
        """
        # Extract domain keywords from original
        domain_keywords = set()
        keywords_to_check = [
            "developer", "engineer", "analyst", "designer", "scientist",
            "consultant", "administrator", "architect", "manager",
            "software", "data", "product", "ux", "ui", "qa", "test",
            "frontend", "backend", "full stack", "mobile", "web",
            "devops", "cloud", "security", "network", "database",
            "machine learning", "ml", "ai", "artificial intelligence"
        ]
        
        original_lower = original.lower()
        transformed_lower = transformed.lower()
        
        for keyword in keywords_to_check:
            if keyword in original_lower:
                domain_keywords.add(keyword)
        
        # Check if at least one domain keyword is preserved
        if not domain_keywords:
            return True  # No specific domain to preserve
        
        for keyword in domain_keywords:
            if keyword in transformed_lower:
                return True
        
        return False

    def transform_skills(
        self,
        skills: List[Dict[str, Any]],
        mask_ratio: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], TransformationResult]:
        """
        Mask advanced skills and downgrade proficiency levels.
        
        Reduces the skills list by:
        1. Masking/removing 20-40% of advanced skills
        2. Downgrading proficiency levels to reflect developing expertise
        
        Args:
            skills: List of skill dictionaries with 'name' and optionally 'proficiency'
            mask_ratio: Optional override for mask ratio (default uses config range)
            
        Returns:
            Tuple of (transformed_skills, TransformationResult)
            
        Requirements: 5.3, 5.4
        """
        if not skills:
            return [], TransformationResult(
                success=True,
                transformed_text="",
                original_text="",
                term_preservation_rate=1.0
            )
        
        original_text = json.dumps(skills)
        context = context or {}
        domain = context.get("domain", "Technology")
        esco_context = context.get("esco_context")

        try:
            prompt = self.prompts["skills"].format(
                skills=original_text,
                domain=domain,
                esco_context=self._format_esco_context(esco_context)
            )
            transformed_skills = self._generate_with_retry(
                prompt, self._extract_json_array)

            if transformed_skills and self._validate_skill_objects(transformed_skills):
                return transformed_skills, TransformationResult(
                    success=True,
                    transformed_text=json.dumps(transformed_skills),
                    original_text=original_text,
                    term_preservation_rate=1.0
                )

            logger.warning("Falling back to rule-based skill masking.")
            return self._rule_based_transform_skills(skills, mask_ratio, original_text)

        except Exception as e:
            logger.error(f"Error during skills transformation: {e}")
            return self._rule_based_transform_skills(skills, mask_ratio, original_text, error=str(e))

    def _rule_based_transform_skills(
        self,
        skills: List[Dict[str, Any]],
        mask_ratio: Optional[float],
        original_text: str,
        error: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], TransformationResult]:
        """Fallback rule-based skill masking."""
        if mask_ratio is None:
            min_ratio = self.config.skills_mask_ratio_min
            max_ratio = self.config.skills_mask_ratio_max
            mask_ratio = random.uniform(min_ratio, max_ratio)

        advanced_skills_indices = self._identify_advanced_skills(skills)
        num_to_mask = max(1, int(len(advanced_skills_indices) * mask_ratio))
        skills_to_mask = set()
        if advanced_skills_indices:
            mask_indices = random.sample(
                advanced_skills_indices,
                min(num_to_mask, len(advanced_skills_indices))
            )
            skills_to_mask = set(mask_indices)

        transformed_skills = []
        for i, skill in enumerate(skills):
            if i in skills_to_mask:
                continue
            downgraded_skill = self._downgrade_skill(skill)
            transformed_skills.append(downgraded_skill)

        return transformed_skills, TransformationResult(
            success=False,
            transformed_text=json.dumps(transformed_skills),
            original_text=original_text,
            term_preservation_rate=1.0,
            error_message=error
        )

    def _identify_advanced_skills(self, skills: List[Dict[str, Any]]) -> List[int]:
        """
        Identify indices of advanced/senior-level skills.
        
        Args:
            skills: List of skill dictionaries
            
        Returns:
            List of indices for advanced skills
        """
        advanced_indices = []
        
        for i, skill in enumerate(skills):
            skill_name = ""
            if isinstance(skill, dict):
                skill_name = skill.get("name", "")
            elif isinstance(skill, str):
                skill_name = skill
            
            skill_name_lower = skill_name.lower()
            
            # Check if skill is in the advanced skills list
            is_advanced = any(
                adv_skill in skill_name_lower
                for adv_skill in self.ADVANCED_SKILLS_TO_MASK
            )
            
            # Also check proficiency level
            if isinstance(skill, dict):
                proficiency = skill.get("proficiency", "").lower()
                if proficiency in ["expert", "advanced"]:
                    is_advanced = True
            
            if is_advanced:
                advanced_indices.append(i)
        
        return advanced_indices

    def _downgrade_skill(self, skill: Any) -> Dict[str, Any]:
        """
        Downgrade a single skill's proficiency level.
        
        Args:
            skill: Skill dict or string
            
        Returns:
            Downgraded skill dictionary
        """
        if isinstance(skill, str):
            return {
                "name": skill,
                "proficiency": "beginner"
            }
        
        if not isinstance(skill, dict):
            return {"name": str(skill), "proficiency": "beginner"}
        
        downgraded = skill.copy()
        
        # Downgrade proficiency if present
        if "proficiency" in downgraded:
            current = downgraded["proficiency"].lower() if isinstance(downgraded["proficiency"], str) else "intermediate"
            downgraded["proficiency"] = self.PROFICIENCY_DOWNGRADES.get(current, "beginner")
        elif "level" in downgraded:
            current = downgraded["level"].lower() if isinstance(downgraded["level"], str) else "intermediate"
            downgraded["level"] = self.PROFICIENCY_DOWNGRADES.get(current, "beginner")
        else:
            # Add proficiency if not present
            downgraded["proficiency"] = "beginner"
        
        return downgraded

    def transform_resume(
        self,
        resume: Dict[str, Any],
        job_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, TransformationResult]]:
        """
        Transform an entire resume to junior-level perspective.
        
        Convenience method that applies all transformations to a resume.
        
        Args:
            resume: Resume dictionary with experience, responsibilities, role, skills
            job_context: Optional job context for domain-aware transformation
            
        Returns:
            Tuple of (transformed_resume, results_dict)
        """
        job_context = job_context or {}
        results: Dict[str, TransformationResult] = {}
        
        # Create a copy of the resume
        transformed = resume.copy()
        
        esco_context = self.esco_context_builder.build_context(
            role=resume.get("role", ""),
            job_title=job_context.get("title", ""),
            skills=resume.get("skills", [])
        )
        # Determine context
        context = {
            "role": resume.get("role", job_context.get("title", "Professional")),
            "domain": self._detect_domain(resume, job_context),
            "esco_context": esco_context
        }
        
        # Transform experience
        if "experience" in resume and resume["experience"]:
            result = self.transform_experience(resume["experience"], context)
            transformed["experience"] = result.transformed_text
            results["experience"] = result
            
            # Also update original_text if present
            if "original_text" in resume:
                transformed["original_text"] = result.transformed_text
        
        # Transform responsibilities
        if "responsibilities" in resume and resume["responsibilities"]:
            transformed_resp, result = self.transform_responsibilities(
                resume["responsibilities"],
                context=context
            )
            transformed["responsibilities"] = transformed_resp
            results["responsibilities"] = result
        
        # Transform role
        if "role" in resume and resume["role"]:
            result = self.transform_role(resume["role"])
            transformed["role"] = result.transformed_text
            results["role"] = result
        
        # Transform skills
        if "skills" in resume and resume["skills"]:
            transformed_skills, result = self.transform_skills(
                resume["skills"],
                context=context
            )
            transformed["skills"] = transformed_skills
            results["skills"] = result
        
        return transformed, results

    def _detect_domain(
        self,
        resume: Dict[str, Any],
        job_context: Dict[str, Any]
    ) -> str:
        """
        Detect the domain from resume and job context.
        
        Args:
            resume: Resume dictionary
            job_context: Job context dictionary
            
        Returns:
            Detected domain string
        """
        # Check job title first
        job_title = job_context.get("title", "").lower()
        
        domain_keywords = {
            "data": ["data", "analytics", "ml", "machine learning", "ai"],
            "devops": ["devops", "sre", "infrastructure", "platform", "cloud"],
            "product": ["product", "ux", "design"],
            "software": ["software", "developer", "engineer", "backend", "frontend"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in job_title for kw in keywords):
                return domain
        
        # Check resume role
        role = resume.get("role", "").lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in role for kw in keywords):
                return domain
        
        # Check skills
        skills = resume.get("skills", [])
        skill_text = " ".join(
            s.get("name", "") if isinstance(s, dict) else str(s)
            for s in skills
        ).lower()
        
        for domain, keywords in domain_keywords.items():
            if any(kw in skill_text for kw in keywords):
                return domain
        
        return "Technology"  # Default domain
