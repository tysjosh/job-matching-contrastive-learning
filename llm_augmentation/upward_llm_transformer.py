"""
Upward LLM Transformer for Career-Aware Data Augmentation System

This module transforms resume content to senior-level perspective using LLM prompts.
It adds leadership context, business impact language, and strategic thinking indicators
while preserving technical terms and maintaining semantic coherence.

Requirements: 2.1, 3.1, 4.1, 5.1, 5.2
"""

import json
import logging
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


class UpwardLLMTransformer:
    """
    Transforms resume content to senior-level perspective using LLM prompts.
    
    This transformer elevates resume descriptions to reflect how a more senior
    professional would describe the same work, adding ownership, impact, and scale.
    
    Attributes:
        llm_client: LLM client for generating transformations
        config: Transformation configuration
        term_protector: Technical term protector for preserving technical terms
    """

    # Default prompts for senior-level transformation
    DEFAULT_EXPERIENCE_PROMPT = """Transform the following work experience description to reflect a senior-level professional with leadership responsibilities, strategic thinking, and business impact.

Guidelines:
- Add language demonstrating ownership, mentorship, and cross-functional collaboration
- Include strategic thinking and business impact indicators
- Preserve ALL technical terms exactly as written (programming languages, frameworks, tools)
- Maintain the core technical content and achievements
- Use action verbs like: led, architected, designed, mentored, spearheaded, drove, scaled

Original experience:
{text}

Context (role: {role}, domain: {domain})
{esco_context}

Transformed senior-level experience (respond with ONLY the transformed text, no explanations):"""

    DEFAULT_RESPONSIBILITIES_PROMPT = """Transform these responsibilities to reflect senior-level ownership and leadership.

Guidelines:
- Add strategic oversight and mentorship aspects
- Include cross-functional collaboration language
- Preserve the core duties and technical content
- Use ownership language: "owned", "led", "drove", "architected"
- Maintain approximately the same number of responsibilities

Original responsibilities:
{responsibilities}
{esco_context}

Transformed senior-level responsibilities (respond with ONLY a numbered list, one per line):"""

    DEFAULT_ROLE_PROMPT = """Elevate this job title to a senior-level position while keeping it in the same job family/domain.

Guidelines:
- Add appropriate seniority prefix (Senior, Lead, Principal, Staff)
- Keep the core job function/domain intact
- Do not change the fundamental role type

Original title: {title}

Senior-level title (respond with ONLY the new title):"""

    DEFAULT_SKILLS_PROMPT = """Enhance this skills list for a senior-level professional.

Guidelines:
- Add 2-3 senior-level skills relevant to the domain (architecture, leadership, mentoring)
- Upgrade proficiency levels to reflect senior expertise
- Preserve all existing technical skills
- Add skills like: System Design, Technical Leadership, Architecture, Mentoring

Original skills:
{skills}

Domain context: {domain}
{esco_context}

Enhanced senior-level skills (respond with a JSON array of skill objects with 'name' and 'proficiency' keys):"""

    # Senior-level skills to potentially add
    SENIOR_SKILLS = [
        {"name": "System Architecture", "proficiency": "expert"},
        {"name": "Technical Leadership", "proficiency": "expert"},
        {"name": "Team Mentoring", "proficiency": "advanced"},
        {"name": "Cross-functional Collaboration", "proficiency": "expert"},
        {"name": "Strategic Planning", "proficiency": "advanced"},
        {"name": "Code Review & Best Practices", "proficiency": "expert"},
        {"name": "Stakeholder Management", "proficiency": "advanced"},
        {"name": "Technical Documentation", "proficiency": "expert"},
    ]

    # Proficiency upgrade mapping
    PROFICIENCY_UPGRADES = {
        "beginner": "intermediate",
        "basic": "intermediate",
        "intermediate": "advanced",
        "proficient": "advanced",
        "advanced": "expert",
        "expert": "expert",
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
        Initialize the upward LLM transformer.
        
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
            "experience": prompts.get("upward_experience", self.DEFAULT_EXPERIENCE_PROMPT) if prompts else self.DEFAULT_EXPERIENCE_PROMPT,
            "responsibilities": prompts.get("upward_responsibilities", self.DEFAULT_RESPONSIBILITIES_PROMPT) if prompts else self.DEFAULT_RESPONSIBILITIES_PROMPT,
            "role": prompts.get("upward_role", self.DEFAULT_ROLE_PROMPT) if prompts else self.DEFAULT_ROLE_PROMPT,
            "skills": prompts.get("upward_skills", self.DEFAULT_SKILLS_PROMPT) if prompts else self.DEFAULT_SKILLS_PROMPT,
        }
        
        logger.info(
            f"UpwardLLMTransformer initialized with target level: {config.upward_target_level}"
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
        Transform experience text to senior-level language.
        
        Uses LLM to add leadership context, business impact language, and
        strategic thinking indicators while preserving technical terms.
        
        Args:
            experience: Original experience text
            context: Optional context dict with 'role' and 'domain' keys
            
        Returns:
            TransformationResult with transformed text and metadata
            
        Requirements: 2.1
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
        Transform responsibilities to ownership/leadership language.
        
        Uses LLM to add strategic oversight and mentorship aspects while
        preserving the core duties and maintaining responsibility count
        within 20% of original.
        
        Args:
            responsibilities: List of original responsibility strings
            
        Returns:
            Tuple of (transformed_responsibilities, TransformationResult)
            
        Requirements: 3.1
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
            
            for i, resp in enumerate(responsibilities):
                protected, mapping = self.term_protector.protect_terms(resp)
                protected_responsibilities.append(protected)
                # Prefix mapping keys to avoid collisions
                for k, v in mapping.items():
                    all_term_mappings[f"{k}_{i}"] = v
                    protected = protected.replace(k, f"{k}_{i}")
                protected_responsibilities[i] = protected
            
            # Re-protect with unique keys
            protected_responsibilities = []
            all_term_mappings = {}
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
        # Simple ownership language additions
        ownership_prefixes = [
            "Led and ",
            "Owned ",
            "Drove ",
            "Spearheaded ",
        ]
        
        # Check if already has ownership language
        lower_resp = responsibility.lower()
        if any(prefix.lower() in lower_resp for prefix in ownership_prefixes):
            return responsibility
        
        # Add ownership prefix
        prefix = ownership_prefixes[hash(responsibility) % len(ownership_prefixes)]
        
        # Lowercase first letter if needed
        if responsibility and responsibility[0].isupper():
            transformed = prefix + responsibility[0].lower() + responsibility[1:]
        else:
            transformed = prefix + responsibility
        
        return transformed

    def transform_role(self, role: str) -> TransformationResult:
        """
        Elevate job title to senior level.
        
        Uses LLM to transform the title while keeping it in the same
        job family/domain.
        
        Args:
            role: Original job title
            
        Returns:
            TransformationResult with elevated title
            
        Requirements: 4.1
        """
        if not role or not role.strip():
            return TransformationResult(
                success=True,
                transformed_text=role,
                original_text=role,
                term_preservation_rate=1.0
            )
        
        try:
            # Step 1: Check if already senior-level
            if self._is_already_senior(role):
                logger.debug(f"Role '{role}' is already senior-level")
                return TransformationResult(
                    success=True,
                    transformed_text=role,
                    original_text=role,
                    term_preservation_rate=1.0
                )
            
            # Step 2: Try rule-based elevation first (faster, more predictable)
            rule_based_result = self._rule_based_role_elevation(role)
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
                transformed_role = f"Senior {role}"
            
            logger.debug(f"Role transformation: '{role}' -> '{transformed_role}'")
            
            return TransformationResult(
                success=True,
                transformed_text=transformed_role,
                original_text=role,
                term_preservation_rate=1.0
            )
            
        except LLMClientError as e:
            logger.error(f"LLM error during role transformation: {e}")
            # Fallback to simple prefix
            return TransformationResult(
                success=False,
                transformed_text=f"Senior {role}",
                original_text=role,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during role transformation: {e}")
            return TransformationResult(
                success=False,
                transformed_text=f"Senior {role}",
                original_text=role,
                error_message=str(e)
            )

    def _is_already_senior(self, role: str) -> bool:
        """Check if a role is already senior-level."""
        senior_indicators = [
            "senior", "lead", "principal", "staff", "architect",
            "director", "head", "chief", "vp", "vice president",
            "manager", "supervisor"
        ]
        role_lower = role.lower()
        return any(indicator in role_lower for indicator in senior_indicators)

    def _rule_based_role_elevation(self, role: str) -> Optional[str]:
        """
        Apply rule-based role elevation for common patterns.
        
        Returns elevated role or None if no rule matches.
        """
        role_lower = role.lower().strip()
        
        # Common role elevation patterns
        elevation_rules = {
            "developer": "Senior Developer",
            "software developer": "Senior Software Developer",
            "software engineer": "Senior Software Engineer",
            "engineer": "Senior Engineer",
            "programmer": "Senior Software Engineer",
            "analyst": "Senior Analyst",
            "data analyst": "Senior Data Analyst",
            "business analyst": "Senior Business Analyst",
            "designer": "Senior Designer",
            "ux designer": "Senior UX Designer",
            "ui designer": "Senior UI Designer",
            "product designer": "Senior Product Designer",
            "scientist": "Senior Scientist",
            "data scientist": "Senior Data Scientist",
            "consultant": "Senior Consultant",
            "administrator": "Senior Administrator",
            "system administrator": "Senior System Administrator",
            "devops engineer": "Senior DevOps Engineer",
            "qa engineer": "Senior QA Engineer",
            "test engineer": "Senior Test Engineer",
            "frontend developer": "Senior Frontend Developer",
            "backend developer": "Senior Backend Developer",
            "full stack developer": "Senior Full Stack Developer",
            "mobile developer": "Senior Mobile Developer",
            "web developer": "Senior Web Developer",
        }
        
        # Check for exact match
        if role_lower in elevation_rules:
            return elevation_rules[role_lower]
        
        # Check for partial matches
        for base_role, elevated_role in elevation_rules.items():
            if base_role in role_lower and "senior" not in role_lower:
                # Preserve original casing style
                if role[0].isupper():
                    return f"Senior {role}"
                return f"senior {role}"
        
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
        domain: str = "Technology",
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], TransformationResult]:
        """
        Add senior skills and upgrade proficiency levels.
        
        Enhances the skills list by:
        1. Adding 2-3 senior-level skills relevant to the domain
        2. Upgrading proficiency levels to reflect senior expertise
        
        Args:
            skills: List of skill dictionaries with 'name' and optionally 'proficiency'
            domain: Domain context for selecting relevant senior skills
            
        Returns:
            Tuple of (transformed_skills, TransformationResult)
            
        Requirements: 5.1, 5.2
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
                merged_skills = self._merge_original_skills(skills, transformed_skills)
                return merged_skills, TransformationResult(
                    success=True,
                    transformed_text=json.dumps(merged_skills),
                    original_text=original_text,
                    term_preservation_rate=1.0
                )

            return skills, TransformationResult(
                success=False,
                transformed_text=original_text,
                original_text=original_text,
                term_preservation_rate=1.0,
                error_message="Failed to parse LLM skills output"
            )

        except Exception as e:
            logger.error(f"Error during skills transformation: {e}")
            return skills, TransformationResult(
                success=False,
                transformed_text=original_text,
                original_text=original_text,
                term_preservation_rate=1.0,
                error_message=str(e)
            )

    def _merge_original_skills(
        self,
        original: List[Dict[str, Any]],
        generated: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure original skills are preserved in generated output."""
        original_names = {
            self._normalize_skill_name(s.get("name", "") if isinstance(s, dict) else str(s))
            for s in original
        }
        generated_names = {
            self._normalize_skill_name(s.get("name", "") if isinstance(s, dict) else str(s))
            for s in generated
        }
        merged = list(generated)
        for skill in original:
            name = self._normalize_skill_name(
                skill.get("name", "") if isinstance(skill, dict) else str(skill))
            if name and name not in generated_names:
                merged.append(self._upgrade_skill(skill))
        return merged

    # Rule-based fallback removed by request.

    def _upgrade_skill(self, skill: Any) -> Dict[str, Any]:
        """
        Upgrade a single skill's proficiency level.
        
        Args:
            skill: Skill dict or string
            
        Returns:
            Upgraded skill dictionary
        """
        if isinstance(skill, str):
            return {
                "name": skill,
                "proficiency": "advanced"
            }
        
        if not isinstance(skill, dict):
            return {"name": str(skill), "proficiency": "advanced"}
        
        upgraded = skill.copy()
        
        # Upgrade proficiency if present
        if "proficiency" in upgraded:
            current = upgraded["proficiency"].lower() if isinstance(upgraded["proficiency"], str) else "intermediate"
            upgraded["proficiency"] = self.PROFICIENCY_UPGRADES.get(current, "expert")
        elif "level" in upgraded:
            current = upgraded["level"].lower() if isinstance(upgraded["level"], str) else "intermediate"
            upgraded["level"] = self.PROFICIENCY_UPGRADES.get(current, "expert")
        else:
            # Add proficiency if not present
            upgraded["proficiency"] = "advanced"
        
        return upgraded

    def _normalize_skill_name(self, name: str) -> str:
        """Normalize skill name for comparison."""
        return name.lower().strip()

    def _select_senior_skills(
        self,
        existing_skills: set,
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Select senior skills to add based on domain and existing skills.
        
        Args:
            existing_skills: Set of normalized existing skill names
            domain: Domain context
            
        Returns:
            List of senior skills to add (2-3 skills)
        """
        # Domain-specific senior skills
        domain_skills = {
            "software": [
                {"name": "System Architecture", "proficiency": "expert"},
                {"name": "Technical Leadership", "proficiency": "expert"},
                {"name": "Code Review & Best Practices", "proficiency": "expert"},
            ],
            "data": [
                {"name": "Data Architecture", "proficiency": "expert"},
                {"name": "ML Pipeline Design", "proficiency": "advanced"},
                {"name": "Data Strategy", "proficiency": "advanced"},
            ],
            "devops": [
                {"name": "Infrastructure Architecture", "proficiency": "expert"},
                {"name": "Platform Engineering", "proficiency": "expert"},
                {"name": "SRE Practices", "proficiency": "advanced"},
            ],
            "product": [
                {"name": "Product Strategy", "proficiency": "expert"},
                {"name": "Stakeholder Management", "proficiency": "expert"},
                {"name": "Roadmap Planning", "proficiency": "advanced"},
            ],
        }
        
        # Determine domain category
        domain_lower = domain.lower()
        selected_domain = "software"  # default
        
        for key in domain_skills.keys():
            if key in domain_lower:
                selected_domain = key
                break
        
        # Get domain-specific skills
        candidate_skills = domain_skills.get(selected_domain, []) + self.SENIOR_SKILLS
        
        # Filter out existing skills
        skills_to_add = []
        for skill in candidate_skills:
            skill_name_normalized = self._normalize_skill_name(skill["name"])
            if skill_name_normalized not in existing_skills:
                skills_to_add.append(skill)
                if len(skills_to_add) >= 3:
                    break
        
        # Ensure we add at least 2 skills
        if len(skills_to_add) < 2:
            for skill in self.SENIOR_SKILLS:
                skill_name_normalized = self._normalize_skill_name(skill["name"])
                if skill_name_normalized not in existing_skills and skill not in skills_to_add:
                    skills_to_add.append(skill)
                    if len(skills_to_add) >= 2:
                        break
        
        return skills_to_add[:3]  # Return max 3 skills

    def transform_resume(
        self,
        resume: Dict[str, Any],
        job_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, TransformationResult]]:
        """
        Transform an entire resume to senior-level perspective.
        
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
                domain=context["domain"],
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
