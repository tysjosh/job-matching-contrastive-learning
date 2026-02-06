"""
LLM-Based Negative Sample Generator for Career-Aware Data Augmentation

This module generates hard negative samples by creating semantically plausible
but incorrect resume-job pairings using career level mismatches.

Negative Types Generated (matching rule-based augmentation):
1. Cross-Match Negative (Foundational→Aspirational): Junior resume → Senior job
2. Cross-Match Negative (Aspirational→Foundational): Senior resume → Junior job
3. Cross-Match Negative (Original→Foundational): Original resume → Junior job
4. Cross-Match Negative (Original→Aspirational): Original resume → Senior job

This module integrates with the existing UpwardLLMTransformer and DownwardLLMTransformer
for comprehensive transformations including:
- Experience text transformation with leadership/learning language
- Responsibilities transformation with ownership/support verbs
- Role/title transformation with rule-based fallbacks
- Skills transformation with proficiency adjustments
- Technical term protection
- ESCO context integration
- Semantic validation to ensure transformation quality
"""

import json
import logging
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient, LLMClientError
from .models import TransformationConfig
from .esco_context import ESCOContextBuilder, ESCOContext
from .technical_term_protector import TechnicalTermProtector
from .upward_llm_transformer import UpwardLLMTransformer
from .downward_llm_transformer import DownwardLLMTransformer
from .semantic_validator import SemanticCoherenceValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class NegativeType:
    """Defines a type of negative sample."""
    name: str
    description: str
    weight: float = 1.0


@dataclass 
class GeneratedNegative:
    """A generated negative sample with metadata."""
    resume: Dict[str, Any]
    job: Dict[str, Any]
    negative_type: str
    transformation_details: Dict[str, Any]
    label: int = 0
    validation_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NegativeGenerationResult:
    """Result of negative generation for a single record."""
    original_record: Dict[str, Any]
    negatives: List[GeneratedNegative]
    success: bool
    error_message: Optional[str] = None


EXPERIENCE_LEVELS = ["intern", "junior", "mid", "senior", "lead", "principal", "executive"]
EXPERIENCE_LEVEL_MAP = {level: idx for idx, level in enumerate(EXPERIENCE_LEVELS)}


class LLMNegativeGenerator:
    """
    Generates hard negative samples using LLM and ESCO ontology guidance.
    
    Creates 4 types of cross-match negatives per original record (matching rule-based):
    1. Cross-Match Negative (Foundational→Aspirational): Junior resume → Senior job
    2. Cross-Match Negative (Aspirational→Foundational): Senior resume → Junior job
    3. Cross-Match Negative (Original→Foundational): Original resume → Junior job
    4. Cross-Match Negative (Original→Aspirational): Original resume → Senior job
    
    This class integrates with UpwardLLMTransformer and DownwardLLMTransformer for
    comprehensive transformations including experience, responsibilities, role, and skills.
    """
    
    NEGATIVE_TYPES = [
        NegativeType("Cross-Match Negative (Foundational→Aspirational)", "Junior resume → Senior job", weight=1.0),
        NegativeType("Cross-Match Negative (Aspirational→Foundational)", "Senior resume → Junior job", weight=1.0),
        NegativeType("Cross-Match Negative (Original→Foundational)", "Original resume → Junior job", weight=1.0),
        NegativeType("Cross-Match Negative (Original→Aspirational)", "Original resume → Senior job", weight=1.0),
    ]
    
    # Job transformation prompts (resume transformations use the dedicated transformers)
    TRANSFORM_JOB_TO_JUNIOR_PROMPT = """Transform this job posting to require junior/entry-level experience.

Original Job:
Title: {title}
Description: {description}
Required Skills: {skills}
{esco_context}

Guidelines:
- Change title to include "Junior", "Associate", or "Entry-Level" prefix
- Reduce years of experience requirements (0-2 years)
- Simplify responsibilities to learning/support tasks
- Keep the same domain/field and ESCO occupation family
- Preserve all technical skill names exactly

Return a JSON object with keys: title, description, required_experience_level
Respond with ONLY the JSON, no explanations:"""

    TRANSFORM_JOB_TO_SENIOR_PROMPT = """Transform this job posting to require senior/lead-level experience.

Original Job:
Title: {title}
Description: {description}
Required Skills: {skills}
{esco_context}

Guidelines:
- Change title to include "Senior", "Lead", or "Principal" prefix
- Increase years of experience requirements (8+ years)
- Add leadership, architecture, and mentoring responsibilities
- Keep the same domain/field and ESCO occupation family
- Preserve all technical skill names exactly

Return a JSON object with keys: title, description, required_experience_level
Respond with ONLY the JSON, no explanations:"""

    def __init__(
        self,
        llm_client: LLMClient,
        config: TransformationConfig,
        esco_context_builder: Optional[ESCOContextBuilder] = None,
        term_protector: Optional[TechnicalTermProtector] = None,
        semantic_validator: Optional[SemanticCoherenceValidator] = None,
        cs_skills_path: str = "dataset/cs_skills.json",
        esco_domains_path: str = "esco_it_career_domains_refined.json",
        num_negatives: int = 4,
        enable_semantic_validation: bool = True,
        min_similarity: float = 0.5,
        max_similarity: float = 0.95
    ):
        self.llm_client = llm_client
        self.config = config
        self.num_negatives = num_negatives
        self.enable_semantic_validation = enable_semantic_validation
        
        self.esco_context_builder = esco_context_builder or ESCOContextBuilder(
            esco_domains_path=esco_domains_path,
            cs_skills_path=cs_skills_path
        )
        
        self.term_protector = term_protector or TechnicalTermProtector(
            cs_skills_path=cs_skills_path,
            llm_client=llm_client
        )
        
        # Initialize semantic validator for quality control
        self.semantic_validator = semantic_validator or SemanticCoherenceValidator(
            min_similarity=min_similarity,
            max_similarity=max_similarity
        )
        
        # Initialize the comprehensive transformers for resume transformations
        self.upward_transformer = UpwardLLMTransformer(
            llm_client=llm_client,
            config=config,
            term_protector=self.term_protector,
            esco_context_builder=self.esco_context_builder,
            cs_skills_path=cs_skills_path,
            esco_domains_path=esco_domains_path
        )
        
        self.downward_transformer = DownwardLLMTransformer(
            llm_client=llm_client,
            config=config,
            term_protector=self.term_protector,
            esco_context_builder=self.esco_context_builder,
            cs_skills_path=cs_skills_path,
            esco_domains_path=esco_domains_path
        )
        
        logger.info(f"LLMNegativeGenerator initialized with {num_negatives} negatives per record")
        logger.info("Using UpwardLLMTransformer and DownwardLLMTransformer for comprehensive transformations")
        logger.info(f"Semantic validation: {'enabled' if enable_semantic_validation else 'disabled'}")

    def _format_esco_context(self, context: Optional[ESCOContext]) -> str:
        """Format ESCO context for prompt injection."""
        if context is None:
            return ""
        block = context.to_prompt_block()
        return f"\n{block}" if block else ""

    def _build_esco_context(self, resume: Dict[str, Any], job: Dict[str, Any]) -> ESCOContext:
        """Build ESCO context from resume and job data."""
        role = resume.get("role", "")
        job_title = job.get("title", "")
        skills = resume.get("skills", [])
        
        return self.esco_context_builder.build_context(
            role=role,
            job_title=job_title,
            skills=skills,
            max_adjacent_roles=6,
            max_skills_per_cluster=6
        )

    def _extract_text_from_experience(self, experience: Any) -> str:
        """Extract text from various experience formats."""
        if isinstance(experience, str):
            return experience
        if isinstance(experience, list) and experience:
            first = experience[0]
            if isinstance(first, dict):
                desc = first.get("description", "")
                if isinstance(desc, list) and desc:
                    if isinstance(desc[0], dict):
                        return desc[0].get("description", "")
                    return str(desc[0])
                return str(desc)
            return str(first)
        return str(experience) if experience else ""

    def _extract_skills_list(self, skills: Any) -> List[str]:
        """Extract skills as a list of strings."""
        if not skills:
            return []
        if isinstance(skills, str):
            return [skills]
        if isinstance(skills, list):
            result = []
            for s in skills:
                if isinstance(s, dict):
                    result.append(s.get("name", str(s)))
                else:
                    result.append(str(s))
            return result
        return []

    def _call_llm_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call LLM and parse JSON response."""
        try:
            import re
            response = self.llm_client.generate(prompt)
            text = response.text.strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return None
        except (LLMClientError, json.JSONDecodeError) as e:
            logger.warning(f"LLM JSON call failed: {e}")
            return None

    def _validate_transformation(
        self,
        original_text: str,
        transformed_text: str,
        target_level: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a transformation using semantic coherence validator.
        
        Args:
            original_text: Original text before transformation
            transformed_text: Text after transformation
            target_level: Target career level ("senior" or "junior")
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        if not self.enable_semantic_validation:
            return True, {"validation_skipped": True}
        
        if not original_text or not transformed_text:
            return True, {"validation_skipped": True, "reason": "empty_text"}
        
        try:
            result = self.semantic_validator.validate_transformation(
                original=original_text,
                transformed=transformed_text,
                target_level=target_level
            )
            
            validation_details = {
                "is_valid": result.is_valid,
                "semantic_similarity": result.semantic_similarity,
                "career_level_valid": result.career_level_valid,
                "rejection_reason": result.rejection_reason
            }
            
            if not result.is_valid:
                logger.warning(
                    f"Transformation validation failed: {result.rejection_reason} "
                    f"(similarity: {result.semantic_similarity:.3f})"
                )
            else:
                logger.debug(
                    f"Transformation validated: similarity={result.semantic_similarity:.3f}, "
                    f"career_level_valid={result.career_level_valid}"
                )
            
            return result.is_valid, validation_details
            
        except Exception as e:
            logger.warning(f"Semantic validation error: {e}")
            return True, {"validation_error": str(e)}

    def _transform_resume_to_junior(self, resume: Dict[str, Any], job: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transform resume to junior/foundational level using DownwardLLMTransformer.
        
        Returns:
            Tuple of (transformed_resume, transformation_results)
        """
        original_experience = self._extract_text_from_experience(resume.get("experience", ""))
        
        try:
            # Use the comprehensive downward transformer
            transformed_resume, results = self.downward_transformer.transform_resume(
                resume=copy.deepcopy(resume),
                job_context=job
            )
            
            # Ensure experience_level is set
            transformed_resume["experience_level"] = "junior"
            
            # Validate the transformation
            transformed_experience = self._extract_text_from_experience(transformed_resume.get("experience", ""))
            is_valid, validation_details = self._validate_transformation(
                original_text=original_experience,
                transformed_text=transformed_experience,
                target_level="junior"
            )
            
            # Build transformation details from results
            transformation_details = {
                "method": "DownwardLLMTransformer",
                "experience_success": results.get("experience", {}).success if "experience" in results else None,
                "responsibilities_success": results.get("responsibilities", {}).success if "responsibilities" in results else None,
                "role_success": results.get("role", {}).success if "role" in results else None,
                "skills_success": results.get("skills", {}).success if "skills" in results else None,
                "semantic_validation": validation_details,
            }
            
            # If validation failed, use fallback
            if not is_valid and validation_details.get("rejection_reason"):
                logger.warning(f"Junior transformation failed validation, using fallback")
                transformed = copy.deepcopy(resume)
                role = transformed.get("role", "")
                if not role.lower().startswith("junior"):
                    transformed["role"] = f"Junior {role}"
                transformed["experience_level"] = "junior"
                transformation_details["method"] = "fallback_after_validation_failure"
                return transformed, transformation_details
            
            logger.debug(f"Resume transformed to junior level: {resume.get('role', '')} -> {transformed_resume.get('role', '')}")
            return transformed_resume, transformation_details
            
        except Exception as e:
            logger.error(f"Error in downward transformation, using fallback: {e}")
            # Fallback to simple transformation
            transformed = copy.deepcopy(resume)
            role = transformed.get("role", "")
            if not role.lower().startswith("junior"):
                transformed["role"] = f"Junior {role}"
            transformed["experience_level"] = "junior"
            return transformed, {"method": "fallback", "error": str(e)}

    def _transform_resume_to_senior(self, resume: Dict[str, Any], job: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transform resume to senior/aspirational level using UpwardLLMTransformer.
        
        Returns:
            Tuple of (transformed_resume, transformation_results)
        """
        original_experience = self._extract_text_from_experience(resume.get("experience", ""))
        
        try:
            # Use the comprehensive upward transformer
            transformed_resume, results = self.upward_transformer.transform_resume(
                resume=copy.deepcopy(resume),
                job_context=job
            )
            
            # Ensure experience_level is set
            transformed_resume["experience_level"] = "senior"
            
            # Validate the transformation
            transformed_experience = self._extract_text_from_experience(transformed_resume.get("experience", ""))
            is_valid, validation_details = self._validate_transformation(
                original_text=original_experience,
                transformed_text=transformed_experience,
                target_level="senior"
            )
            
            # Build transformation details from results
            transformation_details = {
                "method": "UpwardLLMTransformer",
                "experience_success": results.get("experience", {}).success if "experience" in results else None,
                "responsibilities_success": results.get("responsibilities", {}).success if "responsibilities" in results else None,
                "role_success": results.get("role", {}).success if "role" in results else None,
                "skills_success": results.get("skills", {}).success if "skills" in results else None,
                "semantic_validation": validation_details,
            }
            
            # If validation failed, use fallback
            if not is_valid and validation_details.get("rejection_reason"):
                logger.warning(f"Senior transformation failed validation, using fallback")
                transformed = copy.deepcopy(resume)
                role = transformed.get("role", "")
                if not any(p in role.lower() for p in ["senior", "lead", "principal"]):
                    transformed["role"] = f"Senior {role}"
                transformed["experience_level"] = "senior"
                transformation_details["method"] = "fallback_after_validation_failure"
                return transformed, transformation_details
            
            logger.debug(f"Resume transformed to senior level: {resume.get('role', '')} -> {transformed_resume.get('role', '')}")
            return transformed_resume, transformation_details
            
        except Exception as e:
            logger.error(f"Error in upward transformation, using fallback: {e}")
            # Fallback to simple transformation
            transformed = copy.deepcopy(resume)
            role = transformed.get("role", "")
            if not any(p in role.lower() for p in ["senior", "lead", "principal"]):
                transformed["role"] = f"Senior {role}"
            transformed["experience_level"] = "senior"
            return transformed, {"method": "fallback", "error": str(e)}

    def _transform_job_to_junior(self, job: Dict[str, Any], resume: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transform job to junior/foundational level.
        
        Returns:
            Tuple of (transformed_job, transformation_details)
        """
        job_title = job.get("title", "")
        job_desc = job.get("description", {})
        if isinstance(job_desc, dict):
            job_desc_text = job_desc.get("original", "")
        else:
            job_desc_text = str(job_desc)
        job_skills = self._extract_skills_list(job.get("required_skills", job.get("skills", [])))
        
        # Build ESCO context for domain guidance
        esco_context = self._build_esco_context(resume, job)
        
        prompt = self.TRANSFORM_JOB_TO_JUNIOR_PROMPT.format(
            title=job_title,
            description=job_desc_text[:500],
            skills=", ".join(job_skills[:10]),
            esco_context=self._format_esco_context(esco_context)
        )
        
        result = self._call_llm_json(prompt)
        
        transformed = copy.deepcopy(job)
        transformation_details = {"method": "LLM"}
        
        if result:
            transformed["title"] = result.get("title", f"Junior {job_title}")
            transformed_desc = result.get("description", job_desc_text)
            if isinstance(transformed.get("description"), dict):
                transformed["description"]["original"] = transformed_desc
            else:
                transformed["description"] = {"original": transformed_desc}
            transformed["required_experience_level"] = result.get("required_experience_level", "junior")
            transformed["experience_level"] = "junior"
            transformation_details["llm_success"] = True
            
            # Validate the job description transformation
            is_valid, validation_details = self._validate_transformation(
                original_text=job_desc_text,
                transformed_text=transformed_desc,
                target_level="junior"
            )
            transformation_details["semantic_validation"] = validation_details
            
            if not is_valid and validation_details.get("rejection_reason"):
                logger.warning(f"Junior job transformation failed validation, using fallback")
                transformed = copy.deepcopy(job)
                if not job_title.lower().startswith("junior"):
                    transformed["title"] = f"Junior {job_title}"
                transformed["required_experience_level"] = "junior"
                transformed["experience_level"] = "junior"
                transformation_details["method"] = "fallback_after_validation_failure"
        else:
            if not job_title.lower().startswith("junior"):
                transformed["title"] = f"Junior {job_title}"
            transformed["required_experience_level"] = "junior"
            transformed["experience_level"] = "junior"
            transformation_details["llm_success"] = False
            transformation_details["method"] = "fallback"
        
        return transformed, transformation_details

    def _transform_job_to_senior(self, job: Dict[str, Any], resume: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transform job to senior/aspirational level.
        
        Returns:
            Tuple of (transformed_job, transformation_details)
        """
        job_title = job.get("title", "")
        job_desc = job.get("description", {})
        if isinstance(job_desc, dict):
            job_desc_text = job_desc.get("original", "")
        else:
            job_desc_text = str(job_desc)
        job_skills = self._extract_skills_list(job.get("required_skills", job.get("skills", [])))
        
        # Build ESCO context for domain guidance
        esco_context = self._build_esco_context(resume, job)
        
        prompt = self.TRANSFORM_JOB_TO_SENIOR_PROMPT.format(
            title=job_title,
            description=job_desc_text[:500],
            skills=", ".join(job_skills[:10]),
            esco_context=self._format_esco_context(esco_context)
        )
        
        result = self._call_llm_json(prompt)
        
        transformed = copy.deepcopy(job)
        transformation_details = {"method": "LLM"}
        
        if result:
            transformed["title"] = result.get("title", f"Senior {job_title}")
            transformed_desc = result.get("description", job_desc_text)
            if isinstance(transformed.get("description"), dict):
                transformed["description"]["original"] = transformed_desc
            else:
                transformed["description"] = {"original": transformed_desc}
            transformed["required_experience_level"] = result.get("required_experience_level", "senior")
            transformed["experience_level"] = "senior"
            transformation_details["llm_success"] = True
            
            # Validate the job description transformation
            is_valid, validation_details = self._validate_transformation(
                original_text=job_desc_text,
                transformed_text=transformed_desc,
                target_level="senior"
            )
            transformation_details["semantic_validation"] = validation_details
            
            if not is_valid and validation_details.get("rejection_reason"):
                logger.warning(f"Senior job transformation failed validation, using fallback")
                transformed = copy.deepcopy(job)
                if not any(p in job_title.lower() for p in ["senior", "lead", "principal"]):
                    transformed["title"] = f"Senior {job_title}"
                transformed["required_experience_level"] = "senior"
                transformed["experience_level"] = "senior"
                transformation_details["method"] = "fallback_after_validation_failure"
        else:
            if not any(p in job_title.lower() for p in ["senior", "lead", "principal"]):
                transformed["title"] = f"Senior {job_title}"
            transformed["required_experience_level"] = "senior"
            transformed["experience_level"] = "senior"
            transformation_details["llm_success"] = False
            transformation_details["method"] = "fallback"
        
        return transformed, transformation_details


    def generate_foundational_to_aspirational_negative(
        self,
        resume: Dict[str, Any],
        job: Dict[str, Any]
    ) -> Optional[GeneratedNegative]:
        """Generate Cross-Match Negative (Foundational→Aspirational): Junior resume → Senior job."""
        try:
            junior_resume, resume_details = self._transform_resume_to_junior(resume, job)
            senior_job, job_details = self._transform_job_to_senior(job, resume)
            
            return GeneratedNegative(
                resume=junior_resume,
                job=senior_job,
                negative_type="Cross-Match Negative (Foundational→Aspirational)",
                transformation_details={
                    "resume_transformation": "junior/foundational",
                    "job_transformation": "senior/aspirational",
                    "original_resume_role": resume.get("role", ""),
                    "transformed_resume_role": junior_resume.get("role", ""),
                    "original_job_title": job.get("title", ""),
                    "transformed_job_title": senior_job.get("title", ""),
                    "resume_transform_method": resume_details.get("method", "unknown"),
                    "job_transform_method": job_details.get("method", "unknown"),
                }
            )
        except Exception as e:
            logger.error(f"Error generating Foundational→Aspirational negative: {e}")
            return None

    def generate_aspirational_to_foundational_negative(
        self,
        resume: Dict[str, Any],
        job: Dict[str, Any]
    ) -> Optional[GeneratedNegative]:
        """Generate Cross-Match Negative (Aspirational→Foundational): Senior resume → Junior job."""
        try:
            senior_resume, resume_details = self._transform_resume_to_senior(resume, job)
            junior_job, job_details = self._transform_job_to_junior(job, resume)
            
            return GeneratedNegative(
                resume=senior_resume,
                job=junior_job,
                negative_type="Cross-Match Negative (Aspirational→Foundational)",
                transformation_details={
                    "resume_transformation": "senior/aspirational",
                    "job_transformation": "junior/foundational",
                    "original_resume_role": resume.get("role", ""),
                    "transformed_resume_role": senior_resume.get("role", ""),
                    "original_job_title": job.get("title", ""),
                    "transformed_job_title": junior_job.get("title", ""),
                    "resume_transform_method": resume_details.get("method", "unknown"),
                    "job_transform_method": job_details.get("method", "unknown"),
                }
            )
        except Exception as e:
            logger.error(f"Error generating Aspirational→Foundational negative: {e}")
            return None

    def generate_original_to_foundational_negative(
        self,
        resume: Dict[str, Any],
        job: Dict[str, Any]
    ) -> Optional[GeneratedNegative]:
        """Generate Cross-Match Negative (Original→Foundational): Original resume → Junior job."""
        try:
            junior_job, job_details = self._transform_job_to_junior(job, resume)
            
            return GeneratedNegative(
                resume=copy.deepcopy(resume),
                job=junior_job,
                negative_type="Cross-Match Negative (Original→Foundational)",
                transformation_details={
                    "resume_transformation": "original",
                    "job_transformation": "junior/foundational",
                    "original_resume_role": resume.get("role", ""),
                    "original_job_title": job.get("title", ""),
                    "transformed_job_title": junior_job.get("title", ""),
                    "job_transform_method": job_details.get("method", "unknown"),
                }
            )
        except Exception as e:
            logger.error(f"Error generating Original→Foundational negative: {e}")
            return None

    def generate_original_to_aspirational_negative(
        self,
        resume: Dict[str, Any],
        job: Dict[str, Any]
    ) -> Optional[GeneratedNegative]:
        """Generate Cross-Match Negative (Original→Aspirational): Original resume → Senior job."""
        try:
            senior_job, job_details = self._transform_job_to_senior(job, resume)
            
            return GeneratedNegative(
                resume=copy.deepcopy(resume),
                job=senior_job,
                negative_type="Cross-Match Negative (Original→Aspirational)",
                transformation_details={
                    "resume_transformation": "original",
                    "job_transformation": "senior/aspirational",
                    "original_resume_role": resume.get("role", ""),
                    "original_job_title": job.get("title", ""),
                    "transformed_job_title": senior_job.get("title", ""),
                    "job_transform_method": job_details.get("method", "unknown"),
                }
            )
        except Exception as e:
            logger.error(f"Error generating Original→Aspirational negative: {e}")
            return None

    def generate_negatives(self, record: Dict[str, Any]) -> NegativeGenerationResult:
        """Generate all negative samples for a single record."""
        resume = record.get("resume", {})
        job = record.get("job", {})
        
        negatives = []
        
        generators = [
            ("Foundational→Aspirational", self.generate_foundational_to_aspirational_negative),
            ("Aspirational→Foundational", self.generate_aspirational_to_foundational_negative),
            ("Original→Foundational", self.generate_original_to_foundational_negative),
            ("Original→Aspirational", self.generate_original_to_aspirational_negative),
        ]
        
        for neg_type, generator in generators:
            try:
                negative = generator(resume, job)
                if negative:
                    negatives.append(negative)
                    logger.debug(f"Generated {neg_type} negative")
                else:
                    logger.warning(f"Failed to generate {neg_type} negative")
            except Exception as e:
                logger.error(f"Error generating {neg_type} negative: {e}")
        
        return NegativeGenerationResult(
            original_record=record,
            negatives=negatives,
            success=len(negatives) > 0,
            error_message=None if negatives else "No negatives generated"
        )

    def _get_career_view(self, negative_type: str) -> str:
        """Get career_view string matching rule-based augmentation format."""
        career_view_map = {
            "Cross-Match Negative (Foundational→Aspirational)": "foundational_resume__aspirational_job",
            "Cross-Match Negative (Aspirational→Foundational)": "aspirational_resume__foundational_job",
            "Cross-Match Negative (Original→Foundational)": "original_resume__foundational_job",
            "Cross-Match Negative (Original→Aspirational)": "original_resume__aspirational_job",
        }
        return career_view_map.get(negative_type, "unknown")

    def format_output_record(
        self,
        original_record: Dict[str, Any],
        negative: GeneratedNegative,
        original_sample_id: str
    ) -> Dict[str, Any]:
        """Format a negative sample for output."""
        type_suffix = negative.negative_type.replace("Cross-Match Negative ", "").replace("(", "").replace(")", "").replace("→", "_to_")
        
        # Extract job_applicant_id from original record
        job_applicant_id = original_record.get("job_applicant_id", 
                                               original_record.get("metadata", {}).get("job_applicant_id"))
        
        return {
            "resume": negative.resume,
            "job": negative.job,
            "label": 0,
            "sample_id": f"{original_sample_id}_{type_suffix}",
            "metadata": {
                "original_record_id": original_record.get("metadata", {}).get("original_record_id"),
                "job_applicant_id": job_applicant_id,
                "augmentation_type": negative.negative_type,
                "label": 0,
                "original_label": "negative",
                "career_view": self._get_career_view(negative.negative_type),
                "transformation_details": negative.transformation_details,
            },
            "job_applicant_id": job_applicant_id
        }

    def format_original_record(
        self,
        record: Dict[str, Any],
        sample_id: str,
        original_label: int
    ) -> Dict[str, Any]:
        """Format the original record for output."""
        # Extract job_applicant_id from original record
        job_applicant_id = record.get("job_applicant_id", 
                                      record.get("metadata", {}).get("job_applicant_id"))
        
        return {
            "resume": record.get("resume", {}),
            "job": record.get("job", {}),
            "label": original_label,
            "sample_id": sample_id,
            "metadata": {
                "original_record_id": record.get("metadata", {}).get("original_record_id"),
                "job_applicant_id": job_applicant_id,
                "augmentation_type": "Original",
                "label": original_label,
                "original_label": "positive" if original_label == 1 else "negative",
                "career_view": "original"
            },
            "job_applicant_id": job_applicant_id
        }
