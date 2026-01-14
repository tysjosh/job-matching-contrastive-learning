"""
Metadata Synchronization Engine: Ensures metadata consistency with transformed content

This component synchronizes metadata fields with transformed content to maintain
semantic coherence across experience levels, skill proficiency, and job titles.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Import config loader - required dependency
from augmentation.transformation_config_loader import get_config_loader


class ExperienceLevel(Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    EXECUTIVE = "executive"


class SkillProficiency(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class SynchronizationResult:
    """Result of metadata synchronization operation"""
    success: bool
    updated_fields: List[str]
    consistency_score: float
    warnings: List[str]
    errors: List[str]
    synchronized_metadata: Dict[str, Any]


@dataclass
class ConsistencyReport:
    """Report on metadata consistency validation"""
    experience_level_aligned: bool
    skill_proficiency_aligned: bool
    job_title_coherent: bool
    years_experience_consistent: bool
    consistency_score: float
    misalignments: List[str]
    recommendations: List[str]


class MetadataSynchronizer:
    """
    Synchronizes metadata fields with transformed content to ensure consistency.
    
    Key features:
    - Experience level alignment with content
    - Skill proficiency synchronization
    - Job title coherence validation
    - Years of experience consistency
    - Automatic metadata correction
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize metadata synchronizer with mapping rules.
        
        Args:
            config_dir: Optional custom config directory path
        
        Raises:
            RuntimeError: If configuration files cannot be loaded
        """
        # Initialize config loader
        self.config_loader = get_config_loader(config_dir)
        
        self._load_mappings()
    
    def _load_mappings(self):
        """Load mappings from config files"""
        
        self._load_mappings_from_config()
        
        # Validate required data was loaded
        if not self.experience_level_mappings:
            raise RuntimeError("Failed to load experience level mappings from config. Check config/metadata_mappings/experience_levels.yaml")
        if not self.skill_proficiency_mappings:
            raise RuntimeError("Failed to load skill proficiency mappings from config. Check config/metadata_mappings/skill_proficiency.yaml")
        
        logger.info("Loaded metadata mappings from config files")
    
    def _load_mappings_from_config(self):
        """Load mappings from configuration files"""
        # Load experience level mappings
        exp_config = self.config_loader.load_experience_level_mappings()
        self.experience_level_mappings = {}
        
        for level_name, level_data in exp_config.items():
            level_enum = self._get_experience_level_enum(level_name)
            if level_enum:
                years_range = level_data.get('years_range', {})
                skill_levels = [self._get_skill_proficiency_enum(s) for s in level_data.get('skill_levels', [])]
                skill_levels = [s for s in skill_levels if s is not None]
                
                self.experience_level_mappings[level_enum] = {
                    'keywords': level_data.get('keywords', []),
                    'years_range': (years_range.get('min', 0), years_range.get('max', 20)),
                    'skill_levels': skill_levels,
                    'title_indicators': level_data.get('title_indicators', [])
                }
        
        # Load skill proficiency mappings
        skill_config = self.config_loader.load_skill_proficiency_mappings()
        self.skill_proficiency_mappings = {}
        
        for prof_name, prof_data in skill_config.items():
            prof_enum = self._get_skill_proficiency_enum(prof_name)
            if prof_enum:
                years_range = prof_data.get('years_range', {})
                self.skill_proficiency_mappings[prof_enum] = {
                    'keywords': prof_data.get('keywords', []),
                    'experience_indicators': prof_data.get('experience_indicators', []),
                    'years_range': (years_range.get('min', 0), years_range.get('max', 15))
                }
        
        # Load transformation patterns
        self.transformation_patterns = self.config_loader.load_transformation_patterns()
    
    def _get_experience_level_enum(self, level_name: str) -> Optional[ExperienceLevel]:
        """Convert level name string to ExperienceLevel enum"""
        level_map = {
            'junior': ExperienceLevel.JUNIOR,
            'mid': ExperienceLevel.MID,
            'senior': ExperienceLevel.SENIOR,
            'executive': ExperienceLevel.EXECUTIVE
        }
        return level_map.get(level_name.lower())
    
    def _get_skill_proficiency_enum(self, prof_name: str) -> Optional[SkillProficiency]:
        """Convert proficiency name string to SkillProficiency enum"""
        prof_map = {
            'beginner': SkillProficiency.BEGINNER,
            'intermediate': SkillProficiency.INTERMEDIATE,
            'advanced': SkillProficiency.ADVANCED,
            'expert': SkillProficiency.EXPERT
        }
        return prof_map.get(prof_name.lower())

    def synchronize_experience_metadata(self,
                                        resume: Dict[str, Any],
                                        transformation_type: str,
                                        target_level: Optional[str] = None) -> SynchronizationResult:
        """
        Synchronize experience metadata with transformed content.
        
        Args:
            resume: Resume data with transformed content
            transformation_type: 'upward' or 'downward'
            target_level: Optional target experience level
            
        Returns:
            SynchronizationResult with updated metadata and consistency metrics
        """
        updated_fields = []
        warnings = []
        errors = []
        
        try:
            # Extract current metadata
            current_metadata = resume.get('metadata', {})
            synchronized_metadata = current_metadata.copy()
            
            # Extract text content for analysis
            text_content = self._extract_text_content(resume)
            
            # Determine target experience level
            if not target_level:
                target_level = self._infer_target_experience_level(
                    text_content, transformation_type, current_metadata.get('experience_level')
                )
            
            # Synchronize experience level
            if self._update_experience_level(synchronized_metadata, target_level, text_content):
                updated_fields.append('experience_level')
            
            # Synchronize years of experience
            if self._update_years_experience(synchronized_metadata, target_level, text_content):
                updated_fields.append('years_experience')
            
            # Synchronize skill proficiency levels
            skill_updates = self._update_skill_proficiency(resume, target_level, transformation_type)
            if skill_updates:
                updated_fields.extend(skill_updates)
            
            # Synchronize job title if present
            if self._update_job_title_metadata(synchronized_metadata, target_level, text_content):
                updated_fields.append('job_title_level')
            
            # Validate consistency
            consistency_report = self.validate_skill_metadata_consistency(
                resume.get('skills', []), text_content, target_level
            )
            
            # Generate warnings for inconsistencies
            if consistency_report.consistency_score < 0.8:
                warnings.extend(consistency_report.misalignments)
            
            return SynchronizationResult(
                success=True,
                updated_fields=updated_fields,
                consistency_score=consistency_report.consistency_score,
                warnings=warnings,
                errors=errors,
                synchronized_metadata=synchronized_metadata
            )
            
        except Exception as e:
            logger.error(f"Metadata synchronization error: {e}")
            errors.append(f"Synchronization failed: {str(e)}")
            
            return SynchronizationResult(
                success=False,
                updated_fields=[],
                consistency_score=0.0,
                warnings=warnings,
                errors=errors,
                synchronized_metadata=resume.get('metadata', {})
            )

    def validate_skill_metadata_consistency(self,
                                            skills_array: List[Any],
                                            experience_text: str,
                                            experience_level: str) -> ConsistencyReport:
        """
        Validate consistency between skills array, experience text, and experience level.
        
        Args:
            skills_array: List of skills with proficiency levels
            experience_text: Experience description text
            experience_level: Overall experience level
            
        Returns:
            ConsistencyReport with detailed consistency analysis
        """
        misalignments = []
        recommendations = []
        
        try:
            # Parse experience level
            exp_level_enum = self._parse_experience_level(experience_level)
            if not exp_level_enum:
                misalignments.append(f"Unknown experience level: {experience_level}")
                return self._create_failed_consistency_report(misalignments)
            
            # Extract skill proficiency from text
            text_proficiency_indicators = self._extract_proficiency_indicators(experience_text)
            
            # Extract years of experience
            years_experience = self._extract_years_experience(experience_text)
            
            # Validate experience level alignment
            experience_aligned = self._validate_experience_level_alignment(
                exp_level_enum, text_proficiency_indicators, years_experience
            )
            if not experience_aligned:
                misalignments.append("Experience level doesn't match text proficiency indicators")
                recommendations.append("Update experience level to match text content")
            
            # Validate skill proficiency alignment
            skill_aligned = self._validate_skill_array_alignment(
                skills_array, exp_level_enum, text_proficiency_indicators
            )
            if not skill_aligned:
                misalignments.append("Skills array proficiency doesn't match experience level")
                recommendations.append("Update skill proficiency levels to match experience level")
            
            # Validate job title coherence (if available in text)
            job_title_level = self._extract_job_title_level(experience_text)
            title_coherent = self._validate_job_title_coherence(job_title_level, exp_level_enum)
            if not title_coherent:
                misalignments.append("Job title seniority doesn't match experience level")
                recommendations.append("Ensure job title reflects appropriate seniority")
            
            # Validate years of experience consistency
            years_consistent = self._validate_years_consistency(years_experience, exp_level_enum)
            if not years_consistent:
                misalignments.append("Years of experience inconsistent with experience level")
                recommendations.append("Adjust years of experience to match seniority level")
            
            # Calculate overall consistency score
            consistency_components = [experience_aligned, skill_aligned, title_coherent, years_consistent]
            consistency_score = sum(consistency_components) / len(consistency_components)
            
            return ConsistencyReport(
                experience_level_aligned=experience_aligned,
                skill_proficiency_aligned=skill_aligned,
                job_title_coherent=title_coherent,
                years_experience_consistent=years_consistent,
                consistency_score=consistency_score,
                misalignments=misalignments,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
            misalignments.append(f"Validation error: {str(e)}")
            return self._create_failed_consistency_report(misalignments)

    def _infer_target_experience_level(self,
                                       text_content: str,
                                       transformation_type: str,
                                       current_level: Optional[str]) -> str:
        """Infer target experience level from transformation type and content"""
        
        # Parse current level
        current_enum = self._parse_experience_level(current_level) if current_level else None
        
        # Extract indicators from text
        text_indicators = self._extract_experience_level_indicators(text_content)
        
        # Determine target based on transformation type
        if transformation_type == 'upward':
            if current_enum == ExperienceLevel.JUNIOR:
                return ExperienceLevel.MID.value
            elif current_enum == ExperienceLevel.MID:
                return ExperienceLevel.SENIOR.value
            elif current_enum == ExperienceLevel.SENIOR:
                return ExperienceLevel.EXECUTIVE.value
            else:
                # Infer from text if no current level
                if any(indicator in text_indicators for indicator in ['senior', 'lead', 'principal']):
                    return ExperienceLevel.SENIOR.value
                elif any(indicator in text_indicators for indicator in ['experienced', 'proficient']):
                    return ExperienceLevel.MID.value
                else:
                    return ExperienceLevel.MID.value  # Default upward target
        
        else:  # downward transformation
            if current_enum == ExperienceLevel.EXECUTIVE:
                return ExperienceLevel.SENIOR.value
            elif current_enum == ExperienceLevel.SENIOR:
                return ExperienceLevel.MID.value
            elif current_enum == ExperienceLevel.MID:
                return ExperienceLevel.JUNIOR.value
            else:
                # Infer from text if no current level
                if any(indicator in text_indicators for indicator in ['junior', 'entry', 'beginner']):
                    return ExperienceLevel.JUNIOR.value
                elif any(indicator in text_indicators for indicator in ['intermediate', 'mid']):
                    return ExperienceLevel.MID.value
                else:
                    return ExperienceLevel.JUNIOR.value  # Default downward target

    def _update_experience_level(self,
                                 metadata: Dict[str, Any],
                                 target_level: str,
                                 text_content: str) -> bool:
        """Update experience level in metadata"""
        current_level = metadata.get('experience_level')
        
        if current_level != target_level:
            metadata['experience_level'] = target_level
            logger.debug(f"Updated experience level: {current_level} -> {target_level}")
            return True
        
        return False

    def _update_years_experience(self,
                                 metadata: Dict[str, Any],
                                 target_level: str,
                                 text_content: str) -> bool:
        """Update years of experience to match target level"""
        target_enum = self._parse_experience_level(target_level)
        if not target_enum:
            return False
        
        # Get expected years range for target level
        expected_range = self.experience_level_mappings[target_enum]['years_range']
        
        # Extract current years from text or metadata
        current_years = self._extract_years_experience(text_content)
        if not current_years:
            current_years = metadata.get('years_experience', 0)
        
        # Determine target years within expected range
        min_years, max_years = expected_range
        
        if current_years < min_years:
            target_years = min_years
        elif current_years > max_years:
            target_years = max_years
        else:
            target_years = current_years  # Already in range
        
        if metadata.get('years_experience') != target_years:
            metadata['years_experience'] = target_years
            logger.debug(f"Updated years of experience: {current_years} -> {target_years}")
            return True
        
        return False

    def _update_skill_proficiency(self,
                                  resume: Dict[str, Any],
                                  target_level: str,
                                  transformation_type: str) -> List[str]:
        """Update skill proficiency levels to match target experience level"""
        updated_fields = []
        
        target_enum = self._parse_experience_level(target_level)
        if not target_enum:
            return updated_fields
        
        # Get expected skill levels for target experience level
        expected_skill_levels = self.experience_level_mappings[target_enum]['skill_levels']
        
        # Update skills array if present
        if 'skills' in resume and isinstance(resume['skills'], list):
            for i, skill in enumerate(resume['skills']):
                if isinstance(skill, dict) and 'proficiency' in skill:
                    current_proficiency = skill['proficiency']
                    
                    # Determine appropriate proficiency for this skill
                    target_proficiency = self._determine_target_skill_proficiency(
                        current_proficiency, expected_skill_levels, transformation_type
                    )
                    
                    if current_proficiency != target_proficiency:
                        resume['skills'][i]['proficiency'] = target_proficiency
                        updated_fields.append(f'skills[{i}].proficiency')
                        logger.debug(f"Updated skill proficiency: {current_proficiency} -> {target_proficiency}")
        
        return updated_fields

    def _update_job_title_metadata(self,
                                   metadata: Dict[str, Any],
                                   target_level: str,
                                   text_content: str) -> bool:
        """Update job title level metadata"""
        target_enum = self._parse_experience_level(target_level)
        if not target_enum:
            return False
        
        # Extract job title level from text
        job_title_level = self._extract_job_title_level(text_content)
        
        # Map to experience level
        title_level_mapping = {
            'junior': ExperienceLevel.JUNIOR,
            'mid': ExperienceLevel.MID,
            'senior': ExperienceLevel.SENIOR,
            'executive': ExperienceLevel.EXECUTIVE
        }
        
        expected_title_level = None
        for level, enum_val in title_level_mapping.items():
            if enum_val == target_enum:
                expected_title_level = level
                break
        
        if expected_title_level and metadata.get('job_title_level') != expected_title_level:
            metadata['job_title_level'] = expected_title_level
            logger.debug(f"Updated job title level: {job_title_level} -> {expected_title_level}")
            return True
        
        return False

    def _extract_text_content(self, resume: Dict[str, Any]) -> str:
        """Extract all text content from resume"""
        text_parts = []

        # Extract experience text
        if 'experience' in resume:
            exp = resume['experience']
            if isinstance(exp, str):
                text_parts.append(exp)
            elif isinstance(exp, list):
                for item in exp:
                    if isinstance(item, dict):
                        if 'responsibilities' in item:
                            resp = item['responsibilities']
                            if isinstance(resp, str):
                                text_parts.append(resp)
                            elif isinstance(resp, dict):
                                text_parts.extend(str(v) for v in resp.values())
                    elif isinstance(item, str):
                        text_parts.append(item)

        # Extract summary
        if 'summary' in resume:
            summary = resume['summary']
            if isinstance(summary, str):
                text_parts.append(summary)
            elif isinstance(summary, dict) and 'text' in summary:
                text_parts.append(summary['text'])

        # Extract job title
        if 'job_title' in resume:
            text_parts.append(str(resume['job_title']))
        elif 'title' in resume:
            text_parts.append(str(resume['title']))

        return ' '.join(text_parts)

    def _parse_experience_level(self, level: Optional[str]) -> Optional[ExperienceLevel]:
        """Parse experience level string to enum"""
        if not level:
            return None
        
        level_str = str(level).lower().strip()
        
        for exp_level in ExperienceLevel:
            if exp_level.value == level_str:
                return exp_level
        
        # Try to match keywords
        for exp_level, mapping in self.experience_level_mappings.items():
            if level_str in mapping['keywords']:
                return exp_level
        
        return None

    def _extract_experience_level_indicators(self, text: str) -> List[str]:
        """Extract experience level indicators from text"""
        indicators = []
        text_lower = text.lower()
        
        for exp_level, mapping in self.experience_level_mappings.items():
            for keyword in mapping['keywords']:
                if keyword in text_lower:
                    indicators.append(keyword)
        
        return indicators

    def _extract_proficiency_indicators(self, text: str) -> List[str]:
        """Extract skill proficiency indicators from text"""
        indicators = []
        
        for pattern in self.transformation_patterns['skill_proficiency']:
            matches = re.findall(pattern, text.lower())
            indicators.extend(matches)
        
        return indicators

    def _extract_years_experience(self, text: str) -> Optional[int]:
        """Extract years of experience from text"""
        for pattern in self.transformation_patterns['years_experience']:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None

    def _extract_job_title_level(self, text: str) -> str:
        """Extract job title seniority level from text"""
        text_lower = text.lower()
        
        # Check for senior indicators
        if any(indicator in text_lower for indicator in ['senior', 'lead', 'principal', 'staff', 'architect']):
            return 'senior'
        
        # Check for junior indicators
        if any(indicator in text_lower for indicator in ['junior', 'jr', 'entry', 'associate', 'trainee']):
            return 'junior'
        
        # Check for executive indicators
        if any(indicator in text_lower for indicator in ['director', 'vp', 'cto', 'head', 'chief', 'manager']):
            return 'executive'
        
        return 'mid'  # Default to mid-level

    def _validate_experience_level_alignment(self,
                                             exp_level: ExperienceLevel,
                                             text_indicators: List[str],
                                             years_experience: Optional[int],
                                             strict_mode: bool = False) -> bool:
        """
        Validate that experience level aligns with text indicators and years.
        
        Args:
            exp_level: Expected experience level
            text_indicators: Text indicators extracted from content
            years_experience: Years of experience (if available)
            strict_mode: If True, requires BOTH text and years to align (when years available)
                        If False (default), requires EITHER to align (more permissive)
        
        Returns:
            bool: True if alignment is valid
        """
        expected_mapping = self.experience_level_mappings[exp_level]
        
        # Check text indicators alignment
        text_aligned = any(indicator in expected_mapping['keywords'] for indicator in text_indicators)
        
        # Check years alignment
        years_aligned = True
        years_available = years_experience is not None
        if years_available:
            min_years, max_years = expected_mapping['years_range']
            # Allow some flexibility: ±1 year tolerance
            years_aligned = (min_years - 1) <= years_experience <= (max_years + 1)
        
        if strict_mode and years_available:
            # Strict mode: require both text AND years to align when years are available
            return text_aligned and years_aligned
        else:
            # Permissive mode (default): either should align
            # This is appropriate for transformations where not all metadata may be present
            return text_aligned or years_aligned

    def _validate_skill_array_alignment(self,
                                        skills_array: List[Any],
                                        exp_level: ExperienceLevel,
                                        text_indicators: List[str]) -> bool:
        """Validate that skills array proficiency aligns with experience level"""
        if not skills_array:
            return True  # No skills to validate
        
        expected_skill_levels = self.experience_level_mappings[exp_level]['skill_levels']
        expected_proficiency_names = [level.value for level in expected_skill_levels]
        
        # Count skills at appropriate proficiency levels
        aligned_skills = 0
        total_skills_with_proficiency = 0
        
        for skill in skills_array:
            if isinstance(skill, dict) and 'proficiency' in skill:
                total_skills_with_proficiency += 1
                proficiency = str(skill['proficiency']).lower()
                
                if proficiency in expected_proficiency_names:
                    aligned_skills += 1
        
        if total_skills_with_proficiency == 0:
            return True  # No proficiency data to validate
        
        # Require at least 60% of skills to be at appropriate proficiency
        alignment_ratio = aligned_skills / total_skills_with_proficiency
        return alignment_ratio >= 0.6

    def _validate_job_title_coherence(self, job_title_level: str, exp_level: ExperienceLevel) -> bool:
        """Validate that job title level is coherent with experience level"""
        title_to_exp_mapping = {
            'junior': [ExperienceLevel.JUNIOR],
            'mid': [ExperienceLevel.JUNIOR, ExperienceLevel.MID],
            'senior': [ExperienceLevel.MID, ExperienceLevel.SENIOR],
            'executive': [ExperienceLevel.SENIOR, ExperienceLevel.EXECUTIVE]
        }
        
        expected_exp_levels = title_to_exp_mapping.get(job_title_level, [])
        return exp_level in expected_exp_levels

    def _validate_years_consistency(self, years_experience: Optional[int], exp_level: ExperienceLevel) -> bool:
        """Validate that years of experience is consistent with experience level"""
        if years_experience is None:
            return True  # No years data to validate
        
        expected_range = self.experience_level_mappings[exp_level]['years_range']
        min_years, max_years = expected_range
        
        # Allow some flexibility in ranges
        return (min_years - 1) <= years_experience <= (max_years + 2)

    def _determine_target_skill_proficiency(self,
                                            current_proficiency: str,
                                            expected_levels: List[SkillProficiency],
                                            transformation_type: str) -> str:
        """Determine target skill proficiency based on transformation type"""
        current_enum = None
        
        # Parse current proficiency
        for proficiency in SkillProficiency:
            if proficiency.value == current_proficiency.lower():
                current_enum = proficiency
                break
        
        if not current_enum:
            # Default to intermediate if unknown
            return SkillProficiency.INTERMEDIATE.value
        
        # Select appropriate target from expected levels
        if transformation_type == 'upward':
            # Choose higher proficiency from expected levels
            target = max(expected_levels, key=lambda x: list(SkillProficiency).index(x))
        else:  # downward
            # Choose lower proficiency from expected levels
            target = min(expected_levels, key=lambda x: list(SkillProficiency).index(x))
        
        return target.value

    def _create_failed_consistency_report(self, misalignments: List[str]) -> ConsistencyReport:
        """Create a failed consistency report"""
        return ConsistencyReport(
            experience_level_aligned=False,
            skill_proficiency_aligned=False,
            job_title_coherent=False,
            years_experience_consistent=False,
            consistency_score=0.0,
            misalignments=misalignments,
            recommendations=["Fix metadata synchronization system"]
        )