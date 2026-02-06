"""
Dataset Augmentation Orchestrator: Implements the complete 1→N record expansion strategy

This module generates 4 new training records from each original record using the
Career Time Machine approach with proper match labels and transformations.
"""

import logging
import json
import copy
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from augmentation.career_aware_augmenter import CareerAwareAugmenter
from augmentation.job_transformer import JobTransformer
from augmentation.job_pool_manager import JobPoolManager
from augmentation.enhanced_semantic_validator import EnhancedSemanticValidator
from augmentation.metadata_synchronizer import MetadataSynchronizer
from augmentation.augmentation_config import AugmentationConfig, load_augmentation_config

logger = logging.getLogger(__name__)


@dataclass
class AugmentedRecord:
    """Container for an augmented training record"""
    job_applicant_id: str
    resume: Dict[str, Any]
    job: Dict[str, Any]
    label: int
    augmentation_type: str
    original_id: str


class DatasetAugmentationOrchestrator:
    """
    Orchestrates the complete dataset augmentation strategy.

    Generates 4 new training records from each original record:
    1. Aspirational Match (match=1): Tup candidate + Tup job
    2. Foundational Match (match=1): Tdown candidate + Tdown job
    3. Underqualified Hard Negative (match=0): Tdown candidate + Original job
    """

    def __init__(self,
                 esco_skills_hierarchy: Dict,
                 career_graph: Any,
                 lambda1: Optional[float] = None,
                 lambda2: Optional[float] = None,
                 enable_enhanced_validation: bool = True,
                 augmentation_config: Optional[AugmentationConfig] = None,
                 quality_profile: Optional[str] = None):
        """
        Initialize the dataset augmentation orchestrator.

        Args:
            esco_skills_hierarchy: ESCO skills hierarchy (required)
            career_graph: Career graph for validation (required)
            lambda1: Weight for aspirational view (if None, uses config value)
            lambda2: Weight for foundational view (if None, uses config value)
            enable_enhanced_validation: Enable enhanced semantic validation
            augmentation_config: Augmentation configuration object
            quality_profile: Quality profile to use ('fast', 'balanced', 'high_quality')
        """
        self.career_graph = career_graph
        
        # Load augmentation configuration if not provided
        if augmentation_config is None:
            self.augmentation_config = load_augmentation_config(quality_profile=quality_profile)
        else:
            self.augmentation_config = augmentation_config
            if quality_profile:
                self.augmentation_config.set_active_profile(quality_profile)
        
        # Use config values if lambda weights not explicitly provided
        if lambda1 is None:
            lambda1 = self.augmentation_config.lambda1
        if lambda2 is None:
            lambda2 = self.augmentation_config.lambda2
        
        self.career_augmenter = CareerAwareAugmenter(
            esco_skills_hierarchy=esco_skills_hierarchy,
            career_graph=career_graph,
            lambda1=lambda1,
            lambda2=lambda2
        )
        self.job_transformer = JobTransformer()
        self.job_pool_manager = None  # Will be initialized when dataset is loaded
        
        # Get active profile
        self.active_profile = self.augmentation_config.get_active_profile()
        logger.info(f"Using augmentation quality profile: {self.active_profile.name}")
        
        # Enhanced validation components
        self.enable_enhanced_validation = enable_enhanced_validation and self.active_profile.enhanced_validation
        if self.enable_enhanced_validation:
            # Initialize with profile-specific parameters
            profile_thresholds = self.active_profile.validation_thresholds
            profile_quality_gates = self.active_profile.quality_gates
            profile_diversity = self.active_profile.diversity_requirements
            
            self.enhanced_validator = EnhancedSemanticValidator(
                upward_min_threshold=profile_thresholds.upward_min,
                upward_max_threshold=profile_thresholds.upward_max,
                downward_min_threshold=profile_thresholds.downward_min,
                downward_max_threshold=profile_thresholds.downward_max,
                min_transformation_quality=profile_quality_gates.min_transformation_quality,
                min_metadata_consistency=profile_quality_gates.min_metadata_consistency,
                min_technical_preservation=profile_quality_gates.min_technical_preservation,
                min_diversity_threshold=profile_diversity.min_diversity_threshold,
                max_collapse_risk=profile_diversity.max_collapse_risk
            )
            
            if self.active_profile.metadata_sync_enabled:
                self.metadata_synchronizer = MetadataSynchronizer()
            else:
                self.metadata_synchronizer = None
        else:
            self.enhanced_validator = None
            self.metadata_synchronizer = None

        # Statistics tracking
        self.stats = {
            'original_records': 0,
            'generated_records': 0,
            'aspirational_matches': 0,
            'foundational_matches': 0,
            'overqualified_matches': 0,
            'underqualified_negatives': 0,
            'failed_transformations': 0,
            'validation_failures': 0,
            'metadata_sync_failures': 0,
            'quality_gate_failures': 0
        }

    def augment_dataset(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Augment complete dataset from input file to output file.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output augmented JSONL file

        Returns:
            Dict: Augmentation statistics
        """
        logger.info(
            f"Starting dataset augmentation: {input_file} → {output_file}")

        # Initialize job pool manager for negative generation
        if self.job_pool_manager is None:
            logger.info("Initializing job pool manager...")
            self.job_pool_manager = JobPoolManager(
                dataset_path=input_file,
                esco_config_file='esco_it_career_domains_refined.json',
                esco_csv_path='dataset/esco/',
                esco_skills_hierarchy=self.career_augmenter.esco_skills_hierarchy,
                career_graph=self.career_graph
            )
            stats = self.job_pool_manager.get_stats()
            logger.info(
                f"Job pool loaded: {stats['total_jobs']} jobs across {len(stats['domains'])} domains")

            # Log occupation detection results
            if 'coverage' in stats:
                occupation_rate = stats['coverage']['occupation_detection_rate']
                logger.info(
                    f"Occupation detection rate: {occupation_rate:.1%}")

            if stats['total_occupations_detected'] > 0:
                logger.info(
                    f"Detected {stats['total_occupations_detected']} unique occupations")
            else:
                logger.warning(
                    "No occupations detected - check career graph structure")

        augmented_records = []

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            original_record = json.loads(line.strip())
                            self.stats['original_records'] += 1

                            # Generate 4 augmented records from this original
                            new_records = self._generate_augmented_records(
                                original_record)
                            augmented_records.extend(new_records)

                            # Keep original record too
                            original_record['augmentation_type'] = 'Original'
                            original_record['original_record_id'] = str(
                                original_record['job_applicant_id'])
                            augmented_records.append(original_record)

                            if line_num % 100 == 0:
                                logger.info(
                                    f"Processed {line_num} original records...")

                        except json.JSONDecodeError as e:
                            logger.error(
                                f"JSON decode error on line {line_num}: {e}")
                            continue
                        except Exception as e:
                            logger.error(
                                f"Error processing record {line_num}: {e}")
                            self.stats['failed_transformations'] += 1
                            continue

            # Write augmented dataset
            self._write_augmented_dataset(augmented_records, output_file)

            # Update final statistics
            self.stats['generated_records'] = len(augmented_records)

            logger.info(
                f"Dataset augmentation completed: {len(augmented_records)} total records")
            return self.stats

        except Exception as e:
            logger.error(f"Dataset augmentation failed: {e}")
            raise

    def _generate_augmented_records(self, original_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 4 augmented records from one original record.
        Strategy depends on the original label:
        - label=1 (matches): Generate progression-based augmentations
        - label=0 (mismatches): Generate diverse negative augmentations

        Args:
            original_record: Original training record

        Returns:
            List of 4 augmented records
        """
        original_label = original_record.get('label', 1)

        if original_label == 1:
            return self._generate_match_augmentations(original_record)
        else:
            return []

    def _generate_match_augmentations(self, original_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 3 augmented records for original matches (label=1).
        Creates: 2 positives + 1 negative (existing logic)

        Args:
            original_record: Original training record with label=1

        Returns:
            List of 3 augmented records
        """
        augmented_records = []
        original_id = str(original_record.get('job_applicant_id', 'unknown'))

        try:
            # Extract current experience level from resume
            current_level = self._extract_experience_level(
                original_record['resume'])

            # Generate career views for resume
            career_views = self.career_augmenter.generate_career_views(
                resume=original_record['resume'],
                job=original_record['job'],
                current_level=current_level
            )

            # Get target levels
            senior_level = self._get_next_level(current_level)
            junior_level = self._get_previous_level(current_level)

            # Generate job transformations
            senior_job = self.job_transformer.transform_job_upward(
                original_record['job'], senior_level)
            junior_job = self.job_transformer.transform_job_downward(
                original_record['job'], junior_level)

            # 1. Aspirational Match (match=1): Tup candidate + Tup job
            aspirational_record = self._create_validated_augmented_record(
                original_record=original_record,
                original_id=original_id,
                resume=career_views.aspirational,
                job=senior_job,
                label=1,
                augmentation_type="Aspirational Match (Tup/Tup)",
                transformation_type="upward",
                suffix="1"
            )
            if aspirational_record:
                augmented_records.append(aspirational_record)
                self.stats['aspirational_matches'] += 1

            # 2. Foundational Match (match=1): Tdown candidate + Tdown job
            foundational_record = self._create_validated_augmented_record(
                original_record=original_record,
                original_id=original_id,
                resume=career_views.foundational,
                job=junior_job,
                label=1,
                augmentation_type="Foundational Match (Tdown/Tdown)",
                transformation_type="downward",
                suffix="2"
            )
            if foundational_record:
                augmented_records.append(foundational_record)
                self.stats['foundational_matches'] += 1

            # 3. Overqualified Near-Match (match=1): Tup candidate + Original job
            # overqualified_record = self._create_augmented_record(
            #     original_id=original_id,
            #     resume=career_views.aspirational,
            #     job=original_record['job'],
            #     label=1,
            #     augmentation_type="Overqualified Near-Match (Tup/Original)",
            #     suffix="3"
            # )
            # augmented_records.append(overqualified_record)
            # self.stats['overqualified_matches'] += 1

            # 4. Underqualified Hard Negative (match=0): Tdown candidate + Original job
            underqualified_record = self._create_validated_augmented_record(
                original_record=original_record,
                original_id=original_id,
                resume=career_views.foundational,
                job=original_record['job'],
                label=0,
                augmentation_type="Underqualified Hard Negative (Tdown/Original)",
                transformation_type="downward",
                suffix="4"
            )
            if underqualified_record:
                augmented_records.append(underqualified_record)
                self.stats['underqualified_negatives'] += 1

            return augmented_records

        except Exception as e:
            logger.error(
                f"Failed to generate match augmentations for {original_id}: {e}")
            self.stats['failed_transformations'] += 1
            return []

    def _generate_mismatch_augmentations(self, original_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 4 diverse negative augmentations for original mismatches (label=0).
        Creates: 4 negatives (with rare level-adjusted positives)

        Args:
            original_record: Original training record with label=0

        Returns:
            List of 4 augmented records
        """
        augmented_records = []
        original_id = str(original_record.get('job_applicant_id', 'unknown'))

        try:
            # 1. Domain Mismatch (Hard Negative): Original resume + Job from different domain
            domain_mismatch_record = self._create_domain_mismatch_negative(
                original_record, original_id)
            augmented_records.append(domain_mismatch_record)
            self.stats.setdefault('domain_mismatch_negatives', 0)
            self.stats['domain_mismatch_negatives'] += 1

            # 2. Skill Mismatch (Hard Negative): Original resume + Job with different skills
            skill_mismatch_record = self._create_skill_mismatch_negative(
                original_record, original_id)
            augmented_records.append(skill_mismatch_record)
            self.stats.setdefault('skill_mismatch_negatives', 0)
            self.stats['skill_mismatch_negatives'] += 1

            # 3. Random Negative (Easy Negative): Original resume + Random job
            random_negative_record = self._create_random_negative(
                original_record, original_id)
            augmented_records.append(random_negative_record)
            self.stats.setdefault('random_negatives', 0)
            self.stats['random_negatives'] += 1

            # 4. Positive Correction (Role-Match): Try to fix the mismatch
            positive_correction_record = self._create_role_match_correction_positive(
                original_record, original_id)
            augmented_records.append(positive_correction_record)
            self.stats.setdefault('positive_corrections', 0)
            self.stats['positive_corrections'] += 1

            return augmented_records

        except Exception as e:
            logger.error(
                f"Failed to generate mismatch augmentations for {original_id}: {e}")
            self.stats['failed_transformations'] += 1
            return []

    def _create_validated_augmented_record(self,
                                           original_record: Dict[str, Any],
                                           original_id: str,
                                           resume: Dict[str, Any],
                                           job: Dict[str, Any],
                                           label: int,
                                           augmentation_type: str,
                                           transformation_type: str,
                                           suffix: str) -> Optional[Dict[str, Any]]:
        """Create a new augmented training record with enhanced validation"""
        
        if not self.enable_enhanced_validation:
            return self._create_augmented_record(original_id, resume, job, label, augmentation_type, suffix)
        
        try:
            # 1. Validate transformation quality
            validation_result = self.enhanced_validator.validate_transformation_with_metadata(
                original=original_record['resume'],
                transformed=resume,
                transformation_type=transformation_type
            )
            
            if not validation_result.is_valid:
                logger.debug(f"Validation failed for {augmentation_type}: {validation_result.failure_reasons}")
                self.stats['validation_failures'] += 1
                return None
            
            # 2. Synchronize metadata
            sync_result = self.metadata_synchronizer.synchronize_experience_metadata(
                resume=resume,
                transformation_type=transformation_type
            )
            
            if not sync_result.success:
                logger.debug(f"Metadata synchronization failed for {augmentation_type}: {sync_result.errors}")
                self.stats['metadata_sync_failures'] += 1
                return None
            
            # Update resume with synchronized metadata
            if sync_result.updated_fields:
                resume['metadata'] = sync_result.synchronized_metadata
                logger.debug(f"Updated metadata fields: {sync_result.updated_fields}")
            
            # 3. Quality gate check
            if validation_result.overall_quality_score < self.enhanced_validator.min_transformation_quality:
                logger.debug(f"Quality gate failed for {augmentation_type}: {validation_result.overall_quality_score}")
                self.stats['quality_gate_failures'] += 1
                return None
            
            # Create the record if all validations pass
            return self._create_augmented_record(original_id, resume, job, label, augmentation_type, suffix)
            
        except Exception as e:
            logger.error(f"Enhanced validation error for {augmentation_type}: {e}")
            self.stats['validation_failures'] += 1
            return None

    def _create_augmented_record(self,
                                 original_id: str,
                                 resume: Dict[str, Any],
                                 job: Dict[str, Any],
                                 label: int,
                                 augmentation_type: str,
                                 suffix: str) -> Dict[str, Any]:
        """Create a new augmented training record"""

        # Use original job_applicant_id as integer (not string with suffix)
        try:
            original_job_applicant_id = int(original_id)
        except (ValueError, TypeError):
            # Fallback to original if can't convert
            original_job_applicant_id = original_id

        # Clean resume and job of transformation metadata
        clean_resume = self._clean_transformation_metadata(resume)
        clean_job = self._clean_transformation_metadata(job)

        return {
            'job_applicant_id': original_job_applicant_id,
            'resume': clean_resume,
            'job': clean_job,
            'label': label,
            'augmentation_type': augmentation_type,
            'original_record_id': original_id
        }

    def _clean_transformation_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove transformation metadata from data"""
        cleaned = copy.deepcopy(data)
        if '_transformation_meta' in cleaned:
            del cleaned['_transformation_meta']
        return cleaned

    def _extract_experience_level(self, resume: Dict[str, Any]) -> str:
        """Extract experience level from resume data"""
        # Check if experience_level is explicitly set
        if 'experience_level' in resume:
            return resume['experience_level']

        # Fallback: infer from text
        experience_text = str(resume.get('experience', '')).lower()

        if any(term in experience_text for term in ['senior', 'lead', 'principal']):
            return 'senior'
        elif any(term in experience_text for term in ['junior', 'entry', 'intern']):
            return 'entry'
        else:
            return 'mid'  # Default to mid-level

    def _get_next_level(self, current_level: str) -> str:
        """Get next career level for upward transformation"""
        level_progression = {
            'entry': 'mid',
            'junior': 'mid',
            'mid': 'senior',
            'senior': 'lead',
            'lead': 'principal',
            'principal': 'principal'  # Stay at top
        }
        return level_progression.get(current_level, 'senior')

    def _get_previous_level(self, current_level: str) -> str:
        """Get previous career level for downward transformation"""
        level_regression = {
            'principal': 'lead',
            'lead': 'senior',
            'senior': 'mid',
            'mid': 'entry',
            'junior': 'junior',
            'entry': 'entry'  # Stay at bottom
        }
        return level_regression.get(current_level, 'entry')

    def _write_augmented_dataset(self, records: List[Dict[str, Any]], output_file: str):
        """Write augmented records to JSONL file"""
        logger.info(
            f"Writing {len(records)} augmented records to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')

    def get_augmentation_statistics(self) -> Dict[str, Any]:
        """Get detailed augmentation statistics"""
        total_generated = self.stats['generated_records']
        original_count = self.stats['original_records']

        return {
            **self.stats,
            'expansion_ratio': total_generated / original_count if original_count > 0 else 0,
            'success_rate': 1 - (self.stats['failed_transformations'] / original_count) if original_count > 0 else 0,
            'validation_success_rate': 1 - (self.stats['validation_failures'] / original_count) if original_count > 0 else 0,
            'augmentation_breakdown': {
                'aspirational_matches': self.stats['aspirational_matches'],
                'foundational_matches': self.stats['foundational_matches'],
                'overqualified_matches': self.stats['overqualified_matches'],
                'underqualified_negatives': self.stats['underqualified_negatives']
            },
            'quality_metrics': {
                'validation_failures': self.stats['validation_failures'],
                'metadata_sync_failures': self.stats['metadata_sync_failures'],
                'quality_gate_failures': self.stats['quality_gate_failures']
            }
        }

    def print_statistics(self):
        """Print augmentation statistics"""
        stats = self.get_augmentation_statistics()

        print("\nDataset Augmentation Statistics")
        print("=" * 50)
        print(f"Original records: {stats['original_records']}")
        print(f"Total generated records: {stats['generated_records']}")
        print(f"Expansion ratio: {stats['expansion_ratio']:.1f}x")
        print(f"Success rate: {stats['success_rate']:.1%}")
        
        if self.enable_enhanced_validation:
            print(f"Validation success rate: {stats['validation_success_rate']:.1%}")

        print("\nAugmentation Breakdown:")
        print("-" * 30)
        breakdown = stats['augmentation_breakdown']
        print(
            f"Aspirational Matches (match=1): {breakdown['aspirational_matches']}")
        print(
            f"Foundational Matches (match=1): {breakdown['foundational_matches']}")
        print(
            f"Overqualified Matches (match=1): {breakdown['overqualified_matches']}")
        print(
            f"Underqualified Negatives (match=0): {breakdown['underqualified_negatives']}")

        if self.enable_enhanced_validation:
            print("\nQuality Metrics:")
            print("-" * 20)
            quality = stats['quality_metrics']
            print(f"Validation failures: {quality['validation_failures']}")
            print(f"Metadata sync failures: {quality['metadata_sync_failures']}")
            print(f"Quality gate failures: {quality['quality_gate_failures']}")

        if stats['failed_transformations'] > 0:
            print(
                f"\nFailed transformations: {stats['failed_transformations']}")

    def _create_domain_mismatch_negative(self, original_record: Dict, original_id: str) -> Dict:
        """Create cross-domain hard negative"""
        original_resume = original_record['resume']

        # Get a job from a completely different domain
        different_domain_job = self.job_pool_manager.select_cross_domain_job(
            original_record['job'])

        return self._create_augmented_record(
            original_id=original_id,
            resume=original_resume,
            job=different_domain_job,
            label=0,  # Always negative - cross-domain mismatches
            augmentation_type="Domain Mismatch (Original/Other_Domain)",
            suffix="dm"
        )

    def _create_skill_mismatch_negative(self, original_record: Dict, original_id: str) -> Dict:
        """Create skill-based hard negative"""
        original_resume = original_record['resume']

        # Get a job with significantly different skill requirements
        different_skills_job = self.job_pool_manager.select_skill_mismatch_job(
            original_record['job'])

        return self._create_augmented_record(
            original_id=original_id,
            resume=original_resume,
            job=different_skills_job,
            label=0,  # Always negative - skill gaps can't be fixed by level changes
            augmentation_type="Skill Mismatch (Original/Different_Skills)",
            suffix="sm"
        )

    def _create_random_negative(self, original_record: Dict, original_id: str, suffix: str = "rn") -> Dict:
        """Create random easy negative"""
        original_resume = original_record['resume']

        # Select a completely random job from the dataset
        random_job = self.job_pool_manager.select_random_job_excluding(
            original_record['job'])

        return self._create_augmented_record(
            original_id=original_id,
            resume=original_resume,
            job=random_job,
            label=0,  # Always negative - random pairing
            augmentation_type="Random Negative (Original/Random)",
            suffix=suffix
        )

    def _create_role_match_correction_positive(self, original_record: Dict, original_id: str) -> Dict:
        """Tries to create a positive pair from a mismatch by correcting the job."""
        original_resume = original_record['resume']
        original_job = original_record['job']

        try:
            # Adapt resume structure for occupation detection
            resume_as_job = {
                'title': original_resume.get('role'),
                'description': {'original': original_resume.get('experience')}
            }

            # Step 1: Extract occupations and level
            resume_occupation = self.job_pool_manager._extract_job_occupation_esco(
                resume_as_job)
            job_occupation = self.job_pool_manager._extract_job_occupation_esco(
                original_job)
            resume_level = self._extract_experience_level(original_resume)

            # Step 2: Check for Role Compatibility
            if resume_occupation and resume_occupation == job_occupation:
                # Step 3: Find a new, matching job
                new_job = self.job_pool_manager.find_matching_job(
                    occupation=resume_occupation,
                    level=resume_level
                )

                if new_job:
                    # Step 4: Create the new positive record
                    self.stats.setdefault('positive_corrections_success', 0)
                    self.stats['positive_corrections_success'] += 1
                    return self._create_augmented_record(
                        original_id=original_id,
                        resume=original_resume,
                        job=new_job,
                        label=1,  # This is a corrected POSITIVE
                        augmentation_type="Positive Correction (Role-Match)",
                        suffix="pc"
                    )

            # Step 5 (Fallback): If roles don't match or no job found, create a random negative
            self.stats.setdefault('positive_corrections_fallback', 0)
            self.stats['positive_corrections_fallback'] += 1
            return self._create_random_negative(original_record, original_id, suffix="pc_fallback")

        except Exception as e:
            logger.error(
                f"Role match correction failed for {original_id}: {e}")
            # Fallback: create a simple negative
            return self._create_random_negative(original_record, original_id, suffix="pc_fallback")


def augment_processed_dataset(input_file: str,
                              output_file: str,
                              esco_skills_hierarchy: Dict,
                              career_graph: Any) -> Dict[str, Any]:
    """
    Convenience function to augment the processed dataset.

    Args:
        input_file: Path to processed_combined_data.jsonl
        output_file: Path to output augmented dataset
        esco_skills_hierarchy: ESCO skills hierarchy (required)
        career_graph: Career graph (required)

    Returns:
        Dict: Augmentation statistics
    """
    orchestrator = DatasetAugmentationOrchestrator(
        esco_skills_hierarchy=esco_skills_hierarchy,
        career_graph=career_graph
    )

    stats = orchestrator.augment_dataset(input_file, output_file)
    orchestrator.print_statistics()

    return stats
