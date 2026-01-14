"""
Main augmentation pipeline orchestrator.
"""

from typing import Dict, List, Any, Optional
import json
import logging
import copy
from pathlib import Path

from .paraphraser import SentenceTransformerParaphraser
from .masker import FieldMasker
from .output_manager import OutputManager
from .utils import load_configuration, validate_record_structure, validate_resume_fields


class SimpleAugmentationPipeline:
    """
    Main orchestrator class that coordinates SentenceTransformer paraphrasing and masking operations.
    Generates exactly 2 augmentations per record while preserving original job_applicant_id and job information.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the augmentation pipeline with configuration and components.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        self.config_path = config_path
        self.config = load_configuration(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components with error handling coordination
        try:
            self._initialize_components()
            self.logger.info("SimpleAugmentationPipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def _initialize_components(self) -> None:
        """Initialize SentenceTransformer paraphraser and masking components with error handling."""
        try:
            # Initialize SentenceTransformer paraphraser with model management
            st_config = self.config['sentence_transformer_paraphrasing']
            self.paraphraser = SentenceTransformerParaphraser(
                model_name=st_config['model_name'],
                preserve_terms=st_config['preserve_terms'],
                similarity_threshold=st_config['synonym_similarity_threshold'],
                max_replacements_per_sentence=st_config['max_replacements_per_sentence'],
                protected_patterns=st_config['protected_patterns']
            )
            self.logger.info(f"Initialized SentenceTransformer paraphraser with model: {st_config['model_name']}")
            
            # Initialize field masker
            masking_config = self.config['masking']
            self.masker = FieldMasker(masking_config)
            self.logger.info("Initialized field masker")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration based on config settings."""
        logger = logging.getLogger(__name__)
        log_level = self.config['processing']['log_level']
        logger.setLevel(getattr(logging, log_level))
        
        # Add console handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_file(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process input JSONL file and generate augmented records.
        
        Args:
            input_path: Path to input JSONL file
            output_dir: Directory to save augmented data
            
        Returns:
            Dictionary with processing statistics
        """
        self.logger.info(f"Starting processing of {input_path}")
        
        # Initialize statistics
        stats = {
            'total_records': 0,
            'processed_records': 0,
            'augmented_records': 0,
            'failed_records': 0,
            'errors': []
        }
        
        try:
            # Load and process records
            records = self._load_jsonl_file(input_path)
            stats['total_records'] = len(records)
            
            augmented_records = []
            batch_size = self.config['processing']['batch_size']
            
            # Process records in batches
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_results = self._process_batch(batch, stats)
                augmented_records.extend(batch_results)
                
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
            
            # Save results using OutputManager
            output_manager = OutputManager(output_dir)
            output_file = output_manager.save_augmented_data(augmented_records)
            summary_file = output_manager.generate_summary_report(stats)
            readme_file = output_manager.create_readme()
            
            stats['output_file'] = output_file
            stats['summary_file'] = summary_file
            stats['readme_file'] = readme_file
            
            self.logger.info(f"Processing completed. Generated {stats['augmented_records']} augmented records")
            
        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            stats['errors'].append(str(e))
            raise
        
        return stats
    
    def _process_batch(self, batch: List[Dict], stats: Dict[str, Any]) -> List[Dict]:
        """
        Process a batch of records with error handling.
        
        Args:
            batch: List of records to process
            stats: Statistics dictionary to update
            
        Returns:
            List of augmented records from the batch
        """
        batch_results = []
        
        for record in batch:
            try:
                # Generate augmentations for this record
                augmentations = self._generate_augmentations(record)
                batch_results.extend(augmentations)
                
                stats['processed_records'] += 1
                stats['augmented_records'] += len(augmentations)
                
            except Exception as e:
                self.logger.warning(f"Failed to process record {record.get('job_applicant_id', 'unknown')}: {e}")
                stats['failed_records'] += 1
                stats['errors'].append(f"Record {record.get('job_applicant_id', 'unknown')}: {str(e)}")
        
        return batch_results
    
    def _generate_augmentations(self, record: Dict) -> List[Dict]:
        """
        Generate exactly 2 augmentations per record using SentenceTransformer paraphrasing and masking.
        Preserves original job_applicant_id and job information.
        
        Args:
            record: Source record to augment
            
        Returns:
            List of exactly 2 augmented records
        """
        # Validate record structure first
        if not validate_record_structure(record):
            raise ValueError("Invalid record structure")
        
        if not validate_resume_fields(record['resume']):
            raise ValueError("Invalid resume fields")
        
        augmentations = []
        augmentations_per_record = self.config['output']['augmentations_per_record']
        
        # Generate exactly 2 augmentations as specified in requirements
        for i in range(augmentations_per_record):
            try:
                # Create deep copy to preserve original structure
                augmented_record = copy.deepcopy(record)
                
                # Preserve original job_applicant_id and job information (Requirements 1.2)
                if not self.config['processing']['preserve_applicant_id']:
                    self.logger.warning("preserve_applicant_id is False, but requirements mandate preservation")
                
                if not self.config['processing']['preserve_job_info']:
                    self.logger.warning("preserve_job_info is False, but requirements mandate preservation")
                
                # Apply augmentation strategy based on iteration
                if i == 0:
                    # First augmentation: SentenceTransformer paraphrasing
                    augmented_record = self._apply_paraphrasing(augmented_record)
                    # Add augmentation metadata for Phase 1 training compatibility
                    augmented_record['metadata'] = {
                        'augmentation_type': 'Aspirational Match',
                        'source': 'simple_augmentation',
                        'method': 'paraphrasing',
                        'record_type': 'augmented'
                    }
                    self.logger.debug(f"Applied paraphrasing to record {record['job_applicant_id']}")
                else:
                    # Second augmentation: Field masking
                    augmented_record, masking_type = self._apply_masking(augmented_record)
                    # Add augmentation metadata for Phase 1 training compatibility
                    augmented_record['metadata'] = {
                        'augmentation_type': 'Foundational Match',
                        'source': 'simple_augmentation',
                        'method': masking_type,
                        'record_type': 'augmented'
                    }
                    self.logger.debug(f"Applied {masking_type} to record {record['job_applicant_id']}")
                
                # Maintain original JSON structure (Requirements 1.4)
                self._validate_json_structure(record, augmented_record)
                
                augmentations.append(augmented_record)
                
            except Exception as e:
                self.logger.error(f"Error generating augmentation {i+1} for record {record.get('job_applicant_id', 'unknown')}: {e}")
                # Continue with other augmentations even if one fails
                continue
        
        # Ensure we have exactly 2 augmentations (Requirements 1.1)
        if len(augmentations) != augmentations_per_record:
            self.logger.warning(f"Generated {len(augmentations)} augmentations instead of {augmentations_per_record} for record {record.get('job_applicant_id', 'unknown')}")
        
        return augmentations
    
    def _apply_paraphrasing(self, record: Dict) -> Dict:
        """
        Apply SentenceTransformer-based paraphrasing to resume experience text.
        
        Args:
            record: Record to apply paraphrasing to
            
        Returns:
            Record with paraphrased experience text
        """
        try:
            # Process resume experience using SentenceTransformer paraphraser
            record['resume'] = self.paraphraser.process_resume_experience(record['resume'])
            return record
            
        except Exception as e:
            self.logger.error(f"Error applying paraphrasing: {e}")
            # Return original record on paraphrasing failure
            return record
    
    def _apply_masking(self, record: Dict) -> tuple[Dict, str]:
        """
        Apply strategic field masking to the record.
        
        Args:
            record: Record to apply masking to
            
        Returns:
            Tuple of (record with applied masking, masking_type)
        """
        try:
            # Apply masking using FieldMasker (randomly selects field type)
            masked_record, masking_type = self.masker.apply_masking(record)
            return masked_record, masking_type
            
        except Exception as e:
            self.logger.error(f"Error applying masking: {e}")
            # Return original record on masking failure
            return record, "no_masking"
    
    def _validate_json_structure(self, original: Dict, augmented: Dict) -> None:
        """
        Validate that augmented record maintains original JSON structure.
        
        Args:
            original: Original record
            augmented: Augmented record
            
        Raises:
            ValueError: If structure validation fails
        """
        # Check that all original top-level keys are preserved
        original_keys = set(original.keys())
        augmented_keys = set(augmented.keys())
        
        # Allow for the new metadata field
        expected_extra_keys = {'metadata'}
        extra_keys = augmented_keys - original_keys
        unexpected_extra_keys = extra_keys - expected_extra_keys
        
        missing_keys = original_keys - augmented_keys
        
        if missing_keys:
            raise ValueError(f"JSON structure mismatch. Missing keys: {missing_keys}")
        
        if unexpected_extra_keys:
            raise ValueError(f"JSON structure mismatch. Unexpected extra keys: {unexpected_extra_keys}")
        
        # Check that job_applicant_id and job are preserved
        if original['job_applicant_id'] != augmented['job_applicant_id']:
            raise ValueError("job_applicant_id was not preserved")
        
        # Check that job structure is preserved (deep comparison not needed, just presence)
        if 'job' not in augmented or not isinstance(augmented['job'], dict):
            raise ValueError("job information structure was not preserved")
        
        # Check that resume structure exists
        if 'resume' not in augmented or not isinstance(augmented['resume'], dict):
            raise ValueError("resume structure was not preserved")
        
        # Validate augmentation_type field - check in metadata where it's actually stored
        augmentation_type = augmented.get('metadata', {}).get('augmentation_type')
        if not augmentation_type:
            raise ValueError("augmentation_type field is missing from metadata")
        
        # Valid augmentation types include both simple augmentation methods and Phase 1 training types
        valid_augmentation_types = {
            # Simple augmentation methods
            'paraphrasing', 'role_masking', 'skills_masking', 'education_masking', 'no_masking',
            # Phase 1 training compatibility types
            'Aspirational Match', 'Foundational Match'
        }
        if augmentation_type not in valid_augmentation_types:
            raise ValueError(f"Invalid augmentation_type: {augmentation_type}. Must be one of {valid_augmentation_types}")
    
    def _load_jsonl_file(self, file_path: str) -> List[Dict]:
        """
        Load and parse JSONL file with error handling for malformed records.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of parsed records
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        records = []
        line_number = 0
        
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    line = line.strip()
                    
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Malformed JSON on line {line_number}: {e}")
                        # Continue processing other records
                        continue
            
            self.logger.info(f"Loaded {len(records)} records from {file_path}")
            
            if len(records) == 0:
                raise ValueError(f"No valid records found in {file_path}")
            
            return records
            
        except Exception as e:
            self.logger.error(f"Error loading JSONL file {file_path}: {e}")
            raise