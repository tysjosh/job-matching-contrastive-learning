"""
Utility functions for data validation and configuration management.
"""

from typing import Dict, Any, List
import json
import logging
from pathlib import Path

def validate_record_structure(record: Dict) -> bool:
    """
    Validate input JSONL record structure.
    
    Args:
        record: Record to validate
        
    Returns:
        True if record structure is valid
    """
    try:
        # Check top-level required fields
        required_top_fields = ['job_applicant_id', 'resume', 'job', 'label']
        for field in required_top_fields:
            if field not in record:
                logging.error(f"Missing required top-level field: {field}")
                return False
        
        # Validate job_applicant_id is integer
        if not isinstance(record['job_applicant_id'], int):
            logging.error("job_applicant_id must be an integer")
            return False
        
        # Validate label is integer
        if not isinstance(record['label'], int):
            logging.error("label must be an integer")
            return False
        
        # Validate resume structure
        if not isinstance(record['resume'], dict):
            logging.error("resume must be a dictionary")
            return False
        
        # Validate job structure
        if not isinstance(record['job'], dict):
            logging.error("job must be a dictionary")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating record structure: {e}")
        return False

def validate_resume_fields(resume: Dict) -> bool:
    """
    Validate required resume fields.
    
    Args:
        resume: Resume data to validate
        
    Returns:
        True if required fields are present
    """
    try:
        # Required resume fields based on the data structure
        required_fields = [
            'original_text', 'role', 'experience', 'experience_level',
            'skills', 'education', 'certifications', 'responsibilities',
            'keywords', 'text_stats', 'skill_stats'
        ]
        
        for field in required_fields:
            if field not in resume:
                logging.error(f"Missing required resume field: {field}")
                return False
        
        # Validate field types
        string_fields = ['original_text', 'role', 'experience', 'experience_level', 'education']
        for field in string_fields:
            if not isinstance(resume[field], str):
                logging.error(f"Resume field '{field}' must be a string")
                return False
        
        # Validate array fields
        array_fields = ['skills', 'certifications', 'responsibilities', 'keywords']
        for field in array_fields:
            if not isinstance(resume[field], list):
                logging.error(f"Resume field '{field}' must be a list")
                return False
        
        # Validate skills structure
        for skill in resume['skills']:
            if not isinstance(skill, dict):
                logging.error("Each skill must be a dictionary")
                return False
            skill_required_fields = ['name', 'level', 'category', 'source']
            for skill_field in skill_required_fields:
                if skill_field not in skill:
                    logging.error(f"Missing required skill field: {skill_field}")
                    return False
                if not isinstance(skill[skill_field], str):
                    logging.error(f"Skill field '{skill_field}' must be a string")
                    return False
        
        # Validate stats are dictionaries
        if not isinstance(resume['text_stats'], dict):
            logging.error("text_stats must be a dictionary")
            return False
        
        if not isinstance(resume['skill_stats'], dict):
            logging.error("skill_stats must be a dictionary")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating resume fields: {e}")
        return False

def load_configuration(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If configuration validation fails
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate the loaded configuration
        if not validate_configuration(config):
            raise ValueError("Configuration validation failed")
        
        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise

def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
    """
    try:
        # Check required top-level sections
        required_sections = ['sentence_transformer_paraphrasing', 'masking', 'output', 'processing']
        for section in required_sections:
            if section not in config:
                logging.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate sentence_transformer_paraphrasing section
        st_config = config['sentence_transformer_paraphrasing']
        st_required_fields = ['model_name', 'preserve_terms', 'synonym_similarity_threshold', 
                             'max_replacements_per_sentence', 'protected_patterns']
        for field in st_required_fields:
            if field not in st_config:
                logging.error(f"Missing SentenceTransformer config field: {field}")
                return False
        
        # Validate SentenceTransformer model configuration
        if not isinstance(st_config['model_name'], str) or not st_config['model_name'].strip():
            logging.error("model_name must be a non-empty string")
            return False
        
        if not isinstance(st_config['preserve_terms'], list):
            logging.error("preserve_terms must be a list")
            return False
        
        if not isinstance(st_config['synonym_similarity_threshold'], (int, float)):
            logging.error("synonym_similarity_threshold must be a number")
            return False
        
        if not (0.0 <= st_config['synonym_similarity_threshold'] <= 1.0):
            logging.error("synonym_similarity_threshold must be between 0.0 and 1.0")
            return False
        
        if not isinstance(st_config['max_replacements_per_sentence'], int) or st_config['max_replacements_per_sentence'] < 0:
            logging.error("max_replacements_per_sentence must be a non-negative integer")
            return False
        
        if not isinstance(st_config['protected_patterns'], list):
            logging.error("protected_patterns must be a list")
            return False
        
        # Validate masking section
        masking_config = config['masking']
        masking_required_fields = ['role_placeholder', 'education_placeholder', 'skills_removal_rate', 'field_types']
        for field in masking_required_fields:
            if field not in masking_config:
                logging.error(f"Missing masking config field: {field}")
                return False
        
        if not isinstance(masking_config['skills_removal_rate'], list) or len(masking_config['skills_removal_rate']) != 2:
            logging.error("skills_removal_rate must be a list of two numbers")
            return False
        
        min_rate, max_rate = masking_config['skills_removal_rate']
        if not (0.0 <= min_rate <= max_rate <= 1.0):
            logging.error("skills_removal_rate values must be between 0.0 and 1.0, with min <= max")
            return False
        
        # Validate output section
        output_config = config['output']
        output_required_fields = ['augmentations_per_record', 'include_original', 'output_file', 'summary_file']
        for field in output_required_fields:
            if field not in output_config:
                logging.error(f"Missing output config field: {field}")
                return False
        
        if not isinstance(output_config['augmentations_per_record'], int) or output_config['augmentations_per_record'] <= 0:
            logging.error("augmentations_per_record must be a positive integer")
            return False
        
        if not isinstance(output_config['include_original'], bool):
            logging.error("include_original must be a boolean")
            return False
        
        # Validate processing section
        processing_config = config['processing']
        processing_required_fields = ['batch_size', 'log_level', 'preserve_job_info', 'preserve_applicant_id']
        for field in processing_required_fields:
            if field not in processing_config:
                logging.error(f"Missing processing config field: {field}")
                return False
        
        if not isinstance(processing_config['batch_size'], int) or processing_config['batch_size'] <= 0:
            logging.error("batch_size must be a positive integer")
            return False
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if processing_config['log_level'] not in valid_log_levels:
            logging.error(f"log_level must be one of: {valid_log_levels}")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating configuration: {e}")
        return False

def is_augmented_record(record: Dict) -> bool:
    """
    Check if a record is an augmented record.
    
    Args:
        record: Record to check
        
    Returns:
        True if record is augmented (has augmentation_type field)
    """
    return 'augmentation_type' in record and record['augmentation_type'] is not None

def get_augmentation_type(record: Dict) -> str:
    """
    Get the augmentation type of a record.
    
    Args:
        record: Record to check
        
    Returns:
        Augmentation type string, or 'original' if not augmented
    """
    if is_augmented_record(record):
        return record['augmentation_type']
    return 'original'

def filter_augmented_records(records: List[Dict]) -> List[Dict]:
    """
    Filter records to return only augmented ones.
    
    Args:
        records: List of records to filter
        
    Returns:
        List containing only augmented records
    """
    return [record for record in records if is_augmented_record(record)]

def filter_original_records(records: List[Dict]) -> List[Dict]:
    """
    Filter records to return only original (non-augmented) ones.
    
    Args:
        records: List of records to filter
        
    Returns:
        List containing only original records
    """
    return [record for record in records if not is_augmented_record(record)]