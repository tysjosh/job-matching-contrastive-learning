"""
Field masking functionality for strategic resume field masking.
"""

from typing import Dict, List
import random
import copy

class FieldMasker:
    """
    Applies strategic masking to resume fields.
    """
    
    def __init__(self, masking_config: Dict):
        """
        Initialize the field masker with masking configuration.
        
        Args:
            masking_config: Dictionary containing masking configuration parameters
        """
        self.role_placeholder = masking_config.get("role_placeholder", "[ROLE_MASKED]")
        self.education_placeholder = masking_config.get("education_placeholder", "[EDUCATION_MASKED]")
        self.skills_removal_rate = masking_config.get("skills_removal_rate", [0.3, 0.5])
        self.field_types = masking_config.get("field_types", ["roles", "skills", "education"])
    
    def select_random_field_type(self) -> str:
        """
        Randomly select a field type for masking.
        
        Returns:
            Random field type from available options
        """
        return random.choice(self.field_types)
    
    def apply_masking(self, record: Dict, field_type: str = None) -> tuple[Dict, str]:
        """
        Apply masking to specified field type.
        
        Args:
            record: Resume record to mask
            field_type: Type of field to mask (roles, skills, education). 
                       If None, randomly selects one.
            
        Returns:
            Tuple of (record with applied masking, field_type that was masked)
        """
        # Create a deep copy to avoid modifying the original record
        masked_record = copy.deepcopy(record)
        
        # Select field type if not provided
        if field_type is None:
            field_type = self.select_random_field_type()
        
        # Apply appropriate masking based on field type
        if field_type == "roles":
            return self._mask_role(masked_record), "role_masking"
        elif field_type == "skills":
            return self._mask_skills(masked_record), "skills_masking"
        elif field_type == "education":
            return self._mask_education(masked_record), "education_masking"
        else:
            # If invalid field type, return original record
            return masked_record, "no_masking"
    
    def _mask_role(self, record: Dict) -> Dict:
        """
        Mask role field with generic placeholder.
        
        Args:
            record: Resume record to mask
            
        Returns:
            Record with masked role field
        """
        if "resume" in record and "role" in record["resume"]:
            record["resume"]["role"] = self.role_placeholder
        return record
    
    def _mask_skills(self, record: Dict) -> Dict:
        """
        Mask skills by removing 30-50% randomly.
        
        Args:
            record: Resume record to mask
            
        Returns:
            Record with masked skills array
        """
        if "resume" in record and "skills" in record["resume"] and isinstance(record["resume"]["skills"], list):
            skills = record["resume"]["skills"]
            if len(skills) > 0:
                # Calculate removal rate between 30-50%
                min_rate, max_rate = self.skills_removal_rate
                removal_rate = random.uniform(min_rate, max_rate)
                
                # Calculate number of skills to keep
                num_to_keep = max(1, int(len(skills) * (1 - removal_rate)))
                
                # Randomly select skills to keep
                record["resume"]["skills"] = random.sample(skills, num_to_keep)
        
        return record
    
    def _mask_education(self, record: Dict) -> Dict:
        """
        Mask education field with generic placeholder.
        
        Args:
            record: Resume record to mask
            
        Returns:
            Record with masked education field
        """
        if "resume" in record and "education" in record["resume"]:
            record["resume"]["education"] = self.education_placeholder
        return record