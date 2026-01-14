"""
Output management for saving augmented data and generating reports.
"""

from typing import List, Dict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Manages file output and report generation.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize output manager with output directory.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"OutputManager initialized with directory: {self.output_dir}")
    
    def save_augmented_data(self, records: List[Dict]) -> str:
        """
        Save augmented records to JSONL file.
        
        Args:
            records: List of augmented records
            
        Returns:
            Path to saved file
        """
        output_file = self.output_dir / "augmented_data.jsonl"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"Saved {len(records)} augmented records to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving augmented data: {e}")
            raise
    
    def generate_summary_report(self, stats: Dict) -> str:
        """
        Generate augmentation summary report.
        
        Args:
            stats: Processing statistics
            
        Returns:
            Path to summary report file
        """
        summary_file = self.output_dir / "augmentation_summary.json"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated summary report: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise
    
    def create_readme(self) -> str:
        """
        Create README file explaining the augmentation process.
        
        Returns:
            Path to README file
        """
        readme_file = self.output_dir / "README.md"
        
        readme_content = """# Simple Resume Augmentation Results

This directory contains the results of the simple resume augmentation process.

## Files

- `augmented_data.jsonl`: Contains all augmented resume records
- `augmentation_summary.json`: Contains processing statistics and summary
- `README.md`: This file explaining the augmentation process

## Augmentation Process

The simple augmentation system generates 2 augmented versions of each resume record using:

1. **SentenceTransformer-based Paraphrasing**: Uses sentence embeddings to generate semantically similar alternative phrasings of resume experience text while preserving technical terms and professional language.

2. **Strategic Field Masking**: Randomly masks one of the following field types:
   - Roles: Replaces role field with generic placeholder
   - Skills: Removes 30-50% of skills randomly
   - Education: Replaces education field with generic placeholder

## Data Structure

Each augmented record maintains the original JSON structure with preserved:
- `job_applicant_id`: Original applicant identifier
- `job`: Complete job information
- `label`: Original label value
- `augmentation_type`: Identifies the augmentation method used

The `resume` section may contain modified content based on the augmentation strategy applied.

### Augmentation Types

- `paraphrasing`: SentenceTransformer-based text paraphrasing applied
- `role_masking`: Role field replaced with placeholder
- `skills_masking`: 30-50% of skills randomly removed
- `education_masking`: Education field replaced with placeholder

## Technical Details

- Model used: all-MiniLM-L6-v2 (SentenceTransformer)
- Protected terms: Technical skills, certifications, experience levels preserved
- Augmentations per record: 2
- Processing maintains professional language standards
"""
        
        try:
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            logger.info(f"Created README file: {readme_file}")
            return str(readme_file)
            
        except Exception as e:
            logger.error(f"Error creating README file: {e}")
            raise