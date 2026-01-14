# Simple Resume Augmentation Results

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
