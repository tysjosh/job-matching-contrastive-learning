# LLM Career-Aware Data Augmentation System

## Overview

The LLM Career-Aware Data Augmentation System transforms resume-job pairs into career progression views using Large Language Models. Unlike the rule-based `augmentation` module, this system leverages LLM capabilities for more natural, contextually appropriate transformations while maintaining semantic coherence and career realism.

The system implements a "Career Time Machine" approach that generates 2x training data by creating:

- **Aspirational View**: Senior-level perspective with leadership, impact, and strategic language
- **Foundational View**: Junior-level perspective with learning, support, and task-focused language

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM Augmentation System                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Entry Point                                                                │
│  └── LLMAugmentationOrchestrator                                           │
│      ├── Loads configuration from llm_augmentation/config.json             │
│      ├── Processes Source_Records from processed_combined_data.jsonl       │
│      └── Outputs to llm_augmentation/augmented_data.jsonl                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Transformation Layer                                                       │
│  ├── UpwardLLMTransformer (aspirational views)                             │
│  ├── DownwardLLMTransformer (foundational views)                           │
│  └── JobLLMTransformer (job description transformation)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Protection & Validation Layer                                             │
│  ├── TechnicalTermProtector                                                │
│  └── SemanticCoherenceValidator                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Fallback Layer                                                            │
│  └── RuleBasedFallback (uses existing augmentation module)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

Ensure you have the required dependencies:

```bash
pip install openai sentence-transformers
```

Set your LLM API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

Run the augmentation system from the project root:

```bash
python run_llm_augmentation.py
```

### With Custom Input/Output

```bash
python run_llm_augmentation.py \
    --input preprocess/processed_combined_enriched_data.jsonl \
    --output llm_augmentation/augmented_data.jsonl \
    --config llm_augmentation/config.json
```

### Programmatic Usage

```python
from llm_augmentation import LLMAugmentationOrchestrator

# Initialize orchestrator
orchestrator = LLMAugmentationOrchestrator(config_path="llm_augmentation/config.json")

# Process a single record
source_record = {
    "job_applicant_id": "123",
    "resume": {
        "experience": "Developed web applications...",
        "responsibilities": ["Built features", "Fixed bugs"],
        "role": "Software Developer",
        "skills": [{"name": "Python", "proficiency": "intermediate"}]
    },
    "job": {
        "title": "Software Engineer",
        "description": {"original": "Looking for a developer..."}
    }
}

augmented_views = orchestrator.augment_record(source_record)
# Returns 2 views: aspirational and foundational

# Process entire dataset
stats = orchestrator.augment_dataset(
    input_file="preprocess/processed_combined_enriched_data.jsonl",
    output_file="llm_augmentation/augmented_data.jsonl"
)
```

## Configuration

The system is configured via `llm_augmentation/config.json`:

### LLM Provider Settings

```json
{
  "llm_provider": {
    "model_name": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1024,
    "api_key_env": "OPENAI_API_KEY",
    "provider_type": "openai"
  }
}
```

### Transformation Settings

```json
{
  "transformation": {
    "upward_target_level": "senior",
    "downward_target_level": "junior",
    "skills_mask_ratio_min": 0.2,
    "skills_mask_ratio_max": 0.4
  }
}
```

### Validation Thresholds

```json
{
  "validation": {
    "min_semantic_similarity": 0.5,
    "max_semantic_similarity": 0.95,
    "min_technical_preservation": 0.95
  }
}
```

### Fallback Configuration

```json
{
  "fallback": {
    "enabled": true,
    "max_retries": 3,
    "use_rule_based": true
  }
}
```

## Components

### LLMAugmentationOrchestrator

Main entry point that coordinates the augmentation pipeline:
- Loads configuration
- Initializes all transformer components
- Processes records and generates augmented views
- Handles validation and fallback logic

### UpwardLLMTransformer

Transforms content to senior-level perspective:
- Adds leadership context and business impact language
- Elevates job titles to senior positions
- Adds advanced skills and upgrades proficiency levels

### DownwardLLMTransformer

Transforms content to junior-level perspective:
- Adds learning context and support language
- Reduces job titles to junior positions
- Masks advanced skills and downgrades proficiency levels

### JobLLMTransformer

Transforms job descriptions to match career levels:
- Adjusts requirements and expectations
- Transforms job titles appropriately

### TechnicalTermProtector

Preserves technical terms during LLM transformation:
- Identifies terms from `dataset/cs_skills.json`
- Uses placeholder substitution during LLM processing
- Restores original terms after transformation

### SemanticCoherenceValidator

Validates transformation quality:
- Computes semantic similarity using sentence embeddings
- Validates career level indicators
- Triggers fallback on validation failure

## Output Format

### Augmented Data (augmented_data.jsonl)

Each line contains an augmented view with metadata:

```json
{
  "job_applicant_id": "123",
  "resume": { ... },
  "job": { ... },
  "_augmentation_meta": {
    "view_type": "aspirational",
    "target_level": "senior",
    "transformation_method": "llm",
    "semantic_similarity": 0.78,
    "technical_term_preservation": 0.98,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Summary (augmentation_summary.json)

Processing statistics:

```json
{
  "total_records_processed": 1000,
  "total_views_generated": 2000,
  "aspirational_views": 1000,
  "foundational_views": 1000,
  "llm_transformations": 1850,
  "fallback_transformations": 150,
  "validation_failures": 50,
  "average_semantic_similarity": 0.75,
  "average_technical_preservation": 0.97,
  "processing_time_seconds": 3600
}
```

## Troubleshooting

### API Key Issues

Ensure your API key is set correctly:

```bash
echo $OPENAI_API_KEY
```

### Rate Limiting

If you encounter rate limits, adjust batch settings in config:

```json
{
  "batch_processing": {
    "batch_size": 5,
    "save_interval": 50
  }
}
```

### Low Semantic Similarity

If transformations are being rejected:
1. Check the `min_semantic_similarity` threshold
2. Review LLM prompts for appropriate transformation level
3. Consider adjusting temperature for more/less variation

### Technical Term Corruption

If technical terms are being modified:
1. Verify `dataset/cs_skills.json` contains the terms
2. Check `min_technical_preservation` threshold
3. Review TechnicalTermProtector logs

## Comparison with Rule-Based Augmentation

| Feature | LLM Augmentation | Rule-Based Augmentation |
|---------|------------------|------------------------|
| Transformation Quality | More natural, contextual | Pattern-based, predictable |
| Processing Speed | Slower (API calls) | Faster (local processing) |
| Cost | API costs | Free |
| Customization | Prompt engineering | Rule configuration |
| Fallback | Uses rule-based | N/A |

## License

Part of the Career-Aware Contrastive Learning project.
