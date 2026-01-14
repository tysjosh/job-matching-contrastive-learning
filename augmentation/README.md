# Career-Aware Data Augmentation System

A comprehensive system for generating high-quality, career-progression-aware training data for contrastive learning models in the career matching domain.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Configuration System](#configuration-system)
- [Usage Guide](#usage-guide)
- [Quality Profiles](#quality-profiles)
- [Integration with Training Pipeline](#integration-with-training-pipeline)
- [Validation and Quality Gates](#validation-and-quality-gates)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The Career-Aware Data Augmentation System transforms original resume-job pairs into multiple career progression views, generating 4x more training data while maintaining semantic coherence and career realism. The system implements a "Career Time Machine" approach that creates aspirational (senior-level) and foundational (junior-level) views of candidates.

### Key Features

- **Career Progression Modeling**: Generates realistic career advancement and regression scenarios
- **Intelligent Paraphrasing**: Career-aware paraphrasing that increases linguistic diversity while preserving technical terms and professional context
- **Enhanced Quality Validation**: Multi-layered validation with embedding monitoring and metadata consistency
- **Configurable Quality Profiles**: Fast, balanced, and high-quality modes for different use cases
- **Technical Term Preservation**: Advanced handling of programming languages, frameworks, and technical concepts
- **Metadata Synchronization**: Ensures experience levels, skill proficiencies, and job titles remain coherent
- **Embedding Diversity Monitoring**: Prevents model collapse through diversity tracking and paraphrasing
- **Pipeline Integration**: Seamless integration with existing training workflows

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Augmentation System                          │
├─────────────────────────────────────────────────────────────────┤
│  Configuration Layer                                            │
│  ├── Quality Profiles (Fast/Balanced/High-Quality)             │
│  ├── Validation Thresholds                                     │
│  └── Integration Parameters                                     │
├─────────────────────────────────────────────────────────────────┤
│  Core Transformation Engine                                     │
│  ├── Career-Aware Augmenter                                    │
│  ├── Upward Transformer (Senior-level views)                   │
│  ├── Downward Transformer (Junior-level views)                 │
│  ├── Career-Aware Paraphraser (Diversity enhancement)          │
│  └── Job Transformer                                           │
├─────────────────────────────────────────────────────────────────┤
│  Quality Assurance Layer                                       │
│  ├── Enhanced Semantic Validator                               │
│  ├── Metadata Synchronizer                                     │
│  ├── Technical Term Preservation                               │
│  └── Embedding Diversity Monitor                               │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration & Integration                                   │
│  ├── Dataset Augmentation Orchestrator                         │
│  ├── Pipeline Integration Utilities                            │
│  └── Job Pool Manager                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Career-Aware Augmenter (`career_aware_augmenter.py`)

The central component that generates career progression views:

```python
from augmentation.career_aware_augmenter import CareerAwareAugmenter

augmenter = CareerAwareAugmenter(
    esco_skills_hierarchy=esco_skills,
    career_graph=career_graph,
    lambda1=0.3,  # Aspirational view weight
    lambda2=0.2   # Foundational view weight
)

# Generate career views
views = augmenter.generate_career_views(
    resume=resume_data,
    job=job_data,
    current_level="mid"
)
```

**Outputs:**
- `views.original`: Original resume
- `views.aspirational`: Senior-level progression
- `views.foundational`: Junior-level regression

### 2. Transformation Components

#### Upward Transformer (`upward_transformer.py`)
Elevates resumes to senior levels by:
- Adding leadership context and impact language
- Enhancing technical responsibilities
- Upgrading skill proficiency levels
- Adjusting job titles appropriately
- **Paraphrasing for diversity**: Generates varied expressions while maintaining professional tone

#### Downward Transformer (`downward_transformer.py`)
Reduces resumes to junior levels by:
- Adding learning and support language
- Simplifying technical responsibilities
- Reducing skill complexity
- Adjusting experience indicators
- **Paraphrasing for diversity**: Creates varied junior-level expressions

#### Career-Aware Paraphraser (`career_aware_paraphraser.py`)
Intelligent paraphrasing system that:
- Maintains career-appropriate language for each level
- Preserves technical terms (programming languages, frameworks)
- Increases linguistic diversity to prevent embedding collapse
- Provides configurable diversity targets and quality thresholds
- Tracks usage patterns to ensure varied expressions

#### Job Transformer (`job_transformer.py`)
Transforms job descriptions to match candidate levels:
- Adjusts seniority requirements
- Modifies responsibility complexity
- Updates skill requirements

### 3. Quality Assurance System

#### Enhanced Semantic Validator (`enhanced_semantic_validator.py`)
Advanced validation with comprehensive quality gates:

```python
from augmentation.enhanced_semantic_validator import EnhancedSemanticValidator

validator = EnhancedSemanticValidator(
    upward_min_threshold=0.5,
    upward_max_threshold=0.8,
    min_transformation_quality=0.4,
    min_metadata_consistency=0.7
)

result = validator.validate_transformation_with_metadata(
    original=original_resume,
    transformed=transformed_resume,
    transformation_type="upward"
)
```

#### Metadata Synchronizer (`metadata_synchronizer.py`)
Ensures metadata consistency across transformations:

```python
from augmentation.metadata_synchronizer import MetadataSynchronizer

synchronizer = MetadataSynchronizer()
sync_result = synchronizer.synchronize_experience_metadata(
    resume=transformed_resume,
    transformation_type="upward"
)
```

### 4. Dataset Orchestration

#### Dataset Augmentation Orchestrator (`dataset_augmentation_orchestrator.py`)
Manages the complete augmentation pipeline:

```python
from augmentation.dataset_augmentation_orchestrator import DatasetAugmentationOrchestrator

orchestrator = DatasetAugmentationOrchestrator(
    esco_skills_hierarchy=esco_skills,
    career_graph=career_graph,
    enable_enhanced_validation=True,
    quality_profile="balanced"
)

stats = orchestrator.augment_dataset(
    input_file="processed_data.jsonl",
    output_file="augmented_data.jsonl"
)
```

## Configuration System

### Augmentation Configuration (`augmentation_config.py`)

The system uses a comprehensive configuration system with three quality profiles:

```json
{
  "enhanced_validation": {
    "enabled": true,
    "similarity_thresholds": {
      "upward": {"min": 0.5, "max": 0.8},
      "downward": {"min": 0.4, "max": 0.85}
    },
    "quality_gates": {
      "min_transformation_quality": 0.4,
      "min_metadata_consistency": 0.7,
      "min_technical_preservation": 0.8
    }
  },
  "quality_vs_speed_profiles": {
    "fast": { "enhanced_validation": false },
    "balanced": { "enhanced_validation": true },
    "high_quality": { "strict_thresholds": true }
  }
}
```

### Training Configuration Integration

Training configurations now support augmentation settings:

```json
{
  "augmentation_config_path": "config/augmentation_config.json",
  "augmentation_quality_profile": "balanced",
  "enhanced_augmentation_validation": true,
  "augmentation_diversity_monitoring": true,
  "augmentation_metadata_sync": true
}
```

## Usage Guide

### Basic Usage

1. **Setup Configuration**:
```python
from augmentation.augmentation_config import load_augmentation_config

config = load_augmentation_config(
    config_path="config/augmentation_config.json",
    quality_profile="balanced"  # Enables paraphrasing by default
)
```

2. **Initialize Components**:
```python
from augmentation.dataset_augmentation_orchestrator import DatasetAugmentationOrchestrator

orchestrator = DatasetAugmentationOrchestrator(
    esco_skills_hierarchy=esco_skills,
    career_graph=career_graph,
    augmentation_config=config
)
```

3. **Augment Dataset**:
```python
stats = orchestrator.augment_dataset(
    input_file="input_data.jsonl",
    output_file="augmented_data.jsonl"
)

orchestrator.print_statistics()
```

### Paraphrasing Usage

1. **Direct Paraphraser Usage**:
```python
from augmentation.career_aware_paraphraser import CareerAwareParaphraser

paraphraser = CareerAwareParaphraser(
    preserve_technical_terms=True,
    min_diversity_score=0.3
)

result = paraphraser.paraphrase_experience_text(
    text="Developed web applications using React and Node.js",
    career_level="senior",
    diversity_target=0.4
)

print(f"Original: {result.original_text}")
print(f"Paraphrased: {result.paraphrased_text}")
print(f"Diversity Score: {result.diversity_score:.3f}")
```

2. **Transformer with Paraphrasing**:
```python
from augmentation.upward_transformer import UpwardTransformer

transformer = UpwardTransformer(
    enable_paraphrasing=True,
    paraphrasing_config={
        "preserve_technical_terms": True,
        "min_diversity_score": 0.3,
        "max_semantic_drift": 0.8
    }
)

transformed_resume = transformer.transform(
    resume=resume_data,
    target_level="senior",
    job_context=job_context
)

# Get paraphrasing statistics
stats = transformer.get_paraphrasing_statistics()
print(f"Paraphrasing enabled: {stats['paraphrasing_enabled']}")
```

### Advanced Usage with Pipeline Integration

```python
from augmentation.pipeline_integration import create_pipeline_integrator

# Load training configuration
with open("config/training_config.json") as f:
    training_config = json.load(f)

# Create pipeline integrator
integrator = create_pipeline_integrator(training_config)

# Initialize enhanced components
validator = integrator.initialize_enhanced_validator()
synchronizer = integrator.initialize_metadata_synchronizer()
orchestrator = integrator.initialize_orchestrator(
    esco_skills_hierarchy, career_graph
)

# Apply configuration overrides
overrides = integrator.apply_training_config_overrides()
```

## Quality Profiles

### Fast Profile
- **Use Case**: Rapid prototyping, development testing
- **Features**: Basic validation, minimal quality gates, **paraphrasing disabled**
- **Performance**: ~3x faster processing
- **Quality**: Moderate, suitable for experimentation

```python
config = load_augmentation_config(quality_profile="fast")
```

### Balanced Profile (Default)
- **Use Case**: Production training, general use
- **Features**: Enhanced validation, moderate quality gates, **paraphrasing enabled**
- **Performance**: Standard processing speed
- **Quality**: High, suitable for most applications

```python
config = load_augmentation_config(quality_profile="balanced")
```

### High-Quality Profile
- **Use Case**: Critical applications, research
- **Features**: Strict validation, comprehensive quality gates, **advanced paraphrasing**
- **Performance**: ~2x slower processing
- **Quality**: Maximum, suitable for production models

```python
config = load_augmentation_config(quality_profile="high_quality")
```

## Integration with Training Pipeline

### Automatic Integration

The system automatically integrates with the training pipeline through configuration:

```python
# In training configuration
{
  "augmentation_config_path": "config/augmentation_config.json",
  "augmentation_quality_profile": "balanced"
}
```

### Manual Integration

```python
from contrastive_learning.pipeline import MLPipeline
from contrastive_learning.pipeline_config import PipelineConfig

# Create pipeline config with augmentation settings
pipeline_config = PipelineConfig(
    training_config_path="config/training_config.json",
    augmentation_config_path="config/augmentation_config.json",
    augmentation_quality_profile="balanced"
)

# Run pipeline
pipeline = MLPipeline(pipeline_config)
results = pipeline.run_complete_pipeline("dataset.jsonl")
```

## Validation and Quality Gates

### Quality Gates

The system implements multiple quality gates:

1. **Semantic Similarity**: Ensures transformations maintain meaning
2. **Embedding Distance**: Prevents model collapse
3. **Technical Preservation**: Protects programming languages and frameworks
4. **Metadata Consistency**: Aligns experience levels with content
5. **Length Ratio**: Controls transformation verbosity
6. **Forbidden Transformations**: Prevents semantic violations

### Validation Metrics

```python
# Validation result structure
ValidationResult(
    is_valid=True,
    semantic_score=0.72,
    embedding_similarity=0.68,
    metadata_consistency=0.85,
    technical_preservation=0.91,
    quality_gates_passed=["semantic_similarity", "embedding_distance"],
    quality_gates_failed=[],
    overall_quality_score=0.79
)
```

### Diversity Monitoring

```python
# Check embedding diversity across transformations
diversity_report = validator.validate_embedding_diversity(transformations)

print(f"Diversity Index: {diversity_report.diversity_index:.3f}")
print(f"Collapse Risk: {diversity_report.collapse_risk_score:.3f}")
```

## Examples

### Example 1: Basic Dataset Augmentation

```python
import json
from augmentation.dataset_augmentation_orchestrator import DatasetAugmentationOrchestrator

# Load required data
with open("esco_skills_hierarchy.json") as f:
    esco_skills = json.load(f)

# Initialize orchestrator
orchestrator = DatasetAugmentationOrchestrator(
    esco_skills_hierarchy=esco_skills,
    career_graph=None,  # Optional
    quality_profile="balanced"
)

# Augment dataset
stats = orchestrator.augment_dataset(
    input_file="original_data.jsonl",
    output_file="augmented_data.jsonl"
)

print(f"Generated {stats['generated_records']} records from {stats['original_records']} originals")
print(f"Expansion ratio: {stats['expansion_ratio']:.1f}x")
```

### Example 2: Custom Quality Configuration

```python
from augmentation.augmentation_config import AugmentationConfig, QualityGates

# Create custom configuration
config = AugmentationConfig(
    enhanced_validation_enabled=True,
    quality_gates=QualityGates(
        min_transformation_quality=0.6,
        min_metadata_consistency=0.8,
        min_technical_preservation=0.9
    )
)

# Use with orchestrator
orchestrator = DatasetAugmentationOrchestrator(
    esco_skills_hierarchy=esco_skills,
    career_graph=career_graph,
    augmentation_config=config
)
```

### Example 3: Pipeline Integration

```python
from contrastive_learning.pipeline import MLPipeline
from contrastive_learning.pipeline_config import PipelineConfig

# Configure pipeline with augmentation
config = PipelineConfig.from_file("config/pipeline_config.json")
config.augmentation_quality_profile = "high_quality"

# Run complete pipeline
pipeline = MLPipeline(config)
results = pipeline.run_complete_pipeline(
    dataset_path="dataset.jsonl"
)

print(f"Training completed with augmented data")
print(f"Final model: {results.final_model_path}")
```

## Paraphrasing Configuration

### Configuration Options

```json
{
  "paraphrasing": {
    "enabled": true,
    "preserve_technical_terms": true,
    "min_diversity_score": 0.3,
    "max_semantic_drift": 0.8,
    "career_level_aware": true
  }
}
```

### Dynamic Configuration

```python
# Configure paraphrasing at runtime
augmenter.configure_paraphrasing(
    enable=True,
    min_diversity_score=0.4,
    preserve_technical_terms=True
)

# Reset diversity tracking for new batch
augmenter.reset_diversity_tracking()

# Get paraphrasing statistics
stats = augmenter.get_paraphrasing_statistics()
```

### Quality vs Speed Trade-offs

| Profile | Paraphrasing | Diversity Target | Performance | Use Case |
|---------|-------------|------------------|-------------|----------|
| Fast | Disabled | N/A | 3x faster | Development |
| Balanced | Enabled | 0.3 | Standard | Production |
| High-Quality | Advanced | 0.4 | 2x slower | Research |

## Troubleshooting

### Common Issues

#### 1. Low Transformation Quality
**Symptoms**: High validation failure rate, poor semantic similarity scores
**Solutions**:
- Switch to "fast" profile for development
- Adjust similarity thresholds in configuration
- Check input data quality

#### 2. Embedding Collapse
**Symptoms**: High embedding similarity, low diversity scores
**Solutions**:
- **Enable paraphrasing** (most effective solution)
- Enable diversity monitoring
- Use "high_quality" profile with advanced paraphrasing
- Increase diversity targets in configuration

#### 3. Technical Term Corruption
**Symptoms**: Programming languages or frameworks being modified incorrectly
**Solutions**:
- Enable enhanced validation
- **Ensure paraphrasing preserves technical terms**
- Check technical term preservation settings
- Review CS skills database

#### 4. Metadata Inconsistency
**Symptoms**: Experience levels don't match transformed content
**Solutions**:
- Enable metadata synchronization
- Check experience level mappings
- Validate transformation logic

#### 5. Low Paraphrasing Diversity
**Symptoms**: Paraphrasing not achieving target diversity scores
**Solutions**:
- Increase `min_diversity_score` in configuration
- Check if technical term preservation is too restrictive
- Review original text quality and length
- Use fallback paraphrasing strategies

### Performance Optimization

#### For Speed
```python
# Use fast profile
config = load_augmentation_config(quality_profile="fast")

# Disable expensive validations
config.enhanced_validation_enabled = False
config.metadata_synchronization_enabled = False
```

#### For Quality
```python
# Use high-quality profile
config = load_augmentation_config(quality_profile="high_quality")

# Enable all validations
config.enhanced_validation_enabled = True
config.metadata_synchronization_enabled = True
config.embedding_diversity_monitoring = True
```

### Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Components will output detailed validation information
```

#### Validation Reports
```python
# Get detailed validation report
result = validator.validate_transformation_with_metadata(
    original, transformed, "upward"
)

print("Quality Gates Passed:", result.quality_gates_passed)
print("Quality Gates Failed:", result.quality_gates_failed)
print("Recommendations:", result.recommendations)
```

#### Statistics and Monitoring
```python
# Get augmentation statistics
stats = orchestrator.get_augmentation_statistics()

print(f"Success Rate: {stats['success_rate']:.1%}")
print(f"Validation Failures: {stats['validation_failures']}")
print(f"Quality Gate Failures: {stats['quality_gate_failures']}")
```

## Configuration Files

### Key Configuration Files

- `config/augmentation_config.json`: Main augmentation configuration
- `config/training_config.json`: Training configuration with augmentation settings
- `config/colab_enhanced_training_config.json`: Enhanced training configuration
- `config/fast_training_config.json`: Fast training configuration
- `config/high_quality_training_config.json`: High-quality training configuration

### Testing Configuration

Use the provided test script to validate your configuration:

```bash
python test_augmentation_config_integration.py
```

This will verify:
- Configuration file validity
- Component initialization
- Pipeline integration
- Quality profile functionality

## Contributing

When extending the augmentation system:

1. **Follow the validation pattern**: All new transformations should implement quality gates
2. **Maintain backward compatibility**: Existing configurations should continue working
3. **Add comprehensive tests**: Include unit tests for new components
4. **Update configuration schema**: Document new configuration options
5. **Preserve technical terms**: Ensure programming languages and frameworks are handled correctly

## License

This augmentation system is part of the Career-Aware Contrastive Learning project.