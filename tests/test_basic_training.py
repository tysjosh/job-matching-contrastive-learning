#!/usr/bin/env python3
"""
Test basic training functionality with minimal configuration.
"""

import sys
from pathlib import Path
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.trainer import ContrastiveLearningTrainer


def create_minimal_test_data():
    """Create minimal test data for verification."""
    
    test_samples = [
        {
            "resume": {
                "role": "data scientist",
                "experience": [{"description": "Python developer with 3 years experience"}],
                "skills": [{"name": "Python", "level": "expert"}],
                "keywords": ["python", "data", "science"]
            },
            "job": {
                "title": "Data Scientist",
                "description": {
                    "original": "Looking for a data scientist with Python skills",
                    "keywords": ["python", "data", "scientist"]
                },
                "skills": []
            },
            "label": "positive",
            "sample_id": "test_1",
            "metadata": {"augmentation_type": "Original", "label": 1}
        },
        {
            "resume": {
                "role": "software engineer",
                "experience": [{"description": "Java developer with 2 years experience"}],
                "skills": [{"name": "Java", "level": "intermediate"}],
                "keywords": ["java", "software", "engineering"]
            },
            "job": {
                "title": "Marketing Manager",
                "description": {
                    "original": "Looking for a marketing professional",
                    "keywords": ["marketing", "manager", "professional"]
                },
                "skills": []
            },
            "label": "negative",
            "sample_id": "test_2",
            "metadata": {"augmentation_type": "Original", "label": 0}
        }
    ]
    
    # Write test data
    with open("test_data.jsonl", "w") as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + "\n")
    
    print("Created minimal test data: test_data.jsonl")


def test_basic_training():
    """Test basic training functionality."""
    
    print("=" * 50)
    print("BASIC TRAINING TEST")
    print("=" * 50)
    
    # Create minimal test data
    create_minimal_test_data()
    
    # Create minimal configuration
    config = TrainingConfig(
        # Use supervised mode (simpler than self-supervised)
        training_phase="supervised",
        
        # Minimal training parameters
        batch_size=32,
        learning_rate=0.001,
        num_epochs=1,
        temperature=0.2,
        
        # Disable complex features
        use_pathway_negatives=False,
        use_view_augmentation=False,
        global_negative_sampling=False,
        
        # Simple text encoder
        text_encoder_model='sentence-transformers/all-MiniLM-L6-v2',
        freeze_text_encoder=True,
        
        # Minimal cache
        embedding_cache_size=100,
        enable_embedding_preload=False,
        
        # Frequent logging
        log_frequency=1,
        checkpoint_frequency=10
    )
    
    print(f"Configuration:")
    print(f"  - Training phase: {config.training_phase}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Use pathway negatives: {config.use_pathway_negatives}")
    print(f"  - Use view augmentation: {config.use_view_augmentation}")
    print()
    
    # Initialize trainer
    try:
        trainer = ContrastiveLearningTrainer(
            config=config,
            output_dir="test_output"
        )
        print("✅ Trainer initialized successfully")
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        return False
    
    # Run training
    try:
        print("Starting basic training test...")
        results = trainer.train("test_data.jsonl")
        
        print("✅ Training completed successfully!")
        print(f"  - Final loss: {results.final_loss:.4f}")
        print(f"  - Training time: {results.training_time:.2f}s")
        print(f"  - Total samples: {results.total_samples}")
        print(f"  - Total batches: {results.total_batches}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False


def main():
    """Run the basic test."""
    
    success = test_basic_training()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ BASIC TEST PASSED!")
        print("=" * 50)
        print("The training system is working correctly.")
        print("You can now try the two-phase training examples.")
    else:
        print("\n" + "=" * 50)
        print("❌ BASIC TEST FAILED!")
        print("=" * 50)
        print("There may be an issue with the training system.")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main()