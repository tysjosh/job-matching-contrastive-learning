#!/usr/bin/env python3
"""
Test the gradient fix with a small dataset.
"""

import sys
from pathlib import Path
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.trainer import ContrastiveLearningTrainer


def create_small_test_data():
    """Create a small test dataset."""
    
    samples = []
    
    # Create 10 samples to test multiple batches
    for i in range(10):
        sample = {
            "resume": {
                "role": f"developer_{i}",
                "experience": [{"description": f"Developer with {i+1} years experience in Python"}],
                "skills": [{"name": "Python", "level": "expert"}],
                "keywords": ["python", "developer", "software"]
            },
            "job": {
                "title": f"Software Engineer {i}",
                "description": {
                    "original": f"Looking for a software engineer with Python skills. Job {i}",
                    "keywords": ["python", "software", "engineer"]
                },
                "skills": []
            },
            "label": "positive",
            "sample_id": f"test_{i}",
            "metadata": {"augmentation_type": "Original", "label": 1}
        }
        samples.append(sample)
    
    # Write test data
    with open("small_test_data.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {len(samples)} test samples")


def test_gradient_fix():
    """Test that the gradient fix works with multiple batches."""
    
    print("=" * 50)
    print("GRADIENT FIX TEST")
    print("=" * 50)
    
    # Create test data
    create_small_test_data()
    
    # Create configuration with small batch size to force multiple batches
    config = TrainingConfig(
        training_phase="supervised",
        batch_size=4,  # Small batch size to create multiple batches
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
        
        # Small cache
        embedding_cache_size=50,
        enable_embedding_preload=False,
        
        # Frequent logging to see progress
        log_frequency=1,
        checkpoint_frequency=5
    )
    
    print(f"Configuration:")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Use diagnostics: True (enabled by default)")
    print()
    
    # Initialize trainer
    try:
        trainer = ContrastiveLearningTrainer(
            config=config,
            output_dir="gradient_test_output"
        )
        print("✅ Trainer initialized successfully")
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        return False
    
    # Run training
    try:
        print("Starting training with gradient fix...")
        results = trainer.train("small_test_data.jsonl")
        
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
    """Run the gradient fix test."""
    
    success = test_gradient_fix()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ GRADIENT FIX TEST PASSED!")
        print("=" * 50)
        print("The gradient computation issue has been resolved.")
        print("Training can now run with diagnostics enabled.")
    else:
        print("\n" + "=" * 50)
        print("❌ GRADIENT FIX TEST FAILED!")
        print("=" * 50)
        print("The gradient issue may still exist.")


if __name__ == "__main__":
    main()