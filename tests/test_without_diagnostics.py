#!/usr/bin/env python3
"""
Test training without diagnostic integration to isolate the gradient issue.
"""

from contrastive_learning.trainer import ContrastiveLearningTrainer
from contrastive_learning.data_structures import TrainingConfig
import sys
from pathlib import Path
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_without_diagnostics():
    """Test training with diagnostics disabled."""

    print("=" * 50)
    print("TEST WITHOUT DIAGNOSTICS")
    print("=" * 50)

    # Create configuration
    config = TrainingConfig(
        training_phase="supervised",
        batch_size=4,
        learning_rate=0.001,
        num_epochs=1,
        temperature=0.2,

        # Disable complex features
        use_pathway_negatives=False,
        use_view_augmentation=False,
        global_negative_sampling=False,

        # Simple text encoder
        text_encoder_model='sentence-transformers/all-mpnet-base-v2',
        freeze_text_encoder=True,

        # Small cache
        embedding_cache_size=50,
        enable_embedding_preload=False,

        # Frequent logging
        log_frequency=1,
        checkpoint_frequency=5
    )

    print("Testing with diagnostics disabled...")

    # Temporarily disable diagnostics by modifying the trainer
    class TrainerWithoutDiagnostics(ContrastiveLearningTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Disable diagnostic integration
            self.diagnostic_integration = None
            print("   ✅ Diagnostics disabled")

    try:
        trainer = TrainerWithoutDiagnostics(
            config=config,
            output_dir="no_diagnostics_test"
        )

        # Use the small test data from previous test
        results = trainer.train("small_test_data.jsonl")

        print("✅ Training completed successfully without diagnostics!")
        print(f"  - Final loss: {results.final_loss:.4f}")
        print(f"  - Total batches: {results.total_batches}")

        return True

    except Exception as e:
        print(f"❌ Training failed even without diagnostics: {e}")
        return False


def main():
    """Run the test."""

    success = test_without_diagnostics()

    if success:
        print("\n" + "=" * 50)
        print("✅ CONFIRMED: Issue is with diagnostic integration")
        print("=" * 50)
        print("Training works fine without diagnostics.")
        print("The gradient issue is caused by the diagnostic system.")
    else:
        print("\n" + "=" * 50)
        print("❌ Issue persists even without diagnostics")
        print("=" * 50)
        print("The gradient issue may be elsewhere in the code.")


if __name__ == "__main__":
    main()
