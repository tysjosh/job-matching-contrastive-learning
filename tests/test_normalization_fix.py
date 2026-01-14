#!/usr/bin/env python3
"""
Test script to demonstrate the normalization fix.

This script shows how the normalization fix resolves training instability
by ensuring consistent embedding scales across batches.
"""

import torch
import torch.nn.functional as F
import numpy as np
from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.trainer import ContrastiveLearningTrainer

def test_normalization_consistency():
    """Test that embeddings are consistently normalized."""
    
    print("🎯 Testing Normalization Fix")
    print("=" * 50)
    
    # Create a trainer to test the model
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        global_negative_sampling=False,
        use_pathway_negatives=False  # Disable for testing
    )
    
    trainer = ContrastiveLearningTrainer(config)
    
    # Test with different input scales to simulate the problem
    print("📊 Testing Embedding Normalization Consistency")
    print("-" * 45)
    
    # Simulate different batch scenarios
    test_cases = [
        {
            "name": "Large magnitude inputs",
            "inputs": torch.randn(5, 384) * 10.0,  # Large scale
            "description": "Simulates batch with large embedding magnitudes"
        },
        {
            "name": "Small magnitude inputs", 
            "inputs": torch.randn(5, 384) * 0.1,   # Small scale
            "description": "Simulates batch with small embedding magnitudes"
        },
        {
            "name": "Mixed magnitude inputs",
            "inputs": torch.cat([
                torch.randn(2, 384) * 10.0,  # Large
                torch.randn(3, 384) * 0.1    # Small
            ]),
            "description": "Simulates batch with mixed embedding magnitudes"
        }
    ]
    
    print("🔍 Before Fix (Problematic Behavior):")
    print("  • Model returns unnormalized embeddings")
    print("  • Loss function normalizes inconsistently") 
    print("  • Different gradient scales across batches")
    print()
    
    print("✅ After Fix (Consistent Behavior):")
    print("  • Model returns L2-normalized embeddings")
    print("  • All embeddings have unit norm (magnitude = 1.0)")
    print("  • Consistent gradient flow across batches")
    print()
    
    # Test the fixed model
    trainer.model.eval()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Description: {test_case['description']}")
        
        inputs = test_case['inputs']
        
        # Show input statistics
        input_norms = torch.norm(inputs, p=2, dim=1)
        print(f"  Input norms: min={input_norms.min():.3f}, max={input_norms.max():.3f}, mean={input_norms.mean():.3f}")
        
        # Get model outputs (should be normalized)
        with torch.no_grad():
            outputs = trainer.model(inputs)
        
        # Check output normalization
        output_norms = torch.norm(outputs, p=2, dim=1)
        print(f"  Output norms: min={output_norms.min():.6f}, max={output_norms.max():.6f}, mean={output_norms.mean():.6f}")
        
        # Verify normalization (should all be ~1.0)
        is_normalized = torch.allclose(output_norms, torch.ones_like(output_norms), atol=1e-6)
        print(f"  ✅ Properly normalized: {is_normalized}")
        
        # Show gradient scale consistency (only if model is in training mode)
        if trainer.model.training:
            outputs.sum().backward()
            grad_norms = []
            for param in trainer.model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            
            if grad_norms:
                print(f"  Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")
            
            # Clear gradients for next test
            trainer.model.zero_grad()
        else:
            print(f"  Gradient analysis: Skipped (model in eval mode)")
        print()
    
    print("🎯 Key Benefits of the Fix:")
    print("  • Consistent embedding magnitudes (all unit norm)")
    print("  • Stable gradient flow across different input scales")
    print("  • Eliminates batch-dependent training dynamics")
    print("  • Better convergence and training stability")
    print("  • Cosine similarity works correctly without extra normalization")

def demonstrate_gradient_stability():
    """Demonstrate how the fix improves gradient stability."""
    
    print("\n" + "=" * 60)
    print("🔬 Gradient Stability Analysis")
    print("=" * 60)
    
    config = TrainingConfig(batch_size=32, use_pathway_negatives=False)
    trainer = ContrastiveLearningTrainer(config)
    
    # Simulate training steps with different input scales
    scales = [0.1, 1.0, 10.0, 100.0]
    gradient_stats = []
    
    print("Testing gradient consistency across different input scales...")
    print()
    
    for scale in scales:
        # Create inputs with different scales
        inputs = torch.randn(10, 384) * scale
        
        # Forward pass
        trainer.model.train()
        outputs = trainer.model(inputs)
        
        # Simple loss (sum of outputs)
        loss = outputs.sum()
        loss.backward()
        
        # Collect gradient statistics
        total_grad_norm = 0
        param_count = 0
        
        for param in trainer.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        
        avg_grad_norm = (total_grad_norm / param_count) ** 0.5 if param_count > 0 else 0
        gradient_stats.append(avg_grad_norm)
        
        print(f"Input scale: {scale:6.1f} → Avg gradient norm: {avg_grad_norm:.6f}")
        
        # Clear gradients
        trainer.model.zero_grad()
    
    # Analyze gradient consistency
    grad_variance = np.var(gradient_stats)
    grad_mean = np.mean(gradient_stats)
    coefficient_of_variation = grad_variance / grad_mean if grad_mean > 0 else float('inf')
    
    print()
    print(f"📊 Gradient Statistics:")
    print(f"  Mean gradient norm: {grad_mean:.6f}")
    print(f"  Gradient variance: {grad_variance:.6f}")
    print(f"  Coefficient of variation: {coefficient_of_variation:.6f}")
    
    if coefficient_of_variation < 0.1:
        print("  ✅ EXCELLENT: Very stable gradients across input scales")
    elif coefficient_of_variation < 0.5:
        print("  ✅ GOOD: Reasonably stable gradients")
    else:
        print("  ⚠️  WARNING: High gradient variance - may cause training instability")

if __name__ == "__main__":
    test_normalization_consistency()
    demonstrate_gradient_stability()