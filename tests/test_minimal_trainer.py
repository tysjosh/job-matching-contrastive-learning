#!/usr/bin/env python3
"""
Test with a completely minimal trainer to isolate the gradient issue.
"""

import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from contrastive_learning.data_structures import TrainingConfig


class MinimalModel(nn.Module):
    """Minimal model for testing."""
    
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(384, 128)  # MiniLM embedding size to projection
    
    def forward(self, embeddings):
        return torch.nn.functional.normalize(self.projection(embeddings), p=2, dim=1)


def test_minimal_training():
    """Test with minimal training loop."""
    
    print("=" * 50)
    print("MINIMAL TRAINER TEST")
    print("=" * 50)
    
    # Create minimal model
    model = MinimalModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create fake embeddings (simulating SentenceTransformer output)
    batch_size = 4
    embedding_dim = 384
    
    print("Testing minimal training loop...")
    
    for batch_idx in range(3):  # Test 3 batches
        print(f"Batch {batch_idx + 1}:")
        
        try:
            # Create fake embeddings
            anchor_embs = torch.randn(batch_size, embedding_dim, requires_grad=False)
            positive_embs = torch.randn(batch_size, embedding_dim, requires_grad=False)
            negative_embs = torch.randn(batch_size, embedding_dim, requires_grad=False)
            
            # Forward pass
            anchor_proj = model(anchor_embs)
            positive_proj = model(positive_embs)
            negative_proj = model(negative_embs)
            
            # Simple contrastive loss
            pos_sim = torch.sum(anchor_proj * positive_proj, dim=1) / 0.1
            neg_sim = torch.sum(anchor_proj * negative_proj, dim=1) / 0.1
            
            # InfoNCE loss
            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim)
            loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
            
            print(f"  Loss: {loss.item():.4f}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  ✅ Batch {batch_idx + 1} completed successfully")
            
        except Exception as e:
            print(f"  ❌ Batch {batch_idx + 1} failed: {e}")
            return False
    
    print("✅ All batches completed successfully!")
    return True


def test_with_sentence_transformer():
    """Test with actual SentenceTransformer to see if that's the issue."""
    
    print("\n" + "=" * 50)
    print("SENTENCE TRANSFORMER TEST")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the same model used in training
        text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Freeze it
        for param in text_encoder.parameters():
            param.requires_grad = False
        
        model = MinimalModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Test texts
        texts = [
            "Python developer with 5 years experience",
            "Java developer with 3 years experience", 
            "Chef with culinary training",
            "Doctor with medical experience"
        ]
        
        print("Testing with SentenceTransformer embeddings...")
        
        for batch_idx in range(2):  # Test 2 batches
            print(f"Batch {batch_idx + 1}:")
            
            try:
                # Generate embeddings
                embeddings = text_encoder.encode(texts, convert_to_tensor=True)
                
                # Split into anchor, positive, negative
                anchor_embs = embeddings[:2]  # First 2
                positive_embs = embeddings[1:3]  # Middle 2
                negative_embs = embeddings[2:]  # Last 2
                
                # Forward pass
                anchor_proj = model(anchor_embs)
                positive_proj = model(positive_embs)
                negative_proj = model(negative_embs)
                
                # Simple contrastive loss
                pos_sim = torch.sum(anchor_proj * positive_proj, dim=1) / 0.1
                neg_sim = torch.sum(anchor_proj * negative_proj, dim=1) / 0.1
                
                # InfoNCE loss
                pos_exp = torch.exp(pos_sim)
                neg_exp = torch.exp(neg_sim)
                loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
                
                print(f"  Loss: {loss.item():.4f}")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"  ✅ Batch {batch_idx + 1} completed successfully")
                
            except Exception as e:
                print(f"  ❌ Batch {batch_idx + 1} failed: {e}")
                return False
        
        print("✅ SentenceTransformer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformer test failed: {e}")
        return False


def main():
    """Run all tests."""
    
    print("Testing gradient computation in isolation...")
    
    # Test 1: Minimal training loop
    minimal_success = test_minimal_training()
    
    # Test 2: With SentenceTransformer
    st_success = test_with_sentence_transformer()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    if minimal_success and st_success:
        print("✅ Both tests passed!")
        print("The gradient issue is likely in the CACL trainer implementation,")
        print("not in the basic PyTorch training loop or SentenceTransformer.")
    elif minimal_success:
        print("✅ Minimal test passed, ❌ SentenceTransformer test failed")
        print("The issue might be related to SentenceTransformer integration.")
    else:
        print("❌ Even minimal test failed")
        print("There might be a fundamental PyTorch environment issue.")


if __name__ == "__main__":
    main()