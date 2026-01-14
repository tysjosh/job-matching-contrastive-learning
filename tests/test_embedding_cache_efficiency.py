#!/usr/bin/env python3
"""
Test script to demonstrate the efficiency improvements of the embedding cache.

This script compares the old vs new embedding generation approaches and shows
the performance benefits of caching.
"""

import time
import json
import torch
from typing import List, Dict, Any
from pathlib import Path

# Import the new cache system
from contrastive_learning.embedding_cache import EmbeddingCache, BatchEfficientEncoder
from contrastive_learning.data_structures import TrainingConfig


def create_mock_content(num_resumes: int = 100, num_jobs: int = 50) -> List[Dict[str, Any]]:
    """Create mock resume and job content for testing."""
    content_items = []

    # Create mock resumes
    for i in range(num_resumes):
        resume = {
            'experience': f'Experienced software engineer with {i+1} years in Python development. '
            f'Worked on web applications, APIs, and data processing systems.',
            'role': 'software engineer',
            'experience_level': 'senior' if i > 50 else 'mid',
            'skills': [
                {'name': 'Python', 'level': 'senior',
                    'category': 'Programming Languages'},
                {'name': 'Django', 'level': 'mid', 'category': 'Frameworks'},
                {'name': 'PostgreSQL', 'level': 'mid', 'category': 'Databases'}
            ],
            'keywords': ['python', 'web', 'api', 'database', 'software']
        }
        content_items.append((resume, 'resume'))

    # Create mock jobs
    for i in range(num_jobs):
        job = {
            'title': f'Senior Software Engineer {i+1}',
            'description': {
                'original': f'We are looking for a senior software engineer with expertise in Python '
                f'and web development. Job ID: {i+1}'
            },
            'skills': [
                {'name': 'Python', 'level': 'expert'},
                {'name': 'Django', 'level': 'advanced'},
                {'name': 'REST APIs', 'level': 'advanced'}
            ]
        }
        content_items.append((job, 'job'))

    return content_items


def simulate_batch_processing(content_items: List, batch_size: int = 32, num_batches: int = 10):
    """
    Simulate the batch processing that happens during training.
    This creates overlapping content across batches (realistic scenario).
    """
    import random

    batches = []
    for batch_idx in range(num_batches):
        # Create a batch with some overlap from previous batches
        batch = []

        # Add some random content
        random_items = random.sample(content_items, batch_size // 2)
        batch.extend(random_items)

        # Add some content that appeared in previous batches (simulate overlap)
        if batch_idx > 0:
            overlap_items = random.sample(
                content_items[:batch_size], batch_size // 2)
            batch.extend(overlap_items)
        else:
            # First batch, just add more random items
            more_items = random.sample(content_items, batch_size // 2)
            batch.extend(more_items)

        batches.append(batch[:batch_size])  # Ensure exact batch size

    return batches


def test_old_approach(batches: List[List], mock_encoder):
    """Test the old approach without caching."""
    print("Testing OLD approach (no caching)...")

    start_time = time.time()
    total_encodings = 0

    for batch_idx, batch in enumerate(batches):
        # Simulate the old approach: encode everything every time
        batch_start = time.time()

        # Old approach: no deduplication across batches
        embeddings = mock_encoder.encode_batch(batch)
        total_encodings += len(batch)

        batch_time = time.time() - batch_start
        if batch_idx % 5 == 0:
            print(
                f"  Batch {batch_idx}: {len(batch)} items, {batch_time:.3f}s")

    total_time = time.time() - start_time

    print(f"OLD approach results:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Total encodings: {total_encodings}")
    print(f"  Avg time per encoding: {total_time/total_encodings:.4f}s")
    print(f"  Encodings per second: {total_encodings/total_time:.1f}")

    return {
        'total_time': total_time,
        'total_encodings': total_encodings,
        'avg_time_per_encoding': total_time / total_encodings
    }


def test_new_approach(batches: List[List], mock_encoder):
    """Test the new approach with caching."""
    print("\nTesting NEW approach (with caching)...")

    # Initialize cache
    cache = EmbeddingCache(max_cache_size=1000, enable_stats=True)

    start_time = time.time()
    total_requests = 0

    for batch_idx, batch in enumerate(batches):
        batch_start = time.time()

        # New approach: use cache
        embeddings = cache.get_embeddings_batch(
            batch, mock_encoder.encode_batch)
        total_requests += len(batch)

        batch_time = time.time() - batch_start
        if batch_idx % 5 == 0:
            cache_stats = cache.get_cache_stats()
            print(f"  Batch {batch_idx}: {len(batch)} items, {batch_time:.3f}s, "
                  f"hit rate: {cache_stats['hit_rate']:.1%}")

    total_time = time.time() - start_time
    cache_stats = cache.get_cache_stats()

    print(f"NEW approach results:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Total requests: {total_requests}")
    print(f"  Actual encodings: {cache_stats['total_encodings']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Time saved: {cache_stats['time_saved_seconds']:.3f}s")
    print(f"  Cache size: {cache_stats['cache_size']}")
    print(f"  Memory usage: {cache_stats['memory_usage_mb']:.1f}MB")

    return {
        'total_time': total_time,
        'total_requests': total_requests,
        'actual_encodings': cache_stats['total_encodings'],
        'cache_hit_rate': cache_stats['hit_rate'],
        'time_saved': cache_stats['time_saved_seconds']
    }


class MockEncoder:
    """Mock encoder that simulates the time cost of encoding."""

    def __init__(self, encoding_time_per_item: float = 0.01):
        """
        Args:
            encoding_time_per_item: Simulated time to encode each item (seconds)
        """
        self.encoding_time_per_item = encoding_time_per_item
        self.device = torch.device('cpu')

    def encode_batch(self, content_items: List) -> List[torch.Tensor]:
        """Mock batch encoding with simulated delay."""
        # Simulate encoding time
        time.sleep(len(content_items) * self.encoding_time_per_item)

        # Return mock embeddings
        embeddings = []
        for _ in content_items:
            # Create a random 256-dimensional embedding
            embedding = torch.randn(256, device=self.device)
            embeddings.append(embedding)

        return embeddings


def main():
    """Run the efficiency comparison test."""
    print("Embedding Cache Efficiency Test")
    print("=" * 50)

    # Test parameters
    num_resumes = 100
    num_jobs = 50
    batch_size = 32
    num_batches = 20
    # 5ms per encoding (realistic for SentenceTransformer)
    encoding_time = 0.005

    print(f"Test setup:")
    print(f"  Resumes: {num_resumes}")
    print(f"  Jobs: {num_jobs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Simulated encoding time: {encoding_time*1000:.1f}ms per item")
    print()

    # Create test data
    print("Creating test data...")
    content_items = create_mock_content(num_resumes, num_jobs)
    batches = simulate_batch_processing(content_items, batch_size, num_batches)

    # Create mock encoder
    mock_encoder = MockEncoder(encoding_time)

    # Test old approach
    old_results = test_old_approach(batches, mock_encoder)

    # Test new approach
    new_results = test_new_approach(batches, mock_encoder)

    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)

    speedup = old_results['total_time'] / new_results['total_time']
    encoding_reduction = 1 - \
        (new_results['actual_encodings'] / old_results['total_encodings'])

    print(f"Speedup: {speedup:.2f}x faster")
    print(f"Encoding reduction: {encoding_reduction:.1%}")
    print(
        f"Time saved: {old_results['total_time'] - new_results['total_time']:.3f}s")
    print()

    print("Detailed comparison:")
    print(
        f"  Old approach: {old_results['total_encodings']} encodings in {old_results['total_time']:.3f}s")
    print(
        f"  New approach: {new_results['actual_encodings']} encodings in {new_results['total_time']:.3f}s")
    print(f"  Cache hit rate: {new_results['cache_hit_rate']:.1%}")

    # Extrapolate to real training
    print("\n" + "=" * 50)
    print("REAL TRAINING EXTRAPOLATION")
    print("=" * 50)

    # Real training parameters (based on your dataset)
    real_samples = 8572
    real_epochs = 10
    real_batch_size = 32
    real_batches_per_epoch = real_samples // real_batch_size
    real_total_batches = real_batches_per_epoch * real_epochs

    print(f"Your dataset: {real_samples} samples")
    print(f"Training: {real_epochs} epochs, batch size {real_batch_size}")
    print(f"Total batches: {real_total_batches}")

    # Estimate time savings
    old_time_per_batch = old_results['total_time'] / len(batches)
    new_time_per_batch = new_results['total_time'] / len(batches)

    estimated_old_time = old_time_per_batch * real_total_batches
    estimated_new_time = new_time_per_batch * real_total_batches
    estimated_savings = estimated_old_time - estimated_new_time

    print(f"\nEstimated training time:")
    print(f"  Without cache: {estimated_old_time/60:.1f} minutes")
    print(f"  With cache: {estimated_new_time/60:.1f} minutes")
    print(
        f"  Time saved: {estimated_savings/60:.1f} minutes ({estimated_savings/3600:.1f} hours)")
    print(f"  Speedup: {estimated_old_time/estimated_new_time:.1f}x")


if __name__ == "__main__":
    main()
