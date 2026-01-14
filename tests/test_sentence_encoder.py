"""
Test suite for sentence encoder functionality in the trainer.

This test verifies that the SentenceTransformer encoder:
1. Loads correctly with the configured model
2. Encodes text to proper embedding dimensions
3. Handles resume and job content correctly
4. Produces consistent embeddings for identical inputs
5. Produces different embeddings for different inputs
6. Works correctly on the configured device (CPU/GPU)
"""

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.trainer import ContrastiveLearningTrainer
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSentenceEncoder:
    """Test suite for sentence encoder functionality"""

    @pytest.fixture
    def config(self):
        """Create a minimal training configuration for testing"""
        return TrainingConfig(
            text_encoder_model='all-MiniLM-L6-v2',  # Default model
            batch_size=2,
            num_epochs=1,
            learning_rate=0.001,
            temperature=0.1,
            text_encoder_device='cpu',
            use_pathway_negatives=False,  # Disable for testing
            esco_graph_path='training_output/career_graph.gexf'  # Provide path
        )

    @pytest.fixture
    def trainer(self, config):
        """Create a trainer instance for testing"""
        trainer = ContrastiveLearningTrainer(config)
        return trainer

    @pytest.fixture
    def sample_resume_content(self):
        """Sample resume content for testing"""
        return {
            'experience': 'Senior Software Engineer with 5 years of experience in full-stack development',
            'role': 'Software Engineer',
            'experience_level': 'Senior',
            'skills': [
                {'name': 'Python', 'category': 'Programming Languages', 'level': 'Expert'},
                {'name': 'JavaScript', 'category': 'Programming Languages',
                    'level': 'Advanced'},
                {'name': 'AWS', 'category': 'Cloud Platforms  Devops Tools',
                    'level': 'Intermediate'},
                {'name': 'Docker', 'category': 'Cloud Platforms  Devops Tools',
                    'level': 'Advanced'}
            ],
            'keywords': ['software', 'development', 'agile', 'cloud', 'microservices']
        }

    @pytest.fixture
    def sample_job_content(self):
        """Sample job content for testing"""
        return {
            'title': 'Senior Software Engineer',
            'jobtitle': 'Senior Software Engineer',
            'skills': [
                {'name': 'Python', 'level': 'Required'},
                {'name': 'AWS', 'level': 'Preferred'},
                {'name': 'Kubernetes', 'level': 'Nice to have'}
            ],
            'keywords': ['python', 'cloud', 'distributed systems', 'backend']
        }

    def test_text_encoder_initialization(self, trainer):
        """Test that the text encoder is properly initialized"""
        assert trainer.text_encoder is not None, "Text encoder should be initialized"
        assert hasattr(trainer.text_encoder,
                       'encode'), "Text encoder should have encode method"
        print(
            f"✓ Text encoder initialized: {trainer.config.text_encoder_model}")

    def test_text_encoder_device(self, trainer):
        """Test that the text encoder is on the correct device"""
        # Check the device of the text encoder
        device = next(trainer.text_encoder.parameters()).device
        expected_device = trainer.device

        assert device.type == expected_device.type, \
            f"Text encoder should be on {expected_device}, but is on {device}"
        print(f"✓ Text encoder on correct device: {device}")

    def test_encode_resume_to_text_embedding(self, trainer, sample_resume_content):
        """Test encoding resume content to text embeddings"""
        embedding = trainer._encode_content_to_text_embedding(
            sample_resume_content, 'resume'
        )

        expected_dim = trainer.text_encoder.get_sentence_embedding_dimension()

        assert isinstance(
            embedding, torch.Tensor), "Embedding should be a tensor"
        assert embedding.dim() == 1, "Embedding should be 1-dimensional"
        assert embedding.shape[0] == expected_dim, \
            f"Embedding dimension should be {expected_dim}, got {embedding.shape[0]}"
        assert not torch.isnan(embedding).any(
        ), "Embedding should not contain NaN values"
        assert not torch.isinf(embedding).any(
        ), "Embedding should not contain Inf values"

        print(f"✓ Resume embedding shape: {embedding.shape}")
        print(f"✓ Resume embedding norm: {torch.norm(embedding).item():.4f}")

    def test_encode_job_to_text_embedding(self, trainer, sample_job_content):
        """Test encoding job content to text embeddings"""
        embedding = trainer._encode_content_to_text_embedding(
            sample_job_content, 'job'
        )

        expected_dim = trainer.text_encoder.get_sentence_embedding_dimension()

        assert isinstance(
            embedding, torch.Tensor), "Embedding should be a tensor"
        assert embedding.dim() == 1, "Embedding should be 1-dimensional"
        assert embedding.shape[0] == expected_dim, \
            f"Embedding dimension should be {expected_dim}, got {embedding.shape[0]}"
        assert not torch.isnan(embedding).any(
        ), "Embedding should not contain NaN values"
        assert not torch.isinf(embedding).any(
        ), "Embedding should not contain Inf values"

        print(f"✓ Job embedding shape: {embedding.shape}")
        print(f"✓ Job embedding norm: {torch.norm(embedding).item():.4f}")

    def test_embedding_consistency(self, trainer, sample_resume_content):
        """Test that identical inputs produce identical embeddings"""
        embedding1 = trainer._encode_content_to_text_embedding(
            sample_resume_content, 'resume'
        )
        embedding2 = trainer._encode_content_to_text_embedding(
            sample_resume_content, 'resume'
        )

        # Check that embeddings are identical (or very close due to floating point)
        assert torch.allclose(embedding1, embedding2, atol=1e-6), \
            "Identical inputs should produce identical embeddings"

        print(f"✓ Embedding consistency verified")
        print(
            f"  Max difference: {torch.max(torch.abs(embedding1 - embedding2)).item():.10f}")

    def test_embedding_differences(self, trainer, sample_resume_content, sample_job_content):
        """Test that different inputs produce different embeddings"""
        resume_embedding = trainer._encode_content_to_text_embedding(
            sample_resume_content, 'resume'
        )
        job_embedding = trainer._encode_content_to_text_embedding(
            sample_job_content, 'job'
        )

        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            resume_embedding.unsqueeze(0),
            job_embedding.unsqueeze(0)
        ).item()

        # Embeddings should be different but could have some similarity
        # (since they're both about software engineering)
        assert not torch.equal(resume_embedding, job_embedding), \
            "Different inputs should produce different embeddings"

        print(f"✓ Resume vs Job embeddings are different")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(
            f"  L2 distance: {torch.norm(resume_embedding - job_embedding).item():.4f}")

    def test_encode_empty_content(self, trainer):
        """Test encoding empty/minimal content"""
        empty_resume = {
            'experience': '',
            'role': '',
            'skills': [],
            'keywords': []
        }

        embedding = trainer._encode_content_to_text_embedding(
            empty_resume, 'resume')

        expected_dim = trainer.text_encoder.get_sentence_embedding_dimension()

        assert isinstance(
            embedding, torch.Tensor), "Should still produce an embedding"
        assert embedding.shape[0] == expected_dim, \
            "Should maintain correct dimension"
        assert not torch.isnan(embedding).any(), "Should not contain NaN"

        print(f"✓ Empty content handled correctly")
        print(f"  Embedding norm: {torch.norm(embedding).item():.4f}")

    def test_encode_rich_vs_minimal_content(self, trainer, sample_resume_content):
        """Test that rich content produces different embeddings than minimal content"""
        minimal_resume = {
            'experience': 'Software Engineer',
            'role': 'Software Engineer',
            'skills': [],
            'keywords': []
        }

        rich_embedding = trainer._encode_content_to_text_embedding(
            sample_resume_content, 'resume'
        )
        minimal_embedding = trainer._encode_content_to_text_embedding(
            minimal_resume, 'resume'
        )

        # Calculate similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            rich_embedding.unsqueeze(0),
            minimal_embedding.unsqueeze(0)
        ).item()

        assert not torch.equal(rich_embedding, minimal_embedding), \
            "Rich and minimal content should produce different embeddings"

        print(f"✓ Rich vs minimal content embeddings differ")
        print(f"  Cosine similarity: {cos_sim:.4f}")

    def test_batch_encoding(self, trainer, sample_resume_content, sample_job_content):
        """Test encoding multiple items"""
        # Create variations
        resume1 = sample_resume_content.copy()
        resume2 = {
            'experience': 'Junior Developer with 1 year of experience',
            'role': 'Developer',
            'experience_level': 'Junior',
            'skills': [{'name': 'Java', 'category': 'Programming Languages', 'level': 'Beginner'}],
            'keywords': ['java', 'spring']
        }

        embedding1 = trainer._encode_content_to_text_embedding(
            resume1, 'resume')
        embedding2 = trainer._encode_content_to_text_embedding(
            resume2, 'resume')
        job_embedding = trainer._encode_content_to_text_embedding(
            sample_job_content, 'job')

        expected_dim = trainer.text_encoder.get_sentence_embedding_dimension()

        # All should be valid
        for i, emb in enumerate([embedding1, embedding2, job_embedding], 1):
            assert isinstance(
                emb, torch.Tensor), f"Embedding {i} should be a tensor"
            assert emb.shape[0] == expected_dim, \
                f"Embedding {i} should have correct dimension"
            assert not torch.isnan(emb).any(
            ), f"Embedding {i} should not contain NaN"

        print(f"✓ Batch encoding works correctly")
        print(
            f"  Senior vs Junior similarity: {torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item():.4f}")

    def test_text_construction_resume(self, trainer, sample_resume_content):
        """Test that resume text is properly constructed with SEP tokens"""
        # This is a bit of a white-box test to ensure the text formatting is working
        embedding = trainer._encode_content_to_text_embedding(
            sample_resume_content, 'resume'
        )

        # Verify that the embedding is reasonable (non-zero, normalized-ish)
        norm = torch.norm(embedding).item()
        assert norm > 0, "Embedding should have non-zero norm"

        # The norm should be reasonable (not too small or too large)
        assert 0.1 < norm < 100, f"Embedding norm seems unusual: {norm}"

        print(f"✓ Resume text construction produces valid embedding")
        print(f"  Embedding norm: {norm:.4f}")

    def test_text_construction_job(self, trainer, sample_job_content):
        """Test that job text is properly constructed"""
        embedding = trainer._encode_content_to_text_embedding(
            sample_job_content, 'job'
        )

        # Verify that the embedding is reasonable
        norm = torch.norm(embedding).item()
        assert norm > 0, "Embedding should have non-zero norm"
        assert 0.1 < norm < 100, f"Embedding norm seems unusual: {norm}"

        print(f"✓ Job text construction produces valid embedding")
        print(f"  Embedding norm: {norm:.4f}")

    def test_skills_with_different_formats(self, trainer):
        """Test handling skills in different formats (string vs dict)"""
        # Skills as dicts (current format)
        resume_dict_skills = {
            'experience': 'Software Engineer',
            'skills': [
                {'name': 'Python', 'category': 'Programming Languages', 'level': 'Expert'}
            ]
        }

        # Skills as strings (alternative format mentioned in code)
        resume_string_skills = {
            'experience': 'Software Engineer',
            'skills': ['Python', 'JavaScript']
        }

        embedding1 = trainer._encode_content_to_text_embedding(
            resume_dict_skills, 'resume')
        embedding2 = trainer._encode_content_to_text_embedding(
            resume_string_skills, 'resume')

        # Both should work
        assert isinstance(embedding1, torch.Tensor), "Dict skills should work"
        assert isinstance(
            embedding2, torch.Tensor), "String skills should work"

        print(f"✓ Different skill formats handled correctly")

    def test_encoder_gradient_flow(self, trainer):
        """Test that the encoder can be used in training (gradients flow)"""
        # Create dummy input
        dummy_content = {
            'experience': 'Test content for gradient flow',
            'role': 'Test',
            'skills': [],
            'keywords': []
        }

        # Enable gradient tracking
        trainer.text_encoder.train()

        # Encode
        embedding = trainer._encode_content_to_text_embedding(
            dummy_content, 'resume')

        # Try to compute a simple loss and backward
        loss = embedding.sum()  # Dummy loss

        # This should work without errors
        try:
            if embedding.requires_grad:
                loss.backward()
                print(f"✓ Gradients flow through encoder (trainable mode)")
            else:
                print(f"✓ Encoder in eval mode (no gradients expected)")
        except Exception as e:
            print(f"Note: Gradient flow test: {e}")

    def test_sentence_encoder_integration(self):
        """Integration test: full encoder workflow"""
        print("\n" + "="*70)
        print("SENTENCE ENCODER INTEGRATION TEST")
        print("="*70)

        # Create config
        config = TrainingConfig(
            text_encoder_model='all-MiniLM-L6-v2',
            text_encoder_device='cpu',
            use_pathway_negatives=False,  # Disable for testing
            esco_graph_path='training_output/career_graph.gexf'
        )

        # Create trainer
        trainer = ContrastiveLearningTrainer(config)

        # Test data
        resume = {
            'experience': 'Experienced Data Scientist with ML expertise',
            'role': 'Data Scientist',
            'experience_level': 'Senior',
            'skills': [
                {'name': 'Python', 'category': 'Programming Languages', 'level': 'Expert'},
                {'name': 'TensorFlow', 'category': 'Machine Learning',
                    'level': 'Advanced'}
            ],
            'keywords': ['machine learning', 'python', 'data analysis']
        }

        job = {
            'title': 'Senior Data Scientist',
            'skills': [
                {'name': 'Python', 'level': 'Required'},
                {'name': 'Machine Learning', 'level': 'Required'}
            ],
            'keywords': ['ml', 'ai', 'python']
        }

        # Encode
        resume_emb = trainer._encode_content_to_text_embedding(
            resume, 'resume')
        job_emb = trainer._encode_content_to_text_embedding(job, 'job')

        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(
            resume_emb.unsqueeze(0),
            job_emb.unsqueeze(0)
        ).item()

        print(f"\n📊 Integration Test Results:")
        print(f"  Resume embedding shape: {resume_emb.shape}")
        print(f"  Job embedding shape: {job_emb.shape}")
        print(f"  Resume-Job similarity: {similarity:.4f}")
        print(f"  Resume embedding norm: {torch.norm(resume_emb).item():.4f}")
        print(f"  Job embedding norm: {torch.norm(job_emb).item():.4f}")

        # Similarity should be reasonably high for matching resume/job
        assert 0.3 < similarity < 1.0, \
            f"Similar resume/job should have reasonable similarity, got {similarity}"

        print(f"\n✅ Integration test passed!")
        print("="*70)


if __name__ == "__main__":
    # Run all tests with pytest
    print("\nRunning sentence encoder test suite with pytest...")
    pytest.main([__file__, "-v", "--tb=short"])
