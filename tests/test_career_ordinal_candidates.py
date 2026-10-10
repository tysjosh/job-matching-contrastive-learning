"""Career ordinal supervision must include binary-negative partial matches."""

import torch

from contrastive_learning.batch_processor import BatchProcessor
from contrastive_learning.data_structures import TrainingConfig, TrainingSample
from contrastive_learning.loss_engine import ContrastiveLossEngine


def test_potential_fit_is_ordinal_candidate_not_infonce_positive_or_negative():
    config = TrainingConfig(use_pathway_negatives=False, pathway_weight=0.0,
                            loss_type='ordinal', max_negatives_per_anchor=2)
    processor = BatchProcessor(config)
    resume = {'resume_id': 'r1', 'text': 'resume'}
    good = {'job_id': 'good', 'title': 'Good job'}
    potential = {'job_id': 'potential', 'title': 'Partial job'}
    no_fit = {'job_id': 'no', 'title': 'Unrelated job'}
    batch = [
        TrainingSample(resume, good, 'positive', 'good',
                       {'resume_id': 'r1', 'original_label': 'good_fit'}),
        TrainingSample(resume, potential, 'negative', 'potential',
                       {'resume_id': 'r1', 'original_label': 'potential_fit'}),
        TrainingSample(resume, no_fit, 'negative', 'no',
                       {'resume_id': 'r1', 'original_label': 'no_fit'}),
    ]
    triplets = processor.process_batch(batch)
    assert len(triplets) == 1
    assert triplets[0].view_metadata['ordinal_candidates'] == [potential]
    assert potential not in triplets[0].negatives

    # With the same InfoNCE and level-0 terms, only the level-2 > level-1
    # margin can account for a change when lambda1 is enabled.
    triplet = triplets[0]
    engine = ContrastiveLossEngine(config)
    engine.set_epoch(5)
    embeddings = {
        engine._get_content_key(resume): torch.tensor([1.0, 0.0]),
        engine._get_content_key(good): torch.tensor([1.0, 0.0]),
        engine._get_content_key(potential): torch.tensor([1.0, 0.0]),
        engine._get_content_key(no_fit): torch.tensor([0.0, 1.0]),
    }
    with_margin = engine._compute_ordinal_loss(triplets, embeddings)
    engine.ordinal_lambda1 = 0.0
    without_margin = engine._compute_ordinal_loss(triplets, embeddings)
    assert with_margin > without_margin
