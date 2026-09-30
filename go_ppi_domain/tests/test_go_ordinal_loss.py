"""The GO loss switch and evidence-grade ranking contract."""

import json
from pathlib import Path

import pytest
import torch

from contrastive_learning.data_structures import ContrastiveTriplet, TrainingConfig
from contrastive_learning.loss_engine import ContrastiveLossEngine
from contrastive_learning.trainer import ContrastiveLearningTrainer
from scripts.run_go_decomposition import arm_config, materialize, config_dir


def setup_loss(loss_type="go_ordinal", weak_grade=1, **overrides):
    cfg = TrainingConfig(domain_adapter="go_ppi", loss_type=loss_type,
                         ontology_weight=0.0, **overrides)
    engine = ContrastiveLossEngine(cfg)
    slots = [
        {"encoder_view": "anchor"},
        {"encoder_view": "high"},
        {"encoder_view": "weak", "grade": weak_grade},
        {"encoder_view": "unobserved", "grade": 0},
    ]
    vectors = ([1.0, 0.0], [0.8, 0.6], [0.9, 0.4358899], [0.0, 1.0])
    embeddings = {
        engine._get_content_key(slot): torch.tensor(vector, requires_grad=True)
        for slot, vector in zip(slots, vectors)
    }
    triplet = ContrastiveTriplet(slots[0], slots[1], slots[2:], [5.0, 5.0],
                                 {"positive_original_label": "good_fit"})
    return engine, triplet, embeddings


def test_go_ordinal_penalizes_high_below_weak_and_backpropagates():
    ordinal, triplet, embeddings = setup_loss()
    baseline, _, _ = setup_loss("infonce")
    go_loss = ordinal.compute_loss([triplet], embeddings)
    base_loss = baseline.compute_loss([triplet], embeddings)
    assert go_loss.item() > base_loss.item()
    go_loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all()
               for v in embeddings.values())


def test_go_ordinal_zero_weights_is_exactly_infonce():
    ordinal, triplet, embeddings = setup_loss(
        go_ordinal_lambda_high_weak=0.0,
        go_ordinal_lambda_weak_unobserved=0.0)
    baseline, _, _ = setup_loss("infonce")
    assert torch.equal(ordinal.compute_loss([triplet], embeddings),
                       baseline.compute_loss([triplet], embeddings))


def test_weak_unobserved_term_is_optional_and_grade_sensitive():
    with_term, triplet, embeddings = setup_loss(
        go_ordinal_lambda_high_weak=0.0)
    without_term, _, _ = setup_loss(
        go_ordinal_lambda_high_weak=0.0,
        go_ordinal_lambda_weak_unobserved=0.0)
    assert with_term.compute_loss([triplet], embeddings) > without_term.compute_loss(
        [triplet], embeddings)
    triplet.negatives[0]["grade"] = None
    with pytest.raises(ValueError, match="requires grade"):
        with_term.compute_loss([triplet], embeddings)


def test_validation_dummy_has_no_ordinal_rank():
    engine, triplet, embeddings = setup_loss()
    triplet.negatives[0]["job_id"] = "dummy_negative"
    # The shared batch processor's placeholder still contributes to InfoNCE,
    # but cannot be assigned GO evidence grade 0 or 1.
    triplet.negatives[0]["grade"] = None
    loss = engine.compute_loss([triplet], embeddings)
    assert torch.isfinite(loss)


def test_go_ordinal_requires_go_domain_and_valid_parameters():
    with pytest.raises(ValueError, match="go_ppi"):
        ContrastiveLossEngine(TrainingConfig(loss_type="go_ordinal"))
    with pytest.raises(ValueError, match="finite and positive"):
        ContrastiveLossEngine(TrainingConfig(
            domain_adapter="go_ppi", loss_type="go_ordinal",
            go_ordinal_rank_temperature=0.0))


def test_runner_materializes_switchable_matched_controls():
    materialize("go_ordinal")
    arm = json.loads((config_dir("go_ordinal") / "go_A_simgic.json").read_text())
    control = json.loads((config_dir("go_ordinal") / "randwin.json").read_text())
    assert arm["loss_type"] == control["loss_type"] == "go_ordinal"
    assert arm["go_ppi_go_tiered_negatives"] is True
    assert control["go_ppi_tier_sampling"] == "random_window"
    assert arm_config("A", "simgic", "infonce")["loss_type"] == "infonce"


def test_trainer_seeds_before_encoder_and_projection_initialization(
        monkeypatch, tmp_path):
    seed = 3917

    def check_seed_before_encoder(_model_name):
        assert torch.initial_seed() == seed
        raise RuntimeError("seed checked before model construction")

    monkeypatch.setattr("contrastive_learning.trainer.SentenceTransformer",
                        check_seed_before_encoder)
    cfg = TrainingConfig(training_seed=seed, domain_adapter="go_ppi",
                         use_pathway_negatives=False, pathway_weight=0.0)
    with pytest.raises(RuntimeError, match="seed checked"):
        ContrastiveLearningTrainer(cfg, output_dir=str(tmp_path / "seed_order"))
