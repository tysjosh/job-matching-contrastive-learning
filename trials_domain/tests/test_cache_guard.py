"""Tests for the embedding-cache validity guard.

Motivated by an actual failure in this study: ``preencode_pool`` wrote the
projection head's 128-d output into a cache the training path fills with 768-d
text embeddings, and the mix only surfaced ~90 minutes later inside a
``torch.stack``. Five epochs had already run on wrong inputs.

The subtler case is the one that must not regress: a cache that is *uniformly*
the wrong dimension never mismatches, so a homogeneity-only check passes it and
training proceeds silently on wrong inputs. The corrupted artifact really was
uniformly 128-d.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trials_domain.label_budget_runner import assert_cache_valid, expected_text_dim


def _trainer(cache: dict, text_dim: int = 768, projection_dim: int = 128):
    encoder = types.SimpleNamespace(
        get_sentence_embedding_dimension=lambda: text_dim
    )
    return types.SimpleNamespace(
        text_encoder=encoder,
        embedding_cache=types.SimpleNamespace(cache=cache),
        config=types.SimpleNamespace(
            embedding_cache_path="embedding_cache/x.pt",
            projection_dim=projection_dim,
        ),
    )


def test_correct_text_embeddings_pass():
    t = _trainer({"a": torch.zeros(768), "b": torch.zeros(768)})
    report = assert_cache_valid(t)
    assert report["entries"] == 2
    assert report["expected_text_dim"] == 768


def test_mixed_dimensions_raise():
    t = _trainer({"a": torch.zeros(768), "b": torch.zeros(128)})
    with pytest.raises(RuntimeError, match="mixed dimensions"):
        assert_cache_valid(t)


def test_uniformly_projected_cache_raises():
    """The regression that matters: uniform 128-d must NOT pass.

    A homogeneity-only check accepts this, and the run then trains on the
    projection head's own stale output with no error anywhere.
    """
    t = _trainer({f"k{i}": torch.zeros(128) for i in range(1995)})
    with pytest.raises(RuntimeError) as excinfo:
        assert_cache_valid(t)
    message = str(excinfo.value)
    assert "128" in message and "768" in message
    # Should name the likely cause rather than just reporting a mismatch.
    assert "projection" in message.lower()


def test_uniformly_wrong_but_not_projection_dim_still_raises():
    t = _trainer({"a": torch.zeros(384)}, text_dim=768, projection_dim=128)
    with pytest.raises(RuntimeError, match="384"):
        assert_cache_valid(t)


def test_empty_cache_is_allowed():
    """A cold start has nothing to validate and must not be blocked."""
    assert assert_cache_valid(_trainer({}))["entries"] == 0


def test_passes_when_expected_dim_undiscoverable():
    """Without a discoverable encoder dim, fall back to the homogeneity check only."""
    t = _trainer({"a": torch.zeros(999), "b": torch.zeros(999)})
    t.text_encoder = types.SimpleNamespace()  # no dimension getter
    report = assert_cache_valid(t)
    assert report["expected_text_dim"] is None

    t.embedding_cache.cache = {"a": torch.zeros(999), "b": torch.zeros(128)}
    with pytest.raises(RuntimeError, match="mixed dimensions"):
        assert_cache_valid(t)


def test_expected_text_dim_prefers_encoder_then_attrs():
    t = _trainer({})
    assert expected_text_dim(t) == 768

    t.text_encoder = types.SimpleNamespace()
    t.text_encoder_dim = 384
    assert expected_text_dim(t) == 384

    t2 = types.SimpleNamespace(
        text_encoder=types.SimpleNamespace(),
        config=types.SimpleNamespace(),
    )
    assert expected_text_dim(t2) is None


def test_guard_catches_the_real_quarantined_artifact():
    """If the corrupted file is still around, confirm the guard rejects it."""
    path = Path("/tmp/corrupted_trials_cache.pt")
    if not path.exists():
        pytest.skip("quarantined artifact not present")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    store = payload.get("cache", payload)
    t = _trainer(store)
    with pytest.raises(RuntimeError):
        assert_cache_valid(t)
