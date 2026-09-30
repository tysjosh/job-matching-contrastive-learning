"""Small correctness checks for the MeSH decomposition diagnostic."""

import pytest

from scripts.audit_mesh_decomposition import auc, exact_bma, jaccard


def test_rank_auc_handles_separation_and_ties():
    assert auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert auc([1.0, 1.0], [1.0, 1.0]) == 0.5
    assert auc([], [1.0]) is None


def test_exact_scores_keep_set_denominators_distinct():
    a, b = {"D1", "D2"}, {"D1"}
    assert jaccard(a, b) == pytest.approx(0.5)
    assert exact_bma(a, b) == pytest.approx(0.75)
