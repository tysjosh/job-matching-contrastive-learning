"""Inference-time GO/PPI text + GO fusion: scorer, phi library and honest search.

Nothing here trains a model. The 15 published text-only checkpoints are scored
read-only and only the way the text and GO scores are combined changes.

Modules:
    guard    -- the no-test-access guard for search mode.
    data     -- read-only port of ``evaluate_publication_fusion.prepare_split``.
    metrics  -- vectorised within-anchor metrics (checked against the published code).
    phi      -- GO similarity variants (simGIC/ancestor/exact/IC-overlap/Resnik, BP/MF/CC,
                is_a-only, pruned vs unpruned).
    fusion   -- linear z, per-anchor rank, RRF, gated and stacked fusion.
    search   -- anchor-grouped cross-fitted validation search.
    final    -- report step: per-anchor deltas, two-level bootstrap, shuffled-GO
                derangement, protein-disjoint CV sensitivity.
"""
