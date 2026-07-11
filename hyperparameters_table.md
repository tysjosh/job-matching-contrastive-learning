# Important Hyperparameters

## Ontology Metric Computation

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Graph distance decay | $\beta$ | 0.7 | Exponential decay rate for skill similarity: $s(u,v) = \exp(-\beta \cdot \delta(u,v))$ |
| Maximum path length | - | 8 hops | Skills beyond 8 hops are considered disconnected |
| OT regularization | $\lambda$ | 0.4 | Entropic regularization for Sinkhorn algorithm |
| OT disconnected cost | - | 10.0 | Cost assigned to skill pairs with no path within 8 hops |
| Sinkhorn iterations | - | 200 | Maximum iterations for Sinkhorn convergence |
| Sinkhorn tolerance | - | 1e-6 | Convergence threshold for Sinkhorn algorithm |

## ESCO Enrichment (Skill Linking)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Fuzzy match threshold (skills) | 86 | RapidFuzz WRatio cutoff for skill string matching |
| Fuzzy match threshold (occupations) | 82 | RapidFuzz WRatio cutoff for occupation title matching |
| Semantic match threshold | 0.55 | Cosine similarity cutoff for embedding-based skill matching |
| Semantic embedding model | all-MiniLM-L6-v2 | Sentence-transformer model for semantic matching |
| Top-k semantic matches | 3 | Maximum number of ESCO candidates per skill string |

## Contrastive Learning

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Batch size | - | 64 | Number of resume-job pairs per batch |
| Learning rate | $\eta$ | 8.5e-5 | Adam optimizer learning rate |
| Temperature | $\tau$ | 0.07 | InfoNCE temperature scaling parameter |
| Number of epochs | - | 15 | Training epochs for phase 1 (pretraining) |
| Negatives per anchor | $K$ | 7 | Number of negative jobs per resume |
| Text encoder | - | all-mpnet-base-v2 | Sentence-transformer backbone (frozen) |
| Projection dimension | $d$ | 128 | Dimensionality of contrastive projection head |
| Projection dropout | - | 0.3 | Dropout rate in projection head |

## OSCAR Loss Weighting

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Ontology weight | $\alpha$ | 0.3 | Strength of ontology signal in loss weighting |
| OT distance scale | $\tau_{\text{OT}}$ | 10.0 | Normalization scale for OT distance: $\sigma_{\text{OT}} = \max(0, 1 - d_{\text{OT}}/\tau_{\text{OT}})$ |
| Use OT distance | - | True | Whether to include OT distance in ontology signal |
| ISCO loss weight | - | True | Whether to include ISCO proximity in ontology signal |
| Weight bounds | - | [0.5, 1.5] | Clamp range for sample weights |

## Negative Selection Strategy

| Parameter | Value | Description |
|-----------|-------|-------------|
| Use pathway negatives | True | Enable ontology-based negative selection |
| Use ISCO negatives | True | Enable ISCO hierarchy for negative selection |
| ISCO weight | 0.4 | Weight for ISCO distance in hybrid negative selection (0=skill-only, 1=ISCO-only) |
| Global negative pool size | 1000 | Size of candidate pool for negative sampling |
| Negative curriculum | False | Whether to use curriculum learning (easy→hard negatives) |
| Hard negative ratio | 0.33 | Proportion of hard negatives (high ontology similarity) |
| Medium negative ratio | 0.34 | Proportion of medium negatives |
| Easy negative ratio | 0.33 | Proportion of easy negatives (low ontology similarity) |

## Quality Tier Weights

| Tier | Criteria | Weight ($w_{\text{tier}}$) |
|------|----------|---------------------------|
| A | Both sides have skill URIs + occupation coverage | 1.0 |
| B | Both sides have skill URIs, no occupation | 0.9 |
| C | One side has skill URIs | 0.75 |
| D | Neither side has skill URIs (but raw skills exist) | 0.6 |
| F | Missing raw skills entirely | 0.5 |

## Ablation-Specific Parameters

### E3 Ablation (Negative Selection Strategies)
- **E3-A**: `use_pathway_negatives=True`, `ontology_weight=0.3` (full system)
- **E3-B/C**: `use_pathway_negatives=True`, `ontology_weight=0.0` (pathway negatives, no loss weighting)
- **E3-D**: `use_pathway_negatives=False`, `max_negatives_per_anchor=20` (random negatives, quantity over quality)
- **E3-E**: `use_pathway_negatives=True`, `use_isco_negatives=True`, `isco_weight=1.0`, `ontology_weight=0.0` (ISCO negatives only)
- **E3-F**: `use_pathway_negatives=True`, `use_isco_negatives=True`, `isco_weight=0.4`, `ontology_weight=0.0` (hybrid negatives)

### E4 Ablation (OSCAR Loss Weighting)
- **E4-InfoNCE**: `ontology_weight=0.0` (baseline, no ontology weighting)
- **E4-OSCAR-Skill**: `ontology_weight=0.3`, `use_ot_distance=True`, `isco_loss_weight=False` (skill-based only)
- **E4-OSCAR-ISCO**: `ontology_weight=0.3`, `use_ot_distance=False`, `isco_loss_weight=True` (ISCO hierarchy only)
- **E4-OSCAR-Hybrid**: `ontology_weight=0.3`, `use_ot_distance=True`, `isco_loss_weight=True` (all signals)

---

## Notes

1. **Frozen encoder**: The text encoder (all-mpnet-base-v2) is frozen during phase 1 pretraining. Only the projection head is trained.
2. **Embedding cache**: Embeddings are cached with size 10,000 to avoid redundant encoding of repeated resumes/jobs.
3. **View augmentation**: Multiple text views are generated per resume/job (max 5 each) for data augmentation.
4. **Validation frequency**: Model is validated every epoch on held-out validation set.
5. **Checkpoint frequency**: Model checkpoints saved every 500 batches.
