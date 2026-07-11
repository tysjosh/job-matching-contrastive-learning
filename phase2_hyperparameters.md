# Phase 2 Fine-Tuning Hyperparameters

Phase 2 fine-tunes a supervised classification head on top of the frozen Phase 1 contrastive encoder using labeled data (good_fit/potential_fit/no_fit).

## Training Configuration

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Batch size | - | 32 | Training batch size (reduced from Phase 1's 64) |
| Learning rate | $\eta$ | 5e-4 | Adam optimizer learning rate (higher than Phase 1's 8.5e-5) |
| Number of epochs | - | 10 | Training epochs (fewer than Phase 1's 15) |
| Weight decay | $\lambda$ | 0.001 | L2 regularization coefficient |
| Validation frequency | - | Every epoch | Model evaluated on validation set each epoch |

## Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| Text encoder | all-mpnet-base-v2 | Sentence-transformer backbone (frozen) |
| Freeze text encoder | True | Text encoder weights frozen from Phase 1 |
| Freeze contrastive layers | True | **Phase 1 projection head frozen** |
| Projection dimension | 128 | Dimensionality of contrastive embeddings |
| Projection dropout | 0.3 | Dropout rate in projection head |
| Classification dropout | 0.3 | Dropout rate in classification head |

## Loss Function

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Temperature | $\tau$ | 0.2 | Softmax temperature (higher than Phase 1's 0.07) |
| Positive class weight | $w_{\text{pos}}$ | 2.5 | Upweight positive class to handle imbalance |
| Ontology weight | $\alpha$ | 0.3 | OSCAR sample-level weighting strength |
| OT distance scale | $\tau_{\text{OT}}$ | 10.0 | Normalization for optimal transport distance |

## Negative Sampling

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max negatives per anchor | 20 | Number of negative jobs per resume (increased from Phase 1's 7) |
| Use pathway negatives | True | Ontology-guided negative selection |
| Global negative pool size | 1000 | Candidate pool for negative sampling |
| Negative sampling ratio | 0.7 | Proportion of hard negatives |

## Data Augmentation

| Parameter | Value | Description |
|-----------|-------|-------------|
| Use view augmentation | False | **Disabled** (was True in Phase 1) |
| Max resume views | 5 | Maximum augmented views per resume (unused) |
| Max job views | 5 | Maximum augmented views per job (unused) |

## Key Differences from Phase 1

### Architecture Changes
- **Freeze contrastive layers**: Phase 1 projection head is frozen; only the new classification head is trained
- **Classification head**: New 3-way classifier (good_fit/potential_fit/no_fit) added on top of frozen embeddings

### Training Changes
- **Higher learning rate** (5e-4 vs 8.5e-5): Faster convergence for classification head
- **Smaller batch size** (32 vs 64): More frequent gradient updates
- **Fewer epochs** (10 vs 15): Prevent overfitting on smaller labeled set
- **Weight decay** (0.001): L2 regularization to prevent overfitting

### Loss Changes
- **Higher temperature** (0.2 vs 0.07): Softer probability distribution for classification
- **Positive class weight** (2.5): Handle class imbalance (more no_fit samples than good_fit)
- **More negatives** (20 vs 7): Harder discrimination task for fine-tuning

### Data Changes
- **No view augmentation**: Use original data only (no paraphrasing/transformation)
- **Clear cache between epochs**: Free memory after each epoch

## Training Strategy

Phase 2 follows a **feature extraction** approach:

1. **Load Phase 1 checkpoint**: Initialize from best Phase 1 contrastive model
2. **Freeze encoder + projection**: Keep all Phase 1 weights fixed
3. **Add classification head**: New linear layer: $\mathbb{R}^{128} \rightarrow \mathbb{R}^3$
4. **Train classifier only**: Optimize classification head with supervised labels
5. **Class weighting**: Upweight positive class (good_fit) to handle imbalance

This approach leverages the rich contrastive representations learned in Phase 1 while adapting to the supervised classification task with minimal overfitting risk.

## Pretrained Model Path

```
pretrained_model_path: "results_infonce_ontology/phase1_pretraining/best_checkpoint.pt"
```

The Phase 2 model is initialized from the best Phase 1 checkpoint (selected by validation loss), ensuring that fine-tuning starts from high-quality contrastive embeddings.

## Validation Data

```
validation_path: "preprocess/data_splits_v6/validation.jsonl"
```

Phase 2 uses the same validation split as Phase 1 for consistent evaluation across training phases.

---

## Rationale for Hyperparameter Choices

### Why higher learning rate (5e-4)?
The classification head is randomly initialized, so it needs a higher learning rate to converge quickly. The frozen encoder prevents catastrophic forgetting of Phase 1 representations.

### Why smaller batch size (32)?
Smaller batches provide more frequent gradient updates, which is beneficial when training only a small classification head. This also reduces memory usage since we're computing embeddings for more negatives (20 vs 7).

### Why higher temperature (0.2)?
Phase 1 uses low temperature (0.07) to create sharp contrastive boundaries. Phase 2 uses higher temperature (0.2) to produce softer classification probabilities, which is more appropriate for ordinal labels where boundaries are less distinct.

### Why positive class weight (2.5)?
The dataset has class imbalance: more no_fit samples than good_fit. Upweighting the positive class (good_fit) by 2.5× ensures the model doesn't simply predict no_fit for everything.

### Why freeze contrastive layers?
Freezing Phase 1 weights prevents overfitting on the smaller labeled dataset and preserves the general-purpose contrastive representations. Only the task-specific classification head is adapted.

### Why no view augmentation?
Phase 1 uses augmentation to increase training data diversity for contrastive learning. Phase 2 has supervised labels, so augmentation is less critical and could introduce noise that hurts classification accuracy.
