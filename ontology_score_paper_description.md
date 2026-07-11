# Ontology-Aware Sample Weighting (OSCAR)

## Camera-Ready Description for Conference Paper

### Overview

We introduce **OSCAR** (Ontology-Structured Career-Aware Regularization), a sample-level weighting mechanism that modulates contrastive loss based on ontology-derived confidence signals. OSCAR addresses the challenge of noisy labels in career trajectory data by upweighting samples where ontology evidence confirms the label signal and downweighting samples with weak or missing ontology support.

### Mathematical Formulation

For each training triplet $(r, j^+, \{j^-_i\})$ consisting of resume $r$, positive job $j^+$, and negative jobs $\{j^-_i\}$, we compute an ontology-aware weight $w_{\text{ont}}$ that modulates the sample's contribution to the loss:

$$
\mathcal{L}_{\text{weighted}} = w_{\text{ont}} \cdot \mathcal{L}_{\text{base}}(r, j^+, \{j^-_i\})
$$

where $\mathcal{L}_{\text{base}}$ is the base contrastive loss (InfoNCE or ordinal).

### Weight Computation

The ontology weight $w_{\text{ont}} \in [0.5, 1.5]$ is computed as:

$$
w_{\text{ont}} = \max(0.5, \min(1.5, w_{\text{tier}} \cdot (1 + \alpha \cdot (2\sigma - 1))))
$$

where:
- $w_{\text{tier}}$ is a quality tier base weight
- $\alpha$ is the ontology weight hyperparameter (default: 0.3)
- $\sigma \in [0, 1]$ is the aggregated ontology signal

### Components

#### 1. Quality Tier Base Weight ($w_{\text{tier}}$)

Data quality tiers reflect metadata completeness and label confidence:

| Tier | Criteria | Weight |
|------|----------|--------|
| A | Complete metadata, high confidence | 1.0 |
| B | Good metadata, verified labels | 0.9 |
| C | Partial metadata | 0.75 |
| D | Minimal metadata | 0.6 |
| F | Missing critical fields | 0.5 |

#### 2. Ontology Signal ($\sigma$)

The ontology signal aggregates multiple evidence sources:

$$
\sigma = \frac{1}{N} \sum_{k=1}^{N} \sigma_k
$$

where $N$ is the number of available signals and $\sigma_k \in [0, 1]$ are individual signal components:

**a) Skill-Based Ontology Similarity ($\sigma_{\text{skill}}$)**

Measures semantic overlap between resume skill set $A$ and job skill set $B$ using ESCO knowledge graph $G$. For skill URIs $u, v$, let $\delta(u, v)$ be the shortest-path length on $G$. We convert graph distance to pairwise skill similarity via exponential decay:

$$
s_{\text{skill}}(u, v) = \exp(-\beta \cdot \delta(u, v))
$$

where $\beta = 0.7$ is the decay parameter. Directional coverage is computed as:

$$
s(A \rightarrow B) = \frac{1}{|A|} \sum_{a \in A} \max_{b \in B} s_{\text{skill}}(a, b)
$$

The symmetric ontology similarity is:

$$
\sigma_{\text{skill}} = s_{\text{ont}}(A, B) = \frac{1}{2} [s(A \rightarrow B) + s(B \rightarrow A)]
$$

This best-match averaging approach credits partial skill overlap through the ontology structure (e.g., "Python" and "Java" both connect through "programming languages").

**b) Optimal Transport Distance ($\sigma_{\text{OT}}$)**

Captures fine-grained skill distribution alignment:

$$
\sigma_{\text{OT}} = \max\left(0, 1 - \frac{d_{\text{OT}}(r, j^+)}{\tau}\right)
$$

where $d_{\text{OT}}$ is the Wasserstein-1 distance between skill distributions and $\tau$ is a normalization scale (default: 10.0). Lower OT distance indicates better alignment.

**c) ISCO Hierarchy Proximity ($\sigma_{\text{ISCO}}$)** *(optional)*

Leverages occupational taxonomy structure:

$$
\sigma_{\text{ISCO}} = \begin{cases}
1.0 & \text{if } \text{ISCO}_r = \text{ISCO}_{j^+} \text{ (4-digit match)} \\
0.8 & \text{if } \text{ISCO}_r[:3] = \text{ISCO}_{j^+}[:3] \text{ (unit group)} \\
0.6 & \text{if } \text{ISCO}_r[:2] = \text{ISCO}_{j^+}[:2] \text{ (minor group)} \\
0.3 & \text{if } \text{ISCO}_r[:1] = \text{ISCO}_{j^+}[:1] \text{ (major group)} \\
0.0 & \text{otherwise}
\end{cases}
$$

This 5-level hierarchy captures occupational relatedness at multiple granularities.

### Algorithm

```
Algorithm: OSCAR Weight Computation
Input: triplet metadata M, config parameters α, τ
Output: sample weight w_ont ∈ [0.5, 1.5]

1. Extract quality tier T from M
2. Set w_tier ← tier_weight_map[T]
3. Initialize σ ← 0, N ← 0

4. // Aggregate ontology signals
5. if ontology_similarity available in M:
6.     σ ← σ + M.ontology_similarity
7.     N ← N + 1

8. if use_ot_distance and ot_distance available in M:
9.     σ_ot ← max(0, 1 - M.ot_distance / τ)
10.    σ ← σ + σ_ot
11.    N ← N + 1

12. if isco_loss_weight and occupation URIs available:
13.    σ_isco ← isco_proximity(M.job_occ, M.resume_occ)
14.    σ ← σ + σ_isco
15.    N ← N + 1

16. // Compute final weight
17. if N > 0:
18.    σ ← σ / N  // average signal
19.    w_ont ← w_tier × (1 + α × (2σ - 1))
20. else:
21.    w_ont ← w_tier  // fallback to tier weight

22. return clamp(w_ont, 0.5, 1.5)
```

### Interpretation

The ontology weight acts as a **confidence modulator**:

- **High ontology signal** ($\sigma \approx 1$): Skills and occupations strongly align → weight increases toward 1.5 → sample contributes more to learning
- **Low ontology signal** ($\sigma \approx 0$): Weak or contradictory ontology evidence → weight decreases toward 0.5 → sample contribution is dampened
- **Missing ontology data**: Falls back to quality tier weight → graceful degradation

This mechanism enables the model to focus on high-confidence career transitions while remaining robust to label noise and incomplete metadata.

### Ablation Variants

We evaluate three OSCAR configurations:

1. **OSCAR-Skill**: Uses $\sigma_{\text{skill}}$ and $\sigma_{\text{OT}}$ (skill-based ontology)
2. **OSCAR-ISCO**: Uses $\sigma_{\text{ISCO}}$ only (occupational hierarchy)
3. **OSCAR-Hybrid**: Combines all three signals ($\sigma_{\text{skill}}$, $\sigma_{\text{OT}}$, $\sigma_{\text{ISCO}}$)

### Hyperparameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Ontology weight | $\alpha$ | 0.3 | [0, 1] | Strength of ontology adjustment |
| OT distance scale | $\tau$ | 10.0 | [5, 20] | Normalization for OT distance |
| Weight bounds | - | [0.5, 1.5] | - | Prevents extreme deweighting |

### Implementation Notes

- Ontology scores are **precomputed** during data preprocessing using ESCO knowledge graphs
- Weight computation is **differentiable** and integrated into the loss backward pass
- Computational overhead is **negligible** (simple arithmetic on cached metadata)
- Compatible with any base contrastive loss (InfoNCE, ordinal, Wasserstein)

---

## Citation Format (if needed)

```
We introduce OSCAR (Ontology-Structured Career-Aware Regularization), 
a sample-level weighting mechanism that modulates contrastive loss based 
on ontology-derived confidence signals from the ESCO knowledge graph. 
OSCAR computes a weight w_ont ∈ [0.5, 1.5] for each training sample by 
aggregating skill-based similarity, optimal transport distance, and ISCO 
hierarchy proximity, enabling the model to focus on high-confidence career 
transitions while remaining robust to label noise.
```

---

## Key Advantages

1. **Theoretically grounded**: Leverages established ontology structures (ESCO, ISCO)
2. **Empirically effective**: Ablation studies show consistent improvements (see E4 experiments)
3. **Computationally efficient**: No additional forward passes, only metadata lookup
4. **Robust to missing data**: Graceful degradation when ontology signals unavailable
5. **Interpretable**: Clear semantic meaning for each component
6. **Modular**: Can be combined with any contrastive learning framework
