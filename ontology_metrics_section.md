# Ontology Metrics Computed from ESCO Graph

We compute three ontology-derived metrics from the ESCO knowledge graph to quantify semantic alignment between resume and job skill sets. These metrics are precomputed during data preprocessing and stored as metadata for downstream use in sample weighting and negative selection.

## 1. Skill-Based Ontology Similarity

Measures semantic overlap between resume skill set $A$ and job skill set $B$ using the ESCO knowledge graph $G$. For skill URIs $u, v$, let $\delta(u, v)$ be the shortest-path length on $G$. We convert graph distance to pairwise skill similarity via exponential decay:

$$
s_{\text{skill}}(u, v) = \exp(-\beta \cdot \delta(u, v))
$$

where $\beta = 0.7$ is the decay parameter and paths longer than 8 hops are considered disconnected ($s_{\text{skill}} = 0$). Directional coverage is computed as:

$$
s(A \rightarrow B) = \frac{1}{|A|} \sum_{a \in A} \max_{b \in B} s_{\text{skill}}(a, b)
$$

The symmetric ontology similarity is:

$$
s_{\text{ont}}(A, B) = \frac{1}{2} [s(A \rightarrow B) + s(B \rightarrow A)]
$$

This best-match averaging approach credits partial skill overlap through the ontology structure. For example, "Python" and "Java" receive non-zero similarity by connecting through intermediate nodes like "programming languages" in the ESCO graph, whereas simple set-based metrics (e.g., Jaccard) would assign zero similarity.

**Range**: $s_{\text{ont}} \in [0, 1]$, where 1 indicates perfect alignment and 0 indicates no semantic overlap within 8 hops.

## 2. Optimal Transport Distance

Captures fine-grained skill distribution alignment using the Wasserstein-1 distance. We treat resume skill set $A$ and job skill set $B$ as uniform distributions and compute the optimal transport cost using the Sinkhorn algorithm with graph distance as the ground metric:

$$
d_{\text{OT}}(A, B) = \text{Sinkhorn}(\mathbf{a}, \mathbf{b}, \mathbf{C}; \lambda)
$$

where:
- $\mathbf{a} = \frac{1}{|A|}\mathbf{1}_{|A|}$ and $\mathbf{b} = \frac{1}{|B|}\mathbf{1}_{|B|}$ are uniform distributions
- $\mathbf{C}_{ij} = \delta(a_i, b_j)$ is the cost matrix (graph distances)
- $\lambda = 0.4$ is the entropic regularization parameter
- Disconnected pairs (no path within 8 hops) are assigned cost 10.0

The Sinkhorn algorithm iteratively computes the transport plan $\mathbf{P}$ that minimizes $\sum_{ij} P_{ij} C_{ij}$ subject to marginal constraints. Unlike the ontology similarity metric which uses best-match averaging, OT considers the full distribution alignment and penalizes imbalanced skill coverage.

**Range**: $d_{\text{OT}} \in [0, \infty)$, where lower values indicate better alignment. Typical values range from 0 (identical skill sets) to 10 (completely disjoint sets).

## 3. ISCO Hierarchy Proximity

Leverages the International Standard Classification of Occupations (ISCO-08) taxonomy to measure occupational relatedness. Given resume occupation with ISCO code $\text{ISCO}_r$ and job occupation with code $\text{ISCO}_j$, proximity is computed via hierarchical prefix matching:

$$
\sigma_{\text{ISCO}} = \begin{cases}
1.0 & \text{if } \text{ISCO}_r = \text{ISCO}_j \text{ (4-digit exact match)} \\
0.8 & \text{if } \text{ISCO}_r[:3] = \text{ISCO}_j[:3] \text{ (unit group)} \\
0.6 & \text{if } \text{ISCO}_r[:2] = \text{ISCO}_j[:2] \text{ (minor group)} \\
0.3 & \text{if } \text{ISCO}_r[:1] = \text{ISCO}_j[:1] \text{ (major group)} \\
0.0 & \text{otherwise}
\end{cases}
$$

This 5-level hierarchy captures occupational relatedness at multiple granularities. For example:
- ISCO 2512 (Software Developers) vs 2513 (Web Developers) → 0.8 (same unit group)
- ISCO 2512 (Software Developers) vs 2519 (Software Professionals n.e.c.) → 0.6 (same minor group)
- ISCO 2512 (Software Developers) vs 3512 (ICT User Support) → 0.3 (same major group: ICT)

**Range**: $\sigma_{\text{ISCO}} \in \{0, 0.3, 0.6, 0.8, 1.0\}$, discrete values reflecting ISCO hierarchy levels.

---