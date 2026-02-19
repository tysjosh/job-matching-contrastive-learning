# Career-Aware Contrastive Learning (CACL) — Methodology

## 1. Problem Statement

Resume-job matching is a semantic matching problem: given a resume and a job description, predict whether the candidate is a good fit (i.e., likely to receive an interview). Traditional keyword-based approaches fail to capture the nuanced relationships between career trajectories, transferable skills, and job requirements.

CACL addresses this by learning a shared embedding space where semantically compatible resume-job pairs are pulled closer together, while incompatible pairs are pushed apart — guided by occupational ontology knowledge from the ESCO (European Skills, Competences, Qualifications and Occupations) framework.

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CACL Training Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Structured   │    │  ESCO KG     │    │  Raw Dataset          │  │
│  │  Dataset      │───▶│  Enrichment  │───▶│  (JSONL)              │  │
│  │  (Resume+Job) │    │  (Skill URIs)│    │  4,143 train samples  │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │              │
│                                          ┌────────────▼────────────┐ │
│                                          │   80/10/10 Data Split   │ │
│                                          │  Train / Val / Test     │ │
│                                          └────────────┬────────────┘ │
│                                                       │              │
│                    ┌──────────────────────────────────┐│              │
│                    │                                  ││              │
│              ┌─────▼─────┐                  ┌─────────▼──────────┐   │
│              │  Phase 1   │                  │     Phase 2        │   │
│              │ Contrastive│─── best ckpt ──▶│  Classification    │   │
│              │ Pretraining│                  │  Fine-tuning       │   │
│              └─────┬─────┘                  └─────────┬──────────┘   │
│                    │                                  │              │
│              ┌─────▼─────┐                  ┌─────────▼──────────┐   │
│              │  Phase 1   │                  │     Phase 2        │   │
│              │ Evaluation │                  │  Evaluation        │   │
│              │ (Val set)  │                  │  (Test set)        │   │
│              └───────────┘                  └────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Dataset and Data Representation

### 3.1 Dataset Overview

The dataset consists of 5,179 resume-job pairs derived from real recruitment data. Each sample was originally labeled with one of three categories by domain experts:

| Original Label | Count | Description |
|---------------|-------|-------------|
| good_fit | 1,353 | Strong match — candidate likely to receive interview |
| potential_fit | 1,139 | Partial match — candidate has some relevant qualifications |
| no_fit | 2,687 | Poor match — candidate lacks key requirements |

For binary classification, these are mapped as: `good_fit → 1 (positive)`, `potential_fit → 0 (negative)`, `no_fit → 0 (negative)`. This means the negative class contains both clearly unqualified candidates and borderline candidates, which introduces label noise and makes the classification task harder.

### 3.2 Data Splits

The dataset is split sequentially (80/10/10) to avoid temporal leakage:

| Split | Total | Positive (good_fit) | Negative | Pos % | Unique Jobs |
|-------|-------|---------------------|----------|-------|-------------|
| Train | 4,143 | 1,086 | 3,057 (920 potential + 2,137 no_fit) | 26.2% | 381 |
| Validation | 517 | 122 | 395 (107 potential + 288 no_fit) | 23.6% | 149 |
| Test | 519 | 145 | 374 (112 potential + 262 no_fit) | 27.9% | 155 |

### 3.3 ESCO Enrichment

Each sample is pre-enriched with ESCO ontology data during preprocessing. Skill names from resumes and jobs are mapped to ESCO skill URIs, enabling ontology-based distance computation. 78% of training samples have both resume and job skill URIs available.

Quality tier distribution (training set): Tier A: 77.6% | Tier B: 0.4% | Tier C: 22.0%

### 3.4 Sample Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Sample                          │
├──────────────────────────┬──────────────────────────────────┤
│        Resume            │           Job                     │
├──────────────────────────┼──────────────────────────────────┤
│ • role                   │ • title                           │
│ • experience_level       │ • description                     │
│ • skills [{name, uri}]   │ • required_skills [{name, uri}]   │
│ • experience [entries]   │ • skill_uris (from ESCO)          │
│ • skill_uris (from ESCO) │                                   │
├──────────────────────────┴──────────────────────────────────┤
│ label: 1 (positive/match) or 0 (negative/no match)          │
│ metadata: quality_tier, ontology_similarity, ot_distance     │
└─────────────────────────────────────────────────────────────┘
```

Precomputed metadata fields:
- **ontology_similarity**: Symmetric best-match skill similarity between resume and job skill URI sets (0–1)
- **ot_distance**: Sinkhorn optimal transport distance between skill sets on the ESCO graph
- **quality_tier**: A/B/C grade based on ESCO skill URI coverage (A = both have rich skill URIs, C = one or both missing)

## 4. ESCO Knowledge Graph Integration

### 4.1 What is ESCO?

ESCO (European Skills, Competences, Qualifications and Occupations) is a standardized occupational taxonomy maintained by the European Commission. It provides a structured graph connecting occupations to the skills they require. CACL uses the ESCO knowledge graph to inject domain knowledge about skill relationships into the training process.

### 4.2 Graph Structure

The ESCO knowledge graph used in CACL contains 18,237 nodes and 132,679 edges:

| Node Type | Count | Description |
|-----------|-------|-------------|
| Skills | 14,359 | Individual skills/competences (e.g., "Python programming", "project management") |
| Occupations | 3,039 | Job roles (e.g., "Software developer", "Data analyst") |
| ISCO Groups | 619 | Hierarchical occupation categories from the ISCO-08 classification |
| Other | 220 | Additional classification nodes |

Edges connect occupations to the skills they require (`occupation → skill`) and occupations to their ISCO group (`occupation → isco`). When converted to an undirected graph, this creates implicit paths between skills: two skills are "close" if they are required by the same or similar occupations.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ESCO Knowledge Graph Structure                      │
│                                                                       │
│  Example: How "Python" and "Java" are connected                       │
│                                                                       │
│  skill:Python ◄────── occupation:Software_Developer ──────► skill:Java│
│       │                        │                                │     │
│       │                   isco:C2514                             │     │
│       │                  (Software                               │     │
│       │                  developers)                             │     │
│       │                        │                                │     │
│       └──── occupation:Data_Scientist ────► skill:Statistics ───┘     │
│                                                                       │
│  shortest_path(Python, Java) = 2  (through Software_Developer)        │
│  shortest_path(Python, Statistics) = 2  (through Data_Scientist)      │
│  shortest_path(Java, Statistics) = 4  (through two occupations)       │
│                                                                       │
│  Skills required by the same occupation are distance 2 apart.         │
│  Skills in related occupations are further apart.                     │
│  Skills in unrelated fields have no path (distance = ∞).             │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 How CACL Uses the ESCO Graph

The ESCO graph is used in two distinct ways during training:

```
┌───────────────────────────────────────────────────────────────────┐
│                    Two Uses of ESCO in CACL                        │
│                                                                    │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐ │
│  │  Usage 1: Negative          │  │  Usage 2: Sample-level      │ │
│  │  Selection (§5.6)           │  │  Loss Weighting (§5.7)      │ │
│  │                             │  │                             │ │
│  │  COMPUTED AT TRAINING TIME  │  │  PRECOMPUTED DURING         │ │
│  │                             │  │  DATA PREPROCESSING         │ │
│  │  For each anchor resume,    │  │                             │ │
│  │  compute skill similarity   │  │  Each sample stores:        │ │
│  │  to every candidate         │  │  • ontology_similarity      │ │
│  │  negative job. Bucket       │  │  • ot_distance              │ │
│  │  into hard/medium/easy      │  │  • quality_tier             │ │
│  │  based on distance.         │  │                             │ │
│  │                             │  │  These scale each sample's  │ │
│  │  Uses: ontology_set_        │  │  loss contribution by       │ │
│  │  similarity (§4.4)          │  │  0.5x – 1.5x               │ │
│  └─────────────────────────────┘  └─────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### 4.4 Pairwise Skill Similarity (ontology_set_similarity)

To compare the skill sets of a resume and a job, CACL computes a symmetric similarity score using shortest-path distances on the ESCO graph. This is done in three steps:

**Step 1: Skill-to-skill similarity.** Given two individual skill URIs `u` and `v`, their similarity decays exponentially with their shortest-path distance on the graph:

```
skill_sim(u, v) = exp(-α · d(u, v))       where α = 0.7, d = shortest path length

Examples (using the graph above):
  skill_sim(Python, Java)       = exp(-0.7 × 2) = 0.247   (same occupation → moderate)
  skill_sim(Python, Statistics) = exp(-0.7 × 2) = 0.247   (same occupation → moderate)
  skill_sim(Java, Statistics)   = exp(-0.7 × 4) = 0.061   (different occupations → low)
  skill_sim(Python, Python)     = exp(-0.7 × 0) = 1.000   (identical → perfect)
  skill_sim(Python, Cooking)    = 0.000                     (no path → zero)
```

The decay parameter α = 0.7 means similarity drops to ~0.25 at distance 2 (skills in the same occupation) and to ~0.06 at distance 4 (skills in related but different occupations). Skills with no connecting path get similarity 0.

**Step 2: Directional set score.** For a set of skills X (e.g., resume skills) and a set Y (e.g., job skills), compute how well X is "covered" by Y. For each skill in X, find its best match in Y:

```
dir_score(X → Y) = (1 / |X|) · Σ_{x ∈ X} max_{y ∈ Y} skill_sim(x, y)
```

This measures: "on average, how well is each skill in X represented by at least one skill in Y?" A score of 1.0 means every skill in X has an identical match in Y. A score near 0 means X's skills are unrelated to anything in Y.

**Step 3: Symmetric similarity.** The final similarity is the average of both directions:

```
ontology_set_similarity(A, B) = 0.5 · (dir_score(A → B) + dir_score(B → A))
```

This is symmetric: similarity(resume_skills, job_skills) = similarity(job_skills, resume_skills). It captures both "does the candidate have the skills the job needs?" and "are the candidate's skills relevant to this job?"

### 4.5 Optimal Transport Distance (ot_distance)

While `ontology_set_similarity` uses a greedy best-match approach (each skill matches independently), the Sinkhorn Optimal Transport (OT) distance provides a more holistic comparison by finding the minimum-cost assignment between two skill sets.

**Intuition:** Imagine you need to "transport" the resume's skill distribution to match the job's skill distribution. Each unit of skill must be moved to some skill in the other set, and the cost of moving skill `u` to skill `v` is their shortest-path distance on the ESCO graph. OT finds the assignment that minimizes total transport cost.

**Computation:**

```
Given:
  A = {a₁, a₂, ..., aₙ}  (resume skill URIs)
  B = {b₁, b₂, ..., bₘ}  (job skill URIs)

1. Build cost matrix C ∈ ℝⁿˣᵐ:
   C[i,j] = shortest_path(aᵢ, bⱼ)  on the ESCO graph
   (If no path exists, C[i,j] = 10.0 as a penalty)

2. Define uniform distributions:
   p = (1/n, 1/n, ..., 1/n)   (each resume skill has equal weight)
   q = (1/m, 1/m, ..., 1/m)   (each job skill has equal weight)

3. Solve for the optimal transport plan P* using Sinkhorn iteration:
   P* = argmin_P  Σᵢⱼ Pᵢⱼ · Cᵢⱼ  +  ε · KL(P || pqᵀ)

   where ε = 0.4 (entropic regularization) and KL is the Kullback-Leibler divergence.

4. OT distance = Σᵢⱼ P*ᵢⱼ · Cᵢⱼ
```

The Sinkhorn algorithm approximates the exact OT solution efficiently using iterative matrix scaling (200 iterations max, convergence tolerance 1e-6). The entropic regularization ε = 0.4 smooths the transport plan, making it differentiable and more robust to noise.

**Difference from ontology_set_similarity:** The greedy best-match approach can "reuse" the same skill in Y to match multiple skills in X. OT enforces that each skill contributes proportionally, giving a more balanced comparison. For example, if a resume has 5 Python-related skills and a job needs 1 Python skill + 4 management skills, the greedy approach would score high (all 5 match Python), but OT would score lower (4 skills have no good match).

**Usage in CACL:** Both `ontology_similarity` and `ot_distance` are precomputed during data preprocessing and stored in each sample's metadata. Neither is used directly as a training target or label. Instead, they serve as confidence signals in the sample-level loss weighting (§5.7):

```
┌──────────────────────────────────────────────────────────────────────┐
│         How ot_distance and ontology_similarity flow into training    │
│                                                                       │
│  Data Preprocessing (offline, before training):                       │
│    For each resume-job pair:                                          │
│      1. Map skill names → ESCO skill URIs                             │
│      2. Compute ontology_similarity (greedy best-match, §4.4)         │
│      3. Compute ot_distance (Sinkhorn OT, §4.5)                      │
│      4. Assign quality_tier (A/B/C based on URI coverage)             │
│      5. Store all three in sample metadata                            │
│                                                                       │
│  During Phase 1 Training (per triplet):                               │
│    1. Compute InfoNCE loss as usual                                   │
│    2. Read ontology_similarity, ot_distance, quality_tier             │
│       from the anchor sample's metadata                               │
│    3. Combine into a single weight w ∈ [0.5, 1.5]                    │
│    4. Final loss = w × InfoNCE loss                                   │
│                                                                       │
│  Effect:                                                              │
│    • High ontology_similarity + low ot_distance → w ≈ 1.5            │
│      (ontology confirms this is a strong match → trust this sample)   │
│    • Low ontology_similarity + high ot_distance → w ≈ 0.5            │
│      (ontology says skills don't align → downweight this sample)      │
│    • Missing URIs (tier C) → w ≈ 0.75                                │
│      (no ontology signal → reduce influence)                          │
└──────────────────────────────────────────────────────────────────────┘
```

The key idea: not all training samples are equally informative. Samples where the ESCO ontology agrees with the label (e.g., a positive pair with high skill overlap) should influence the model more than samples where the ontology data is missing or contradicts the label. The `ot_distance` and `ontology_similarity` together provide two complementary views of skill alignment — one greedy, one holistic — and their average forms the ontology signal used in the weight calculation.


## 5. Phase 1 — Contrastive Pretraining

### 5.1 Objective

Learn a shared embedding space where matching resume-job pairs have higher cosine similarity than non-matching pairs, without relying on binary labels directly. The contrastive objective forces the model to learn fine-grained semantic distinctions.

### 5.2 Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Phase 1 Model Architecture                     │
│                                                                  │
│  Resume Text ──┐                                                 │
│                │    ┌────────────────────┐    ┌──────────────┐   │
│                ├───▶│  SentenceTransformer│───▶│  Projection  │──▶ 128-d embedding
│                │    │  (all-mpnet-base-v2)│    │  Head (MLP)  │   │
│  Job Text ─────┘    │  768-d output       │    │  768→256→128 │   │
│                     │  ❄ FROZEN ❄        │    │  🔥 TRAINABLE│   │
│                     └────────────────────┘    └──────────────┘   │
│                                                                  │
│  Projection Head:                                                │
│    Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 128)    │
│                                                                  │
│  Trainable params:  ~230K  (projection head only)                │
│  Frozen params:     ~110M  (SentenceTransformer)                 │
└─────────────────────────────────────────────────────────────────┘
```

The SentenceTransformer (`all-mpnet-base-v2`) is frozen to prevent catastrophic forgetting. Only the lightweight projection head is trained. This is a deliberate research design choice: the pre-trained language model already captures rich semantic representations; the projection head learns to map these into a task-specific space optimized for resume-job matching.

### 5.3 Text Encoding Priority

Content is serialized to text with a deliberate field priority to fit within the 512-token window:

```
Resume: "Position: {exp_level} {role} [SEP] Skills: {skill1, skill2, ...} [SEP] Profile: {experience_text}"
Job:    "Position: {title} [SEP] Required Skills: {skill1, skill2, ...} [SEP] Description: {desc_text}"
```

Skills are placed before experience text to guarantee they are always encoded (experience is truncated at ~800 chars).

### 5.4 Embedding Cache Strategy

```
┌──────────────────────────────────────────────────────────────┐
│                    Embedding Cache Flow                        │
│                                                               │
│  Content ──▶ SHA-256 hash ──▶ Cache lookup                    │
│                                    │                          │
│                          ┌─────────┴──────────┐               │
│                          │                    │               │
│                       HIT ✓               MISS ✗              │
│                          │                    │               │
│                   Return cached         Encode with frozen    │
│                   768-d text emb        SentenceTransformer   │
│                          │                    │               │
│                          │              Store in cache         │
│                          │                    │               │
│                          └────────┬───────────┘               │
│                                   │                           │
│                          Pass through trainable               │
│                          projection head (fresh                │
│                          each batch → grad flows)             │
│                                   │                           │
│                          128-d final embedding                │
│                          (with computational graph)           │
└──────────────────────────────────────────────────────────────┘
```

Key insight: only the frozen 768-d text embeddings are cached. The trainable projection head processes them fresh each batch, creating a new computational graph so gradients can flow during backpropagation. This gives the speed benefit of caching without breaking gradient flow.

### 5.5 Triplet Construction

Within each batch, only positive samples (label=1, i.e., `good_fit`) are used as anchors. Negative samples (label=0) in the batch are not used as anchors — they only contribute their jobs to the candidate negative pool. This means Phase 1 learns from the perspective of matching resumes: "given a resume that matches this job, learn to distinguish it from non-matching jobs."

Each anchor produces one triplet: the resume is the anchor, the matched job is the positive, and up to 7 negatives are selected from the global pool of 381 unique jobs.

```
┌──────────────────────────────────────────────────────────┐
│                  Contrastive Triplet                      │
│                                                           │
│  Anchor (Resume) ◄──── positive pair ────► Positive (Job) │
│        │                                                  │
│        │──── negative pairs ────► Negative Job 1          │
│        │                         Negative Job 2           │
│        │                         ...                      │
│        │                         Negative Job 7           │
│        │                    (max 7 per anchor)            │
└──────────────────────────────────────────────────────────┘
```

With 4,143 training samples at batch size 64, this produces ~65 batches per epoch. Since only ~26% of samples are positive, each batch yields roughly 16-18 triplets.


### 5.6 Ontology-Aware Negative Selection with Curriculum Learning

This is a core contribution of CACL. Instead of random negative sampling, negatives are selected based on their skill-level ontology distance to the anchor resume, using the ESCO knowledge graph.

```
┌──────────────────────────────────────────────────────────────────┐
│            Ontology-Aware Negative Selection                      │
│                                                                   │
│  For each anchor resume (with skill URIs):                        │
│                                                                   │
│  1. Compute ontology_set_similarity(resume_skills, job_skills)    │
│     for every candidate negative job                              │
│                                                                   │
│  2. Convert to distance: d = 1 - similarity                      │
│                                                                   │
│  3. Bucket candidates:                                            │
│     ┌──────────┬─────────────────┬──────────────────────────┐     │
│     │ Bucket   │ Distance Range  │ Meaning                  │     │
│     ├──────────┼─────────────────┼──────────────────────────┤     │
│     │ HARD     │ d ≤ 0.3         │ Very similar skills      │     │
│     │ MEDIUM   │ 0.3 < d ≤ 0.6  │ Partially overlapping    │     │
│     │ EASY     │ d > 0.6         │ Very different skills    │     │
│     └──────────┴─────────────────┴──────────────────────────┘     │
│                                                                   │
│  4. Curriculum learning shifts ratios over epochs:                 │
│                                                                   │
│     epoch_ratio = current_epoch / total_epochs  (0.0 → 1.0)      │
│                                                                   │
│     ┌──────────┬──────────────┬──────────────┐                    │
│     │ Bucket   │ Early (ε=0)  │ Late (ε=1)   │                    │
│     ├──────────┼──────────────┼──────────────┤                    │
│     │ HARD     │    20%       │    60%        │                    │
│     │ MEDIUM   │    30%       │    30%        │                    │
│     │ EASY     │    50%       │    10%        │                    │
│     └──────────┴──────────────┴──────────────┘                    │
│                                                                   │
│  Rationale: Early training uses mostly easy negatives so the      │
│  model learns basic distinctions first. As training progresses,   │
│  harder negatives force the model to learn fine-grained           │
│  skill-level differences — similar to curriculum learning in       │
│  education.                                                       │
│                                                                   │
│  Fallback: Samples without skill URIs get random negatives.       │
│  The loss engine downweights these via quality tier (§7).         │
└──────────────────────────────────────────────────────────────────┘
```

### 5.7 Loss Function — InfoNCE with Sample-Level Ontology Weighting

The loss has two components: standard InfoNCE for the contrastive objective, and a sample-level multiplicative weight based on ontology data quality.

#### InfoNCE Loss

```
L_InfoNCE = -log( exp(sim(a, p⁺) / τ) / (exp(sim(a, p⁺) / τ) + Σᵢ exp(sim(a, nᵢ⁻) / τ)) )

where:
  a   = anchor embedding (resume)
  p⁺  = positive embedding (matching job)
  nᵢ⁻ = negative embeddings (non-matching jobs)
  τ   = temperature = 0.07
  sim = cosine similarity (dot product on L2-normalized vectors)
```

The temperature τ=0.07 sharpens the similarity distribution, making the model more sensitive to small differences.

#### Sample-Level Ontology Weight

Each triplet's loss is scaled by a weight w ∈ [0.5, 1.5] based on the precomputed ontology enrichment scores:

```
┌──────────────────────────────────────────────────────────────┐
│              Sample-Level Ontology Weighting                  │
│                                                               │
│  1. Base weight from quality tier:                            │
│     A → 1.0  |  B → 0.9  |  C → 0.75  |  D → 0.6  |  F → 0.5│
│                                                               │
│  2. Ontology signal (average of available scores):            │
│     ont_signal = mean(ontology_similarity, 1 - ot_dist/10)   │
│                                                               │
│  3. Final weight:                                             │
│     w = base × (1 + 0.3 × (2 × ont_signal - 1))             │
│     w = clamp(w, 0.5, 1.5)                                   │
│                                                               │
│  Effect: Samples with rich ESCO coverage and strong ontology  │
│  agreement contribute more to the loss. Samples with missing  │
│  or weak ontology data are downweighted, reducing noise.      │
│                                                               │
│  Final loss per triplet:                                      │
│     L = w × L_InfoNCE                                         │
└──────────────────────────────────────────────────────────────┘
```

### 5.8 Training Procedure

Each epoch processes all 65 batches. After each batch:
1. Gradients are clipped to max norm 1.0 to prevent gradient explosion (large norms are logged as warnings)
2. The Adam optimizer updates only the projection head parameters

After each epoch:
1. Validation loss is computed on the held-out validation set (517 samples) using the same triplet construction and InfoNCE loss, but with no gradient updates
2. If the validation loss improves, the model is saved as `best_checkpoint.pt`
3. The best checkpoint (lowest validation loss) is passed to Phase 2

### 5.9 Training Configuration (Phase 1)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 15 | Val loss best at epoch 13 |
| Batch size | 64 | Balance between gradient stability and memory |
| Learning rate | 8.5e-5 | Conservative for projection head |
| Temperature | 0.07 | Sharp similarity distribution |
| Max negatives/anchor | 7 | Sufficient contrast without memory issues |
| Projection dim | 128 | Compact embedding space |
| Projection dropout | 0.3 | Regularization |
| Global neg pool | 381 unique jobs | All unique jobs from training set (deduplicated by job_id) |
| Embedding cache | Enabled, not cleared between epochs | Text encoder is frozen → embeddings don't change |
| Validation | Every epoch, on held-out validation set | Early stopping via best checkpoint |


## 6. Phase 2 — Classification Fine-tuning

### 6.1 Objective

Use the embedding space learned in Phase 1 as a feature extractor, and train a classification head to predict binary match/no-match labels. This converts the unsupervised contrastive signal into a supervised prediction.

### 6.2 Model Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Phase 2 Model Architecture                         │
│                                                                       │
│                     ┌──────────────────┐                              │
│  Resume Text ──────▶│ SentenceTransformer│──▶ 768-d ──┐               │
│                     │ ❄ FROZEN ❄       │             │               │
│                     └──────────────────┘             │               │
│                                                      │               │
│                     ┌──────────────────┐             │               │
│                     │ Pre-trained       │             ▼               │
│                     │ Projection Head   │──▶ 128-d resume_emb        │
│                     │ ❄ FROZEN ❄       │                    ┐        │
│                     │ (from Phase 1)    │                    │        │
│                     └──────────────────┘                    │        │
│                                                             │ concat │
│                     ┌──────────────────┐                    │ 256-d  │
│                     │ SentenceTransformer│──▶ 768-d ──┐     │        │
│  Job Text ─────────▶│ ❄ FROZEN ❄       │             │     │        │
│                     └──────────────────┘             │     │        │
│                                                      │     │        │
│                     ┌──────────────────┐             ▼     │        │
│                     │ Pre-trained       │──▶ 128-d job_emb  │        │
│                     │ Projection Head   │                    ┘        │
│                     │ ❄ FROZEN ❄       │                    │        │
│                     └──────────────────┘                    │        │
│                                                             ▼        │
│                     ┌──────────────────────────────────────────┐      │
│                     │         Classification Head               │      │
│                     │  🔥 TRAINABLE (~41K params)               │      │
│                     │                                           │      │
│                     │  Linear(256, 128) → ReLU → Dropout(0.3)  │      │
│                     │  Linear(128, 64)  → ReLU → Dropout(0.3)  │      │
│                     │  Linear(64, 1)    → Sigmoid              │      │
│                     │                                           │      │
│                     │  Output: P(match) ∈ [0, 1]               │      │
│                     └──────────────────────────────────────────┘      │
│                                                                       │
│  Total trainable:  ~41K  (classification head only)                   │
│  Total frozen:     ~110M + 230K  (text encoder + projection head)     │
└──────────────────────────────────────────────────────────────────────┘
```

The entire Phase 1 model (SentenceTransformer + projection head) is frozen. Only the new classification head is trained. This preserves the contrastive embedding space while learning a decision boundary on top of it.

### 6.3 Class Imbalance Handling

The dataset has ~74% negative / ~26% positive samples (after binarization, where `potential_fit` and `no_fit` are both label=0). Without correction, the model collapses to predicting all-negative. CACL uses weighted BCE loss:

```
L_BCE = -(1/N) Σᵢ wᵢ · [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

where wᵢ = 2.5  if yᵢ = 1 (positive sample)
      wᵢ = 1.0  if yᵢ = 0 (negative sample)
```

The weight 2.5 compensates for the ~2.6:1 class imbalance ratio.

### 6.4 Training Configuration (Phase 2)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 10 | Loss still decreasing; more possible |
| Batch size | 32 | Smaller batches for classification stability |
| Learning rate | 5e-4 | Higher than Phase 1 (only training small head) |
| Weight decay | 0.001 | L2 regularization |
| Classification dropout | 0.3 | Prevent overfitting |
| pos_class_weight | 2.5 | Compensate ~74/26 class imbalance |
| Freeze contrastive layers | true | Preserve Phase 1 embedding space |
| Pretrained model | Phase 1 best checkpoint (epoch 13) | Best validation loss |

### 6.5 Training Procedure

Phase 2 uses all labeled samples (both positive and negative) for supervised training. Each epoch processes all 4,143 training samples in batches of 32 (~130 batches per epoch). Validation is run every epoch on the validation set (517 samples), tracking both loss and accuracy. The best checkpoint is saved based on validation loss.

Unlike Phase 1, Phase 2 does not use triplet construction, negative sampling, or the ESCO ontology. It is a straightforward binary classification task on top of frozen contrastive embeddings.

## 7. Evaluation Methodology

### 7.1 Phase 1 Evaluation (Embedding Quality)

Evaluated on the validation set (517 samples). Measures how well the contrastive embedding space separates matching from non-matching pairs using cosine similarity:

- **AUC-ROC**: Area under the ROC curve for similarity-based classification
- **Embedding separation**: Difference between mean positive and mean negative cosine similarity
- **Optimal threshold**: Similarity threshold that maximizes F1

### 7.2 Phase 2 Evaluation (Classification Performance)

Evaluated on the held-out test set (519 samples, never seen during training). The optimal classification threshold is tuned on the validation set, then applied to the test set:

- **Binary classification**: Accuracy, Precision, Recall, F1, AUC-ROC at both default (0.5) and optimal thresholds
- **Job ranking (MAP, MRR, NDCG)**: For each resume, rank all candidate jobs by predicted match probability. Measures how highly the correct job is ranked.
- **Resume ranking**: For each job, rank all candidate resumes. Measures retrieval quality from the employer's perspective.

Note: The test set contains 145 `good_fit`, 112 `potential_fit`, and 262 `no_fit` samples. Since `potential_fit` is labeled as negative (0), the model is evaluated on its ability to distinguish `good_fit` from both `potential_fit` and `no_fit`. This is a conservative evaluation — some "false positives" may be `potential_fit` samples that the model reasonably scores highly.

### 7.3 Results Summary

```
┌────────────────────────────────────────────────────────────────┐
│                      Results Summary                            │
├────────────────────────┬───────────────┬───────────────────────┤
│ Metric                 │ Phase 1       │ Phase 2               │
├────────────────────────┼───────────────┼───────────────────────┤
│ AUC-ROC                │ 0.720         │ 0.799                 │
│ F1 Score               │ 0.482         │ 0.598                 │
│ Embedding Separation   │ 0.078         │ 0.249                 │
│ Positive Mean Sim      │ 0.821         │ 0.647 (probability)   │
│ Negative Mean Sim      │ 0.743         │ 0.398 (probability)   │
├────────────────────────┼───────────────┼───────────────────────┤
│ Job MAP                │ —             │ 0.260                 │
│ Resume MAP             │ —             │ 0.877                 │
│ Job NDCG               │ —             │ 0.418                 │
├────────────────────────┼───────────────┼───────────────────────┤
│ Train Loss (final)     │ 0.515         │ 0.721                 │
│ Val Loss (final)       │ 0.806         │ 0.748                 │
│ Best Val Loss          │ 0.720 (ep 13) │ 0.748 (ep 10)        │
│ Training Time          │ ~10 min       │ ~94 min               │
└────────────────────────┴───────────────┴───────────────────────┘
```

## 8. End-to-End Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. DATA PREPARATION                                                    │
│     Dataset (JSONL) ──▶ ESCO enrichment (skill URIs, ontology scores)   │
│                    ──▶ 80/10/10 sequential split                        │
│                                                                         │
│  2. PHASE 1: CONTRASTIVE PRETRAINING                                    │
│     For each epoch:                                                     │
│       For each batch of 64 samples:                                     │
│         ├─ Filter positive samples (label=1) as anchors                 │
│         ├─ For each anchor:                                             │
│         │   ├─ Resume = anchor, matched job = positive                  │
│         │   ├─ Select 7 negatives from global pool (ontology-aware)     │
│         │   └─ Curriculum learning shifts hard/easy ratio               │
│         ├─ Encode all content → 768-d (cached) → 128-d (projection)    │
│         ├─ Compute InfoNCE loss × ontology sample weight                │
│         ├─ Backprop through projection head only                        │
│         └─ Gradient clipping (max norm = 1.0)                           │
│       Run validation on val set → save best checkpoint                  │
│                                                                         │
│  3. PHASE 1 EVALUATION                                                  │
│     Load best checkpoint → compute cosine similarities on val set       │
│     → AUC-ROC, separation, threshold analysis                           │
│                                                                         │
│  4. PHASE 2: CLASSIFICATION FINE-TUNING                                 │
│     Load Phase 1 best checkpoint (frozen)                               │
│     For each epoch:                                                     │
│       For each batch of 32 labeled samples:                             │
│         ├─ Encode resume → 768-d → 128-d (frozen pipeline)              │
│         ├─ Encode job → 768-d → 128-d (frozen pipeline)                 │
│         ├─ Concatenate [resume_emb; job_emb] → 256-d                    │
│         ├─ Classification head → P(match)                               │
│         ├─ Weighted BCE loss (pos_weight=2.5)                           │
│         └─ Backprop through classification head only                    │
│       Run validation on val set                                         │
│                                                                         │
│  5. PHASE 2 EVALUATION                                                  │
│     Load best checkpoint → predict on test set                          │
│     → Classification metrics + ranking metrics (MAP, NDCG)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 9. Key Design Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| Freeze SentenceTransformer | Prevents catastrophic forgetting of pre-trained language knowledge; only ~230K params trained vs ~110M frozen |
| Two-phase training | Phase 1 learns embedding geometry without label bias; Phase 2 leverages it for classification |
| Ontology-aware negatives | Random negatives are too easy; ESCO-guided selection creates informative contrasts that teach skill-level distinctions |
| Curriculum learning | Prevents early training collapse from too-hard negatives; gradually increases difficulty as the model improves |
| Cache text embeddings only | Frozen encoder produces identical outputs → safe to cache. Projection head must run fresh for gradient flow |
| Global negative pool | Ensures consistent negative difficulty across batches (vs in-batch sampling which varies with batch composition) |
| Sample-level ontology weight | Downweights noisy samples (missing ESCO data) and upweights high-quality samples where ontology confirms the label |
| Weighted BCE in Phase 2 | Prevents collapse to majority-class prediction under ~74/26 class imbalance |
| Sequential data split | Avoids temporal leakage if data has any chronological ordering |
| Binary label mapping | `good_fit` → 1, `potential_fit` + `no_fit` → 0; conservative choice that treats borderline candidates as negative |
