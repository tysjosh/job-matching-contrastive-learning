# ESCO Enrichment Process - Camera-Ready Description

## Two-Paragraph Summary

We enrich each resume-job pair with structured ontology metadata from the European Skills, Competences, Qualifications and Occupations (ESCO) knowledge graph through a multi-stage pipeline. First, raw skill strings undergo cleaning (CamelCase splitting, alias normalization, proficiency suffix removal) and are linked to ESCO skill URIs via a three-tier matching strategy: (1) exact string matching against ESCO's preferred and alternative labels, (2) vendor-specific alias mapping for common technology terms lacking ESCO labels (e.g., "AWS"→cloud technologies, "Docker"→virtualization), and (3) semantic matching using sentence-transformer embeddings (all-MiniLM-L6-v2) with cosine similarity threshold 0.55 to handle vocabulary mismatch between industry terminology and ESCO's standardized labels. Job occupation titles are resolved to canonical ESCO occupation URIs through hierarchical matching: direct UUID recognition, slug-to-title extraction with fuzzy matching (RapidFuzz WRatio, cutoff=82), and fallback to job title matching. For each resolved occupation, we compute essential and optional skill coverage by intersecting the candidate's linked skill URIs with the occupation's required skill profile from ESCO's occupationSkillRelations. Each record is assigned a quality tier (A-F) based on metadata completeness: tier A indicates both sides have linked skill URIs with occupation coverage, while tier F indicates missing raw skills entirely.

Following URI linking, we compute two ontology-aware similarity scores that quantify semantic alignment beyond surface-level overlap. The ontology similarity score uses symmetric best-match averaging: for each skill in set A, we find its closest match in set B using graph distance d in the ESCO knowledge graph, compute similarity as exp(-αd) with decay parameter α=0.7 and maximum path length of 8 hops, then average bidirectionally to yield a score in [0,1]. The optimal transport (OT) distance applies the Sinkhorn algorithm to compute the Wasserstein-1 distance between skill distributions, treating each skill set as a uniform distribution and using graph distance as the ground metric (disconnected pairs assigned cost 10.0, regularization λ=0.4). These precomputed scores—ontology_similarity (higher is better), ot_distance (lower is better), quality tier, and occupation coverage—are stored in the esco_enrichment_v3 metadata block and subsequently used by OSCAR to modulate sample-level loss weights during contrastive training, enabling the model to focus on high-confidence career transitions while remaining robust to label noise.

---

## Key Technical Details

### Stage 1: Skill Linking (Three-Tier Matching)
- **Input**: Raw skill strings (e.g., "JavaJ2EE (expert)", "HTML/CSS", "AWS")
- **Cleaning**: CamelCase splitting, alias normalization (j2ee→java ee), proficiency removal
- **Tier 1 - Exact Match**: Direct lookup against ESCO preferred/alternative labels
- **Tier 2 - Vendor Aliases**: 150+ hardcoded mappings for tech terms (AWS→cloud technologies, Docker→virtualization, React→JavaScript Framework)
- **Tier 3 - Semantic Match**: Sentence-transformer embeddings (all-MiniLM-L6-v2) with cosine similarity ≥0.55
- **Output**: Set of ESCO skill URIs per resume/job

### Stage 2: Occupation Resolution
- **Input**: Raw occupation URI (slug or plain text) + job title
- **Resolution**: UUID detection → slug extraction → fuzzy matching (RapidFuzz WRatio ≥82) → job title fallback
- **Coverage**: Compute essential/optional skill coverage using ESCO occupationSkillRelations
- **Output**: Canonical occupation URI + coverage metrics

### Stage 3: Ontology Scoring
- **Ontology Similarity**: Symmetric best-match averaging with exponential decay
  - Pairwise similarity: $s_{\text{skill}}(u,v) = \exp(-\beta \cdot \delta(u,v))$ where $\beta=0.7$
  - Directional coverage: $s(A \rightarrow B) = \frac{1}{|A|} \sum_{a \in A} \max_{b \in B} s_{\text{skill}}(a,b)$
  - Symmetric score: $s_{\text{ont}}(A,B) = \frac{1}{2}[s(A \rightarrow B) + s(B \rightarrow A)]$
  - Maximum path length: 8 hops
- **Optimal Transport**: Sinkhorn algorithm with graph-distance cost matrix, λ=0.4 regularization
- **Quality Tier**: A (full metadata + occupation) → F (missing skills)

### Integration with OSCAR
The enrichment outputs feed directly into OSCAR's weight computation:
- `ontology_similarity` → σ_skill component
- `ot_distance` → σ_OT component (normalized)
- `quality_tier` → w_tier base weight
- `occupation_uri` → enables ISCO proximity (σ_ISCO)

This preprocessing enables efficient, differentiable loss weighting without runtime ontology queries.

### Why Semantic Matching Matters
The semantic matching stage (v4 enrichment) addresses vocabulary mismatch between industry terminology and ESCO's standardized European taxonomy. For example:
- "AWS" has no ESCO label → semantic embedding maps to "cloud technologies" (0.72 similarity)
- "Docker" → "manage ICT virtualisation environments" (0.68 similarity)
- "React" → "JavaScript Framework" (0.81 similarity)

This dramatically improves skill coverage for technology-heavy resumes/jobs where exact string matching fails, increasing linked skill URIs by ~40% in our dataset.
