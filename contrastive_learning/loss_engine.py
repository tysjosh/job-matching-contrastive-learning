"""
Contrastive loss engine for pathway-aware contrastive learning.

This module implements the ContrastiveLossEngine that computes contrastive loss
with temperature scaling, pathway weighting, and support for multiple view combinations.
"""

import torch
import logging
from typing import List, Dict, Any, Optional

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .data_structures import ContrastiveTriplet, TrainingConfig

logger = logging.getLogger(__name__)


class ContrastiveLossEngine:
    """
    Computes pathway-weighted contrastive loss for resume-job matching.

    This engine implements contrastive loss with temperature scaling and pathway-aware
    weighting that gives higher importance to career-relevant distinctions. It supports
    multiple view combinations and includes numerical stability checks.
    """

    def __init__(self, config: TrainingConfig, skill_matcher=None):
        """
        Initialize the contrastive loss engine.

        Args:
            config: Training configuration containing temperature, pathway_weight, etc.
            skill_matcher: Optional OntologySkillMatcher for on-the-fly φ computation
                          in ordinal loss L₂. If None, falls back to precomputed φ.

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ContrastiveLossEngine. "
                "Install with: pip install torch"
            )

        self.config = config
        self.temperature = config.temperature
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Numerical stability parameters
        self.eps = 1e-8
        self.max_exp = 50.0  # Prevent overflow in exp
        self.gradient_clip_value = 1.0

        # Sample-level weighting: quality tier + ontology coverage
        self.ontology_weight = getattr(config, 'ontology_weight', 0.3)
        self.ot_distance_scale = getattr(config, 'ot_distance_scale', 10.0)
        self.use_ot_distance = getattr(config, 'use_ot_distance', True)

        # Loss function selection: "infonce", "wasserstein", "hybrid", or "ordinal"
        self.loss_type = getattr(config, 'loss_type', 'infonce')
        self.ws2_weight = getattr(config, 'ws2_weight', 0.3)

        # Ordinal contrastive loss (OCL) parameters
        self.ordinal_alpha = getattr(config, 'ordinal_alpha', 0.5)
        self.ordinal_lambda1 = getattr(config, 'ordinal_lambda1', 1.0)
        self.ordinal_lambda2 = getattr(config, 'ordinal_lambda2', 1.0)
        self.ordinal_m2 = getattr(config, 'ordinal_m2', 0.3)
        self.ordinal_curriculum_switch = getattr(config, 'ordinal_curriculum_switch', 0.3)
        self.ordinal_fixed_m1 = getattr(config, 'ordinal_fixed_m1', False)

        # Curriculum learning state (set by trainer each epoch)
        self.current_epoch = 0
        self.total_epochs = getattr(config, 'num_epochs', 15)

        # Graduated relevance scores for Wasserstein loss
        # good_fit=3 (strong positive), potential_fit=1 (weak positive), no_fit=0 (negative)
        self.relevance_scores = {'good_fit': 3.0, 'potential_fit': 1.0, 'no_fit': 0.0}

        # Skill matcher for on-the-fly φ(resume_p, job_α) in ordinal L₂
        self.skill_matcher = skill_matcher
        if skill_matcher and self.loss_type == 'ordinal' and not self.ordinal_fixed_m1:
            logger.info("OntologySkillMatcher available — L₂ will compute φ(resume_p, job_α) on the fly")
        elif self.loss_type == 'ordinal' and not self.ordinal_fixed_m1 and not skill_matcher:
            logger.warning("No skill_matcher provided — L₂ will fall back to precomputed φ(resume_p, job_p)")

        # ── Enhanced φ: essential/optional skill weighting ──
        self.phi_use_weighted = getattr(config, 'phi_use_weighted', False)
        self.phi_essential_weight = getattr(config, 'phi_essential_weight', 1.0)
        self.phi_optional_weight = getattr(config, 'phi_optional_weight', 0.5)
        self.essential_skills = {}  # occupation_uri -> set of essential skill URIs
        self.optional_skills = {}   # occupation_uri -> set of optional skill URIs

        if self.phi_use_weighted:
            esco_relations_path = getattr(config, 'esco_relations_path', None)
            if esco_relations_path:
                self._load_essential_optional_skills(esco_relations_path)
            else:
                logger.warning("phi_use_weighted=True but no esco_relations_path — falling back to uniform weighting")
                self.phi_use_weighted = False

        # ── Adaptive margin annealing ──
        self.margin_anneal = getattr(config, 'margin_anneal', False)
        self.margin_anneal_rate = getattr(config, 'margin_anneal_rate', 1.0)

        # ── Confidence-gated margins ──
        self.phi_gate_threshold = getattr(config, 'phi_gate_threshold', 1.0)

        logger.info(f"Initialized ContrastiveLossEngine with temperature={self.temperature}, "
                    f"loss_type={self.loss_type}, ws2_weight={self.ws2_weight}, "
                    f"ontology_weight={self.ontology_weight}, use_ot_distance={self.use_ot_distance}, "
                    f"ordinal_alpha={self.ordinal_alpha}, ordinal_m2={self.ordinal_m2}, "
                    f"device={self.device}")

    def compute_loss(self, triplets: List[ContrastiveTriplet],
                     embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss for a batch of triplets.

        For InfoNCE: computes per-triplet loss and averages.
        For Wasserstein: collects all triplets into a batch matrix and computes
        2-Wasserstein distance once (following SyCL's batch-level approach).

        Args:
            triplets: List of ContrastiveTriplet objects
            embeddings: Dictionary mapping content to embeddings

        Returns:
            torch.Tensor: Computed contrastive loss

        Raises:
            ValueError: If triplets is empty or embeddings are missing
        """
        if not triplets:
            raise ValueError("Cannot compute loss for empty triplets list")

        if not embeddings:
            raise ValueError("Embeddings dictionary cannot be empty")

        if self.loss_type == 'wasserstein':
            return self._compute_batch_wasserstein_loss(triplets, embeddings)

        if self.loss_type == 'hybrid':
            return self._compute_hybrid_loss(triplets, embeddings)

        if self.loss_type == 'ordinal':
            return self._compute_ordinal_loss(triplets, embeddings)

        # InfoNCE: per-triplet loss averaged
        losses = []
        
        for triplet in triplets:
            try:
                triplet_loss = self._compute_triplet_loss(triplet, embeddings)
                if triplet_loss is not None:
                    losses.append(triplet_loss)
            except Exception as e:
                logger.warning(f"Failed to compute loss for triplet: {e}")
                continue

        if not losses:
            logger.warning("No valid triplets found for loss computation")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if len(losses) == 1:
            return losses[0]
        else:
            return torch.stack(losses).mean()

    def _compute_triplet_loss(self, triplet: ContrastiveTriplet,
                              embeddings: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute contrastive loss for a single triplet.

        Applies ontology-aware positive weighting when scores are available
        in view_metadata (from precomputed ESCO enrichment).

        Args:
            triplet: ContrastiveTriplet containing anchor, positive, and negatives
            embeddings: Dictionary mapping content to embeddings

        Returns:
            torch.Tensor: Loss for this triplet, or None if computation fails
        """
        try:
            # Get embeddings for anchor (resume), positive (job), and negatives
            anchor_key = self._get_content_key(triplet.anchor)
            positive_key = self._get_content_key(triplet.positive)

            if anchor_key not in embeddings or positive_key not in embeddings:
                logger.warning("Missing embeddings for anchor or positive")
                return None

            anchor_emb = embeddings[anchor_key]
            positive_emb = embeddings[positive_key]

            # Get negative embeddings
            negative_embs = []

            for i, negative in enumerate(triplet.negatives):
                negative_key = self._get_content_key(negative)
                if negative_key in embeddings:
                    negative_embs.append(embeddings[negative_key])

            if not negative_embs:
                logger.warning("No valid negative embeddings found")
                return None

            # Dispatch to selected loss function
            if self.loss_type == 'wasserstein':
                # Pure Wasserstein: per-triplet EMD fallback
                neg_labels = triplet.view_metadata.get('negative_original_labels', [])
                pos_label = triplet.view_metadata.get('positive_original_label', 'good_fit')
                loss = self._wasserstein_loss(
                    anchor_emb, positive_emb, negative_embs, pos_label, neg_labels)
            else:
                # InfoNCE for both "infonce" and "hybrid" modes
                # (hybrid's WS2 component is computed at batch level in _compute_hybrid_loss)
                loss = self._infonce_loss(anchor_emb, positive_emb, negative_embs)

            if loss is None:
                return None

            # Apply sample-level ontology weight (label confidence + data quality)
            ont_weight = self._compute_ontology_weight(triplet.view_metadata)
            loss = loss * ont_weight

            return loss

        except Exception as e:
            logger.error(f"Error computing triplet loss: {e}")
            return None

    def _compute_ontology_weight(self, view_metadata: Dict[str, Any]) -> float:
        """
        Compute a sample-level weight based on precomputed ontology scores.

        Uses ontology_similarity and ot_distance from ESCO enrichment to
        upweight samples where the ontology confirms the label signal,
        and downweight samples with weak/missing ontology evidence.

        Args:
            view_metadata: Triplet metadata containing ontology scores

        Returns:
            float: Multiplicative weight for this sample's loss (0.5 to 1.5)
        """
        if self.ontology_weight == 0.0:
            return 1.0

        ont_sim = view_metadata.get('ontology_similarity')
        ot_dist = view_metadata.get('ot_distance') if self.use_ot_distance else None
        tier = view_metadata.get('quality_tier', 'F')

        # Base weight from quality tier
        tier_weights = {'A': 1.0, 'B': 0.9, 'C': 0.75, 'D': 0.6, 'F': 0.5}
        base = tier_weights.get(tier, 0.5)

        if ont_sim is None and ot_dist is None:
            return base

        # Ontology signal: blend similarity and normalized OT distance
        ont_signal = 0.0
        count = 0
        if ont_sim is not None:
            ont_signal += ont_sim  # 0-1, higher = more similar
            count += 1
        if ot_dist is not None:
            # Normalize OT distance to 0-1 (inverted: lower distance = higher signal)
            normalized_ot = max(0.0, 1.0 - ot_dist / self.ot_distance_scale)
            ont_signal += normalized_ot
            count += 1

        if count > 0:
            ont_signal /= count  # average of available signals, 0-1

        # Final weight: base ± ontology adjustment
        weight = base * (1.0 + self.ontology_weight * (2.0 * ont_signal - 1.0))

        # Clamp to reasonable range
        return max(0.5, min(1.5, weight))

    def _infonce_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                      negatives: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute standard InfoNCE contrastive loss.

        Negative difficulty is handled by the selection strategy (ontology-based
        bucketing with curriculum learning), not by per-negative weighting.

        Args:
            anchor: Anchor embedding (resume), L2-normalized
            positive: Positive embedding (matching job), L2-normalized
            negatives: List of negative embeddings (non-matching jobs), L2-normalized

        Returns:
            torch.Tensor: InfoNCE loss
        """
        # Ensure all tensors are on the same device
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negatives = [neg.to(self.device) for neg in negatives]

        # Compute similarities (dot product on normalized vectors = cosine similarity)
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        neg_sims = []
        for neg_emb in negatives:
            neg_sim = torch.sum(anchor * neg_emb, dim=-1) / self.temperature
            neg_sims.append(neg_sim)

        # Clamp similarities to prevent overflow
        pos_sim = torch.clamp(pos_sim, max=self.max_exp)
        neg_sims = [torch.clamp(sim, max=self.max_exp) for sim in neg_sims]

        # Standard InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg_i))))
        pos_exp = torch.exp(pos_sim)
        neg_exps = torch.stack([torch.exp(s) for s in neg_sims])
        denominator = pos_exp + torch.sum(neg_exps)

        # Numerical stability
        denominator = torch.clamp(denominator, min=self.eps, max=1e10)
        ratio = pos_exp / denominator
        ratio = torch.clamp(ratio, min=self.eps, max=1.0)
        loss = -torch.log(ratio)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN or infinite loss detected, returning zero loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

    def _wasserstein_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                          negatives: List[torch.Tensor],
                          positive_label: str,
                          negative_labels: List[str]) -> torch.Tensor:
        """
        Compute Wasserstein-based contrastive loss with graduated relevance.

        Per-triplet fallback used when batch-level WS2 fails or for single triplets.
        Uses 1-Wasserstein (EMD) as a simpler alternative.

        Args:
            anchor: Anchor embedding (resume), L2-normalized
            positive: Positive embedding (matching job), L2-normalized
            negatives: List of negative embeddings, L2-normalized
            positive_label: Original label of the positive (good_fit/potential_fit)
            negative_labels: Original labels of each negative

        Returns:
            torch.Tensor: Wasserstein loss (scalar)
        """
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negatives = [neg.to(self.device) for neg in negatives]

        # Build similarity logits: [positive, neg_0, neg_1, ...]
        pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature
        neg_sims = [torch.sum(anchor * neg, dim=-1, keepdim=True) / self.temperature for neg in negatives]
        logits = torch.cat([pos_sim] + neg_sims, dim=-1)
        logits = torch.clamp(logits, max=self.max_exp)

        # Build target relevance scores
        pos_score = self.relevance_scores.get(positive_label, 3.0)
        neg_scores = []
        for i, neg_emb in enumerate(negatives):
            label = negative_labels[i] if i < len(negative_labels) else 'no_fit'
            neg_scores.append(self.relevance_scores.get(label, 0.0))

        target_scores = torch.tensor(
            [pos_score] + neg_scores, dtype=torch.float64, device=self.device
        )

        # Convert to distributions and compute EMD
        pred_dist = F.softmax(logits.double(), dim=-1)
        target_dist = F.softmax(target_scores, dim=-1)

        sorted_indices = torch.argsort(target_scores, descending=True)
        pred_sorted = pred_dist[sorted_indices]
        target_sorted = target_dist[sorted_indices]

        pred_cdf = torch.cumsum(pred_sorted, dim=-1)
        target_cdf = torch.cumsum(target_sorted, dim=-1)
        emd = torch.sum(torch.abs(pred_cdf - target_cdf))

        loss = emd.float()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf in Wasserstein loss, returning zero")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for curriculum learning in ordinal loss."""
        self.current_epoch = epoch

    def _load_essential_optional_skills(self, csv_path: str) -> None:
        """Load essential/optional skill sets per occupation from ESCO relations CSV."""
        import csv
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                occ_uri = row['occupationUri']
                skill_uri = row['skillUri']
                rel_type = row['relationType']
                if rel_type == 'essential':
                    self.essential_skills.setdefault(occ_uri, set()).add(skill_uri)
                elif rel_type == 'optional':
                    self.optional_skills.setdefault(occ_uri, set()).add(skill_uri)
                count += 1
        logger.info(f"Loaded {count} ESCO skill-occupation relations "
                    f"({len(self.essential_skills)} occupations with essential skills)")

    def _compute_phi_weighted(self, resume_skill_uris: List[str],
                               job_skill_uris: List[str],
                               job_occupation_uri: Optional[str] = None) -> Optional[float]:
        """
        Compute weighted φ where essential skills count more than optional.

        Each job skill's contribution to the denominator is weighted:
          - essential skill: phi_essential_weight (default 1.0)
          - optional skill:  phi_optional_weight (default 0.5)
          - unknown:         1.0 (fallback)

        Numerator credit per skill uses the same Strategy D as before.
        """
        if not self.skill_matcher:
            return None

        job_skills = list(set(job_skill_uris))
        resume_skills = list(set(resume_skill_uris))

        if not job_skills:
            return None
        if not resume_skills:
            return 0.0

        # Look up essential/optional sets for this occupation
        ess_set = self.essential_skills.get(job_occupation_uri, set()) if job_occupation_uri else set()
        opt_set = self.optional_skills.get(job_occupation_uri, set()) if job_occupation_uri else set()

        total_credit = 0.0
        total_weight = 0.0
        for j_uri in job_skills:
            # Determine weight for this skill
            if j_uri in ess_set:
                w = self.phi_essential_weight
            elif j_uri in opt_set:
                w = self.phi_optional_weight
            else:
                w = 1.0  # unknown relation — treat as essential

            # Compute credit (Strategy D)
            best_credit = 0.0
            for r_uri in resume_skills:
                d = self.skill_matcher.skill_distance(j_uri, r_uri)
                if d is not None:
                    if d == 0:
                        best_credit = 1.0
                        break
                    elif d <= 2 and best_credit < 0.5:
                        best_credit = 0.5

            total_credit += w * best_credit
            total_weight += w

        return total_credit / total_weight if total_weight > 0 else 0.0

    def _get_effective_lambda1(self) -> float:
        """Get λ₁ with optional adaptive annealing: λ₁(t) = λ₁ · (1 − rate · t/T)."""
        if not self.margin_anneal:
            return self.ordinal_lambda1
        epoch_ratio = self.current_epoch / max(1, self.total_epochs)
        decay = 1.0 - self.margin_anneal_rate * epoch_ratio
        return self.ordinal_lambda1 * max(0.0, decay)

    def _compute_phi_on_the_fly(self, resume_skill_uris: List[str],
                                job_skill_uris: List[str]) -> Optional[float]:
        """
        Compute φ(resume, job) on the fly using Strategy D.

        This computes skill coverage of resume against a SPECIFIC job,
        fixing the bug where precomputed φ used the candidate's own job
        instead of the anchor's job.

        Strategy D (tight + reduced credit):
          - Exact URI match:     credit = 1.0
          - Within 2 graph hops: credit = 0.5
          - No match:            credit = 0.0

        Args:
            resume_skill_uris: Skill URIs from the candidate's resume
            job_skill_uris: Skill URIs from the TARGET job (anchor's job)

        Returns:
            φ score (0.0 to 1.0), or None if job has no skill URIs
        """
        if not self.skill_matcher:
            return None

        job_skills = list(set(job_skill_uris))
        resume_skills = list(set(resume_skill_uris))

        if not job_skills:
            return None
        if not resume_skills:
            return 0.0

        total_credit = 0.0
        for j_uri in job_skills:
            best_credit = 0.0
            for r_uri in resume_skills:
                d = self.skill_matcher.skill_distance(j_uri, r_uri)
                if d is not None:
                    if d == 0:
                        best_credit = 1.0
                        break  # exact match, no need to check further
                    elif d <= 2 and best_credit < 0.5:
                        best_credit = 0.5
            total_credit += best_credit

        return total_credit / len(job_skills)

    def _compute_ordinal_loss(self, triplets: List[ContrastiveTriplet],
                               embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute ordinal contrastive loss (OCL) = InfoNCE + ordinal margins.

        L₁ = per-triplet InfoNCE (existing pipeline with 7 ontology-selected negatives)
        L₂ = max(0, m₁(φₚ) − (sᵅ − sₚ))  — good_fit above potential_fit (φ-guided)
        L₃ = max(0, m₂ − (sₚ − sₙ))       — potential_fit above no_fit (fixed margin)

        Curriculum: easy phase (L₁ + λ₂·L₃), full phase (L₁ + λ₁·L₂ + λ₂·L₃).
        Strategy B: in-batch selection for cₚ (hardest potential_fit) and cₙ (clearest no_fit).

        Args:
            triplets: List of ContrastiveTriplet objects
            embeddings: Dictionary mapping content to embeddings

        Returns:
            torch.Tensor: Combined ordinal loss
        """
        # ── L₁: Use existing per-triplet InfoNCE (proven, 7 ontology-selected negatives) ──
        infonce_losses = []
        for triplet in triplets:
            try:
                tl = self._compute_triplet_loss(triplet, embeddings)
                if tl is not None:
                    infonce_losses.append(tl)
            except Exception:
                continue

        if not infonce_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        l1_mean = torch.stack(infonce_losses).mean()

        # ── Determine curriculum phase ──
        epoch_ratio = self.current_epoch / max(1, self.total_epochs)
        is_full_phase = epoch_ratio >= self.ordinal_curriculum_switch

        # ── Collect batch items for ordinal margin computation (L₂, L₃) ──
        # Include skill_uris so we can compute φ(resume_p, job_α) on the fly
        batch_items = []
        for triplet in triplets:
            job_key = self._get_content_key(triplet.positive)
            resume_key = self._get_content_key(triplet.anchor)
            if job_key not in embeddings or resume_key not in embeddings:
                continue

            label = triplet.view_metadata.get('positive_original_label', 'good_fit')
            phi = triplet.view_metadata.get('phi')
            tier = triplet.view_metadata.get('quality_tier', 'F')

            batch_items.append({
                'job_emb': embeddings[job_key].to(self.device),
                'resume_emb': embeddings[resume_key].to(self.device),
                'label': label,
                'phi': phi if phi is not None else 0.5,
                'tier': tier,
                # Skill URIs for on-the-fly φ computation
                'resume_skill_uris': triplet.anchor.get('skill_uris', []),
                'job_skill_uris': triplet.positive.get('skill_uris', []),
                'job_occupation_uri': triplet.positive.get('occupation_uri'),
            })

            # Also collect negatives with their labels
            neg_labels = triplet.view_metadata.get('negative_original_labels', [])
            for i, negative in enumerate(triplet.negatives):
                neg_key = self._get_content_key(negative)
                if neg_key not in embeddings:
                    continue
                neg_label = neg_labels[i] if i < len(neg_labels) else 'no_fit'
                batch_items.append({
                    'job_emb': embeddings[neg_key].to(self.device),
                    'resume_emb': embeddings[resume_key].to(self.device),
                    'label': neg_label,
                    'phi': None,
                    'tier': tier,
                    # For negatives: resume is the anchor's resume, job is the negative's job
                    'resume_skill_uris': triplet.anchor.get('skill_uris', []),
                    'job_skill_uris': negative.get('skill_uris', []),
                    'job_occupation_uri': negative.get('occupation_uri'),
                })

        # ── Group by label for Strategy B tuple selection ──
        good_fit_items = [it for it in batch_items if it['label'] == 'good_fit']
        potential_fit_items = [it for it in batch_items if it['label'] == 'potential_fit']
        no_fit_items = [it for it in batch_items if it['label'] == 'no_fit']

        # If no good_fit items for ordinal margins, return just L₁
        if not good_fit_items:
            return l1_mean

        # ── Quality tier weights ──
        tier_weights = {'A': 1.0, 'B': 0.9, 'C': 0.75, 'D': 0.6, 'F': 0.5}

        # ── Compute L₂ and L₃ margins (Strategy B: in-batch selection) ──
        margin_losses = []

        for gf in good_fit_items:
            w_q = tier_weights.get(gf['tier'], 0.5)
            s_alpha = torch.sum(gf['job_emb'] * gf['resume_emb'], dim=-1)

            # ── Select cₙ: no_fit whose job is LEAST similar to j (clearest negative) ──
            best_nf = None
            best_nf_job_sim = float('inf')
            for nf in no_fit_items:
                job_sim = torch.sum(gf['job_emb'] * nf['job_emb'], dim=-1).item()
                if job_sim < best_nf_job_sim:
                    best_nf_job_sim = job_sim
                    best_nf = nf

            # ── Select cₚ: potential_fit whose job is MOST similar to j (hardest) ──
            best_pf = None
            best_pf_job_sim = float('-inf')
            if is_full_phase:
                for pf in potential_fit_items:
                    job_sim = torch.sum(gf['job_emb'] * pf['job_emb'], dim=-1).item()
                    if job_sim > best_pf_job_sim:
                        best_pf_job_sim = job_sim
                        best_pf = pf

            # ── L₃ (active in BOTH phases) ──
            l3_term = torch.tensor(0.0, device=self.device)
            if best_nf is not None:
                s_n = torch.sum(gf['job_emb'] * best_nf['resume_emb'], dim=-1)
                if best_pf is not None and is_full_phase:
                    s_p = torch.sum(gf['job_emb'] * best_pf['resume_emb'], dim=-1)
                    l3_term = F.relu(self.ordinal_m2 - (s_p - s_n))
                else:
                    l3_term = F.relu(self.ordinal_m2 - (s_alpha - s_n))

            # ── L₂ (only in full phase) ──
            l2_term = torch.tensor(0.0, device=self.device)
            if is_full_phase and best_pf is not None:
                s_p = torch.sum(gf['job_emb'] * best_pf['resume_emb'], dim=-1)
                if self.ordinal_fixed_m1:
                    # Ablation: fixed margin (same as m₂), no φ dependency
                    m1 = self.ordinal_m2
                else:
                    # φ-guided margin: m₁ = α·(1 − φ(resume_p, job_α))
                    # FIXED: compute φ against the ANCHOR's job, not the pf candidate's own job
                    phi_p = None
                    if self.skill_matcher:
                        if self.phi_use_weighted:
                            # Improvement 1: weighted φ (essential > optional)
                            job_occ_uri = gf.get('job_occupation_uri')
                            phi_p = self._compute_phi_weighted(
                                best_pf['resume_skill_uris'],
                                gf['job_skill_uris'],
                                job_occupation_uri=job_occ_uri,
                            )
                        else:
                            phi_p = self._compute_phi_on_the_fly(
                                best_pf['resume_skill_uris'],
                                gf['job_skill_uris'],
                            )
                    if phi_p is None:
                        # Fallback to precomputed φ (old behavior, against own job)
                        phi_p = best_pf['phi'] if best_pf['phi'] is not None else 0.5

                    # Improvement 3: confidence gating — skip φ-derived L₂ when φ is high
                    if phi_p >= self.phi_gate_threshold:
                        # φ too high → margin unreliable, fall back to fixed margin
                        m1 = self.ordinal_m2
                    else:
                        m1 = self.ordinal_alpha * (1.0 - phi_p)
                l2_term = F.relu(m1 - (s_alpha - s_p))

            # ── Combine margins for this tuple ──
            # Improvement 2: adaptive margin annealing for λ₁
            effective_lambda1 = self._get_effective_lambda1()
            if is_full_phase:
                margin = effective_lambda1 * l2_term + self.ordinal_lambda2 * l3_term
            else:
                margin = self.ordinal_lambda2 * l3_term

            margin_losses.append(w_q * margin)

        # ── Final loss: L₁ (InfoNCE) + ordinal margins ──
        ordinal_margin = torch.stack(margin_losses).mean()
        return l1_mean + ordinal_margin

    def _compute_hybrid_loss(self, triplets: List[ContrastiveTriplet],
                              embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute hybrid InfoNCE + WS2 loss.

        Combines the contrastive separation signal from InfoNCE with the
        ordinal awareness from 2-Wasserstein distance:
            loss = infonce_loss + ws2_weight * ws2_loss

        InfoNCE ensures positive/negative separation in the embedding space.
        WS2 adds graduated relevance awareness (good_fit > potential_fit > no_fit).

        Args:
            triplets: List of ContrastiveTriplet objects
            embeddings: Dictionary mapping content keys to embeddings

        Returns:
            torch.Tensor: Combined hybrid loss
        """
        # Compute InfoNCE component (per-triplet, averaged)
        infonce_losses = []
        for triplet in triplets:
            try:
                triplet_loss = self._compute_triplet_loss(triplet, embeddings)
                if triplet_loss is not None:
                    infonce_losses.append(triplet_loss)
            except Exception as e:
                logger.warning(f"Failed InfoNCE for triplet: {e}")
                continue

        if not infonce_losses:
            logger.warning("No valid InfoNCE losses, falling back to WS2 only")
            return self._compute_batch_wasserstein_loss(triplets, embeddings)

        infonce_loss = torch.stack(infonce_losses).mean()

        # Compute WS2 component (batch-level)
        try:
            ws2_loss = self._compute_batch_wasserstein_loss(triplets, embeddings)
        except Exception as e:
            logger.warning(f"WS2 computation failed, using InfoNCE only: {e}")
            return infonce_loss

        if torch.isnan(ws2_loss) or torch.isinf(ws2_loss):
            logger.warning("Invalid WS2 loss, using InfoNCE only")
            return infonce_loss

        # Combine: InfoNCE for separation + WS2 for ordinal awareness
        hybrid_loss = infonce_loss + self.ws2_weight * ws2_loss

        return hybrid_loss

    def _compute_batch_wasserstein_loss(self, triplets: List[ContrastiveTriplet],
                                         embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute 2-Wasserstein loss at batch level following SyCL (2503.23239).

        Collects all triplets into a batch logits matrix [batch_size, num_items]
        and a corresponding label matrix, then computes the 2-Wasserstein distance
        between the predicted and target distributions across the entire batch.

        The batch-level computation captures cross-sample covariance structure
        that per-triplet computation misses.

        Args:
            triplets: List of ContrastiveTriplet objects
            embeddings: Dictionary mapping content keys to embeddings

        Returns:
            torch.Tensor: Batch-level 2-Wasserstein loss
        """
        logit_rows = []
        label_rows = []
        ont_weights = []

        for triplet in triplets:
            try:
                anchor_key = self._get_content_key(triplet.anchor)
                positive_key = self._get_content_key(triplet.positive)

                if anchor_key not in embeddings or positive_key not in embeddings:
                    continue

                anchor_emb = embeddings[anchor_key]
                positive_emb = embeddings[positive_key]

                neg_embs = []
                neg_labels_list = triplet.view_metadata.get('negative_original_labels', [])
                valid_neg_labels = []

                for i, negative in enumerate(triplet.negatives):
                    neg_key = self._get_content_key(negative)
                    if neg_key in embeddings:
                        neg_embs.append(embeddings[neg_key])
                        label = neg_labels_list[i] if i < len(neg_labels_list) else 'no_fit'
                        valid_neg_labels.append(label)

                if not neg_embs:
                    continue

                # Compute similarity logits: [pos_sim, neg_0_sim, neg_1_sim, ...]
                pos_sim = torch.sum(anchor_emb * positive_emb, dim=-1) / self.temperature
                neg_sims = [torch.sum(anchor_emb * neg, dim=-1) / self.temperature for neg in neg_embs]
                row_logits = torch.stack([pos_sim] + neg_sims)
                row_logits = torch.clamp(row_logits, max=self.max_exp)

                # Build target relevance row
                pos_label = triplet.view_metadata.get('positive_original_label', 'good_fit')
                pos_score = self.relevance_scores.get(pos_label, 3.0)
                neg_scores = [self.relevance_scores.get(l, 0.0) for l in valid_neg_labels]
                row_labels = torch.tensor([pos_score] + neg_scores, dtype=torch.float64, device=self.device)

                logit_rows.append(row_logits)
                label_rows.append(row_labels)
                ont_weights.append(self._compute_ontology_weight(triplet.view_metadata))

            except Exception as e:
                logger.warning(f"Failed to process triplet for batch WS2: {e}")
                continue

        if not logit_rows:
            logger.warning("No valid triplets for batch Wasserstein loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Pad rows to same length (triplets may have different negative counts)
        max_len = max(row.shape[0] for row in logit_rows)

        padded_logits = []
        padded_labels = []
        for logits_row, labels_row in zip(logit_rows, label_rows):
            pad_size = max_len - logits_row.shape[0]
            if pad_size > 0:
                # Pad with large negative logit (will get ~0 probability after softmax)
                logits_row = torch.cat([logits_row, torch.full((pad_size,), -50.0, device=self.device)])
                # Pad labels with 0 (no_fit)
                labels_row = torch.cat([labels_row, torch.zeros(pad_size, dtype=torch.float64, device=self.device)])
            padded_logits.append(logits_row)
            padded_labels.append(labels_row)

        # Stack into batch matrices [batch_size, num_items]
        logits_matrix = torch.stack(padded_logits)
        labels_matrix = torch.stack(padded_labels)

        # Apply ontology weights: scale each row's logits by its weight
        # This integrates sample-level quality weighting into the batch loss
        weight_tensor = torch.tensor(ont_weights, dtype=torch.float32, device=self.device)
        avg_weight = weight_tensor.mean()

        # Compute softmax distributions (following SyCL exactly)
        # Note: logits already divided by temperature during per-triplet computation,
        # so we don't divide again here (SyCL divides once in forward())
        preds = F.softmax(logits_matrix.double(), dim=1)
        targets = F.softmax(labels_matrix, dim=1)

        # Compute 2-Wasserstein distance between batch distributions
        ws2_loss = self._ws2_distance(preds, targets)

        # Scale by average ontology weight
        ws2_loss = ws2_loss.float() * avg_weight

        if torch.isnan(ws2_loss) or torch.isinf(ws2_loss):
            logger.warning("NaN/Inf in batch WS2 loss, falling back to per-triplet")
            # Fall back to per-triplet 1-Wasserstein
            losses = []
            for triplet in triplets:
                try:
                    loss = self._compute_triplet_loss(triplet, embeddings)
                    if loss is not None:
                        losses.append(loss)
                except Exception:
                    continue
            if losses:
                return torch.stack(losses).mean()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return ws2_loss

    def _ws2_distance(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute 2-Wasserstein distance between two batch distributions.

        Follows the SyCL implementation: treats rows as samples from multivariate
        Gaussian distributions and computes the closed-form W2 distance using
        mean difference and covariance trace terms.

        d(P_X, P_Y) = |mu_X - mu_Y|^2 + Tr(cov_X + cov_Y - 2*(cov_X @ cov_Y)^{1/2})

        Fast eigenvalue method from: https://arxiv.org/pdf/2009.14075.pdf

        Args:
            X: Predicted distributions [batch_size, num_items] (float64)
            Y: Target distributions [batch_size, num_items] (float64)

        Returns:
            torch.Tensor: 2-Wasserstein distance (scalar, float64)
        """
        import torch.linalg as linalg
        import math

        # Transpose to [num_items, batch_size] for covariance computation
        X = X.transpose(0, 1).double()
        Y = Y.transpose(0, 1).double()

        mu_X = torch.mean(X, dim=1, keepdim=True)
        mu_Y = torch.mean(Y, dim=1, keepdim=True)

        _, b = X.shape
        fact = 1.0 if b < 2 else 1.0 / (b - 1)
        fact_sqrt = math.sqrt(fact)

        # Covariance matrices
        E_X = X - mu_X
        E_Y = Y - mu_Y
        cov_X = torch.matmul(E_X, E_X.t()) * fact
        cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

        # Tr((cov_X @ cov_Y)^{1/2}) via eigenvalue decomposition
        C_X = E_X * fact_sqrt
        C_Y = E_Y * fact_sqrt
        M_l = torch.matmul(C_X.t(), C_Y)
        M_r = torch.matmul(C_Y.t(), C_X)
        M = torch.matmul(M_l, M_r)
        S = linalg.eigvals(M) + 1e-15
        sq_tr_cov = S.sqrt().abs().sum()

        trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov

        # Mean difference term
        diff = mu_X - mu_Y
        mean_term = torch.sum(torch.mul(diff, diff))

        return (trace_term + mean_term).real

    def compute_view_combinations_loss(self, triplets: List[ContrastiveTriplet],
                                       resume_embeddings: Dict[str, torch.Tensor],
                                       job_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss across multiple view combinations.

        This method handles cases where we have multiple views of resumes and jobs,
        computing loss across all valid combinations as specified in Requirement 4.2.

        Args:
            triplets: List of ContrastiveTriplet objects with view metadata
            resume_embeddings: Embeddings for different resume views
            job_embeddings: Embeddings for different job views

        Returns:
            torch.Tensor: Combined loss across all view combinations
        """
        if not triplets:
            raise ValueError("Cannot compute loss for empty triplets list")

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_combinations = 0

        for triplet in triplets:
            # Get view combinations from metadata
            view_combinations = triplet.view_metadata.get(
                'view_combinations', [])

            if not view_combinations:
                # Fallback to original embeddings
                all_embeddings = {**resume_embeddings, **job_embeddings}
                triplet_loss = self._compute_triplet_loss(
                    triplet, all_embeddings)
                if triplet_loss is not None:
                    total_loss = total_loss + triplet_loss
                    total_combinations += 1
            else:
                # Compute loss for each view combination
                for combo in view_combinations:
                    try:
                        combo_loss = self._compute_view_combination_loss(
                            triplet, combo, resume_embeddings, job_embeddings
                        )
                        if combo_loss is not None:
                            total_loss = total_loss + combo_loss
                            total_combinations += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute loss for view combination: {e}")
                        continue

        if total_combinations == 0:
            logger.warning(
                "No valid view combinations found for loss computation")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Average loss across all combinations
        avg_loss = total_loss / total_combinations

        return avg_loss

    def _compute_view_combination_loss(self, triplet: ContrastiveTriplet,
                                       view_combination: Dict[str, Any],
                                       resume_embeddings: Dict[str, torch.Tensor],
                                       job_embeddings: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute loss for a specific view combination.

        Args:
            triplet: ContrastiveTriplet object
            view_combination: Dictionary specifying which views to use
            resume_embeddings: Resume view embeddings
            job_embeddings: Job view embeddings

        Returns:
            torch.Tensor: Loss for this view combination, or None if computation fails
        """
        try:
            # Extract view keys from combination
            resume_view_key = view_combination.get('resume_view_key')
            positive_view_key = view_combination.get('positive_view_key')
            negative_view_keys = view_combination.get('negative_view_keys', [])

            # Get embeddings
            if resume_view_key not in resume_embeddings:
                return None
            if positive_view_key not in job_embeddings:
                return None

            anchor_emb = resume_embeddings[resume_view_key]
            positive_emb = job_embeddings[positive_view_key]

            # Get negative embeddings
            negative_embs = []

            for i, neg_key in enumerate(negative_view_keys):
                if neg_key in job_embeddings and i < len(triplet.career_distances):
                    negative_embs.append(job_embeddings[neg_key])

            if not negative_embs:
                return None

            return self._infonce_loss(
                anchor_emb, positive_emb, negative_embs
            )

        except Exception as e:
            logger.error(f"Error computing view combination loss: {e}")
            return None

    def _get_content_key(self, content: Dict[str, Any]) -> str:
        """
        Generate a consistent key for content to look up embeddings.
        
        Uses the same key generation method as EmbeddingCache for consistency.

        Args:
            content: Resume or job content dictionary

        Returns:
            str: Key for embedding lookup
        """
        # Use the same key generation as EmbeddingCache for consistency
        import hashlib
        import json
        try:
            # Create a normalized representation for consistent hashing
            normalized_content = self._normalize_content_for_hashing(content)
            content_str = json.dumps(normalized_content, sort_keys=True, default=str)
            
            # Use SHA-256 for better collision resistance than hash()
            return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:16]
        except (TypeError, ValueError) as e:
            logger.warning(f"Error creating content key: {e}")
            # Fallback to string representation
            content_str = str(sorted(content.items()))
            return hashlib.md5(content_str.encode('utf-8')).hexdigest()[:16]
    
    def _normalize_content_for_hashing(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize content for consistent hashing by removing non-essential fields.
        
        Args:
            content: Original content dictionary
            
        Returns:
            Dict: Normalized content for hashing
        """
        # Create a copy and remove metadata that doesn't affect embeddings
        normalized = {}
        
        # Essential fields that affect text encoding
        essential_fields = {
            'experience', 'role', 'experience_level', 'skills', 'keywords',
            'title', 'jobtitle', 'description', 'jobdescription'
        }
        
        for key, value in content.items():
            if key in essential_fields:
                if isinstance(value, (dict, list)):
                    # Recursively normalize nested structures
                    normalized[key] = self._normalize_nested_structure(value)
                else:
                    normalized[key] = value
        
        return normalized
    
    def _normalize_nested_structure(self, obj: Any) -> Any:
        """Recursively normalize nested dictionaries and lists."""
        if isinstance(obj, dict):
            return {k: self._normalize_nested_structure(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_nested_structure(item) for item in obj]
        else:
            return obj

    def validate_gradients(self, loss: torch.Tensor, parameters: List[torch.Tensor] = None) -> bool:
        """
        Validate that gradients are well-behaved (no NaN or infinite values).

        Args:
            loss: Loss tensor to validate
            parameters: List of parameters to check gradients for (optional)

        Returns:
            bool: True if gradients are valid, False otherwise
        """
        if not loss.requires_grad:
            return True

        try:
            # For testing purposes, just check if the loss tensor itself is valid
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Invalid loss value detected")
                return False

            # If parameters are provided, check their gradients
            if parameters:
                # Compute gradients
                loss.backward(retain_graph=True)

                for param in parameters:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.warning(
                                "Invalid gradients detected in parameters")
                            return False

            return True

        except Exception as e:
            logger.error(f"Error validating gradients: {e}")
            return False

    def get_loss_components(self, triplets: List[ContrastiveTriplet],
                            embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Get detailed loss components for analysis and debugging.

        Args:
            triplets: List of ContrastiveTriplet objects
            embeddings: Dictionary mapping content to embeddings

        Returns:
            Dict[str, float]: Dictionary containing loss components and statistics
        """
        components = {
            'total_loss': 0.0,
            'avg_positive_similarity': 0.0,
            'avg_negative_similarity': 0.0,
            'avg_ontology_weight': 0.0,
            'valid_triplets': 0,
            'failed_triplets': 0
        }

        total_pos_sim = 0.0
        total_neg_sim = 0.0
        total_ontology_weight = 0.0
        total_negatives = 0

        for triplet in triplets:
            try:
                # Get embeddings
                anchor_key = self._get_content_key(triplet.anchor)
                positive_key = self._get_content_key(triplet.positive)

                if anchor_key not in embeddings or positive_key not in embeddings:
                    components['failed_triplets'] += 1
                    continue

                # Embeddings are already normalized by the model
                anchor_emb = embeddings[anchor_key]
                positive_emb = embeddings[positive_key]

                # Compute positive similarity
                pos_sim = torch.sum(anchor_emb * positive_emb, dim=-1).item()
                total_pos_sim += pos_sim

                # Track ontology weight
                ontology_weight = self._compute_ontology_weight(triplet.view_metadata)
                total_ontology_weight += ontology_weight

                # Compute negative similarities
                for i, negative in enumerate(triplet.negatives):
                    negative_key = self._get_content_key(negative)
                    if negative_key in embeddings:
                        neg_emb = embeddings[negative_key]
                        neg_sim = torch.sum(
                            anchor_emb * neg_emb, dim=-1).item()
                        total_neg_sim += neg_sim
                        total_negatives += 1

                components['valid_triplets'] += 1

            except Exception as e:
                logger.warning(f"Error analyzing triplet: {e}")
                components['failed_triplets'] += 1

        # Compute averages
        if components['valid_triplets'] > 0:
            components['avg_positive_similarity'] = total_pos_sim / \
                components['valid_triplets']
            components['avg_ontology_weight'] = total_ontology_weight / \
                components['valid_triplets']

        if total_negatives > 0:
            components['avg_negative_similarity'] = total_neg_sim / \
                total_negatives

        # Compute total loss
        total_loss = self.compute_loss(triplets, embeddings)
        components['total_loss'] = total_loss.item(
        ) if total_loss.requires_grad else 0.0

        return components

    def clip_gradients(self, parameters: List[torch.Tensor]) -> None:
        """
        Apply gradient clipping to model parameters.

        Args:
            parameters: List of model parameters to clip gradients for
        """
        if parameters:
            torch.nn.utils.clip_grad_value_(
                parameters, self.gradient_clip_value)
