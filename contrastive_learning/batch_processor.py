"""
BatchProcessor for contrastive triplet generation.

This module implements the BatchProcessor class that converts batches of training samples
into contrastive triplets for training. It supports configurable negative sampling ratios
and in-batch negative sampling for computational efficiency.
"""

import random
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging

from .data_structures import TrainingSample, ContrastiveTriplet, TrainingConfig

if TYPE_CHECKING:
    from .career_graph import CareerGraph
    # CVE-domain tiered negative selector. Imported only for typing so the core
    # never takes a hard runtime dependency on the cve_domain package (the seam
    # is reached through injection / the Ontology_Adapter interface).
    from cve_domain.negative_selector import CVENegativeSelector

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes batches of training samples to create contrastive triplets.

    The BatchProcessor converts each positive training sample into a contrastive triplet
    where the resume serves as the anchor, the matched job as the positive, and other
    jobs from the same batch serve as negatives. This in-batch negative sampling
    strategy provides computational efficiency while maintaining training effectiveness.
    """

    def __init__(self, config: TrainingConfig, career_graph: Optional['CareerGraph'] = None,
                 esco_graph_path: Optional[str] = None,
                 cve_negative_selector: Optional['CVENegativeSelector'] = None,
                 cve_view_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
                 cve_present_ids: Optional[set] = None):
        """
        Initialize the BatchProcessor with training configuration.

        Args:
            config: TrainingConfig containing negative sampling parameters
            career_graph: Optional CareerGraph for pathway-aware negative sampling
            esco_graph_path: Path to ESCO graph file (required if career_graph is None and use_pathway_negatives=True)
            cve_negative_selector: Optional CVE-domain tiered negative selector
                (``cve_domain.negative_selector.CVENegativeSelector``). When
                provided, the negative-selection point routes through it to supply
                ontology-tiered CVE negatives. When ``None`` (the default), the
                career negative-selection path is used unchanged. This is the
                Ontology_Adapter seam for the CVE domain (Req 5.3) and is purely
                additive — the loss engine is not touched.
            cve_view_lookup: Optional mapping of ``cve`` id -> its
                ``CVE_View_Record`` (carrying ``encoder_view``), used to materialize
                the selected negative cve ids into negative "job" slots the loss
                engine already consumes.
            cve_present_ids: Optional set of cve ids present among the converted
                records (the validity universe for pooled negatives). Defaults to
                the keys of ``cve_view_lookup`` when omitted.
        """
        self.config = config
        self.negative_sampling_ratio = config.negative_sampling_ratio
        self.use_pathway_negatives = config.use_pathway_negatives
        self.max_negatives_per_anchor = config.max_negatives_per_anchor

        if not self.use_pathway_negatives and self.config.pathway_weight != 0:
            logger.info("Pathway negatives disabled; setting pathway_weight to 0.0.")
            self.config.pathway_weight = 0.0

        # Initialize or create CareerGraph with ESCO graph
        if career_graph is not None:
            self.career_graph = career_graph
        elif self.use_pathway_negatives:
            if not esco_graph_path:
                raise ValueError(
                    "esco_graph_path is required when use_pathway_negatives=True and no CareerGraph is provided")
            # Create CareerGraph with ESCO graph path
            self.career_graph = self.create_career_graph(
                config, esco_graph_path)
        else:
            self.career_graph = None

        # Initialize OntologySkillMatcher for skill-level negative selection
        self.skill_matcher = None
        use_ontology = getattr(config, 'ontology_weight', 0.0) > 0.0
        if use_ontology and esco_graph_path:
            try:
                from .ontology_skill_matcher import OntologySkillMatcher
                # Use the full ESCO KG for skill matching (not the career graph)
                # Fall back to esco_graph_path if no separate KG path configured
                kg_path = getattr(config, 'esco_kg_path', None) or esco_graph_path
                # Build reuse weights if configured
                reuse_weights = None
                skills_path = None
                if getattr(config, 'use_reuse_weighting', False):
                    skills_path = getattr(config, 'esco_skills_path', None)
                    if skills_path:
                        reuse_weights = {
                            'transversal': getattr(config, 'reuse_weight_transversal', 0.3),
                            'cross-sector': getattr(config, 'reuse_weight_cross_sector', 0.6),
                            'sector-specific': getattr(config, 'reuse_weight_sector_specific', 1.0),
                        }
                self.skill_matcher = OntologySkillMatcher(
                    kg_path, esco_skills_path=skills_path, reuse_weights=reuse_weights)
                # Load transversal skills mask if available
                transversal_path = "dataset/esco/transversalSkillsCollection_en.csv"
                import os
                if os.path.exists(transversal_path) and getattr(config, 'use_reuse_weighting', False):
                    self.skill_matcher.load_transversal_skills(transversal_path, weight=0.3)
                # Load precomputed skill distances if available
                dist_cache_path = "embedding_cache/skill_distances.pkl"
                if os.path.exists(dist_cache_path):
                    self.skill_matcher.load_precomputed_distances(dist_cache_path)
                logger.info("OntologySkillMatcher enabled for skill-level negative selection")
            except Exception as e:
                logger.warning(f"Failed to initialize OntologySkillMatcher: {e}")

        # Set random seed for reproducible negative sampling
        random.seed(42)

        # Curriculum learning state for ontology negative selection
        self.current_epoch = 0
        self.total_epochs = config.num_epochs

        # ISCO group distance for negative selection
        self.use_isco_negatives = getattr(config, 'use_isco_negatives', False)
        self.isco_weight = getattr(config, 'isco_weight', 0.4)
        self.occ_to_isco = {}
        if self.use_isco_negatives:
            esco_occ_path = getattr(config, 'esco_occupations_path', None)
            if esco_occ_path:
                self._load_isco_codes(esco_occ_path)
            else:
                logger.warning("use_isco_negatives=True but no esco_occupations_path")
                self.use_isco_negatives = False

        logger.info(f"BatchProcessor initialized with pathway_negatives={'enabled' if self.career_graph else 'disabled'}, "
                    f"skill_matcher={'enabled' if self.skill_matcher else 'disabled'}, "
                    f"hard_distance_threshold={config.hard_negative_max_distance}, "
                    f"medium_distance_threshold={config.medium_negative_max_distance}")

        # ConFit-inspired: rejection-based hard negatives and in-batch negatives
        self.use_in_batch_negatives = getattr(config, 'use_in_batch_negatives', False)
        self.use_rejection_hard_negatives = getattr(config, 'use_rejection_hard_negatives', False)
        self.rejection_hard_neg_count = getattr(config, 'rejection_hard_neg_count', 4)
        self.rejection_index = {}  # job_id -> list of rejected resume dicts

        # ── CVE-domain tiered negative selection (additive; Req 5.3) ──
        # When a CVENegativeSelector is injected (career default: None), the
        # negative-selection point routes through it instead of the career logic.
        # This keeps the career path byte-identical whenever no selector is set.
        self.cve_negative_selector = cve_negative_selector
        self.cve_view_lookup: Dict[str, Dict[str, Any]] = dict(cve_view_lookup or {})
        self._cve_split_ids: List[str] = list(self.cve_view_lookup.keys())
        self._cve_present_ids: set = (
            set(cve_present_ids) if cve_present_ids is not None
            else set(self._cve_split_ids)
        )
        if self.cve_negative_selector is not None:
            logger.info(
                "BatchProcessor: CVE tiered negative selector enabled "
                "(%d view records indexed)", len(self.cve_view_lookup))

        # ── ORCA negative-selector injection seam (additive; no-op default) ──
        # An external ORCA phase orchestrator (living in the isolated top-level
        # ``orca/`` package) may inject a duck-typed negative selector during
        # ORCA Phase 4 via ``set_negative_selector``. This module never imports
        # ``orca`` (Isolation constraint, Requirement 7.6); the selector arrives
        # by injection. When ``None`` (the default, and the only state on the
        # career / ORCA-Denominator paths) the negative-selection point is
        # byte-identical to the pre-ORCA OSCAR bucket logic (Requirements 5.7,
        # 7.5).
        self.negative_selector = None

    def set_cve_negative_selector(
        self,
        selector: 'CVENegativeSelector',
        view_lookup: Dict[str, Dict[str, Any]],
        present_ids: Optional[set] = None,
    ) -> None:
        """Inject the CVE tiered negative selector and its view-record lookup.

        This is the setter counterpart to the constructor params, letting callers
        wire the CVE Ontology_Adapter seam after construction. Purely additive:
        the career path is unaffected until a selector is set.

        Args:
            selector: The ``CVENegativeSelector`` supplying tiered CVE negatives.
            view_lookup: Mapping of ``cve`` id -> its ``CVE_View_Record`` (carrying
                ``encoder_view``), used to materialize selected negative ids into
                negative "job" slots.
            present_ids: Optional validity universe for pooled negatives; defaults
                to the keys of ``view_lookup``.
        """
        self.cve_negative_selector = selector
        self.cve_view_lookup = dict(view_lookup or {})
        self._cve_split_ids = list(self.cve_view_lookup.keys())
        self._cve_present_ids = (
            set(present_ids) if present_ids is not None
            else set(self._cve_split_ids)
        )
        logger.info(
            "BatchProcessor: CVE tiered negative selector set "
            "(%d view records indexed)", len(self.cve_view_lookup))

    def set_negative_selector(self, selector) -> None:
        """Inject (or clear) a duck-typed ORCA negative selector (ORCA seam).

        This is the batch-processor end of the ORCA negative-selector injection
        point (design integration seams). The trainer's ``set_negative_selector``
        forwards a duck-typed selector here during ORCA Phase 4; passing ``None``
        clears it and restores the OSCAR bucket selection path.

        Purely additive: while ``selector`` is ``None`` (the default, and the
        only state on the career and ORCA-Denominator MVP paths) negative
        selection is byte-identical to the pre-ORCA OSCAR bucket logic
        (Requirements 5.7, 7.5). This module never imports ``orca`` — the
        selector is duck-typed and consulted only when set (Requirement 7.6).

        Args:
            selector: A duck-typed negative selector, or ``None`` to clear. When
                set, it is consulted at the negative-selection point only if it
                exposes a compatible batch-level selection method; otherwise the
                OSCAR bucket logic is used as a safe fallback.
        """
        self.negative_selector = selector
        logger.info(
            "BatchProcessor: ORCA negative selector %s",
            "set" if selector is not None else "cleared")

    def build_rejection_index(self, dataset_path: str) -> None:
        """Build index of rejected resumes per job from the training data.
        Also builds reverse index (job_id -> rejected job dicts for resumes that applied)."""
        import json
        from collections import defaultdict
        job_rejections = defaultdict(list)  # job_title -> list of rejected resume dicts
        with open(dataset_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("label") == 0:  # rejected/not_satisfied
                    job_id = d["job"].get("title", "")
                    job_rejections[job_id].append(d["job"])  # Store the job as a negative
        self.rejection_index = dict(job_rejections)
        total = sum(len(v) for v in self.rejection_index.values())
        logger.info(f"Rejection index: {len(self.rejection_index)} jobs, {total} rejected pairs")

    def process_batch(self, batch: List[TrainingSample], global_job_pool: Optional[List[Dict[str, Any]]] = None, global_resume_pool: Optional[List[Dict[str, Any]]] = None) -> List[ContrastiveTriplet]:
        """
        Process a batch of training samples to create contrastive triplets.

        Args:
            batch: List of TrainingSample objects
            global_job_pool: Optional global pool of jobs for negative sampling
            global_resume_pool: Optional global pool of resumes for symmetric loss reverse direction

        Returns:
            List of ContrastiveTriplet objects

        Raises:
            ValueError: If batch is empty or contains insufficient positive samples
        """
        if not batch:
            raise ValueError("Batch cannot be empty")

        # Filter positive samples (these will become anchors)
        positive_samples = [
            sample for sample in batch if sample.label == 'positive']

        if not positive_samples:
            raise ValueError("Batch must contain at least one positive sample")

        logger.debug(
            f"Processing batch with {len(batch)} samples, {len(positive_samples)} positive")

        triplets = []

        # Create triplets for each positive sample
        for anchor_sample in positive_samples:
            try:
                triplet = self._create_triplet(anchor_sample, batch, global_job_pool, global_resume_pool)
                triplets.append(triplet)
            except Exception as e:
                logger.warning(
                    f"Failed to create triplet for sample {anchor_sample.sample_id}: {e}")
                continue

        logger.debug(f"Created {len(triplets)} triplets from batch")
        return triplets

    def _create_triplet(self, anchor_sample: TrainingSample, batch: List[TrainingSample], global_job_pool: Optional[List[Dict[str, Any]]] = None, global_resume_pool: Optional[List[Dict[str, Any]]] = None) -> ContrastiveTriplet:
        """
        Create a contrastive triplet for a single anchor sample.

        Args:
            anchor_sample: The positive sample to use as anchor
            batch: Full batch of samples for negative selection
            global_job_pool: Optional global pool of jobs for negative sampling
            global_resume_pool: Optional global pool of resumes for symmetric loss reverse direction

        Returns:
            ContrastiveTriplet with anchor, positive, and negatives
        """
        # Use resume as anchor and matched job as positive
        anchor = anchor_sample.resume
        positive = anchor_sample.job

        # Select negative jobs from the batch or global pool
        negatives, career_distances = self._select_negatives(
            anchor_sample, batch, global_job_pool)

        # Add rejection-based hard negatives (ConFit-inspired)
        if self.use_rejection_hard_negatives and self.rejection_index:
            job_title = anchor_sample.job.get('title', '')
            rejected_jobs = self.rejection_index.get(job_title, [])
            if rejected_jobs:
                # These are jobs from rejected pairs with the same job title
                # They serve as hard negatives because they're real jobs candidates applied to
                n_hard = min(self.rejection_hard_neg_count, len(rejected_jobs))
                hard_negs = random.sample(rejected_jobs, n_hard)
                negatives.extend(hard_negs)
                career_distances.extend([0.0] * n_hard)

        # Select resume negatives for symmetric loss reverse direction
        resume_negatives = []
        if global_resume_pool and getattr(self.config, 'use_symmetric_loss', False):
            anchor_resume_id = anchor_sample.resume.get('resume_id', anchor_sample.resume.get('name', ''))
            candidates = [r for r in global_resume_pool
                          if r.get('resume_id', r.get('name', '')) != anchor_resume_id]
            n_resume_neg = min(self.max_negatives_per_anchor, len(candidates))
            if candidates and n_resume_neg > 0:
                resume_negatives = random.sample(candidates, n_resume_neg)

        # Determine sampling strategy for metadata
        sampling_strategy = 'global' if global_job_pool else 'in_batch'

        # Create view metadata including ontology scores for loss weighting
        view_metadata = {
            'anchor_id': anchor_sample.sample_id,
            'positive_job_applicant_id': anchor_sample.metadata.get('job_applicant_id', 'unknown'),
            'negative_count': len(negatives),
            'sampling_strategy': sampling_strategy,
            'ontology_similarity': anchor_sample.metadata.get('ontology_similarity'),
            'ot_distance': anchor_sample.metadata.get('ot_distance'),
            'quality_tier': anchor_sample.metadata.get('quality_tier'),
            'phi': anchor_sample.metadata.get('phi'),
            # Graduated relevance labels for Wasserstein loss
            'positive_original_label': anchor_sample.metadata.get('original_label', 'good_fit'),
            'negative_original_labels': [neg.get('original_label', 'no_fit') for neg in negatives],
            # Occupation URIs for ISCO proximity weighting
            'job_occupation_uri': anchor_sample.job.get('occupation_uri', ''),
            'resume_occupation_uri': anchor_sample.metadata.get('resume_occupation_uri', ''),
            # Resume negatives for symmetric loss reverse direction
            'resume_negatives': resume_negatives,
            # Stable resume identity for query-anchored ordinal grouping.
            # Falls back to a content hash if the loader did not inject one.
            'resume_id': anchor_sample.metadata.get('resume_id')
                          or self._compute_resume_id(anchor_sample.resume),
        }

        # ── ORCA per-negative ontology-feature capture (additive; Req 9.1) ──
        # Only runs when ORCA is enabled; leaves the career path byte-identical
        # otherwise. Records the per-negative ontology scalars ORCA's
        # ReliabilityMLP / weak targets need (d_esco, d_isco, d_ot, s_esco,
        # s_isco) so the loss engine sources them from OSCAR's skill matcher
        # rather than proxying a single blended distance.
        if getattr(self.config, 'orca_enabled', False):
            feats = self._compute_negative_ontology_features(
                anchor_sample, negatives)
            if feats is not None:
                view_metadata['negative_ontology_features'] = feats

        return ContrastiveTriplet(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            career_distances=career_distances,
            view_metadata=view_metadata
        )

    def _compute_negative_ontology_features(
        self, anchor_sample: TrainingSample, negatives: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, float]]]:
        """Per-negative ontology scalars for ORCA, aligned 1:1 with ``negatives``.

        Reuses OSCAR's ``OntologySkillMatcher`` (no reimplementation, Req 9.1) to
        produce ``{d_esco, d_isco, d_ot, s_esco, s_isco}`` for each
        (resume, negative_job) pair:

          * ``s_esco`` = ``ontology_set_similarity(resume_uris, job_uris)`` and
            ``d_esco`` = ``1 - s_esco``;
          * ``d_isco`` = ISCO occupation-hierarchy distance (``s_isco = 1 - d_isco``);
          * ``d_ot``  = Sinkhorn optimal-transport distance, computed only when
            ``orca_capture_ot_distance`` is set (it is expensive); otherwise a
            neutral ``0.0``.

        Returns ``None`` when the skill matcher is unavailable or the resume has
        no skill URIs, so the ORCA adapter falls back to its per-negative
        ``career_distances`` proxy. Missing per-pair signals degrade to neutral
        values (Req 9.3) rather than raising.
        """
        matcher = self.skill_matcher
        if matcher is None:
            return None

        resume = anchor_sample.resume
        resume_uris = resume.get('skill_uris', []) if isinstance(resume, dict) else []
        if not resume_uris:
            return None

        resume_occ = anchor_sample.metadata.get('resume_occupation_uri', '') \
            or resume.get('occupation_uri', '')
        capture_ot = getattr(self.config, 'orca_capture_ot_distance', False)

        features: List[Dict[str, float]] = []
        for job in negatives:
            job_uris = job.get('skill_uris', []) if isinstance(job, dict) else []

            # ESCO skill-set similarity → s_esco / d_esco.
            if job_uris:
                try:
                    s_esco = float(matcher.ontology_set_similarity(resume_uris, job_uris))
                except Exception:
                    s_esco = 0.0
            else:
                s_esco = 0.0  # neutral: no ontology signal for this negative
            d_esco = 1.0 - s_esco

            # ISCO occupation-hierarchy distance → d_isco / s_isco.
            job_occ = job.get('occupation_uri', '') if isinstance(job, dict) else ''
            if self.use_isco_negatives and resume_occ and job_occ:
                d_isco = float(self._isco_distance(resume_occ, job_occ))
            else:
                d_isco = 0.5  # neutral when ISCO data is unavailable
            s_isco = 1.0 - d_isco

            # Optimal-transport distance (expensive; opt-in).
            d_ot = 0.0
            if capture_ot and job_uris:
                try:
                    ot = matcher.ot_distance(resume_uris, job_uris)
                    d_ot = float(ot) if ot is not None else 0.0
                except Exception:
                    d_ot = 0.0

            features.append({
                'd_esco': d_esco,
                'd_isco': d_isco,
                'd_ot': d_ot,
                's_esco': s_esco,
                's_isco': s_isco,
            })

        return features

    @staticmethod
    def _compute_resume_id(resume: Dict[str, Any]) -> str:
        """
        Deterministic resume identifier from content (role + first experience +
        sorted skills). Mirrors DataLoader._compute_resume_id so the ordinal loss
        can group graded jobs per query even when the loader did not inject one.
        """
        import hashlib
        role = str(resume.get('role', ''))
        exp = resume.get('experience', [])
        first_desc = ''
        if isinstance(exp, list) and exp:
            first = exp[0]
            first_desc = str(first.get('description', ''))[:300] if isinstance(first, dict) else str(first)[:300]
        skills = resume.get('skills', [])
        skills_str = '|'.join(sorted(map(str, skills))) if isinstance(skills, list) else str(skills)
        basis = f"{role}||{first_desc}||{skills_str}"
        return hashlib.sha256(basis.encode('utf-8')).hexdigest()[:16]

    def _select_cve_negatives(
        self, anchor_sample: TrainingSample
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Select tiered CVE negatives via the injected CVENegativeSelector (Req 5.3).

        The CVE domain packs the anchor CVE into the sample's first view slot
        (``resume``) and the ontology-related positive CVE into the second
        (``job``); ``metadata['resume_id']`` is the anchor cve id. This method
        asks the ``CVENegativeSelector`` for tiered negative cve ids for the
        anchor, then materializes each id into a negative "job" slot
        (``{cve, encoder_view}``) — the same shape as the positive job slot the
        loss engine already consumes. The loss math is untouched: this only
        chooses *which* CVE records become negatives.

        Args:
            anchor_sample: The positive (anchor+positive) CVE TrainingSample.

        Returns:
            Tuple of (negative_job_slots, career_distances). ``career_distances``
            are zeros (tier structure lives in the selector, not the loss).
        """
        # Anchor cve id: prefer the injected group id, fall back to the anchor
        # view slot's cve.
        anchor_cve = anchor_sample.metadata.get('resume_id')
        if not anchor_cve:
            anchor_cve = anchor_sample.resume.get('cve', '')
        anchor_cve = str(anchor_cve).strip()

        neg_ids = self.cve_negative_selector.select_negatives(
            anchor_cve, self._cve_split_ids, self._cve_present_ids)

        negatives: List[Dict[str, Any]] = []
        for cid in neg_ids:
            record = self.cve_view_lookup.get(cid)
            if record is None:
                # Selector already filters against present_ids, but guard anyway.
                continue
            negatives.append({
                'cve': cid,
                'encoder_view': record.get('encoder_view', ''),
            })

        if not negatives:
            logger.warning(
                "No CVE negatives selected for anchor %s; using dummy negative",
                anchor_cve or anchor_sample.sample_id)
            dummy_negative = {
                'cve': 'dummy_negative',
                'encoder_view': 'No Match Available',
            }
            return [dummy_negative], [self.config.medium_negative_max_distance]

        career_distances = [0.0] * len(negatives)
        logger.debug(
            "Selected %d CVE tiered negatives for anchor %s",
            len(negatives), anchor_cve)
        return negatives, career_distances

    def _select_with_injected_selector(
        self,
        anchor_sample: TrainingSample,
        candidate_negatives: List[Dict[str, Any]],
        max_negatives: int,
    ) -> Optional[tuple[List[Dict[str, Any]], List[float]]]:
        """Route negative selection through an injected ORCA selector, if usable.

        The injected selector is duck-typed. It is used only if it exposes a
        batch-level ``select_batch_negatives(anchor_sample, candidate_negatives,
        max_negatives, epoch)`` method returning either a list of selected
        negative job dicts or a ``(negatives, career_distances)`` tuple. Any
        other selector shape (e.g. a purely embedding/tensor-level sampler that
        needs projected embeddings not available at this stage) is not usable
        here, so this returns ``None`` and the caller falls back to the OSCAR
        bucket logic.

        Returning ``None`` on incompatibility (rather than raising) keeps the
        seam purely additive: a set-but-incompatible selector degrades to OSCAR
        behavior instead of breaking negative selection.

        Args:
            anchor_sample: The anchor (positive) sample.
            candidate_negatives: The candidate negative job dicts.
            max_negatives: The maximum number of negatives to select.

        Returns:
            A ``(negatives, career_distances)`` tuple when the selector produced
            a selection, otherwise ``None`` to signal a fallback to OSCAR logic.
        """
        select = getattr(self.negative_selector, "select_batch_negatives", None)
        if not callable(select):
            # Not a batch-dict-level selector (e.g. a tensor-level sampler that
            # operates after embedding). Fall back to OSCAR bucket selection.
            return None

        try:
            result = select(
                anchor_sample, candidate_negatives, max_negatives,
                self.current_epoch)
        except Exception as e:  # pragma: no cover - defensive additive guard
            logger.warning(
                "Injected ORCA negative selector failed (%s); falling back to "
                "OSCAR bucket selection.", e)
            return None

        if result is None:
            return None

        # Accept either a bare list of negatives or a (negatives, distances) pair.
        if isinstance(result, tuple) and len(result) == 2:
            selected, career_distances = result
        else:
            selected = result
            career_distances = [0.0] * len(selected)

        logger.debug(
            "Selected %d negatives via injected ORCA selector", len(selected))
        return list(selected), list(career_distances)

    def _select_negatives(self, anchor_sample: TrainingSample, batch: List[TrainingSample], global_job_pool: Optional[List[Dict[str, Any]]] = None) -> tuple[List[Dict[str, Any]], List[float]]:
        """
        Select negative jobs using either global pool or batch-based sampling.

        This method supports both global and in-batch negative sampling:
        - If global_job_pool is provided: Uses global negative sampling for consistent training
        - If global_job_pool is None: Falls back to in-batch sampling (original behavior)
        
        Also respects the use_pathway_negatives configuration setting:
        - If True: Uses CareerGraph for intelligent career-aware negative selection
        - If False: Uses simple random sampling from available candidates

        Args:
            anchor_sample: The anchor sample (positive)
            batch: Full batch of samples (used for fallback)
            global_job_pool: Optional global pool of jobs for negative sampling

        Returns:
            Tuple of (negative_jobs, career_distances)
        """
        # ── CVE domain: tiered ontology negatives (Req 5.3) ──
        # When a CVENegativeSelector is injected, route negative selection through
        # it. This branch is skipped entirely for the career default (selector is
        # None), so the career negative-selection behavior below is unchanged.
        if self.cve_negative_selector is not None:
            return self._select_cve_negatives(anchor_sample)

        # Choose negative candidate source
        if self.use_in_batch_negatives:
            # ConFit-style: use other jobs in the batch as negatives
            candidate_negatives = []
            anchor_job_id = anchor_sample.job.get('job_id', anchor_sample.job.get('title', ''))
            for sample in batch:
                job = sample.job
                job_id = job.get('job_id', job.get('title', ''))
                if job_id != anchor_job_id:
                    candidate_negatives.append(job)
            logger.debug(f"Using in-batch negatives: {len(candidate_negatives)} candidates")
        elif global_job_pool:
            # Use global negative sampling for consistent training
            candidate_negatives = []
            anchor_job_id = anchor_sample.job.get('job_id', anchor_sample.job.get('title', ''))
            
            for job in global_job_pool:
                job_id = job.get('job_id', job.get('title', ''))
                # Skip the same job as the positive
                if job_id != anchor_job_id:
                    candidate_negatives.append(job)
            
            logger.debug(f"Using global negative sampling with {len(candidate_negatives)} candidates")
        else:
            # Fall back to in-batch negative sampling (original behavior)
            candidate_negatives = []
            anchor_job_applicant_id = anchor_sample.metadata.get('job_applicant_id')

            for sample in batch:
                job = sample.job
                sample_job_applicant_id = sample.metadata.get('job_applicant_id')

                # Skip only samples with same job_applicant_id that are positive labels
                if sample_job_applicant_id == anchor_job_applicant_id and sample.label == 'positive':
                    continue

                candidate_negatives.append(job)
            
            logger.debug(f"Using in-batch negative sampling with {len(candidate_negatives)} candidates")

        if not candidate_negatives:
            # If no other jobs available, create a dummy negative
            logger.warning(
                f"No candidate negatives found for sample {anchor_sample.sample_id}")
            dummy_negative = {
                'job_id': 'dummy_negative',
                'title': 'No Match Available',
                'description': 'Placeholder negative sample',
                'level': 'unknown'
            }
            return [dummy_negative], [self.config.medium_negative_max_distance]

        # Limit number of negatives to prevent memory issues
        max_negatives = min(len(candidate_negatives), self.max_negatives_per_anchor)

        # ── ORCA adaptive negative selection (additive; Req 5.7) ──
        # When an ORCA negative selector is injected (career / ORCA-Denominator
        # default: None), route the final selection through it. This branch is
        # skipped entirely when no selector is set, so the OSCAR bucket logic
        # below stays byte-identical (Requirements 5.7, 7.5). Any incompatible
        # selector falls back safely to the OSCAR logic (``None`` return).
        if self.negative_selector is not None:
            injected = self._select_with_injected_selector(
                anchor_sample, candidate_negatives, max_negatives)
            if injected is not None:
                return injected

        # Check if pathway-aware negative selection is enabled
        if not self.use_pathway_negatives:
            # Use simple random negative sampling
            selected_negatives = random.sample(
                candidate_negatives, min(max_negatives, len(candidate_negatives)))
            # Return zeros for career distances since pathway analysis is disabled
            career_distances = [0.0] * len(selected_negatives)

            logger.debug(
                f"Selected {len(selected_negatives)} random negatives (pathway negatives disabled)")
            return selected_negatives, career_distances

        # ── Skill-level ontology selection (preferred when available) ──
        if self.skill_matcher:
            try:
                resume_uris = anchor_sample.resume.get('skill_uris', [])
                if resume_uris:
                    anchor_occ = anchor_sample.job.get('occupation_uri', '')
                    return self._select_ontology_negatives(
                        resume_uris, candidate_negatives, max_negatives, anchor_occ_uri=anchor_occ)
            except Exception as e:
                logger.warning(f"Ontology negative selection failed: {e}")

        # ── Random fallback (no skill URIs or ontology selection failed) ──
        # Samples without skill URIs get random negatives. The ontology weight
        # in the loss engine already downweights these samples (tier C → 0.75x).
        selected_negatives = random.sample(
            candidate_negatives, min(max_negatives, len(candidate_negatives)))
        career_distances = [0.0] * len(selected_negatives)

        logger.debug(
            f"Selected {len(selected_negatives)} random negatives (no skill URIs available)")
        return selected_negatives, career_distances

    @staticmethod
    def create_career_graph(config: TrainingConfig, esco_graph_path: str) -> 'CareerGraph':
        """
        Create a CareerGraph instance using configuration parameters and ESCO graph.

        This is a shared factory method that can be used by any component needing
        a CareerGraph instance with consistent configuration.

        Args:
            config: TrainingConfig with distance thresholds
            esco_graph_path: Path to the ESCO graph file

        Returns:
            CareerGraph instance configured with training parameters
        """
        from .career_graph import CareerGraph

        return CareerGraph(
            esco_graph_path=esco_graph_path,
            hard_negative_max_distance=config.hard_negative_max_distance,
            medium_negative_max_distance=config.medium_negative_max_distance,
            use_distance_cache=True
        )

    def get_batch_statistics(self, batch: List[TrainingSample]) -> Dict[str, Any]:
        """
        Get statistics about a batch for monitoring and debugging.

        Args:
            batch: List of TrainingSample objects

        Returns:
            Dictionary with batch statistics
        """
        if not batch:
            return {'total_samples': 0, 'positive_samples': 0, 'negative_samples': 0}

        positive_count = sum(
            1 for sample in batch if sample.label == 'positive')
        negative_count = len(batch) - positive_count

        # Get unique job count for negative sampling potential
        unique_jobs = set()
        for sample in batch:
            job_id = sample.job.get(
                'job_id', sample.job.get('title', 'unknown'))
            unique_jobs.add(job_id)

        return {
            'total_samples': len(batch),
            'positive_samples': positive_count,
            'negative_samples': negative_count,
            'unique_jobs': len(unique_jobs),
            'avg_negatives_per_positive': max(0, len(unique_jobs) - 1) if positive_count > 0 else 0
        }
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for curriculum learning in negative selection."""
        self.current_epoch = epoch

    def update_scheduler_metrics(self, metrics: Dict[str, float]) -> None:
        """Feed validation metrics back to the negative scheduler.
        
        Used by adaptive_val_dgp and performance_gated_triplet schedulers.
        
        Args:
            metrics: Dict with keys like 'val_loss', 'd_gp' (mean positive distance),
                     'triplet_accuracy', etc.
        """
        self._scheduler_metrics = getattr(self, '_scheduler_metrics', {})
        self._scheduler_metrics.update(metrics)
        # Track d(g,p) history for adaptive scheduler
        if 'd_gp' in metrics:
            history = getattr(self, '_dgp_history', [])
            history.append(metrics['d_gp'])
            self._dgp_history = history
        if 'triplet_accuracy' in metrics:
            self._last_triplet_acc = metrics['triplet_accuracy']

    def _compute_negative_ratios(self) -> tuple:
        """Compute hard/medium/easy ratios based on the configured scheduler.
        
        Returns:
            Tuple of (hard_ratio, medium_ratio, easy_ratio)
        """
        scheduler = getattr(self.config, 'negative_scheduler', 'fixed')
        use_curriculum = getattr(self.config, 'negative_curriculum', True)
        epoch_ratio = self.current_epoch / max(1, self.total_epochs)

        # Legacy: negative_curriculum=True with no explicit scheduler → linear_easy_to_hard
        if use_curriculum and scheduler == 'fixed':
            scheduler = 'linear_easy_to_hard'

        if scheduler == 'fixed':
            return (
                getattr(self.config, 'negative_hard_ratio', 0.33),
                getattr(self.config, 'negative_medium_ratio', 0.34),
                getattr(self.config, 'negative_easy_ratio', 0.33),
            )

        elif scheduler == 'linear_easy_to_hard':
            # Linear interpolation: easy-heavy → hard-heavy over training
            # Early: 20% hard, 30% medium, 50% easy
            # Late:  60% hard, 30% medium, 10% easy
            hard_ratio = 0.2 + 0.4 * epoch_ratio
            easy_ratio = 0.5 - 0.4 * epoch_ratio
            medium_ratio = 1.0 - hard_ratio - easy_ratio
            return (hard_ratio, medium_ratio, easy_ratio)

        elif scheduler == 'adaptive_val_dgp':
            # Adaptive: increase hard negatives when d(g,p) separation improves.
            # If d(g,p) is growing (model separating well), push harder negatives.
            # If d(g,p) stalls or drops, ease off.
            dgp_history = getattr(self, '_dgp_history', [])
            if len(dgp_history) >= 2:
                # Compare latest d(g,p) to the one 2 epochs ago (smoothed trend)
                recent = dgp_history[-1]
                prev = dgp_history[-2]
                improving = recent > prev
            else:
                improving = False

            # Base: start conservative, ramp up if improving
            if improving:
                # Model is separating well → push harder
                hard_ratio = min(0.6, 0.3 + 0.05 * len(dgp_history))
                easy_ratio = max(0.1, 0.4 - 0.05 * len(dgp_history))
            else:
                # Model struggling → stay moderate
                hard_ratio = 0.25
                easy_ratio = 0.45
            medium_ratio = 1.0 - hard_ratio - easy_ratio
            return (hard_ratio, medium_ratio, easy_ratio)

        elif scheduler == 'performance_gated_triplet':
            # Only increase hard negatives when triplet accuracy exceeds a threshold.
            # Below threshold: mostly easy. Above: shift to hard.
            triplet_acc = getattr(self, '_last_triplet_acc', 0.0)
            threshold = 0.6  # Gate: need 60% triplet accuracy before ramping hard

            if triplet_acc >= threshold:
                # Scale hard ratio linearly from 0.3 to 0.6 as accuracy goes 0.6→0.9
                progress = min(1.0, (triplet_acc - threshold) / 0.3)
                hard_ratio = 0.3 + 0.3 * progress
                easy_ratio = 0.4 - 0.3 * progress
            else:
                # Below threshold: conservative
                hard_ratio = 0.15
                easy_ratio = 0.55
            medium_ratio = 1.0 - hard_ratio - easy_ratio
            return (hard_ratio, medium_ratio, easy_ratio)

        else:
            # Unknown scheduler, fall back to fixed
            logger.warning(f"Unknown negative_scheduler '{scheduler}', using fixed ratios")
            return (0.33, 0.34, 0.33)

    def _load_isco_codes(self, csv_path: str) -> None:
        """Load occupation_uri -> ISCO code mapping from occupations CSV."""
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uri = row.get('conceptUri', '')
                isco = row.get('iscoGroup', '')
                if uri and isco:
                    self.occ_to_isco[uri] = isco
        logger.info(f"Loaded {len(self.occ_to_isco)} ISCO codes for negative selection")

    def _isco_distance(self, occ_uri_a: str, occ_uri_b: str) -> float:
        """Compute ISCO group distance between two occupations (0-1 scale).
        Uses 5-level hierarchy: same 4-digit=0.0, 3-digit=0.2, 2-digit=0.4, 1-digit=0.7, different=1.0."""
        isco_a = self.occ_to_isco.get(occ_uri_a, '')
        isco_b = self.occ_to_isco.get(occ_uri_b, '')
        if not isco_a or not isco_b:
            return 0.5
        if isco_a == isco_b:
            return 0.0
        if len(isco_a) >= 3 and len(isco_b) >= 3 and isco_a[:3] == isco_b[:3]:
            return 0.2
        if len(isco_a) >= 2 and len(isco_b) >= 2 and isco_a[:2] == isco_b[:2]:
            return 0.4
        if isco_a[:1] == isco_b[:1]:
            return 0.7
        return 1.0

    def _select_ontology_negatives(
        self,
        resume_skill_uris: List[str],
        candidate_negatives: List[Dict[str, Any]],
        max_negatives: int,
        anchor_occ_uri: str = '',
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """
        Select negatives based on skill-level ontology distance to the resume,
        optionally combined with ISCO group distance.

        Uses curriculum learning to shift hard/medium/easy ratios over epochs:
        - Early training: mostly easy negatives (model learns basic distinctions)
        - Late training: mostly hard negatives (model learns fine-grained distinctions)

        Args:
            resume_skill_uris: Skill URIs from the anchor resume
            candidate_negatives: Candidate negative jobs
            max_negatives: Number of negatives to select

        Returns:
            Tuple of (selected_negatives, ontology_distances)
        """
        # Compute ontology distance for each candidate
        scored = []
        # Skip expensive skill matching when isco_weight=1.0 (pure ISCO mode)
        skip_skill_matching = (self.use_isco_negatives and self.isco_weight >= 1.0)
        for job in candidate_negatives:
            # Skill-level distance
            if skip_skill_matching:
                skill_distance = 0.5  # placeholder, won't be used
            else:
                job_uris = job.get('skill_uris', [])
                if job_uris and resume_skill_uris:
                    sim = self.skill_matcher.ontology_set_similarity(resume_skill_uris, job_uris)
                    skill_distance = 1.0 - sim
                else:
                    skill_distance = 0.5

            # Blend with ISCO distance if enabled
            if self.use_isco_negatives and anchor_occ_uri:
                job_occ = job.get('occupation_uri', '')
                isco_dist = self._isco_distance(anchor_occ_uri, job_occ)
                w = self.isco_weight
                distance = (1.0 - w) * skill_distance + w * isco_dist
            else:
                distance = skill_distance

            scored.append((job, distance))

        # Bucket by ontology distance
        hard = [(j, d) for j, d in scored if d <= 0.3]       # very similar skills
        medium = [(j, d) for j, d in scored if 0.3 < d <= 0.6]
        easy = [(j, d) for j, d in scored if d > 0.6]        # very different skills

        # Get ratios from the configured scheduler
        hard_ratio, medium_ratio, easy_ratio = self._compute_negative_ratios()

        hard_count = int(max_negatives * hard_ratio)
        medium_count = int(max_negatives * medium_ratio)
        easy_count = max_negatives - hard_count - medium_count

        selected = []
        if hard and hard_count > 0:
            selected.extend(random.sample(hard, min(hard_count, len(hard))))
        if medium and medium_count > 0:
            selected.extend(random.sample(medium, min(medium_count, len(medium))))
        if easy and easy_count > 0:
            selected.extend(random.sample(easy, min(easy_count, len(easy))))

        # Fill remaining from any bucket
        if len(selected) < max_negatives:
            used = set(id(j) for j, _ in selected)
            remaining = [(j, d) for j, d in scored if id(j) not in used]
            random.shuffle(remaining)
            selected.extend(remaining[:max_negatives - len(selected)])

        selected = selected[:max_negatives]

        # Convert ontology distance to 0-10 scale for career_distances field
        negatives = [j for j, _ in selected]
        distances = [d * 10.0 for _, d in selected]

        logger.debug(
            f"Ontology negatives (epoch {self.current_epoch}): "
            f"ratios=hard:{hard_ratio:.0%}/med:{medium_ratio:.0%}/easy:{easy_ratio:.0%}, "
            f"selected=hard:{min(hard_count, len(hard))}/med:{min(medium_count, len(medium))}/easy:{min(easy_count, len(easy))}"
        )

        return negatives, distances


