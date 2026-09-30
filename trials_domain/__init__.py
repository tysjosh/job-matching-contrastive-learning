"""TREC Clinical Trials domain for the CDCL/OSCAR + ORCA training stack.

This package is the clinical-trials analogue of ``cve_domain``: it adapts the
TREC 2021/2022 Clinical Trials track (patient topics + graded qrels + the
ClinicalTrials.gov snapshot) onto the shared ``TrainingSample`` /
``ContrastiveTriplet`` contracts so ORCA can train on it without any change to
the ``orca`` package.

Why this dataset
----------------
TREC-CT carries a **gold label for the exact phenomenon ORCA models**. The
official qrels are graded:

  * ``2`` — eligible (a true positive),
  * ``1`` — the patient has the target condition but fails eligibility,
  * ``0`` — not relevant.

Grade ``1`` is an expert-annotated *ambiguous negative*: topically relevant,
genuinely not a match. ORCA's weak reliability target ``r_tilde`` is defined to
sit near ``0`` for possibly-false negatives and near ``1`` for true negatives,
so the grade-1/grade-0 split is a direct external validation of the learned
reliability ``r_hat`` — something the career v7 split cannot provide, because
its ``soft_label`` is itself derived from the same ontology signals that feed
``r_ont``.

Ontology mapping
----------------
MeSH 2021 replaces ESCO/ISCO. The correspondence is structural, not cosmetic:

  =================  ===============================  ==============================
  ORCA feature       Career (ESCO/ISCO)               Clinical trials (MeSH)
  =================  ===============================  ==============================
  ``s_esco``         skill-URI set similarity over     descriptor-set similarity over
                     the ESCO skill graph              MeSH tree distances
  ``d_isco``         ISCO occupation-code prefix       condition-branch tree-prefix
                     agreement (coarse group)          agreement (coarse branch)
  ``d_ot``           Sinkhorn over skill-graph hops    Sinkhorn over tree distances
  =================  ===============================  ==============================

Isolation
---------
Like ``cve_domain``, this package imports from ``contrastive_learning`` but is
never imported by it; registration happens as an import-time side effect of
``trials_domain.record_adapter``. Nothing here imports ``orca``.
"""

__all__ = [
    "concept_extractor",
    "data_converter",
    "data_splitter",
    "mesh_ontology",
    "record_adapter",
]
