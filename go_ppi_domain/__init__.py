"""Gene Ontology / protein-protein-interaction domain (``domain_adapter="go_ppi"``).

The high-ceiling arm of the ontology-injection study. Career (ESCO/ISCO), trials
(MeSH) and patents (CPC) all measure near-chance ontology signal on their hard
contrast, so a null training result there is uninformative on its own: it cannot
distinguish "ontology injection does not work" from "this ontology carries no
signal about this label". GO/PPI is the case where the ontology demonstrably does
carry the signal --

    hard contrast (established interaction vs weak-evidence interaction)
    simGIC rank-AUC 0.8195, 95% CI [0.8063, 0.8335], n=3000/tier

-- measured before any training, by ``scripts/go_bulk_signal_diagnosis.py``. That
makes this the domain where injection *should* help if the ceiling diagnostic is
predictive at all.

Structural parallel to ``trials_domain``, deliberately three grades so the
existing graded machinery is reused unchanged:

    grade 2  high-confidence interaction   (STRING experimental >= 700)  positive
    grade 1  weak-evidence interaction     (0 < experimental < 150)      hard negative
    grade 0  no interaction recorded                                     easy negative

Grade 1 is the analogue of trials' "ineligible" and career's "potential_fit": a
pair that is topically plausible (both proteins studied together, some assay
signal) but not an established interaction. Grade 0 is "not_relevant".

Nothing in ``contrastive_learning/`` or ``orca/`` is modified; this package
injects through the existing additive seams (``set_ontology_matcher``,
``set_domain_negative_selector``) and reuses the shared record key names
``encoder_view`` / ``skill_uris`` / ``coarse_uris`` / ``grade`` /
``original_label`` / ``metadata.resume_id``.
"""
