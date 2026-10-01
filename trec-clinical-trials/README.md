# TREC Clinical Trials 2021/2022

Local, read-only source data for patient-to-clinical-trial retrieval experiments.

## Contents

- `raw/corpus/ClinicalTrials.2021-04-27.part{1..5}.zip`: the shared April 27, 2021 ClinicalTrials.gov snapshot. Keep these archives compressed; preprocessing can stream the XML members.
- `raw/2021/topics2021.xml`: 75 synthetic patient case descriptions.
- `raw/2021/qrels2021.txt`: 35,832 expert relevance judgments.
- `raw/2022/topics2022.xml`: 50 synthetic patient case descriptions.
- `raw/2022/qrels2022.txt`: 35,394 expert relevance judgments.
- `raw/ontology/desc2021.gz`: the time-matched 2021 MeSH descriptor hierarchy.
- `SHA256SUMS`: local integrity manifest.

## Relevance labels

The official qrels format is `topic iteration NCT_ID relevance`:

- `0`: not relevant
- `1`: excluded/ineligible (the patient has the target condition but fails eligibility criteria)
- `2`: eligible

Unjudged patient-trial pairs are **not** negative labels.

## Validated statistics

| Component | Count |
|---|---:|
| Clinical-trial XML records | 375,580 |
| Unique NCT identifiers | 375,580 |
| 2021 topics | 75 |
| 2021 qrels (0 / 1 / 2) | 24,243 / 6,019 / 5,570 |
| 2022 topics | 50 |
| 2022 qrels (0 / 1 / 2) | 28,419 / 3,036 / 3,939 |
| 2021 MeSH descriptors | 29,917 |
| 2021 MeSH tree-number assignments | 61,314 |

All five corpus archives match the MD5 checksums published by `ir_datasets`. All archives pass Python ZIP CRC testing, and every NCT identifier referenced by either qrels file exists in the corpus.

## Sources

- TREC 2021 data: <https://trec.nist.gov/data/trials2021.html>
- TREC 2022 data: <https://trec.nist.gov/data/trials2022.html>
- Track documentation and corpus: <https://www.trec-cds.org/2022.html>
- `ir_datasets` catalog: <https://ir-datasets.com/clinicaltrials.html>
- NLM MeSH data: <https://www.nlm.nih.gov/databases/download/mesh.html>

Consult the source sites' terms before redistributing the raw files. Kaggle data is not used in this copy.
