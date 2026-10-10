# GO/PPI withholding study — setup baseline

Repo: /Users/olukotunjosh/Downloads/CDCL, branch orca-experiments-and-runner, HEAD 699e0cb. Apple M2, 8 GB, CPU-only.

## Raw data sha256 (GOA/OBO are "current" downloads)
| File | sha256 |
|---|---|
| dataset/go_ppi/bulk/9606.protein.info.v12.0.txt.gz | 144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c |
| dataset/go_ppi/bulk/9606.protein.links.detailed.v12.0.txt.gz | 28a2ea3b22c964e691c0f2f934c5e3e397a4b0a12c833d8787fa02b7847b91b6 |
| dataset/go_ppi/bulk/9606.protein.aliases.v12.0.txt.gz | b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680 |
| dataset/go_ppi/bulk/9606.protein.physical.links.v12.0.txt.gz | 1ca87209a93cf7685e9c2997fbcfbfc69fedc6a803ccc1df917318a651df0823 |
| dataset/go_ppi/bulk/goa_human.gaf.gz | db472faff1785878521693af62646546cea6af4a386609dd01c72c0554a46a30 |
| dataset/go_ppi/bulk/go-basic.obo | b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc |
| dataset/go_ppi/independent/HuRI.tsv | 4cdf94366ff6e23e71ef9ec81922e85b2b3a1fb08e938f7b00d86c027cf1ecd6 |
| dataset/go_ppi/independent/Lit-BM.tsv | f39c9c4b8a372286c3eca77d0ff651d837a643d32235eae866daae082663391d |

## Frozen test file
preprocess/go_ppi_splits/test.jsonl: c65a1d2ca0e5c16a1c69ecd70d921e670531d325c52625aa15dba88c2e2481e4 (matches expected prefix c65a1d2c).

## Baseline tests
- `.venv/bin/python -m pytest go_ppi_domain/tests/ -q` → 20 passed, 0 failed.
- `.venv/bin/python -m pytest tests/test_career_ordinal_candidates.py tests/test_mesh_soft_sampling.py tests/test_career_adapter_equivalence.py tests/test_career_golden_regression.py -q` → 28 passed, 0 failed (loss_engine/trainer-touching tests).

## Output directories
preprocess/go_ppi and preprocess/go_ppi_splits are read-only for this study. Leak-pruned rebuilds go to new sibling directories, named in the design step.

## Git safety
Uncommitted edits in contrastive_learning/batch_processor.py, loss_engine.py and trainer.py (225+/20-) were observed present; they will only receive additive, career-neutral edits. No commit, stash, reset or checkout in any step.
