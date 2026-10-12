# GO function study: setup baseline (gap item 1)

## Downloads

Source: DGZ public release `https://bio2vec.net/data/deepgo/deepgozero/data/` (README's KAUST URL is 404). Script: `scripts/go_function/download_dgz.py` (never overwrites; verifies Content-Length). Files land in `dataset/go_function/dgz/`; machine-readable manifest with URL, size, sha256 and retrieval time: `dataset/go_function/MANIFEST.json`. Total 513 MB.

URL = base + path.

| path | size (bytes) | sha256 |
|---|---|---|
| go.obo | 33820417 | 0a6b4876004554c7bfa784bba6e6e8558b5711eb1d1abf85a751e6c70909af83 |
| go.owl | 132095312 | af109c7b8fbec1db7f0d60d603197b39844f65d9f6663f822a7ab1c812c3f692 |
| go.norm | 4790388 | 870f2caa1abc57b3b90a5e66c880ff9fd41a5bf1d5035cc908e3ff07da9e0dc3 |
| definitions_go.txt | 637592 | 42c3f2df428f4e68f654bcb61feb59929b441cb7d30d6674887d8c5e053594d2 |
| swissprot_exp.fa | 41865100 | ca65b30bc59bc425d89bcee55d2b704a4f764a3f886bca956c97d0d6f73902c7 |
| bp/train_data.pkl | 86396014 | 09bf45263f6c9227c07fba528c14332c1bfd6fcf801049b01b32b07acd47332e |
| bp/valid_data.pkl | 10461151 | d679ce708fad727d6be0d8cde58c46b01cee8a4056a58420cbd70e1e0f1c628d |
| bp/test_data.pkl | 10100306 | bbbafafcf347edd310794f5290db8de9cd52a1b6a44531247d6167aeae30efad |
| bp/interpros.pkl | 317643 | db53f566f2ee54a90384be2b925b7c0075575bfdaa50c57fac780cf5d2a2a4fe |
| bp/terms.pkl | 278708 | 2c79415a69c9d1cad4e1829331fcd0ba6446ae169e9d44efe5c8711c79a5d024 |
| bp/terms_zero.pkl | 278617 | 8ea8a9128c2d8b0ad8622ee64d02da0cc78bff32fdfa6f4d598ef8d498892fd6 |
| bp/terms_zero_10.pkl | 132028 | c12e49fefde3fe7795bf7cccdc57f6bfda1d2f401906f913c225744bd7dbf617 |
| bp/test_diamond.res | 1958902 | 01ff07881fc1b79ce17c2363102bea154c6cb780df153ddb1ef5059fb55cba43 |
| mf/train_data.pkl | 66200048 | 87c0c5888364b8cde2be41180ec38d0a4295f74e370575335a735c1eeec7d4b5 |
| mf/valid_data.pkl | 7707408 | 22ec90c2d94f02ca86aa42996aea078809bb82d14b0a2d2d79b3535945a934a0 |
| mf/test_data.pkl | 9350734 | 4481027c1fb6e4e568608f2bba1b8fcdc6fe420b3210d404487af82d15707f9c |
| mf/interpros.pkl | 317643 | db53f566f2ee54a90384be2b925b7c0075575bfdaa50c57fac780cf5d2a2a4fe |
| mf/terms.pkl | 89982 | 5dfa6627d52b1699d34db5de99cd0c429825a28d36ec74025995371667ee3e4c |
| mf/terms_zero.pkl | 89917 | 750ba52669d7bebdf65c41100757f1aea93af8cc5ad1695353abc91493d32954 |
| mf/terms_zero_10.pkl | 27214 | 4bd50ea9dfff84548ba704669cf9573c45e513b301985396037903540a187270 |
| mf/test_diamond.res | 1653158 | 6b5a9915839afb8314b1e03f41bdeeb3240258fdc9172513accc756370e28539 |
| cc/train_data.pkl | 83105324 | 7f461fa7d36674154b801e80c797f2a482464770577f58c33951cc8a82e62f1e |
| cc/valid_data.pkl | 8816926 | a5a3eb2e95d148f328e6e59a45c7b922dd65561a4dcf50625533f3161dec7f74 |
| cc/test_data.pkl | 10663017 | f932a1ccffeb41dcf41b4855b67b942d4b0b7b1374dcc9b33cbc66d8d67ce60b |
| cc/interpros.pkl | 317643 | db53f566f2ee54a90384be2b925b7c0075575bfdaa50c57fac780cf5d2a2a4fe |
| cc/terms.pkl | 37497 | 4ee40170356f20cd03441e9f91cbbdccd9e157ca2266210cb84f6a0138c5ebfb |
| cc/terms_zero.pkl | 37445 | 0597b9810fb3176a8e9458d529f76370d94849980f168fc286726156b612ff32 |
| cc/terms_zero_10.pkl | 18008 | 7ac537a7d4e29ce4ce27b11d45ce5ea3b0805589ac86359e5bd3eecb2a16eca1 |
| cc/test_diamond.res | 1816401 | b910473ef6cc8e7779f7d5187fca40c0066d641416e6f47c3f8ff35c55eb82d1 |

Not downloaded (not in gap item 1): `swissprot_exp.{pkl,dmnd,sim}`, `*.th` checkpoints, `predictions_*.pkl` (0.94–1.41 GB each; needed later for the gap-4 reproduction gate), the tarballs, ESM-2 weights.

## Baseline tests

All run from the repo root with `.venv/bin/python -m pytest <target> -q -p no:cacheprovider`.

| target | result |
|---|---|
| `go_ppi_domain/tests` | 334 passed, 3 skipped, **1 failed** |
| `trials_domain/tests` | 108 passed, 1 skipped |
| `tests/test_career_adapter_equivalence.py tests/test_career_golden_regression.py tests/test_career_ordinal_candidates.py tests/cve_domain/test_selector_batch_loss_integration.py` | 8 passed |

The go_ppi loss tests (`test_go_ordinal_loss.py`, `test_go_withholding_loss.py`) are inside the first target and pass.

## Deviations

- Pre-existing failure, unrelated to this work: `go_ppi_domain/tests/test_sweep_driver.py::test_real_protocol_amendments_table_parses` asserts the amendments table in `protocol_go_ppi.md` has only `AM1`; the working copy (uncommitted modification, not by this step) now contains `AM1, AM2`. Left as-is; protected file.
- `interpros.pkl` is identical across bp/mf/cc (same sha256).
- "Commit or tag the code" in gap item 1 is left to the user (no git writes by this step).
