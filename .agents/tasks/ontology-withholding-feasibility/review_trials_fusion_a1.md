# Review of §9 A1: cohort 2022 retrained on one device

A1 replaces the cohort-2022 trained population. The previous population was 15 Mac-trained published checkpoints plus 4 CUDA seeds. The new one is the same 19 (F, S) cells, all retrained on the pinned A100 with `--device cuda`. The published 15 move to gate 2 only. The amendment is made before pins revision 1, and no confirmatory output exists, so the §7 rules allow it. It removes the device–seed confound that §8 had disclosed. It also makes every cohort-2022 model new, and it takes the 15 checkpoints previewed by the A0 shuffled-control run out of the population.
Watch for: nothing blocking. A1 is a self-review by the implementer, not an independent reviewer (confirmed).

**Verdict**: APPROVED

## High-level view

The design change touches only the trained-model population of cohort 2022. Hypotheses, metrics, controls, the weight grid, the normaliser, the bootstrap, Holm, labels, the ≥ 17 of 19 stop rule, cohort 2021 and gates 2 and 3 are unchanged. The bootstrap still resamples 19 checkpoints at equal weight, so the fraction-stratified sensitivity keeps its meaning.

The code change has a single source. `fc.COHORT2022_RUNS` drives pins, training order and the tests. `score.py` no longer adds the published checkpoints to cohort 2022. The new test checks the run set against the §3b table, the order, the train files and the `--fraction` values. The three added input rows are hash-checked at the transfer check, and the parser minimum rises to 21 so a protocol copy missing one aborts.

<details>
<summary>Issues (2)</summary>

1. **Self-review only** (LOW, confirmed). A1 changes the preregistered population and was reviewed by the implementer. An independent reviewer can check it quickly, because A1 lists every unchanged rule.
2. **Dev-local 2022 analysis has no trained models** (NIT, confirmed). With no retrained run present locally, `analyze.py --dev-local` cannot assemble cohort 2022. That path is not needed again. If it ever is, use a toy retrained run.

</details>

<details>
<summary>File map</summary>

- `trials_domain/fusion_confirm.py`: `COHORT2022_RUNS`, `COHORT2022_SEEDS`, `COHORT2022_FRACTION_ARG`, the 34-run `training_order`, reworded A0 disclosure, parser minimum 21.
- `scripts/trials_fusion_confirm/make_pins.py`: `NEW_2022` set to the 19 runs.
- `scripts/trials_fusion_confirm/train.py`: `--fraction` for every F, updated docstring.
- `scripts/trials_fusion_confirm/score.py`: cohort 2022 scores retrained runs only.
- `trials_domain/tests/test_fusion_confirm.py`: A1 plan test, parser count 21, toy root with 21 input rows.
- `protocol_trials_fusion.md`: §9 A1 and pointers; three §7 input rows.
- `REMOTE_RUN_TRIALS_FUSION.md`: transfer list, run counts, wall-time note.
- `verify_trials_fusion.md`: pass-4 section.

</details>
