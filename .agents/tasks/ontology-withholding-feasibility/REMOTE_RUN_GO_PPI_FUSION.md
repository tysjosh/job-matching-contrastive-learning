# Running the GO/PPI fusion confirmation on the remote L40S host

Operational runbook for `protocol_go_ppi_fusion.md` (DRAFT v4), §6.2–6.4. It cannot change the protocol; where the two differ, the protocol wins.

Everything confirmatory runs on one Linux x86_64 host with one NVIDIA L40S:
- training (34 runs) uses the GPU, `--device cuda`;
- the transfer check, gate 1, data builds, fresh masked encodes, gate 2, scoring and analysis run on that host's CPU;
- `make_pins.py` is the one step-0 script run without the CUDA mask. It only reads the GPU name, which must contain `L40S`, with exactly one device visible.

Nothing confirmatory runs on the Mac. Mac dev-smoke outputs live in `results/go_ppi_fusion_confirm_dev/`; never copy them, and no confirmatory script reads them.

All paths are repo-relative. No machine path enters a config, manifest, ledger or pins file. Every block below assumes this prefix on the remote:

```bash
cd /path/to/CDCL                 # the remote checkout (any location)
R="$PWD"; PY="$R/.venv/bin/python"; S="$R/scripts/go_ppi_fusion_confirm"
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
P=.agents/tasks/ontology-withholding-feasibility/protocol_go_ppi_fusion.md
O=results/go_ppi_fusion_confirm
```

Device rules:
- prefix every CPU step with `CUDA_VISIBLE_DEVICES=""` and pass `--device cpu` where the flag exists;
- never set `CUDA_VISIBLE_DEVICES` for `make_pins.py` or `train.py`. Both refuse to run when it is `""`.

Every script asserts `PYTHONHASHSEED=0` (simGIC sums iterate frozensets) and refuses to overwrite its `--out` leaf. There is no `--force`.

## 1. Transfer (step −1)

### 1-0. Mac, right before any transfer: dump the code-tree hashes

```bash
# on the Mac, repo root. Rerun after any code edit; delete the old file first (the script refuses to overwrite).
PYTHONHASHSEED=0 .venv/bin/python scripts/go_ppi_fusion_confirm/dump_code_hashes.py
#   -> .agents/tasks/ontology-withholding-feasibility/code_hashes_go_ppi_fusion.mac.json
```

The manifest covers the new study files plus every reused module: `go_ppi_fusion/`, `go_ppi_domain/`, `trials_domain/`, `contrastive_learning/`, `scripts/evaluate_publication_fusion.py`, `scripts/run_learning_curve_point.py`, `scripts/go_independent_signal_diagnosis.py`, `run_phase1_embedding_evaluation.py`, and `scripts/trials_fusion_confirm/requirements-remote.txt`. `verify_transfer.py` fails on any missing, extra or differing file.

**Never ship:**
- `/300` (a curl cookie file);
- `reports/`;
- `results/go_ppi_fusion_confirm_dev/`.

### 1A. Preferred: clone the branch, pull data from Hugging Face, copy the small Mac-only set

The study branch must carry the code tree and the manifest from 1-0:
- this protocol and runbook;
- `go_ppi_domain/fusion_confirm.py` and `go_ppi_domain/tests/test_fusion_confirm.py`;
- `scripts/go_ppi_fusion_confirm/`;
- `go_ppi_fusion/` (frozen, import only);
- `code_hashes_go_ppi_fusion.mac.json`.

The public dataset `olukotunjosh/cdcl-data` holds the `preprocess/`, `dataset/` and `embedding_cache/` inputs at their repo-relative paths. It does **not** hold the GO checkpoints, the GO prepared caches or the GO result JSONs; 1A-3 copies those from the Mac. `verify_transfer.py` re-hashes everything against the protocol regardless of source.

```bash
# 1A-1 remote
git clone -b <study branch> git@github.com:tysjosh/job-matching-contrastive-learning.git CDCL
cd CDCL
python3.13 -m venv .venv && .venv/bin/pip install "huggingface-hub==0.34.4"   # full env in section 2
unset HF_HUB_OFFLINE      # downloads need the network; turn it back on afterwards

# 1A-2 remote: data from Hugging Face
.venv/bin/huggingface-cli download olukotunjosh/cdcl-data --repo-type dataset --local-dir . \
  --include "preprocess/go_ppi/*" "preprocess/go_ppi_lc/*" "preprocess/go_ppi_splits/*" \
            "preprocess/go_withholding/converted/*" "preprocess/go_withholding/splits/*" \
            "dataset/go_ppi/bulk/*" "dataset/go_ppi/independent/*" \
            "embedding_cache/go_ppi_text_embeddings.pt" "embedding_cache/go_withholding_text_embeddings.pt"

# MPNet: the FULL snapshot (a subset left config/tokenizer files out in the trials run, trials §9 A3).
# The pinned revision is e8c3b32edf5434bc2275fc9bab85f82640a19130; if the hub's main has moved, use 1B-3.
.venv/bin/hf download sentence-transformers/all-mpnet-base-v2
export HF_HUB_OFFLINE=1
```

```bash
# 1A-3 Mac, repo root: the files that are not on Hugging Face (about 75 MB)
HOST=user@gpu-host; DEST="$HOST:/path/to/CDCL"
cat > /tmp/go_ppi_fusion_mac_only.txt <<'EOF'
results/publication_phase1_fixedval/go_ppi_fusion_learning_curve.json
results/publication_phase1_fixedval/go_ppi_fusion_learning_curve_ndcg_selected.json
results/publication_phase1_fixedval/prepared_embeddings/go_ppi/test_c65a1d2ca0e5c16a_sentence-transformers_all-mpnet-base-v2
results/publication_phase1_fixedval/prepared_embeddings/go_ppi/validation_07e432fb64f3e271_sentence-transformers_all-mpnet-base-v2
results/go_ppi_fusion_best/test_results.json
results/go_ppi_fusion_best/finalists.json
results/go_ppi_fusion_best/test_plan_amendment.json
results/ontology_ceiling/go_independent_signal.json
EOF
for F in 10 25 50 75 100; do for S in 42 13 21; do
  d=results/publication_phase1_fixedval/checkpoints/go_ppi_baseline_f${F}_s${S}
  printf '%s\n' "$d/best_checkpoint.pt" "$d/training_config.json" "$d/lc_manifest.json" >> /tmp/go_ppi_fusion_mac_only.txt
done; done
rsync -a -r --relative --files-from=/tmp/go_ppi_fusion_mac_only.txt ./ "$DEST/"
#   -r is needed: with --files-from, -a does not recurse into the two prepared-cache dirs (5 + 2 chunks).
```

On the remote, `git status` should then show only untracked data paths.

### 1B. Alternative: copy everything from the Mac

Use this if the clone route is not available or a Hugging Face file fails the hash check.

```bash
HOST=user@gpu-host; DEST="$HOST:/path/to/CDCL"

# 1B-1 code tree (.git, configs, scripts, packages, .agents with this protocol and the 1-0 manifest).
#      Only untracked heavy paths are excluded; never /300, /reports/ or the dev-smoke dir.
rsync -a --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '/results/' --exclude '/preprocess/' --exclude '/embedding_cache/' \
  --exclude '/dataset/' --exclude '/training_output/' --exclude '/trec-clinical-trials/raw/' \
  --exclude '/.cache/' --exclude '/cybersecurity-vulnerability-ranking/' \
  --exclude '/cve_domain/runs/experiments/' --exclude '/cve_domain/runs/prof/' \
  --exclude '/cve_domain/runs/experiments_smoke/_shared/view_records_full.jsonl' \
  --exclude '/colab_package.tar.gz' --exclude '/300' --exclude '/reports/' \
  ./ "$DEST/"

# 1B-2 every pinned data input (protocol §7 table, §3b checkpoints, §2.1 prepared caches), about 790 MB
cp /tmp/go_ppi_fusion_mac_only.txt /tmp/go_ppi_fusion_transfer.txt      # the 1A-3 list
cat >> /tmp/go_ppi_fusion_transfer.txt <<'EOF'
preprocess/go_ppi_splits/train.jsonl
preprocess/go_ppi_splits/validation.jsonl
preprocess/go_ppi_splits/test.jsonl
preprocess/go_ppi_splits/negative_pools.jsonl
preprocess/go_ppi_splits/split_manifest.json
preprocess/go_ppi_lc/frac_10/train.jsonl
preprocess/go_ppi_lc/frac_25/train.jsonl
preprocess/go_ppi_lc/frac_50/train.jsonl
preprocess/go_ppi_lc/frac_75/train.jsonl
preprocess/go_ppi_lc/frac_100/train.jsonl
preprocess/go_ppi_lc/validation_positive.jsonl
preprocess/go_ppi_lc/fraction_manifest.json
preprocess/go_ppi/go_index_P.pkl
preprocess/go_ppi/proteins.jsonl
preprocess/go_ppi/pairs.jsonl
preprocess/go_ppi/convert_manifest.json
preprocess/go_withholding/splits/train.jsonl
preprocess/go_withholding/splits/validation.jsonl
preprocess/go_withholding/splits/test.jsonl
preprocess/go_withholding/splits/negative_pools.jsonl
preprocess/go_withholding/splits/split_manifest.json
preprocess/go_withholding/converted/go_index_P.pkl
preprocess/go_withholding/converted/proteins.jsonl
preprocess/go_withholding/converted/pairs.jsonl
preprocess/go_withholding/converted/mask_lexicon.json
preprocess/go_withholding/converted/mask_manifest.json
preprocess/go_withholding/converted/convert_manifest.json
embedding_cache/go_ppi_text_embeddings.pt
embedding_cache/go_withholding_text_embeddings.pt
dataset/go_ppi/bulk/go-basic.obo
dataset/go_ppi/bulk/goa_human.gaf.gz
dataset/go_ppi/bulk/9606.protein.aliases.v12.0.txt.gz
dataset/go_ppi/bulk/9606.protein.info.v12.0.txt.gz
dataset/go_ppi/bulk/9606.protein.links.detailed.v12.0.txt.gz
dataset/go_ppi/independent/HuRI.tsv
dataset/go_ppi/independent/Lit-BM.tsv
EOF
rsync -a -r --relative --files-from=/tmp/go_ppi_fusion_transfer.txt ./ "$DEST/"

# 1B-3 the full MPNet snapshot (revision e8c3b32edf5434bc2275fc9bab85f82640a19130) into the remote HF cache
ssh "$HOST" mkdir -p .cache/huggingface/hub
rsync -a ~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2 "$HOST:.cache/huggingface/hub/"
```

What you should see:
- 1B-1 ships roughly 0.5 GB, most of it the `.git` pack (about 470 MB on the Mac).
- 1B-2 ships about 790 MB, mostly `dataset/go_ppi`, the two text caches and the split files.
- 1B-3 adds about 420 MB.
- `preprocess/go_ppi_fusion_confirm/` and `results/go_ppi_fusion_confirm/` are **not** copied. They are built on the remote in step 0.

Record the method actually used in the run log at the end of this file.

## 2. Environment

```bash
python3.13 -m venv .venv
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # driver < 570: .../whl/cu126
"$PY" -m pip install -r scripts/trials_fusion_confirm/requirements-remote.txt
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
nvidia-smi
```

The GPU name must contain `L40S`. `make_pins.py` records the versions, GPU name, `nvidia-smi` output, BLAS and the device policy in `pins.environment`. Keep the same shell environment for every later step, including thread settings such as `OMP_NUM_THREADS` if you set any.

## 3. Verify the transfer (step −1, CPU)

`verify_transfer.py` parses the protocol for the expected hashes:
- exactly 15 checkpoint rows (§3b) and at least 30 inputs rows (§7);
- both published prepared-cache dirs: 5 test chunks with digest `42abdb8c…`, 2 validation chunks with digest `cefa1e04…`;
- the MPNet revision, followed by a CPU load test (construct, encode one string, assert width 768);
- the code tree against the 1-0 manifest.

It also records the parsed protocol's sha256.

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/verify_transfer.py" --protocol "$P" \
    --code-manifest .agents/tasks/ontology-withholding-feasibility/code_hashes_go_ppi_fusion.mac.json \
    --out $O/transfer_check.json
```

A pass exits 0 with `PASSED`. On any failure it exits 2 and writes `transfer_check.FAILED.<utc>.json`; fix the transfer and rerun. A code failure means the transfer was partial or the Mac tree changed after 1-0: redo 1-0 and the code transfer. Nothing else runs until this passes.

## 4. Gate 1: unit tests (CPU)

```bash
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest go_ppi_domain/tests/test_fusion_confirm.py -q     # must pass
```

Mac runs do not count. If this fails, stop and report; do not continue.

## 5. Step 0: label-free builds, fresh masked encodes, cache copy (CPU)

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/build_inputs.py" --protocol "$P" \
    --fold-seed 2027 --n-folds 5 --shuffle-seed 11 --device cpu \
    --out-data preprocess/go_ppi_fusion_confirm --out-phi $O/phi --out-configs $O/configs
CUDA_VISIBLE_DEVICES="" "$PY" "$S/encode.py" --device cpu \
    --inputs preprocess/go_withholding/splits preprocess/go_ppi_fusion_confirm \
    --out $O/prepared_embeddings/go_ppi_masked
mkdir -p $O/cache
cp -n embedding_cache/go_withholding_text_embeddings.pt $O/cache/go_withholding_text_embeddings.pt
```

`build_inputs.py` writes everything below without computing any text score or fused score, plus `$O/build_manifest.json`:
- the cohort-A masked fractions (row counts 1,571 / 3,927 / 7,854 / 11,781 / 15,708) and `validation_positive.jsonl` (321);
- the 5 cohort-B folds;
- the HuRI cohort;
- the shuffle maps for all 14 scored files;
- `n_pmid.json`;
- the is_a-only index (if the typed check passes);
- the cohort-A mention flags;
- the derived configs and their `report_factor` audit.

It stops on any build assert. Each assert is resolved by a pre-pins amendment of the protocol (§9), never by editing code silently:
- identical-view groups must be 14 groups / 31 proteins / max size 4;
- two independent fold builds must give identical sha256 for every file;
- hard-defined `Bclean` must be ≥ 2,400. The Mac fold-structure count under seed 2027 was 2,669;
- HuRI anchors must be within 769 ± 5;
- no shuffle stratum may exceed 10,000 draws;
- the config audit set must be exactly `["embedding_cache_path", "go_ppi_converted_dir", "go_ppi_split_dir"]`.

If the typed-parser check fails, T is dropped and C1t is reported as not computed. This is printed but is not a stop.

`encode.py` encodes every distinct masked text of the scored files on the CPU (`SentenceTransformer(<snapshot>, device="cpu")`, batch 64, no normalisation). Work goes to `…/go_ppi_masked.partial/`; rerunning the same command after an interruption resumes from it. The directory is renamed on completion. Expect roughly 15–30 min CPU (estimate, not measured on the remote).

## 6. Pins (GPU name only, no CUDA mask)

```bash
"$PY" "$S/make_pins.py" --protocol "$P" --out $O/pins.json --host-label "<short host label>"
```

`make_pins.py` checks all of the following and refuses if any check fails:
- a passing `transfer_check.json` that parsed this protocol file, and a code tree still equal to the Mac manifest;
- sha256(cache copy) == sha256(original) == `3aed5148…`;
- every §7 input, published checkpoint and prepared chunk re-hashed against the protocol;
- every `build_inputs.py` output re-hashed against `build_manifest.json`;
- the masked-encode digest;
- the published near-tie cells equal to the three pinned cells (ndcg f10_s42; pooled f10_s42; pooled f75_s21);
- exactly one visible CUDA device whose name contains `L40S`.

It writes no config; the derived configs come from `build_inputs.py` and are only hashed here.

## 7. Gate 2: reproduction and integrity (CPU)

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/score.py" --gate-reproduce \
    --pins $O/pins.json --device cpu --out $O/gate_reproduction.json
```

Parts, all of which must pass:
- **2a.** The 15 published checkpoints with U on the published files, grid 0–1.0, both rules, leaf by leaf against the two published JSONs.
  - Tolerance tiers: exact counts and `selected_weight`; per-anchor rank leaves ≤ 1e-3; rank leaves ≤ 1e-4; continuous leaves ≤ 1e-5.
  - Near-tie exception: in the three pinned cells either top weight passes, and test leaves are compared at the published weight. The event is logged.
  - Zero-shot on the published test file: hard 0.786, pooled 0.838, nDCG@10 0.772 (3 dp).
- **2b.** Alignment of the masked and published splits, the cohort-A fractions and `validation_positive` key match, and record URIs equal to the pruned index.
- **2c.** The typed-parser check. The pinned is_a-only index must re-derive identically.
- **2d.** Every derived file is pinned and unchanged, with the asserted counts: groups 14/31, two-build identity, `Bclean` sha256 and floor, HuRI anchors and both grade-1 counts, masked digest, config audit.
- **2e.** φ path equality. X, P, N, S, S_P and UX on cohort-A masked validation/test must equal direct `GoMatcher` calls exactly.
- **2f.** AZ-X and GZ-P on the published checkpoints against `results/go_ppi_fusion_best/test_results.json` (`F3_P`, `PUB_ndcg_P`). The selected w must match exactly, except near ties (top-two gap < 1e-4, logged). Per-checkpoint Δ for hard, pooled, easy, AP and nDCG@10 must be ≤ 1e-4.

**On failure** (retry rule, §5.2):
1. Fix the code.
2. Log a §9 entry in the protocol.
3. Redo 1-0 and the code transfer.
4. Write a new pins revision and rerun gate 2 into a new leaf:

```bash
"$PY" "$S/make_pins.py" --revise --previous <latest pins> --amendment A<m> --protocol "$P" \
    --changed-fields "" --confirmatory-output-seen false --out $O/pins_r<n>.json
CUDA_VISIBLE_DEVICES="" "$PY" "$S/score.py" --gate-reproduce --pins $O/pins_r<n>.json \
    --device cpu --out $O/gate_reproduction_r<n>.json
```

`--revise` recomputes everything, never rewrites configs (it requires their hashes unchanged), and requires `--changed-fields` to equal the actual non-whitelisted changes. Tolerances are fixed. A failure that persists after a code fix stops the study.

Once gate 2 passes, `pins.code` is frozen. Below, `<PINS>` is the latest revision; every script verifies the `supersedes` chain and refuses a non-latest revision.

## 8. Training: 34 runs = 19 cohort A + 15 cohort B (GPU), under tmux

```bash
tmux new -s gpfc          # reattach later with: tmux attach -t gpfc
"$PY" "$S/train.py" --pins <PINS> --cohort A --device cuda 2>&1 | tee -a $O/train_A.out
"$PY" "$S/train.py" --pins <PINS> --cohort B --device cuda 2>&1 | tee -a $O/train_B.out
```

`train.py` issues exactly the §6.3 commands (`run_learning_curve_point.py --domain go_ppi … --compare-config config/lc_go_ppi_baseline.json --device cuda`) in the fixed order:
- cohort A: F 10, 25, 50, 75, 100; seeds 42, 13, 21, then 87, 123 at F 25 and 100. These use the masked fractions and `lc_go_ppi_baseline_masked.json`;
- cohort B: folds k 0..4; seeds 13, 21, 42; `--fraction 1.0`, with `lc_go_ppi_baseline_cv_fold{k}.json`.

It sets `PYTHONHASHSEED=0` in the child and writes per-attempt logs to `$O/train_logs/`.

After each run it applies gate 3:
- exit code 0;
- `lc_manifest.json` and `best_checkpoint.pt` present, and no `run_error.json`;
- `unexpected_fields == ["embedding_cache_path", "go_ppi_converted_dir", "go_ppi_split_dir"]`;
- `attach.cache_check` passes and `attach.matcher` is null;
- the device block is `cuda`/`cuda` with `cuda_device_name == pins.environment.gpu_name`;
- both original caches are unchanged.

It then appends to `$O/run_ledger.json`, recording the checkpoint sha256 and the cache-copy entries and sha256 before and after the run.

Rules:
- `--cohort B` refuses to start until every cohort-A run is passed or excluded.
- Never parallelise or shard runs. The order is part of the condition because the cache copy grows.
- `--max-runs N` advances in chunks; a later call resumes in order.

Stop rules:
- cohort A needs ≥ 17 of 19 checkpoints;
- cohort B needs ≥ 4 folds with all 3 seeds.

Otherwise stop and amend through §7's declared-change branch.

## 9. Scoring, once (CPU)

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/score.py" --pins <PINS> --ledger $O/run_ledger.json \
    --device cpu --out $O/scores
```

`score.py` re-verifies every pinned hash, then exports the following groups:

| group | models | files |
|---|---|---|
| `A/all` | passing retrained runs + zero-shot | masked cohort-A validation/test |
| `huri/all` | the same models | HuRI test + `test_grade1_unrestricted` |
| `B/fold_k` | passing seeds + zero-shot | the fold's validation/test |
| `apub/all` | the 15 published checkpoints | published files (descriptive) |

The sources it uses:
- masked texts come only from the fresh encodes; a missing text aborts;
- published texts come from the frozen cache plus the published chunks;
- φ comes from the pinned shuffle maps and indexes.

Each group gets `phis_<file>.npz`, `<model>.npz` and `selected_w.json` (w, curve and top-two gap per model × arm).

## 10. Analysis, once (CPU)

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/analyze.py" --pins <PINS> --scores $O/scores \
    --out $O/analysis.json --report $O/report.md
```

`analyze.py` recomputes every selected weight, which must equal `selected_w.json`. It then runs:
- the two-level bootstrap (B = 2000; rng A [31415,1], Bclean [31415,2], HuRI [31415,3], all_B [31415,4]);
- Holm(3) for H1–H3, H1z–H3z and H1g–H3g;
- the C2 / C5 / trials-C2 states, TOST and the four-state labels;
- DE, the 7-label rule, the headline and every §4.5 sensitivity.

The headline is the first line of `report.md`.

Wall time, all estimates, not remote measurements:
- each 15-epoch GPU run should take a few minutes, so 34 runs is likely 1–3 h in total (trials: 62 min for 34 runs);
- `build_inputs.py` takes minutes, most of it parsing the STRING links file;
- scoring and analysis take tens of minutes on CPU.

## 11. Monitoring and resume

```bash
"$PY" "$S/train.py" --pins <PINS> --cohort A --dry-run       # plan + ledger status per run, trains nothing
tail -f $O/train_logs/*.log
"$PY" -c "import json;[print(e['run'],e['status'],e.get('gate3_failures')) for e in json.load(open('$O/run_ledger.json'))]"
nvidia-smi
df -h .
```

**Resume** after a dropped session or crash: rerun the same `train.py` command.
- Passed and excluded runs are skipped.
- An output dir without a passing entry is renamed `.failed1` and retried once; a second failure is recorded as `excluded`.
- Never edit or delete `run_ledger.json`.

For other steps:
- `encode.py` resumes from `….partial/`.
- `build_inputs.py`, `score.py` and `analyze.py` refuse an existing `--out`. If one was interrupted, move its partial output aside (never delete a confirmatory output that was read) and record that in the run log.

## 12. What to send back

```bash
tar czf go_ppi_fusion_confirm_return.tgz \
  $O/{transfer_check.json,build_manifest.json,pins*.json,configs,gate_reproduction*.json,run_ledger.json,train_logs,train_*.out,analysis.json,report.md} \
  $O/scores/score_manifest.json $O/scores/*/*/selected_w.json \
  $O/phi/shuffle_maps $O/phi/n_pmid.json $O/phi/mention_flags_A_test.json \
  $O/prepared_embeddings/go_ppi_masked/encode_manifest.json \
  preprocess/go_ppi_fusion_confirm/cv/cv_manifest.json preprocess/go_ppi_fusion_confirm/cv/fold_*/fold_manifest.json \
  preprocess/go_ppi_fusion_confirm/huri/huri_manifest.json
"$PY" -m pip freeze > pip_freeze.txt; nvidia-smi > nvidia_smi.txt
```

Also send:
- `pip_freeze.txt` and `nvidia_smi.txt`;
- any `*.FAILED.*.json`;
- the run log below.

Optionally send the checkpoint dirs `$O/cohortA/` and `$O/cohortB/` (about 2.6 MB per `best_checkpoint.pt` plus logs) and the full `$O/scores/` (per-record arrays, tens of MB). Do not send the cache copy (about 60 MB, mutable, not needed).

## Run log (fill in on the remote)

| step | date (UTC) | result / notes |
|---|---|---|
| transfer method (1A / 1B) | | |
| −1 verify_transfer | | |
| gate 1 (pytest) | | |
| 0 build_inputs / encode / cache copy | | |
| pins | | |
| gate 2 (2a–2f) | | |
| training A (19) | | |
| training B (15) | | |
| scoring | | |
| analysis | | |
