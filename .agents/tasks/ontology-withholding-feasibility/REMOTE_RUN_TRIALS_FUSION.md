# Running the TREC-CT MeSH fusion confirmation on a remote GPU host

This is the operational runbook for `protocol_trials_fusion.md` (DRAFT v7, frozen), §6.2–6.4. It cannot change the protocol; where the two differ, the protocol wins.

The target is one Linux x86_64 host with one NVIDIA GPU (the planned host is the user's remote NVIDIA L40S; any single GPU works, and `pins.environment.gpu_name` records whichever is used). Everything confirmatory runs on that one host:
- training uses the GPU;
- the transfer check, gate 1, fold building, pins, gate 2, scoring and analysis run on that host's CPU.

Nothing confirmatory runs on the Mac. The Mac outputs under `results/trials_fusion_confirm/dev_local/` are non-confirmatory. Never copy them, and nothing reads them. (The protocol header names `results/trials_fusion_confirm_smoke/`; §9 A0 records the change to `dev_local/`. A0 also discloses what the earlier Mac smoke exposed. Do not run further 2021 dev smokes on the real folds.)

All paths are repo-relative. No machine path enters a config, manifest, ledger or pins file. Every block below assumes this prefix on the remote:

```bash
cd /path/to/CDCL                 # the remote checkout (any location)
R="$PWD"; PY="$R/.venv/bin/python"; S="$R/scripts/trials_fusion_confirm"
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
P=.agents/tasks/ontology-withholding-feasibility/protocol_trials_fusion.md
```

Device rule:
- prefix every CPU step with `CUDA_VISIBLE_DEVICES=""`;
- never set `CUDA_VISIBLE_DEVICES` for training. `train.py` refuses `--device cuda` when it is `""`.

## 1. Transfer (step −1), from the local repo root

None of the pinned inputs is git-tracked, so a plain clone has none of them. Copy the working tree, which includes `.git` and the uncommitted study files, and then the data, both to the same repo-relative paths.

**Never ship:**
- `/300` (a curl cookie file);
- `reports/`;
- `results/trials_fusion_confirm/dev_local/`.

```bash
HOST=user@gpu-host; DEST="$HOST:/path/to/CDCL"

# 1-0. Mac: dump the code-tree hashes right before 1a (protocol §9 A0). Rerun after any code
#      edit: delete the old file first (the script refuses to overwrite).
PYTHONHASHSEED=0 .venv/bin/python scripts/trials_fusion_confirm/dump_code_hashes.py
#   -> .agents/tasks/ontology-withholding-feasibility/code_hashes_trials_fusion.mac.json (shipped by 1a)

# 1a. code tree: .git, configs, scripts, packages, .agents (protocol), root modules.
rsync -a --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '/results/' --exclude '/preprocess/' --exclude '/embedding_cache/' \
  --exclude '/dataset/' --exclude '/training_output/' --exclude '/trec-clinical-trials/raw/' \
  --exclude '/.cache/' --exclude '/cybersecurity-vulnerability-ranking/' \
  --exclude '/cve_domain/runs/experiments/' --exclude '/cve_domain/runs/prof/' \
  --exclude '/cve_domain/runs/experiments_smoke/_shared/view_records_full.jsonl' \
  --exclude '/colab_package.tar.gz' --exclude '/300' --exclude '/reports/' \
  ./ "$DEST/"
#   The three cve_domain/runs excludes are untracked CVE-study outputs (~1 GB) that the
#   trials study never reads. Only untracked paths are excluded, so `git status` on the
#   remote (pins.git.dirty_paths) shows no spurious deletions.

# 1b. every untracked pinned input (protocol §7 table, §3b checkpoints, §2.1 prepared caches).
cat > /tmp/trials_fusion_transfer.txt <<'EOF'
preprocess/trec_ct/qrels.jsonl
preprocess/trec_ct/topics.jsonl
preprocess/trec_ct/trials.jsonl
preprocess/trec_ct_splits/train.jsonl
preprocess/trec_ct_splits/validation.jsonl
preprocess/trec_ct_splits/test.jsonl
preprocess/trec_ct_splits/negative_pools.jsonl
preprocess/trec_ct_splits/split_manifest.json
preprocess/trec_ct_lc/frac_10/train.jsonl
preprocess/trec_ct_lc/frac_25/train.jsonl
preprocess/trec_ct_lc/frac_50/train.jsonl
preprocess/trec_ct_lc/frac_75/train.jsonl
preprocess/trec_ct_lc/frac_100/train.jsonl
preprocess/trec_ct_lc/validation_positive.jsonl
embedding_cache/trials_text_embeddings.pt
embedding_cache/mesh2021_index.pkl
trec-clinical-trials/raw/ontology/desc2021.gz
results/publication_phase1_fixedval/trials_fusion_learning_curve_ndcg_selected.json
results/publication_phase1_fixedval/prepared_embeddings/trials/test_4030cc85ff1b4ef3_sentence-transformers_all-mpnet-base-v2
results/publication_phase1_fixedval/prepared_embeddings/trials/validation_5defb59b858bdeee_sentence-transformers_all-mpnet-base-v2
EOF
for F in 10 25 50 75 100; do for S in 42 13 21; do
  d=results/publication_phase1_fixedval/checkpoints/trials_baseline_f${F}_s${S}
  printf '%s\n' "$d/best_checkpoint.pt" "$d/training_config.json" "$d/lc_manifest.json" >> /tmp/trials_fusion_transfer.txt
done; done
ls preprocess/*.py >> /tmp/trials_fusion_transfer.txt   # optional imports of contrastive_learning.view_generator
rsync -a -r --relative --files-from=/tmp/trials_fusion_transfer.txt ./ "$DEST/"
#   -r is needed: with --files-from, -a does not recurse into the two prepared-cache dirs
#   (36 chunks + the empty validation dir).

# 1c. the MPNet snapshot (revision e8c3b32edf5434bc2275fc9bab85f82640a19130) into the remote HF cache
ssh "$HOST" mkdir -p .cache/huggingface/hub
rsync -a ~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2 "$HOST:.cache/huggingface/hub/"
```

What you should see:
- 1a ships about 520 MB, most of it the `.git` pack (about 465 MB).
- 1b ships about 560 MB. Most of it is `trials.jsonl`, the text cache, the published fusion JSON and the 36 chunks. 1c adds about 420 MB.
- The fold files (`preprocess/trials_fusion_confirm/cv2021/`) are not copied. They are rebuilt on the remote in step 0.
- Record the method actually used in the run log at the end of this file.

## 2. Environment

```bash
python3.13 -m venv .venv
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # driver < 570: use .../whl/cu126
"$PY" -m pip install -r "$S/requirements-remote.txt"
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
nvidia-smi
```

`make_pins.py` records the versions, GPU name, driver, CUDA runtime, torch CUDA build, cuDNN and BLAS in `pins.environment`. Keep the same shell environment for every later step, including thread settings such as `OMP_NUM_THREADS`, if you set any.

## 3. Step −1: verify the transfer (CPU)

This step parses the protocol for the expected hashes:
- the 15 checkpoint rows and the 21 inputs rows (18 + the three `frac_{10,50,75}` files of §9 A1), matched by anchored regexes;
- the 36-chunk manifest digest `3d015e60…3f11`;
- the 0-chunk validation dir;
- the MPNet revision;
- the code tree against the Mac manifest from step 1-0: any missing, extra or differing study/reused code file fails (§9 A0). `make_pins.py` refuses to run without this passing check.

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/verify_transfer.py" --protocol "$P" \
    --code-manifest .agents/tasks/ontology-withholding-feasibility/code_hashes_trials_fusion.mac.json \
    --out results/trials_fusion_confirm/transfer_check.json
```

A code failure means 1a was partial or the Mac tree changed after 1-0: rerun 1-0 and 1a, then this step.

It exits 0 with `PASSED`. On any failure, it exits 2 and writes `transfer_check.FAILED.<utc>.json`. Fix the transfer and rerun. Nothing else runs until this passes.

## 4. Step 0: gate 1, folds, cache copy, pins (CPU)

```bash
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest trials_domain/tests/test_fusion_confirm.py -q     # gate 1: must pass
CUDA_VISIBLE_DEVICES="" "$PY" "$S/build_cv2021.py" \
    --converted-dir preprocess/trec_ct --out preprocess/trials_fusion_confirm/cv2021 \
    --fold-seed 2021 --n-folds 5 --cosine-threshold 0.80 --max-component 6 --device cpu
mkdir -p results/trials_fusion_confirm/cache
cp -n embedding_cache/trials_text_embeddings.pt results/trials_fusion_confirm/cache/trials_text_embeddings.pt
CUDA_VISIBLE_DEVICES="" "$PY" "$S/make_pins.py" --protocol "$P" \
    --out results/trials_fusion_confirm/pins.json --host-label "<short host label>"
```

- `build_cv2021.py` asserts the following and stops on any deviation (amend through §9):
  - the 3 cosine-added edges 26–63, 26–57 and 15–21;
  - the 5-topic component;
  - the 23-topic `exposed_closed`;
  - topic-disjointness and co-located groups.

  It prints 5 folds of 15 topics.
- `make_pins.py`:
  - requires a passing `transfer_check.json` that parsed this protocol file;
  - requires sha256(cache copy) == sha256(original) == `fa83e10a…1d4e`;
  - re-checks every input, checkpoint and chunk hash;
  - writes `results/trials_fusion_confirm/configs/*.json` (one time only);
  - probes the GPU in a child process with CUDA visible, so `gpu_name` is filled even though the step runs with `CUDA_VISIBLE_DEVICES=""`.

  It refuses to run if no GPU is found.

## 5. Gate 2: reproduction on the 15 published checkpoints (CPU)

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/score.py" --gate-reproduce \
    --pins results/trials_fusion_confirm/pins.json --device cpu \
    --out results/trials_fusion_confirm/gate_reproduction.json
```

Pass condition:
- `selected_weight` matches exactly;
- count leaves match exactly;
- rank leaves are within abs 1e-4;
- continuous leaves are within abs 1e-6;
- no published non-threshold leaf is missing;
- zero-shot macro pooled AUC is 0.795 at 3 dp.

Leaf paths are key tuples, and `per_query` rows are paired by their `query` field. The output records the maximum absolute difference per class and per leaf group. It also reports, without gating, the exploratory restricted-grid estimate. Expect about +0.013.

On the Mac the code check gave exact zeros (see `verify_trials_fusion.md`).

**On failure:**
1. Fix the code.
2. Log a §9 entry.
3. Write a new revision and rerun gate 2 into a new leaf:

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/make_pins.py" --revise --previous <latest pins> --amendment A<m> \
    --protocol "$P" --changed-fields "" --confirmatory-output-seen false \
    --out results/trials_fusion_confirm/pins_r<n>.json
CUDA_VISIBLE_DEVICES="" "$PY" "$S/score.py" --gate-reproduce --pins results/trials_fusion_confirm/pins_r<n>.json \
    --device cpu --out results/trials_fusion_confirm/gate_reproduction_r<n>.json
```

Once gate 2 passes, `pins.code` is frozen. Below, `<PINS>` is the latest revision.

## 6. Training: 19 + 15 runs (GPU), under tmux

```bash
tmux new -s tfc          # reattach later with: tmux attach -t tfc
"$PY" "$S/train.py" --pins <PINS> --cohort 2022 --device cuda 2>&1 | tee -a results/trials_fusion_confirm/train_2022.out
"$PY" "$S/train.py" --pins <PINS> --cohort 2021 --device cuda 2>&1 | tee -a results/trials_fusion_confirm/train_2021.out
```

What `train.py` does:
- issues the exact §6.3 `run_learning_curve_point.py` commands in the fixed §3b order:
  - 2022 (§9 A1, all 19 retrained): F 10, 25, 50, 75, 100, each with s42, s13, s21, and at F 25 and 100 then s87, s123;
  - 2021: k0..k4, each with s13, s21, s42;
- passes `--compare-config config/lc_trials_baseline.json` and `--device cuda`;
- sets `PYTHONHASHSEED=0` for the child;
- writes per-attempt logs to `results/trials_fusion_confirm/train_logs/`.

After each run it checks gate 3:
- exit code 0;
- `lc_manifest.json` and `best_checkpoint.pt` are present;
- no `run_error.json`;
- `unexpected_fields` is exactly `["embedding_cache_path"]` (2022) or `["embedding_cache_path", "trials_split_dir"]` (2021);
- `cache_check` passes;
- the device block is `cuda`/`cuda`, and `cuda_device_name == pins.environment.gpu_name`;
- the original cache hash is unchanged.

It then appends an entry to `run_ledger.json`.

Notes:
- `--cohort 2021` refuses to start until every 2022 run is passed or excluded.
- Runs are never parallelised or sharded across GPUs. The order is part of the condition, because the cache copy grows. `--max-runs N` advances the grid in chunks, and a later call resumes in order.

**Monitoring:**

```bash
"$PY" "$S/train.py" --pins <PINS> --cohort 2021 --dry-run      # plan + ledger status per run, trains nothing
tail -f results/trials_fusion_confirm/train_logs/*.log
"$PY" -c "import json;[print(e['run'],e['status'],e.get('gate3_failures')) for e in json.load(open('results/trials_fusion_confirm/run_ledger.json'))]"
nvidia-smi
```

**Resume** after a dropped session or crash: rerun the same `train.py` command.
- Passed and excluded runs are skipped.
- An output dir without a passing entry is renamed `.failed1` and retried once.
- A second failure is recorded as `excluded`.
- Never edit or delete `run_ledger.json`.

Stop rules:
- Cohort 2022 needs ≥ 17 of 19 checkpoints.
- Cohort 2021 needs ≥ 4 folds with all 3 seeds.

Otherwise stop and amend (§5, §7 declared-change branch).

## 7. Scoring and analysis, once (CPU)

```bash
CUDA_VISIBLE_DEVICES="" "$PY" "$S/score.py" --pins <PINS> \
    --ledger results/trials_fusion_confirm/run_ledger.json --device cpu \
    --out results/trials_fusion_confirm/scores
CUDA_VISIBLE_DEVICES="" "$PY" "$S/analyze.py" --pins <PINS> \
    --scores results/trials_fusion_confirm/scores \
    --out results/trials_fusion_confirm/analysis.json \
    --report results/trials_fusion_confirm/report.md
```

`score.py`:
- re-verifies every pinned hash;
- scores the passing retrained cohort-2022 runs and zero-shot on the 2022 files (the 15 published checkpoints are used only by gate 2, §9 A1);
- scores the passing seeds and zero-shot on each 2021 fold. The 2021 vectors come from the frozen cache. Any 2021 text missing from it is fresh-encoded on CPU into `results/trials_fusion_confirm/prepared_embeddings/trials/`. On the Mac every fold text resolved from the frozen cache, so expect 0 fresh encodes.

`analyze.py` recomputes every selected weight (it must equal `selected_w.json`) and writes the headline first in `report.md`.

On wall time:
- Each 2-epoch CPU smoke run took about 20 s on the Mac.
- 15-epoch GPU runs should take a few minutes each, so 34 runs is likely 1-3 hours in total.
- 2022 scoring took about 30 s per 16 models on the Mac.

These are estimates, not measurements on the remote.

## 8. What to send back

```bash
tar czf trials_fusion_confirm_return.tgz \
  results/trials_fusion_confirm/{transfer_check.json,pins*.json,configs,gate_reproduction*.json,run_ledger.json,train_logs,train_*.out,scores,analysis.json,report.md} \
  results/trials_fusion_confirm/cohort2022 results/trials_fusion_confirm/cohort2021 \
  preprocess/trials_fusion_confirm/cv2021/cv2021_manifest.json preprocess/trials_fusion_confirm/cv2021/fold_*/fold_manifest.json
"$PY" -m pip freeze > pip_freeze.txt; nvidia-smi > nvidia_smi.txt
```

Also send:
- `pip_freeze.txt` and `nvidia_smi.txt`;
- any `*.FAILED.*.json`;
- the run log below.

The checkpoint dirs are small: about 2.6 MB per `best_checkpoint.pt`, plus logs. Do not send the cache copy (about 95 MB, mutable, not needed).

## Run log (fill in on the remote)

| step | date (UTC) | result / notes |
|---|---|---|
| transfer method | | |
| −1 verify_transfer | | |
| 0 gate 1 / folds / pins | | |
| gate 2 | | |
| training 2022 | | |
| training 2021 | | |
| scoring / analysis | | |
