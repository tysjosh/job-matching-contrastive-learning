# Running the GO/PPI withholding grid on a remote A100

Operational guide for protocol gates G4 (if not finalised locally), G5, G6 and G7 on a remote Linux GPU machine. It does not change the protocol (`protocol_go_ppi.md`) or the design (`design_go_ppi.md`); where they differ, they win. All paths are relative to the repository root. Nothing in configs, manifests, condition files or pins carries a machine path: study configs use `$ROOT/...` values that resolve against the checkout, and pins hold hashes only.

Every block below assumes this prefix on the remote machine:

```bash
cd /path/to/CDCL                       # the remote checkout (any location)
R="$PWD"; PY="$R/.venv/bin/python"; S="$R/scripts/go_withholding"
```

## 0. Before leaving the local machine

- G0–G3 (and ideally G4) have run locally per protocol §10, so `results/go_withholding/protocol_pins.json` holds G1, G1-leak, G2-targets, G2-cache, G2-covariate, G2-validation, G3 (and G4-lambda, G4-configs).
- The code is committed or tagged, or the content pin is explicitly accepted (protocol G5). Note the local code hash; the remote must reproduce it:

  ```bash
  .venv/bin/python -c "from go_ppi_domain import withholding_constants as wc; print(wc.code_sha256()[0])"
  ```

## 1. Copy list (rsync)

`code_sha256()` runs `git ls-files -co --exclude-standard` over `contrastive_learning/`, `go_ppi_domain/`, `scripts/`, `trials_domain/` and `run_phase1_embedding_evaluation.py`. The remote therefore needs the git metadata, the `.gitignore` files and exactly the same tracked and untracked `.py` files. Copy the working tree, not a fresh clone, so the uncommitted edits come along.

Run from the local repository root:

```bash
HOST=user@a100; DEST="$HOST:/path/to/CDCL"

# 1a. code, git metadata, .gitignore files, configs, protocol (heavy and ignored dirs excluded).
#     '/300' is an untracked root-level cookie jar unrelated to the study: never ship it.
rsync -a --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude 'preprocess/' --exclude 'embedding_cache/' \
  --exclude 'dataset/' --exclude 'training_output/' --exclude 'colab_package.tar.gz' \
  --exclude '/300' \
  ./ "$DEST/"

# 1b. study data: pruned features, splits, target table + covariate, target-free validation
#     (--relative recreates the repo-relative paths under DEST), plus the read-only reference
#     data preprocess/go_ppi + preprocess/go_ppi_splits (~170 MB): G0's test_go_ppi_wiring.py reads them
rsync -a --relative preprocess/go_withholding/converted preprocess/go_withholding/splits \
  preprocess/go_withholding/targets preprocess/go_withholding/validation_targetfree \
  preprocess/go_ppi preprocess/go_ppi_splits "$DEST/"

# 1c. pins (hashes only), eval cache, lambda pilot and code pin (when made locally)
rsync -a --relative results/go_withholding/protocol_pins.json results/go_withholding/eval_cache "$DEST/"
[ -d results/go_withholding/lambda_pilot ] && rsync -a --relative results/go_withholding/lambda_pilot "$DEST/"
[ -f results/go_withholding/code_pin.json ] && rsync -a --relative results/go_withholding/code_pin.json "$DEST/"

# 1d. raw inputs: HuRI / Lit-BM and STRING aliases for the offline negative audit (and a remote G1 rebuild)
rsync -a --relative dataset/go_ppi "$DEST/"

# 1e. the exact MPNet snapshot (revision e8c3b32edf5434bc2275fc9bab85f82640a19130), so the remote
#     encoder is byte-identical to the one that built the pinned eval cache
ssh "$HOST" mkdir -p .cache/huggingface/hub
rsync -a ~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2 "$HOST:.cache/huggingface/hub/"
```

Not copied: `preprocess/go_withholding/conditions/` (about 4.4 GB). It is regenerated on the remote and verified against the G3 pin (step 3). Copying it instead also works; the sweep preflight hashes every condition manifest either way.

Remote G1 rebuild instead of 1b/1c: copy only `preprocess/go_ppi` and `preprocess/go_ppi_splits` from 1b (the reference data, read-only) and run protocol §10 G1–G4 verbatim with `R="$PWD"`.

## 2. Environment

```bash
python3.13 -m venv .venv
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r "$S/requirements-remote.txt"
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

export HF_HUB_OFFLINE=1                 # use the copied MPNet snapshot only
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # avoid CPU oversubscription with parallel workers; keep fixed for the whole grid
```

G0 on the remote (same commands as protocol §10 G0) and the code-hash check:

```bash
"$PY" -m pytest "$R/go_ppi_domain/tests" -q
"$PY" -m pytest "$R/tests/test_career_ordinal_candidates.py" "$R/tests/test_mesh_soft_sampling.py" \
  "$R/tests/test_career_adapter_equivalence.py" "$R/tests/test_career_golden_regression.py" -q
"$PY" -c "from go_ppi_domain import withholding_constants as wc; print(wc.code_sha256()[0])"   # must equal the local value
```

If the hash differs, stop: a file differs or is missing (often an untracked `.py` file or a `.gitignore`).

## 3. Rebuild the conditions, verified against the pins

```bash
"$PY" "$S/regenerate_conditions.py"      # all n_conditions (121 at 24 targets)
```

It verifies the base files against G1, the table against G2-targets and the target-free validation against G2-validation, and checks each condition's manifest sha256 against G3 before writing. Any mismatch exits 2 and nothing is written for that condition. Needs about 4.4 GB; the preflight refuses below 5 GB free.

## 4. λ pilot (only if G4 was not finalised locally)

```bash
"$PY" "$S/materialize_configs.py" --lambda none
"$PY" "$S/lambda_pilot.py" --out "$R/results/go_withholding/lambda_pilot" --device cuda
"$PY" -c "import json;print(json.load(open('$R/results/go_withholding/lambda_pilot/lambda_choice.json'))['lambda'])"
```

`lambda_pilot.py` already calls `materialize_configs.py --lambda <λ>` (protocol §10 G4's last line), which writes the four cell configs and the `G4-configs` pin; running it again would append a second `G4-configs` entry. An exit 3 is a protocol stop (fewer than 4 validation batches, or λ not confirmed): stop and ask.

## 5. G5 pin and G6 smoke

```bash
"$PY" "$S/sweep.py" pin                  # only if code_pin.json was not made locally; refuses if it exists
CACHE=$(ls "$R"/results/go_withholding/eval_cache/test_*.npz)
```

Run the protocol §10 G6 block verbatim (with `--device cuda` appended to the runner and evaluator calls if wanted). If G6 already passed locally on CPU, running the same block once more on the A100 into a different directory (for example `results/go_withholding/smoke_a100/`) is a useful device check: the encoder cross-check (cos ≥ 0.9999 against the CPU-built cache) and `check-group` must pass. That extra run is not a protocol gate.

Readiness check, read-only (full pin preflight, nothing trained):

```bash
"$PY" "$S/sweep.py" run --max-runs 1 --dry-run
"$PY" "$S/sweep.py" plan
```

## 6. Launch the grid (G7)

First warm the shared frozen-encoder cache with one run. The first run in enumeration order is `all__kall__baseline__s13`, whose condition contains every train text, so later runs never write the cache again. The sweep refuses `--workers > 1` or `--shard` until the cache file exists.

```bash
"$PY" "$S/sweep.py" run --max-runs 1 --device cuda
```

Then either one process with parallel workers:

```bash
W=6
nohup "$PY" "$S/sweep.py" run --max-runs 100 --workers "$W" --device cuda > sweep_w.log 2>&1 &
```

or N shard processes, which can share one GPU (or run on several machines over a shared `results/`; see "Crashed host" in §7 for the manual recovery that setup needs):

```bash
N=3; W=2
for i in $(seq 1 "$N"); do
  nohup "$PY" "$S/sweep.py" run --max-runs 100 --shard "$i/$N" --workers "$W" --device cuda > "sweep_s$i.log" 2>&1 &
done
```

Each invocation runs at most 100 runs and then stops; repeat until `status` shows the full count (protocol G7: the grid is a repeated, explicit decision). A repeat loop that stops when nothing is pending or on any non-zero exit (2 refused, 4 grid stopped):

```bash
i=1; N=1; W=6
while :; do
  P=$("$PY" "$S/sweep.py" plan --shard "$i/$N" 2>/dev/null | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["n_pending"])')
  [ "$P" -eq 0 ] && break
  "$PY" "$S/sweep.py" run --max-runs 100 --shard "$i/$N" --workers "$W" --device cuda || break
done
```

Shards split the grid by (condition, seed) group: group g (seed-major) goes to shard `g % N + 1`. They are disjoint, together cover every run, keep a group's four cells together and stay seed-major. Keep `N` fixed for the whole grid.

Only one `sweep.py run` per shard at a time. Concurrent invocations must use disjoint shards (different `i` of the same `N`); never run two unsharded invocations, or two copies of the same shard, side by side. `execute_run` rechecks a run's claim right before it renames a leftover dir, which narrows but does not close the race between overlapping invocations. Use `--workers` for more parallelism inside one invocation instead.

GPU memory: each concurrent run holds its own CUDA context plus MPNet (roughly 1–1.5 GB), so `N × W` up to about 8 fits easily on a 40 GB or 80 GB A100. The limit is usually CPU cores: keep `N × W × OMP_NUM_THREADS` at or below the core count.

## 7. Monitoring, interruption and resume

```bash
"$PY" "$S/sweep.py" status               # also group-checks complete groups; must end with n_groups_unchecked = 0
"$PY" "$S/sweep.py" status --shard 2/3
tail -f "$R/results/go_withholding/sweep_log.jsonl"   # one line per train/eval attempt, with exit_code and wall_seconds
nvidia-smi
```

- **Resume.** Just relaunch the same command. Runs with `DONE.json` are skipped (a no-op). A run dir with neither `DONE.json` nor `FAILED.json` whose `RUNNING.json` claim is absent, unreadable or belongs to a dead process on this host was interrupted: it is renamed `<run_id>.interrupted<n>` (kept, never analysed) and retrained from scratch. A live claim is skipped.
- **Hard reboot.** All claims belong to dead processes, so every partial run restarts cleanly on the next launch.
- **Crashed host (multi-machine only).** A claim from another host is always treated as live, because its pid cannot be probed. If a host dies, its claimed runs stay `in_progress`, and the repeat loop ends with them undone (`status` keeps showing `n_in_progress > 0` while nothing runs). Recovery is manual. First confirm the host is really down and none of its sweep processes survive. Then, for each run dir whose `RUNNING.json` names that host and that has neither `DONE.json` nor `FAILED.json`, rename the dir to the next free `<run_id>.interrupted<n>` (for example `mv runs/<run_id> runs/<run_id>.interrupted1`, raising `n` if that name exists). Never delete it. The next launch on a live host then retrains it from scratch, exactly like an interrupted run on the same host:

  ```bash
  DEAD=deadhost   # hostname recorded in RUNNING.json, after confirming that host is down
  cd "$R/results/go_withholding/runs"
  for d in */; do d=${d%/}
    [ -f "$d/RUNNING.json" ] && [ ! -f "$d/DONE.json" ] && [ ! -f "$d/FAILED.json" ] || continue
    "$PY" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["host"]!=sys.argv[2])' "$d/RUNNING.json" "$DEAD" || continue
    n=1; while [ -e "$d.interrupted$n" ]; do n=$((n+1)); done
    mv "$d" "$d.interrupted$n"
  done
  ```
- **Training or scoring failures** follow design §12.3: one retry, then `FAILED.json`; the driver continues. An evaluation-stage failure can be repaired later with `sweep.py rescore --run-id <run_id>` after the cause is fixed (protocol §10).
- **Exit 4 / `GROUP_FAILED.json`.** The grid is stopped by an identity mismatch. Do not delete anything. Diagnose, record an amendment row in protocol §13, then `sweep.py clear-group` (protocol §3, §10). Stop and ask before doing this.
- **Disk.** About 7.7 GB for 2,420 runs plus 4.4 GB of conditions, so have at least ≈ 12 GB free before regenerating the conditions. `regenerate_conditions.py` / `make_conditions.py` refuse below 5 GB free. The sweep preflight refuses below 3 GB, checked on every invocation; one invocation writes at most 100 runs (≈ 0.32 GB), so that guard does not cover the whole grid. Check `df -h` between invocations.

## 8. Offline audits and statistics

```bash
for D in "$R"/preprocess/go_withholding/conditions/*/; do
  "$PY" "$S/audit_negatives.py" --cond "$(basename "$D")" --cells baseline loss_only neg_only both --seed 13 --epochs 0 7 14
done
"$PY" "$S/analyze_withholding.py" --runs "$R/results/go_withholding/runs" \
  --target-table "$R/preprocess/go_withholding/targets/target_table.json" \
  --covariate "$R/preprocess/go_withholding/targets/text_covariate.json" \
  --out "$R/results/go_withholding/analysis" --seed 20251009 --B 10000
```

The audits and the analysis are CPU-only and cheap; they can also run locally after copying `results/go_withholding/` back:

```bash
rsync -a a100:/path/to/CDCL/results/go_withholding/ results/go_withholding/
```

The analysis exits 2 on any unchecked or `GROUP_FAILED` group and 3 when more than 10% of the 720 (target, k, seed) tuples are missing (`--allow-incomplete` labels everything exploratory).

## 9. Time estimate (rough estimate, not measured on an A100)

**This is an estimate.** It extrapolates from CPU measurements; it has not been timed on the target machine.

- Measured on CPU (Apple M2): an existing GO f100 run took about 278 s for 15 epochs. The design puts the full grid at about 136–257 h of CPU time (2,420 runs).
- Each run is mostly Python-side work: negative selection, batch assembly, per-epoch diagnostics and the cached-embedding evaluation. The frozen MPNet only encodes on cache misses, and the trainable projection head is small. So a GPU shortens a single run only modestly. Rough guess: 90–200 s per run on an A100, including about 20–30 s of start-up and scoring.
- Throughput comes from concurrency. With 4–8 concurrent runs (`N × W`) on a node with enough CPU cores, 2,420 runs would take roughly 8–40 h of wall time; about 15–20 h is a plausible middle.
- To replace the guess with a measurement, look at the `wall_seconds` of the first 10–20 `stage: train` and `stage: eval` lines in `sweep_log.jsonl`. Then: hours ≈ 2,420 × (mean train + eval seconds) / (N × W) / 3600, adjusted for the slowdown concurrent runs cause each other.
