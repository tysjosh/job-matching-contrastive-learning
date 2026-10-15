# Running the GO protein-function study on a remote L40S

Operational guide for the remote gates of `protocol_go_function.md` §15: G5 (ESM-2 and MPNet term text), G4, G7, G8, G9b, G10 and G11, on one Linux L40S (48 GB). It does not change the protocol or `design_go_function.md`; where they differ, they win. Every command runs from the root of the remote clone and every path is repo-relative. Nothing written by the study carries a machine path: configs, manifests, pins, specs and run markers hold repo-relative paths and hashes only, and subprocess logs are scrubbed.

Every block below assumes:

```bash
cd CDCL                                   # the remote clone (any location)
PY=.venv/bin/python
```

Exit codes of every script: 0 ok, 2 fatal, 3 protocol stop (stop and ask the user), 4 refused by state (a pin, E* or a code pin is missing).

## 0. Prerequisites (before anything runs remotely)

- Local gates done on the Mac (protocol §15): G0, G1, G2, G2b, G3, InterPro features (G5 local part), G6 and the G9a smoke. `results/go_function/protocol_pins.json` holds G1-labels, G1-norm, G2, G2b, G3, G5 (InterPro) and G6.
- **The user commits and pushes** the study code (`go_function/`, `scripts/go_function/`, `config/go_function/`) to a branch. No step of this study makes git writes.
- **Sign-offs needed before G8** (protocol §10 G8):
  - D19, closed-world W2/W1 (a deviation from spec §5 "Masking"): **approved 2026-10-12**;
  - plan decision 3, the G9a smoke with reduced batches (`max_train_batches` 40, 2 epochs, CPU).
- **Obsolete go.norm classes (decided, protocol A3).** go.norm names 728 GO classes that are obsolete in the pinned go.obo. `embed_terms.py` gives each its own stanza's text (name without the leading "obsolete ", definition as written), so step 4b runs without a further decision.
- **Pick the ESM-2 revision** (a commit sha of `facebook/esm2_t33_650M_UR50D`). It is pinned at G5.

## 1. Clone and copy the pins

On the remote:

```bash
git clone <repository-url> CDCL && cd CDCL && git checkout <study-branch>
```

From the local repository root (only the pins and gate results; everything else is rebuilt and verified on the remote):

```bash
HOST=user@l40s; DEST="$HOST:CDCL"
rsync -a --relative results/go_function/protocol_pins.json results/go_function/gates "$DEST/"
```

Never ship `/300` (an untracked cookie jar at the repository root) or `reports/`. Do not ship `preprocess/go_function/` or `dataset/go_function/`: they are rebuilt (step 3) and must reproduce the pins exactly.

## 2. Environment

```bash
python3.13 -m venv .venv
$PY -m pip install --upgrade pip
$PY -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # driver < 570: .../whl/cu126
$PY -m pip install -r scripts/go_function/requirements-remote.txt
$PY -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

# models at pinned revisions (full snapshots), then offline
.venv/bin/hf download facebook/esm2_t33_650M_UR50D --revision <ESM_SHA>
.venv/bin/hf download sentence-transformers/all-mpnet-base-v2 --revision e8c3b32edf5434bc2275fc9bab85f82640a19130
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Keep these exports in every shell (and in the tmux session of step 9).

**Hosts with a wall-time cap (for example a 10-hour VCL reservation).** Every training run snapshots its full state to `<run_dir>/resume.pt` after an epoch once 15 minutes have passed since the last snapshot (`GO_FUNCTION_RESUME_EVERY_S`, in seconds). Re-running the same `run.py --spec`, `pilot.py` or `sweep.py run` command on a run dir that holds `resume.pt` continues from that epoch, after checking that the spec, `study.json` and the input hashes are unchanged. Nothing is renamed aside, and `resume.pt` is deleted when the run writes `DONE.json`. Before a reservation ends, copy `results/go_function/` including the `resume.pt` files to the next host (for example a private Hugging Face dataset). Do not copy `*.CLAIM` files: a claim from another host is treated as live. The G0 unit tests on the remote are a cheap check of the install:

```bash
GO_FUNCTION_SLOW=1 $PY -m pytest go_function/tests -q -p no:cacheprovider
```

## 3. Re-download, verify, rebuild

```bash
$PY scripts/go_function/download_dgz.py                     # DGZ release -> dataset/go_function/dgz/ + MANIFEST.json
$PY scripts/go_function/download_dgz_gate.py --only all     # gate files (about 5.2 GB): pinned in input_hashes.json
$PY scripts/go_function/pin_inputs.py --gate                # every file vs config/go_function/input_hashes.json; mismatch = exit 2

for A in bp mf; do
  $PY scripts/go_function/build_labels.py --aspect $A --regime all --verify-pins
  $PY scripts/go_function/build_features.py --aspect $A     # InterPro; pin G5 must match
  $PY scripts/go_function/build_targets.py --aspect $A      # pin G6 must match
  $PY scripts/go_function/make_conditions.py --aspect $A --block all   # every condition + index; pin G6 must match
done
```

`build_labels.py --verify-pins` compares with the shipped G1 pins. The other builders record their pins, and recording a pin that differs from the shipped value is fatal, so a rebuild that does not reproduce the Mac files exits 2 and writes no pin. Disk: dataset about 10.6 GB, `preprocess/go_function/` about 0.4 GB.

## 4. Features on the GPU (G5)

```bash
$PY scripts/go_function/embed_esm.py --device cuda --revision <ESM_SHA>   # about 0.5 h (estimate); fp32 cross-check; pins G5 esm2_650m/*
$PY scripts/go_function/embed_terms.py --device cuda                     # 4b; pins G5 term text
```

`embed_esm.py` refuses an existing output dir (exit 4) and fails on a cross-check cosine below 0.999 (exit 2). `embed_terms.py` gives the 728 obsolete go.norm classes their own stanza text (A3) and records `n_obsolete_text` (728) in `term_text_mpnet.json`; a class with no stanza is still fatal (exit 2).

## 5. G4: trainer reproduction

```bash
$PY scripts/go_function/gate_trainer.py --write-specs
for s in results/go_function/specs/gates/*.json; do $PY scripts/go_function/run.py --spec "$s" --device cuda; done
$PY scripts/go_function/gate_trainer.py               # compare with dgz_published.json (+-0.01); pin G4
```

The V0 gate specs use DGZ's `deepgopro.py` settings (lr 1e-3, milestones [1, 3], dropout 0.3; protocol A6); the V4 gate specs use study.json. If `results/go_function/specs/gates/` or `results/go_function/gates/runs/` still hold V0 specs or runs from before A6 (study.json settings), `--write-specs` exits 2 on the spec mismatch: rename those spec files and run dirs with a `.studyhp1` suffix first (never delete them).

Exit 4 means a seed-42 cell missed and the seed 13/21 fallback runs are needed:

```bash
$PY scripts/go_function/gate_trainer.py --write-specs --fallback
for s in results/go_function/specs/gates/*_s13.json results/go_function/specs/gates/*_s21.json; do
  $PY scripts/go_function/run.py --spec "$s" --device cuda; done
$PY scripts/go_function/gate_trainer.py
```

Exit 3 (the 3-seed mean misses too) is a protocol stop: stop and ask. Optional D10 diagnostic: `--write-specs --diagnostic`, then run `results/go_function/specs/gates/diagnostic/*.json`.

## 6. G7: pilots

```bash
$PY scripts/go_function/pilot.py --aspect bp --feature esm2_650m --device cuda
$PY scripts/go_function/pilot.py --aspect mf --feature esm2_650m --device cuda
$PY scripts/go_function/pilot.py --aspect bp --feature interpro --device cuda
$PY scripts/go_function/pilot.py --aspect mf --feature interpro --device cuda
```

Each writes `results/go_function/pilots/<aspect>_<feature>/choice.json`, sets `epochs_star["<aspect>_<feature>"]` in `config/go_function/study.json` and records pin G7. The ESM pilots also need the term text (step 4b) for the V5/V6 record. Per aspect and feature: 6 E* runs (64 epochs), 16 λ-record runs (E* epochs) and, for ESM, 2 text runs (64 epochs). Runs with `DONE.json` are reused, so a pilot can be restarted. Pilot runs never read test data (protocol §6.4, A4): their `eval.json` is a stub and they write no `scores.npz`. `--stage estar|lambda|text` runs one stage.

`study.json` is tracked and inside the code pin: after G7 the user commits it (or accepts a content pin at step 7).

## 7. G8: specs, audit and code pin

```bash
for A in bp mf; do for F in esm2_650m interpro; do
  $PY scripts/go_function/materialize.py --aspect $A --feature $F
done; done
```

Under protocol A7 (Core A) this writes 522 + 2 = 524 BP and 1 + 2 = 3 MF specs (the preregistered v4 grid was 2,782 BP and 3,157 MF) under `results/go_function/specs/<aspect>/<feature>/`, asserts the per-block counts and runs the single-factor audit (exit 2 on any undeclared difference). Without E* it exits 4.

After the D19 sign-off and the user's commit (study code plus `study.json`):

```bash
$PY scripts/go_function/sweep.py pin                         # needs a clean git status for the study paths
$PY scripts/go_function/sweep.py pin --accept-content-pin    # only if the user explicitly accepts a content pin
```

`pin` writes `results/go_function/code_pin.json` and pin G8 (code pin hash, spec-set hashes, git HEAD). From here, study-mode runs and `analyze.py` refuse any imported module that is not covered by the pin and any pinned file whose on-disk hash changed (checked at preflight and again before `run.json` is written). Do not edit study code, tests or `config/go_function/*.json` after the pin.

## 8. G9b: remote smoke and the 10-day projection

```bash
$PY scripts/go_function/sweep.py run --smoke --feature esm2_650m --device cuda --workers 4 --max-runs 13
$PY scripts/go_function/analyze.py --runs results/go_function/smoke/runs --out results/go_function/smoke/analysis --smoke
cat results/go_function/smoke/smoke_report_esm2_650m.json
```

The smoke runs BP p01 × k ∈ {0, 5} × {V0, V3, V3c, V4, V6} at E*, plus V4 under DGZ with W2, WDGZrel and W2drop at k = 0 (13 runs, mode smoke, full epochs). The report records seconds, peak RSS and peak CUDA memory per run, and the projected grid wall time (5,939 runs × mean seconds / workers). **If the projection exceeds 10 days, the command exits 3: stop and ask the user.** The options are in protocol §11 risk 7: drop the MF noIPI block (−375 runs); W1 to V0 only (−480); MF main block to V0, V1, V4, V5, V6 (−615). Use the peak CUDA memory to choose `--workers` for step 9 (4 is the default).

## 9. G10: the grid, in tmux

```bash
tmux new -s gof
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
W=4
while :; do
  P=$($PY scripts/go_function/sweep.py plan 2>/dev/null | $PY -c 'import json,sys; print(json.load(sys.stdin)["n_pending"])')
  [ "$P" -eq 0 ] && break
  $PY scripts/go_function/sweep.py run --max-runs 200 --workers $W --device cuda || break
done
```

- **Order.** Seed-major: every run of seed 13 first (BP before MF, ESM before InterPro), so an interrupted grid still has complete seeds.
- **Bounded invocations.** `--max-runs` is required. Each invocation starts at most that many runs; the loop repeats until nothing is pending, or stops on any non-zero exit.
- **Shards.** Several processes split the grid with `--shard i/N`. Run index r goes to shard `r % N + 1` over the full ordered grid, so the shards are disjoint and exhaustive; this needs all four spec indexes. Keep N fixed for the whole grid and run one process per shard:

  ```bash
  N=2
  for i in $(seq 1 $N); do
    tmux new-window -t gof "$PY scripts/go_function/sweep.py run --max-runs 200 --shard $i/$N --workers 2 --device cuda"
  done
  ```

- **Claims and resume.** Each run is claimed by `<run_dir>.CLAIM` (O_EXCL, pid and host). A run with `DONE.json` is never re-run, so relaunching the same command resumes.
- **Failures.** A failed attempt is renamed `<run_dir>.attempt<n>` and retried once. A second failure leaves `FAILED.json`, and the run is missing for the analysis. A CUDA OOM lowers the invocation's worker count by one before the retry.
- **Refusals and systemic preflight errors.** A run refused by the study's state (exit 4: norm unset, a pin or the code pin missing) uses no attempt: its dir becomes `<run_dir>.refused<n>`, the run stays pending, and the invocation exits 4, which ends the loop above. Three consecutive runs that fail at stage preflight (for example a pin or code-pin mismatch after an edit) stop the invocation with exit 2. Fix the cause before relaunching. Those few runs are left `failed`; to re-queue one, rename its run dir and its `.attempt1` dir with a `.refused<n>` suffix (the analysis ignores `.refused` dirs).

Monitoring and repair:

```bash
$PY scripts/go_function/sweep.py status                  # counts per state
$PY scripts/go_function/sweep.py status --release-stale  # after a crash/reboot: dead-pid claims released, partial dirs -> .interrupted<n>
$PY scripts/go_function/sweep.py rescore --max-runs 100 --device cuda   # runs whose FAILED.json has stage eval
nvidia-smi; df -h .
tail -f results/go_function/runs/logs/bp/esm2_650m/<run_name>.attempt1.log
```

A claim records pid, host and the host's boot id, so after a reboot a same-host claim is stale even if its pid has been reused. A claim from another host (including a container whose hostname changed) is always treated as live. Release it by hand only after confirming that host is down.

Disk (estimate): about 1 MB per run for run outputs. Kept checkpoints are larger, roughly 0.1–0.3 GB each: the `all`/seed-13 study runs (31) and every pilot run (about 92). Plan on at least 60 GB free beyond the dataset.

## 10. G11: analysis

```bash
$PY scripts/go_function/analyze.py --runs results/go_function/runs --out results/go_function/analysis --seed 20261011 --B 10000
```

The command writes `long_table.csv`, `fusion_F.csv`, `baseline_DS.csv`, `contrasts.json` and `decisions.json`. It exits 2 if a `DONE.json` or `run.json` hash disagrees with its files, and 4 without the code pin. If more than 10% of the main block's (t, k, s, arm) values are missing, every output is labelled exploratory. The analysis needs CPU only. It can also run locally after the return transfer, if `preprocess/go_function/` is present there.

## 11. Return transfer

```bash
find results/go_function/runs \( -name run.json -o -name epochs.jsonl -o -name eval.json \
  -o -name scores.npz -o -name DONE.json -o -name FAILED.json \) -not -path '*.attempt*' \
  -not -path '*.refused*' > /tmp/gof_runs.txt
tar czf go_function_results.tgz --exclude='*.pt' \
  results/go_function/gates results/go_function/pilots results/go_function/specs \
  results/go_function/analysis results/go_function/smoke results/go_function/protocol_pins.json \
  results/go_function/code_pin.json config/go_function/study.json -T /tmp/gof_runs.txt
```

No checkpoints are returned: the kept `all`/seed-13 and pilot checkpoints stay on the remote.

## 12. Time estimate (an estimate, not measured on an L40S)

**This is an estimate.** It extrapolates from the local G9a smoke on an 8-core Apple CPU. No run has been timed on the L40S, and G9b (step 8) replaces this estimate with a measurement.

- **Local G9a, BP InterPro, 40 batches per epoch, 2 epochs, CPU.** A V0 run took about 45 s, mostly start-up, preflight and evaluation. An EL run (V3/V3c) took about 145 s, with about 50–70 s per 40-batch epoch: roughly 1.5 s per batch, dominated by the EL loss over about 110k BP axioms.
- **A full BP epoch is 1,290 batches.** Assuming the L40S runs the EL step 20–50× faster than this CPU, an EL epoch takes 40–100 s and a V0 epoch a few seconds. At E* epochs (8–64, fixed at G7), an EL run takes about 5–30 min and a V0/V5 run about 2–5 min, including about 1–2 min of preflight and evaluation.
- **Grid.** About three quarters of the 5,939 runs are EL arms. At 4 concurrent workers that gives roughly 4–20 days of wall time; the spread comes mostly from E* and the GPU speed-up. The upper part of this range crosses the 10-day G9b stop, so read the G9b projection before launching G10.
- **G4 (step 5), estimate.** Four gate runs of 256 epochs each (selection picks the best epoch, but every epoch is trained), run one after another by the step 5 loop. At the rates above: V4 BP about 3–7 h (256 × 40–100 s), V4 MF about 2–5 h (about 940 batches per epoch, EL-dominated), V0 BP and V0 MF well under 1 h together. Total about 5–13 h. Each missed cell adds its seed-13/21 fallback runs: about 4–14 h more for a V4 cell, under 1 h for a V0 cell; up to about 10–26 h if every cell misses. The optional D10 diagnostic (two V4 runs) adds about 5–12 h.
- ESM embedding: about 0.5 h. Pilots: 4 × (6 + 16 [+ 2]) runs, roughly 1–3 days in total at the same per-epoch rates.
