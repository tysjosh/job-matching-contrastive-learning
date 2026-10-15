# Remote run: GO protein-function fusion study (S1 + S2 add-on)

Preregistration: `protocol_go_function_fusion.md` (v3, approved round 3). Code: `go_function_fusion/`, `scripts/go_function_fusion/`, `config/go_function_fusion/`. Every path is repo-relative; run from the repo root. `PY=.venv/bin/python`.

Ground rules: no edit to a protected path (protocol §15); no git commit/stash/reset/checkout/revert/clean by any step; a script that exits 3 stops the study for the user; exit 2 is fatal; exit 4 is a refusal (nothing written). Validation steps run in search mode (the guard blocks every open of a test artefact; a violation exits 2 and is logged in `search_log.jsonl`). `score_test.py` runs once.

## 0. Mac, before the push (CPU, `CUDA_VISIBLE_DEVICES=""`)

1. **G-0:** `CUDA_VISIBLE_DEVICES="" $PY -m pytest go_function_fusion/tests -q`; `git status --porcelain` shows no protected path.
2. **G-R(b):** transcribe `config/go_function_fusion/dgz_published_ds.json` (DGZ Tables 3–4, MLP+DS and DGZ+DS cells, with cell references) first. Then per (aspect, base) in {bp, mf} × {mlp, deepgozero}:
   `CUDA_VISIBLE_DEVICES="" $PY scripts/go_function_fusion/gate_repro.py --part b --aspect <a> --base <b> --pilot-dir /tmp/gofunc_fusion --results results/go_function_fusion`
   (BP needs about 3 GB RAM; `df -h .` first.) Exit 3 = stop and ask.
3. **G-R(c):** same with `--part c` (reported, not gated).
4. **G-R(a)** reads the released test predictions and their labels (the GF G2 exposure only, no fusion): run it on the remote (step 2 below) with `--allow-g2-test-read`, or on the Mac only if the user allows a test-label read there.
5. **G-V:** `CUDA_VISIBLE_DEVICES="" $PY scripts/go_function_fusion/search_valid.py --aspect <a> --base <b> --sim /tmp/gofunc_fusion/swissprot_exp.sim --results results/go_function_fusion` for the four Tier A pairs.
6. **G-F:** `$PY scripts/go_function_fusion/freeze_finalists.py --results results/go_function_fusion` (writes `finalists_tierA.json`, `code_pin.json`, pin G-F). Copy the hashes into protocol §13 by hand.
7. **G-S2dry:** `CUDA_VISIBLE_DEVICES="" $PY scripts/go_function_fusion/s2_grid_fusion.py --smoke --fusion-results results/go_function_fusion` (label-free F parity on `fusion_F.csv`; smoke, non-informative).
8. The user commits and pushes a branch (never main). `results/` is gitignored, so ship `results/go_function_fusion/` (small: per-term tables, pins) with `git add -f results/go_function_fusion`; never ship `/300` or `reports/`. The remote also needs GF's `results/go_function/protocol_pins.json` (G1-labels pin, checked by the fusion label loaders) and `results/go_function/gates/gate_dgz_eval.json` (G-R(a)): `git add -f` both, or copy them over before step 2.

Dev runs of this step's implementation went to `results/go_function_fusion_dev/` and are not gate outputs.

## 1. Transfer and environment (remote)

```bash
git clone -b <branch> <repo> CDCL && cd CDCL
python3.13 -m venv .venv && .venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r scripts/go_function/requirements-remote.txt
.venv/bin/python scripts/go_function/download_dgz.py          # then verify against config/go_function/input_hashes.json
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PY=.venv/bin/python
$PY scripts/go_function_fusion/make_pins.py code --results results/go_function_fusion > /tmp/code_now.json
$PY - <<'EOF'
import json; a=json.load(open("results/go_function_fusion/code_pin.json"))["code"]; b=json.load(open("/tmp/code_now.json"))
assert a == b, sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k)); print("code pin OK")
EOF
```

Rebuild `preprocess/go_function/{bp,mf}/labels_TP.*` with the GF scripts (`build_labels.py --verify-pins`); the fusion loaders check the G1-labels pin.

## 2. G-R(a) (remote, optional here)

`$PY scripts/go_function_fusion/gate_repro.py --part a --aspect bp --allow-g2-test-read --results results/go_function_fusion` (then mf). Every value must equal `results/go_function/gates/gate_dgz_eval.json` to 1e-9.

## 3. G-D: train-only DIAMOND (CPU)

Install the DIAMOND release chosen at G-D (record its version and sha256), then per aspect:

```bash
bash scripts/go_function_fusion/diamond_valid.sh bp $(command -v diamond)
$PY scripts/go_function_fusion/make_pins.py diamond --aspect bp --diamond-version "$(diamond version)" --diamond-binary "$(command -v diamond)"
# repeat for mf
```

Fatal checks: every hit is a train protein; no query id is a train id. The pin fills `config/go_function_fusion/input_hashes.json` `diamond_valid_by_aspect`.

## 4. G-TP: Tier B bases (V0-InterPro, V4-InterPro, V0-ESM; `all`, seed 13)

- Source (a), GF G10 wrote them: `results/go_function/runs/<aspect>/<feature>/<run>/checkpoint.pt`.
- Source (b), otherwise (needs GF G7 and G8): `$PY scripts/go_function_fusion/train_tp_base.py --spec results/go_function/specs/<aspect>/<feature>/<run>.json --device cuda` (writes `results/go_function_fusion/tp_bases/<run>/` with `tp_base_provenance.json`; never open its `eval.json` / `scores.npz`).

Verify and pin each: `$PY scripts/go_function_fusion/verify_tp_bases.py --name V0IPR --aspect bp --spec <spec> --checkpoint <ckpt> --source a|b --device cuda --results results/go_function_fusion`. It checks the spec (variant/feature/aspect, study mode, `all`, seed 13), the checkpoint and spec sha256, for source (b) the provenance (spec sha256, exit code 0, `go_function/*.py` hashes equal to `results/go_function/code_pin.json`), records the §10.4 exposure facts (listings only), then in search mode does the strict load and two validation forwards (max abs diff 0). Any failure exits 2 and pins nothing. If neither source works: stop and ask (drop or defer Tier B).

## 5. G-V2 / G-F2 (search mode)

```bash
for a in bp mf; do for b in mlp deepgozero; do
  $PY scripts/go_function_fusion/search_valid.py --aspect $a --base $b --diamond-valid --results results/go_function_fusion --device cuda
done; for b in V0IPR V4IPR V0ESM; do
  $PY scripts/go_function_fusion/search_valid.py --aspect $a --base $b --results results/go_function_fusion --device cuda
done; done
$PY scripts/go_function_fusion/freeze_finalists.py --remote --results results/go_function_fusion
```

The Tier A `--diamond-valid` reruns write to `search/<aspect>_<base>_dv/` (the Mac G-F inputs in `search/<aspect>_<base>/` are untouched). They build the w_H, shuffled-hits and w_R grids at the G-F frozen (c*, τ*) read from `finalists_tierA.json` (hash-checked against the G-F pin, protocol §3.2) and record whether the rerun's own pass-1 AP-T selection equals it (`apt_star_equal_gf`, also in `finalists_remote.json` as `rerun_apt_equal_gf`; descriptive). G-F2 refuses (exit 2) if any `_dv` w_H grid was not built at the G-F cell. G-F2 refuses if the code differs from the G-F pin (a §14 amendment needs the user) and records `gf_grid_snapshot` for the §11.2 rule.

If Tier B is unavailable when G-T is otherwise ready: stop and ask (Tier A only, or wait).

## 6. G-T: single test scoring

```bash
$PY scripts/go_function_fusion/score_test.py --results results/go_function_fusion --device cuda
$PY scripts/go_function_fusion/report.py --results results/go_function_fusion
```

`score_test.py` refuses on macOS, without CUDA, without the G-F/G-D/G-F2 pins, on a code-hash difference, or if `results/go_function_fusion/test/` exists. It writes `test/test_read.json` first. After a crash, rerun only with the identical command plus `--rerun-identical` (logged). Outputs: `test/` per-term and per-protein tables, `dgz_faithful_<aspect>.json` (descriptive); `analysis/contrasts.json`, `analysis/decisions.json`.

## 7. G-S2 (after GF G10)

`$PY scripts/go_function_fusion/s2_grid_fusion.py --runs results/go_function/runs --with-labels --out results/go_function_fusion/s2 --fusion-results results/go_function_fusion` (descriptive; two-way pack × seed bootstrap, B = 10,000, seed 20261011; remote parameters only if the §11.2 snapshot rule allows).

## 8. Return

A tarball of `results/go_function_fusion/` without checkpoints, `tp_bases/*/checkpoint.pt` or Diamond DBs (`*.dmnd`, `*.fa`).
