#!/usr/bin/env bash
# One 10-hour VCL reservation for the GO function fusion study's single test scoring
# (G-T), rerun with --rerun-identical after the previous VCL run was stopped
# (test/STOPPED_vcl.json). Fresh host: env, DGZ data + pin check, wait for
# `hf auth login`, restore only what G-T needs, then score_test.py, report.py, upload.
# Progress: ~/gof_s1_gt.log; the bundle ~/orca.bundle brings the unpushed commits.
set -euo pipefail
LOG=~/gof_logs; mkdir -p $LOG
echo "== $(date '+%F %T') env"
curl -LsSf https://astral.sh/uv/0.8.17/install.sh | env UV_NO_MODIFY_PATH=1 sh > $LOG/uv.log 2>&1
~/.local/bin/uv python install 3.13.5 >> $LOG/uv.log 2>&1
cd ~; [ -d CDCL ] || GIT_TERMINAL_PROMPT=0 git clone -q --branch orca-experiments-and-runner https://github.com/tysjosh/job-matching-contrastive-learning.git CDCL
cd CDCL
[ -f ~/orca.bundle ] && git fetch -q ~/orca.bundle orca-experiments-and-runner && git merge -q --ff-only FETCH_HEAD
git log --oneline -1
~/.local/bin/uv venv --python 3.13.5 --seed .venv -q
PY=.venv/bin/python
$PY -m pip install -q --upgrade pip
$PY -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q -r scripts/go_function/requirements-remote.txt
echo "ENV READY: run 'cd ~/CDCL && .venv/bin/hf auth login' now"
echo "== $(date '+%F %T') DGZ data"
$PY scripts/go_function/download_dgz.py > $LOG/download_dgz.log 2>&1
$PY scripts/go_function/download_dgz_gate.py --only all > $LOG/download_gate.log 2>&1
$PY scripts/go_function/pin_inputs.py --gate > $LOG/pin_inputs.log 2>&1
$PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); print('cuda ok', torch.cuda.get_device_name(0))"
echo "== $(date '+%F %T') waiting for the Hugging Face login"
until $PY -c "from huggingface_hub import HfApi; HfApi().whoami()" > /dev/null 2>&1; do sleep 15; done
echo "== $(date '+%F %T') restore"
$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("olukotunjosh/cdcl-go-function-results", repo_type="dataset", local_dir=".",
    allow_patterns=["preprocess/go_function/**", "results/go_function_fusion/**",
                    "results/go_function/protocol_pins.json", "results/go_function/code_pin.json*",
                    "results/go_function/specs/**",
                    "results/go_function/runs/*/*/V0__TP__W2__all__kall__s13/**",
                    "results/go_function/runs/*/*/V4__TP__W2__all__kall__s13/**"],
    ignore_patterns=["*.CLAIM", "*.tmp", "*/resume.pt"])
PY
rm -rf .cache/huggingface/download
ls results/go_function/runs/*/*/ results/go_function_fusion/test
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "== $(date '+%F %T') G-T rerun"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > results/go_function_fusion/test/gpu_vcl_rerun.txt 2>&1 || true
$PY scripts/go_function_fusion/score_test.py --results results/go_function_fusion --device cuda --rerun-identical > $LOG/s1_gt_rerun.log 2>&1
echo "== $(date '+%F %T') report"
$PY scripts/go_function_fusion/report.py --results results/go_function_fusion > $LOG/s1_report.log 2>&1
unset HF_HUB_OFFLINE
.venv/bin/hf upload olukotunjosh/cdcl-go-function-results results/go_function_fusion results/go_function_fusion \
  --repo-type dataset --exclude "*.dmnd" "*.fa" "*.CLAIM" "*.tmp" --commit-message "S1 G-T rerun + report (VCL) $(date +%F_%H%M)" | tail -1
.venv/bin/hf upload olukotunjosh/cdcl-go-function-results $LOG remote_logs/gof_logs_vcl_s1gt_$(date +%F) --repo-type dataset --commit-message "S1 G-T logs" | tail -1
echo "== $(date '+%F %T') S1 GT DONE"
