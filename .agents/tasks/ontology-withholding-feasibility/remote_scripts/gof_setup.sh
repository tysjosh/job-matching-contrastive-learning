#!/usr/bin/env bash
# Fresh host: env + clone + public downloads, then wait for `hf auth login`, then
# restore results/ and preprocess/ from the private HF dataset and start the pipeline.
set -euo pipefail
LOG=~/gof_logs; mkdir -p $LOG
curl -LsSf https://astral.sh/uv/0.8.17/install.sh | env UV_NO_MODIFY_PATH=1 sh > $LOG/uv.log 2>&1
~/.local/bin/uv python install 3.13.5 >> $LOG/uv.log 2>&1
cd ~; [ -d CDCL ] || GIT_TERMINAL_PROMPT=0 git clone -q --branch orca-experiments-and-runner https://github.com/tysjosh/job-matching-contrastive-learning.git CDCL
cd CDCL; git log --oneline -1
~/.local/bin/uv venv --python 3.13.5 --seed .venv -q
PY=.venv/bin/python
$PY -m pip install -q --upgrade pip
$PY -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q -r scripts/go_function/requirements-remote.txt
echo "ENV READY: run 'cd ~/CDCL && .venv/bin/hf auth login' now"
.venv/bin/hf download facebook/esm2_t33_650M_UR50D --revision 08e4846e537177426273712802403f7ba8261b6c > $LOG/hf_esm.log 2>&1
.venv/bin/hf download sentence-transformers/all-mpnet-base-v2 --revision e8c3b32edf5434bc2275fc9bab85f82640a19130 > $LOG/hf_mpnet.log 2>&1
$PY scripts/go_function/download_dgz.py > $LOG/download_dgz.log 2>&1
$PY scripts/go_function/download_dgz_gate.py --only all > $LOG/download_gate.log 2>&1
$PY scripts/go_function/pin_inputs.py --gate > $LOG/pin_inputs.log 2>&1
$PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); print('cuda ok', torch.cuda.get_device_name(0))"
echo "waiting for the Hugging Face login"
until .venv/bin/hf auth whoami > /dev/null 2>&1; do sleep 15; done
echo "restoring from Hugging Face"
$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("olukotunjosh/cdcl-go-function-results", repo_type="dataset", local_dir=".",
    allow_patterns=["preprocess/go_function/**", "results/go_function/**"],
    ignore_patterns=["*.CLAIM", "*.tmp"])
PY
rm -rf .cache/huggingface/download
# a finished run never needs its snapshot (train.run deletes it after DONE.json)
for d in $(find results/go_function -name DONE.json -printf '%h\n'); do rm -f "$d/resume.pt"; done
ls results/go_function/gates/runs/
echo SETUP DONE
