#!/bin/bash
#SBATCH --job-name=gof_restore
#SBATCH --partition=xfer
#SBATCH --qos=xfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/share/nesylearning/%u/logs/restore.%j.out
# After `hf auth login`: results/ and preprocess/ from the private HF dataset.
set -euo pipefail
source /share/nesylearning/$USER/hazel/env.sh
cd $R
$PY -c "from huggingface_hub import HfApi; print('HF user', HfApi().whoami()['name'])"
$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("olukotunjosh/cdcl-go-function-results", repo_type="dataset", local_dir=".",
    allow_patterns=["preprocess/go_function/**", "results/go_function/**"],
    ignore_patterns=["*.CLAIM", "*.tmp", "*/resume.pt", "*/checkpoint.pt", "results/go_function/runs/**"])
PY
rm -rf .cache/huggingface/download
echo "RESTORE XFER DONE"
