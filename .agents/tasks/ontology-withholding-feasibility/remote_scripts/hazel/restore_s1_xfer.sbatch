#!/bin/bash
#SBATCH --job-name=s1_restore
#SBATCH --partition=xfer
#SBATCH --qos=xfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/share/nesylearning/%u/logs/s1_restore.%j.out
# Bring the VCL S1 state to Hazel: results/go_function_fusion (incl. the partial
# test/ and the Diamond hits), the G8 code pin and pins, and the six Tier B run dirs
# with their checkpoints. The smoke dir is not touched.
set -euo pipefail
source /share/nesylearning/$USER/hazel/env.sh
cd $R
$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("olukotunjosh/cdcl-go-function-results", repo_type="dataset", local_dir=".",
    allow_patterns=["results/go_function_fusion/**", "results/go_function/protocol_pins.json",
                    "results/go_function/code_pin.json*", "results/go_function/specs/**",
                    "results/go_function/runs/*/*/V0__TP__W2__all__kall__s13/**",
                    "results/go_function/runs/*/*/V4__TP__W2__all__kall__s13/**"],
    ignore_patterns=["*.CLAIM", "*.tmp", "*/resume.pt"])
PY
rm -rf .cache/huggingface/download
mkdir -p config/go_function_fusion
$PY - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download("olukotunjosh/cdcl-go-function-results", "remote_logs/s1/input_hashes.json", repo_type="dataset")
shutil.copy(p, "config/go_function_fusion/input_hashes.json")
PY
git status --short | head; ls results/go_function/runs/*/*/ ; ls results/go_function_fusion/test
echo "S1 RESTORE DONE"
