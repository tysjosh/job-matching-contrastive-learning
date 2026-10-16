#!/bin/bash
#SBATCH --job-name=gof_g9b
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=/share/nesylearning/%u/logs/g9b.%j.out
# G9b (protocol §10): the 13-run CUDA smoke at full E* epochs, 4 workers on one
# H200, then analyze.py --smoke. short_gpu caps a job at 2 h: if this job ends
# early, submit it again; every run resumes from its resume.pt. The smoke dir is
# used by this job only, so leftover claims (from a node that is gone) are removed.
set -uo pipefail
source /share/nesylearning/$USER/hazel/env.sh
cd $R
export HF_HUB_OFFLINE=1 GO_FUNCTION_RESUME_EVERY_S=600
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
find results/go_function/smoke -name "*.CLAIM" -print -delete 2>/dev/null
$PY scripts/go_function/sweep.py run --smoke --feature esm2_650m --device cuda --workers 4 --max-runs 13
rc=$?; echo "sweep exit $rc"
if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then
  $PY scripts/go_function/analyze.py --runs results/go_function/smoke/runs --out results/go_function/smoke/analysis --smoke
  cat results/go_function/smoke/smoke_report_esm2_650m.json
fi
# uploads run from the xfer node (upload_xfer.sbatch): GPU nodes have no outbound internet
exit $rc
