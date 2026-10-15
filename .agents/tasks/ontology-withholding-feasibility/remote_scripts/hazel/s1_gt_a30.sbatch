#!/bin/bash
#SBATCH --job-name=s1_gt
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-12:00:00
#SBATCH --output=/share/nesylearning/%u/logs/s1_gt.%j.out
# GO function fusion S1 gate G-T, rerun once (--rerun-identical) after the VCL run
# was stopped before its reservation ended (test/STOPPED_vcl.json). The identical
# command, code hashes and pins are checked by score_test.py and the rerun is
# logged in test/test_read.rerun<n>.json. Then report.py, then upload.
set -euo pipefail
source /share/nesylearning/$USER/hazel/env.sh
cd $R
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee results/go_function_fusion/test/gpu_hazel.txt
$PY scripts/go_function_fusion/score_test.py --results results/go_function_fusion --device cuda --rerun-identical
$PY scripts/go_function_fusion/report.py --results results/go_function_fusion
unset HF_HUB_OFFLINE
.venv/bin/hf upload olukotunjosh/cdcl-go-function-results results/go_function_fusion results/go_function_fusion \
  --repo-type dataset --exclude "*.dmnd" "*.fa" "*.CLAIM" "*.tmp" --commit-message "S1 G-T + report (Hazel) $(date +%F_%H%M)" | tail -1
echo "S1 GT DONE"
