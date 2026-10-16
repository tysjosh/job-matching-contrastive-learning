#!/bin/bash
#SBATCH --job-name=gof_grid
#SBATCH --partition=gpu_partners
#SBATCH --qos=scavenger_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=/share/nesylearning/%u/logs/grid.%j.out
# Core A grid shard on one H200 (scavenger_gpu, preemptable with resume.pt checkpointing).
# Expects: SHARD=i/N  e.g. 1/4  (set via --export or sbatch --export=SHARD=1/4)
set -uo pipefail
source /share/nesylearning/$USER/hazel/env.sh
cd $R
export HF_HUB_OFFLINE=1 GO_FUNCTION_RESUME_EVERY_S=600
echo "== $(date '+%F %T') SHARD=$SHARD on $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
find results/go_function/runs -name "*.CLAIM" -print -delete 2>/dev/null || true
$PY scripts/go_function/sweep.py run --shard "$SHARD" --device cuda --workers 4 --max-runs 9999
rc=$?; echo "== $(date '+%F %T') sweep exit $rc"
exit $rc
