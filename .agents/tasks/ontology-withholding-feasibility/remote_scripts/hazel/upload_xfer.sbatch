#!/bin/bash
#SBATCH --job-name=gof_upload
#SBATCH --partition=xfer
#SBATCH --qos=xfer
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/share/nesylearning/%u/logs/upload.%j.out
# Uploads from the transfer node: GPU compute nodes cannot reach huggingface.co.
# Usage: sbatch upload_xfer.sbatch <local dir under CDCL> <path in the HF dataset> <message>
set -euo pipefail
source /share/nesylearning/$USER/hazel/env.sh
cd $R
.venv/bin/hf upload olukotunjosh/cdcl-go-function-results "$1" "$2" --repo-type dataset \
  --exclude "*.CLAIM" "*.tmp" --commit-message "$3" | tail -1
echo "UPLOAD DONE $1 -> $2"
