#!/bin/bash
#SBATCH --job-name=gof_verify
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/nesylearning/%u/logs/verify.%j.out
# Unit tests, then rebuild every input against the pins (must report unchanged).
set -euo pipefail
source /share/nesylearning/$USER/hazel/env.sh
cd $R
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
$PY -m pytest go_function/tests -q -p no:cacheprovider
for A in bp mf; do
  $PY scripts/go_function/build_labels.py --aspect $A --regime all --verify-pins
  $PY scripts/go_function/build_features.py --aspect $A
  $PY scripts/go_function/build_targets.py --aspect $A
  $PY scripts/go_function/make_conditions.py --aspect $A --block all
done
GO_FUNCTION_SLOW=1 $PY -m pytest go_function/tests -q -p no:cacheprovider
echo "VERIFY CPU DONE"
