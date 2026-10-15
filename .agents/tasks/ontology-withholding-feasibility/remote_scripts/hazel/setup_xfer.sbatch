#!/bin/bash
#SBATCH --job-name=gof_setup
#SBATCH --partition=xfer
#SBATCH --qos=xfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/share/nesylearning/%u/logs/setup.%j.out
# Downloads only (xfer partition): uv + Python 3.13.5, the clone (+ the bundle of
# unpushed commits), the venv, pinned models, the DGZ release and its pin check.
set -euo pipefail
source /share/nesylearning/$USER/hazel/env.sh
cd $S
[ -x $S/uv/bin/uv ] || curl -LsSf https://astral.sh/uv/0.8.17/install.sh | env UV_INSTALL_DIR=$S/uv/bin UV_NO_MODIFY_PATH=1 sh
$S/uv/bin/uv python install 3.13.5
[ -d $R ] || GIT_TERMINAL_PROMPT=0 git clone -q --branch orca-experiments-and-runner https://github.com/tysjosh/job-matching-contrastive-learning.git $R
cd $R
if [ -f $S/orca.bundle ]; then git fetch -q $S/orca.bundle orca-experiments-and-runner && git merge -q --ff-only FETCH_HEAD; fi
git log --oneline -1
[ -x $PY ] || $S/uv/bin/uv venv --python 3.13.5 --seed .venv -q
$PY -m pip install -q --upgrade pip
$PY -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q -r scripts/go_function/requirements-remote.txt
.venv/bin/hf download facebook/esm2_t33_650M_UR50D --revision 08e4846e537177426273712802403f7ba8261b6c > /dev/null
.venv/bin/hf download sentence-transformers/all-mpnet-base-v2 --revision e8c3b32edf5434bc2275fc9bab85f82640a19130 > /dev/null
$PY scripts/go_function/download_dgz.py
$PY scripts/go_function/download_dgz_gate.py --only all
$PY scripts/go_function/pin_inputs.py --gate
echo "SETUP XFER DONE"
