#!/usr/bin/env bash
# S1 (GO function fusion) CPU-only remote steps, REMOTE_RUN_GO_FUNCTION_FUSION.md §1-3:
# code-pin check, G-R(a) (G2 reproduction with the test read, --allow-g2-test-read),
# G-D (train-only DIAMOND, valid + test queries) and its pins. Run beside training as
#   setsid nohup nice -n 19 taskset -c 4-15 bash ~/gof_s1_cpu_steps.sh > ~/gof_logs/s1.log 2>&1 &
set -euo pipefail
cd ~/CDCL; export PY=.venv/bin/python CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
DV=v2.2.8; DDIR=~/tools/diamond-$DV
echo "== $(date '+%F %T') code pin"
$PY scripts/go_function_fusion/make_pins.py code --results results/go_function_fusion > /tmp/code_now.json
$PY - <<'P'
import json; a=json.load(open("results/go_function_fusion/code_pin.json"))["code"]; b=json.load(open("/tmp/code_now.json"))
assert a == b, sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k)); print("code pin OK")
P
for A in bp mf; do
  echo "== $(date '+%F %T') G-R(a) $A"
  $PY scripts/go_function_fusion/gate_repro.py --part a --aspect $A --allow-g2-test-read --results results/go_function_fusion
done
echo "== $(date '+%F %T') DIAMOND $DV"
mkdir -p $DDIR
[ -x $DDIR/diamond ] || { curl -sSL -o $DDIR/diamond-linux64.tar.gz https://github.com/bbuchfink/diamond/releases/download/$DV/diamond-linux64.tar.gz
  tar -xzf $DDIR/diamond-linux64.tar.gz -C $DDIR; }
sha256sum $DDIR/diamond-linux64.tar.gz $DDIR/diamond | tee $DDIR/SHA256SUMS
$DDIR/diamond version
for A in bp mf; do
  echo "== $(date '+%F %T') G-D $A"
  bash scripts/go_function_fusion/diamond_valid.sh $A $DDIR/diamond
  $PY scripts/go_function_fusion/make_pins.py diamond --aspect $A --diamond-version "$($DDIR/diamond version)" --diamond-binary $DDIR/diamond
done
echo "== $(date '+%F %T') upload S1 outputs"
unset HF_HUB_OFFLINE
.venv/bin/hf upload olukotunjosh/cdcl-go-function-results results/go_function_fusion results/go_function_fusion \
  --repo-type dataset --exclude "*.dmnd" "*.fa" "*.CLAIM" "*.tmp" --commit-message "S1 G-R(a) + G-D $(date +%F_%H%M)" | tail -1
.venv/bin/hf upload olukotunjosh/cdcl-go-function-results config/go_function_fusion/input_hashes.json \
  remote_logs/s1/input_hashes.json --repo-type dataset --commit-message "S1 G-D input_hashes" | tail -1
echo "== $(date '+%F %T') S1 CPU STEPS DONE"
