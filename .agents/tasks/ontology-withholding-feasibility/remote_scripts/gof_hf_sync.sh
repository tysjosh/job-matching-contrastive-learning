#!/usr/bin/env bash
# Every 30 min: results/go_function (incl. resume.pt, no CLAIMs) + logs -> HF.
# Every 3 h (reservations last 10 h): squash the history and delete
# orphaned LFS objects (old resume.pt versions) right after an upload.
cd ~/CDCL; unset HF_HUB_OFFLINE; R=olukotunjosh/cdcl-go-function-results
last_squash=$(date +%s)
while :; do
  echo "== $(date '+%F %T') sync"
  .venv/bin/hf upload $R results/go_function results/go_function --repo-type dataset \
    --exclude "*.CLAIM" "*.tmp" --commit-message "sync $(date +%F_%H%M)" 2>&1 | tail -1
  .venv/bin/hf upload $R ~/gof_logs remote_logs/gof_logs_$(date +%F) --repo-type dataset \
    --commit-message "logs $(date +%F_%H%M)" 2>&1 | tail -1
  if [ $(( $(date +%s) - last_squash )) -ge ${SQUASH_EVERY:-10800} ]; then
    echo "== $(date '+%F %T') squash"; .venv/bin/python ~/gof_hf_squash.py 2>&1 | tail -1
    last_squash=$(date +%s)
  fi
  sleep ${SYNC_EVERY:-1800}
done
