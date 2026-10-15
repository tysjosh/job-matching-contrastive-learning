#!/usr/bin/env bash
# Second host of a pilot split: every 30 min upload only this host's pilot dirs
# (PILOTS) and its logs. It never uploads protocol_pins.json or other shared files
# and never squashes (the main host's gof_hf_sync.sh does that).
cd ~/CDCL; unset HF_HUB_OFFLINE; R=olukotunjosh/cdcl-go-function-results
: "${PILOTS:?set PILOTS}"; TAG=${HOST_TAG:-$(hostname -s)}
while :; do
  echo "== $(date '+%F %T') sync $PILOTS"
  for P in $PILOTS; do
    d=results/go_function/pilots/$P
    [ -d "$d" ] && .venv/bin/hf upload $R "$d" "$d" --repo-type dataset \
      --exclude "*.CLAIM" "*.tmp" --commit-message "sync $TAG $P $(date +%F_%H%M)" 2>&1 | tail -1
  done
  .venv/bin/hf upload $R ~/gof_logs remote_logs/gof_logs_${TAG}_$(date +%F) --repo-type dataset \
    --commit-message "logs $TAG $(date +%F_%H%M)" 2>&1 | tail -1
  sleep ${SYNC_EVERY:-1800}
done
