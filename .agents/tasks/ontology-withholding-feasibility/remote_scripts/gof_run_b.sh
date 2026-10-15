#!/usr/bin/env bash
# Machine B of the pilot split: after gof_setup.sh restores, set A's pilots aside,
# verify the rebuild, run B's pilots, and upload only B's pilot dirs.
export PILOTS="bp_interpro mf_esm2_650m" HOST_TAG=B
until grep -q "SETUP DONE" ~/gof_setup.log 2>/dev/null; do
  pgrep -f "bash /home/ojolukot/gof_setup.sh" > /dev/null || { echo "SETUP FAILED"; exit 1; }; sleep 15; done
cd ~/CDCL; mkdir -p ~/not_on_B
for P in bp_esm2_650m mf_interpro; do [ -d results/go_function/pilots/$P ] && mv results/go_function/pilots/$P ~/not_on_B/; done
ls results/go_function/pilots
setsid nohup bash ~/gof_hf_sync_pilots.sh >> ~/gof_hf_sync.log 2>&1 < /dev/null &
VERIFY=1 bash ~/gof_pilots.sh
