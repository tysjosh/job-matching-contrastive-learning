#!/usr/bin/env bash
# One host: after gof_setup.sh restores, verify the rebuild and run only the E* stage
# of the two BP pilots (pending an E*-rule decision, no lambda/text stage runs).
until grep -q "SETUP DONE" ~/gof_setup.log 2>/dev/null; do
  pgrep -f "bash /home/ojolukot/gof_setup.sh" > /dev/null || { echo "SETUP FAILED"; exit 1; }; sleep 15; done
setsid nohup bash ~/gof_hf_sync.sh >> ~/gof_hf_sync.log 2>&1 < /dev/null &
VERIFY=1 STAGE=estar PILOTS="bp_esm2_650m bp_interpro" bash ~/gof_pilots.sh
