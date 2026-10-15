#!/usr/bin/env bash
# Run a subset of the G7 pilots in parallel on this host, e.g.
#   PILOTS="bp_interpro mf_esm2_650m" bash ~/gof_pilots.sh
# VERIFY=1 first runs the unit tests and the pin-verified rebuild (fresh host).
# Each pilot continues its runs from resume.pt; re-running a finished pilot trains
# nothing. When the pilots are split over two hosts, each host writes only its own
# pilots' E* into study.json and its own G7 pin entries; reconcile afterwards by
# merging the pilot run dirs onto one host and running all four pilots there
# (trains nothing; writes every E* and the G7 pin).
set -uo pipefail
cd ~/CDCL; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG=~/gof_logs; mkdir -p "$LOG"; touch "$LOG/STATUS"
: "${PILOTS:?set PILOTS, e.g. PILOTS=\"bp_interpro mf_esm2_650m\"}"
step() { local name=$1; shift; if grep -qx "$name 0" "$LOG/STATUS"; then echo "=== skip $name"; return 0; fi
         echo "=== $(date '+%F %T') START $name"; "$@" >> "$LOG/$name.log" 2>&1; local rc=$?
         echo "=== $(date '+%F %T') END $name rc=$rc"; echo "$name $rc" >> "$LOG/STATUS"; return $rc; }

if [ "${VERIFY:-0}" = 1 ]; then
  step tests $PY -m pytest go_function/tests -q -p no:cacheprovider || exit 2
  for A in bp mf; do
    step labels_$A     $PY scripts/go_function/build_labels.py --aspect $A --regime all --verify-pins || exit 2
    step features_$A   $PY scripts/go_function/build_features.py --aspect $A || exit 2
    step targets_$A    $PY scripts/go_function/build_targets.py --aspect $A || exit 2
    step conditions_$A $PY scripts/go_function/make_conditions.py --aspect $A --block all || exit 2
  done
  step tests_slow env GO_FUNCTION_SLOW=1 $PY -m pytest go_function/tests -q -p no:cacheprovider || exit 2
fi

pids=()
for P in $PILOTS; do
  A=${P%%_*}; F=${P#*_}
  step "pilot_$P" $PY scripts/go_function/pilot.py --aspect "$A" --feature "$F" --device cuda &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== $(date '+%F %T') PILOTS DONE ($PILOTS) rc=$rc"
exit $rc
