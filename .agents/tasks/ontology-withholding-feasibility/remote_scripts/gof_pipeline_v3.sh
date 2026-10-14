#!/usr/bin/env bash
# Verify rebuild, then G4 (gate runs in parallel; finished runs skipped, resume.pt dirs
# continued) and G7 (pilots in parallel, then a sequential pass).
set -uo pipefail
cd ~/CDCL; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG=~/gof_logs; mkdir -p "$LOG"; touch "$LOG/STATUS"; sed -i "/^STOPPED /d" "$LOG/STATUS"
step() { local name=$1; shift; if grep -qx "$name 0" "$LOG/STATUS"; then echo "=== skip $name"; return 0; fi
         echo "=== $(date '+%F %T') START $name"; "$@" >> "$LOG/$name.log" 2>&1; local rc=$?
         echo "=== $(date '+%F %T') END $name rc=$rc"; echo "$name $rc" >> "$LOG/STATUS"; return $rc; }
stop() { echo "=== STOPPED at $1 (rc=$2)"; echo "STOPPED $1 $2" >> "$LOG/STATUS"; exit "$2"; }
parallel() { local label=$1; shift; local pids=() names=() i fail=0
  for job in "$@"; do local name=${job%%$'\t'*} cmd=${job#*$'\t'}
    step "$name" bash -c "$cmd" & pids+=($!); names+=("$name"); done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "=== $label: ${names[$i]} failed"; fail=1; }; done
  return $fail; }
gate_job() { local s=$1 n; n=$(basename "$s" .json)
  printf '%s\t%s' "g4_run_$n" "[ -f results/go_function/gates/runs/$n/DONE.json ] || $PY scripts/go_function/run.py --spec $s --device cuda"; }

step tests $PY -m pytest go_function/tests -q -p no:cacheprovider || stop tests $?
for A in bp mf; do
  step labels_$A     $PY scripts/go_function/build_labels.py --aspect $A --regime all --verify-pins || stop labels_$A $?
  step features_$A   $PY scripts/go_function/build_features.py --aspect $A || stop features_$A $?
  step targets_$A    $PY scripts/go_function/build_targets.py --aspect $A || stop targets_$A $?
  step conditions_$A $PY scripts/go_function/make_conditions.py --aspect $A --block all || stop conditions_$A $?
done
step tests_slow env GO_FUNCTION_SLOW=1 $PY -m pytest go_function/tests -q -p no:cacheprovider || stop tests_slow $?

step g4_specs $PY scripts/go_function/gate_trainer.py --write-specs || stop g4_specs $?
jobs=(); for s in results/go_function/specs/gates/*_s42.json; do jobs+=("$(gate_job "$s")"); done
parallel g4 "${jobs[@]}" || stop g4_runs 2
step g4_compare $PY scripts/go_function/gate_trainer.py; rc=$?
if [ $rc -eq 4 ]; then
  step g4_fallback_specs $PY scripts/go_function/gate_trainer.py --write-specs --fallback || stop g4_fallback_specs $?
  jobs=(); for s in results/go_function/specs/gates/*_s13.json results/go_function/specs/gates/*_s21.json; do
    [ -e "$s" ] && jobs+=("$(gate_job "$s")"); done
  parallel g4_fallback "${jobs[@]}" || stop g4_fallback_runs 2
  step g4_compare_fallback $PY scripts/go_function/gate_trainer.py; rc=$?
fi
[ $rc -eq 0 ] || stop g4 $rc

jobs=(); for A in bp mf; do for F in esm2_650m interpro; do
  jobs+=("pilot_par_${A}_$F"$'\t'"$PY scripts/go_function/pilot.py --aspect $A --feature $F --device cuda"); done; done
parallel g7 "${jobs[@]}" || echo "=== g7: a parallel pilot failed; the sequential pass decides"
for A in bp mf; do for F in esm2_650m interpro; do
  step pilot_${A}_$F $PY scripts/go_function/pilot.py --aspect $A --feature $F --device cuda || stop pilot_${A}_$F $?
done; done
echo "=== $(date '+%F %T') PIPELINE DONE through G7"; echo "DONE G7" >> "$LOG/STATUS"
