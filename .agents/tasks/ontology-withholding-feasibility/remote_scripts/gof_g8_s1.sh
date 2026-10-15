#!/usr/bin/env bash
# GO function G8 (Core A, protocol A7) and the GO function fusion study S1 remote steps
# (REMOTE_RUN_GO_FUNCTION_FUSION.md §4-6): materialize + code pin, the six Tier B
# grid cells (source a), verify_tp_bases, G-V2/G-F2, then the single test scoring G-T.
# Stops at the first non-zero exit (3 = protocol stop: ask the user). Re-running
# skips finished steps (~/gof_logs/STATUS). G9b/G10 are NOT started here.
set -uo pipefail
cd ~/CDCL; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG=~/gof_logs; mkdir -p "$LOG"; touch "$LOG/STATUS"
step() { local name=$1; shift; if grep -qx "$name 0" "$LOG/STATUS"; then echo "=== skip $name"; return 0; fi
         echo "=== $(date '+%F %T') START $name"; "$@" >> "$LOG/$name.log" 2>&1; local rc=$?
         echo "=== $(date '+%F %T') END $name rc=$rc"; echo "$name $rc" >> "$LOG/STATUS"; return $rc; }
stop() { echo "=== STOPPED at $1 (rc=$2)"; echo "STOPPED $1 $2" >> "$LOG/STATUS"; exit "$2"; }
TB=V0__TP__W2__all__kall__s13; TB4=V4__TP__W2__all__kall__s13
SP=results/go_function/specs; RU=results/go_function/runs; FR=results/go_function_fusion

# ---- G8: Core A specs and the code pin
step g8_tests $PY -m pytest go_function/tests -q -p no:cacheprovider || stop g8_tests $?
for A in bp mf; do for F in esm2_650m interpro; do
  step g8_materialize_${A}_$F $PY scripts/go_function/materialize.py --aspect $A --feature $F || stop g8_materialize_${A}_$F $?
done; done
step g8_pin $PY scripts/go_function/sweep.py pin || stop g8_pin $?

# ---- S1 G-TP: the six Tier B cells as ordinary grid runs (parallel), then verify + pin
pids=(); names=()
for c in bp/esm2_650m/$TB bp/interpro/$TB bp/interpro/$TB4 mf/esm2_650m/$TB mf/interpro/$TB mf/interpro/$TB4; do
  n=tierb_$(echo $c | tr / _)
  if [ -f $RU/$c/DONE.json ]; then echo "=== skip $n (DONE)"; continue; fi
  step $n $PY scripts/go_function/run.py --spec $SP/$c.json --device cuda & pids+=($!); names+=($n)
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || stop "${names[$i]}" 2; done
unset HF_HUB_OFFLINE
step s1_diamond_restore $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('olukotunjosh/cdcl-go-function-results', repo_type='dataset', local_dir='.',
                  allow_patterns=['$FR/diamond/*/*.res', '$FR/diamond/*/checks.json', '$FR/diamond/*/diamond_version.txt'])" || stop s1_diamond_restore $?
export HF_HUB_OFFLINE=1
for A in bp mf; do
  for nm in V0IPR:interpro/$TB V4IPR:interpro/$TB4 V0ESM:esm2_650m/$TB; do
    N=${nm%%:*}; C=${nm#*:}
    step s1_verify_${N}_$A $PY scripts/go_function_fusion/verify_tp_bases.py --name $N --aspect $A \
      --spec $SP/$A/$C.json --checkpoint $RU/$A/$C/checkpoint.pt --source a --device cuda --results $FR \
      || stop s1_verify_${N}_$A $?
  done
done

# ---- S1 G-V2 / G-F2 (search mode), then G-T once
for A in bp mf; do
  for B in mlp deepgozero; do
    step s1_gv2_${A}_$B $PY scripts/go_function_fusion/search_valid.py --aspect $A --base $B --diamond-valid --results $FR --device cuda || stop s1_gv2_${A}_$B $?
  done
  for B in V0IPR V4IPR V0ESM; do
    step s1_gv2_${A}_$B $PY scripts/go_function_fusion/search_valid.py --aspect $A --base $B --results $FR --device cuda || stop s1_gv2_${A}_$B $?
  done
done
step s1_gf2 $PY scripts/go_function_fusion/freeze_finalists.py --remote --results $FR || stop s1_gf2 $?
step s1_gt_score $PY scripts/go_function_fusion/score_test.py --results $FR --device cuda || stop s1_gt_score $?
step s1_gt_report $PY scripts/go_function_fusion/report.py --results $FR || stop s1_gt_report $?

# ---- save everything
unset HF_HUB_OFFLINE
.venv/bin/hf upload olukotunjosh/cdcl-go-function-results $FR $FR --repo-type dataset \
  --exclude "*.dmnd" "*.fa" "*.CLAIM" "*.tmp" --commit-message "S1 G-TP..G-T $(date +%F_%H%M)" 2>&1 | tail -1
.venv/bin/hf upload olukotunjosh/cdcl-go-function-results results/go_function results/go_function --repo-type dataset \
  --exclude "*.CLAIM" "*.tmp" --commit-message "G8 + Tier B $(date +%F_%H%M)" 2>&1 | tail -1
echo "=== $(date '+%F %T') G8 + S1 DONE"; echo "DONE S1" >> "$LOG/STATUS"
