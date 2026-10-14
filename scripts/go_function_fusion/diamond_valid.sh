#!/usr/bin/env bash
# Gate G-D (REMOTE ONLY; protocol §4.3): train-only DIAMOND DB, valid and test queries.
# Flags of DGZ's run_diamond.sh (blastp --more-sensitive --outfmt 6 qseqid sseqid
# bitscore pident), but the DB holds the TRAIN sequences only (DGZ appends valid).
# Fatal checks (python): every hit id is a train protein; no query id is a train id.
# Then pin with: python scripts/go_function_fusion/make_pins.py diamond --aspect <a> ...
#
# Usage (repo root, remote):  bash scripts/go_function_fusion/diamond_valid.sh bp [path/to/diamond]
set -euo pipefail
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "REFUSED: diamond_valid.sh is remote only (protocol U4); nothing is installed on the Mac" >&2
  exit 4
fi
ASPECT="${1:?aspect bp|mf}"
DIAMOND="${2:-diamond}"
PY="${PY:-.venv/bin/python}"
OUT="results/go_function_fusion/diamond/${ASPECT}"
DATA="dataset/go_function/dgz/${ASPECT}"
if [[ -e "${OUT}/checks.json" ]]; then
  echo "REFUSED: ${OUT}/checks.json exists (G-D is run once)" >&2
  exit 4
fi
mkdir -p "${OUT}"
"${DIAMOND}" version | tee "${OUT}/diamond_version.txt"
for split in train valid test; do
  "${PY}" - "${DATA}/${split}_data.pkl" "${OUT}/${split}_data.fa" <<'EOF'
import sys
import pandas as pd
df = pd.read_pickle(sys.argv[1])
with open(sys.argv[2], "w") as f:          # DGZ diamond_data.py format
    for row in df.itertuples():
        f.write(">" + row.proteins + "\n")
        f.write(row.sequences + "\n")
EOF
done
"${DIAMOND}" makedb --in "${OUT}/train_data.fa" --db "${OUT}/train_only.dmnd"
"${DIAMOND}" blastp --more-sensitive -d "${OUT}/train_only.dmnd" -q "${OUT}/valid_data.fa" \
  --outfmt 6 qseqid sseqid bitscore pident > "${OUT}/valid_diamond_trainonly.res"
"${DIAMOND}" blastp --more-sensitive -d "${OUT}/train_only.dmnd" -q "${OUT}/test_data.fa" \
  --outfmt 6 qseqid sseqid bitscore pident > "${OUT}/test_diamond_trainonly.res"
"${PY}" - "${ASPECT}" "${OUT}" <<'EOF'
import json, sys
sys.path.insert(0, ".")
import pandas as pd
from go_function_fusion import priors as PR
aspect, out = sys.argv[1], sys.argv[2]
d = f"dataset/go_function/dgz/{aspect}"
train = list(pd.read_pickle(f"{d}/train_data.pkl").proteins)
checks = {}
for split in ("valid", "test"):
    q = list(pd.read_pickle(f"{d}/{split}_data.pkl").proteins)
    checks[split] = PR.check_train_only(f"{out}/{split}_diamond_trainonly.res", q, train)
with open(f"{out}/checks.json", "w") as f:
    json.dump(checks, f, indent=2, sort_keys=True)
print(json.dumps(checks))
EOF
rm -f "${OUT}/valid_data.fa" "${OUT}/test_data.fa"
echo "G-D ${ASPECT} done; now: make_pins.py diamond --aspect ${ASPECT} --diamond-version \"\$(${DIAMOND} version)\" --diamond-binary \$(command -v ${DIAMOND})"
