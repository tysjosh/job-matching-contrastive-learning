#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Example:
# ./run_all.sh "python train.py --config {manifest}" --workdir ~/Downloads/CDCL --dataset cnamuangtoun --limit 10

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 '<runner-cmd-with-{manifest}>' [extra run_all.py args...]"
  echo "Example: $0 'python train.py --config {manifest}' --dataset cnamuangtoun --limit 10"
  exit 1
fi

RUNNER_CMD="$1"
shift

python3 run_all.py --index manifest_index.json --runner-cmd "$RUNNER_CMD" "$@"
