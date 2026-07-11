#!/usr/bin/env python3
import argparse, json, subprocess, shlex
from pathlib import Path

# Dataset label -> dataset JSONL path (relative to --workdir when not absolute)
DATASET_PATH_MAP = {
    'cnamuangtoun': 'preprocess/combined_all_8000_scored.jsonl',
    'cnamuangtou_sparse_v6': 'preprocess/combined_all_8000_scored.jsonl',
    'indian': 'preprocess/indian_dataset_scored_labeled.jsonl',
}

PRIORITY = [
    # Fast -> slow (from ablation plan)
    'E3-D','E3-A',
    'E1-A','E1-B','E1-C','E1-D',
    'E4-InfoNCE','E4-OSCAR-Skill','E4-OSCAR-Hybrid','E4-OSCAR-ISCO',
    'E2-A','E2-B','E2-C',
    'E5-D64','E5-D96','E5-D128','E5-D192','E5-IVFPQ','E5-INT8',
    'other'
]
PRIORITY_IDX = {k:i for i,k in enumerate(PRIORITY)}

def exp_rank(exp_id:str)->int:
    return PRIORITY_IDX.get(exp_id, PRIORITY_IDX['other'])

def load_index(path:Path):
    return json.loads(path.read_text())

def main():
    ap=argparse.ArgumentParser(description='Priority launcher for OSCAR/CDCL manifests')
    ap.add_argument('--index', default='manifest_index.json', help='Path to manifest_index.json')
    ap.add_argument('--runner-cmd', required=True,
                    help='Command template. Placeholders: {manifest}, {run_id}, {experiment_id}, {dataset}, {dataset_path}, {seed}. Example: "python train.py --config {manifest}"')
    ap.add_argument('--dataset', action='append', choices=['cnamuangtoun','cnamuangtou_sparse_v6','indian'],
                    help='Optional dataset filter (repeatable)')
    ap.add_argument('--experiment', action='append', help='Optional experiment filter (repeatable, e.g. E1-A)')
    ap.add_argument('--seed', type=int, action='append', help='Optional seed filter (repeatable)')
    ap.add_argument('--format', choices=['yaml','json'], default='yaml', help='Manifest format to execute')
    ap.add_argument('--limit', type=int, default=0, help='Run at most N manifests (0 = all)')
    ap.add_argument('--execute', action='store_true', help='Actually execute commands (default: dry-run)')
    ap.add_argument('--start-at', default='', help='Optional run_id to start from (inclusive)')
    ap.add_argument('--workdir', default='', help='Working directory where runner command should execute')
    args=ap.parse_args()

    idx_path=Path(args.index).resolve()
    rows=load_index(idx_path)

    # filters
    out=[]
    started=(args.start_at=='')
    for r in rows:
        if not started:
            if r['run_id']==args.start_at:
                started=True
            else:
                continue
        if args.dataset and r['dataset'] not in args.dataset:
            continue
        if args.experiment and r['experiment_id'] not in args.experiment:
            continue
        if args.seed and r['seed'] not in args.seed:
            continue
        out.append(r)

    # sort by priority, then dataset, then seed, then run_id
    out.sort(key=lambda r:(exp_rank(r['experiment_id']), r['dataset'], r['seed'], r['run_id']))

    if args.limit and args.limit>0:
        out=out[:args.limit]

    cwd = Path(args.workdir).expanduser().resolve() if args.workdir else None

    print(f"Selected manifests: {len(out)}")
    if cwd:
        print(f"Execution working directory: {cwd}")
    for i,r in enumerate(out,1):
        manifest = Path(r[args.format])
        # If manifest path in index is stale (moved folders), fall back to local sibling path.
        if not manifest.exists():
            candidate = idx_path.parent / args.format / manifest.name
            if candidate.exists():
                manifest = candidate

        dataset_path = DATASET_PATH_MAP.get(r['dataset'], r['dataset'])
        dataset_path = str((cwd / dataset_path).resolve()) if (cwd and not Path(dataset_path).is_absolute()) else str(Path(dataset_path).resolve())

        values = {
            'manifest': str(manifest),
            'run_id': r['run_id'],
            'experiment_id': r['experiment_id'],
            'dataset': r['dataset'],
            'dataset_path': dataset_path,
            'seed': str(r['seed']),
        }

        cmd = args.runner_cmd
        for k,v in values.items():
            cmd = cmd.replace('{' + k + '}', shlex.quote(v))

        print(f"[{i}/{len(out)}] {r['run_id']} -> {manifest}")
        print(f"  CMD: {cmd}")
        if args.execute:
            rc=subprocess.call(cmd, shell=True, cwd=str(cwd) if cwd else None)
            if rc!=0:
                print(f"  ERROR: command failed with exit code {rc}; stopping.")
                raise SystemExit(rc)

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to launch.")

if __name__=='__main__':
    main()
