#!/usr/bin/env python3
"""
Pull ORCA/ordinal run RESULTS back from a Hugging Face repo into
results/research_runs/ so the local aggregators and significance tests can see
them.

Counterpart to scripts/hf_upload_data.py: that pushes training data UP to the
GPU machine, this pulls evaluation artifacts BACK for analysis. Only small JSON
artifacts are fetched by default (not checkpoints), so this is cheap.

Layout-agnostic: the repo can nest the runs however it likes. Every downloaded
file whose path contains a run id (``<VARIANT>__<dataset>__s<seed>``) is copied
to::

    results/research_runs/<run_id>/phase1_evaluation/<filename>

which is where scripts/orca_table4.py and scripts/orca_sig_test.py look for
``ordinal_evaluation_results.json``.

Prereq (one-time auth, only needed for a private repo):
  huggingface-cli login      # or: export HF_TOKEN=hf_xxx

Usage:
  python3 scripts/hf_pull_results.py --repo-id <user>/cdcl-orca-results
  python3 scripts/hf_pull_results.py --repo-id <user>/cdcl-orca-results --dry-run
  python3 scripts/hf_pull_results.py --repo-id <user>/cdcl-orca-results \
      --runs 'ER-*' --include-checkpoints
"""
import argparse
import fnmatch
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = ROOT / "results" / "research_runs"

# results/research_runs/<VARIANT>__<dataset>__s<seed>/...
RUN_RE = re.compile(r"([A-Za-z0-9.\-]+__[A-Za-z0-9_]+__s\d+)")

# The evaluation artifacts the local analysis scripts consume.
RESULT_FILES = [
    "ordinal_evaluation_results.json",     # orca_table4.py / orca_sig_test.py
    "phase1_evaluation_results.json",      # aggregate_orca_results.py
    "training_config.json",                # provenance
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pull run results from Hugging Face into results/research_runs/.")
    ap.add_argument("--repo-id", required=True,
                    help="Source repo, e.g. yourname/cdcl-orca-results")
    ap.add_argument("--repo-type", default="dataset",
                    choices=["dataset", "model"],
                    help="HF repo type (default: dataset).")
    ap.add_argument("--runs", default="*",
                    help="Glob over run ids to keep, e.g. 'ER-*' (default: all).")
    ap.add_argument("--include-checkpoints", action="store_true",
                    help="Also fetch best_checkpoint.pt (large).")
    ap.add_argument("--cache-dir", default="hf_results",
                    help="Local download staging dir (default: hf_results).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied without writing.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing local artifacts (default: skip).")
    args = ap.parse_args()

    allow = [f"**/{name}" for name in RESULT_FILES]
    if args.include_checkpoints:
        allow.append("**/best_checkpoint.pt")

    staging = ROOT / args.cache_dir
    print(f"Downloading {args.repo_id} ({args.repo_type}) -> {staging}")
    try:
        local = snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=str(staging),
            allow_patterns=allow,
        )
    except Exception as exc:
        print(f"ERROR: download failed: {exc}")
        print("If the repo is private, authenticate first: huggingface-cli login")
        sys.exit(1)

    copied = skipped = unmatched = 0
    runs_seen = set()
    for src in sorted(Path(local).rglob("*")):
        if not src.is_file():
            continue
        m = RUN_RE.search(str(src.relative_to(local)))
        if not m:
            unmatched += 1
            continue
        run_id = m.group(1)
        if not fnmatch.fnmatch(run_id, args.runs):
            continue

        dest = DEST_ROOT / run_id / (
            "" if src.name == "training_config.json" else "phase1_evaluation"
        ) / src.name
        runs_seen.add(run_id)

        if dest.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.dry_run:
            print(f"  [would copy] {run_id}/{src.name}")
            copied += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1

    verb = "would copy" if args.dry_run else "copied"
    print(f"\n{verb} {copied} file(s) across {len(runs_seen)} run(s); "
          f"skipped {skipped} already present; {unmatched} path(s) had no run id.")
    if skipped and not args.overwrite:
        print("Pass --overwrite to replace artifacts that already exist locally.")
    if not runs_seen:
        print("No run ids matched. Expected paths containing "
              "'<VARIANT>__<dataset>__s<seed>' (e.g. ER-DEN__cnamuangtoun__s13).")
        return

    print("\nNext:")
    print("  python3 scripts/orca_table4.py")
    print("  python3 scripts/orca_sig_test.py E4-OSCAR-Skill ER-DEN")
    print("  python3 scripts/orca_sig_test.py ER-DEN ER-EXT")


if __name__ == "__main__":
    main()
