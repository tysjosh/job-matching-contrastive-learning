#!/usr/bin/env python3
"""
Upload the large training data to a PRIVATE Hugging Face dataset repo so it can
be pulled on the GPU machine (these files are too big for GitHub and are
gitignored).

Uploads:
  preprocess/data_splits_v7/   (train/validation/test.jsonl + split_indices.json)
  dataset/esco/esco_kg.gexf    (ESCO knowledge graph)

PRIVACY: data_splits_v7 contains resume text (PII). This script creates a
PRIVATE repo by default. Do NOT pass --public unless you are certain the data
is safe to share.

Prereq (one-time auth):
  huggingface-cli login          # paste a token with write access
  # or: export HF_TOKEN=hf_xxx

Usage:
  python3 scripts/hf_upload_data.py --repo-id <user>/cdcl-v7-data
  python3 scripts/hf_upload_data.py --repo-id <user>/cdcl-v7-data --include-esco
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True,
                    help="Target dataset repo, e.g. yourname/cdcl-v7-data")
    ap.add_argument("--include-esco", action="store_true",
                    help="Also upload dataset/esco/esco_kg.gexf (77MB)")
    ap.add_argument("--public", action="store_true",
                    help="Make the repo PUBLIC (default: private). Resumes are PII!")
    args = ap.parse_args()

    try:
        user = whoami().get("name")
        print(f"Authenticated as: {user}")
    except Exception:
        print("ERROR: not authenticated. Run `huggingface-cli login` "
              "or set HF_TOKEN, then retry.")
        sys.exit(1)

    api = HfApi()
    private = not args.public
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=private, exist_ok=True)
    print(f"Repo ready: {args.repo_id} (private={private})")

    # Upload data_splits_v7 under data_splits_v7/ in the repo
    splits = ROOT / "preprocess" / "data_splits_v7"
    if not splits.exists():
        print(f"ERROR: {splits} not found"); sys.exit(1)
    print(f"Uploading {splits} ...")
    api.upload_folder(
        folder_path=str(splits),
        path_in_repo="data_splits_v7",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add v7 data splits",
    )

    if args.include_esco:
        esco = ROOT / "dataset" / "esco" / "esco_kg.gexf"
        if esco.exists():
            print(f"Uploading {esco} ...")
            api.upload_file(
                path_or_fileobj=str(esco),
                path_in_repo="esco/esco_kg.gexf",
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Add ESCO knowledge graph",
            )
        else:
            print(f"WARN: {esco} not found, skipping ESCO upload")

    print("\nDone. On the GPU machine, download with:")
    print(f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id="{args.repo_id}", repo_type="dataset",
                  local_dir="hf_data")
# then place files:
#   hf_data/data_splits_v7/*  -> preprocess/data_splits_v7/
#   hf_data/esco/esco_kg.gexf -> dataset/esco/esco_kg.gexf
""")


if __name__ == "__main__":
    main()
