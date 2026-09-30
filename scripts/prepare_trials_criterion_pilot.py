#!/usr/bin/env python3
"""Acquire immutable TrialGPT annotation files and prepare a criterion pilot.

Public source files only; no authenticated services or patient uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(config: dict) -> dict:
    raw = ROOT / config["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "acquisition_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] != config["dataset_revision"]:
            raise ValueError("Raw data revision differs; use a new output directory.")
        for item in manifest["files"]:
            if sha256(raw / item["path"]) != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {item['path']}")
        return manifest
    files = []
    for name in config["source_files"]:
        url = (f"https://huggingface.co/datasets/{config['dataset_id']}/resolve/"
               f"{config['dataset_revision']}/{name}")
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != response.content:
            raise ValueError(f"Refusing to replace a differing raw file: {path}")
        path.write_bytes(response.content)
        files.append({"path": name, "url": url, "bytes": len(response.content),
                      "sha256": sha256(path)})
        print(f"Acquired {name}: {len(response.content):,} bytes", flush=True)
    manifest = {"dataset_id": config["dataset_id"], "revision": config["dataset_revision"],
                "acquired_at": datetime.now(timezone.utc).isoformat(), "files": files,
                "license_declared_in_card": "public-domain",
                "license_file_empty": (raw / "LICENSE").stat().st_size == 0}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/trials_criterion_pilot.json")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest = acquire(config)
    if not args.download_only:
        from trials_domain.criterion_pilot import prepare
        result = prepare(ROOT, config, manifest)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
