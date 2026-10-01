#!/usr/bin/env python3
"""Download or verify the pinned GO/PPI and PatentMatch/CPC candidate data.

The large files live under dataset/, which is gitignored. The tracked manifest
records sources, sizes, checksums, licenses, and leakage cautions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "external_candidate_data.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(artifact: dict) -> bool:
    path = REPO_ROOT / artifact["path"]
    if not path.is_file():
        print(f"MISSING  {artifact['id']}: {path}")
        return False
    actual_size = path.stat().st_size
    actual_hash = sha256(path)
    ok = actual_size == artifact["bytes"] and actual_hash == artifact["sha256"]
    status = "OK" if ok else "BAD"
    print(f"{status:7} {artifact['id']} ({actual_size:,} bytes)")
    if not ok:
        print(f"         expected size={artifact['bytes']:,} sha256={artifact['sha256']}")
        print(f"         actual   size={actual_size:,} sha256={actual_hash}")
    return ok


def download(artifact: dict, allow_unverified_patent_mirror: bool) -> bool:
    path = REPO_ROOT / artifact["path"]
    if path.exists():
        return verify(artifact)
    if artifact["license"] == "UNVERIFIED" and not allow_unverified_patent_mirror:
        print(f"SKIP    {artifact['id']}: pass --allow-unverified-patent-mirror to download")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    request = urllib.request.Request(
        artifact["url"], headers={"User-Agent": "CDCL-research-data/1.0"}
    )
    print(f"GET     {artifact['id']} <- {artifact['url']}")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)

    if partial.stat().st_size != artifact["bytes"] or sha256(partial) != artifact["sha256"]:
        print(f"BAD     downloaded content did not match pinned manifest; retained at {partial}")
        return False
    os.replace(partial, path)
    path.chmod(0o644)
    return verify(artifact)


def extract_cpc(artifacts: list[dict]) -> bool:
    artifact = next(item for item in artifacts if item["id"] == "cpc-scheme-2020-08")
    archive = REPO_ROOT / artifact["path"]
    target = archive.parent / "CPCSchemeXML202008"
    if not archive.is_file():
        print(f"MISSING  CPC archive: {archive}")
        return False
    if target.exists() and any(target.iterdir()):
        print(f"OK      CPC already extracted: {target}")
        return True

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        target_root = target.resolve()
        for member in package.infolist():
            resolved = (target / member.filename).resolve()
            if target_root not in resolved.parents and resolved != target_root:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        package.extractall(target)
    print(f"OK      extracted CPC to {target}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--group", choices=("all", "go_ppi", "patent_cpc"), default="all")
    parser.add_argument("--download", action="store_true", help="Download missing files before verification")
    parser.add_argument("--extract-cpc", action="store_true", help="Extract the pinned CPC archive")
    parser.add_argument(
        "--allow-unverified-patent-mirror",
        action="store_true",
        help="Allow downloading the third-party PatentMatch mirror with unverified upstream terms",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    artifacts = [
        item for item in manifest["artifacts"] if args.group == "all" or item["group"] == args.group
    ]
    results = []
    for artifact in artifacts:
        if args.download:
            results.append(download(artifact, args.allow_unverified_patent_mirror))
        else:
            results.append(verify(artifact))
    if args.extract_cpc and args.group in ("all", "patent_cpc"):
        results.append(extract_cpc(manifest["artifacts"]))
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
