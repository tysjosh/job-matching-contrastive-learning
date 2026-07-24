#!/usr/bin/env python3
"""Report CVSS coverage in the CVE source data (CSV and/or ontology profiles).

The converter fills a CVSS target from ``cvss_base_score`` / ``cvss_base_severity``,
reading the CSV row first and falling back to the joined ontology profile. This
script checks BOTH sources so you can tell, before launching the CVSS-target runs,
whether CVSS coverage is high enough to be worth it.

Run on the box that has the enriched data (the GPU box):

    python scripts/check_cvss_coverage.py
    python scripts/check_cvss_coverage.py --csv path/to.csv --profiles path/to.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

_CSV_CANDIDATES = [
    "cybersecurity-vulnerability-ranking/data/processed/vulnerability_priority_ranking_nvd_all.csv",
    "cybersecurity-vulnerability-ranking/data/processed/vulnerability_priority_ranking.csv",
]
_PROFILES_CANDIDATES = [
    "cybersecurity-vulnerability-ranking/data/ontology/cve_ontology_profiles.jsonl",
]


def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


def _report(n: int, have_sev: int, have_score: int, sev: Counter, label: str) -> None:
    if n == 0:
        print(f"\n[{label}] no rows read.")
        return
    print(f"\n[{label}] {n} records")
    print(f"  cvss_base_severity present: {have_sev}/{n} ({100*have_sev/n:.1f}%)")
    print(f"  cvss_base_score present   : {have_score}/{n} ({100*have_score/n:.1f}%)")
    if sev:
        print("  severity distribution     :", dict(sev.most_common()))


def check_csv(path: str) -> None:
    n = have_sev = have_score = 0
    sev: Counter = Counter()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        print(f"CSV columns: {reader.fieldnames}")
        for row in reader:
            n += 1
            s = (row.get("cvss_base_severity") or "").strip()
            sc = (row.get("cvss_base_score") or "").strip()
            if s:
                have_sev += 1
                sev[s.lower()] += 1
            if sc:
                have_score += 1
    _report(n, have_sev, have_score, sev, f"CSV {Path(path).name}")


def check_profiles(path: str) -> None:
    n = have_sev = have_score = 0
    sev: Counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if first:
                print(f"\nProfile keys (sample): {sorted(d.keys())}")
                first = False
            n += 1
            s = str(d.get("cvss_base_severity") or "").strip()
            sc = str(d.get("cvss_base_score") or "").strip()
            if s:
                have_sev += 1
                sev[s.lower()] += 1
            if sc:
                have_score += 1
    _report(n, have_sev, have_score, sev, f"profiles {Path(path).name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--profiles", default=None)
    args = ap.parse_args()

    csv_path = args.csv or _first_existing(_CSV_CANDIDATES)
    prof_path = args.profiles or _first_existing(_PROFILES_CANDIDATES)

    if csv_path:
        print(f"Checking CSV: {csv_path}")
        check_csv(csv_path)
    else:
        print("No CSV found among candidates:", _CSV_CANDIDATES)

    if prof_path:
        print(f"\nChecking profiles: {prof_path}")
        check_profiles(prof_path)
    else:
        print("\nNo ontology profiles found among candidates:", _PROFILES_CANDIDATES)

    print("\nRule of thumb: want cvss_base_severity coverage well above ~70% and a "
          "spread across low/medium/high/critical for the CVSS target to be worthwhile.")


if __name__ == "__main__":
    main()
