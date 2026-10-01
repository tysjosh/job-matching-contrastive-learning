#!/usr/bin/env python3
"""Fetch CPC classification codes for the EP patents referenced by PatentMatch.

PatentMatch supplies the graded label (X = cited against novelty/inventive step,
A = cited as background) but carries no classification codes, and the CPC scheme
XML supplies the hierarchy but no patent-to-code assignments. This bridges the
two by pulling the CPC codes Google Patents publishes per document.

Why Google Patents: every USPTO route for patent-to-CPC assignment now needs a
registered API key (see dataset/patent_cpc/ACQUISITION_BLOCKED.md), and EPO OPS
does too. Google Patents serves the codes in the public document page. Only
~970 documents are needed for PatentMatch's test split, so this is a small,
one-off, cached fetch rather than bulk scraping.

Politeness / reproducibility
  - one request at a time, with a delay between requests (default 1.1s)
  - every response cached to disk; re-runs skip anything already fetched, so an
    interrupted run resumes rather than refetching
  - failures recorded rather than retried forever, so coverage stays visible

Output
  dataset/patent_cpc/cpc_by_patent.json
      {"EP3185416A1": ["G01C19/5776", "H03F1/3211", ...], ...}
  dataset/patent_cpc/cpc_fetch_failures.json

Usage
  .venv/bin/python3 scripts/fetch_cpc_codes.py                  # test split
  .venv/bin/python3 scripts/fetch_cpc_codes.py --split train --limit 2000
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "dataset" / "patent_cpc"
MIRROR = PC / "patentmatch_mirror"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# <span itemprop="Code">H03F1/3211</span>
CODE_RE = re.compile(r'<span itemprop="Code">([^<]+)</span>')
# a full CPC code has a group/subgroup part; bare "H", "H03", "H03F" are ancestors
FULL_CODE_RE = re.compile(r'^[A-HY]\d{2}[A-Z]\d+/\d+$')


def load_ids(split, limit):
    tsv = MIRROR / f"{split}_balanced.tsv"
    zp = MIRROR / f"{split}_balanced.tsv.zip"
    if not tsv.exists():
        if not zp.exists():
            sys.exit(f"missing {tsv.name} and {zp.name}")
        import zipfile
        print(f"extracting {zp.name} ...")
        with zipfile.ZipFile(zp) as z:
            z.extractall(MIRROR)
    import pandas as pd
    df = pd.read_csv(tsv, sep="\t", engine="python", on_bad_lines="skip",
                     usecols=["patent_application_id", "cited_document_id"])
    ids = set(df.patent_application_id.dropna().astype(str))
    ids |= set(df.cited_document_id.dropna().astype(str))
    ids = sorted(i for i in ids if i.startswith("EP"))
    if limit:
        ids = ids[:limit]
    return ids, df


def fetch_one(pid, timeout=30):
    url = f"https://patents.google.com/patent/{pid}/en"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf-8", "replace")
    codes = sorted({c.strip() for c in CODE_RE.findall(html)})
    full = [c for c in codes if FULL_CODE_RE.match(c)]
    return full, len(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=1.1)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    out_path = PC / "cpc_by_patent.json"
    fail_path = PC / "cpc_fetch_failures.json"
    cache = json.loads(out_path.read_text()) if out_path.exists() else {}
    failures = json.loads(fail_path.read_text()) if fail_path.exists() else {}

    ids, _ = load_ids(args.split, args.limit)
    todo = [i for i in ids if i not in cache]
    print(f"{args.split} split: {len(ids)} unique EP documents, "
          f"{len(cache)} already cached, {len(todo)} to fetch")
    if not todo:
        print("nothing to do")
        return

    eta = len(todo) * args.delay / 60
    print(f"estimated wall time at {args.delay}s/request: {eta:.1f} min")

    done = 0
    for i, pid in enumerate(todo, 1):
        ok = False
        for attempt in range(args.retries + 1):
            try:
                full, n_all = fetch_one(pid)
                cache[pid] = full
                failures.pop(pid, None)
                ok = True
                if i <= 3 or i % 100 == 0:
                    print(f"  [{i}/{len(todo)}] {pid}: {len(full)} full codes "
                          f"({n_all} spans incl. ancestors)  e.g. {full[:3]}")
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    failures[pid] = "404"
                    break
                failures[pid] = f"HTTP {e.code}"
                time.sleep(2.0 * (attempt + 1))
            except Exception as e:
                failures[pid] = f"{type(e).__name__}: {e}"
                time.sleep(2.0 * (attempt + 1))
        if ok:
            done += 1
        time.sleep(args.delay)
        if i % 50 == 0:
            out_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
            fail_path.write_text(json.dumps(failures, indent=1, sort_keys=True))

    out_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    fail_path.write_text(json.dumps(failures, indent=1, sort_keys=True))

    with_codes = sum(1 for v in cache.values() if v)
    print()
    print(f"fetched this run : {done}/{len(todo)}")
    print(f"cache total      : {len(cache)} documents, {with_codes} with >=1 full CPC code")
    print(f"failures         : {len(failures)}")
    if failures:
        for k, v in list(failures.items())[:10]:
            print(f"    {k}: {v}")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
