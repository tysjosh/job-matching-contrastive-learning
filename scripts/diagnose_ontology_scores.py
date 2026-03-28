#!/usr/bin/env python3
"""
Diagnose why ontology_similarity and ot_distance have weak separation between labels.
Analyzes the underlying skill URI data to find root causes.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import statistics

def load_data(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main():
    path = "preprocess/data_splits_v4/train.jsonl"
    records = load_data(path)
    print(f"Total records: {len(records)}")
    
    # Group by label
    by_label = defaultdict(list)
    for r in records:
        label = r["metadata"].get("original_label", "unknown")
        by_label[label].append(r)
    
    print(f"\nLabel distribution:")
    for label, recs in sorted(by_label.items()):
        print(f"  {label}: {len(recs)}")
    
    # ── 1. Skill URI coverage analysis ──
    print("\n" + "="*70)
    print("1. SKILL URI COVERAGE BY LABEL")
    print("="*70)
    for label in ["good_fit", "potential_fit", "no_fit"]:
        recs = by_label[label]
        r_counts = [r["metadata"].get("resume_skill_uri_count", 0) for r in recs]
        j_counts = [r["metadata"].get("job_skill_uri_count", 0) for r in recs]
        r_zero = sum(1 for c in r_counts if c == 0)
        j_zero = sum(1 for c in j_counts if c == 0)
        both_zero = sum(1 for r, j in zip(r_counts, j_counts) if r == 0 or j == 0)
        
        print(f"\n  [{label}] ({len(recs)} records)")
        print(f"    Resume URIs: mean={statistics.mean(r_counts):.1f}, median={statistics.median(r_counts):.0f}, zero={r_zero} ({100*r_zero/len(recs):.1f}%)")
        print(f"    Job URIs:    mean={statistics.mean(j_counts):.1f}, median={statistics.median(j_counts):.0f}, zero={j_zero} ({100*j_zero/len(recs):.1f}%)")
        print(f"    Either zero: {both_zero} ({100*both_zero/len(recs):.1f}%) — these get null scores")
    
    # ── 2. Actual skill overlap analysis ──
    print("\n" + "="*70)
    print("2. DIRECT SKILL OVERLAP (exact URI match)")
    print("="*70)
    for label in ["good_fit", "potential_fit", "no_fit"]:
        recs = by_label[label]
        overlaps = []
        jaccard_scores = []
        for r in recs:
            r_uris = set(r["resume"].get("skill_uris", []))
            j_uris = set(r["job"].get("skill_uris", []))
            if r_uris and j_uris:
                overlap = len(r_uris & j_uris)
                jaccard = overlap / len(r_uris | j_uris) if (r_uris | j_uris) else 0
                overlaps.append(overlap)
                jaccard_scores.append(jaccard)
        
        if overlaps:
            print(f"\n  [{label}] ({len(overlaps)} with both URIs)")
            print(f"    Exact overlap count: mean={statistics.mean(overlaps):.2f}, median={statistics.median(overlaps):.0f}")
            print(f"    Jaccard similarity:  mean={statistics.mean(jaccard_scores):.4f}, median={statistics.median(jaccard_scores):.4f}")
            print(f"    Zero overlap: {sum(1 for o in overlaps if o == 0)} ({100*sum(1 for o in overlaps if o == 0)/len(overlaps):.1f}%)")
    
    # ── 3. Ontology similarity distribution detail ──
    print("\n" + "="*70)
    print("3. ONTOLOGY SIMILARITY DISTRIBUTION (non-null only)")
    print("="*70)
    for label in ["good_fit", "potential_fit", "no_fit"]:
        recs = by_label[label]
        scores = [r["metadata"]["ontology_similarity"] for r in recs 
                  if r["metadata"].get("ontology_similarity") is not None]
        if scores:
            # Bucket into ranges
            buckets = Counter()
            for s in scores:
                if s < 0.1: buckets["0.0-0.1"] += 1
                elif s < 0.2: buckets["0.1-0.2"] += 1
                elif s < 0.3: buckets["0.2-0.3"] += 1
                elif s < 0.4: buckets["0.3-0.4"] += 1
                elif s < 0.5: buckets["0.4-0.5"] += 1
                elif s < 0.6: buckets["0.5-0.6"] += 1
                elif s < 0.7: buckets["0.6-0.7"] += 1
                else: buckets["0.7+"] += 1
            
            print(f"\n  [{label}] ({len(scores)} scored)")
            print(f"    mean={statistics.mean(scores):.4f}, stdev={statistics.stdev(scores):.4f}")
            print(f"    p10={sorted(scores)[len(scores)//10]:.3f}, p25={sorted(scores)[len(scores)//4]:.3f}, "
                  f"p50={statistics.median(scores):.3f}, p75={sorted(scores)[3*len(scores)//4]:.3f}, "
                  f"p90={sorted(scores)[9*len(scores)//10]:.3f}")
            for bucket in ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7+"]:
                pct = 100 * buckets.get(bucket, 0) / len(scores)
                bar = "█" * int(pct / 2)
                print(f"    {bucket}: {buckets.get(bucket, 0):4d} ({pct:5.1f}%) {bar}")
    
    # ── 4. OT distance distribution detail ──
    print("\n" + "="*70)
    print("4. OT DISTANCE DISTRIBUTION (non-null only)")
    print("="*70)
    for label in ["good_fit", "potential_fit", "no_fit"]:
        recs = by_label[label]
        scores = [r["metadata"]["ot_distance"] for r in recs 
                  if r["metadata"].get("ot_distance") is not None]
        if scores:
            buckets = Counter()
            for s in scores:
                if s < 1.0: buckets["0-1"] += 1
                elif s < 2.0: buckets["1-2"] += 1
                elif s < 3.0: buckets["2-3"] += 1
                elif s < 4.0: buckets["3-4"] += 1
                elif s < 5.0: buckets["4-5"] += 1
                else: buckets["5+"] += 1
            
            print(f"\n  [{label}] ({len(scores)} scored)")
            print(f"    mean={statistics.mean(scores):.4f}, stdev={statistics.stdev(scores):.4f}")
            print(f"    p10={sorted(scores)[len(scores)//10]:.3f}, p25={sorted(scores)[len(scores)//4]:.3f}, "
                  f"p50={statistics.median(scores):.3f}, p75={sorted(scores)[3*len(scores)//4]:.3f}, "
                  f"p90={sorted(scores)[9*len(scores)//10]:.3f}")
            for bucket in ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]:
                pct = 100 * buckets.get(bucket, 0) / len(scores)
                bar = "█" * int(pct / 2)
                print(f"    {bucket}: {buckets.get(bucket, 0):4d} ({pct:5.1f}%) {bar}")
    
    # ── 5. Root cause: graph distance distribution ──
    print("\n" + "="*70)
    print("5. SKILL_OVERLAP (metadata field) BY LABEL")
    print("="*70)
    for label in ["good_fit", "potential_fit", "no_fit"]:
        recs = by_label[label]
        overlaps = [r["metadata"].get("skill_overlap", 0) for r in recs]
        if overlaps:
            print(f"\n  [{label}]")
            print(f"    mean={statistics.mean(overlaps):.4f}, median={statistics.median(overlaps):.4f}")
            print(f"    zero: {sum(1 for o in overlaps if o == 0)} ({100*sum(1 for o in overlaps if o == 0)/len(overlaps):.1f}%)")
    
    # ── 6. Essential/optional coverage by label ──
    print("\n" + "="*70)
    print("6. ESSENTIAL & OPTIONAL COVERAGE BY LABEL")
    print("="*70)
    for label in ["good_fit", "potential_fit", "no_fit"]:
        recs = by_label[label]
        ess = [r["metadata"].get("essential_coverage", 0) or 0 for r in recs]
        opt = [r["metadata"].get("optional_coverage", 0) or 0 for r in recs]
        print(f"\n  [{label}]")
        print(f"    Essential coverage: mean={statistics.mean(ess):.4f}, zero={sum(1 for e in ess if e == 0)} ({100*sum(1 for e in ess if e == 0)/len(ess):.1f}%)")
        print(f"    Optional coverage:  mean={statistics.mean(opt):.4f}, zero={sum(1 for o in opt if o == 0)} ({100*sum(1 for o in opt if o == 0)/len(opt):.1f}%)")
    
    # ── 7. How many resume skills are actually in the ESCO graph? ──
    print("\n" + "="*70)
    print("7. URI COUNT RATIO: resume_uris / total_resume_skills")
    print("="*70)
    for label in ["good_fit", "potential_fit", "no_fit"]:
        recs = by_label[label]
        ratios = []
        for r in recs:
            total_skills = len(r["resume"].get("skills", []))
            uri_count = len(r["resume"].get("skill_uris", []))
            if total_skills > 0:
                ratios.append(uri_count / total_skills)
        if ratios:
            print(f"\n  [{label}] (resume)")
            print(f"    URI coverage ratio: mean={statistics.mean(ratios):.3f}, median={statistics.median(ratios):.3f}")
            print(f"    <25% coverage: {sum(1 for r in ratios if r < 0.25)} ({100*sum(1 for r in ratios if r < 0.25)/len(ratios):.1f}%)")
            print(f"    <50% coverage: {sum(1 for r in ratios if r < 0.50)} ({100*sum(1 for r in ratios if r < 0.50)/len(ratios):.1f}%)")
    
    # ── 8. Occupation match analysis ──
    print("\n" + "="*70)
    print("8. OCCUPATION MATCH MODE BY LABEL")
    print("="*70)
    for label in ["good_fit", "potential_fit", "no_fit"]:
        recs = by_label[label]
        modes = Counter(r["metadata"].get("occupation_match_mode", "none") for r in recs)
        print(f"\n  [{label}]")
        for mode, count in modes.most_common():
            print(f"    {mode}: {count} ({100*count/len(recs):.1f}%)")

if __name__ == "__main__":
    main()
