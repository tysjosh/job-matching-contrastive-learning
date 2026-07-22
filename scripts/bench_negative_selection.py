#!/usr/bin/env python3
"""Micro-benchmark: E4-OSCAR-Skill vs ORCA negative-selection cost on v7.

Times the real ``BatchProcessor.process_batch`` path (which contains the
skill-graph negative selection that dominated the observed ~50s/batch wall
time) on the v7 training data with the real ESCO KG + precomputed skill
distances.

We deliberately exclude the neural forward/backward: it runs on GPU and is
*identical* for OSCAR-Skill and ORCA, so it cannot explain any difference
between them. What differs on CPU is (a) ORCA's extra per-negative ontology
feature capture (``_compute_negative_ontology_features``, gated on
``orca_enabled``) and the isco blend, and (b) whether the set-similarity memo
is active. For each config we time N batches with the memo OFF (pre-fix
behaviour) and ON across two passes (pass 1 cold, pass 2 warm = epoch 2+).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.data_loader import DataLoader
from contrastive_learning.batch_processor import BatchProcessor

CONFIG = "config/orca_denominator_config.json"
TRAIN = "preprocess/data_splits_v7/train.jsonl"
N_BATCHES = 3
POOL = 1000

# E4-OSCAR-Skill overlay (run_manifests/manifest_adapter._overlay_oscar_skill):
# skill-graph negatives + skill weighting, NO isco, NO ORCA.
OSCAR_SKILL_OVERLAY = {
    "orca_enabled": False,
    "use_pathway_negatives": True,
    "ontology_weight": 0.3,
    "use_ot_distance": True,
    "use_isco_negatives": False,
    "negative_curriculum": False,
}


def _fmt(times):
    return ", ".join(f"{t:6.2f}s" for t in times) + f"  (mean {sum(times)/len(times):6.2f}s)"


def _make_config(overlay=None):
    cfg = TrainingConfig.from_json(CONFIG)
    for k, v in (overlay or {}).items():
        setattr(cfg, k, v)
    return cfg


def _bench_processor(label, cfg, pool, batches):
    print(f"\nBuilding BatchProcessor for {label} (loads ESCO KG once)...")
    bp = BatchProcessor(cfg, esco_graph_path=cfg.esco_graph_path)
    matcher = bp.skill_matcher
    print(f"  orca_enabled={getattr(cfg,'orca_enabled',False)} "
          f"ontology_weight={cfg.ontology_weight} "
          f"use_isco_negatives={getattr(bp,'use_isco_negatives',None)} "
          f"skill_matcher={'ENABLED' if matcher else 'DISABLED'}")
    if matcher is None:
        print("  (skill matcher disabled — ontology path not active)")
        return None

    def reset(enabled):
        matcher._set_sim_cache.clear()
        matcher._set_sim_cache_max = 4_000_000 if enabled else 0

    def run_pass():
        out = []
        for batch in batches:
            t0 = time.perf_counter()
            bp.process_batch(batch, global_job_pool=pool)
            out.append(time.perf_counter() - t0)
        return out

    reset(False)
    nocache = run_pass()
    reset(True)
    warm1 = run_pass()
    warm2 = run_pass()
    print(f"  memo OFF (pre-fix)    : {_fmt(nocache)}")
    print(f"  memo ON  pass1 (cold) : {_fmt(warm1)}")
    print(f"  memo ON  pass2 (warm) : {_fmt(warm2)}")
    return {
        "nocache": sum(nocache) / len(nocache),
        "cold": sum(warm1) / len(warm1),
        "warm": sum(warm2) / len(warm2),
    }


def main():
    base = _make_config()
    print(f"pool={POOL}, batches={N_BATCHES}, dataset=v7 train")

    loader = DataLoader(base)
    print("Loading global job pool...")
    pool = loader.load_global_job_pool(TRAIN, max_jobs=POOL)
    print(f"  pool size: {len(pool)}")
    print("Collecting benchmark batches...")
    batches = []
    for b in loader.load_batches(TRAIN):
        batches.append(b)
        if len(batches) >= N_BATCHES:
            break
    print(f"  {len(batches)} batches of size ~{len(batches[0])}")

    oscar = _bench_processor("E4-OSCAR-Skill", _make_config(OSCAR_SKILL_OVERLAY), pool, batches)
    orca = _bench_processor("ORCA (denominator)", _make_config(), pool, batches)

    print("\n=== summary (mean s/batch, selection path only) ===")
    if oscar:
        print(f"  OSCAR-Skill  memo OFF : {oscar['nocache']:6.2f}s   ON warm: {oscar['warm']:6.2f}s")
    if orca:
        print(f"  ORCA         memo OFF : {orca['nocache']:6.2f}s   ON warm: {orca['warm']:6.2f}s")
    if oscar and orca:
        print(f"  ORCA vs OSCAR (memo OFF): {orca['nocache'] - oscar['nocache']:+.2f}s "
              f"({orca['nocache']/max(oscar['nocache'],1e-9):.2f}x)")
        print(f"  memo speedup (warm/off) : OSCAR {oscar['nocache']/max(oscar['warm'],1e-9):.1f}x, "
              f"ORCA {orca['nocache']/max(orca['warm'],1e-9):.1f}x")


if __name__ == "__main__":
    main()
