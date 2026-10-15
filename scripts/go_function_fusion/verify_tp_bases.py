"""Gate G-TP (REMOTE ONLY; protocol §9, §4.2, §10.4): verify one Tier B TP base and pin it.

Checks, in order (any failure is fatal, exit 2; nothing is pinned):

1. the spec is a GF materialized spec under ``results/go_function/specs/`` for the named
   base (V0IPR = V0/interpro, V4IPR = V4/interpro, V0ESM = V0/esm2_650m; study mode,
   ``pack`` None, seed 13, the named aspect); spec and checkpoint sha256 recorded;
2. source (a): the checkpoint is under ``results/go_function/runs/``;
   source (b): it is under ``results/go_function_fusion/tp_bases/`` and
   ``tp_base_provenance.json`` next to it has the spec sha256 of the GF spec, exit code 0,
   and ``go_function_code`` equal to GF's code pin (``results/go_function/code_pin.json``
   ``files``) for every ``go_function/*.py`` file in either;
3. the §10.4 exposure facts (directory listings only, no file is opened): whether any GF
   ``eval.json`` or ``results/go_function/analysis/`` file existed, and which test-derived
   outputs ``train.run`` left in the source (b) directory;
4. in search mode (the guard is active): strict load and two validation forwards, max
   abs difference 0.

Usage (remote):
  $PY scripts/go_function_fusion/verify_tp_bases.py --name V0IPR --aspect bp \\
      --spec results/go_function/specs/bp/interpro/<run>.json \\
      --checkpoint results/go_function/runs/bp/interpro/<run>/checkpoint.pt --source a --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_pins as MP  # noqa: E402
from go_function_fusion import guard  # noqa: E402

BASES = {"V0IPR": ("V0", "interpro"), "V4IPR": ("V4", "interpro"), "V0ESM": ("V0", "esm2_650m")}
GF_CODE_PIN = "results/go_function/code_pin.json"
SOURCE_PREFIX = {"a": "results/go_function/runs/", "b": "results/go_function_fusion/tp_bases/"}
TP_TEST_OUTPUTS = ("eval.json", "scores.npz", "eval_inputs.npz", "run.json")


def check_spec(spec: Mapping[str, Any], name: str, aspect: str) -> None:
    variant, feature = BASES[name]
    want = {"aspect": aspect, "variant": variant, "feature": feature, "mode": "study"}
    bad = {k: spec.get(k) for k, v in want.items() if spec.get(k) != v}
    if spec.get("pack") is not None:
        bad["pack"] = spec.get("pack")
    if int(spec.get("seed", -1)) != 13:
        bad["seed"] = spec.get("seed")
    if bad:
        raise MP.Fatal(f"spec is not the {name} {aspect} 'all' seed-13 base: {bad}")


def compare_gf_code(prov_code: Mapping[str, str], pin_files: Mapping[str, str]) -> Dict[str, Any]:
    """``go_function/*.py`` hashes of the provenance vs GF's code pin (both directions)."""
    def top(d):
        return {k: v for k, v in d.items() if k.startswith("go_function/") and k.count("/") == 1
                and k.endswith(".py")}
    a, b = top(prov_code), top(pin_files)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {"equal": not diff and bool(a), "differs": diff, "n_files": len(a)}


def exposure_facts(ck_rel: str, source: str) -> Dict[str, Any]:
    """§10.4: listings only (``Path.glob`` / ``exists`` do not open files)."""
    gf = ROOT / "results/go_function"
    evals = sorted(MP.rel(p) for p in gf.glob("**/eval.json"))
    analysis = sorted(MP.rel(p) for p in (gf / "analysis").rglob("*") if p.is_file()) \
        if (gf / "analysis").is_dir() else []
    out = {"gf_eval_json_existed": bool(evals), "n_gf_eval_json": len(evals),
           "gf_analysis_existed": bool(analysis), "n_gf_analysis_files": len(analysis),
           "opened_by_fusion_pipeline": False,
           "basis": "search-mode guard blocks results/go_function/** except specs, pins, gate "
                    "JSONs and named checkpoints, and results/go_function_fusion*/tp_bases/** "
                    "except checkpoint.pt and tp_base_provenance.json (§10.3)"}
    if source == "b":
        d = (ROOT / ck_rel).parent
        out["tp_base_test_outputs_present"] = sorted(n for n in TP_TEST_OUTPUTS if (d / n).exists())
    return out


def determinism(spec: Mapping[str, Any], ck: Path, aspect: str, device: str) -> Dict[str, Any]:
    """Strict load (``load_tier_b``) and two validation forwards; run in search mode."""
    import numpy as np
    from go_function_fusion import data as D
    ctx = D.aspect_context(aspect, ("train", "valid"))
    proteins = ctx.labels["proteins_valid"]
    x = D.tier_b_features(spec, "valid", proteins)
    model = D.load_tier_b(spec, ck, ctx.vocab, x.shape[1], device)
    l1 = D.base_logits(model, x, device)
    l2 = D.base_logits(model, x, device)
    if l1.shape != (len(proteins), len(ctx.vocab)):
        raise MP.Fatal(f"validation logits shape {l1.shape} != ({len(proteins)}, {len(ctx.vocab)})")
    d = float(np.max(np.abs(l1.astype(np.float64) - l2.astype(np.float64))))
    if d != 0.0 or not np.isfinite(l1).all():
        raise MP.Fatal(f"validation forward not deterministic / finite (max abs diff {d})")
    return {"strict_load": True, "valid_shape": list(l1.shape), "max_abs_diff_two_calls": d,
            "device": device}


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=MP.RESULTS_CHOICES[0])
    ap.add_argument("--name", choices=tuple(BASES), required=True)
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", choices=("a", "b"), required=True)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    a = ap.parse_args(argv)
    MP.refuse_on_mac("G-TP")
    from go_function.pins import sha256_bytes, sha256_file
    res = MP.results_dir(a.results)
    spec_p, ck = ROOT / a.spec, ROOT / a.checkpoint
    spec_rel, ck_rel = MP.rel(spec_p), MP.rel(ck)
    if not (spec_rel.startswith("results/go_function/specs/") and spec_rel.endswith(".json")):
        raise MP.Fatal("--spec must be a GF materialized spec under results/go_function/specs/")
    if not (ck_rel.startswith(SOURCE_PREFIX[a.source]) and ck_rel.endswith("/checkpoint.pt")):
        raise MP.Fatal(f"source ({a.source}) checkpoints live under {SOURCE_PREFIX[a.source]}")
    raw = spec_p.read_bytes()
    spec = json.loads(raw)
    check_spec(spec, a.name, a.aspect)
    entry: Dict[str, Any] = {"spec": spec_rel, "spec_sha256": sha256_bytes(raw),
                             "checkpoint": ck_rel, "checkpoint_sha256": sha256_file(ck),
                             "source": a.source}
    if a.source == "b":
        prov = ck.parent / "tp_base_provenance.json"
        if not prov.exists():
            raise MP.Fatal("source (b) needs tp_base_provenance.json next to the checkpoint")
        pj = json.loads(prov.read_text())
        if pj.get("spec_sha256") != entry["spec_sha256"]:
            raise MP.Fatal("provenance spec sha256 differs from the GF materialized spec")
        if pj.get("exit_code") != 0:
            raise MP.Fatal(f"provenance exit_code {pj.get('exit_code')} (train.run did not finish)")
        pin_p = ROOT / GF_CODE_PIN
        if not pin_p.exists():
            raise MP.Fatal(f"{GF_CODE_PIN} missing (GF G8): source (b) cannot be verified")
        cmp = compare_gf_code(pj.get("go_function_code", {}), json.loads(pin_p.read_text()).get("files", {}))
        if not cmp["equal"]:
            raise MP.Fatal(f"provenance go_function/ code differs from GF's code pin: {cmp['differs']}")
        entry["provenance_sha256"] = sha256_file(prov)
        entry["gf_code_pin_sha256"] = sha256_file(pin_p)
        entry["go_function_code_equal_gf_pin"] = cmp
    entry["exposure_10_4"] = exposure_facts(ck_rel, a.source)
    if ck_rel.startswith(guard.GF_RESULTS):
        guard.allow_checkpoints([ck_rel])
    guard.enter_search_mode()
    entry["determinism"] = determinism(spec, ck, a.aspect, a.device)
    entry["utc"] = guard.utc_now()
    print(MP.record_pin(res, "G-TP", {f"{a.aspect}/{a.name}": entry}))
    return MP.EXIT_OK


if __name__ == "__main__":
    sys.exit(MP.main_wrapper(run))
