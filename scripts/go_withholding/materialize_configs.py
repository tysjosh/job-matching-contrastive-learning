"""Write the four withholding-study cell configs (design 12.1, gate G4).

``--lambda <float>`` writes ``baseline.json``, ``loss_only.json``,
``neg_only.json`` and ``both.json`` into ``withholding_constants.CONFIG_DIR``
plus ``configs.sha256.json`` (exactly those four keys) and appends a
``G4-configs`` pin. ``--lambda none`` writes only ``baseline.json``,
``neg_only.json`` and ``loss_pilot_template.json`` (the loss_only config with
weight 0.0, for the lambda pilot) and appends no pin.

Baseline/loss_only start from ``config/go_decomposition_go_ordinal/randwin.json``,
neg_only/both from ``go_P_simgic.json``. The same overrides go into all four,
including ``go_ppi_unannotated_anchor_window: true`` (a shared field, design 5.6);
``validation_path`` is removed.

Paths (user decision B, remote execution): every ``_path``/``_dir``/``_cache``
value written is repo-root-anchored, ``"$ROOT/<relative>"``; the runner and the
study scripts expand ``$ROOT`` to the checkout they run from
(``withholding_constants.expand_root_token``). No absolute path is written. This
replaces the design's "absolute values" rule with the same fail-closed intent: a
bare relative (cwd-dependent) value is still refused.

Factor audit (fatal, nothing written, exit 2): the candidate dicts are written to
a temporary dir under ``CONFIG_DIR/.staging/`` and every pair must satisfy
``report_factor(...)["unexpected_fields"] == []`` and
``set(diff) == EXPECTED_PAIR_DIFF[pair]``.

Usage::

    "$PY" "$R/scripts/go_withholding/materialize_configs.py" --lambda none
    "$PY" "$R/scripts/go_withholding/materialize_configs.py" --lambda 0.3
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("materialize_configs")

BASE_DIR = ROOT / "config" / "go_decomposition_go_ordinal"
BASES = {"baseline": "randwin.json", "loss_only": "randwin.json",
         "neg_only": "go_P_simgic.json", "both": "go_P_simgic.json"}
TEMPLATE = "loss_pilot_template"
STUDY_TAG = "go_withholding_v4"

EXPECTED_PAIR_DIFF = {
    ("baseline", "loss_only"): {"go_ontology_loss_weight"},
    ("neg_only", "both"): {"go_ontology_loss_weight"},
    ("baseline", "neg_only"): {"go_ppi_go_tiered_negatives", "go_ppi_tier_sampling"},
    ("loss_only", "both"): {"go_ppi_go_tiered_negatives", "go_ppi_tier_sampling"},
}
#: Pairs audited under ``--lambda none`` (only files it writes).
EXPECTED_PAIR_DIFF_NONE = {
    ("baseline", "neg_only"): {"go_ppi_go_tiered_negatives", "go_ppi_tier_sampling"},
    ("baseline", TEMPLATE): set(),
}


class ConfigError(RuntimeError):
    pass


def overrides() -> Dict[str, Any]:
    """The design 12.1 overrides, paths as ``$ROOT/...`` tokens (read at call time)."""
    return {
        "go_ppi_split_dir": wc.root_token(Path(wc.CONDITIONS) / "all"),
        "go_ppi_converted_dir": wc.root_token(Path(wc.CONVERTED)),
        "go_index_cache": wc.root_token(Path(wc.CONVERTED) / "go_index_P.pkl"),
        "go_obo_path": wc.root_token(Path(wc.ROOT) / "dataset/go_ppi/bulk/go-basic.obo"),
        "go_annotation_path": wc.root_token(
            Path(wc.ROOT) / "dataset/go_ppi/bulk/goa_human.gaf.gz"),
        "embedding_cache_path": wc.root_token(Path(wc.EMBED_CACHE)),
        "checkpoint_frequency": 1000000000,
        "go_similarity_mode": "simgic",
        "go_aspect_weights": {"P": 1.0, "F": 1.0, "C": 1.0},
        "go_ppi_unannotated_anchor_window": True,
        "_study": STUDY_TAG,
    }


def build_candidates(lam: Optional[float]) -> Dict[str, Dict[str, Any]]:
    """Candidate config dicts by name; ``lam=None`` -> the ``--lambda none`` set."""
    out: Dict[str, Dict[str, Any]] = {}
    names = (["baseline", "loss_only", "neg_only", "both"] if lam is not None
             else ["baseline", "neg_only", TEMPLATE])
    for name in names:
        cell = "loss_only" if name == TEMPLATE else name
        cfg = json.loads((BASE_DIR / BASES[cell]).read_text(encoding="utf-8"))
        cfg.pop("validation_path", None)
        cfg.update(overrides())
        cfg["_cell"] = cell
        if name == TEMPLATE:
            cfg["_template"] = "lambda pilot copies this and sets go_ontology_loss_weight"
        loss_on = cell in ("loss_only", "both")
        cfg["go_ontology_loss_weight"] = float(lam) if (loss_on and lam is not None) else 0.0
        out[name] = cfg
    return out


def check_paths(candidates: Dict[str, Dict[str, Any]]) -> None:
    """Every path-scope value must be root-anchored (no absolute, no bare relative)."""
    bad = {name: wc.config_path_problems(cfg, allow_absolute=False)
           for name, cfg in candidates.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise ConfigError(f"non-root-anchored path values: {bad}")


def _dump(cfg: Dict[str, Any]) -> bytes:
    return (json.dumps(cfg, indent=2) + "\n").encode("utf-8")


def audit(candidates: Dict[str, Dict[str, Any]], expected: Dict[tuple, set],
          config_dir: Path) -> Dict[str, Any]:
    """The exact-diff-set factor audit, on files written under ``.staging``."""
    from scripts.run_learning_curve_point import report_factor

    staging_root = config_dir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=staging_root))
    try:
        paths = {}
        for name, cfg in candidates.items():
            paths[name] = tmp / f"{name}.json"
            paths[name].write_bytes(_dump(cfg))
        report, failures = {}, []
        for (a, b), want in expected.items():
            rep = report_factor("go_ppi", paths[a], paths[b])
            got = set(rep["diff"])
            report[f"{a}__{b}"] = sorted(got)
            if rep["unexpected_fields"]:
                failures.append(f"{a} vs {b}: unexpected_fields {rep['unexpected_fields']}")
            if got != want:
                failures.append(f"{a} vs {b}: diff {sorted(got)} != expected {sorted(want)}")
        if failures:
            raise ConfigError("factor audit failed: " + "; ".join(failures))
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            staging_root.rmdir()
        except OSError:
            pass


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def materialize(lam: Optional[float]) -> Dict[str, Any]:
    if lam is not None and not (math.isfinite(lam) and lam > 0):
        raise ConfigError(f"--lambda must be a finite float > 0 or 'none', got {lam!r}")
    config_dir = Path(wc.CONFIG_DIR)
    wc.assert_writable_target(config_dir)
    candidates = build_candidates(lam)
    check_paths(candidates)
    report = audit(candidates, EXPECTED_PAIR_DIFF if lam is not None
                   else EXPECTED_PAIR_DIFF_NONE, config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, cfg in candidates.items():
        path = config_dir / f"{name}.json"
        _write(path, _dump(cfg))
        files[path.name] = wc.sha256_file(path)
    summary: Dict[str, Any] = {"lambda": lam, "files": files, "audit": report,
                               "pinned": False}
    if lam is not None:
        cells = {k: files[k] for k in ("baseline.json", "loss_only.json", "neg_only.json",
                                       "both.json")}
        sha_path = config_dir / "configs.sha256.json"
        _write(sha_path, (json.dumps(cells, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        wc.append_pin("G4-configs", {"configs_sha256_json": wc.sha256_file(sha_path),
                                     "files": cells, "lambda": float(lam)})
        summary["pinned"] = True
    return summary


def _parse_lambda(text: str) -> Optional[float]:
    if text.strip().lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--lambda must be a float or 'none', got {text!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lambda", dest="lam", required=True, type=_parse_lambda,
                    help="the pilot's lambda (float > 0), or 'none' before the pilot")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        summary = materialize(args.lam)
    except Exception as exc:  # noqa: BLE001 - nothing is written on any failure
        logger.error("materialize_configs failed: %s: %s", type(exc).__name__, exc)
        return 2
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
