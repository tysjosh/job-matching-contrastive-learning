"""A complete, synthetic withholding-study layout for runner / audit / pilot tests.

Builds on ``test_withholding_generator.make_fixture`` (singleton-pack target
table, base splits, converted dir, pruned-looking GoIndex) and adds what the
runner needs: neighbour sets in the target table, a full ``frozen_hashes.json``,
the target-free validation set, conditions, four cell configs (absolute tmp
paths; the real configs use ``$ROOT`` tokens) and ``lambda_choice.json``.
``patch_study(monkeypatch, s)`` points every ``withholding_constants`` study
path at the layout and redirects the pins file.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from go_ppi_domain import withholding as wh
from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.tests.test_withholding_generator import make_fixture

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "config" / "go_decomposition_go_ordinal"
LAMBDA = 0.3


def cell_config(s: Dict[str, Any], cell: str, cond: str = "all", **over) -> Dict[str, Any]:
    tiered = cell in ("neg_only", "both")
    raw = json.loads((BASE_DIR / ("go_P_simgic.json" if tiered else "randwin.json"))
                     .read_text(encoding="utf-8"))
    raw.update({
        "go_ppi_split_dir": str(s["conditions"] / cond),
        "go_ppi_converted_dir": str(s["converted"]),
        "go_index_cache": str(s["converted"] / "go_index_P.pkl"),
        "go_obo_path": str(s["root"] / "absent.obo"),
        "go_annotation_path": str(s["root"] / "absent.gaf.gz"),
        "embedding_cache_path": str(s["root"] / "emb" / f"{cell}.pt"),
        "checkpoint_frequency": 1000000000, "go_similarity_mode": "simgic",
        "go_aspect_weights": {"P": 1.0, "F": 1.0, "C": 1.0},
        "go_ppi_unannotated_anchor_window": True,
        "go_ontology_loss_weight": LAMBDA if cell in ("loss_only", "both") else 0.0,
        "num_epochs": 1, "batch_size": 16, "log_frequency": 1000,
        "_study": "fixture", "_cell": cell,
    })
    raw.update(over)
    return raw


def build_study(root: Path, n_targets: int = 6,
                conds: Iterable[str] = ("all", "p01_k0", "p01_k5"),
                text_encoder_model: str = "fake") -> Dict[str, Any]:
    root = Path(root)
    f = make_fixture(root, n_targets=n_targets)
    table = f["table"]
    targets = table["targets"]
    for i, t in enumerate(targets):
        nxt = targets[(i + 1) % len(targets)]
        t["nbr_train_proteins_sib"] = sorted(nxt["train_proteins"][:6])
        t["nbr_test_proteins_sib"] = []
        t["nbr_train_proteins_2"] = sorted(nxt["train_proteins"][:3])
        t["nbr_test_proteins_2"] = []
    f["table_path"].write_text(json.dumps(table, indent=1, sort_keys=True) + "\n",
                               encoding="utf-8")
    (f["table_path"].parent / "target_table.json.sha256").write_text(
        wc.sha256_file(f["table_path"]) + "\n")
    splits, conv = f["splits"], f["converted"]
    frozen = {"schema_version": 2,
              "test_sha256": wc.sha256_file(splits / "test.jsonl"),
              "train_sha256": wc.sha256_file(splits / "train.jsonl"),
              "validation_sha256": wc.sha256_file(splits / "validation.jsonl"),
              "negative_pools_sha256": wc.sha256_file(splits / "negative_pools.jsonl"),
              "proteins_sha256": wc.sha256_file(conv / "proteins.jsonl"),
              "go_index_P_sha256": wc.sha256_file(conv / "go_index_P.pkl")}
    (splits / "frozen_hashes.json").write_text(json.dumps(frozen, indent=2, sort_keys=True)
                                               + "\n", encoding="utf-8")
    table_sha = wc.sha256_file(f["table_path"])
    base = wh.load_base(splits, conv)
    val_dir = root / "validation_targetfree"
    vman = wh.build_targetfree_validation(table, splits, val_dir, base=base, table_sha=table_sha)
    cond_root = root / "conditions"
    manifests = {}
    for cid in conds:
        m = wh.build_condition(table, cid, splits, val_dir, cond_root / cid, base=base,
                               table_sha=table_sha, index=f["index"])
        manifests[cid] = m
    s = {"root": root, "splits": splits, "converted": conv, "table": table,
         "table_path": f["table_path"], "targets_dir": f["table_path"].parent,
         "val_dir": val_dir, "val_manifest": vman, "conditions": cond_root,
         "manifests": manifests, "frozen": frozen, "index": f["index"],
         "results": root / "results", "config_dir": root / "config"}
    s["config_dir"].mkdir(parents=True, exist_ok=True)
    for cell in wc.CELLS:
        cfg = cell_config(s, cell, text_encoder_model=text_encoder_model)
        (s["config_dir"] / f"{cell}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    lam_dir = s["results"] / "lambda_pilot"
    lam_dir.mkdir(parents=True, exist_ok=True)
    (lam_dir / "lambda_choice.json").write_text(json.dumps({"lambda": LAMBDA}))
    return s


def patch_study(monkeypatch, s: Dict[str, Any]) -> None:
    monkeypatch.setattr(wc, "TARGETS", s["targets_dir"])
    monkeypatch.setattr(wc, "VALIDATION_TF", s["val_dir"])
    monkeypatch.setattr(wc, "CONVERTED", s["converted"])
    monkeypatch.setattr(wc, "SPLITS", s["splits"])
    monkeypatch.setattr(wc, "CONDITIONS", s["conditions"])
    monkeypatch.setattr(wc, "STUDY_RES", s["results"])
    monkeypatch.setattr(wc, "CONFIG_DIR", s["config_dir"])
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(s["root"] / "pins.json"))


def runner_args(s: Dict[str, Any], cell: str, out: Path, cond: str = "all",
                config: Path | None = None, seed: int = 13, extra=()) -> list:
    cdir = s["conditions"] / cond
    return ["--domain", "go_ppi", "--config", str(config or s["config_dir"] / f"{cell}.json"),
            "--train-file", str(cdir / "train.jsonl"), "--go-ppi-split-dir", str(cdir),
            "--withholding-manifest", str(cdir / wh.MANIFEST), "--study-cell", cell,
            "--go-diagnostics", str(out / "go_diagnostics.jsonl"), "--output-dir", str(out),
            "--seed", str(seed), *extra]


def fake_encoder_class():
    """``FakeSentenceTransformer`` that also accepts a single string (``preencode_pool``)."""
    from go_ppi_domain.tests.test_epoch_guard_and_recorder import FakeSentenceTransformer

    class FakeEncoder(FakeSentenceTransformer):
        def encode(self, texts, *a, **k):
            if isinstance(texts, str):
                return super().encode([texts], *a, **k)[0]
            return super().encode(texts, *a, **k)

    return FakeEncoder


def training_config_names() -> set:
    from contrastive_learning.data_structures import TrainingConfig

    return {f.name for f in dataclasses.fields(TrainingConfig)}
