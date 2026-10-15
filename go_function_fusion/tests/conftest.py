"""Shared fixtures for go_function_fusion tests (CPU only; no test label is read)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MINI_OBO = ROOT / "go_function" / "tests" / "fixtures" / "mini.obo"
DGZ = ROOT / "dataset" / "go_function" / "dgz"
SMOKE = ROOT / "results" / "go_function" / "smoke"

#: hand vocabulary over mini.obo; column 3 is the alt id of column 2 (a duplicate)
MINI_IDS = ["GO:0008150", "GO:0000010", "GO:0000011", "GO:0000099", "GO:0000012",
            "GO:0000013", "GO:0000014", "GO:0000015", "GO:0000016"]


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: needs the DGZ release and GF smoke outputs")


def pytest_collection_modifyitems(config, items):
    ok = (DGZ / "go.obo").exists() and (SMOKE / "analysis" / "fusion_F.csv").exists()
    skip = pytest.mark.skip(reason="DGZ release or GF smoke outputs absent")
    for item in items:
        if "needs_data" in item.keywords and not ok:
            item.add_marker(skip)


@pytest.fixture
def mini():
    from go_function.labels import build_vocab
    from go_function.ontology import ancestors, descendants, parse_obo
    ont = parse_obo(MINI_OBO)
    vocab = build_vocab(MINI_IDS, ont)
    anc = ancestors(ont)
    return {"ont": ont, "vocab": vocab, "anc": anc, "desc": descendants(anc), "root": 0,
            "ntr": np.full(len(MINI_IDS), 50)}
