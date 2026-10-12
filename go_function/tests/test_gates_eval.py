"""Gate G2 driver on synthetic predictions (plan item 7): pass, re-run unchanged,
planted miss -> exit 3."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from go_function import evaluate as E
from go_function.constants import EXIT_OK, EXIT_STOP
from go_function.ontology import DgzRule
from go_function.tests.test_evaluate import _dgz_reference

ROOT = Path(__file__).resolve().parents[2]
TERMS = ["GO:0000010", "GO:0000011", "GO:0000012", "GO:0000013", "GO:0000014",
         "GO:0000015", "GO:0000016"]
DIRECT = ["GO:0000011", "GO:0000012", "GO:0000013", "GO:0000016", "GO:0000099"]


def _gate():
    spec = importlib.util.spec_from_file_location(
        "gate_dgz_eval", ROOT / "scripts" / "go_function" / "gate_dgz_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _props(dgz, rng, n):
    return [sorted(dgz.prop(rng.choice(DIRECT, size=rng.integers(1, 3), replace=False)))
            for _ in range(n)]


@pytest.fixture
def world(tmp_path, mini_obo):
    rng = np.random.default_rng(7)
    d = tmp_path / "dgz"
    (d / "bp").mkdir(parents=True)
    shutil.copy(mini_obo, d / "go.obo")
    dgz = DgzRule(d / "go.obo")
    pd.DataFrame({"gos": TERMS}).to_pickle(d / "bp" / "terms.pkl")
    for split, n in (("train", 50), ("valid", 10)):
        pd.DataFrame({"prop_annotations": _props(dgz, rng, n)}).to_pickle(d / "bp" / f"{split}_data.pkl")
    test_prop = _props(dgz, rng, 40)
    y = np.array([[g in set(x) for g in TERMS] for x in test_prop], dtype=float)
    expected = {}
    for model, noise in (("mlp", 0.6), ("deepgozero", 0.4)):
        preds = np.clip(0.5 * y + noise * rng.random(y.shape), 0, 1).astype(np.float32)
        pd.DataFrame({"preds": list(preds), "prop_annotations": test_prop}).to_pickle(
            d / "bp" / f"predictions_{model}.pkl")
        tv = [list(x) for s in ("train", "valid")
              for x in pd.read_pickle(d / "bp" / f"{s}_data.pkl")["prop_annotations"]]
        ic = E.dgz_ic(dgz, tv + test_prop)
        go_set = E.dgz_namespace_terms(dgz, "bp")
        f, s, a = _dgz_reference(preds.astype(np.float64), TERMS, test_prop, go_set, ic)
        aucs = [roc_auc_score(y[:, j], preds[:, j]) for j in range(len(TERMS))
                if 0 < y[:, j].sum() < len(y)]
        expected[model] = {"fmax": f, "smin": s, "aupr": a, "auc": float(np.mean(aucs))}
    return d, expected


def _published(tmp_path, expected, bump=None):
    vals = {m: {k: {"value": round(v, 3), "cell": f"synthetic {m} {k}"} for k, v in e.items()}
            for m, e in expected.items()}
    if bump:
        m, k, delta = bump
        vals[m][k]["value"] = round(vals[m][k]["value"] + delta, 3)
    p = tmp_path / "published.json"
    p.write_text(json.dumps({"schema": 1, "values": {"bp": vals}}))
    return p


def test_gate_pass_then_unchanged(tmp_path, world):
    d, expected = world
    gate = _gate()
    kw = dict(data_dir=d, published_path=_published(tmp_path, expected),
              out_path=tmp_path / "gates" / "gate_dgz_eval.json",
              pins_path=tmp_path / "pins.json", verify=False)
    assert gate.main(["--aspect", "bp", "--device", "cpu"], **kw) == EXIT_OK
    doc = json.loads((tmp_path / "gates" / "gate_dgz_eval.json").read_text())
    sec = doc["aspects"]["bp"]
    assert sec["passed"] is True
    for m in ("mlp", "deepgozero"):
        for k in ("fmax", "smin", "aupr", "auc"):
            assert sec["results"][m][k] == pytest.approx(expected[m][k], abs=1e-10)
    pins = json.loads((tmp_path / "pins.json").read_text())
    assert pins["pins"]["G2"]["bp"]["passed"] is True
    before = (tmp_path / "gates" / "gate_dgz_eval.json").read_bytes()
    assert gate.main(["--aspect", "bp", "--device", "cpu"], **kw) == EXIT_OK
    assert (tmp_path / "gates" / "gate_dgz_eval.json").read_bytes() == before


def test_gate_planted_miss_is_protocol_stop(tmp_path, world):
    d, expected = world
    gate = _gate()
    rc = gate.main(["--aspect", "bp", "--device", "cpu"], data_dir=d,
                   published_path=_published(tmp_path, expected, bump=("deepgozero", "smin", 0.02)),
                   out_path=tmp_path / "gates" / "gate_dgz_eval.json",
                   pins_path=tmp_path / "pins.json", verify=False)
    assert rc == EXIT_STOP
    assert not (tmp_path / "gates" / "gate_dgz_eval.json").exists()
    assert not (tmp_path / "pins.json").exists()
    fail = json.loads((tmp_path / "gates" / "gate_dgz_eval_bp.FAILED.json").read_text())
    missed = [r for r in fail["comparison"] if not r["within_tolerance"]]
    assert [(r["model"], r["metric"]) for r in missed] == [("deepgozero", "smin")]
