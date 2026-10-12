"""Checkpoint gates (plan item 13) on synthetic checkpoints and predictions:
G3 pass / miss / norm detection, G2b status logic plus a synthetic pass and an
L1 load failure ("loaded": false), and G4 spec writing and comparison."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_curve

from go_function import model as M
from go_function import normforms as NF
from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_REFUSED, EXIT_STOP
from go_function.evaluate import propagate_max
from go_function.ontology import DgzRule
from go_function.pins import canonical_json
from go_function.train import load_study

ROOT = Path(__file__).resolve().parents[2]
TERMS = ["GO:0000010", "GO:0000011", "GO:0000013", "GO:0000014", "GO:0000016"]
IPR = [f"IPR{i:06d}" for i in range(10)]
NF_KW = {"expected_unparsed": None, "expected_nothing": None}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "go_function" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.NF_KW = dict(NF_KW) if hasattr(mod, "NF_KW") else None
    return mod


def _iprs(rng, n):
    return [list(rng.choice(IPR, size=rng.integers(1, 4), replace=False)) for _ in range(n)]


def _x(rows):
    from go_function.features import interpro_csr
    return interpro_csr(rows, {g: i for i, g in enumerate(IPR)})[0]


def _study_file(tmp_path):
    s = load_study()
    s["norm"] = None
    p = tmp_path / "study.json"
    p.write_bytes(canonical_json(s))
    return p


# ------------------------------------------------------------------ G3
def _g3_world(tmp_path, mini_obo, mini_norm, norm="layernorm", perturb=0.0):
    d = tmp_path / "dgz"
    (d / "bp").mkdir(parents=True)
    shutil.copy(mini_obo, d / "go.obo")
    shutil.copy(mini_norm, d / "go.norm")
    pd.DataFrame({"gos": TERMS}).to_pickle(d / "bp" / "terms.pkl")
    pd.DataFrame({"interpros": IPR}).to_pickle(d / "bp" / "interpros.pkl")
    nf = NF.load_normal_forms(d / "go.norm", {g: i for i, g in enumerate(TERMS)}, **NF_KW)
    dgz = DgzRule(d / "go.obo")
    rng = np.random.default_rng(5)
    rows = _iprs(rng, 30)
    x = torch.as_tensor(_x(rows).toarray().astype(np.float32))
    src, dst = M.dgz_propagation_pairs(TERMS, dgz)
    torch.manual_seed(0)
    for kind, arm in (("mlp", "V0"), ("deepgozero", "V4")):
        m = M.build_model(arm, len(IPR), len(TERMS), nf if arm == "V4" else None, hidden_dim=16,
                          norm=norm).eval()
        sd = m.state_dict()
        if kind == "mlp":
            sd = {("net.2." + k[3:]) if k.startswith("fc.") else k: v for k, v in sd.items()}
        torch.save(sd, d / "bp" / f"{kind}.th")
        with torch.no_grad():
            p = torch.sigmoid(m(x))
        p = propagate_max(p, src, dst).numpy().astype(np.float64) + perturb
        pd.DataFrame({"interpros": rows, "preds": list(p)}).to_pickle(d / "bp" / f"predictions_{kind}.pkl")
    return d


def _g3(tmp_path, d):
    mod = _script("gate_checkpoints")
    out, pins, study = tmp_path / "g3.json", tmp_path / "pins.json", _study_file(tmp_path)
    code = mod.main(["--aspect", "bp", "--device", "cpu"], data_dir=d, out_path=out,
                    pins_path=pins, study_path=study, verify=False)
    return code, out, pins, study


def test_g3_pass_sets_norm_and_pin(tmp_path, mini_obo, mini_norm):
    code, out, pins, study = _g3(tmp_path, _g3_world(tmp_path, mini_obo, mini_norm))
    assert code == EXIT_OK
    sec = json.loads(out.read_text())["aspects"]["bp"]
    assert sec["passed"] and sec["norm"] == "layernorm"
    assert all(m["max_abs_diff"] <= 1e-6 for m in sec["results"]["models"].values())
    assert json.loads(study.read_text())["norm"] == "layernorm"
    assert json.loads(pins.read_text())["pins"]["G3"]["bp"]["passed"] is True
    mod = _script("gate_checkpoints")                  # re-run: unchanged
    assert mod.main(["--aspect", "bp"], data_dir=tmp_path / "dgz", out_path=out, pins_path=pins,
                    study_path=study, verify=False) == EXIT_OK


def test_g3_miss_is_a_protocol_stop(tmp_path, mini_obo, mini_norm):
    code, out, pins, study = _g3(tmp_path, _g3_world(tmp_path, mini_obo, mini_norm, perturb=1e-3))
    assert code == EXIT_STOP
    assert not out.exists() and (tmp_path / "g3_bp.FAILED.json").exists()
    assert json.loads(study.read_text())["norm"] is None
    assert not pins.exists()


def test_g3_detects_batchnorm(tmp_path, mini_obo, mini_norm):
    code, out, _, study = _g3(tmp_path, _g3_world(tmp_path, mini_obo, mini_norm, norm="batchnorm1d"))
    assert code == EXIT_OK
    assert json.loads(study.read_text())["norm"] == "batchnorm1d"


def test_g3_checkpoint_mismatch_is_fatal(tmp_path, mini_obo, mini_norm):
    d = _g3_world(tmp_path, mini_obo, mini_norm)
    sd = torch.load(d / "bp" / "deepgozero.th", weights_only=True)
    sd.pop("rel_embed.weight")
    torch.save(sd, d / "bp" / "deepgozero.th")
    code, *_ = _g3(tmp_path, d)
    assert code == EXIT_FATAL


# ------------------------------------------------------------------ G2b
def test_g2b_status_logic():
    mod = _script("gate_dgz_zero")
    assert mod.status_of("bp", True, True, True) == "confirmed"
    assert mod.status_of("bp", True, True, False) == "not confirmed"
    assert mod.status_of("bp", False, True, True) == "not checkable"
    assert mod.status_of("bp", True, False, True) == "not checkable"
    assert mod.status_of("mf", True, True, None) == "reproduced"
    assert mod.status_of("mf", True, False, None) == "not reproduced"
    assert mod.status_of("mf", False, False, None) == "not reproduced"
    pc = {f"t{i}": {"match": i < 6} for i in range(7)}
    assert mod.l2_decision("bp", pc)["passed"]
    pc["t5"]["match"] = False
    assert not mod.l2_decision("bp", pc)["passed"]
    assert mod.l2_decision("mf", {f"t{i}": {"match": i < 4} for i in range(5)})["passed"]
    assert mod.l3_class_pass(0.80, 0.81, 0.30)
    assert not mod.l3_class_pass(0.78, 0.81, 0.5)
    assert not mod.l3_class_pass(0.90, 0.81, 0.29)
    assert not mod.l3_class_pass(None, 0.81, 0.5)


def _g2b_world(tmp_path, mini_obo, mini_norm, break_ckpt=False):
    """Withheld class t = GO:0000016 (positively_regulates X = GO:0000011)."""
    t, X = "GO:0000016", "GO:0000011"
    zterms = [g for g in TERMS if g != t]
    d = tmp_path / "dgz"
    (d / "bp").mkdir(parents=True)
    shutil.copy(mini_obo, d / "go.obo")
    shutil.copy(mini_norm, d / "go.norm")
    pd.DataFrame({"gos": TERMS}).to_pickle(d / "bp" / "terms.pkl")
    pd.DataFrame({"gos": zterms}).to_pickle(d / "bp" / "terms_zero.pkl")
    pd.DataFrame({"interpros": IPR}).to_pickle(d / "bp" / "interpros.pkl")
    nf = NF.load_normal_forms(d / "go.norm", {g: i for i, g in enumerate(zterms)}, **NF_KW)
    torch.manual_seed(1)
    m = M.build_model("V4", len(IPR), len(zterms), nf, hidden_dim=16).eval()
    sd = m.state_dict()
    if break_ckpt:
        sd["go_rad.weight"] = torch.zeros(3, 1)
    torch.save(sd, d / "bp" / "deepgozero_zero.th")
    dgz = DgzRule(d / "go.obo")
    rng = np.random.default_rng(2)
    frames = {}
    for split, n in (("train", 40), ("valid", 10), ("test", 30)):
        rows = _iprs(rng, n)
        direct = [[t] if i % 3 == 0 else [X] if i % 3 == 1 else ["GO:0000013"] for i in range(n)]
        prop = [sorted(dgz.prop(a)) for a in direct]
        frames[split] = pd.DataFrame({"proteins": [f"{split}{i}" for i in range(n)],
                                      "interpros": rows, "prop_annotations": prop,
                                      "zero_annotations": prop})
        frames[split].to_pickle(d / "bp" / f"{split}_data.pkl")
    k = nf.class_index[t]
    for name, splits in (("", ("test",)), ("_all", ("train", "valid", "test"))):
        df = pd.concat([frames[s] for s in splits])
        with torch.no_grad():
            s = torch.sigmoid(m.logits_for_classes(
                torch.as_tensor(_x(df["interpros"].tolist()).toarray().astype(np.float32)),
                torch.tensor([k]))).numpy()[:, 0]
        y = np.asarray([t in p for p in df["prop_annotations"]], dtype=np.float32)
        fpr, tpr, _ = roc_curve(y, s)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_pickle(d / "bp" / f"zero_{t}_auc{name}.pkl")
    vocab = TERMS

    def labels(aspect):
        def csr(df):
            return sp.csr_matrix(np.asarray([[g in set(p) for g in vocab]
                                             for p in df["prop_annotations"]], dtype=np.uint8))
        return {r: {"vocab": vocab, "test": csr(frames["test"]), "train": csr(frames["train"]),
                    "proteins_test": frames["test"]["proteins"].tolist()} for r in ("TP", "DGZ")}
    return d, labels


def _g2b(tmp_path, d, labels):
    mod = _script("gate_dgz_zero")
    out, pins = tmp_path / "g2b.json", tmp_path / "pins.json"
    code = mod.main(["--aspect", "bp", "--device", "cpu"], data_dir=d, out_path=out,
                    pins_path=pins, l3_labels=labels, verify=False)
    return code, out, pins


def test_g2b_synthetic_pass(tmp_path, mini_obo, mini_norm):
    d, labels = _g2b_world(tmp_path, mini_obo, mini_norm)
    code, out, pins = _g2b(tmp_path, d, labels)
    assert code == EXIT_OK
    sec = json.loads(out.read_text())["aspects"]["bp"]
    res = sec["results"]
    assert sec["loaded"] is True and res["withheld_classes"] == ["GO:0000016"]
    e = res["L2"]["per_class"]["GO:0000016"]
    assert e["match"] and abs(e["test"]["diff"]) < 1e-9 and abs(e["all"]["diff"]) < 1e-9
    assert res["L2"]["n_match"] == 1 and not res["L2"]["passed"]     # 1 < 6 of 7 needed
    assert res["L3"]["per_class"]["GO:0000016"]["fillers"][0]["X"] == "GO:0000011"
    assert sec["status"] == "not checkable"
    assert res["released_zero_predictions"]["scored"] is False
    assert json.loads(pins.read_text())["pins"]["G2b"]["bp"]["status"] == "not checkable"


def test_g2b_load_failure_writes_loaded_false(tmp_path, mini_obo, mini_norm):
    d, labels = _g2b_world(tmp_path, mini_obo, mini_norm, break_ckpt=True)
    code, out, _ = _g2b(tmp_path, d, labels)
    assert code == EXIT_OK                                           # G2b never blocks
    sec = json.loads(out.read_text())["aspects"]["bp"]
    assert sec["loaded"] is False and sec["status"] == "not checkable"
    assert "go_rad.weight" in sec["results"]["L1"]["error"]


# ------------------------------------------------------------------ G4
def test_g4_write_specs_and_compare(tmp_path):
    mod = _script("gate_trainer")
    sd = tmp_path / "specs"
    assert mod.main(["--write-specs"], spec_dir=sd) == EXIT_OK
    names = sorted(p.name for p in sd.iterdir())
    assert names == ["bp_V0_s42.json", "bp_V4_s42.json", "mf_V0_s42.json", "mf_V4_s42.json"]
    s = json.loads((sd / "bp_V4_s42.json").read_text())
    assert (s["regime"], s["pack"], s["k"], s["selection"], s["seed"], s["epochs"],
            s["lambda_el"], s["mode"], s["feature"]) == ("DGZ", None, "all", "best_valid_loss",
                                                         42, 256, 1.0, "gate", "interpro")
    assert json.loads((sd / "bp_V0_s42.json").read_text())["lambda_el"] is None
    assert mod.main(["--write-specs"], spec_dir=sd) == EXIT_OK      # unchanged re-run
    assert mod.main(["--write-specs", "--fallback", "--diagnostic"], spec_dir=sd) == EXIT_OK
    assert len(list(sd.glob("*.json"))) == 12 and len(list((sd / "diagnostic").glob("*.json"))) == 2

    pub = json.loads((ROOT / "config" / "go_function" / "dgz_published.json").read_text())["values"]

    def reader_for(offsets):
        def read(out_dir):
            name = Path(out_dir).name                  # <aspect>_<variant>_s<seed>
            a, v, s = name.split("_")
            off = offsets.get((a, v, int(s[1:])), 0.0)
            if off is None:
                return None
            model = mod.PUBLISHED_MODEL[v]
            return {m: pub[a][model][m]["value"] + off for m in ("auc", "fmax")}
        return read

    out, pins = tmp_path / "g4.json", tmp_path / "pins.json"
    assert mod.main([], out_path=out, pins_path=pins, reader=reader_for({})) == EXIT_OK
    assert json.loads(out.read_text())["passed"]
    assert "G4" in json.loads(pins.read_text())["pins"]
    out2 = tmp_path / "g4b.json"
    miss = {("bp", "V4", 42): 0.02, ("bp", "V4", 13): None, ("bp", "V4", 21): None}
    assert mod.main([], out_path=out2, reader=reader_for(miss)) == EXIT_REFUSED   # needs fallback
    fb = {("bp", "V4", 42): 0.02, ("bp", "V4", 13): -0.01, ("bp", "V4", 21): -0.005}
    assert mod.main([], out_path=out2, pins_path=tmp_path / "p2.json",
                    reader=reader_for(fb)) == EXIT_OK                            # 3-seed mean
    assert json.loads(out2.read_text())["cells"]["bp_V4"]["status"] == "pass_3_seed"
    bad = {("bp", "V4", 42): 0.03, ("bp", "V4", 13): 0.03, ("bp", "V4", 21): 0.03}
    out3 = tmp_path / "g4c.json"
    assert mod.main([], out_path=out3, reader=reader_for(bad)) == EXIT_STOP
    assert (tmp_path / "g4c.FAILED.json").exists() and not out3.exists()
