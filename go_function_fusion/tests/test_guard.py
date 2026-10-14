"""Guard: in search mode the process is unable to open test labels or test artefacts."""
import os
import subprocess
import sys

from go_function_fusion import guard
from go_function_fusion.tests.conftest import ROOT


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_is_test_path_rules():
    assert guard.is_test_path(ROOT / "dataset/go_function/dgz/bp/test_data.pkl")
    assert guard.is_test_path(ROOT / "dataset/go_function/dgz/mf/predictions_mlp.pkl")
    assert guard.is_test_path(ROOT / "dataset/go_function/dgz/bp/test_diamond.res")
    assert guard.is_test_path(ROOT / "results/go_function_fusion/test/test_read.json")
    assert guard.is_test_path(ROOT / "results/go_function_fusion/diamond/bp/test_diamond_trainonly.res")
    assert guard.is_test_path(ROOT / "results/go_function/runs/bp/x/scores.npz")
    assert guard.is_test_path(ROOT / "results/go_function/smoke/analysis/fusion_F.csv")
    assert not guard.is_test_path(ROOT / "results/go_function/gates/gate_dgz_eval.json")
    assert not guard.is_test_path(ROOT / "results/go_function/protocol_pins.json")
    assert not guard.is_test_path(ROOT / "dataset/go_function/dgz/bp/valid_data.pkl")
    assert not guard.is_test_path(ROOT / "preprocess/go_function/bp/labels_TP.npz")
    assert not guard.is_test_path("/tmp/elsewhere/test_x")


def test_search_mode_cannot_open_test_labels():
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "import numpy as np, pickle\n"
        "from go_function_fusion import guard, data as D\n"
        "guard.enter_search_mode()\n"
        "attempts = [\n"
        "  lambda: D.load_tp_labels('bp', ('train', 'test')),\n"
        "  lambda: D.read_split_frame('bp', 'test'),\n"
        "  lambda: open('dataset/go_function/dgz/bp/test_data.pkl', 'rb'),\n"
        "  lambda: np.load('results/go_function/smoke/runs/bp/interpro/V0__TP__W2__p01__k0__s13/scores.npz'),\n"
        "  lambda: open('results/go_function/smoke/analysis/fusion_F.csv'),\n"
        "  lambda: open('dataset/go_function/dgz/mf/predictions_mlp.pkl', 'rb'),\n"
        "  lambda: open('results/go_function_fusion/test/test_read.json', 'w'),\n"
        "  lambda: guard.check_split('test'),\n"
        "]\n"
        "for fn in attempts:\n"
        "    try:\n"
        "        fn(); print('LEAK'); sys.exit(3)\n"
        "    except guard.TestAccessError:\n"
        "        pass\n"
        "open('results/go_function/gates/gate_dgz_eval.json').close()\n"
        "print('BLOCKED')\n")
    out = _run(code)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "BLOCKED" in out.stdout


def test_search_mode_label_loader_returns_no_test_member():
    out = _run("import sys; sys.path.insert(0, '.')\n"
               "from go_function_fusion import guard, data as D\n"
               "guard.enter_search_mode()\n"
               "lab = D.load_tp_labels('mf')\n"
               "assert 'test' not in lab and 'proteins_test' not in lab\n"
               "print('OK', lab['valid'].shape)\n")
    if not (ROOT / "preprocess/go_function/mf/labels_TP.npz").exists():
        return
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr


def test_guard_inactive_outside_search_mode_and_one_way():
    out = _run("import sys; sys.path.insert(0, '.')\n"
               "from go_function_fusion import guard\n"
               "guard.check_path('dataset/go_function/dgz/bp/test_data.pkl'); print('OK')\n"
               "guard.enter_search_mode()\n"
               "try:\n"
               "    guard.allow_checkpoints(['results/go_function/runs/x/checkpoint.pt'])\n"
               "except guard.TestAccessError:\n"
               "    print('ONEWAY')\n")
    assert out.returncode == 0 and "OK" in out.stdout and "ONEWAY" in out.stdout, out.stderr


def test_allowed_checkpoint_is_openable_in_search_mode(tmp_path):
    out = _run("import sys; sys.path.insert(0, '.')\n"
               "from go_function_fusion import guard\n"
               "p = 'results/go_function/runs/bp/esm/V0__TP__W2__all__s13/checkpoint.pt'\n"
               "guard.allow_checkpoints([p])\n"
               "guard.enter_search_mode()\n"
               "assert not guard.is_test_path(p)\n"
               "assert guard.is_test_path(p.replace('checkpoint.pt', 'eval.json'))\n"
               "print('OK')\n")
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr


def test_tier_b_source_b_outputs_blocked():
    d = "results/go_function_fusion/tp_bases/V0__TP__W2__all__s13/"
    for name in ("eval.json", "scores.npz", "eval_inputs.npz", "run.json", "epochs.jsonl"):
        assert guard.is_test_path(ROOT / (d + name)), name
        assert guard.is_test_path(ROOT / (d.replace("fusion/", "fusion_dev/") + name)), name
    assert not guard.is_test_path(ROOT / (d + "checkpoint.pt"))
    assert not guard.is_test_path(ROOT / (d + "tp_base_provenance.json"))
    out = _run("import sys; sys.path.insert(0, '.')\n"
               "from go_function_fusion import guard\n"
               "guard.enter_search_mode()\n"
               f"d = '{d}'\n"
               "for n in ('eval.json', 'scores.npz', 'eval_inputs.npz'):\n"
               "    try:\n"
               "        open(d + n, 'rb'); print('LEAK'); sys.exit(3)\n"
               "    except guard.TestAccessError:\n"
               "        pass\n"
               "print('BLOCKED')\n")
    assert out.returncode == 0 and "BLOCKED" in out.stdout, out.stdout + out.stderr


def _script(name):
    import importlib.util
    sdir = ROOT / "scripts/go_function_fusion"
    if str(sdir) not in sys.path:
        sys.path.insert(0, str(sdir))
    spec = importlib.util.spec_from_file_location(f"_gff_{name}", sdir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frozen_star_reads_gf_pinned_finalists(tmp_path):
    import json
    import pytest
    from go_function.pins import sha256_file
    sv = _script("search_valid")
    fin = tmp_path / "finalists_tierA.json"
    fin.write_text(json.dumps({"bases": {"bp/mlp": {"apt": {"c": 3.0, "tau": 10.0}}}}))
    pins = {"schema": 1, "pins": {"G-F": {"finalists_tierA.json": sha256_file(fin)}}}
    (tmp_path / "protocol_pins.json").write_text(json.dumps(pins))
    assert sv.frozen_star(tmp_path, "bp", "mlp") == {"c": 3.0, "tau": 10.0}
    assert sv.search_dir(tmp_path, "bp", "mlp", True).name == "bp_mlp_dv"
    assert sv.search_dir(tmp_path, "bp", "mlp", False).name == "bp_mlp"
    fin.write_text(json.dumps({"bases": {"bp/mlp": {"apt": {"c": 2.0, "tau": 10.0}}}}))
    with pytest.raises(sv.MP.Fatal):
        sv.frozen_star(tmp_path, "bp", "mlp")


def test_verify_tp_bases_checks():
    import pytest
    vt = _script("verify_tp_bases")
    ok = {"aspect": "bp", "variant": "V4", "feature": "interpro", "mode": "study", "pack": None, "seed": 13}
    vt.check_spec(ok, "V4IPR", "bp")
    for bad in ({**ok, "seed": 14}, {**ok, "pack": "p01"}, {**ok, "feature": "esm2_650m"}):
        with pytest.raises(vt.MP.Fatal):
            vt.check_spec(bad, "V4IPR", "bp")
    pin = {"go_function/train.py": "a", "go_function/model.py": "b", "scripts/go_function/x.py": "c"}
    assert vt.compare_gf_code({"go_function/train.py": "a", "go_function/model.py": "b"}, pin)["equal"]
    r = vt.compare_gf_code({"go_function/train.py": "z", "go_function/model.py": "b"}, pin)
    assert not r["equal"] and r["differs"] == ["go_function/train.py"]
    assert not vt.compare_gf_code({"go_function/train.py": "a"}, pin)["equal"]
    assert not vt.compare_gf_code({}, {})["equal"]


def test_search_log_refuses_absolute_paths(tmp_path):
    import pytest
    log = guard.SearchLog(tmp_path / "search_log.jsonl")
    log.log("selection", family="apt", selected={"c": 2})
    with pytest.raises(ValueError):
        log.log("bad", path=str(ROOT / "x"))
    assert (tmp_path / "search_log.jsonl").read_text().count("\n") == 1


def test_remote_only_scripts_refuse_on_mac():
    import platform
    if platform.system() != "Darwin":
        return
    for script, args in (("score_test.py", []), ("train_tp_base.py", ["--spec", "x.json"]),
                         ("s2_grid_fusion.py", ["--with-labels", "--smoke"]),
                         ("verify_tp_bases.py", ["--name", "V0IPR", "--aspect", "bp", "--spec", "x.json",
                                                 "--checkpoint", "x/checkpoint.pt", "--source", "a"]),
                         ("search_valid.py", ["--aspect", "bp", "--base", "mlp", "--diamond-valid",
                                              "--results", "results/go_function_fusion_dev"])):
        out = subprocess.run([sys.executable, f"scripts/go_function_fusion/{script}", *args], cwd=ROOT,
                             capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        assert out.returncode == 4 and "REFUSED" in out.stderr, (script, out.stderr)
    out = subprocess.run(["bash", "scripts/go_function_fusion/diamond_valid.sh", "bp"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 4 and "REFUSED" in out.stderr
