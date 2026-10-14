"""Trainer and run driver (design §7, plan item 12) on tiny synthetic data, CPU only:
end to end with every output, schema rejections, non-finite loss -> FAILED stage
train, eval failure keeps eval_inputs.npz, seed determinism, study.json field
rules, code-pin import coverage, and the orchestrator-required closed-world test
(the trainer's per-row loss mask for a W2 condition equals the base data's, empty)."""
import json
import logging

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from go_function import train as T
from go_function import withholding as WH
from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_STOP
from go_function.pins import canonical_json, csr_sha256
from go_function.runtime import FatalError, ROOT
from go_function.tests.test_withholding import T1, T2, Fixture

LOG = logging.getLogger("test_train")
N_IN, N_TERMS = 12, 6


def _study(**kw):
    s = T.load_study()
    s.update({"norm": "layernorm", "dim": 8, "epochs_star": {"bp_interpro": 2}})
    s.update(kw)
    return s


def _spec(**kw):
    s = {"aspect": "bp", "feature": "interpro", "variant": "V0", "regime": "TP", "rule": "W2",
         "retention_draw": 1, "pack": "p01", "k": 0, "seed": 13, "epochs": 2, "lambda_el": None,
         "batch_size": 37, "lr": 0.0005, "milestones": [5, 20], "gamma": 0.1,
         "selection": "final", "out_dir": "results/go_function/test_runs/x", "mode": "smoke"}
    s.update(kw)
    return s


def _data(seed=0, nan=False):
    rng = np.random.default_rng(seed)
    x = sp.csr_matrix((rng.random((80, N_IN)) < 0.3).astype(np.uint8))
    w = rng.normal(size=(N_IN, N_TERMS))
    y = sp.csr_matrix(((x.toarray() @ w) > 0.5).astype(np.uint8))
    xt = x.toarray().astype(np.float32)
    if nan:
        xt[0, 0] = np.nan
    targets = [T.TargetCols(id=f"GO:000000{c}", column=c, Pa=[(c + 1) % N_TERMS],
                            siblings=[(c + 2) % N_TERMS], X=[(c + 3) % N_TERMS], Desc=[])
               for c in (0, 1)]
    return T.RunData(x_train=xt[:50] if nan else x[:50], x_valid=x[50:60], x_eval=x[60:],
                     y_train=y[:50], train_mask=np.zeros((0, 2), np.int32), y_valid=y[50:60],
                     valid_keep_cols=np.arange(1, N_TERMS), vocab_ids=[f"GO:000000{c}" for c in
                                                                      range(N_TERMS)],
                     y_eval=y[60:], eval_rows=np.arange(20),
                     targets=targets,
                     collateral={"cols": np.arange(2, N_TERMS), "ic": np.ones(N_TERMS - 2),
                                 "src": np.zeros(0, np.int64), "dst": np.zeros(0, np.int64)},
                     hashes={"synthetic": "0"})


def _run(tmp_path, name="r", spec=None, data=None, study=None):
    out = tmp_path / name
    code = T.run(spec or _spec(), "cpu", log=LOG, loader=lambda *a: data or _data(),
                 study=study or _study(), out_dir=out)
    return code, out


# ------------------------------------------------------------------ end to end
def test_end_to_end_writes_every_output(tmp_path):
    code, out = _run(tmp_path)
    assert code == EXIT_OK
    names = sorted(p.name for p in out.iterdir())
    assert names == ["DONE.json", "epochs.jsonl", "eval.json", "run.json", "scores.npz"]
    ep = [json.loads(line) for line in (out / "epochs.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in ep] == [0, 1]
    assert {"train_bce", "valid_bce_target_free", "lr", "seconds", "grad_norms", "el"} <= set(ep[0])
    g = ep[0]["grad_norms"][0]
    assert g["batch"] == 0 and g["bce_protein"] > 0 and g["el_protein"] is None
    ev = json.loads((out / "eval.json").read_text())
    assert set(ev["targets"]) == {"GO:0000000", "GO:0000001"}
    assert {"fmax", "smin", "aupr_micro"} <= set(ev["collateral"])
    z = np.load(out / "scores.npz")
    assert z["columns"].tolist() == [0, 1, 2, 3]       # t, Pa, siblings; no X for V0
    assert z["logits"].shape == (20, 4)
    rj = json.loads((out / "run.json").read_text())
    assert rj["spec"]["seed"] == 13
    assert rj["checkpoint"] is None                   # smoke runs keep no checkpoint
    _, sm = _run(tmp_path, "smoke", spec=_spec(mode="smoke", max_train_batches=1))
    assert not (sm / "checkpoint.pt").exists()        # smoke runs keep no checkpoint
    assert json.loads((sm / "epochs.jsonl").read_text().splitlines()[0])["n_batches"] == 1
    for f in out.iterdir():
        if f.suffix in (".json", ".jsonl"):
            assert "/Users/" not in f.read_text() and str(ROOT) not in f.read_text()


def test_el_arm_logs_el_and_both_tower_grad_norms(tmp_path, mini_norm):
    from go_function import normforms as NF
    vocab = ["GO:0000010", "GO:0000011", "GO:0000013", "GO:0000014", "GO:0000015", "GO:0000016"]
    nf = NF.select(NF.load_normal_forms(mini_norm, {g: i for i, g in enumerate(vocab)},
                                        expected_unparsed=None, expected_nothing=None), "V4")
    d = _data()
    d.nf, d.vocab_ids = nf, vocab
    code, out = _run(tmp_path, spec=_spec(variant="V4", lambda_el=1.0, k=0), data=d)
    assert code == EXIT_OK
    ep = json.loads((out / "epochs.jsonl").read_text().splitlines()[0])
    assert set(ep["el"]) == {"nf1", "nf2", "nf3", "nf4"} and ep["el_total"] > 0
    g = ep["grad_norms"][0]
    assert g["el_protein"] == 0.0 and g["el_term"] > 0 and g["ratio_term"] > 0
    z = np.load(out / "scores.npz")
    assert 3 in z["columns"].tolist()                 # X(t) columns stored for V4 at k = 0


def test_seed_determinism_on_cpu(tmp_path):
    _, a = _run(tmp_path, "a")
    _, b = _run(tmp_path, "b")
    assert (a / "eval.json").read_bytes() == (b / "eval.json").read_bytes()
    assert (a / "scores.npz").read_bytes() == (b / "scores.npz").read_bytes()
    la = [json.loads(x)["train_bce"] for x in (a / "epochs.jsonl").read_text().splitlines()]
    lb = [json.loads(x)["train_bce"] for x in (b / "epochs.jsonl").read_text().splitlines()]
    assert la == lb
    _, c = _run(tmp_path, "c", spec=_spec(seed=21))
    assert (a / "scores.npz").read_bytes() != (c / "scores.npz").read_bytes()


def test_non_finite_loss_fails_in_stage_train(tmp_path):
    code, out = _run(tmp_path, data=_data(nan=True))
    assert code == EXIT_FATAL
    f = json.loads((out / "FAILED.json").read_text())
    assert f["stage"] == "train" and f["error"] == "NonFiniteLoss"
    assert not (out / "DONE.json").exists()


def test_eval_failure_keeps_eval_inputs(tmp_path):
    d = _data()
    d.y_eval = sp.csr_matrix(d.y_eval.toarray() * 0)      # n_pos = 0: protocol stop
    code, out = _run(tmp_path, data=d)
    assert code == EXIT_STOP
    assert json.loads((out / "FAILED.json").read_text())["stage"] == "eval"
    assert (out / "eval_inputs.npz").exists()
    # sweep.py rescore -> run.py --rescore: evaluation only, from eval_inputs.npz
    code = T.rescore(_spec(), "cpu", log=LOG, loader=lambda *a: _data(), study=_study(),
                     out_dir=out)
    assert code == EXIT_OK and (out / "DONE.json").exists()
    assert (out / "FAILED.rescored.json").exists() and not (out / "FAILED.json").exists()
    assert not (out / "eval_inputs.npz").exists()
    rj = json.loads((out / "run.json").read_text())
    assert rj["train"]["rescored"] is True and rj["resources"]["max_rss_bytes"] > 0
    with pytest.raises(T.Refused):                        # nothing left to rescore
        T.rescore(_spec(), "cpu", log=LOG, loader=lambda *a: _data(), study=_study(), out_dir=out)


def _epochs_wo_time(out):
    rows = [json.loads(x) for x in (out / "epochs.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("seconds")
    return rows


def _interrupt_at(monkeypatch, call):
    """Raise KeyboardInterrupt in the ``call``-th validation pass (epoch call - 1)."""
    real, n = T.validation_bce, {"i": 0}

    def flaky(*a, **k):
        n["i"] += 1
        if n["i"] == call:
            raise KeyboardInterrupt
        return real(*a, **k)
    monkeypatch.setattr(T, "validation_bce", flaky)
    return real


@pytest.mark.parametrize("variant", ["V0", "V4"])
def test_resume_reproduces_the_uninterrupted_run(tmp_path, monkeypatch, mini_norm, variant):
    """Checkpoint-and-resume: a run interrupted after epoch 2 of 4 and resumed from
    resume.pt gives the same epochs, eval.json and scores.npz as an uninterrupted run."""
    from go_function import normforms as NF
    monkeypatch.setattr(T, "RESUME_EVERY_S", 0.0)       # snapshot after every epoch

    def data():
        d = _data()
        if variant == "V4":
            vocab = ["GO:0000010", "GO:0000011", "GO:0000013", "GO:0000014", "GO:0000015",
                     "GO:0000016"]
            d.nf = NF.select(NF.load_normal_forms(mini_norm, {g: i for i, g in enumerate(vocab)},
                                                  expected_unparsed=None, expected_nothing=None),
                             "V4")
            d.vocab_ids = vocab
        return d
    spec = _spec(epochs=4, variant=variant, lambda_el=1.0 if variant == "V4" else None)
    code, ref = _run(tmp_path, "ref", spec=spec, data=data())
    assert code == EXIT_OK and not (ref / T.RESUME_FILE).exists()

    real = _interrupt_at(monkeypatch, 3)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, "cut", spec=spec, data=data())
    cut = tmp_path / "cut"
    assert (cut / T.RESUME_FILE).exists() and not (cut / "DONE.json").exists()
    assert json.loads((cut / "FAILED.json").read_text())["error"] == "KeyboardInterrupt"
    assert len((cut / "epochs.jsonl").read_text().splitlines()) == 2
    assert T.resumable(cut)

    monkeypatch.setattr(T, "validation_bce", real)
    code, _ = _run(tmp_path, "cut", spec=spec, data=data())
    assert code == EXIT_OK
    assert not (cut / T.RESUME_FILE).exists() and (cut / "FAILED.interrupted1.json").exists()
    assert _epochs_wo_time(cut) == _epochs_wo_time(ref)
    assert (cut / "eval.json").read_bytes() == (ref / "eval.json").read_bytes()
    assert (cut / "scores.npz").read_bytes() == (ref / "scores.npz").read_bytes()
    assert json.loads((cut / "run.json").read_text())["train"]["resumed_at_epochs"] == [2]


def test_resume_refuses_a_changed_spec_and_real_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RESUME_EVERY_S", 0.0)
    _interrupt_at(monkeypatch, 2)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, "r", spec=_spec(epochs=3))
    monkeypatch.undo()
    code, out = _run(tmp_path, "r", spec=_spec(epochs=3, seed=21))   # different spec
    assert code == EXIT_FATAL
    f = json.loads((out / "FAILED.json").read_text())
    assert f["stage"] == "train" and "different spec" in f["message"]
    assert not T.resumable(out)                         # a real failure is not resumed
    with pytest.raises(T.Refused):
        _run(tmp_path, "r", spec=_spec(epochs=3))


def test_refuses_non_empty_out_dir(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "x").write_text("")
    with pytest.raises(T.Refused):
        _run(tmp_path)


def test_preflight_failure_writes_failed_preflight(tmp_path):
    def boom(*a):
        raise FatalError("pin mismatch")
    out = tmp_path / "p"
    code = T.run(_spec(), "cpu", log=LOG, loader=boom, study=_study(), out_dir=out)
    assert code == EXIT_FATAL
    assert json.loads((out / "FAILED.json").read_text())["stage"] == "preflight"
    out2 = tmp_path / "q"
    assert T.run(_spec(), "cpu", log=LOG, loader=lambda *a: _data(), study=_study(norm=None),
                 out_dir=out2) == 4                       # norm unset before G3: refused


# ------------------------------------------------------------------ schema (§7.1)
@pytest.mark.parametrize("change", [
    {"extra": 1},                                         # unknown field
    {"max_train_batches": 3, "mode": "pilot"},            # smoke only
    {"rule": "WDGZrel"},                                  # DGZ regime only
    {"regime": "DGZ", "rule": "W2drop", "k": 5},          # k = 0 only
    {"lambda_el": 1.0},                                   # V0 has no EL
    {"variant": "V4"},                                    # EL arm needs lambda
    {"regime": "TPnoIPI"},                                # MF only
    {"pack": None},                                       # pack null only for k = all
    {"k": 3},
    {"retention_draw": 2},                                # draw 2 only k in {1, 5}
    {"lr": 0.001},                                        # must equal study.json
    {"milestones": [1, 3]},
    {"out_dir": "/tmp/x"},
    {"out_dir": "results/other/x"},
    {"selection": "best_valid_loss"},                     # gates only
    {"mode": "sweep"},
    {"epochs": 0},
    {"nf4_radius": "class"},                              # G4 diagnostic only
    {"mode": "study", "seed": 7},                         # study seeds only
    {"mode": "study", "epochs": 3},                       # must equal E*
    {"mode": "gate", "variant": "V4", "lambda_el": 0.5},  # lambda = 1 in gates
])
def test_schema_rejections(change):
    with pytest.raises(FatalError):
        T.validate_spec(_spec(**change), _study())


def test_schema_accepts_smoke_gate_and_all():
    s = T.validate_spec(_spec(mode="smoke", max_train_batches=40), _study())
    assert s.max_train_batches == 40 and s.condition_dir.endswith("TP/W2/p01_k0")
    g = T.validate_spec(_spec(mode="gate", variant="V4", lambda_el=1.0, regime="DGZ", pack=None,
                              k="all", selection="best_valid_loss", seed=42, epochs=256,
                              nf4_radius="class"), _study())
    assert g.condition_dir.endswith("DGZ/all") and g.condition_rule == "all"
    st = T.validate_spec(_spec(mode="study", seed=13), _study())
    assert st.run_name == "V0__TP__W2__p01__k0__s13"
    d2 = T.validate_spec(_spec(retention_draw=2, k=5), _study())
    assert d2.condition_dir.endswith("p01_k5_d2") and d2.run_name.endswith("__d2__s13")


def test_gate_v0_uses_the_dgz_mlp_profile_only(tmp_path):
    """Protocol A6: gate-mode V0 must use deepgopro.py's lr, milestones and dropout;
    no other spec may carry them, and the model gets the spec's dropout."""
    gate = dict(mode="gate", regime="DGZ", pack=None, k="all", selection="best_valid_loss",
                seed=42, epochs=256)
    p = T.DGPRO_MLP
    g = T.validate_spec(_spec(**gate, lr=p["lr"], milestones=p["milestones"],
                              dropout=p["dropout"]), _study())
    assert (g.lr, g.milestones, g.dropout) == (1e-3, (1, 3), 0.3)
    with pytest.raises(FatalError):                       # study settings: not the MLP profile
        T.validate_spec(_spec(**gate), _study())
    with pytest.raises(FatalError):                       # profile without its dropout
        T.validate_spec(_spec(**gate, lr=p["lr"], milestones=p["milestones"]), _study())
    with pytest.raises(FatalError):                       # V4 gate keeps deepgozero.py's settings
        T.validate_spec(_spec(**{**gate, "variant": "V4", "lambda_el": 1.0}, lr=p["lr"],
                              milestones=p["milestones"], dropout=p["dropout"]), _study())
    with pytest.raises(FatalError):                       # study / smoke specs never set dropout
        T.validate_spec(_spec(dropout=0.3), _study())
    d = _data()
    m = T.build_run_model(g, _study(), d, "cpu")
    assert m.net[0].dropout.p == 0.3
    assert T.build_run_model(T.validate_spec(_spec(), _study()), _study(), d,
                             "cpu").net[0].dropout.p == 0.1


def test_set_study_field(tmp_path):
    p = tmp_path / "study.json"
    s = T.load_study()
    s["norm"] = None
    p.write_bytes(canonical_json(s))
    assert T.set_study_field("norm", "layernorm", p) == "written"
    assert T.set_study_field("norm", "layernorm", p) == "unchanged"
    with pytest.raises(FatalError):
        T.set_study_field("norm", "batchnorm1d", p)
    assert json.loads(p.read_text())["norm"] == "layernorm"


def test_study_json_values():
    s = T.load_study()
    assert (s["batch_size"], s["lr"], s["milestones"], s["gamma"], s["dropout"], s["dim"],
            s["margin"], s["lambda_el"]) == (37, 5e-4, [5, 20], 0.1, 0.1, 1024, 0.1, 1.0)
    assert s["seeds"] == [13, 21, 42, 87, 123] and s["ks"] == [0, 1, 5, 10, 25, "all"]
    assert s["retention_seeds"] == {"draw1": 20261011, "draw2": 20261012}
    assert s["analysis_seed"] == 20261011 and s["epochs_star"] == {}


# ------------------------------------------------------------------ code pin (§13.1)
def test_import_coverage(tmp_path):
    from go_function.pins import sha256_file
    mine = str(ROOT / "go_function" / "train.py")
    with pytest.raises(FatalError):
        T.check_import_coverage("pilot", tmp_path / "none.json",
                                [mine, str(ROOT / "go_ppi_domain" / "withholding.py")])
    with pytest.raises(T.Refused):
        T.check_import_coverage("study", tmp_path / "none.json", [mine])
    assert T.check_import_coverage("smoke", tmp_path / "none.json", [mine])["code_pin_sha256"] is None
    pin = tmp_path / "code_pin.json"
    pin.write_text(json.dumps({"files": {"go_function/train.py": sha256_file(ROOT / "go_function" / "train.py")}}))
    assert T.check_import_coverage("study", pin, [mine, "/elsewhere/x.py"])["n_modules"] == 1
    pin.write_text(json.dumps({"files": {"go_function/train.py": "0" * 64}}))
    with pytest.raises(FatalError):
        T.check_import_coverage("study", pin, [mine])
    with pytest.raises(FatalError):
        T.check_import_coverage("study", pin, [mine, str(ROOT / "go_function" / "model.py")])


def test_study_mode_refuses_a_changed_pinned_file_that_is_not_imported(tmp_path):
    """Review finding 1: go_function.model is imported after preflight, so study mode
    hashes every pinned file on disk, not only the imported ones."""
    from go_function.pins import sha256_file
    mine = str(ROOT / "go_function" / "train.py")
    pin = tmp_path / "code_pin.json"
    good = {"go_function/train.py": sha256_file(ROOT / "go_function" / "train.py"),
            "go_function/model.py": sha256_file(ROOT / "go_function" / "model.py")}
    pin.write_text(json.dumps({"files": good}))
    assert T.check_import_coverage("study", pin, [mine])["n_pinned_files_verified"] == 2
    pin.write_text(json.dumps({"files": {**good, "go_function/model.py": "0" * 64}}))
    with pytest.raises(FatalError, match="go_function/model.py"):
        T.check_import_coverage("study", pin, [mine])          # model.py not imported yet
    assert T.check_import_coverage("smoke", pin, [mine])["n_modules"] == 1
    pin.write_text(json.dumps({"files": {**good, "go_function/gone.py": "0" * 64}}))
    with pytest.raises(FatalError, match="missing files"):
        T.check_import_coverage("study", pin, [mine])


def test_pilot_run_never_evaluates_on_test(tmp_path, monkeypatch):
    """Protocol §6.4 (amendment A4): a pilot run makes no test forward pass, writes
    an eval.json stub and no scores.npz, and keeps its checkpoint."""
    def forbidden(*a, **k):
        raise AssertionError("pilot run touched test evaluation")
    monkeypatch.setattr(T, "eval_logits", forbidden)
    monkeypatch.setattr(T.E, "class_metrics", forbidden)
    monkeypatch.setattr(T.E, "load_eval_labels", forbidden)
    d = _data()
    d.x_eval = d.y_eval = d.eval_rows = d.collateral = None
    d.targets = []
    code, out = _run(tmp_path, spec=_spec(mode="pilot", k="all", pack=None), data=d)
    assert code == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == ["DONE.json", "checkpoint.pt", "epochs.jsonl",
                                                    "eval.json", "run.json"]
    ev = json.loads((out / "eval.json").read_text())
    assert ev["skipped"] == T.PILOT_EVAL_SKIPPED and "targets" not in ev
    assert json.loads((out / "run.json").read_text())["outputs"].keys() == {"eval.json"}


@pytest.mark.slow
@pytest.mark.needs_dgz
def test_pilot_preflight_never_loads_test_labels(monkeypatch):
    """Real-data preflight of a pilot spec (BP InterPro, condition all): the
    evaluation-label loader is never called and no test rows or targets reach the run."""
    def forbidden(*a, **k):
        raise AssertionError("pilot preflight loaded evaluation labels")
    monkeypatch.setattr(T.E, "load_eval_labels", forbidden)
    study = T.load_study()
    spec = T.validate_spec(_spec(mode="pilot", k="all", pack=None, epochs=8,
                                 batch_size=study["batch_size"], lr=study["lr"],
                                 milestones=study["milestones"], gamma=study["gamma"]), study)
    data = T.load_run_data(spec, study, "cpu", LOG)
    assert data.x_eval is None and data.y_eval is None and data.eval_rows is None
    assert data.targets == [] and data.collateral is None
    assert data.info["eval_skipped"] == T.PILOT_EVAL_SKIPPED


def test_noipi_dropped_targets_are_not_scored():
    table = {"targets": {"GO:1": {"noipi": {"kept": True}}, "GO:2": {"noipi": {"kept": False}},
                         "GO:3": {}}}
    noipi = T.validate_spec(_spec(aspect="mf", regime="TPnoIPI"), _study(epochs_star={}))
    tp = T.validate_spec(_spec(aspect="mf"), _study(epochs_star={}))
    assert T.eval_target_ids(noipi, table, ["GO:1", "GO:2", "GO:3"]) == ["GO:1", "GO:3"]
    assert T.eval_target_ids(tp, table, ["GO:1", "GO:2", "GO:3"]) == ["GO:1", "GO:2", "GO:3"]


# ------------------------------------------------------------------ closed world (D19)
@pytest.fixture
def fx(mini_obo):
    return Fixture(mini_obo)


def _batches(n, bs=7):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(13)).numpy()
    return [perm[a:a + bs] for a in range(0, n, bs)]


@pytest.mark.parametrize("regime", ["TP", "DGZ"])
@pytest.mark.parametrize("k", [0, 1, 5])
def test_closed_world_w2_trainer_mask_equals_base_mask(fx, regime, k):
    """Orchestrator requirement: the per-row loss mask the trainer builds for a W2
    condition is identical to the base-data mask (empty: no tensor at all); the only
    differences between the two are the erased labels, so no row-level signal
    identifies the withheld proteins."""
    w2 = fx.cond(regime, "W2", k)
    base = fx.cond(regime, "all", None)
    shape = w2.labels.shape
    m_w2 = T.loss_mask(w2.mask, shape, "cpu")
    m_base = T.loss_mask(base.mask, shape, "cpu")
    assert m_w2 is None and m_base is None             # no mask tensor is allocated
    ys_w2, ys_base = T.RowStore(w2.labels, "cpu"), T.RowStore(base.labels, "cpu")
    W_all = set().union(*(set(w.tolist()) for w in w2.check.W.values()))
    erased = {}
    for p in W_all:
        ts = [t for t in (T1, T2) if p in set(w2.check.W[t].tolist())]
        ann = fx.rd[regime].annotations(p)
        kept = [a for a in ann if not any(fx.prop[regime].implies(a, t) for t in ts)]
        erased[p] = fx.rd[regime].row(p) - fx.prop[regime].cols(kept)
    seen = set()
    for rows in _batches(shape[0]):
        y1, mk1 = T.batch_targets(ys_w2, m_w2, rows)
        y0, mk0 = T.batch_targets(ys_base, m_base, rows)
        assert mk1 is None and mk0 is None              # identical per-row masks
        diff = (y0 & ~y1).numpy()
        assert not (y1 & ~y0).any()                     # nothing added under W2
        for i, p in enumerate(rows.tolist()):
            cols = set(np.flatnonzero(diff[i]).tolist())
            assert cols == erased.get(p, set())         # only erased labels differ
            seen.add(p)
    assert seen == set(range(shape[0]))
    assert any(erased.values())


def test_closed_world_trainer_builds_no_mask_and_trains_on_erased_zeros(fx, tmp_path):
    w2 = fx.cond("TP", "W2", 0)
    n = w2.labels.shape[0]
    x = sp.csr_matrix(np.eye(n, 12, dtype=np.uint8) + np.eye(n, 12, k=-1, dtype=np.uint8))
    d = T.RunData(x_train=x, x_valid=x[:5], x_eval=x, y_train=w2.labels, train_mask=w2.mask,
                  y_valid=w2.labels[:5], valid_keep_cols=np.arange(w2.labels.shape[1]),
                  vocab_ids=list(fx.vocab.ids))
    _, summary = T.train(T.validate_spec(_spec(epochs=1), _study()), _study(), d, "cpu",
                         tmp_path, LOG)
    assert summary["mask_tensor"] is False and summary["n_masked_entries"] == 0
    # contrast: WDGZrel masks whole target columns on every train row
    rel = fx.cond("DGZ", "WDGZrel", 0)
    m = T.loss_mask(rel.mask, rel.labels.shape, "cpu")
    dense = m.rows(np.arange(n)).numpy()
    cols = np.flatnonzero(dense.any(axis=0))
    assert sorted(cols.tolist()) == sorted({fx.col[T1], fx.col[T2]})
    assert dense[:, cols].all()
    assert csr_sha256(rel.labels) == csr_sha256(fx.rd["DGZ"].train)
