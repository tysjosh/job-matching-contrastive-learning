"""Target-free pilots (G7; design §9, protocol §6.4, D12; plan item 14). Remote.

  .venv/bin/python scripts/go_function/pilot.py --aspect bp --feature esm2_650m --device cuda
  .venv/bin/python scripts/go_function/pilot.py --aspect bp --feature interpro --device cuda --stage estar

Stages (``--stage all`` runs them in this order):

estar   condition ``all``, TP, V0 and V4 (lambda 1), seeds 13/21/42, 64 epochs. For
        each run i_best = the 0-based epoch with minimum target-free validation BCE
        (epochs.jsonl ``valid_bce_target_free``); E* = clip(ceil(median over the
        six runs of (i_best + 1)), 8, 64). Writes epochs_star['<aspect>_<feature>']
        into config/go_function/study.json (an existing different value is fatal).
lambda  descriptive record (D12): V3 and V4, ``all``, TP, seeds 13/21,
        lambda in {0.1, 0.3, 1, 3}, E* epochs: final target-free validation BCE and
        the lambda*EL/BCE gradient-norm ratio per tower (mean over the final epoch's
        records). It never changes study.json's lambda (asserted).
text    esm2_650m only: V5 and V6, ``all``, TP, seed 13, 64 epochs: i_best + 1 and the
        validation BCE at E* (descriptive; never enters E*).

Writes results/go_function/pilots/<aspect>_<feature>/choice.json (specs/, runs/
and logs/ beside it) and, once every stage of the feature is recorded, pin G7.
Runs are ``run.py --spec ... --device ...`` subprocesses (mode pilot, selection
final); a run with DONE.json is reused. Pilot-mode runs themselves never load
test labels or test features and write only an eval.json stub (protocol §6.4,
amendment A4; go_function.train). The pilot reads only epochs.jsonl: an
import probe (subprocess) asserts that loading this script imports neither
go_function.evaluate nor go_function.train, so no test label or target column
is read here.

Exit codes: 0 ok; 2 fatal (failed pilot run, probe failure, E* conflict); 4
refused (E* missing for the lambda/text stages).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_REFUSED  # noqa: E402
from go_function.pins import assert_writable_target, canonical_json, record_pin, sha256_bytes  # noqa: E402
from go_function.runtime import DEVICES, FatalError, repo_rel, setup_logging  # noqa: E402

STUDY = ROOT / "config" / "go_function" / "study.json"
PILOT_ROOT = ROOT / "results" / "go_function" / "pilots"
ESTAR_ARMS, ESTAR_SEEDS = ("V0", "V4"), (13, 21, 42)
LAMBDA_ARMS, LAMBDA_SEEDS, LAMBDAS = ("V3", "V4"), (13, 21), (0.1, 0.3, 1.0, 3.0)
TEXT_ARMS, TEXT_SEED = ("V5", "V6"), 13
MAX_EPOCHS, E_MIN, E_MAX = 64, 8, 64
FORBIDDEN = ("go_function.evaluate", "go_function.train")


class Refused(RuntimeError):
    pass


def _materialize():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "go_function_materialize", ROOT / "scripts" / "go_function" / "materialize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ formulas
def read_epochs(run_dir: Path) -> List[Dict[str, Any]]:
    p = Path(run_dir) / "epochs.jsonl"
    if not p.exists():
        raise FatalError(f"{Path(run_dir).name}: epochs.jsonl is missing")
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def i_best(epochs: Sequence[Dict[str, Any]]) -> int:
    """0-based epoch of minimum target-free validation BCE (first on ties)."""
    vals = [e.get("valid_bce_target_free") for e in epochs]
    if not vals or any(v is None for v in vals):
        raise FatalError("epochs.jsonl lacks valid_bce_target_free values")
    return int(min(range(len(vals)), key=lambda i: (vals[i], i)))


def e_star(i_bests: Sequence[int]) -> int:
    """E* = clip(ceil(median(i_best + 1)), 8, 64)."""
    if not i_bests:
        raise FatalError("E* needs at least one pilot run")
    med = statistics.median([i + 1 for i in i_bests])
    return int(min(E_MAX, max(E_MIN, math.ceil(med))))


def final_ratios(epochs: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Mean lambda*EL/BCE gradient-norm ratio per tower over the final epoch."""
    last = epochs[-1].get("grad_norms") or []
    out = {}
    for tower in ("protein", "term"):
        v = [g[f"ratio_{tower}"] for g in last if g.get(f"ratio_{tower}") is not None]
        out[f"ratio_{tower}"] = float(sum(v) / len(v)) if v else None
    return out


# ------------------------------------------------------------------ runs
def _scrub(text: str) -> str:
    import re
    text = text.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")
    return re.sub(r"/Users/[^\s'\"]*", "<path>", text)


def subprocess_runner(spec_path: Path, out_dir: Path, device: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/go_function/run.py", "--spec", repo_rel(spec_path),
           "--device", device]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    log_path.write_text(_scrub(out.stdout + out.stderr))
    return out.returncode


def _rel_or_name(p: Path, root: Path) -> str:
    try:
        return repo_rel(p)
    except ValueError:                       # tests: pilot root outside the repo
        return "results/go_function/pilots/" + p.relative_to(root).as_posix()


class Pilot:
    def __init__(self, aspect: str, feature: str, device: str, log, *, root: Path = None,
                 study_path: Path = None, pins_path: Path = None,
                 runner: Callable[..., int] = subprocess_runner):
        self.aspect, self.feature, self.device, self.log = aspect, feature, device, log
        self.root = Path(root) if root is not None else PILOT_ROOT
        self.dir = self.root / f"{aspect}_{feature}"
        self.study_path = Path(study_path) if study_path is not None else STUDY
        self.pins_path = pins_path
        self.runner = runner
        self.mat = _materialize()

    @property
    def key(self) -> str:
        return f"{self.aspect}_{self.feature}"

    def study(self) -> Dict[str, Any]:
        return json.loads(self.study_path.read_text())

    def run(self, variant: str, seed: int, epochs: int, lam: Optional[float] = None,
            tag: str = "") -> Path:
        name = self.mat.run_name(variant, "TP", "W2", None, "all", seed) + tag
        out = self.dir / "runs" / name
        if (out / "DONE.json").exists():
            return out
        from go_function.resume import resumable       # not go_function.train (import probe)
        if out.exists() and not resumable(out):
            raise FatalError(f"pilot run {name} exists without DONE.json; inspect and move it")
        spec = self.mat.make_spec(self.study(), aspect=self.aspect, feature=self.feature,
                                  variant=variant, regime="TP", rule="W2", pack=None, k="all",
                                  seed=seed, epochs=epochs, out_dir=_rel_or_name(out, self.root),
                                  mode="pilot", lambda_el=lam)
        sp = self.dir / "specs" / f"{name}.json"
        assert_writable_target(sp)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_bytes(canonical_json(spec))
        self.log.info("pilot run %s (%d epochs) on %s", name, epochs, self.device)
        rc = self.runner(sp, out, self.device, self.dir / "logs" / f"{name}.log")
        if rc != 0 or not (out / "DONE.json").exists():
            raise FatalError(f"pilot run {name} failed (exit {rc}); see its FAILED.json")
        return out

    # ---------------------------------------------------------------- choice.json
    def choice(self) -> Dict[str, Any]:
        p = self.dir / "choice.json"
        if p.exists():
            return json.loads(p.read_text())
        return {"schema": 1, "aspect": self.aspect, "feature": self.feature}

    def save_choice(self, doc: Dict[str, Any]) -> bytes:
        p = self.dir / "choice.json"
        assert_writable_target(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        b = canonical_json(doc)
        tmp = p.with_name(".choice.json.tmp")
        tmp.write_bytes(b)
        tmp.replace(p)
        return b

    def set_estar(self, value: int) -> str:
        doc = self.study()
        cur = (doc.get("epochs_star") or {}).get(self.key)
        if cur is not None:
            if cur != value:
                raise FatalError(f"study.json epochs_star[{self.key}] is {cur}; pilot gives {value}")
            return "unchanged"
        doc.setdefault("epochs_star", {})[self.key] = int(value)
        assert_writable_target(self.study_path)
        tmp = self.study_path.with_name(f".{self.study_path.name}.tmp")
        tmp.write_bytes(canonical_json(doc))
        tmp.replace(self.study_path)
        return "written"

    def require_estar(self) -> int:
        v = (self.study().get("epochs_star") or {}).get(self.key)
        if v is None:
            raise Refused(f"E* for {self.key} is not in study.json: run --stage estar first")
        return int(v)

    # ---------------------------------------------------------------- stages
    def stage_estar(self) -> Dict[str, Any]:
        lam = float(self.study()["lambda_el"])
        runs = {}
        for v in ESTAR_ARMS:
            for s in ESTAR_SEEDS:
                d = self.run(v, s, MAX_EPOCHS, lam if v != "V0" else None)
                ib = i_best(read_epochs(d))
                runs[d.name] = {"variant": v, "seed": s, "i_best": ib, "i_best_plus_1": ib + 1}
        star = e_star([r["i_best"] for r in runs.values()])
        rec = {"runs": runs, "median_i_best_plus_1": statistics.median(
            [r["i_best_plus_1"] for r in runs.values()]), "E_star": star,
            "formula": "clip(ceil(median(i_best + 1)), 8, 64)", "max_epochs": MAX_EPOCHS}
        self.log.info("E* = %d (%s)", star, self.set_estar(star))
        doc = self.choice()
        doc["estar"] = rec
        self.save_choice(doc)
        return rec

    def stage_lambda(self) -> Dict[str, Any]:
        star = self.require_estar()
        lam0 = self.study()["lambda_el"]
        rec = {}
        for v in LAMBDA_ARMS:
            for s in LAMBDA_SEEDS:
                for lam in LAMBDAS:
                    d = self.run(v, s, star, lam, tag=f"__lam{lam:g}__E{star}")
                    ep = read_epochs(d)
                    rec[d.name] = {"variant": v, "seed": s, "lambda_el": lam, "epochs": star,
                                   "final_valid_bce_target_free": ep[-1]["valid_bce_target_free"],
                                   **final_ratios(ep)}
        if self.study()["lambda_el"] != lam0:
            raise FatalError("study.json lambda_el changed during the lambda record")
        doc = self.choice()
        doc["lambda_record"] = {"runs": rec, "lambda_el_study": lam0,
                                "note": "descriptive only (D12): never used to select lambda"}
        self.save_choice(doc)
        return doc["lambda_record"]

    def stage_text(self) -> Optional[Dict[str, Any]]:
        if self.feature != "esm2_650m":
            self.log.info("text record: ESM pilot only (the InterPro grid has no text arms)")
            return None
        star = self.require_estar()
        lam = float(self.study()["lambda_el"])
        rec = {}
        for v in TEXT_ARMS:
            d = self.run(v, TEXT_SEED, MAX_EPOCHS, lam if v == "V6" else None)
            ep = read_epochs(d)
            ib = i_best(ep)
            rec[d.name] = {"variant": v, "seed": TEXT_SEED, "i_best_plus_1": ib + 1,
                           "valid_bce_at_E_star": ep[star - 1]["valid_bce_target_free"]
                           if len(ep) >= star else None,
                           "gap_from_E_star": ib + 1 - star}
        doc = self.choice()
        doc["text_record"] = {"runs": rec, "E_star": star,
                              "note": "descriptive only: never enters E* (risk 14)"}
        self.save_choice(doc)
        return doc["text_record"]

    def maybe_pin(self) -> Optional[str]:
        doc = self.choice()
        need = {"estar", "lambda_record"} | ({"text_record"} if self.feature == "esm2_650m" else set())
        if not need <= set(doc):
            self.log.info("pin G7 waits for %s", sorted(need - set(doc)))
            return None
        b = canonical_json(doc)
        return record_pin("G7", {f"{self.key}/choice.json": sha256_bytes(b),
                                 f"{self.key}/E_star": doc["estar"]["E_star"]},
                          pins_path=self.pins_path)


# ------------------------------------------------------------------ import probe
PROBE = """
import importlib.util, json, sys
sys.path.insert(0, {root!r})
spec = importlib.util.spec_from_file_location("pilot_probe", {path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.e_star([1, 2]); mod._materialize()
{extra}
print(json.dumps(sorted(m for m in sys.modules if m.startswith("go_function"))))
"""


def import_probe(extra: str = "") -> List[str]:
    """go_function modules loaded by importing this script (fresh interpreter)."""
    code = PROBE.format(root=str(ROOT), path=str(Path(__file__).resolve()), extra=extra)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         check=False)
    if out.returncode != 0:
        raise FatalError(f"import probe failed: {_scrub(out.stderr)[-1000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def assert_no_evaluator(modules: Sequence[str]) -> None:
    bad = [m for m in FORBIDDEN if m in modules]
    if bad:
        raise FatalError(f"the pilot imports {bad}: test data must never be reachable")


def main(argv=None, **kw) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--feature", choices=("esm2_650m", "interpro"), required=True)
    ap.add_argument("--device", choices=DEVICES, default="auto")
    ap.add_argument("--stage", choices=("estar", "lambda", "text", "all"), default="all")
    args = ap.parse_args(argv)
    log = setup_logging("pilot")
    try:
        assert_no_evaluator(import_probe())
        p = Pilot(args.aspect, args.feature, args.device, log, **kw)
        if args.stage in ("estar", "all"):
            p.stage_estar()
        if args.stage in ("lambda", "all"):
            p.stage_lambda()
        if args.stage in ("text", "all"):
            p.stage_text()
        log.info("pin G7: %s", p.maybe_pin())
        assert_no_evaluator(list(sys.modules))
    except Refused as e:
        log.error("%s", e)
        return EXIT_REFUSED
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
