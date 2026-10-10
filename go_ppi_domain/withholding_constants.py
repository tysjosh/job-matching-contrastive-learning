"""Constants and small shared helpers for the GO/PPI per-target withholding study.

Design section 1.3 / 13 (``.agents/tasks/ontology-withholding-feasibility/design_go_ppi.md``).

Every study path is a module attribute and every helper reads it at CALL time, so
tests can ``monkeypatch.setattr(withholding_constants, "...", tmp_path / ...)``.
The pins file can be redirected with the ``GO_WITHHOLDING_PINS`` environment
variable (plan decision 1) so scratch runs never touch the real pin record.

Remote execution (user decision B): the grid runs on a remote Linux machine, so
``ROOT`` is the checkout this file lives in (no literal-path assertion), and no
emitted config, manifest or condition file carries an absolute path. Study
configs write repo-root-anchored values ``"$ROOT/<relative>"``
(``ROOT_TOKEN``), which ``expand_root_token`` resolves against ``ROOT`` at load
time; emitted JSON passes through ``relativize_json``. Pins hold hashes only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ raw inputs
#: The 8 raw files of protocol section 11 (values from setup_baseline.md).
RAW_SHA256: Dict[Path, str] = {
    ROOT / "dataset/go_ppi/bulk/9606.protein.info.v12.0.txt.gz":
        "144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c",
    ROOT / "dataset/go_ppi/bulk/9606.protein.links.detailed.v12.0.txt.gz":
        "28a2ea3b22c964e691c0f2f934c5e3e397a4b0a12c833d8787fa02b7847b91b6",
    ROOT / "dataset/go_ppi/bulk/9606.protein.aliases.v12.0.txt.gz":
        "b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680",
    ROOT / "dataset/go_ppi/bulk/9606.protein.physical.links.v12.0.txt.gz":
        "1ca87209a93cf7685e9c2997fbcfbfc69fedc6a803ccc1df917318a651df0823",
    ROOT / "dataset/go_ppi/bulk/goa_human.gaf.gz":
        "db472faff1785878521693af62646546cea6af4a386609dd01c72c0554a46a30",
    ROOT / "dataset/go_ppi/bulk/go-basic.obo":
        "b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc",
    ROOT / "dataset/go_ppi/independent/HuRI.tsv":
        "4cdf94366ff6e23e71ef9ec81922e85b2b3a1fb08e938f7b00d86c027cf1ecd6",
    ROOT / "dataset/go_ppi/independent/Lit-BM.tsv":
        "f39c9c4b8a372286c3eca77d0ff651d837a643d32235eae866daae082663391d",
}

# ------------------------------------------------------------------ study paths
REFERENCE_CONVERTED = ROOT / "preprocess/go_ppi"
REFERENCE_SPLITS = ROOT / "preprocess/go_ppi_splits"
REFERENCE_TEST_SHA256 = "c65a1d2ca0e5c16a1c69ecd70d921e670531d325c52625aa15dba88c2e2481e4"
STUDY_PRE = ROOT / "preprocess/go_withholding"
CONVERTED = STUDY_PRE / "converted"
SPLITS = STUDY_PRE / "splits"
VALIDATION_TF = STUDY_PRE / "validation_targetfree"
TARGETS = STUDY_PRE / "targets"
CONDITIONS = STUDY_PRE / "conditions"
STUDY_RES = ROOT / "results/go_withholding"
PINS = STUDY_RES / "protocol_pins.json"
CODE_PIN = STUDY_RES / "code_pin.json"
CONFIG_DIR = ROOT / "config/go_withholding"
EMBED_CACHE = ROOT / "embedding_cache/go_withholding_text_embeddings.pt"
PROTECTED = (REFERENCE_CONVERTED, REFERENCE_SPLITS, ROOT / "preprocess/go_ppi_lc",
             ROOT / "dataset/go_ppi")

# ------------------------------------------------------------------ study design
RETENTION_SEED = 20251009
ANALYSIS_SEED = 20251009
KS = (0, 1, 5, 10, 25)
SEEDS = (13, 21, 42, 87, 123)
CELLS = ("baseline", "loss_only", "neg_only", "both")
GRID_EPOCHS = 15

#: Renamed run dirs skipped by the sweep and the analysis (design 11, 12.3).
RUN_DIR_IGNORE = (".attempt", ".groupfailed")

#: Code covered by the content pin (design 13).
CODE_PIN_PATHS = ("contrastive_learning", "go_ppi_domain", "scripts", "trials_domain",
                  "run_phase1_embedding_evaluation.py")

PROTOCOL = ROOT / ".agents/tasks/ontology-withholding-feasibility/protocol_go_ppi.md"
AMENDMENT_ID_RE = r"^AM[1-9][0-9]*$"
AMENDMENT_COLUMNS = ("id", "date", "gate", "groups cleared", "cause", "fix", "re-pin")


#: Study artifacts read by the runner (design 4.2 step 4); read at call time.
def lambda_choice_path() -> Path:
    return Path(STUDY_RES) / "lambda_pilot" / "lambda_choice.json"


# ------------------------------------------------- root-anchored paths (B)
ROOT_TOKEN = "$ROOT"
PATH_KEY_SUFFIXES = ("_path", "_dir", "_cache")


def is_path_key(key: str) -> bool:
    """A config key in the design 4.2 step 1 / 12.1 path scope."""
    return not key.startswith("_") and key.endswith(PATH_KEY_SUFFIXES)


def is_root_anchored(value: str) -> bool:
    return value == ROOT_TOKEN or value.startswith(ROOT_TOKEN + "/")


def expand_root_token(value: Any) -> Any:
    """``"$ROOT/x"`` -> ``str(ROOT / "x")`` (``ROOT`` read at call time); else unchanged."""
    if isinstance(value, str) and is_root_anchored(value):
        rest = value[len(ROOT_TOKEN):].lstrip("/")
        return str(Path(ROOT) / rest) if rest else str(ROOT)
    return value


def root_token(path: Path) -> str:
    """``"$ROOT/<path relative to ROOT>"``; ``ValueError`` if ``path`` is outside ROOT."""
    rel = Path(path).relative_to(Path(ROOT))
    return f"{ROOT_TOKEN}/{rel.as_posix()}" if str(rel) != "." else ROOT_TOKEN


def config_path_problems(raw: Dict[str, Any], allow_absolute: bool = True) -> List[str]:
    """Path-scope violations of a JSON config dict (keys from the file only).

    Every non-null string value whose key ends in ``_path``/``_dir``/``_cache``
    (``_``-prefixed keys skipped) must be root-anchored (``"$ROOT/..."``) or, when
    ``allow_absolute``, absolute. A bare relative value would resolve against the
    working directory and is rejected. ``TrainingConfig`` defaults absent from the
    file are out of scope.
    """
    bad = []
    for key, value in raw.items():
        if not is_path_key(key) or value is None or not isinstance(value, str):
            continue
        if is_root_anchored(value):
            continue
        if allow_absolute and Path(value).is_absolute():
            continue
        bad.append(f"{key}={value!r}")
    return bad


def expand_config_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of ``raw`` with root-anchored path values expanded against ``ROOT``."""
    return {k: (expand_root_token(v) if is_path_key(k) else v) for k, v in raw.items()}


def load_study_config(path: Path):
    """``TrainingConfig`` from a study config file, ``$ROOT`` tokens expanded."""
    from contrastive_learning.data_structures import TrainingConfig

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig.from_dict(expand_config_dict(raw))


def _root_pattern() -> "re.Pattern[str]":
    return re.compile(re.escape(str(ROOT)) + r"(?=/|$|[^\w.\-])")


def scrub_root(text: str) -> str:
    """Replace the machine-specific ``ROOT`` prefix with ``$ROOT`` inside ``text``."""
    return _root_pattern().sub(lambda _m: ROOT_TOKEN, text)


def relativize_json(obj: Any) -> Any:
    """Deep copy of a JSON-able value with every ``ROOT`` prefix replaced by ``$ROOT``."""
    if isinstance(obj, dict):
        return {(scrub_root(k) if isinstance(k, str) else k): relativize_json(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [relativize_json(v) for v in obj]
    if isinstance(obj, str):
        return scrub_root(obj)
    return obj


def rel_to_root(path: Path) -> str:
    """Repo-relative POSIX string for ``path`` under ROOT, else ``str(path)``."""
    try:
        return Path(path).relative_to(Path(ROOT)).as_posix()
    except ValueError:
        return str(path)


# ======================================================================= helpers
def assert_writable_target(path: Path) -> None:
    """Fatal (``RuntimeError``) if ``path`` is, or is inside, a PROTECTED dir."""
    p = Path(path).expanduser().resolve()
    for prot in PROTECTED:
        prot = Path(prot).resolve()
        if p == prot or p.is_relative_to(prot):
            raise RuntimeError(f"refusing to write inside protected directory {prot}: {p}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def structural_sha256(records_path: Path) -> str:
    """sha256 over the sorted (protein_id, partner_id, go_grade) tuples of a split file."""
    rows = []
    with open(records_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            m = json.loads(line)["metadata"]
            rows.append(f"{m['protein_id']}\t{m['partner_id']}\t{m['go_grade']}")
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def code_sha256() -> Tuple[str, List[str]]:
    """Content hash of every tracked or untracked (not ignored) pinned ``.py`` file."""
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", *CODE_PIN_PATHS],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    files = sorted(p for p in out if p.endswith(".py") and (ROOT / p).is_file())
    digest = hashlib.sha256(
        "\n".join(f"{p}\t{sha256_file(ROOT / p)}" for p in files).encode("utf-8"))
    return digest.hexdigest(), files


def imported_repo_modules() -> List[str]:
    """Repo ``.py`` files currently in ``sys.modules`` (``.venv`` excluded), relative to ROOT."""
    venv = ROOT / ".venv"
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not os.path.isabs(f):
            # torch registers pseudo-modules with a relative __file__.
            continue
        p = Path(f).resolve()
        if p.suffix != ".py" or not p.is_file():
            continue
        if p.is_relative_to(ROOT) and not p.is_relative_to(venv):
            out.add(str(p.relative_to(ROOT)))
    return sorted(out)


def pins_path() -> Path:
    """``$GO_WITHHOLDING_PINS`` when set, else ``PINS`` (read at call time)."""
    env = os.environ.get("GO_WITHHOLDING_PINS")
    return Path(env) if env else Path(PINS)


def read_pins() -> List[dict]:
    """The pin entries (a JSON list, oldest first); ``[]`` when the file is absent."""
    path = pins_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"{path} is not a JSON list of pin entries")
    return data


def append_pin(gate: str, values: dict) -> None:
    """Append ``{"gate", "time", **values}`` to the pins file (atomic rewrite)."""
    path = pins_path()
    assert_writable_target(path)
    entries = read_pins()
    entry = {"gate": gate,
             "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    entry.update(values)
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pins.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def run_id_for(cond_id: str, cell: str, seed: int) -> str:
    """``<pack>__k<k>__<cell>__s<seed>`` (design 12.3); ``all`` -> ``all__kall__...``."""
    if cond_id == "all":
        pack, k = "all", "all"
    else:
        pack, sep, k = cond_id.rpartition("_k")
        if not sep or not pack or not k.isdigit():
            raise ValueError(f"invalid cond_id {cond_id!r}")
    return f"{pack}__k{k}__{cell}__s{int(seed)}"


def group_run_ids(cond_id: str, seed: int) -> List[str]:
    """The four run ids of a (cond, seed) group, in ``CELLS`` order."""
    return [run_id_for(cond_id, cell, seed) for cell in CELLS]


def group_ok_valid(run_dirs: List[Path]) -> List[bool]:
    """Per dir (cell order): is its ``GROUP_OK.json`` valid for this exact group?

    Valid: the file exists, its ``run_ids`` equal ``[d.name for d in run_dirs]``,
    and its ``lc_manifest_sha256[d.name]`` equals both the current
    ``sha256(d/"lc_manifest.json")`` and ``DONE.json["lc_manifest_sha256"]``.
    """
    names = [Path(d).name for d in run_dirs]
    out = []
    for d in run_dirs:
        d = Path(d)
        try:
            ok = json.loads((d / "GROUP_OK.json").read_text(encoding="utf-8"))
            done = json.loads((d / "DONE.json").read_text(encoding="utf-8"))
            current = sha256_file(d / "lc_manifest.json")
            out.append(ok.get("run_ids") == names
                       and ok.get("lc_manifest_sha256", {}).get(d.name) == current
                       and done.get("lc_manifest_sha256") == current)
        except (OSError, ValueError, AttributeError):
            out.append(False)
    return out


# ----------------------------------------------------------- amendments (P13)
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


def _split_row(line: str) -> List[str]:
    """Cells of a markdown table row; ``\\|`` is a literal pipe inside a cell."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.replace("\\|", "|").strip() for c in _CELL_SPLIT.split(s)]


def parse_amendment_rows(protocol_path: Path | None = None) -> List[Dict[str, str]]:
    """Rows of the protocol section 13 amendments table, keyed by ``AMENDMENT_COLUMNS``.

    Reads strictly between the line starting ``## 13.`` and the line starting
    ``## 14.``; finds the markdown table whose header cells (stripped, lower-cased)
    equal ``AMENDMENT_COLUMNS``; skips its separator row. Cells are split on
    unescaped ``|`` and ``\\|`` is unescaped (plan decision 4b). Raises
    ``ValueError`` if the section, the table or its header is missing, if the
    table appears twice, or if a row does not have exactly 7 cells.
    ``protocol_path=None`` reads ``PROTOCOL`` at call time.
    """
    path = Path(PROTOCOL if protocol_path is None else protocol_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("## 13.")), None)
    if start is None:
        raise ValueError(f"{path}: no '## 13.' section")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## 14.")),
               None)
    if end is None:
        raise ValueError(f"{path}: no '## 14.' section after '## 13.'")
    span = lines[start + 1:end]

    # Group consecutive table lines.
    tables: List[List[str]] = []
    cur: List[str] = []
    for line in span:
        if line.lstrip().startswith("|"):
            cur.append(line)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)

    found = [t for t in tables
             if tuple(c.lower() for c in _split_row(t[0])) == AMENDMENT_COLUMNS]
    if not found:
        raise ValueError(f"{path}: no amendments table with header {AMENDMENT_COLUMNS} in section 13")
    if len(found) > 1:
        raise ValueError(f"{path}: {len(found)} amendments tables in section 13")
    table = found[0]
    body = table[1:]
    if not body or not all(_SEPARATOR_CELL.match(c) for c in _split_row(body[0])):
        raise ValueError(f"{path}: amendments table has no separator row")
    rows: List[Dict[str, str]] = []
    for line in body[1:]:
        cells = _split_row(line)
        if len(cells) != len(AMENDMENT_COLUMNS):
            raise ValueError(f"{path}: amendments row has {len(cells)} cells, "
                             f"need {len(AMENDMENT_COLUMNS)}: {line!r}")
        rows.append(dict(zip(AMENDMENT_COLUMNS, cells)))
    return rows
