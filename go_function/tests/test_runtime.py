from pathlib import Path

import pytest
import torch

from go_function import runtime
from go_function.pins import canonical_json
from go_function.runtime import FatalError, ROOT, repo_rel, resolve_device


def test_auto_never_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_mps_and_unknown_refused():
    with pytest.raises(ValueError):
        resolve_device("mps")
    with pytest.raises(ValueError):
        resolve_device("gpu")
    assert resolve_device("cpu") == "cpu"


def test_cuda_refused_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        resolve_device("cuda")


def test_repo_rel():
    assert repo_rel(ROOT / "go_function" / "x.json") == "go_function/x.json"
    assert repo_rel("preprocess/go_function/bp") == "preprocess/go_function/bp"
    with pytest.raises(ValueError):
        repo_rel(Path("/tmp/outside.json"))
    assert runtime.repo_path("go_function") == ROOT / "go_function"


def test_canonical_json_sorted_and_no_abs_path():
    data = canonical_json({"b": 1, "a": [repo_rel(ROOT / "results" / "x")]})
    assert data == b'{\n  "a": [\n    "results/x"\n  ],\n  "b": 1\n}\n'
    assert b"/Users/" not in data
    with pytest.raises(FatalError):
        canonical_json({"p": str(ROOT / "results")})
    with pytest.raises(FatalError):
        canonical_json({"p": "/Users/someone/file"})


def test_seed_everything_repeatable():
    runtime.seed_everything(7)
    a = torch.rand(3)
    runtime.seed_everything(7)
    assert torch.equal(a, torch.rand(3))
