"""Feature tests (design §3, plan item 6). No model download: a tiny random
EsmModel and a tokenizer built from a temporary vocab file are injected."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from go_function import features as F
from go_function.constants import EXIT_REFUSED, OWL_NOTHING
from go_function.normforms import load_normal_forms
from go_function.ontology import parse_obo
from go_function.runtime import FatalError

ROOT = Path(__file__).resolve().parents[2]
ESM_TOKENS = (["<cls>", "<pad>", "<eos>", "<unk>"] + list("LAGVSERTIDPKQNFYMHWCXBUZO")
              + [".", "-", "<null_1>", "<mask>"])


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "go_function" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny_esm(tmp_path_factory):
    from transformers import EsmConfig, EsmModel, EsmTokenizer
    d = tmp_path_factory.mktemp("esm")
    (d / "vocab.txt").write_text("\n".join(ESM_TOKENS) + "\n")
    tok = EsmTokenizer(str(d / "vocab.txt"))
    torch.manual_seed(0)
    cfg = EsmConfig(vocab_size=len(ESM_TOKENS), hidden_size=16, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
                    pad_token_id=1, mask_token_id=len(ESM_TOKENS) - 1,
                    position_embedding_type="rotary", token_dropout=False)
    model = EsmModel(cfg, add_pooling_layer=False).eval()
    return model, tok


# ------------------------------------------------------------------ InterPro
def test_interpro_csr_fixture_dataframe():
    vocab = ["IPR1", "IPR2", "IPR3"]
    df = pd.DataFrame({"proteins": ["P1", "P2", "P3"],
                       "interpros": [["IPR2", "IPR1", "IPR2"], [], ["IPR9", "IPR3"]]})
    m, unknown = F.interpro_csr(df["interpros"].tolist(), {g: i for i, g in enumerate(vocab)})
    assert m.dtype == np.uint8 and m.shape == (3, 3)
    np.testing.assert_array_equal(m.toarray(), [[1, 1, 0], [0, 0, 0], [0, 0, 1]])
    assert unknown == 1


# ------------------------------------------------------------------ ESM helpers
def test_collect_sequences_dedup_and_conflict():
    names, seqs = F.collect_sequences([(["B", "A"], ["MK", "LL"]), (["A", "C"], ["LL", "V"])])
    assert names == ["A", "B", "C"] and seqs == ["LL", "MK", "V"]
    with pytest.raises(FatalError):
        F.collect_sequences([(["A"], ["LL"]), (["A"], ["LV"])])


def test_truncate_and_batches():
    seqs, n = F.truncate(["M" * 1030, "MK", "A" * 1022], 1022)
    assert n == 1 and [len(s) for s in seqs] == [1022, 2, 1022]
    batches = F.length_batches([5, 1, 3, 3, 10], budget=16)
    assert [i for b in batches for i in b] == [1, 2, 3, 0, 4]       # sorted by length, stable
    for b in batches:
        assert len(b) * (max([5, 1, 3, 3, 10][i] for i in b) + 2) <= 16


def test_pooling_excludes_bos_eos_pad(tiny_esm):
    model, tok = tiny_esm
    seqs = ["MKVLA", "MK", "GGGGGGGGGG"]
    got = F.embed_batch(model, tok, seqs, "cpu")
    for i, s in enumerate(seqs):
        enc = tok([s], return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[0]
        assert enc["input_ids"][0, 0] == tok.cls_token_id and enc["input_ids"][0, -1] == tok.eos_token_id
        manual = h[1:len(s) + 1].mean(0).numpy()                    # residues only
        np.testing.assert_allclose(got[i], manual, atol=1e-5)
        assert not np.allclose(got[i], h.mean(0).numpy(), atol=1e-4)  # BOS/EOS would change it


def test_truncation_applies_before_tokenisation(tiny_esm):
    model, tok = tiny_esm
    long = "MKVLAGG" * 10                                            # 70 residues
    seqs, n = F.truncate([long], 30)
    v = F.embed_batch(model, tok, seqs, "cpu")[0]
    enc = tok([long[:30]], return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    assert n == 1 and enc["input_ids"].shape[1] == 32
    np.testing.assert_allclose(v, h[1:31].mean(0).numpy(), atol=1e-5)


def test_embed_all_oom_halving(tiny_esm):
    model, tok = tiny_esm
    seqs = ["MKV" * k for k in range(1, 9)]
    calls = []

    def flaky(m, t, batch, device):
        tokens = len(batch) * (max(len(s) for s in batch) + 2)
        calls.append(tokens)
        if tokens > 40:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return F.embed_batch(m, t, batch, device)

    out, info = F.embed_all(model, tok, seqs, "cpu", budget=160, min_budget=20, embed=flaky)
    assert info["n_oom_halvings"] >= 1 and info["final_token_budget"] <= 40
    ref = np.vstack([F.embed_batch(model, tok, [s], "cpu") for s in seqs])
    np.testing.assert_allclose(out, ref, atol=1e-5)
    with pytest.raises(FatalError):
        F.embed_all(model, tok, seqs, "cpu", budget=160, min_budget=80, embed=flaky)


def test_cross_check_pass_and_fail(tiny_esm):
    model, tok = tiny_esm
    names = [f"P{i}" for i in range(10)]
    seqs = ["MKV" * (i + 1) for i in range(10)]
    stored, _ = F.embed_all(model, tok, seqs, "cpu", budget=200)
    res = F.cross_check(model, tok, seqs, names, stored, "cpu", n=4)
    assert res["n"] == 4 and res["min_cosine"] > 0.9999
    bad = stored.copy()
    bad[F.cross_check_ids(names, 4)[0]] *= -1
    with pytest.raises(FatalError):
        F.cross_check(model, tok, seqs, names, bad, "cpu", n=4)


def test_esm_view_missing_is_fatal():
    emb = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.testing.assert_array_equal(F.esm_view(emb, ["A", "B", "C"], ["C", "A"]), emb[[2, 0]])
    with pytest.raises(FatalError):
        F.esm_view(emb, ["A", "B", "C"], ["D"])


def test_embed_esm_refuses_cpu_without_limit():
    mod = _script("embed_esm")
    assert mod.main(["--device", "cpu"]) == EXIT_REFUSED


# ------------------------------------------------------------------ term text
class FakeEncoder:
    def __init__(self):
        self.kwargs = None

    def encode(self, texts, **kwargs):
        self.kwargs = kwargs
        v = np.asarray([[len(t), 1.0] for t in texts], dtype=np.float64)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _norm_classes(mini_norm, vocab):
    nf = load_normal_forms(mini_norm, {g: i for i, g in enumerate(vocab)},
                           expected_unparsed=1, expected_nothing=1)
    return nf.class_ids()


def test_obsolete_class_takes_its_own_stanza_text(mini_obo, tmp_path):
    """Amendment A3: name with one leading 'obsolete ' stripped, definition as written."""
    ont = parse_obo(mini_obo)
    st = {}
    ids, texts, _ = F.term_text_table(ont, ["GO:0000030", "GO:0000010"], stats=st)
    assert texts[ids.index("GO:0000030")] == "process." and st["n_obsolete_text"] == 1
    p = tmp_path / "o.obo"
    p.write_text("[Term]\nid: GO:0000001\nname: obsolete obsolete thing\n"
                 "namespace: molecular_function\n"
                 'def: "OBSOLETE. A thing." [GOC:x]\nalt_id: GO:0000002\nis_obsolete: true\n')
    o = parse_obo(p)
    assert F.obsolete_term_text(o, "GO:0000001") == "obsolete thing. OBSOLETE. A thing."
    assert F.obsolete_term_text(o, "GO:0000002") == "obsolete thing. OBSOLETE. A thing."
    assert F.obsolete_term_text(o, "GO:0000009") is None


@pytest.mark.slow
@pytest.mark.needs_dgz
def test_all_728_obsolete_go_norm_classes_get_text():
    """Review finding 5 / A3: every go.norm class obsolete in the pinned go.obo
    resolves to non-empty own-stanza text; none is left without text (no free vector)."""
    ET = _script("embed_terms")
    ont, ids = ET.class_ids()
    obsolete = sorted(g for g in ids if g != OWL_NOTHING and ont.primary(g) is None)
    assert len(obsolete) == 728
    st = {}
    out_ids, texts, skipped = F.term_text_table(ont, ids, stats=st)
    assert st["n_obsolete_text"] == 728 and skipped == 1
    text = dict(zip(out_ids, texts))
    assert set(obsolete) <= set(text) and len(out_ids) == len(ids) - 1
    for g in obsolete:
        assert text[g] and text[g] != "." and not text[g].startswith(F.OBSOLETE_PREFIX)
        assert text[g] == F.obsolete_term_text(ont, g)


def test_term_text_skips_nothing_and_fails_on_missing(mini_obo, mini_norm):
    ont = parse_obo(mini_obo)
    classes = _norm_classes(mini_norm, ["GO:0000010", "GO:0000099"])
    assert OWL_NOTHING in classes
    with pytest.raises(FatalError, match="GO:0000040"):
        F.term_text_table(ont, classes)                 # GO:0000040-44 are not in mini.obo
    present = [c for c in classes if c == OWL_NOTHING or ont.primary(c) is not None]
    ids, texts, skipped = F.term_text_table(ont, present)
    assert skipped == 1 and OWL_NOTHING not in ids
    assert texts[ids.index("GO:0000010")] == "process A. Process A."
    assert texts[ids.index("GO:0000099")] == "process B. Process B."   # alt id -> primary text
    enc = FakeEncoder()
    vec = F.encode_texts(enc, texts)
    assert vec.dtype == np.float32 and vec.shape == (len(ids), 2)
    assert enc.kwargs["normalize_embeddings"] is True and enc.kwargs["batch_size"] == 256
