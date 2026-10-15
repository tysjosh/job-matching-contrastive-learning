"""Validation-only structural-prior fusion variants (class-centric AUC), released DGZ checkpoints.

score_t(p) = z(L_t) + w * z(prior_t(p)) (+ ...), z over valid eval rows, float64, column chunks.
Priors (all from the same model's logits L, no labels except train counts):
  pa_isa    mean logit over is_a parents (vocab columns, root excluded)
  pa_tp     mean logit over is_a+part_of parents                       (= protocol F without siblings)
  pa_any    mean logit over parents by ANY relation, same namespace     (untyped/DGZ-like conflation)
  anc_tp    mean logit over all TP ancestors
  sib       mean logit over is_a siblings with >= 10 TP train positives (subtracted, protocol F)
  ch_max    max logit over direct TP children
  reg       mean logit over regulates/pos/neg-regulates targets (same namespace)
  H         identity-weighted kNN transfer (swissprot_exp.sim), homology prior
Controls: pa_tp with frequency-matched permuted parents; H from permuted train rows.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

R = Path('/Users/olukotunjosh/Downloads/CDCL')
sys.path.insert(0, str(R))
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from go_function.constants import ROOTS  # noqa: E402
from go_function.features import interpro_csr, read_interpro_vocab  # noqa: E402
from go_function.labels import build_vocab, load_labels, read_terms  # noqa: E402
from go_function.model import DgzCompatModel, predict_probs  # noqa: E402
from go_function.normforms import load_normal_forms  # noqa: E402
from go_function.ontology import ancestors, parse_obo, _isa_children  # noqa: E402

torch.set_num_threads(6)
asp, kind = sys.argv[1], sys.argv[2]
D = R / 'dataset/go_function/dgz'
t0 = time.time()
log = lambda *a: print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)  # noqa: E731
terms = read_terms(D / asp / 'terms.pkl')
lab = load_labels(R / f'preprocess/go_function/{asp}/labels_TP.npz')
assert lab['vocab'] == list(terms)
va = pd.read_pickle(D / asp / 'valid_data.pkl')
ipr = read_interpro_vocab(D / asp / 'interpros.pkl')
x, _ = interpro_csr(va.interpros.tolist(), {g: i for i, g in enumerate(ipr)})
nf = load_normal_forms(D / 'go.norm', {g: i for i, g in enumerate(terms)}) if kind == 'deepgozero' else None
model = DgzCompatModel.load(D / asp / f'{kind}.th', kind=kind, n_in=len(ipr), n_terms=len(terms), nf=nf)
Lall = predict_probs(model, x, sigmoid=False)
del model, nf
gc.collect()
Yv = sp.csr_matrix(lab['valid']).astype(bool)
Ytr = sp.csr_matrix(lab['train']).astype(np.float32)
ntr = np.asarray(Ytr.sum(0)).ravel()
rows = np.where(np.asarray(Yv.sum(1)).ravel() > 0)[0]
L = Lall[rows].astype(np.float64)
del Lall
Yd = Yv[rows].toarray()
root = terms.index(ROOTS[asp]) if ROOTS[asp] in terms else -1
npos = Yd.sum(0)
eval_cols = np.where((np.arange(len(terms)) != root) & (npos > 0) & (npos < len(rows)))[0]
log('rows', len(rows), 'eval terms', len(eval_cols))

ont = parse_obo(D / 'go.obo')
vocab = build_vocab(terms, ont)
anc = ancestors(ont)
REG = {'regulates', 'positively_regulates', 'negatively_regulates'}


def cols_of(ids):
    return sorted({j for g in ids for j in vocab.primary_to_cols.get(g, ()) if j != root})


ch = _isa_children(ont)
tp_children = {}
for tid, t in ont.terms.items():
    for r, p in t.edges:
        if r in ('is_a', 'part_of') and ont.terms[p].namespace == t.namespace:
            tp_children.setdefault(p, set()).add(tid)
nb = {k: [] for k in ('pa_isa', 'pa_tp', 'pa_any', 'anc_tp', 'sib', 'ch', 'reg')}
for c in range(len(terms)):
    g = vocab.primary[c]
    if g is None:
        for k in nb:
            nb[k].append([])
        continue
    t = ont.terms[g]
    same = lambda p: ont.terms[p].namespace == t.namespace  # noqa: E731
    nb['pa_isa'].append(cols_of(p for r, p in t.edges if r == 'is_a' and same(p)))
    nb['pa_tp'].append(cols_of(p for r, p in t.edges if r in ('is_a', 'part_of') and same(p)))
    nb['pa_any'].append(cols_of(p for r, p in t.edges if same(p)))
    nb['anc_tp'].append(cols_of(anc[g] - {g}))
    sib = set()
    for r, p in t.edges:
        if r == 'is_a':
            sib |= ch[p]
    sib -= {g} | set(anc[g])
    nb['sib'].append([j for j in cols_of(sib) if ntr[j] >= 10])
    nb['ch'].append(cols_of(tp_children.get(g, ())))
    nb['reg'].append(cols_of(p for r, p in t.edges if r in REG and same(p)))
for k, v in nb.items():
    log(k, 'eval terms with >=1:', int(sum(len(v[c]) > 0 for c in eval_cols)))

rng = np.random.default_rng(20261013)
dec = np.searchsorted(np.quantile(ntr, np.linspace(0, 1, 11)[1:-1]), ntr)
perm = np.arange(len(terms))
for b in np.unique(dec):
    idx = np.where(dec == b)[0]
    perm[idx] = rng.permutation(idx)
nb['pa_tp_shuf'] = [[int(perm[j]) for j in v] for v in nb['pa_tp']]

# homology prior on eval rows
tr_names = lab['proteins_train']
ti = {n: i for i, n in enumerate(tr_names)}
vi = {n: i for i, n in enumerate(np.asarray(lab['proteins_valid'])[rows])}
r_, c_, w_ = [], [], []
for line in open('/tmp/gofunc_fusion/swissprot_exp.sim'):
    a, b, p = line.rstrip('\n').split('\t')
    if a in vi and b in ti:
        r_.append(vi[a]); c_.append(ti[b]); w_.append(float(p))
W = sp.csr_matrix((w_, (r_, c_)), shape=(len(vi), len(ti)))
tot = np.asarray(W.sum(1)).ravel()
W = sp.csr_matrix(sp.diags(np.divide(1, tot, out=np.zeros_like(tot), where=tot > 0)) @ W)


def z(M):
    m = M.mean(0, keepdims=True)
    s = M.std(0, keepdims=True)
    s[s == 0] = 1
    return (M - m) / s


def agg(key, cols, how='mean'):
    out = np.zeros((L.shape[0], len(cols)))
    has = np.zeros(len(cols), bool)
    for i, c in enumerate(cols):
        v = nb[key][c]
        if v:
            out[:, i] = L[:, v].max(1) if how == 'max' else L[:, v].mean(1)
            has[i] = True
    return z(out) * has


VARIANTS = {
    'own': lambda C, ctx: z(L[:, C]),
    'pa_isa': lambda C, ctx: z(L[:, C]) + agg('pa_isa', C),
    'pa_tp': lambda C, ctx: z(L[:, C]) + ctx['pa_tp'],
    'pa_tp_w0.5': lambda C, ctx: z(L[:, C]) + 0.5 * ctx['pa_tp'],
    'pa_tp_w2': lambda C, ctx: z(L[:, C]) + 2 * ctx['pa_tp'],
    'pa_any': lambda C, ctx: z(L[:, C]) + agg('pa_any', C),
    'anc_tp': lambda C, ctx: z(L[:, C]) + agg('anc_tp', C),
    'F_protocol': lambda C, ctx: z(L[:, C]) + ctx['pa_tp'] - agg('sib', C),
    'ch_max': lambda C, ctx: z(L[:, C]) + agg('ch', C, 'max'),
    'pa_tp+reg': lambda C, ctx: z(L[:, C]) + ctx['pa_tp'] + agg('reg', C),
    'pa_tp+H': lambda C, ctx: z(L[:, C]) + ctx['pa_tp'] + 0.5 * ctx['H'],
    'own+H': lambda C, ctx: z(L[:, C]) + 0.5 * ctx['H'],
    'pa_tp_shuffled': lambda C, ctx: z(L[:, C]) + agg('pa_tp_shuf', C),
    # scarcity-adaptive shrinkage to the parent: w_t = 2 tau / (tau + n_train(t))
    'pa_tp_adapt_tau10': lambda C, ctx: z(L[:, C]) + (20.0 / (10 + ntr[C])) * ctx['pa_tp'],
    'pa_tp_adapt_tau30': lambda C, ctx: z(L[:, C]) + (60.0 / (30 + ntr[C])) * ctx['pa_tp'],
    'pa_tp_adapt_tau100': lambda C, ctx: z(L[:, C]) + (200.0 / (100 + ntr[C])) * ctx['pa_tp'],
    'pa_tp_adapt_tau30+H': lambda C, ctx: z(L[:, C]) + (60.0 / (30 + ntr[C])) * ctx['pa_tp'] + 0.5 * ctx['H'],
}
auc = {k: np.empty(len(eval_cols)) for k in VARIANTS}
for a in range(0, len(eval_cols), 1500):
    C = eval_cols[a:a + 1500]
    ctx = {'pa_tp': agg('pa_tp', C), 'H': z(np.asarray((W @ Ytr[:, C]).todense(), dtype=np.float64))}
    y = Yd[:, C]
    npp = y.sum(0)
    nn = y.shape[0] - npp
    for k, f in VARIANTS.items():
        rk = rankdata(f(C, ctx), axis=0)
        auc[k][a:a + 1500] = ((rk * y).sum(0) - npp * (npp + 1) / 2) / (npp * nn)
    log('chunk', a)
BANDS = [(0, 0), (1, 10), (11, 30), (31, 100), (101, 1000), (1001, 10 ** 9)]
has_reg = np.array([len(nb['reg'][c]) > 0 for c in eval_cols])
res = {'aspect': asp, 'model': kind, 'n_rows': int(len(rows)), 'n_terms': int(len(eval_cols)),
       'n_terms_with_reg_edge': int(has_reg.sum()), 'variants': {}}
nt = ntr[eval_cols]
for k, v in auc.items():
    r = {'mean': round(float(v.mean()), 4)}
    for lo, hi in BANDS:
        m = (nt >= lo) & (nt <= hi)
        r[f'{lo}-{hi}'] = round(float(v[m].mean()), 4)
    r['reg_subset'] = round(float(v[has_reg].mean()), 4) if has_reg.any() else None
    d = v - auc['own']
    r['delta_vs_own'] = round(float(d.mean()), 4)
    r['share_terms_improved'] = round(float((d > 0).mean()), 3)
    res['variants'][k] = r
    log(k, json.dumps(r))
res['band_counts'] = {f'{lo}-{hi}': int(((nt >= lo) & (nt <= hi)).sum()) for lo, hi in BANDS}
Path(f'/tmp/gofunc_fusion/struct_{asp}_{kind}.json').write_text(json.dumps(res, indent=1))
np.savez_compressed(f'/tmp/gofunc_fusion/struct_{asp}_{kind}_perterm.npz', cols=eval_cols, ntr=nt, **auc)
