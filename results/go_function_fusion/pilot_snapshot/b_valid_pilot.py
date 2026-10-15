"""Validation-only fusion pilot on DGZ released checkpoints (READ-ONLY w.r.t. the repo).

For one aspect and one released checkpoint (mlp | deepgozero):
  forward on DGZ valid InterPro -> logits L, probs P
  B_raw    = P
  B_dgz    = DGZ-rule max-propagation of P (what DGZ releases)
  B_tp     = true-path (is_a+part_of, same namespace) max-propagation of P
  B_shuf   = B_tp with the TP graph applied to frequency-matched permuted columns (control)
  H        = identity-weighted kNN label transfer valid->train from swissprot_exp.sim
             (proxy for DiamondScore: no valid Diamond run exists), TP train labels
  convex   = (1-a) B_tp + a H, a in A
  zsum     = z_t(B_tp) + w z_t(H)  (per-term z over valid proteins)
  F_pa     = z_t(L_t) + z_t(mean parent logits)   (protocol F without the sibling term)
Metrics on valid proteins with >=1 TP label (root excluded):
  class-centric mean AUC over terms with 0 < n_pos < n, overall and by n_train band;
  protein-centric Fmax (convex/prob-valued variants only).
No test data are read.
"""
import gc
import json
import sys
import time
import collections
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

R = Path('/Users/olukotunjosh/Downloads/CDCL')
sys.path.insert(0, str(R))
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from go_function.constants import ROOTS  # noqa: E402
from go_function.evaluate import propagate_max, tp_ancestor_pairs  # noqa: E402
from go_function.features import interpro_csr, read_interpro_vocab  # noqa: E402
from go_function.labels import build_vocab, load_labels, read_terms  # noqa: E402
from go_function.model import DgzCompatModel, dgz_propagation_pairs, predict_probs  # noqa: E402
from go_function.normforms import load_normal_forms  # noqa: E402
from go_function.ontology import DgzRule, ancestors, parents, parse_obo  # noqa: E402

torch.set_num_threads(6)
asp, kind = sys.argv[1], sys.argv[2]
D = R / 'dataset/go_function/dgz'
t0 = time.time()
log = lambda *a: print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)  # noqa: E731

terms = read_terms(D / asp / 'terms.pkl')
lab = load_labels(R / f'preprocess/go_function/{asp}/labels_TP.npz')
assert lab['vocab'] == list(terms), 'labels vocab != terms.pkl order'
va = pd.read_pickle(D / asp / 'valid_data.pkl')
assert list(va.proteins) == lab['proteins_valid']
ipr = read_interpro_vocab(D / asp / 'interpros.pkl')
x, _ = interpro_csr(va.interpros.tolist(), {g: i for i, g in enumerate(ipr)})
nf = None
if kind == 'deepgozero':
    nf = load_normal_forms(D / 'go.norm', {g: i for i, g in enumerate(terms)})
model = DgzCompatModel.load(D / asp / f'{kind}.th', kind=kind, n_in=len(ipr), n_terms=len(terms), nf=nf)
L = predict_probs(model, x, sigmoid=False)
del model, nf
gc.collect()
P = (1.0 / (1.0 + np.exp(-L))).astype(np.float32)
log('forward done', L.shape)

Yv = sp.csr_matrix(lab['valid']).astype(bool)
Ytr = sp.csr_matrix(lab['train']).astype(np.float32)
ntr = np.asarray(Ytr.sum(0)).ravel()
root = terms.index(ROOTS[asp]) if ROOTS[asp] in terms else -1
rows = np.where(np.asarray(Yv.sum(1)).ravel() > 0)[0]
cols_keep = np.ones(len(terms), bool)
if root >= 0:
    cols_keep[root] = False
Yd = Yv[rows].toarray()
npos = Yd.sum(0)
eval_cols = np.where(cols_keep & (npos > 0) & (npos < len(rows)))[0]
log('valid rows', len(rows), 'eval terms', len(eval_cols))

BANDS = [(0, 0), (1, 10), (11, 30), (31, 100), (101, 1000), (1001, 10 ** 9)]


def class_auc(S):
    S = S[rows][:, eval_cols]
    Y = Yd[:, eval_cols]
    out = np.empty(len(eval_cols))
    for a in range(0, len(eval_cols), 2000):
        rk = rankdata(S[:, a:a + 2000], axis=0)
        y = Y[:, a:a + 2000]
        npp = y.sum(0)
        nn = y.shape[0] - npp
        out[a:a + 2000] = ((rk * y).sum(0) - npp * (npp + 1) / 2) / (npp * nn)
    res = {'mean': float(out.mean()), 'n': int(len(out))}
    for lo, hi in BANDS:
        m = (ntr[eval_cols] >= lo) & (ntr[eval_cols] <= hi)
        if m.any():
            res[f'{lo}-{hi}'] = [round(float(out[m].mean()), 4), int(m.sum())]
    return res, out


def fmax(S):
    S = S[rows][:, cols_keep]
    Y = Yd[:, cols_keep]
    best = (0.0, 0.0)
    ny = Y.sum(1)
    for t in np.arange(0.01, 1.0, 0.01):
        pr = S >= t
        tp = (pr & Y).sum(1)
        npr = pr.sum(1)
        has = npr > 0
        if not has.any():
            continue
        p = (tp[has] / npr[has]).mean()
        r = (tp / ny).mean()
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        if f > best[0]:
            best = (float(f), float(t))
    return {'fmax': round(best[0], 4), 'tmax': round(best[1], 2)}


def zcol(S):
    m = S[rows].mean(0, keepdims=True)
    s = S[rows].std(0, keepdims=True)
    s[s == 0] = 1
    return ((S - m) / s).astype(np.float32)


results = {'aspect': asp, 'model': kind, 'n_valid_rows': int(len(rows)), 'variants': {}}
per_term = {}


def record(name, S, do_fmax=False):
    r, v = class_auc(S)
    if do_fmax:
        r.update(fmax(S))
    results['variants'][name] = r
    per_term[name] = v
    log(name, json.dumps(r))


record('B_raw', P, True)
# ---- propagation variants
ont = parse_obo(D / 'go.obo')
vocab = build_vocab(terms, ont)
anc = ancestors(ont)
src, dst = tp_ancestor_pairs(list(range(len(terms))), vocab,
                             {**{t: a for t, a in anc.items()}, None: frozenset()})
B_tp = propagate_max(torch.as_tensor(P), src, dst).numpy()
record('B_tp', B_tp, True)
dsrc, ddst = dgz_propagation_pairs(terms, DgzRule(D / 'go.obo'))
log('pairs TP', len(src), 'DGZ', len(dsrc))
B_dgz = propagate_max(torch.as_tensor(P), dsrc, ddst).numpy()
record('B_dgz', B_dgz, True)
del B_dgz
gc.collect()
# shuffled-graph control: permute column identities within n_train deciles
rng = np.random.default_rng(20261013)
dec = np.searchsorted(np.quantile(ntr, np.linspace(0, 1, 11)[1:-1]), ntr)
perm = np.arange(len(terms))
for b in np.unique(dec):
    idx = np.where(dec == b)[0]
    perm[idx] = rng.permutation(idx)
B_shuf = propagate_max(torch.as_tensor(P), perm[src], perm[dst]).numpy()
record('B_tp_shuffledgraph', B_shuf)
del B_shuf
gc.collect()
# ---- F_pa: z(own logit) + z(mean TP-parent logit) (vocab columns, root excluded)
par = parents(ont)
pcols = []
for c in range(len(terms)):
    pr = vocab.primary[c]
    ps = [j for p in (par.get(pr, ()) if pr else ()) for j in vocab.primary_to_cols.get(p, ()) if j != root]
    pcols.append(ps)
PM = np.zeros_like(L)
has_pa = np.array([len(p) > 0 for p in pcols])
for c in np.where(has_pa)[0]:
    PM[:, c] = L[:, pcols[c]].mean(1)
F = zcol(L)
F[:, has_pa] += zcol(PM)[:, has_pa]
record('F_pa', F)
# shuffled-parent control for F_pa
PMs = np.zeros_like(L)
for c in np.where(has_pa)[0]:
    PMs[:, c] = L[:, perm[pcols[c]]].mean(1)
Fs = zcol(L)
Fs[:, has_pa] += zcol(PMs)[:, has_pa]
record('F_pa_shuffled', Fs)
del F, Fs, PM, PMs
gc.collect()
# ---- homology prior from swissprot_exp.sim (pident-weighted kNN over train)
tr_names = lab['proteins_train']
ti = {n: i for i, n in enumerate(tr_names)}
vi = {n: i for i, n in enumerate(lab['proteins_valid'])}
r_, c_, w_ = [], [], []
for line in open('/tmp/gofunc_fusion/swissprot_exp.sim'):
    a, b, p = line.rstrip('\n').split('\t')
    if a in vi and b in ti:
        r_.append(vi[a]); c_.append(ti[b]); w_.append(float(p))
W = sp.csr_matrix((w_, (r_, c_)), shape=(len(vi), len(ti)))
tot = np.asarray(W.sum(1)).ravel()
W = sp.diags(np.divide(1, tot, out=np.zeros_like(tot), where=tot > 0)) @ W
H = np.asarray((W @ Ytr).todense(), dtype=np.float32)
results['valid_with_sim_hit'] = int((tot[rows] > 0).sum())
log('H built; valid eval rows with a train hit', results['valid_with_sim_hit'])
record('H_sim', H, True)
for a in (0.2, 0.4, 0.5, 0.6):
    record(f'convex_a{a}', (1 - a) * B_tp + a * H, True)
zB, zH = zcol(B_tp), zcol(H)
for w in (0.25, 0.5, 1.0):
    record(f'zsum_w{w}', zB + w * zH)
# random-hit control: H from a shuffled train-label row assignment
Hs = np.asarray((W @ Ytr[rng.permutation(Ytr.shape[0])]).todense(), dtype=np.float32)
record('convex_a0.4_shuffledhits', 0.6 * B_tp + 0.4 * Hs)
out = Path(f'/tmp/gofunc_fusion/pilot_{asp}_{kind}.json')
out.write_text(json.dumps(results, indent=1))
np.savez_compressed(f'/tmp/gofunc_fusion/pilot_{asp}_{kind}_perterm.npz', cols=eval_cols,
                    ntr=ntr[eval_cols], **per_term)
log('wrote', out)
