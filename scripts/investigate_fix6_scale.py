#!/usr/bin/env python3
"""
Investigate Fix 6: scale/gradient interaction between InfoNCE (L1) and the
ordinal margin terms in the ordinal contrastive loss.

Question: are the ordinal margins strong enough to influence training when
simply added to a temperature-scaled InfoNCE loss (loss = L1 + margin)?

We measure the GRADIENT NORM each component contributes to the embeddings at
realistic operating points, because parameter updates depend on gradients, not
raw loss values. InfoNCE logits are divided by temperature τ, so its gradient
is amplified by ~1/τ; the margin terms use raw cosine (no 1/τ), so their
gradient is O(1). We quantify the resulting imbalance and what β restores it.

Run: python3 scripts/investigate_fix6_scale.py
"""
import torch
import torch.nn.functional as F

TAU = 0.07          # temperature used in phase-1 configs
DIM = 128           # projection_dim
K_NEG = 7           # negatives per anchor (max_negatives_per_anchor)
M = 0.3             # ordinal_m2 margin


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def make_job_at(r, e_perp, cos_target):
    """Build a unit job vector with a target cosine similarity to r."""
    sin = (1.0 - cos_target ** 2) ** 0.5
    return unit(cos_target * r + sin * e_perp)


def infonce_loss(r, jobs_pos, jobs_neg):
    s_pos = (r * jobs_pos).sum() / TAU
    s_neg = torch.stack([(r * j).sum() / TAU for j in jobs_neg])
    logits = torch.cat([s_pos.view(1), s_neg])
    logits = logits - logits.max()
    return -(logits[0] - torch.logsumexp(logits, dim=0))


def ordinal_margin(r, s_by_level):
    """s_by_level: dict level->list of job unit vectors. Mirrors the fixed loss."""
    jobs, levels = [], []
    for lv, js in s_by_level.items():
        for j in js:
            jobs.append(j); levels.append(lv)
    J = torch.stack(jobs)
    sims = J @ r
    lv = torch.tensor(levels, dtype=torch.float32)
    diff = sims.unsqueeze(1) - sims.unsqueeze(0)
    li, lj = lv.unsqueeze(1), lv.unsqueeze(0)
    loss = torch.tensor(0.0)
    m_vs_nofit = ((li > lj) & (lj == 0)).float()
    if m_vs_nofit.sum() > 0:
        loss = loss + (F.relu(M - diff) * m_vs_nofit).sum() / m_vs_nofit.sum()
    m_good_pot = ((li == 2) & (lj == 1)).float()
    if m_good_pot.sum() > 0:
        loss = loss + (F.relu(M - diff) * m_good_pot).sum() / m_good_pot.sum()
    return loss


def grad_norm_wrt(loss, tensors):
    grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    total = 0.0
    for gg in grads:
        if gg is not None:
            total += gg.pow(2).sum().item()
    return total ** 0.5


def scenario(name, cos_good, cos_pot, cos_neg):
    torch.manual_seed(0)
    r = unit(torch.randn(DIM))
    # distinct perpendicular directions for each job
    perps = [unit(torch.randn(DIM)) for _ in range(3 + K_NEG)]

    good = make_job_at(r, perps[0], cos_good).clone().requires_grad_(True)
    pot = make_job_at(r, perps[1], cos_pot).clone().requires_grad_(True)
    negs = [make_job_at(r, perps[2 + i], cos_neg).clone().requires_grad_(True)
            for i in range(K_NEG)]

    params = [good, pot] + negs

    # L1: anchor r, positive = good, negatives = the no_fit jobs
    l1 = infonce_loss(r, good, negs)
    # Ordinal margins: good(2) > potential(1) > no_fit(0)
    om = ordinal_margin(r, {2: [good], 1: [pot], 0: negs})

    gn_l1 = grad_norm_wrt(l1, params)
    gn_om = grad_norm_wrt(om, params)

    print(f"\n=== {name} (cos good/pot/neg = {cos_good}/{cos_pot}/{cos_neg}) ===")
    print(f"  L1 value            : {l1.item():.4f}")
    print(f"  margin value        : {om.item():.4f}")
    print(f"  ||grad L1||         : {gn_l1:.4f}")
    print(f"  ||grad margin||     : {gn_om:.4f}")
    ratio = gn_om / gn_l1 if gn_l1 > 0 else float('inf')
    print(f"  grad ratio (om/L1)  : {ratio:.4f}   -> margin is {1/ratio:.1f}x weaker" if ratio>0 else "  margin inactive")
    # What beta makes them comparable?
    if gn_om > 0:
        print(f"  beta for parity     : {gn_l1/gn_om:.2f}  (≈1/τ={1/TAU:.1f} expected)")
    return gn_l1, gn_om


def main():
    print("Fix 6 investigation: gradient-magnitude balance of L1 vs ordinal margin")
    print(f"τ={TAU}, dim={DIM}, K_neg={K_NEG}, margin m={M}")

    # Early training: embeddings barely separated (sims near 0)
    scenario("EARLY TRAINING", cos_good=0.05, cos_pot=0.0, cos_neg=-0.02)
    # Mid training: partial separation (approx v7 tier means good.57/pot.40/no.26)
    scenario("MID TRAINING (v7-like)", cos_good=0.57, cos_pot=0.40, cos_neg=0.26)
    # Violation: good ranked BELOW potential (margin strongly active)
    scenario("ORDINAL VIOLATION", cos_good=0.30, cos_pot=0.55, cos_neg=0.20)
    # Well-separated & ordered (margins mostly satisfied)
    scenario("WELL SEPARATED", cos_good=0.80, cos_pot=0.45, cos_neg=0.05)


if __name__ == "__main__":
    main()
