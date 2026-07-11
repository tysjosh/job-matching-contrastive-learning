#!/usr/bin/env python3
"""
Confirm that the unfrozen path (freeze_text_encoder=False) actually lets
gradients reach the SentenceTransformer encoder.

Checks:
  1. Old cached path (torch.no_grad + detach) -> encoder grads are None.
  2. New tokenize+forward path -> encoder grads are non-None (fine-tuning works).

Run: python3 scripts/confirm_encoder_finetune.py
"""
import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-mpnet-base-v2"
texts = ["software engineer python react", "warehouse manager logistics"]


def any_encoder_grad(st):
    return any(p.grad is not None and p.grad.abs().sum() > 0
               for p in st.parameters())


def main():
    st = SentenceTransformer(MODEL, device='cpu')
    for p in st.parameters():
        p.requires_grad = True

    # ── Old behavior: no_grad + detach (cached path) ──
    st.zero_grad()
    with torch.no_grad():
        emb = st.encode(texts, convert_to_tensor=True, normalize_embeddings=False)
    emb = emb.detach().clone().requires_grad_(False)
    # try to make a loss and backward — will fail / no grad to encoder
    proj = torch.nn.Linear(emb.shape[1], 128)
    out = torch.nn.functional.normalize(proj(emb), dim=-1)
    loss = (out[0] * out[1]).sum()
    try:
        loss.backward()
    except Exception as e:
        print(f"[cached path] backward could not reach encoder (expected): {type(e).__name__}")
    print(f"[cached path] encoder has grads: {any_encoder_grad(st)}  (expected: False)")

    # ── New behavior: tokenize + forward WITH grad ──
    st.zero_grad()
    st.train()
    features = st.tokenize(texts)
    out = st(features)
    text_emb = out['sentence_embedding']            # requires grad
    proj2 = torch.nn.Linear(text_emb.shape[1], 128)
    z = torch.nn.functional.normalize(proj2(text_emb), dim=-1)
    loss2 = -(z[0] * z[1]).sum()                     # arbitrary contrastive-ish loss
    loss2.backward()
    has = any_encoder_grad(st)
    print(f"[trainable path] text_emb.requires_grad: {text_emb.requires_grad}")
    print(f"[trainable path] encoder has grads: {has}  (expected: True)")

    # Report a couple of grad norms as evidence
    grad_params = [(n, p.grad.norm().item()) for n, p in st.named_parameters()
                   if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"[trainable path] #encoder tensors with grad: {len(grad_params)}")
    for n, g in grad_params[:3]:
        print(f"    {n}: grad_norm={g:.4e}")

    print("\nRESULT:", "PASS — encoder fine-tuning works" if has
          else "FAIL — encoder still not receiving gradients")


if __name__ == "__main__":
    main()
