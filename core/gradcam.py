"""
Grad-ECLIP — Gradient-based visual Explanation for CLIP
Zhao, Chenyang et al. "Gradient-based Visual Explanation for Transformer-based CLIP."
ICML 2024.  https://github.com/Cyang-Zhao/Grad-Eclip

=== Correct algorithm (image encoder branch) ===

Given image I and text T:  S = cos(f_I, f_T)

For a target attention layer L (H heads, each head dimension d_h):

  1.  Forward — capture per-head value vectors v_h  (H, N, d_h)
               attention weights A_h   (H, N, N)  post-softmax
               per-head attention output o_h = A_h @ v_h  (H, N, d_h)
      → retain_grad() on o_h so the backward pass fills o_h.grad

  2.  Backward — dS/d_o_h is now available.

  3.  Channel weight (per head):
        w_c_h = ReLU( dS/d_o_h[:, 0, :] )        shape: (d_h,)
        The CLS token is position 0.  Its gradient tells which feature
        channels of the aggregated CLS representation push S upward.
        NOTE: this is NOT the same as dS/dv, which would propagate
        through the full A @ v Jacobian across all query positions.

  4.  Spatial weight (per head):
        base: a_cls_h = A_h[0, 1:]    (CLS-to-patches row)  shape: (n_p,)

        Full Grad-ECLIP variant adds key-query similarity:
          k_sim_h = sigmoid( q_h[0] @ k_h[1:].T / sqrt(d_h) )
          w_i_h   = loosen( a_cls_h * k_sim_h )
        where loosen(x) = |x| / sum|x|   (L1-normalise absolute values)

        eclip-wo-ksim variant:
          w_i_h = loosen( a_cls_h )

        Loosening avoids the sparsity of raw softmax, letting multiple
        patches contribute rather than just the top-1.

  5.  Per-head saliency:
        M_h = w_i_h * ( v_h[1:] @ w_c_h )         shape: (n_p,)

  6.  Aggregate over heads and ReLU:
        E = ReLU( sum_h M_h )                       shape: (n_p,)

  7.  Reshape to sqrt(n_p) x sqrt(n_p), bicubic-upsample to image size,
      normalise to [0, 1].
"""

from __future__ import annotations

import io
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image


# ── internal store ────────────────────────────────────────────────────────────

class _AttentionStore:
    """Captures per-head tensors from a single CLIPAttention forward pass."""

    def __init__(self):
        self.v:        torch.Tensor | None = None   # (B, H, N, d) detached
        self.attn:     torch.Tensor | None = None   # (B, H, N, N) detached
        self.q_h:      torch.Tensor | None = None   # (B, H, N, d) detached
        self.k_h:      torch.Tensor | None = None   # (B, H, N, d) detached
        self.attn_out: torch.Tensor | None = None   # (B, H, N, d) retains grad


def _patch_attention(attn_module, store: _AttentionStore):
    """
    Temporarily replaces CLIPAttention.forward.
    Captures per-head v, A, q, k and the per-head attention output
    (before the output projection) with retain_grad() so we can read
    dS/d_o after calling score.backward().
    Returns the original forward for restoration.
    """
    original_forward = attn_module.forward

    def patched_forward(hidden_states, attention_mask=None,
                        causal_attention_mask=None, output_attentions=False,
                        **kwargs):
        bsz, seq, embed_dim = hidden_states.size()
        H     = attn_module.num_heads
        d     = attn_module.head_dim
        scale = attn_module.scale

        q = attn_module.q_proj(hidden_states)
        k = attn_module.k_proj(hidden_states)
        v = attn_module.v_proj(hidden_states)

        def _split(t):
            return t.view(bsz, seq, H, d).transpose(1, 2)   # (B, H, N, d)

        q_h = _split(q)
        k_h = _split(k)
        v_h = _split(v)

        logits = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
        if attention_mask is not None:
            logits = logits + attention_mask
        if causal_attention_mask is not None:
            logits = logits + causal_attention_mask
        A = F.softmax(logits, dim=-1)   # (B, H, N, N)

        # Per-head attention output: o = A @ v  →  (B, H, N, d)
        o = torch.matmul(A, v_h)
        o.retain_grad()   # <── allows reading dS/d_o after backward()

        store.v        = v_h.detach()
        store.attn     = A.detach()
        store.q_h      = q_h.detach()
        store.k_h      = k_h.detach()
        store.attn_out = o    # keep reference so .grad is accessible

        merged = o.transpose(1, 2).contiguous().view(bsz, seq, embed_dim)
        return attn_module.out_proj(merged), None

    attn_module.forward = patched_forward
    return original_forward


def _restore_attention(attn_module, original_forward):
    attn_module.forward = original_forward


# ── main Grad-ECLIP function ──────────────────────────────────────────────────

def grad_eclip(
    pil_img:   Image.Image,
    text:      str,
    model,
    processor,
    device:    str,
    layer_idx: int   = -1,
    alpha:     float = 0.55,
    colormap:  str   = "inferno",
    with_ksim: bool  = True,
) -> tuple[Image.Image, np.ndarray, float]:
    """
    Compute a Grad-ECLIP explanation heatmap for an image–text pair.

    Parameters
    ----------
    pil_img    : input PIL image
    text       : text query
    model      : HuggingFace CLIPModel (will be put in eval mode)
    processor  : HuggingFace CLIPProcessor
    layer_idx  : ViT encoder layer to target; -1 = last (most semantic),
                 earlier layers give more local / low-level explanations
    alpha      : heatmap overlay blend weight [0.1 .. 0.9]
    colormap   : matplotlib colormap name for the heatmap
    with_ksim  : True  → full Grad-ECLIP (spatial weight includes key-query sim)
                 False → eclip-wo-ksim variant (loosened attention only)

    Returns
    -------
    overlay_pil : PIL.Image — original frame blended with the coloured heatmap
    cam_np      : np.ndarray (H, W) float32 in [0, 1] — normalised heatmap
    score       : float — cosine similarity S(f_I, f_T)
    """
    model.eval()
    store = _AttentionStore()

    enc_layers  = model.vision_model.encoder.layers
    target_idx  = layer_idx % len(enc_layers)
    target_attn = enc_layers[target_idx].self_attn

    orig_fwd = _patch_attention(target_attn, store)

    try:
        inputs = processor(
            text=[text], images=[pil_img],
            return_tensors="pt", padding=True,
        ).to(device)

        model.zero_grad()
        out      = model(**inputs)
        img_emb  = out.image_embeds
        txt_emb  = out.text_embeds
        i_norm   = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
        t_norm   = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)
        score_t  = (i_norm * t_norm).sum()
        score    = score_t.item()

        score_t.backward()

        if store.attn_out is None:
            raise RuntimeError(
                "Grad-ECLIP: patched forward was not called — check layer_idx."
            )
        if store.attn_out.grad is None:
            raise RuntimeError(
                "Grad-ECLIP: gradient on attention output is None. "
                "retain_grad() may not have been reached during forward."
            )
    finally:
        _restore_attention(target_attn, orig_fwd)

    # ── drop batch dimension ──────────────────────────────────────────────────
    # shapes below: (H, N, d)
    o_grad = store.attn_out.grad[0].detach()   # dS / d_o  at every position
    v_h    = store.v[0]
    A      = store.attn[0]
    q_h    = store.q_h[0]
    k_h    = store.k_h[0]

    H, N, d   = v_h.shape
    n_patches = N - 1   # position 0 = CLS token

    # ── Step 3: channel weight ─────────────────────────────────────────────
    # w_c_h = ReLU( dS/d_o at the CLS position )
    w_c = F.relu(o_grad[:, 0, :])            # (H, d)

    # ── Step 4: spatial weight ─────────────────────────────────────────────
    a_cls = A[:, 0, 1:]                       # (H, n_patches)  CLS → patches

    if with_ksim:
        # Key-query similarity: how much does each patch's key activate the CLS query?
        q_cls = q_h[:, 0:1, :]               # (H, 1, d)
        k_pts = k_h[:, 1:, :]               # (H, n_patches, d)
        k_sim = torch.sigmoid(
            torch.bmm(q_cls, k_pts.transpose(1, 2)).squeeze(1) / math.sqrt(d)
        )                                     # (H, n_patches)
        w_i = _loosen(a_cls * k_sim)
    else:
        w_i = _loosen(a_cls)                  # (H, n_patches)

    # ── Step 5: per-head saliency map ─────────────────────────────────────
    v_pts        = v_h[:, 1:, :]                              # (H, n_patches, d)
    channel_proj = (v_pts * w_c.unsqueeze(1)).sum(dim=-1)     # (H, n_patches)
    M            = w_i * channel_proj                         # (H, n_patches)

    # ── Step 6: aggregate ─────────────────────────────────────────────────
    E = F.relu(M.sum(dim=0)).cpu().numpy()    # (n_patches,)

    # ── Step 7: reshape, upsample, normalise ──────────────────────────────
    grid = int(math.ceil(math.sqrt(n_patches)))
    pad  = grid * grid - n_patches
    if pad > 0:
        E = np.concatenate([E, np.zeros(pad, dtype=np.float32)])
    cam_2d = E.reshape(grid, grid)

    mn, mx = cam_2d.min(), cam_2d.max()
    cam_2d = (cam_2d - mn) / (mx - mn + 1e-8)

    W, H_img = pil_img.size
    cam_pil  = Image.fromarray((cam_2d * 255).astype(np.uint8))
    cam_pil  = cam_pil.resize((W, H_img), Image.BICUBIC)
    cam_np   = np.array(cam_pil, dtype=np.float32) / 255.0

    overlay_pil = _make_overlay(pil_img, cam_np, alpha, colormap)
    return overlay_pil, cam_np, score


# ── helpers ───────────────────────────────────────────────────────────────────

def _loosen(x: torch.Tensor) -> torch.Tensor:
    """
    L1-normalise absolute values.
    loosen(x)_i = |x_i| / sum_j |x_j|
    Avoids the winner-takes-all concentration of softmax.
    """
    abs_x = x.abs()
    return abs_x / abs_x.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _make_overlay(pil_img: Image.Image, cam_np: np.ndarray,
                  alpha: float, colormap: str) -> Image.Image:
    cmap  = mpl_cm.get_cmap(colormap)
    heat  = (cmap(cam_np)[..., :3] * 255).astype(np.float64)
    orig  = np.array(pil_img.convert("RGB"), dtype=np.float64)
    blend = np.clip((1.0 - alpha) * orig + alpha * heat, 0, 255)
    return Image.fromarray(blend.astype(np.uint8))


# ── multi-panel figure ────────────────────────────────────────────────────────

def make_gradeclip_figure(
    pil_img:     Image.Image,
    overlay_pil: Image.Image,
    cam_np:      np.ndarray,
    text:        str,
    score:       float,
    colormap:    str   = "inferno",
    figsize:     tuple = (15, 5),
) -> Image.Image:
    """
    Three-panel figure: [original] [raw heatmap + colorbar] [overlay].
    Returns a PIL Image for st.image().
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#0f1117")

    panels = [
        ("Original",                    pil_img,     None),
        ("Grad-ECLIP heatmap",          None,        cam_np),
        (f"Overlay  (sim={score:.4f})", overlay_pil, None),
    ]
    for ax, (title, img, hmap) in zip(axes, panels):
        ax.set_title(title, color="white", fontsize=10, pad=5)
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        else:
            im = ax.imshow(hmap, cmap=colormap, vmin=0, vmax=1)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(f'Query: "{text}"', color="#8b90b8", fontsize=10, y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── standalone similarity ─────────────────────────────────────────────────────

def clip_similarity(pil_img: Image.Image, text: str,
                    model, processor, device: str) -> float:
    """Cosine similarity for a single image–text pair, no gradients."""
    inputs = processor(text=[text], images=[pil_img],
                       return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out   = model(**inputs)
        i_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        t_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        return (i_emb * t_emb).sum().item()
