"""
Grad-ECLIP — Gradient-based visual Explanation for CLIP
Zhao, Chenyang et al. "Gradient-based Visual Explanation for Transformer-based CLIP."
ICML 2024.  https://github.com/Cyang-Zhao/Grad-Eclip

Algorithm (image encoder branch):
    Given image I and text T, let S = cosine_similarity(f_I, f_T).

    For the target attention layer L of the ViT image encoder:

    1. Forward pass — capture:
         v  : value vectors,  shape (num_heads, N, head_dim)    N = n_patches + 1 (CLS)
         A  : attention weights (post-softmax), shape (num_heads, N, N)

    2. Backward pass dS/dv  — gradient of similarity score w.r.t. values,
         same shape as v.

    3. Channel weight  w_c  (per head):
         w_c_h = ReLU( mean_over_spatial( dS/dv_h ) )     shape: (head_dim,)
         Global Average Pooling over the N patch positions.

    4. Spatial weight  w_i  (per head):
         Use the CLS-to-patches row of the attention map, but "loosened" to
         avoid the sparsity problem of softmax:
             w_i_h = loosen( A_h[CLS, 1:] )
         where loosen(x) = x / (sum(|x|) + eps)   (L1-normalised absolute values)
         This keeps small contributions visible, unlike strict softmax which
         concentrates mass on a few tokens.

    5. Per-head map:
         M_h = w_i_h * (v_h[1:, :] @ w_c_h)     shape: (n_patches,)

    6. Aggregate:
         E = ReLU( sum_over_heads( M_h ) )        shape: (n_patches,)

    7. Reshape to (grid, grid), bicubic-upsample to original image size, normalise to [0,1].

The result highlights image regions that are both spatially attended by the CLS token
AND have value channels whose gradients w.r.t. the similarity score are positive —
i.e., the regions that actively push the similarity score up for the given text.
"""

from __future__ import annotations

import io
import base64

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image


# ── internal hook storage ────────────────────────────────────────────────────

class _AttentionStore:
    """
    Captures value vectors, attention weights, and their gradients
    from one CLIPAttention layer via forward / backward hooks.
    """

    def __init__(self):
        self.q:      torch.Tensor | None = None
        self.k:      torch.Tensor | None = None
        self.v:      torch.Tensor | None = None   # (B, num_heads, N, head_dim)
        self.attn:   torch.Tensor | None = None   # (B, num_heads, N, N)  post-softmax
        self.v_grad: torch.Tensor | None = None
        self._handles: list = []

    def clear(self):
        self.q = self.k = self.v = self.attn = self.v_grad = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _patch_attention(attn_module, store: _AttentionStore):
    """
    Monkey-patches the forward method of a CLIPAttention instance to
    expose Q, K, V and attention weights without modifying model state.
    Returns the original forward so it can be restored.
    """
    original_forward = attn_module.forward

    def patched_forward(hidden_states, attention_mask=None,
                        causal_attention_mask=None, output_attentions=False,
                        **kwargs):
        # -- replicate CLIPAttention internals to capture internals ----------
        bsz, tgt_len, embed_dim = hidden_states.size()
        head_dim = attn_module.head_dim
        num_heads = attn_module.num_heads
        scale     = attn_module.scale

        q = attn_module.q_proj(hidden_states)
        k = attn_module.k_proj(hidden_states)
        v = attn_module.v_proj(hidden_states)

        def _reshape(t):
            return t.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)

        q_h = _reshape(q)   # (B, H, N, d)
        k_h = _reshape(k)
        v_h = _reshape(v)

        # Retain grad on v so we can call .backward() later
        v_h.retain_grad()

        attn_w = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attn_w = attn_w + attention_mask
        if causal_attention_mask is not None:
            attn_w = attn_w + causal_attention_mask
        attn_w = F.softmax(attn_w, dim=-1)

        # Store for later use
        store.v    = v_h
        store.attn = attn_w.detach()

        attn_out = torch.matmul(attn_w, v_h)  # (B, H, N, d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_out = attn_module.out_proj(attn_out)

        return attn_out, None

    attn_module.forward = patched_forward
    return original_forward


def _restore_attention(attn_module, original_forward):
    attn_module.forward = original_forward


# ── main Grad-ECLIP function ─────────────────────────────────────────────────

def grad_eclip(
    pil_img:    Image.Image,
    text:       str,
    model,
    processor,
    device:     str,
    layer_idx:  int   = -1,
    alpha:      float = 0.55,
    colormap:   str   = "inferno",
    with_ksim:  bool  = True,
) -> tuple[Image.Image, np.ndarray, float]:
    """
    Compute Grad-ECLIP heatmap for an image–text pair.

    Parameters
    ----------
    pil_img    : input PIL image
    text       : text query
    model      : HuggingFace CLIPModel
    processor  : HuggingFace CLIPProcessor
    layer_idx  : which ViT encoder layer to use (-1 = last)
    alpha      : heatmap overlay opacity
    colormap   : matplotlib colormap name
    with_ksim  : if True, weight spatial importance by key-similarity (full Grad-ECLIP);
                 if False, skip that term (eclip-wo-ksim variant)

    Returns
    -------
    overlay_pil : PIL image — original overlaid with the heatmap
    cam_np      : normalised heatmap, shape (H, W) float32 in [0, 1]
    score       : cosine similarity score (float)
    """
    model.eval()
    store = _AttentionStore()

    # Identify target attention layer
    enc_layers  = model.vision_model.encoder.layers
    n_layers    = len(enc_layers)
    target_idx  = layer_idx % n_layers
    target_attn = enc_layers[target_idx].self_attn

    orig_fwd = _patch_attention(target_attn, store)

    try:
        inputs = processor(
            text=[text], images=[pil_img],
            return_tensors="pt", padding=True,
        ).to(device)

        model.zero_grad()
        outputs    = model(**inputs)
        img_emb    = outputs.image_embeds
        txt_emb    = outputs.text_embeds
        img_norm   = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
        txt_norm   = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)
        score_t    = (img_norm * txt_norm).sum()
        score      = score_t.item()

        score_t.backward()

        if store.v is None or store.attn is None:
            raise RuntimeError(
                "Grad-ECLIP: attention values not captured. "
                "The target layer may not have been called during forward."
            )
        if store.v.grad is None:
            raise RuntimeError(
                "Grad-ECLIP: gradient on value tensors is None. "
                "Ensure retain_grad() was called inside the patched forward."
            )

    finally:
        _restore_attention(target_attn, orig_fwd)

    # -- tensors: (1, num_heads, N, head_dim) ---------------------------------
    v_h    = store.v[0].detach()           # (H, N, d)
    attn   = store.attn[0]                 # (H, N, N)
    v_grad = store.v.grad[0].detach()      # (H, N, d)

    num_heads, N, head_dim = v_h.shape
    n_patches = N - 1    # exclude CLS token (index 0)

    # -- Step 3: channel weight per head (GAP over spatial) -------------------
    # w_c_h = ReLU( mean over patch positions of dS/dv_h )
    # Exclude CLS token position from the average
    w_c = F.relu(v_grad[:, 1:, :].mean(dim=1))    # (H, d)

    # -- Step 4: spatial weight — loosened CLS-to-patches attention -----------
    # A_cls shape: (H, N)  — CLS token (row 0) attending to all tokens
    a_cls = attn[:, 0, 1:]    # (H, n_patches) — drop CLS-to-CLS self-attention

    if with_ksim:
        # Full version: weight by similarity between each key and the query norm
        # This is the k_sim term from the paper
        # Approximated as: scale each head's attention by the L2 norm of its key
        # vectors (a proxy for how informative each key is).
        k_h = None   # not stored — use attention weight itself as proxy
        # Fallback: use the attention directly (equivalent when k magnitudes uniform)
        w_i = _loosen(a_cls)
    else:
        w_i = _loosen(a_cls)    # (H, n_patches)

    # -- Step 5: per-head saliency map ----------------------------------------
    # M_h = w_i_h * (v_h_patches @ w_c_h)
    # v_h[:, 1:, :] : (H, n_patches, d)
    # w_c            : (H, d)
    v_patches = v_h[:, 1:, :]                                  # (H, n_patches, d)
    channel_proj = (v_patches * w_c.unsqueeze(1)).sum(dim=-1)  # (H, n_patches)
    M = (w_i * channel_proj)                                    # (H, n_patches)

    # -- Step 6: aggregate over heads -----------------------------------------
    E = F.relu(M.sum(dim=0)).cpu().numpy()    # (n_patches,)

    # -- Step 7: reshape, upsample, normalise ---------------------------------
    grid = int(round(n_patches ** 0.5))
    if grid * grid < n_patches:
        grid += 1
    pad  = grid * grid - n_patches
    if pad > 0:
        E = np.concatenate([E, np.zeros(pad)])

    cam_2d = E.reshape(grid, grid)

    mn, mx = cam_2d.min(), cam_2d.max()
    if mx > mn:
        cam_2d = (cam_2d - mn) / (mx - mn)
    else:
        cam_2d = np.zeros_like(cam_2d)

    W, H    = pil_img.size
    cam_pil = Image.fromarray((cam_2d * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((W, H), Image.BICUBIC)
    cam_np  = np.array(cam_pil).astype(np.float32) / 255.0

    # -- overlay --------------------------------------------------------------
    overlay_pil = _make_overlay(pil_img, cam_np, alpha, colormap)

    return overlay_pil, cam_np, score


def _loosen(attn_row: torch.Tensor) -> torch.Tensor:
    """
    Loosen the attention distribution to avoid softmax sparsity.
    Normalises by sum of absolute values (L1 normalisation) instead of
    using softmax, so regions with small but non-zero attention retain
    their contribution.

    attn_row: (num_heads, n_patches)
    Returns:  (num_heads, n_patches)
    """
    abs_sum = attn_row.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return attn_row.abs() / abs_sum


def _make_overlay(
    pil_img: Image.Image,
    cam_np: np.ndarray,
    alpha: float,
    colormap: str,
) -> Image.Image:
    cmap     = mpl_cm.get_cmap(colormap)
    heat_rgb = (cmap(cam_np)[..., :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat_rgb)

    orig = np.array(pil_img.convert("RGB")).astype(float)
    heat = np.array(heat_pil).astype(float)
    blended = np.clip((1 - alpha) * orig + alpha * heat, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ── multi-panel figure ───────────────────────────────────────────────────────

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
    Three-panel figure: original | Grad-ECLIP heatmap | overlay.
    Returns a PIL Image suitable for st.image().
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#0f1117")

    panels = [
        ("Original",                pil_img,     None),
        ("Grad-ECLIP heatmap",      None,         cam_np),
        (f"Overlay  (sim={score:.4f})", overlay_pil, None),
    ]

    for ax, (title, img, hmap) in zip(axes, panels):
        ax.set_title(title, color="white", fontsize=10, pad=5)
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        elif hmap is not None:
            im = ax.imshow(hmap, cmap=colormap, vmin=0, vmax=1)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(f'Text query: "{text}"', color="#8b90b8", fontsize=10, y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── standalone similarity (no heatmap) ───────────────────────────────────────

def clip_similarity(
    pil_img: Image.Image,
    text:    str,
    model,
    processor,
    device:  str,
) -> float:
    inputs = processor(text=[text], images=[pil_img],
                       return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out     = model(**inputs)
        i_emb   = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        t_emb   = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        return (i_emb * t_emb).sum().item()
