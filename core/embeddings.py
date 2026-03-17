"""
Embeddings CLIP para imagens e texto.
Mantém o modelo em cache de sessão (st.cache_resource via wrapper externo).
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.preprocessing import normalize as sk_normalize


# ── carregamento do modelo ────────────────────────────────────

def load_clip(model_name: str = "openai/clip-vit-base-patch32",
              device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, device


# ── embeddings de imagem ──────────────────────────────────────

def compute_image_embeddings(
    frames_pil: list[Image.Image],
    model, processor, device,
    batch_size: int = 16,
    progress_cb=None,
) -> np.ndarray:
    all_emb   = []
    n         = len(frames_pil)
    n_batches = (n - 1) // batch_size + 1

    for i in range(0, n, batch_size):
        batch  = frames_pil[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            
            # Adiciona compatibilidade para o transformers v5+
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif isinstance(feats, tuple):
                feats = feats[0]
                
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_emb.append(feats.cpu().numpy())
        if progress_cb:
            progress_cb(i // batch_size + 1, n_batches)

    return np.vstack(all_emb)


# ── embedding de texto ────────────────────────────────────────

def compute_text_embedding(text, model, processor, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif isinstance(feats, (list, tuple)):
            feats = feats[0]
        # ---------------------------------------------------------------------

        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

# ── similaridade ─────────────────────────────────────────────

def cosine_scores(embeddings: np.ndarray, text_vec: np.ndarray) -> np.ndarray:
    """
    embeddings: (N, D) já L2-normalizados
    text_vec:   (D,)   já L2-normalizado
    Retorna (N,) de scores em [-1, 1].
    """
    return (embeddings @ text_vec).astype(float)
