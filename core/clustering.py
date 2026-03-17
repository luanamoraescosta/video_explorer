"""
Redução de dimensionalidade (UMAP) e clustering (HDBSCAN).
"""

import numpy as np
import hdbscan
import umap
from sklearn.preprocessing import normalize


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (reduced_nd, coords_2d).
    reduced_nd → entrada para HDBSCAN
    coords_2d  → scatter de visualização
    """
    emb_norm = normalize(embeddings)
    nn = min(15, len(embeddings) - 1)

    r_nd = umap.UMAP(
        n_components=n_components, n_neighbors=nn,
        min_dist=0.0, metric="cosine", random_state=random_state,
    ).fit_transform(emb_norm)

    r_2d = umap.UMAP(
        n_components=2, n_neighbors=nn,
        min_dist=0.1, metric="cosine", random_state=random_state,
    ).fit_transform(emb_norm)

    return r_nd, r_2d


def run_hdbscan(
    reduced: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    method: str = "eom",
) -> np.ndarray:
    """
    Retorna labels (N,). Labels == -1 são ruído.
    """
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        metric="euclidean",
    ).fit_predict(reduced)
    return labels


def cluster_summary(labels: np.ndarray) -> dict:
    real   = sorted(c for c in set(labels) if c != -1)
    noise  = int((labels == -1).sum())
    dist   = {c: int((labels == c).sum()) for c in real}
    return {
        "n_clusters": len(real),
        "n_noise": noise,
        "distribution": dist,
    }


def get_representative_idx(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """
    Para cada cluster, retorna o índice do frame mais próximo do centróide.
    """
    from sklearn.preprocessing import normalize as sk_norm
    emb_n = sk_norm(embeddings)
    reps  = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idxs     = np.where(labels == c)[0]
        centroid = emb_n[idxs].mean(0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        best     = idxs[np.linalg.norm(emb_n[idxs] - centroid, axis=1).argmin()]
        reps[c]  = int(best)
    return reps
