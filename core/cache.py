"""
Gerenciamento de cache em disco.
Todos os artefatos ficam em .cache/ ao lado do vídeo.
"""

from pathlib import Path
import numpy as np
import json


def _cache_dir(video_path: str) -> Path:
    d = Path(video_path).parent / ".clip_cache"
    d.mkdir(exist_ok=True)
    return d


def _stem(video_path: str) -> str:
    return Path(video_path).stem


# ── frames + embeddings ──────────────────────────────────────

def frames_cache_path(video_path: str, mode: str, param) -> Path:
    return _cache_dir(video_path) / f"{_stem(video_path)}_{mode}_{param}.npz"


def save_frames(video_path: str, mode: str, param,
                embeddings, timestamps, frames_b64, scene_ids) -> None:
    p = frames_cache_path(video_path, mode, param)
    np.savez_compressed(
        p,
        embeddings = embeddings,
        timestamps = np.array(timestamps, dtype=float),
        frames_b64 = np.array(frames_b64, dtype=object),
        scene_ids  = np.array(scene_ids,  dtype=int),
    )


def load_frames(video_path: str, mode: str, param):
    """Retorna (embeddings, timestamps, frames_b64, scene_ids) ou None."""
    p = frames_cache_path(video_path, mode, param)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return (
        data["embeddings"],
        data["timestamps"].tolist(),
        data["frames_b64"].tolist(),
        data["scene_ids"].tolist(),
    )


def frames_cache_exists(video_path: str, mode: str, param) -> bool:
    return frames_cache_path(video_path, mode, param).exists()


# ── clusters ─────────────────────────────────────────────────

def clusters_cache_path(video_path: str, mode: str, param,
                        min_cs: int, min_s: int, umap_n: int) -> Path:
    tag = f"hdb_mcs{min_cs}_ms{min_s}_u{umap_n}"
    return _cache_dir(video_path) / f"{_stem(video_path)}_{mode}_{param}_{tag}.npz"


def save_clusters(video_path, mode, param, min_cs, min_s, umap_n,
                  labels, coords_2d) -> None:
    p = clusters_cache_path(video_path, mode, param, min_cs, min_s, umap_n)
    np.savez_compressed(p, labels=labels, coords_2d=coords_2d)


def load_clusters(video_path, mode, param, min_cs, min_s, umap_n):
    """Retorna (labels, coords_2d) ou None."""
    p = clusters_cache_path(video_path, mode, param, min_cs, min_s, umap_n)
    if not p.exists():
        return None
    data = np.load(p)
    return data["labels"], data["coords_2d"]


def clusters_cache_exists(video_path, mode, param, min_cs, min_s, umap_n) -> bool:
    return clusters_cache_path(video_path, mode, param, min_cs, min_s, umap_n).exists()


# ── metadados gerais do projeto ───────────────────────────────

def meta_path(video_path: str) -> Path:
    return _cache_dir(video_path) / f"{_stem(video_path)}_meta.json"


def save_meta(video_path: str, data: dict) -> None:
    with open(meta_path(video_path), "w") as f:
        json.dump(data, f, indent=2)


def load_meta(video_path: str) -> dict:
    p = meta_path(video_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def list_cache_files(video_path: str) -> list[Path]:
    return sorted(_cache_dir(video_path).glob(f"{_stem(video_path)}*"))
