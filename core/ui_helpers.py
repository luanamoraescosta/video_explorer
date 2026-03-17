"""
Helpers de UI compartilhados entre as páginas.
"""

import base64
import io
import streamlit as st
from PIL import Image
import numpy as np


# ── estilo global ─────────────────────────────────────────────

DARK_CSS = """
<style>
[data-testid="stSidebar"] { background: #1a1d27; }
.stApp { background: #0f1117; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
.frame-caption { font-size: .7rem; color: #8b90b8; text-align: center; margin-top: 2px; }
</style>
"""


def apply_global_css():
    st.markdown(DARK_CSS, unsafe_allow_html=True)


# ── estado de sessão ─────────────────────────────────────────

DEFAULTS = {
    "video_path":   None,
    "mode":         "scene",
    "param":        27.0,        # threshold (scene) ou interval_sec (interval)
    "embeddings":   None,
    "timestamps":   None,
    "frames_b64":   None,
    "scene_ids":    None,
    "frames_pil":   None,
    "labels":       None,
    "coords_2d":    None,
    "clip_model":   None,
    "clip_proc":    None,
    "clip_device":  None,
}


def init_session():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def session_has_frames() -> bool:
    return (st.session_state.embeddings is not None and
            st.session_state.timestamps is not None)


def session_has_clusters() -> bool:
    return (session_has_frames() and
            st.session_state.labels is not None and
            st.session_state.coords_2d is not None)


def session_has_model() -> bool:
    return st.session_state.clip_model is not None


# ── conversões de frame ───────────────────────────────────────

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def pil_to_b64(pil: Image.Image, fmt="JPEG", quality=85) -> str:
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── renderização de grid de frames ───────────────────────────

def frame_grid(
    frames_b64: list,
    timestamps: list,
    scene_ids: list | None = None,
    n_cols: int = 4,
    max_show: int | None = None,
    key_prefix: str = "grid",
):
    """
    Renderiza frames em grade. Retorna índice do frame clicado (ou None).
    """
    items = list(zip(frames_b64, timestamps,
                     scene_ids if scene_ids else range(len(frames_b64))))
    if max_show:
        items = items[:max_show]

    cols = st.columns(n_cols)
    clicked = None
    for i, (b64, ts, sc) in enumerate(items):
        with cols[i % n_cols]:
            pil = b64_to_pil(b64)
            st.image(pil, use_container_width=True)
            label = f"{_fmt_ts(ts)}"
            if scene_ids is not None:
                label += f"  ·  cena {sc}"
            st.markdown(f'<div class="frame-caption">{label}</div>',
                        unsafe_allow_html=True)
    return clicked


# ── formatação de timestamp ───────────────────────────────────

def _fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def fmt_ts(sec: float) -> str:
    return _fmt_ts(sec)


# ── banner de aviso de dependência ───────────────────────────

def require_frames():
    if not session_has_frames():
        st.warning("⚠️ Nenhum frame/embedding disponível. "
                   "Execute primeiro a **Extração** e depois os **Embeddings**.")
        st.stop()


def require_clusters():
    if not session_has_clusters():
        st.warning("⚠️ Nenhum cluster disponível. Execute primeiro a página **Clustering**.")
        st.stop()


def require_model():
    if not session_has_model():
        st.warning("⚠️ Modelo CLIP não carregado. Execute a página **Embeddings** primeiro.")
        st.stop()


# ── paleta de cores por cluster ───────────────────────────────

PALETTE = [
    "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
    "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
    "#17becf","#bcbd22","#d62728","#9467bd","#8c564b",
    "#e377c2","#7f7f7f","#2ca02c","#ff7f0e","#1f77b4",
]


def cluster_color(cluster_id: int) -> str:
    if cluster_id == -1:
        return "#888888"
    return PALETTE[cluster_id % len(PALETTE)]
