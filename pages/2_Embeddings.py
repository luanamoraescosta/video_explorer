"""
Page 2 -- CLIP Embeddings
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
from core.ui_helpers import init_session, apply_global_css, require_frames
from core.embeddings import load_clip, compute_image_embeddings
from core import cache

st.set_page_config(page_title="CLIP Embeddings", layout="wide")
init_session()
apply_global_css()

st.title("CLIP Embeddings")
st.caption("Generates CLIP feature vectors for each extracted frame. "
           "Results are cached so subsequent clustering runs skip this step.")

require_frames()

video_path = st.session_state.video_path
mode       = st.session_state.mode
param      = st.session_state.param

# -- configuration ------------------------------------------------------------
st.subheader("Configuration")
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox(
        "CLIP model",
        [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
        ],
        help="Larger models produce better embeddings but are slower.",
    )

with col2:
    batch_size = st.select_slider("Batch size", options=[4, 8, 16, 32, 64], value=16)

# -- cache status -------------------------------------------------------------
has_emb_in_session = st.session_state.embeddings is not None
has_emb_in_cache   = bool(video_path and cache.frames_cache_exists(video_path, mode, param))

if has_emb_in_session:
    emb = st.session_state.embeddings
    st.success(f"Embeddings already in memory: {emb.shape[0]} frames x {emb.shape[1]} dims")
elif has_emb_in_cache:
    st.info("Cache found (may include embeddings if previously generated).")
else:
    st.warning("No embeddings cached. Click Generate to process.")

st.divider()

col_run, col_load = st.columns(2)
run_emb  = col_run.button("Generate embeddings", type="primary")
load_btn = col_load.button("Load from cache", disabled=not has_emb_in_cache)

if load_btn and has_emb_in_cache:
    result = cache.load_frames(video_path, mode, param)
    if result:
        emb, ts, b64, sc = result
        if emb is not None and len(emb) > 0:
            st.session_state.embeddings = emb
            st.session_state.timestamps = ts
            st.session_state.frames_b64 = b64
            st.session_state.scene_ids  = sc
            st.success(f"{len(ts)} frames + embeddings loaded from cache.")
        else:
            st.error("Cache does not contain embeddings. Generate them first.")

if run_emb:
    frames_pil = st.session_state.frames_pil
    if frames_pil is None:
        from core.ui_helpers import b64_to_pil
        if st.session_state.frames_b64:
            with st.spinner("Reconstructing PIL frames from cache..."):
                frames_pil = [b64_to_pil(b) for b in st.session_state.frames_b64]
        else:
            st.error("No frames available. Run extraction first.")
            st.stop()

    @st.cache_resource(show_spinner="Loading CLIP model...")
    def get_model(name):
        return load_clip(name)

    model, processor, device = get_model(model_name)
    st.session_state.clip_model  = model
    st.session_state.clip_proc   = processor
    st.session_state.clip_device = device

    prog = st.progress(0, text="Starting...")
    def cb(done, total):
        prog.progress(done / max(total, 1), text=f"Batch {done}/{total}...")

    with st.spinner("Computing embeddings..."):
        embeddings = compute_image_embeddings(
            frames_pil, model, processor, device,
            batch_size=batch_size,
            progress_cb=cb,
        )

    prog.progress(1.0, text="Done.")
    st.session_state.embeddings = embeddings
    cache.save_frames(
        video_path, mode, param,
        embeddings,
        st.session_state.timestamps,
        st.session_state.frames_b64,
        st.session_state.scene_ids,
    )
    st.success(f"Embeddings generated and cached: {embeddings.shape[0]} x {embeddings.shape[1]}")
    st.session_state.labels    = None
    st.session_state.coords_2d = None

# -- statistics ---------------------------------------------------------------
if st.session_state.embeddings is not None:
    emb = st.session_state.embeddings
    st.divider()
    st.subheader("Embedding statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames",      emb.shape[0])
    c2.metric("Dimensions",  emb.shape[1])
    c3.metric("Mean norm",   f"{np.linalg.norm(emb, axis=1).mean():.4f}")
    c4.metric("Model",       model_name.split("/")[-1])

    with st.expander("L2 norm distribution"):
        import plotly.express as px
        norms = np.linalg.norm(emb, axis=1)
        fig   = px.histogram(x=norms, nbins=40,
                             labels={"x": "L2 norm"},
                             title="Embedding L2 norm distribution",
                             template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Next step: Clustering")
