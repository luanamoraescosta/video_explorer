"""
Page 3 -- UMAP + HDBSCAN Clustering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.ui_helpers import (
    init_session, apply_global_css, require_frames,
    b64_to_pil, fmt_ts, cluster_color,
)
from core.clustering import reduce_umap, run_hdbscan, cluster_summary, get_representative_idx
from core import cache

st.set_page_config(page_title="Clustering", layout="wide")
init_session()
apply_global_css()

st.title("Clustering — UMAP + HDBSCAN")
require_frames()

video_path = st.session_state.video_path
mode       = st.session_state.mode
param      = st.session_state.param

# -- parameters ---------------------------------------------------------------
with st.sidebar:
    st.header("Parameters")

    umap_n = st.slider("UMAP dimensions (pre-HDBSCAN)",
                        min_value=2, max_value=50, value=10,
                        help="Reduces embeddings before clustering. 5-15 works well.")
    min_cs = st.slider("min_cluster_size",
                        min_value=2, max_value=30, value=3,
                        help="Minimum cluster size. Larger = fewer clusters.")
    min_s  = st.slider("min_samples",
                        min_value=1, max_value=20, value=2,
                        help="Noise conservatism. Larger = more points labelled as noise.")
    method = st.radio("Cluster selection method", ["eom", "leaf"],
                       help="'eom' tends to produce compact clusters; 'leaf' more granular.")

    st.divider()
    n_exemplars = st.slider("Exemplars per cluster", min_value=1, max_value=16, value=4)

# -- cache check --------------------------------------------------------------
has_cache = bool(video_path and
                 cache.clusters_cache_exists(video_path, mode, param, min_cs, min_s, umap_n))

col_run, col_load = st.columns(2)
run_btn  = col_run.button("Run clustering", type="primary")
load_btn = col_load.button("Load from cache", disabled=not has_cache)

if has_cache:
    st.info("Cache available for these parameters.")

if load_btn and has_cache:
    result = cache.load_clusters(video_path, mode, param, min_cs, min_s, umap_n)
    if result:
        st.session_state.labels, st.session_state.coords_2d = result
        st.success("Clustering loaded from cache.")

if run_btn:
    emb = st.session_state.embeddings
    with st.spinner("Running UMAP..."):
        reduced_nd, coords_2d = reduce_umap(emb, n_components=umap_n)
    with st.spinner("Running HDBSCAN..."):
        labels = run_hdbscan(reduced_nd, min_cluster_size=min_cs,
                              min_samples=min_s, method=method)
    st.session_state.labels    = labels
    st.session_state.coords_2d = coords_2d
    cache.save_clusters(video_path, mode, param, min_cs, min_s, umap_n, labels, coords_2d)
    st.success("Clustering complete and cached.")

# -- visualisation ------------------------------------------------------------
labels    = st.session_state.labels
coords_2d = st.session_state.coords_2d

if labels is None or coords_2d is None:
    st.info("Set parameters and click Run clustering.")
    st.stop()

summary    = cluster_summary(labels)
emb        = st.session_state.embeddings
timestamps = st.session_state.timestamps
frames_b64 = st.session_state.frames_b64
scene_ids  = st.session_state.scene_ids

# -- metrics ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Clusters found", summary["n_clusters"])
c2.metric("Noise frames",   summary["n_noise"])
c3.metric("Total frames",   len(labels))

st.divider()

# -- UMAP scatter -------------------------------------------------------------
st.subheader("UMAP 2D scatter")

real_clusters = sorted(c for c in set(labels) if c != -1)
noise_mask    = labels == -1

fig = go.Figure()

if noise_mask.any():
    fig.add_trace(go.Scatter(
        x=coords_2d[noise_mask, 0], y=coords_2d[noise_mask, 1],
        mode="markers",
        marker=dict(color="#555555", size=4, opacity=0.5),
        name="Noise",
        text=[f"{fmt_ts(timestamps[i])} | scene {scene_ids[i]}"
              for i in np.where(noise_mask)[0]],
        hovertemplate="%{text}<extra>Noise</extra>",
    ))

for c in real_clusters:
    mask  = labels == c
    color = cluster_color(c)
    hover = [f"{fmt_ts(timestamps[i])} | scene {scene_ids[i]}"
             for i in np.where(mask)[0]]
    fig.add_trace(go.Scatter(
        x=coords_2d[mask, 0], y=coords_2d[mask, 1],
        mode="markers",
        marker=dict(color=color, size=7, opacity=0.85,
                    line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
        name=f"Cluster {c}  ({mask.sum()})",
        text=hover,
        hovertemplate="%{text}<extra>Cluster " + str(c) + "</extra>",
    ))

fig.update_layout(
    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    height=500, margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="v", x=1.01, y=1, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# -- exemplars ----------------------------------------------------------------
st.subheader("Cluster exemplars")
rep_idx = get_representative_idx(emb, labels)

filter_opts = ["All"] + [f"Cluster {c}" for c in real_clusters]
if noise_mask.any():
    filter_opts.append("Noise")
selected = st.selectbox("Filter cluster", filter_opts)


def cluster_section(c, n_show):
    color = cluster_color(c)
    name  = "Noise" if c == -1 else f"Cluster {c}"
    idxs  = np.where(labels == c)[0]

    if c in rep_idx:
        idxs = np.concatenate([[rep_idx[c]], [i for i in idxs if i != rep_idx[c]]])

    st.markdown(
        f'<span style="color:{color};font-weight:700">{name}</span>'
        f'<span style="color:#8b90b8;font-size:.8rem;margin-left:10px">{len(idxs)} frames</span>',
        unsafe_allow_html=True,
    )

    page_key = f"page_{c}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    start = st.session_state[page_key] * n_show
    end   = min(start + n_show, len(idxs))
    shown = idxs[start:end]

    cols = st.columns(min(n_show, 4))
    for j, idx in enumerate(shown):
        with cols[j % 4]:
            pil = b64_to_pil(frames_b64[idx])
            cap = f"{'* ' if idx == rep_idx.get(c) else ''}{fmt_ts(timestamps[idx])}"
            if scene_ids:
                cap += f" | scene {scene_ids[idx]}"
            st.image(pil, caption=cap, use_container_width=True)

    pc1, pc2, pc3 = st.columns([1, 3, 1])
    total_pages = max(1, (len(idxs) - 1) // n_show + 1)
    with pc1:
        if st.button("Prev", key=f"prev_{c}",
                     disabled=st.session_state[page_key] == 0):
            st.session_state[page_key] -= 1
            st.rerun()
    with pc2:
        st.caption(f"Page {st.session_state[page_key]+1} / {total_pages}  "
                   f"({start+1}-{end} of {len(idxs)})")
    with pc3:
        if st.button("Next", key=f"next_{c}",
                     disabled=st.session_state[page_key] >= total_pages - 1):
            st.session_state[page_key] += 1
            st.rerun()
    st.markdown("---")


if selected == "All":
    for c in real_clusters:
        cluster_section(c, n_exemplars)
    if noise_mask.any():
        cluster_section(-1, n_exemplars)
elif selected == "Noise":
    cluster_section(-1, n_exemplars)
else:
    c = int(selected.replace("Cluster ", ""))
    cluster_section(c, n_exemplars)
