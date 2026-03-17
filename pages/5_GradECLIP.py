"""
Page 5 -- Grad-ECLIP Explainability
Zhao et al. "Gradient-based Visual Explanation for Transformer-based CLIP." ICML 2024.
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
from core.embeddings import load_clip, compute_text_embedding, cosine_scores
from core.gradcam import grad_eclip, make_gradeclip_figure, clip_similarity

st.set_page_config(page_title="Grad-ECLIP", layout="wide")
init_session()
apply_global_css()

st.title("Grad-ECLIP — Gradient-based CLIP Explainability")
st.caption(
    "Zhao et al., ICML 2024. "
    "Highlights image regions whose value vectors have the highest positive gradient "
    "w.r.t. the image-text similarity score, weighted by a loosened attention map "
    "to avoid softmax sparsity."
)

require_frames()

# -- ensure model loaded ------------------------------------------------------
if not st.session_state.clip_model:
    with st.spinner("Loading CLIP model..."):
        @st.cache_resource(show_spinner=False)
        def _load():
            return load_clip()
        m, p, d = _load()
        st.session_state.clip_model  = m
        st.session_state.clip_proc   = p
        st.session_state.clip_device = d

model  = st.session_state.clip_model
proc   = st.session_state.clip_proc
device = st.session_state.clip_device
emb    = st.session_state.embeddings
ts     = st.session_state.timestamps
b64s   = st.session_state.frames_b64
sc_ids = st.session_state.scene_ids
labels = st.session_state.labels

# -- sidebar configuration ----------------------------------------------------
with st.sidebar:
    st.header("Grad-ECLIP parameters")

    query = st.text_input(
        "Text query",
        placeholder="e.g. face of a person, car in motion...",
        key="gc_query",
    )

    st.divider()

    layer_idx = st.slider(
        "Target attention layer",
        min_value=-12, max_value=-1, value=-1,
        help="-1 = last layer (most semantic). Earlier layers capture more local features.",
    )
    alpha = st.slider(
        "Overlay opacity (alpha)",
        min_value=0.1, max_value=0.9, value=0.55, step=0.05,
    )
    colormap = st.selectbox(
        "Colormap",
        ["inferno", "plasma", "viridis", "magma", "hot", "jet", "RdYlGn"],
        index=0,
    )
    with_ksim = st.toggle(
        "With key-similarity weighting (full Grad-ECLIP)",
        value=True,
        help="Disabling uses the eclip-wo-ksim variant from the paper.",
    )

    st.divider()
    st.subheader("Frame selection")
    select_mode = st.radio(
        "Mode",
        ["By index", "Top-K by query", "By cluster"],
        key="gc_mode",
    )
    top_k_gc = st.slider("Number of frames to analyse", 1, 12, 4, key="gc_topk")

# -- frame selection ----------------------------------------------------------
def get_frame_indices() -> list[int]:
    mode = st.session_state.gc_mode

    if mode == "By index":
        raw = st.text_input(
            "Frame indices (comma-separated)",
            value="0",
            key="gc_idxs_input",
        )
        try:
            idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
            idxs = [i for i in idxs if 0 <= i < len(b64s)]
        except ValueError:
            idxs = [0]
        return idxs[:top_k_gc]

    elif mode == "Top-K by query" and query.strip():
        txt_vec = compute_text_embedding(query.strip(), model, proc, device)
        scores  = cosine_scores(emb, txt_vec)
        return np.argsort(scores)[::-1][:top_k_gc].tolist()

    elif mode == "By cluster" and labels is not None:
        from core.clustering import get_representative_idx
        reps   = get_representative_idx(emb, labels)
        c_opts = sorted(c for c in set(labels) if c != -1)
        sel_c  = st.selectbox("Cluster", c_opts, key="gc_cluster_sel")
        c_idxs = np.where(labels == sel_c)[0]
        if sel_c in reps:
            c_idxs = np.concatenate([[reps[sel_c]],
                                      [i for i in c_idxs if i != reps[sel_c]]])
        return c_idxs[:top_k_gc].tolist()

    else:
        return list(range(min(top_k_gc, len(b64s))))


frame_idxs = get_frame_indices()
st.markdown(f"**Selected frames:** {frame_idxs}")

if not query.strip():
    st.warning("Enter a text query in the sidebar.")
    st.stop()

if st.button("Generate heatmaps", type="primary"):
    results = []
    prog = st.progress(0, text="Running Grad-ECLIP...")

    for i, fidx in enumerate(frame_idxs):
        prog.progress(i / len(frame_idxs), text=f"Frame {i+1}/{len(frame_idxs)}...")
        pil = b64_to_pil(b64s[fidx])
        try:
            overlay_pil, cam_np, score = grad_eclip(
                pil, query.strip(), model, proc, device,
                layer_idx=layer_idx,
                alpha=alpha,
                colormap=colormap,
                with_ksim=with_ksim,
            )
            results.append({
                "fidx": fidx, "pil": pil,
                "overlay": overlay_pil, "cam": cam_np,
                "score": score,
                "ts": fmt_ts(ts[fidx]),
                "scene": sc_ids[fidx] if sc_ids else fidx,
            })
        except Exception as e:
            st.error(f"Frame {fidx}: {e}")

    prog.progress(1.0, text="Done.")
    st.divider()

    # -- results --------------------------------------------------------------
    for r in results:
        c_label = ""
        if labels is not None:
            c     = int(labels[r["fidx"]])
            name  = "Noise" if c == -1 else f"Cluster {c}"
            color = cluster_color(c)
            c_label = f" | <span style='color:{color}'>{name}</span>"

        st.markdown(
            f"**Frame #{r['fidx']}** | {r['ts']} | scene {r['scene']}"
            f" | similarity <b>{r['score']:.4f}</b>{c_label}",
            unsafe_allow_html=True,
        )

        fig_pil = make_gradeclip_figure(
            r["pil"], r["overlay"], r["cam"],
            text=query.strip(),
            score=r["score"],
            colormap=colormap,
        )
        st.image(fig_pil, use_container_width=True)

        with st.expander(f"Interactive heatmap — frame #{r['fidx']}"):
            fig = go.Figure(go.Heatmap(
                z=r["cam"],
                colorscale=colormap,
                showscale=True,
                hovertemplate="row=%{y}  col=%{x}  activation=%{z:.4f}<extra></extra>",
            ))
            import io as _io, base64 as _b64
            buf = _io.BytesIO()
            r["pil"].save(buf, format="PNG")
            img_b64 = _b64.b64encode(buf.getvalue()).decode()
            grid = r["cam"].shape[0]
            fig.add_layout_image(dict(
                source=f"data:image/png;base64,{img_b64}",
                xref="x", yref="y",
                x=0, y=0,
                sizex=grid, sizey=grid,
                sizing="stretch",
                opacity=0.32,
                layer="below",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117",
                height=430, margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False,
                           scaleanchor="x", autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

    if len(results) > 1:
        with st.expander("Score comparison"):
            labels_x = [f"#{r['fidx']} ({r['ts']})" for r in results]
            vals     = [r["score"] for r in results]
            fig = go.Figure(go.Bar(
                x=labels_x, y=vals,
                marker_color=[
                    f"rgba(124,111,247,{0.4 + 0.6*(s/max(vals))})" for s in vals
                ],
                text=[f"{s:.4f}" for s in vals],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117",
                height=320, margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="Cosine similarity",
                yaxis=dict(range=[0, max(vals) * 1.2]),
            )
            st.plotly_chart(fig, use_container_width=True)

elif query.strip():
    st.info("Click Generate heatmaps to run Grad-ECLIP.")
