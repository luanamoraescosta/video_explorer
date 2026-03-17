"""
Page 4 -- Semantic Search
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np

from core.ui_helpers import (
    init_session, apply_global_css, require_frames,
    b64_to_pil, fmt_ts, cluster_color,
)
from core.embeddings import load_clip, compute_text_embedding, cosine_scores

st.set_page_config(page_title="Semantic Search", layout="wide")
init_session()
apply_global_css()

st.title("Semantic Search")
st.caption("Ranks all frames by cosine similarity with a free-text CLIP query.")

require_frames()

if not st.session_state.clip_model:
    with st.spinner("Loading CLIP model..."):
        @st.cache_resource(show_spinner=False)
        def _load():
            return load_clip()
        m, p, d = _load()
        st.session_state.clip_model  = m
        st.session_state.clip_proc   = p
        st.session_state.clip_device = d

model    = st.session_state.clip_model
proc     = st.session_state.clip_proc
device   = st.session_state.clip_device
emb      = st.session_state.embeddings
ts       = st.session_state.timestamps
b64s     = st.session_state.frames_b64
sc_ids   = st.session_state.scene_ids
labels   = st.session_state.labels

# -- search form --------------------------------------------------------------
with st.form("search_form"):
    col_q, col_top, col_btn = st.columns([4, 1, 1])
    with col_q:
        query = st.text_input(
            "Text query",
            placeholder='e.g. "person running", "night scene", "close-up face"',
            label_visibility="collapsed",
        )
    with col_top:
        top_k = st.number_input("Top K", min_value=1,
                                 max_value=len(b64s), value=min(8, len(b64s)), step=1)
    with col_btn:
        submitted = st.form_submit_button("Search", type="primary",
                                           use_container_width=True)

with st.expander("Display options"):
    col1, col2, col3 = st.columns(3)
    with col1:
        n_cols = st.slider("Columns", 2, 8, 4, key="search_cols")
    with col2:
        show_score = st.toggle("Show score", value=True)
    with col3:
        show_cluster = st.toggle("Show cluster", value=(labels is not None))
        if show_cluster and labels is None:
            st.caption("Run clustering first.")
            show_cluster = False

st.divider()

# -- search -------------------------------------------------------------------
if submitted and query.strip():
    with st.spinner(f'Searching: "{query}"...'):
        txt_vec = compute_text_embedding(query.strip(), model, proc, device)
        scores  = cosine_scores(emb, txt_vec)

    top_idxs   = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_idxs]
    max_score  = top_scores[0] if top_scores[0] > 0 else 1.0

    st.markdown(f"**Top {top_k} results for:** _{query}_")

    cols = st.columns(n_cols)
    for rank, (idx, score) in enumerate(zip(top_idxs, top_scores)):
        with cols[rank % n_cols]:
            pil = b64_to_pil(b64s[idx])
            st.image(pil, use_container_width=True)

            caption_lines = [f"#{rank+1}  {fmt_ts(ts[idx])}"]
            if sc_ids:
                caption_lines.append(f"scene {sc_ids[idx]}")
            if show_score:
                pct = score / max_score * 100
                caption_lines.append(f"score {score:.3f} ({pct:.0f}%)")
            if show_cluster and labels is not None:
                c    = int(labels[idx])
                name = "Noise" if c == -1 else f"Cluster {c}"
                col  = cluster_color(c)
                caption_lines.append(f"<span style='color:{col}'>{name}</span>")

            if show_score:
                bar_pct = score / max_score * 100
                st.markdown(
                    f'<div style="height:3px;background:#2e3250;border-radius:2px;margin-bottom:4px">'
                    f'<div style="width:{bar_pct:.0f}%;height:100%;'
                    f'background:linear-gradient(90deg,#7c6ff7,#5ee7c4);border-radius:2px"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            caption_html = "<br>".join(caption_lines)
            st.markdown(
                f'<div style="font-size:.68rem;color:#8b90b8;text-align:center;line-height:1.5">'
                f'{caption_html}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    with st.expander("Score distribution (all frames)"):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores, nbinsx=50,
                                    marker_color="#7c6ff7", opacity=0.8))
        for rank_i, (i, s) in enumerate(zip(top_idxs[:top_k], top_scores)):
            fig.add_vline(x=s, line_dash="dot", line_color="#5ee7c4",
                          annotation_text=f"#{rank_i+1}",
                          annotation_position="top")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117",
            height=300, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Cosine similarity score",
            yaxis_title="Frame count",
        )
        st.plotly_chart(fig, use_container_width=True)

elif not submitted:
    st.info("Enter a description and click Search.")

# -- search history -----------------------------------------------------------
if "search_history" not in st.session_state:
    st.session_state.search_history = []

if submitted and query.strip():
    entry = {"query": query.strip(), "top_k": top_k}
    if entry not in st.session_state.search_history:
        st.session_state.search_history.insert(0, entry)
        st.session_state.search_history = st.session_state.search_history[:10]

if st.session_state.search_history:
    with st.sidebar:
        st.subheader("Recent searches")
        for h in st.session_state.search_history:
            st.caption(f"{h['query']}  (top {h['top_k']})")
