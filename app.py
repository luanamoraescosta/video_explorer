"""
Video CLIP Explorer -- main entry point
Run with: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from core.ui_helpers import init_session, apply_global_css

st.set_page_config(
    page_title="Video CLIP Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()
apply_global_css()

st.title("Video CLIP Explorer")
st.markdown("""
Analyse videos with CLIP embeddings, semantic clustering, and natural language search.

### Workflow

| Step | Page | Description |
|------|------|-------------|
| 1 | **Frame Extraction** | Scene detection via PySceneDetect or fixed-interval sampling. Saves frames to cache. |
| 2 | **Embeddings** | Computes CLIP feature vectors for each frame. Reuses cache on subsequent runs. |
| 3 | **Clustering** | UMAP dimensionality reduction followed by HDBSCAN. Visualises scatter and cluster exemplars. |
| 4 | **Semantic Search** | Ranks frames by cosine similarity with a free-text CLIP query. |
| 5 | **Grad-ECLIP** | Zhao et al. 2024 — gradient-based visual explanation heatmap for any frame + text pair. |

---
Use the sidebar to navigate between pages.
Start with **Frame Extraction**.
""")
