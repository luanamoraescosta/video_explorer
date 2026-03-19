# video_explorer

A Streamlit app for exploring and clustering video content using deep learning embeddings.

## What is this?

This tool lets you load a video, extract frames, and explore them visually — using UMAP for dimensionality reduction and HDBSCAN for clustering. The idea is to get a bird's-eye view of what's happening across a video without watching it frame by frame.

## Try it online

The app is deployed on Streamlit Community Cloud:

👉 **https://luanamoraescosta-video-explorer-app-jgy4zk.streamlit.app/**

> ⚠️ **The hosted version is unstable.** Streamlit Community Cloud has memory and resource limits that can cause the app to crash or time out, especially with heavier models and video files. If you run into issues, I recommend running it locally (see below).

## Run locally (recommended)

**1. Clone the repository**

```bash
git clone https://github.com/luanamoraescosta/video_explorer.git
cd video_explorer
```

**2. Create a virtual environment**

```bash
python -m venv env
source env/bin/activate        # macOS / Linux
env\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> ⚠️ This installs PyTorch, Transformers, OpenCV, UMAP, and HDBSCAN. It may take a few minutes. A machine with a GPU is recommended but not required.

**4. Run the app**

```bash
streamlit run app.py
```

## Tech stack

| Library | Role |
|---|---|
| `streamlit` | UI and app framework |
| `opencv-python-headless` | Video frame extraction |
| `torch` + `torchvision` | Deep learning backbone |
| `transformers` | Feature extraction models |
| `umap-learn` | Dimensionality reduction |
| `hdbscan` | Density-based clustering |
| `plotly` | Interactive visualizations |
| `scikit-learn` | Supporting utilities |

## Status

The project is under active development. Features and structure may change.
