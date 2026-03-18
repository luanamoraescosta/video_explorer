"""
Page 1 -- Load Frames
Aceita uma pasta local (quando rodando localmente) ou um ZIP de imagens
(Streamlit Cloud). Carrega os frames diretamente na sessão, sem precisar
de um vídeo. As páginas 2-5 funcionam normalmente depois disso.
"""

import sys
import io
import base64
import zipfile
import hashlib
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image

from core.ui_helpers import (
    init_session, apply_global_css, frame_grid, pil_to_b64, clear_session,
)

st.set_page_config(page_title="Load Frames", layout="wide")
init_session()
apply_global_css()

st.title("Load Frames")
st.caption(
    "Load pre-extracted frames directly — no video needed. "
    "Supports JPEG, PNG, WEBP, BMP, TIFF."
)

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ── helpers ────────────────────────────────────────────────────────────────────

def _thumb(pil: Image.Image, size=(320, 180)) -> Image.Image:
    img = pil.convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    return img


def _sort_key(p: Path) -> tuple:
    import re
    parts = re.split(r"(\d+)", p.name)
    return tuple(int(x) if x.isdigit() else x.lower() for x in parts)


def _load_pil_list(paths: list) -> list:
    results = []
    for p in sorted(paths, key=_sort_key):
        try:
            img = Image.open(p)
            img.load()
            results.append((p, img))
        except Exception:
            pass
    return results


def _images_to_session(pairs: list, source_label: str,
                        thumb_w: int, thumb_h: int):
    frames_b64 = []
    timestamps  = []
    scene_ids   = []

    prog = st.progress(0, text="Loading images...")
    n = len(pairs)

    for i, (p, pil) in enumerate(pairs):
        thumb = _thumb(pil, (thumb_w, thumb_h))
        frames_b64.append(pil_to_b64(thumb))
        timestamps.append(float(i))
        scene_ids.append(i)
        prog.progress((i + 1) / n, text=f"{i+1}/{n}")

    prog.progress(1.0, text="Done.")

    st.session_state.frames_b64   = frames_b64
    st.session_state.timestamps   = timestamps
    st.session_state.scene_ids    = scene_ids
    st.session_state.frames_pil   = [p for _, p in pairs]
    st.session_state.embeddings   = None
    st.session_state.labels       = None
    st.session_state.coords_2d    = None
    st.session_state.video_path   = source_label
    st.session_state.mode         = "folder"
    st.session_state.param        = 0


# ── input mode ─────────────────────────────────────────────────────────────────

st.subheader("1. Image source")

input_mode = st.radio(
    "Input mode",
    ["Upload ZIP", "Local folder"],
    horizontal=True,
    help=(
        "**Upload ZIP**: compress your frames into a ZIP and upload. "
        "Works on Streamlit Cloud. "
        "**Local folder**: enter an absolute path when running locally."
    ),
)

thumb_size_label = st.select_slider(
    "Thumbnail resolution",
    options=["160x90", "320x180", "480x270", "640x360"],
    value="320x180",
)
thumb_w, thumb_h = [int(x) for x in thumb_size_label.split("x")]

st.divider()

# ── ZIP upload ─────────────────────────────────────────────────────────────────

if input_mode == "Upload ZIP":
    st.subheader("2. Upload a ZIP of images")
    st.caption(
        "Compress your frames folder into a ZIP (JPEG / PNG / WEBP). "
        "Nested folders are supported."
    )

    uploaded_zip = st.file_uploader(
        "ZIP file",
        type=["zip"],
        help="Files larger than ~500 MB may time out on the free Streamlit Cloud tier.",
    )

    if uploaded_zip is not None:
        zip_id = hashlib.md5(
            f"{uploaded_zip.name}_{uploaded_zip.size}".encode()
        ).hexdigest()[:12]

        if st.button("Load images from ZIP", type="primary"):
            tmp_dir = Path(tempfile.gettempdir()) / f"clip_zip_{zip_id}"
            tmp_dir.mkdir(exist_ok=True)

            with st.spinner("Extracting ZIP..."):
                zip_bytes = uploaded_zip.read()
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    image_members = [
                        m for m in zf.infolist()
                        if Path(m.filename).suffix.lower() in SUPPORTED
                        and not m.filename.startswith("__MACOSX")
                        and not Path(m.filename).name.startswith(".")
                    ]
                    if not image_members:
                        st.error("No supported image files found in the ZIP.")
                        st.stop()

                    paths = []
                    for m in image_members:
                        dest = tmp_dir / Path(m.filename).name
                        dest.write_bytes(zf.read(m.filename))
                        paths.append(dest)

            st.info(f"{len(paths)} image files found.")

            with st.spinner("Loading and thumbnailing..."):
                pairs = _load_pil_list(paths)

            if not pairs:
                st.error("Could not open any images.")
                st.stop()

            _images_to_session(pairs, f"zip:{zip_id}", thumb_w, thumb_h)
            st.success(f"{len(pairs)} frames loaded.")

# ── local folder ───────────────────────────────────────────────────────────────

else:
    st.subheader("2. Local image folder")

    folder_input = st.text_input(
        "Absolute path to folder",
        value="",
        placeholder="/home/user/frames/my_video",
    )

    recursive = st.checkbox("Include subfolders", value=False)

    if folder_input:
        folder = Path(folder_input)
        if not folder.exists() or not folder.is_dir():
            st.error("Folder not found.")
        else:
            pattern = "**/*" if recursive else "*"
            all_files = [
                p for p in folder.glob(pattern)
                if p.is_file() and p.suffix.lower() in SUPPORTED
            ]

            if not all_files:
                st.warning("No supported image files found in that folder.")
            else:
                st.info(f"{len(all_files)} image files found.")

                if st.button("Load images", type="primary"):
                    with st.spinner("Loading and thumbnailing..."):
                        pairs = _load_pil_list(all_files)

                    if not pairs:
                        st.error("Could not open any images.")
                        st.stop()

                    _images_to_session(pairs, str(folder), thumb_w, thumb_h)
                    st.success(f"{len(pairs)} frames loaded.")

# ── preview ────────────────────────────────────────────────────────────────────

if st.session_state.frames_b64:
    st.divider()
    n = len(st.session_state.frames_b64)
    st.subheader(f"Preview — {n} frames")

    col_sliders, _ = st.columns([3, 1])
    with col_sliders:
        n_cols   = st.slider("Columns", 2, 8, 4, key="prev_cols")
        max_show = st.slider("Show up to", 8, min(200, n), min(24, n), step=8)

    frame_grid(
        st.session_state.frames_b64,
        st.session_state.timestamps,
        st.session_state.scene_ids,
        n_cols=n_cols,
        max_show=max_show,
    )

    st.divider()
    col_next, col_clear = st.columns(2)
    with col_next:
        if st.button("Next: Generate Embeddings", type="primary",
                     use_container_width=True):
            st.switch_page("pages/2_Embeddings.py")
    with col_clear:
        if st.button("Clear and restart", use_container_width=True):
            clear_session()
            st.rerun()
