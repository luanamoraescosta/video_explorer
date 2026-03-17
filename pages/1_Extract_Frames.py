"""
Page 1 -- Frame Extraction
Works both locally (path input) and on Streamlit Cloud (file uploader).
Uploaded files are written to a temp file so OpenCV and PySceneDetect
can open them by path; the temp file persists for the session.
"""

import sys
import io
import os
import hashlib
import tempfile
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from core.ui_helpers import (
    init_session, apply_global_css, frame_grid, b64_to_pil,
)
from core.extraction import extract_scenes, extract_interval, video_info
from core import cache

st.set_page_config(page_title="Frame Extraction", layout="wide")
init_session()
apply_global_css()

st.title("Frame Extraction")

# ── video input ────────────────────────────────────────────────────────────────
st.subheader("1. Video source")

input_mode = st.radio(
    "Input mode",
    ["Upload file", "Local path"],
    horizontal=True,
    help="Use 'Upload file' on Streamlit Cloud. Use 'Local path' when running locally.",
)

video_path = None

if input_mode == "Upload file":
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        help="Large files (> 200 MB) require the maxUploadSize setting in .streamlit/config.toml.",
    )

    if uploaded is not None:
        # Compute a short hash of the filename + size to detect re-uploads.
        file_id = hashlib.md5(
            f"{uploaded.name}_{uploaded.size}".encode()
        ).hexdigest()[:12]

        # Write to a stable temp path so we don't re-write on every rerun.
        tmp_dir  = Path(tempfile.gettempdir()) / "clip_explorer_uploads"
        tmp_dir.mkdir(exist_ok=True)
        suffix   = Path(uploaded.name).suffix
        tmp_path = tmp_dir / f"{file_id}{suffix}"

        if not tmp_path.exists():
            with st.spinner(f"Saving '{uploaded.name}' to temp storage..."):
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
            st.success(f"Saved to temp: {tmp_path.name}")
        else:
            st.info(f"Using cached temp file for '{uploaded.name}'.")

        video_path = str(tmp_path)
        st.session_state.video_path = video_path
        st.caption(f"Temp path: `{video_path}`")

else:
    path_input = st.text_input(
        "Absolute path to video file",
        value=st.session_state.video_path or "",
        placeholder="/home/user/videos/my_video.mp4",
    )
    if path_input:
        if Path(path_input).exists():
            video_path = path_input
            st.session_state.video_path = video_path
        else:
            st.error("File not found.")

# ── video info ────────────────────────────────────────────────────────────────
if video_path and Path(video_path).exists():
    info = video_info(video_path)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",     f"{info['duration']:.1f}s")
    c2.metric("FPS",          f"{info['fps']:.1f}")
    c3.metric("Resolution",   f"{info['width']}x{info['height']}")
    c4.metric("Total frames", info["total_frames"])

st.divider()

# ── extraction mode ────────────────────────────────────────────────────────────
st.subheader("2. Extraction mode")

mode = st.radio(
    "Mode",
    ["scene", "interval"],
    format_func=lambda x: "Scene detection (PySceneDetect)" if x == "scene"
                          else "Fixed interval",
    horizontal=True,
)
st.session_state.mode = mode

col1, col2 = st.columns(2)
if mode == "scene":
    with col1:
        threshold = st.slider(
            "Cut threshold",
            min_value=5.0, max_value=60.0, value=27.0, step=1.0,
            help="Lower = more sensitive to cuts. 27 is a good default.",
        )
    with col2:
        min_scene_len = st.number_input(
            "Minimum frames per scene", min_value=5, max_value=120, value=15,
        )
    param = threshold
    st.caption("3 frames are extracted per scene (start, middle, end).")
else:
    with col1:
        interval_sec = st.slider(
            "Interval (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5,
        )
    param = interval_sec
st.session_state.param = param

thumb_size_label = st.select_slider(
    "Thumbnail resolution",
    options=["160x90", "320x180", "480x270", "640x360"],
    value="320x180",
)
thumb_w, thumb_h = [int(x) for x in thumb_size_label.split("x")]

st.divider()

# ── cache status ───────────────────────────────────────────────────────────────
has_cache = bool(video_path and cache.frames_cache_exists(video_path, mode, param))

if has_cache:
    st.success(f"Cache found for mode={mode}, param={param}. Load directly or re-extract.")

col_run, col_load = st.columns(2)
run_extract    = col_run.button("Run extraction", type="primary",
                                 disabled=not video_path)
load_cache_btn = col_load.button("Load from cache", disabled=not has_cache)

# ── execution ──────────────────────────────────────────────────────────────────
if load_cache_btn and has_cache:
    result = cache.load_frames(video_path, mode, param)
    if result:
        emb, ts, b64, sc = result
        st.session_state.embeddings = emb
        st.session_state.timestamps = ts
        st.session_state.frames_b64 = b64
        st.session_state.scene_ids  = sc
        st.session_state.frames_pil = None
        st.success(f"{len(ts)} frames loaded from cache.")

if run_extract and video_path:
    prog = st.progress(0, text="Starting...")

    def cb(done, total):
        prog.progress(done / max(total, 1), text=f"Processing {done}/{total}...")

    with st.spinner("Extracting frames..."):
        try:
            if mode == "scene":
                frames_pil, timestamps, frames_b64, scene_ids = extract_scenes(
                    video_path,
                    threshold=threshold,
                    min_scene_len=min_scene_len,
                    thumb_size=(thumb_w, thumb_h),
                    progress_cb=cb,
                )
            else:
                frames_pil, timestamps, frames_b64, scene_ids = extract_interval(
                    video_path,
                    interval_sec=interval_sec,
                    thumb_size=(thumb_w, thumb_h),
                    progress_cb=cb,
                )
        except Exception as e:
            st.error(f"Extraction error: {e}")
            st.stop()

    prog.progress(1.0, text="Done.")
    st.session_state.frames_pil = frames_pil
    st.session_state.timestamps = timestamps
    st.session_state.frames_b64 = frames_b64
    st.session_state.scene_ids  = scene_ids
    st.session_state.embeddings = None
    st.session_state.labels     = None
    st.session_state.coords_2d  = None
    cache.save_meta(video_path, {"mode": mode, "param": param,
                                  "n_frames": len(timestamps)})
    st.success(f"{len(timestamps)} frames extracted.")

# ── preview + download ─────────────────────────────────────────────────────────
if st.session_state.frames_b64:
    st.divider()
    n = len(st.session_state.frames_b64)
    st.subheader(f"Preview — {n} frames")

    col_sliders, col_download = st.columns([3, 1])
    with col_sliders:
        n_cols   = st.slider("Columns", 2, 8, 4, key="prev_cols")
        max_show = st.slider("Show up to", 8, min(200, n), min(24, n), step=8)

    with col_download:
        st.markdown("**Download frames as ZIP**")
        zip_format = st.radio("Format", ["JPEG", "PNG"], horizontal=True, key="zip_fmt")

        if st.button("Build ZIP", key="build_zip"):
            buf  = io.BytesIO()
            stem = Path(st.session_state.video_path).stem if st.session_state.video_path else "frames"
            ext  = zip_format.lower()

            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, (b64, ts, sc) in enumerate(zip(
                    st.session_state.frames_b64,
                    st.session_state.timestamps,
                    st.session_state.scene_ids or range(n),
                )):
                    pil    = b64_to_pil(b64)
                    m, s   = divmod(int(ts), 60)
                    ts_str = f"{m:02d}m{s:02d}s"
                    fname  = f"{stem}_scene{sc:04d}_t{ts_str}_f{i:04d}.{ext}"
                    ibuf   = io.BytesIO()
                    if zip_format == "JPEG":
                        pil.convert("RGB").save(ibuf, format="JPEG", quality=90)
                    else:
                        pil.save(ibuf, format="PNG")
                    zf.writestr(fname, ibuf.getvalue())

            buf.seek(0)
            st.session_state["_zip_bytes"] = buf.getvalue()
            st.session_state["_zip_name"]  = (
                f"{stem}_frames_{mode}_{param}.zip"
            )

        if "_zip_bytes" in st.session_state:
            st.download_button(
                label=f"Download  ({n} frames)",
                data=st.session_state["_zip_bytes"],
                file_name=st.session_state["_zip_name"],
                mime="application/zip",
                use_container_width=True,
            )

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
        if st.button("Delete and restart", use_container_width=True):
            from core.ui_helpers import clear_session
            clear_session()
            st.rerun()
