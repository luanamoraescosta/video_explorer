"""
Extração de frames:
  - modo 'scene'    : PySceneDetect → 3 frames por cena (início, meio, fim)
  - modo 'interval' : amostragem a cada N segundos
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── utilidades ───────────────────────────────────────────────

def _pil_to_b64(pil: Image.Image, size=(320, 180)) -> str:
    thumb = pil.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _read_frame(cap: cv2.VideoCapture, fidx: int) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": total,
        "duration": total / fps if fps else 0,
        "width": w,
        "height": h,
        "name": Path(video_path).name,
    }


# ── modo scene ───────────────────────────────────────────────

def extract_scenes(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    thumb_size: tuple = (320, 180),
    progress_cb=None,
) -> tuple[list, list, list, list]:
    """
    Usa ContentDetector do PySceneDetect.
    Retorna (frames_pil, timestamps, frames_b64, scene_ids).
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video   = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold,
                                         min_scene_len=min_scene_len))
    manager.detect_scenes(video, show_progress=False)
    scene_list = manager.get_scene_list()

    if not scene_list:
        scene_list = [(None, None)]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_pil, timestamps, frames_b64, scene_ids = [], [], [], []

    for sid, (start, end) in enumerate(scene_list):
        if start is None:
            f_start, f_end = 0, total - 1
        else:
            f_start = start.get_frames()
            f_end   = max(end.get_frames() - 1, f_start)

        # início, meio, fim
        for fidx in sorted(set([f_start, (f_start + f_end) // 2, f_end])):
            pil = _read_frame(cap, fidx)
            if pil is None:
                continue
            frames_pil.append(pil)
            timestamps.append(fidx / fps)
            frames_b64.append(_pil_to_b64(pil, thumb_size))
            scene_ids.append(sid)

        if progress_cb:
            progress_cb(sid + 1, len(scene_list))

    cap.release()
    return frames_pil, timestamps, frames_b64, scene_ids


# ── modo interval ────────────────────────────────────────────

def extract_interval(
    video_path: str,
    interval_sec: float = 2.0,
    thumb_size: tuple = (320, 180),
    progress_cb=None,
) -> tuple[list, list, list, list]:
    """
    Um frame a cada `interval_sec` segundos.
    scene_id = índice sequencial do frame (sem conceito de cena).
    """
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps * interval_sec))

    frames_pil, timestamps, frames_b64, scene_ids = [], [], [], []
    fidx  = 0
    count = 0
    n_expected = total // step + 1

    while True:
        pil = _read_frame(cap, fidx)
        if pil is None:
            break
        frames_pil.append(pil)
        timestamps.append(fidx / fps)
        frames_b64.append(_pil_to_b64(pil, thumb_size))
        scene_ids.append(count)
        fidx  += step
        count += 1
        if progress_cb:
            progress_cb(count, n_expected)

    cap.release()
    return frames_pil, timestamps, frames_b64, scene_ids
