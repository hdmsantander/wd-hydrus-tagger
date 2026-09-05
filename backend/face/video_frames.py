"""Extract evenly spaced frames from video bytes for face detection."""

from __future__ import annotations

import tempfile
from io import BytesIO

import numpy as np
from PIL import Image


def extract_video_frames(video_bytes: bytes, num_frames: int = 30) -> list[Image.Image]:
    """Return RGB PIL images sampled from a video blob."""
    import cv2

    frames: list[Image.Image] = []
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return frames
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
        cap.release()
    return frames


def pil_rgb_to_bgr(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
