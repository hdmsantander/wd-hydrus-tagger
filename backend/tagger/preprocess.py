"""Image preprocessing for WD v3 tagger models."""

import numpy as np
from PIL import Image


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB, compositing RGBA onto white background."""
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert("RGB")


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad image to a square with white background, centered."""
    w, h = image.size
    max_dim = max(w, h)
    if w == h:
        return image
    canvas = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    canvas.paste(image, (offset_x, offset_y))
    return canvas


def preprocess_image(image: Image.Image, target_size: int = 448) -> np.ndarray:
    """Preprocess a single image for WD v3 model inference.

    Returns numpy array of shape (target_size, target_size, 3) in BGR float32 [0,1].
    """
    image = ensure_rgb(image)
    image = pad_to_square(image)
    image = image.resize((target_size, target_size), Image.LANCZOS)

    # WD v3 ONNX models expect pixel values in [0, 255] float32, BGR channel order
    arr = np.array(image, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB to BGR
    return arr


def preprocess_batch(images: list[Image.Image], target_size: int = 448) -> np.ndarray:
    """Preprocess a batch of images.

    Returns numpy array of shape (N, target_size, target_size, 3).
    """
    arrays = [preprocess_image(img, target_size) for img in images]
    return np.stack(arrays, axis=0)
