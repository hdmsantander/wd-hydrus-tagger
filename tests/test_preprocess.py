"""Image preprocessing."""

import numpy as np
from PIL import Image

from backend.tagger.preprocess import ensure_rgb, pad_to_square, preprocess_batch, preprocess_image


def test_preprocess_image_shape_and_range():
    img = Image.new("RGB", (64, 32), color=(10, 20, 30))
    out = preprocess_image(img, target_size=48)
    assert out.shape == (48, 48, 3)
    assert out.dtype == np.float32
    assert out.max() <= 255.0


def test_preprocess_batch_stacks():
    images = [Image.new("RGB", (16, 16), color=(i, i, i)) for i in range(3)]
    batch = preprocess_batch(images, target_size=32)
    assert batch.shape == (3, 32, 32, 3)


def test_rgba_composites_on_white():
    rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    rgb = ensure_rgb(rgba)
    assert rgb.mode == "RGB"


def test_pad_to_square():
    img = Image.new("RGB", (20, 10), color=(0, 0, 0))
    sq = pad_to_square(img)
    assert sq.size[0] == sq.size[1] == 20
