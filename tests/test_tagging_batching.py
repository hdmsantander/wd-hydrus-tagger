"""Inference batch clamp helper."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.tagging_shared import clamp_inference_batch


def test_clamp_inference_batch_uses_fallback_when_none():
    assert clamp_inference_batch(None, 5) == 5


def test_clamp_inference_batch_respects_override():
    assert clamp_inference_batch(3, 99) == 3


def test_clamp_inference_batch_caps_at_256():
    assert clamp_inference_batch(500, 4) == 256


def test_clamp_inference_batch_minimum_one():
    assert clamp_inference_batch(0, 8) == 1
