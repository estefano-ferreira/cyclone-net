"""model.input_channels must agree with input_channels_names.

Fails if: a mismatched explicit count reaches model construction and only
surfaces later as an opaque conv-shape error at the first batch.
"""
from __future__ import annotations

import pytest

from src.training.trainer import _build_model


def _cfg(count, names):
    return {"model": {"input_channels": count, "input_channels_names": names}}


def test_mismatched_input_channels_fails_loudly():
    with pytest.raises(ValueError, match="input_channels .* != len"):
        _build_model(_cfg(9, ["a", "b", "c"]))


def test_matching_input_channels_builds():
    model = _build_model(_cfg(3, ["a", "b", "c"]))
    assert model is not None
