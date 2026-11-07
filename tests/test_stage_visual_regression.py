from __future__ import annotations

import json

import torch

from stereo_runtime.stage_visual_regression import (
    StageVisualRegressionConfig,
    _json_safe,
)


def test_stage_visual_regression_config_keeps_runtime_defaults() -> None:
    config = StageVisualRegressionConfig()

    assert config.depth_strength == 0.48
    assert config.layers == 2
    assert config.hole_fill_mode == "balanced"


def test_stage_visual_regression_json_safe_describes_tensors() -> None:
    payload = _json_safe({"mask": torch.zeros(1, 1, 4, 8), "value": 0.25})

    encoded = json.dumps(payload)

    assert "[1, 1, 4, 8]" in encoded
    assert payload["value"] == 0.25
