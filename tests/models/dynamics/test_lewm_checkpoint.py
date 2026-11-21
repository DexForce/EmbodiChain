# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Checkpoint compatibility across the two Transformers ViT layouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from embodichain.extensions.models.dynamics.lewm import (
    LeWM,
    LeWMCfg,
    export_lewm,
    load_lewm,
)


def test_load_lewm_accepts_older_and_newer_vit_projection_keys(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    model = LeWM(
        LeWMCfg(
            image_size=16,
            patch_size=8,
            embed_dim=16,
            vision_depth=1,
            vision_heads=2,
            predictor_depth=1,
            predictor_heads=2,
            predictor_head_dim=8,
            mlp_dim=32,
        )
    )
    export_lewm(tmp_path, model, {})
    expected = model.state_dict()
    state = dict(expected)
    old_key = "encoder.encoder.layer.0.attention.attention.query.weight"
    new_key = "encoder.layers.0.attention.q_proj.weight"
    source, target = (old_key, new_key) if old_key in state else (new_key, old_key)
    state[target] = state.pop(source)

    weights = tmp_path / "weights.pt"
    torch.save(state, weights)
    manifest = tmp_path / "manifest.json"
    info = json.loads(manifest.read_text())
    info["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(info))

    restored, _ = load_lewm(tmp_path)
    torch.testing.assert_close(restored.state_dict(), expected)
