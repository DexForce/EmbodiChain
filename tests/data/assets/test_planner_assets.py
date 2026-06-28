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
from __future__ import annotations

import os

import pytest

from embodichain.data.assets import planner_assets


def _raise_if_downloaded(*args, **kwargs):
    raise AssertionError("hf_hub_download should not be called")


def test_neural_planner_checkpoint_uses_explicit_local_path(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "franka.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(planner_assets, "hf_hub_download", _raise_if_downloaded)

    resolved_path = planner_assets.download_neural_planner_checkpoint(
        checkpoint_path=checkpoint_path
    )

    assert resolved_path == os.path.abspath(str(checkpoint_path))


def test_neural_planner_checkpoint_uses_env_local_path(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "franka.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setenv(
        planner_assets.NEURAL_PLANNER_CHECKPOINT_ENV,
        str(checkpoint_path),
    )
    monkeypatch.setattr(planner_assets, "hf_hub_download", _raise_if_downloaded)

    resolved_path = planner_assets.download_neural_planner_checkpoint()

    assert resolved_path == os.path.abspath(str(checkpoint_path))


def test_neural_planner_checkpoint_uses_default_data_root(tmp_path, monkeypatch):
    checkpoint_path = (
        tmp_path
        / "checkpoints"
        / "dexforce"
        / "neural_motion_generator"
        / "franka"
        / "franka.pt"
    )
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.delenv(planner_assets.NEURAL_PLANNER_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(planner_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(planner_assets, "hf_hub_download", _raise_if_downloaded)

    resolved_path = planner_assets.download_neural_planner_checkpoint()

    assert resolved_path == str(checkpoint_path)


def test_neural_planner_checkpoint_rejects_missing_explicit_path(tmp_path):
    checkpoint_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="NeuralPlanner checkpoint not found"):
        planner_assets.download_neural_planner_checkpoint(
            checkpoint_path=checkpoint_path
        )
