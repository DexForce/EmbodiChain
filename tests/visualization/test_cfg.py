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

import pytest

from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg


def test_visualization_cfg_copies_environment_selection() -> None:
    env_ids = [0, 1]

    cfg = VisualizationCfg(backend="viser", env_ids=env_ids)
    env_ids.append(2)

    assert cfg.env_ids == [0, 1]


def test_visualization_cfg_enforces_visible_environment_limit() -> None:
    with pytest.raises(ValueError, match="exceeding max_visible_envs"):
        VisualizationCfg(backend="viser", env_ids=[0, 1], max_visible_envs=1)


def test_visualization_cfg_accepts_step_synchronized_images() -> None:
    cfg = VisualizationCfg(backend="viser", sensor_image_fps=None)

    assert cfg.sensor_image_fps is None


def test_visualization_cfg_rejects_non_positive_image_fps() -> None:
    with pytest.raises(ValueError, match="sensor_image_fps"):
        VisualizationCfg(backend="viser", sensor_image_fps=0.0)


def test_visualization_cfg_rejects_mutating_commands_in_phase_v0() -> None:
    with pytest.raises(ValueError, match="read-only"):
        VisualizationCfg(backend="viser", allow_commands=True)


def test_viser_server_cfg_requires_bindable_port() -> None:
    with pytest.raises(ValueError, match="between 1 and 65535"):
        ViserServerCfg(port=0)


def test_visualization_cfg_owns_independent_viser_server_settings() -> None:
    first = VisualizationCfg()
    second = VisualizationCfg()

    first.viser_server.port = 9000

    assert first.viser_server.port == 9000
    assert second.viser_server.port == 8080
