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

"""Regression coverage for the affordance CLI deployment setup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from embodichain.cli import affordance


def test_precompute_disables_the_ur5_recorder_before_world_creation() -> None:
    """Offline labeling must keep the physics reset event and omit the video camera."""
    deployment = (
        Path(__file__).resolve().parents[2]
        / "embodichain/extensions/tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    with (
        patch(
            "gymnasium.make", side_effect=RuntimeError("scene configuration captured")
        ) as make,
        pytest.raises(RuntimeError, match="scene configuration captured"),
    ):
        affordance.main(
            [
                "precompute-grasp-cache",
                "--gym-config",
                str(deployment),
                "--pair",
                "eef",
                "cube",
                "cube_grasp",
            ]
        )

    cfg = make.call_args.kwargs["cfg"]
    assert cfg.sim_cfg.render_cfg.renderer == "no-render"
    assert cfg.enable_sensor is False
    assert cfg.events.record_camera is None
    assert (
        cfg.events.expansion_profile_reset.func.__name__
        == "apply_expansion_profile_reset"
    )
