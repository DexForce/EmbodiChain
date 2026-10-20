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

"""CPU coverage for cuRobo's explicit active-joint motion budgets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
    CuroboPlanner,
    CuroboPlannerCfg,
)
from embodichain.lab.sim.motion.planners.curobo.curobo_yaml import _curobo_joint_limits


def _robot() -> SimpleNamespace:
    """Provide active-joint order differing from the full Robot state."""
    caps = torch.tensor([[2.0, 4.0, 0.8], [3.0, 3.0, 0.6]])
    return SimpleNamespace(
        joint_names=["leader", "second", "follower"],
        active_joint_ids=[1, 0],
        mimic_ids=[2],
        mimic_parents=[0],
        mimic_multipliers=[2.0],
        get_qvel_limits=lambda *, joint_ids: caps[:, joint_ids],
    )


def test_curobo_budget_intersection_preserves_active_joint_order() -> None:
    """Requests tighten speeds and assign acceleration in exported order."""
    actual = _curobo_joint_limits(
        _robot(), max_velocity=[1.5, 1.0], max_acceleration=[8.0, 6.0]
    )
    assert actual["max_velocity"] == pytest.approx([1.5, 0.3])
    assert actual["max_acceleration"] == [8.0, 6.0]


def test_curobo_motion_limits_invalidate_yaml_cache() -> None:
    """Cached URDF/YAML must change when a requested budget changes."""
    planner = object.__new__(CuroboPlanner)
    planner.robot = _robot()
    planner.cfg = CuroboPlannerCfg()
    original = planner._robot_yaml_cache_key(
        "missing.urdf", "arm", None, planner.cfg.auto_gen
    )
    planner.cfg.max_velocity = [1.5, 1.0]
    slower = planner._robot_yaml_cache_key(
        "missing.urdf", "arm", None, planner.cfg.auto_gen
    )
    planner.cfg.max_acceleration = [8.0, 6.0]
    acceleration = planner._robot_yaml_cache_key(
        "missing.urdf", "arm", None, planner.cfg.auto_gen
    )
    assert len({original, slower, acceleration}) == 3


@pytest.mark.parametrize("limit_request", [[1.0], [1.0, -2.0], float("inf")])
def test_curobo_invalid_velocity_request_fails_before_export(
    limit_request: object,
) -> None:
    """Bad budgets must not silently fall back to asset caps."""
    with pytest.raises(ValueError):
        _curobo_joint_limits(_robot(), max_velocity=limit_request)


def test_placeholder_runtime_caps_fall_back_to_assembled_urdf(tmp_path) -> None:
    """Default GPU drive placeholders must not discard authored joint speeds."""
    from embodichain.lab.sim.motion.planners._joint_limits import _joint_velocity_limits

    path = tmp_path / "assembled.urdf"
    path.write_text(
        '<robot name="test"><joint name="leader"><limit velocity="2.0"/></joint><joint name="second"><limit velocity="3.0"/></joint><joint name="follower"><limit velocity="0.8"/></joint></robot>'
    )
    robot = _robot()
    robot.cfg = SimpleNamespace(fpath=str(path))
    caps = torch.tensor([[float("inf"), 0.0, float("inf")], [0.25, 4.0, float("inf")]])
    robot.get_qvel_limits = lambda *, joint_ids: caps[:, joint_ids]
    actual = _joint_velocity_limits(robot)
    torch.testing.assert_close(
        actual, torch.tensor([[0.4, 0.0, 0.8], [0.25, 4.0, 0.8]], dtype=torch.float64)
    )
    exported = _curobo_joint_limits(robot, max_acceleration=10.0)
    assert exported["max_velocity"] == pytest.approx([0.0, 0.25])


def test_authored_speed_fallback_cache_tracks_asset_revision(tmp_path) -> None:
    """Editing the assembled asset cannot reuse a stale authored velocity."""
    from embodichain.lab.sim.motion.planners._joint_limits import _joint_velocity_limits

    path = tmp_path / "assembled.urdf"
    robot = _robot()
    robot.cfg = SimpleNamespace(fpath=str(path))
    robot.mimic_ids = robot.mimic_parents = robot.mimic_multipliers = []
    robot.get_qvel_limits = lambda *, joint_ids: torch.full((1, 3), torch.inf)[
        :, joint_ids
    ]
    path.write_text('<robot><joint name="leader"><limit velocity="2"/></joint></robot>')
    assert _joint_velocity_limits(robot)[0, 0] == 2.0
    path.write_text(
        '<robot><joint name="leader"><limit velocity="0.75"/></joint></robot>'
    )
    assert _joint_velocity_limits(robot)[0, 0] == 0.75
