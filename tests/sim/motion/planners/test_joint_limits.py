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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.objects.robot import Robot

from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners._joint_limits import _resolve_joint_constraints
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
    CuroboPlanner,
    CuroboPlannerCfg,
)
from embodichain.lab.sim.motion.planners.curobo.curobo_yaml import _curobo_joint_limits
from embodichain.lab.sim.motion.planners.toppra_planner import (
    ToppraPlanner,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
    TrapezoidalPlanner,
    TrapezoidalPlannerCfg,
)
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanState

pytestmark = pytest.mark.no_sim


class _Robot:
    get_joint_motion_limits = Robot.get_joint_motion_limits

    def __init__(self) -> None:
        self.joint_names = ["shoulder", "finger", "elbow"]
        self.control_parts = {"arm": ["elbow", "shoulder"], "eef": ["finger"]}
        self.cfg = SimpleNamespace(fpath="", joint_acceleration_limits=None)
        self.velocities = torch.tensor(
            [[0.8, 0.03, 1.6], [0.6, 0.02, 1.2]], dtype=torch.float64
        )
        self.num_instances = 2
        self.device = torch.device("cpu")

    def get_joint_ids(self, name: str) -> list[int]:
        return [self.joint_names.index(joint) for joint in self.control_parts[name]]

    def get_qvel_limits(self, *, joint_ids: list[int]) -> torch.Tensor:
        return self.velocities[:, joint_ids]


def test_part_limits_follow_waypoint_order_and_strictest_environment() -> None:
    robot = _Robot()
    robot.cfg.joint_acceleration_limits = {"elbow": 0.9, "shoulder": 1.4}
    constraints = {"velocity": None, "acceleration": None}
    resolved = _resolve_joint_constraints(robot, constraints, control_part="arm", dof=2)
    torch.testing.assert_close(
        resolved["velocity"], torch.tensor([1.2, 0.6], dtype=torch.float64)
    )
    torch.testing.assert_close(
        resolved["acceleration"], torch.tensor([0.9, 1.4], dtype=torch.float64)
    )
    assert constraints == {"velocity": None, "acceleration": None}


def test_single_row_planning_does_not_read_other_environment_caps() -> None:
    resolved = _resolve_joint_constraints(
        _Robot(), {}, control_part="arm", dof=2, batch_size=1
    )
    torch.testing.assert_close(
        resolved["velocity"], torch.tensor([1.6, 0.8], dtype=torch.float64)
    )


def test_asset_caps_survive_unbounded_drive_overlays(tmp_path: Path) -> None:
    robot = _Robot()
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="test"><joint name="finger" type="prismatic"><limit velocity="0.05"/></joint></robot>'
    )
    robot.cfg.fpath = str(path)
    robot.velocities[:, 1] = 1e10
    result = _resolve_joint_constraints(
        robot, {"velocity": 1.0}, control_part="eef", dof=1
    )
    assert result["velocity"].item() == pytest.approx(0.05)
    path.write_text(path.read_text().replace("0.05", "0.04"))
    result = _resolve_joint_constraints(robot, {}, control_part="eef", dof=1)
    assert result["velocity"].item() == pytest.approx(0.04)


@pytest.mark.parametrize("velocity_request", [None, 0.4])
def test_missing_velocity_metadata_uses_request_or_compatibility_fallback(
    velocity_request: float | None,
) -> None:
    robot = _Robot()
    robot.velocities[:] = torch.nan
    result = _resolve_joint_constraints(robot, {"velocity": velocity_request}, dof=3)
    assert (
        result["velocity"].tolist()
        == [0.2 if velocity_request is None else velocity_request] * 3
    )
    assert result["acceleration"].tolist() == [0.5] * 3


def test_explicit_limits_cannot_exceed_available_robot_caps() -> None:
    robot = _Robot()
    robot.cfg.joint_acceleration_limits = {"arm": 1.0}
    result = _resolve_joint_constraints(
        robot, {"velocity": [5.0, 0.1], "acceleration": 3.0}, control_part="arm", dof=2
    )
    assert result["velocity"].tolist() == [1.2, 0.1]
    assert result["acceleration"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("velocity_scale", [0.5, 2.0])
def test_constraint_check_resolves_default_robot_limits(velocity_scale: float) -> None:
    planner = TrapezoidalPlanner.__new__(TrapezoidalPlanner)
    planner.robot = _Robot()
    velocities = (
        torch.tensor([1.2, 0.6]).reshape(1, 1, 2).expand(2, 5, 2) * velocity_scale
    )
    assert planner.is_satisfied_constraint(
        velocities,
        torch.zeros_like(velocities),
        planner.default_plan_options().constraints,
        control_part="arm",
    ) is (velocity_scale <= 1.0)


def test_partial_acceleration_metadata_keeps_explicit_request_for_other_joints() -> (
    None
):
    robot = _Robot()
    robot.cfg.joint_acceleration_limits = {"shoulder": 0.3}
    result = _resolve_joint_constraints(
        robot, {"acceleration": 1.5}, control_part="arm", dof=2
    )
    assert result["acceleration"].tolist() == [1.5, 0.3]


def test_signed_toppra_requests_keep_asymmetric_bounds() -> None:
    result = _resolve_joint_constraints(
        _Robot(), {"velocity": [[-0.7, 3.0], [-2.0, 0.4]]}, control_part="arm", dof=2
    )
    assert result["velocity"].tolist() == [[-0.7, 1.2], [-0.6, 0.4]]


def test_ambiguous_control_part_requires_explicit_joint_mapping() -> None:
    robot = _Robot()
    robot.control_parts["other_arm"] = ["shoulder", "finger"]
    with pytest.raises(ValueError, match="Set control_part"):
        _resolve_joint_constraints(robot, {}, dof=2)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_acceleration_metadata_fails_before_planning(value: float) -> None:
    robot = _Robot()
    robot.cfg.joint_acceleration_limits = value
    with pytest.raises(ValueError, match="finite positive"):
        _resolve_joint_constraints(robot, {}, dof=3)


def test_overlapping_acceleration_rules_are_rejected() -> None:
    robot = _Robot()
    robot.cfg.joint_acceleration_limits = {"arm": 1.0, "should.*": 0.5}
    with pytest.raises(ValueError, match="Multiple matches"):
        _resolve_joint_constraints(robot, {}, dof=3)


@pytest.mark.parametrize("planner_type", ["toppra", "trapezoidal"])
def test_facade_and_direct_plans_apply_robot_limits(planner_type: str) -> None:
    robot = _Robot()
    robot.cfg.joint_acceleration_limits = {"elbow": 0.9, "shoulder": 1.4}
    planner_class, cfg_class = (
        (ToppraPlanner, ToppraPlannerCfg)
        if planner_type == "toppra"
        else (TrapezoidalPlanner, TrapezoidalPlannerCfg)
    )
    planner = planner_class.__new__(planner_class)
    planner.robot = robot
    planner.device = robot.device
    planner.cfg = cfg_class(robot_uid="robot")
    if planner_type == "toppra":
        planner.cfg.max_workers = 1
        planner._pool = None
    generator = MotionGenerator.__new__(MotionGenerator)
    generator.planner = planner
    states = [
        PlanState.from_qpos(torch.zeros(2, 2)),
        PlanState.from_qpos(torch.tensor([[2.0, 1.0], [1.5, 1.0]])),
    ]
    for options in (
        planner.default_plan_options(),
        generator.resolve_plan_options(None, sample_count=101),
    ):
        options = planner.with_motion_context(
            options, start_qpos=None, control_part="arm"
        )
        result = planner.plan(states, options)
        assert result.is_all_success()
        assert result.positions.shape[1] in (100, 101)
        assert bool(
            (
                result.velocities.abs().amax(dim=(0, 1))
                <= torch.tensor([1.2, 0.6]) * 1.03
            ).all()
        )
        assert bool(
            (
                result.accelerations.abs().amax(dim=(0, 1))
                <= torch.tensor([0.9, 1.4]) * 1.03
            ).all()
        )
        assert options.constraints["velocity"] is None


@pytest.mark.parametrize("override_path", [False, True])
def test_curobo_export_uses_effective_velocity_and_acceleration_caps(
    tmp_path: Path,
    override_path: bool,
) -> None:
    robot = _Robot()
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="test"><joint name="elbow"><limit velocity="3"/></joint><joint name="shoulder"><limit velocity="2"/></joint><joint name="finger"><limit velocity="0.04"/></joint></robot>'
    )
    if not override_path:
        robot.cfg.fpath = str(path)
    robot.cfg.joint_acceleration_limits = {"arm": 1.5}
    exported = _curobo_joint_limits(
        robot, urdf_path=str(path) if override_path else None
    )
    assert exported["velocity_scale"] == pytest.approx([0.3, 0.5, 0.4])
    assert exported["max_acceleration"] == [1.5, 0.5, 1.5]


def test_toppra_spawn_workers_receive_resolved_joint_limits() -> None:
    robot = _Robot()
    planner = ToppraPlanner.__new__(ToppraPlanner)
    planner.robot = robot
    planner.device = robot.device
    planner.cfg = ToppraPlannerCfg(robot_uid="robot", max_workers=2, mp_context="spawn")
    planner._pool = None
    planner._mp_context = "spawn"
    options = planner.default_plan_options()
    options.control_part = "arm"
    try:
        result = planner.plan(
            [
                PlanState.from_qpos(torch.zeros(2, 2)),
                PlanState.from_qpos(torch.ones(2, 2)),
            ],
            options,
        )
        assert result.is_all_success()
        assert result.positions.shape == (2, 100, 2)
        assert bool(
            (
                result.velocities.abs().amax(dim=(0, 1))
                <= torch.tensor([1.2, 0.6]) * 1.03
            ).all()
        )
        assert planner._pool is not None
    finally:
        planner.close()


def test_mimic_follower_caps_constrain_curobo_active_joint(tmp_path: Path) -> None:
    robot = _Robot()
    robot.velocities[:, 1] = 1.0
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="test"><joint name="shoulder"><limit velocity="2"/></joint><joint name="finger"><limit velocity="0.2"/><mimic joint="shoulder" multiplier="-2"/></joint><joint name="elbow"><limit velocity="3"/></joint></robot>'
    )
    robot.cfg.fpath = str(path)
    exported = _curobo_joint_limits(robot)
    assert exported["velocity_scale"] == pytest.approx([0.05, 0.4])
    assert len(exported["max_acceleration"]) == 2


def test_mimic_runtime_and_acceleration_overrides_constrain_leader(
    tmp_path: Path,
) -> None:
    robot = _Robot()
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="test"><joint name="finger"><mimic joint="shoulder" multiplier="-2"/></joint></robot>'
    )
    robot.cfg.fpath = str(path)
    robot.cfg.joint_acceleration_limits = {"finger": 0.2}
    result = _resolve_joint_constraints(robot, {}, control_part="arm", dof=2)
    assert result["velocity"].tolist() == pytest.approx([1.2, 0.01])
    assert result["acceleration"].tolist() == pytest.approx([0.5, 0.1])


def test_curobo_cache_changes_with_robot_metadata_and_closes_stale_backend() -> None:
    robot = _Robot()
    planner = CuroboPlanner.__new__(CuroboPlanner)
    planner.robot = robot
    planner.cfg = CuroboPlannerCfg(robot_uid="robot", use_cuda_graph=False)
    key = planner._robot_yaml_cache_key("", "arm", None, planner.cfg.auto_gen)
    cached = SimpleNamespace(motion_limits_key=key, planner=Mock())
    backend_key = ("arm", 2, False, MoveType.EEF_MOVE)
    planner._backend_cache = {backend_key: cached}
    assert planner._get_backend("arm", 2) is cached
    robot.cfg.joint_acceleration_limits = 0.2
    assert planner._robot_yaml_cache_key("", "arm", None, planner.cfg.auto_gen) != key
    planner._materialize_profile = Mock(side_effect=RuntimeError("rebuild backend"))
    with pytest.raises(RuntimeError, match="rebuild backend"):
        planner._get_backend("arm", 2)
    cached.planner.close.assert_called_once()
    assert backend_key not in planner._backend_cache
