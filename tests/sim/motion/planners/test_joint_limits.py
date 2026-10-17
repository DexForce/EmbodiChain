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
from embodichain.lab.sim.motion.planners.utils import (
    MoveType,
    PlanState,
    TrajectorySampleMethod,
)

pytestmark = pytest.mark.no_sim


class _Robot:
    mimic_ids = ()
    mimic_parents = ()
    mimic_multipliers = ()

    @property
    def active_joint_ids(self) -> list[int]:
        return [i for i in range(len(self.joint_names)) if i not in self.mimic_ids]

    def __init__(self) -> None:
        self.joint_names = ["shoulder", "finger", "elbow"]
        self.control_parts = {"arm": ["elbow", "shoulder"], "eef": ["finger"]}
        self.cfg = SimpleNamespace(fpath="")
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
    constraints = {"velocity": None, "acceleration": [0.9, 1.4]}
    resolved = _resolve_joint_constraints(robot, constraints, control_part="arm", dof=2)
    torch.testing.assert_close(
        resolved["velocity"], torch.tensor([1.2, 0.6], dtype=torch.float64)
    )
    torch.testing.assert_close(
        resolved["acceleration"], torch.tensor([0.9, 1.4], dtype=torch.float64)
    )
    assert constraints == {"velocity": None, "acceleration": [0.9, 1.4]}


def test_single_row_planning_does_not_read_other_environment_caps() -> None:
    resolved = _resolve_joint_constraints(
        _Robot(), {}, control_part="arm", dof=2, batch_size=1
    )
    torch.testing.assert_close(
        resolved["velocity"], torch.tensor([1.6, 0.8], dtype=torch.float64)
    )


@pytest.mark.parametrize("velocity_request", [None, 1.0])
def test_zero_speed_caps_survive_limit_resolution(
    velocity_request: float | None,
) -> None:
    robot = _Robot()
    robot.velocities[:, 2] = 0.0
    constraints = _resolve_joint_constraints(
        robot, {"velocity": velocity_request}, dof=3
    )
    assert constraints["velocity"][2].item() == 0.0


def _zero_speed_planner(planner_type: str) -> ToppraPlanner | TrapezoidalPlanner:
    planner_class, cfg_class = (
        (ToppraPlanner, ToppraPlannerCfg)
        if planner_type == "toppra"
        else (TrapezoidalPlanner, TrapezoidalPlannerCfg)
    )
    planner = planner_class.__new__(planner_class)
    planner.robot = _Robot()
    planner.robot.velocities[:, 2] = 0.0
    planner.device = torch.device("cpu")
    planner.cfg = cfg_class(robot_uid="robot")
    if planner_type == "toppra":
        planner.cfg.max_workers = 1
        planner._pool = None
    return planner


@pytest.mark.parametrize("planner_type", ["toppra", "trapezoidal"])
def test_zero_speed_joint_can_hold_while_another_joint_moves(planner_type: str) -> None:
    planner = _zero_speed_planner(planner_type)
    start = torch.tensor([[0.0, 0.01, 0.25]], dtype=torch.float64).repeat(2, 1)
    goal = start.clone()
    goal[:, 0] = 0.4
    options = planner.default_plan_options()
    options.sample_method = TrajectorySampleMethod.QUANTITY
    options.sample_interval = 101
    result = planner.plan(
        [PlanState.from_qpos(start), PlanState.from_qpos(goal)], options
    )
    assert result.is_all_success()
    assert result.positions.shape[1] == 101
    assert bool((result.positions[..., 2] == start[:, 2:3]).all())
    assert torch.count_nonzero(result.velocities[..., 2]) == 0
    if result.constraint_report is not None:
        assert result.constraint_report["velocity_limit"].tolist() == pytest.approx(
            [0.6, 0.6]
        )


@pytest.mark.parametrize("planner_type", ["toppra", "trapezoidal"])
@pytest.mark.parametrize("displacement", [0.1, 1e-7])
def test_zero_speed_joint_cannot_be_commanded_to_move(
    planner_type: str, displacement: float
) -> None:
    planner = _zero_speed_planner(planner_type)
    start = torch.zeros(2, 3, dtype=torch.float64)
    goal = start.clone()
    goal[:, 2] = displacement
    states = [PlanState.from_qpos(start), PlanState.from_qpos(goal)]
    options = planner.default_plan_options()
    options.sample_method = TrajectorySampleMethod.QUANTITY
    options.sample_interval = 101
    if planner_type == "trapezoidal":
        with pytest.raises(ValueError, match="zero velocity"):
            planner.plan(states, options)
    else:
        assert not planner.plan(states, options).success.any()


@pytest.mark.parametrize("planner_type", ["toppra", "trapezoidal"])
def test_all_zero_speed_joints_can_hold(planner_type: str) -> None:
    planner = _zero_speed_planner(planner_type)
    planner.robot.velocities[:] = 0.0
    start = torch.full((2, 3), 0.25, dtype=torch.float64)
    result = planner.plan([PlanState.from_qpos(start), PlanState.from_qpos(start)])
    assert result.is_all_success()
    assert bool((result.positions == 0.25).all())
    assert torch.count_nonzero(result.velocities) == 0
    if result.constraint_report is not None:
        assert result.constraint_report["velocity_limit"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("profile", ["trapezoidal", "double_s"])
def test_blended_profiles_preserve_zero_speed_joints(profile: str) -> None:
    planner = _zero_speed_planner("trapezoidal")
    options = planner.default_plan_options()
    options.profile = profile
    options.stop_at_waypoints = False
    options.blend_tolerance = 0.02
    options.sample_interval = 101
    options.constraints["jerk"] = 2.0
    states = [
        PlanState.from_qpos(torch.tensor([point], dtype=torch.float64).repeat(2, 1))
        for point in ([0.0, 0.0, 0.25], [0.2, 0.0, 0.25], [0.2, 0.05, 0.25])
    ]
    result = planner.plan(states, options)
    assert result.is_all_success()
    assert bool((result.positions[..., 2] == 0.25).all())
    assert torch.count_nonzero(result.velocities[..., 2]) == 0
    assert result.constraint_report["velocity_limit"].tolist() == pytest.approx(
        [0.6, 0.6]
    )
    assert bool(result.constraint_report["within_limits"].all())


@pytest.mark.parametrize("displacement", [0.0, 0.1])
def test_toppra_signed_requests_preserve_zero_speed_caps(displacement: float) -> None:
    planner = _zero_speed_planner("toppra")
    options = planner.default_plan_options()
    options.constraints["velocity"] = [[-0.5, 0.7], [-0.01, 0.01], [-1.0, 1.0]]
    start = torch.zeros(2, 3, dtype=torch.float64)
    goal = start.clone()
    goal[:, 0] = 0.4
    goal[:, 2] = displacement
    result = planner.plan(
        [PlanState.from_qpos(start), PlanState.from_qpos(goal)], options
    )
    assert result.is_all_success() is (displacement == 0.0)
    if displacement == 0.0:
        assert torch.count_nonzero(result.positions[..., 2]) == 0
        assert torch.count_nonzero(result.velocities[..., 2]) == 0


@pytest.mark.parametrize(("multiplier", "leader_cap"), [(-2.0, 0.0), (0.0, 0.6)])
def test_zero_speed_follower_respects_mimic_ratio(
    multiplier: float, leader_cap: float
) -> None:
    robot = _Robot()
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = (
        [2],
        [0],
        [multiplier],
    )
    robot.velocities[:, 2] = 0.0
    result = _resolve_joint_constraints(robot, {}, dof=3)
    assert result["velocity"].tolist() == [leader_cap, 0.02, 0.0]


def test_curobo_exports_zero_velocity_cap() -> None:
    robot = _Robot()
    robot.velocities[:, 2] = 0.0
    exported = _curobo_joint_limits(robot)
    assert exported["max_velocity"][2] == 0.0
    assert bool(torch.isfinite(torch.tensor(exported["max_velocity"])).all())


def test_planning_uses_effective_limits_without_reopening_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = _Robot()
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="test"><joint name="finger"><limit velocity="0.05"/></joint></robot>'
    )
    robot.cfg.fpath = str(path)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        Mock(side_effect=AssertionError("Planner reopened the asset")),
    )
    robot.velocities[:, 1] = 0.3
    result = _resolve_joint_constraints(
        robot, {"velocity": 1.0}, control_part="eef", dof=1
    )
    assert result["velocity"].item() == pytest.approx(0.3)
    robot.velocities[:, 1] = 0.1
    result = _resolve_joint_constraints(robot, {}, control_part="eef", dof=1)
    assert result["velocity"].item() == pytest.approx(0.1)


@pytest.mark.parametrize("missing", [float("nan"), float("inf"), -1.0, 1e10])
@pytest.mark.parametrize("velocity_request", [None, 0.4])
def test_missing_velocity_metadata_uses_request_or_compatibility_fallback(
    velocity_request: float | None,
    missing: float,
) -> None:
    robot = _Robot()
    robot.velocities[:] = missing
    result = _resolve_joint_constraints(robot, {"velocity": velocity_request}, dof=3)
    assert (
        result["velocity"].tolist()
        == [0.2 if velocity_request is None else velocity_request] * 3
    )
    assert result["acceleration"].tolist() == [0.5] * 3


def test_speed_requests_are_capped_while_acceleration_is_planner_owned() -> None:
    result = _resolve_joint_constraints(
        _Robot(),
        {"velocity": [5.0, 0.1], "acceleration": 3.0},
        control_part="arm",
        dof=2,
    )
    assert result["velocity"].tolist() == [1.2, 0.1]
    assert result["acceleration"].tolist() == [3.0, 3.0]


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


def test_per_joint_acceleration_requests_follow_trajectory_order() -> None:
    result = _resolve_joint_constraints(
        _Robot(), {"acceleration": [1.5, 0.3]}, control_part="arm", dof=2
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
def test_invalid_acceleration_request_fails_before_planning(value: float) -> None:
    with pytest.raises(ValueError, match="finite|positive"):
        _resolve_joint_constraints(_Robot(), {"acceleration": value}, dof=3)


@pytest.mark.parametrize("planner_type", ["toppra", "trapezoidal"])
def test_facade_and_direct_plans_apply_robot_limits(planner_type: str) -> None:
    robot = _Robot()
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
        options.constraints["acceleration"] = [0.9, 1.4]
        options = planner.with_motion_context(
            options, start_qpos=None, control_part="arm"
        )
        result = planner.plan(states, options)
        assert result.is_all_success()
        if options.sample_method == TrajectorySampleMethod.QUANTITY:
            assert result.positions.shape[1] == options.sample_interval
        else:
            assert result.positions.shape[1] > 2
            assert bool((result.dt[:, 1:] <= options.sample_interval + 1e-6).all())
        torch.testing.assert_close(result.positions[:, -1], states[-1].qpos)
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


@pytest.mark.parametrize(
    ("acceleration_request", "expected"), [(None, 15.0), (1.5, 1.5)]
)
def test_curobo_export_uses_robot_speeds_and_planner_acceleration(
    acceleration_request: float | None, expected: float
) -> None:
    exported = _curobo_joint_limits(_Robot(), max_acceleration=acceleration_request)
    assert exported["max_velocity"] == pytest.approx([0.6, 0.02, 1.2])
    assert exported["max_acceleration"] == [expected] * 3


def test_toppra_spawn_workers_receive_resolved_joint_limits() -> None:
    robot = _Robot()
    planner = ToppraPlanner.__new__(ToppraPlanner)
    planner.robot = robot
    planner.device = robot.device
    planner.cfg = ToppraPlannerCfg(
        robot_uid="robot", backend="numpy", max_workers=2, mp_context="spawn"
    )
    planner._pool = None
    planner._mp_context = "spawn"
    options = planner.default_plan_options()
    options.control_part = "arm"
    options.sample_method = TrajectorySampleMethod.QUANTITY
    options.sample_interval = 100
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


def test_mimic_follower_caps_constrain_curobo_active_joint() -> None:
    robot = _Robot()
    robot.velocities[:, 1] = 0.2
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = [1], [0], [-2.0]
    exported = _curobo_joint_limits(robot)
    assert exported["max_velocity"] == pytest.approx([0.1, 1.2])
    assert len(exported["max_acceleration"]) == 2


def test_mimic_speed_caps_preserve_planner_acceleration_request() -> None:
    robot = _Robot()
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = [1], [0], [-2.0]
    result = _resolve_joint_constraints(
        robot, {"acceleration": 1.0}, control_part="arm", dof=2
    )
    assert result["velocity"].tolist() == pytest.approx([1.2, 0.01])
    assert result["acceleration"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("reverse", [False, True])
def test_nested_mimic_caps_do_not_depend_on_relation_order(reverse: bool) -> None:
    robot = _Robot()
    robot.velocities[:] = torch.tensor([2.0, 1.0, 0.12])
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = (
        ([1, 2], [0, 1], [3.0, -2.0]) if reverse else ([2, 1], [1, 0], [-2.0, 3.0])
    )
    result = _resolve_joint_constraints(robot, {}, dof=3)
    assert result["velocity"].tolist() == pytest.approx([0.02, 0.06, 0.12])
    assert result["acceleration"].tolist() == [0.5] * 3


@pytest.mark.parametrize("multiplier", [float("nan"), float("inf")])
def test_nonfinite_runtime_mimic_multiplier_is_rejected(multiplier: float) -> None:
    robot = _Robot()
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = (
        [1],
        [0],
        [multiplier],
    )
    with pytest.raises(ValueError, match="finite"):
        _resolve_joint_constraints(robot, {}, dof=3)


def test_cyclic_runtime_mimic_relations_are_rejected() -> None:
    robot = _Robot()
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = (
        [1, 0],
        [0, 1],
        [1.0, 1.0],
    )
    with pytest.raises(ValueError, match="acyclic"):
        _resolve_joint_constraints(robot, {}, dof=3)


@pytest.mark.parametrize("change", ["velocity", "acceleration"])
def test_curobo_cache_changes_with_limits_and_closes_stale_backend(change: str) -> None:
    robot = _Robot()
    planner = CuroboPlanner.__new__(CuroboPlanner)
    planner.robot = robot
    planner.cfg = CuroboPlannerCfg(robot_uid="robot", use_cuda_graph=False)
    key = planner._robot_yaml_cache_key("", "arm", None, planner.cfg.auto_gen)
    cached = SimpleNamespace(motion_limits_key=key, planner=Mock())
    backend_key = ("arm", 2, False, MoveType.EEF_MOVE)
    planner._backend_cache = {backend_key: cached}
    assert planner._get_backend("arm", 2) is cached
    if change == "velocity":
        robot.velocities[:, 0] = 0.2
    else:
        planner.cfg.max_acceleration = 0.2
    assert planner._robot_yaml_cache_key("", "arm", None, planner.cfg.auto_gen) != key
    planner._materialize_profile = Mock(side_effect=RuntimeError("rebuild backend"))
    with pytest.raises(RuntimeError, match="rebuild backend"):
        planner._get_backend("arm", 2)
    cached.planner.close.assert_called_once()
    assert backend_key not in planner._backend_cache


def test_curobo_regenerates_cache_when_motion_urdf_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planner = CuroboPlanner.__new__(CuroboPlanner)
    planner.robot = _Robot()
    # The new budget must not shift existing positional configuration arguments.
    planner.cfg = CuroboPlannerCfg(
        "robot", "curobo", 0, "warning", max_acceleration=0.8
    )
    assert planner.cfg.log_level == "warning"
    planner.cfg.auto_gen.cache_dir = str(tmp_path)
    planner._curobo_device = torch.device("cpu")
    key = planner._robot_yaml_cache_key("", "arm", "tool", planner.cfg.auto_gen)
    yaml_path = tmp_path / f"{key}.yml"
    motion_path = tmp_path / f"{key}.motion.urdf"
    yaml_path.touch()
    motion_path.touch()
    generate = Mock(return_value=str(yaml_path))
    monkeypatch.setattr(
        "embodichain.lab.sim.motion.planners.curobo.curobo_yaml.generate_curobo_robot_yaml",
        generate,
    )
    assert planner._auto_generate_robot_yaml("arm", "tool") == str(yaml_path)
    generate.assert_not_called()
    motion_path.unlink()
    assert planner._auto_generate_robot_yaml("arm", "tool") == str(yaml_path)
    generate.assert_called_once()
    assert generate.call_args.kwargs["max_acceleration"] == 0.8


def test_unknown_mimic_follower_speed_does_not_freeze_known_leader() -> None:
    robot = _Robot()
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = [1], [0], [-2.0]
    robot.velocities[:, 1] = torch.inf
    result = _resolve_joint_constraints(robot, {}, control_part="arm", dof=2)
    assert result["velocity"].tolist() == pytest.approx([1.2, 0.6])
