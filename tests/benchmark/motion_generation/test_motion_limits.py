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

"""Contract coverage for shared planning and validation motion budgets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.benchmarks.motion_generation.config import PlannerSpecCfg, load_suite
from embodichain.benchmarks.motion_generation.limits import (
    resolve_dynamic_limit,
    resolve_robot_velocity_limits,
    resolve_velocity_limits,
)
from embodichain.benchmarks.motion_generation.models import BenchmarkCase
from embodichain.benchmarks.motion_generation.planners.base import PlannerContext
from embodichain.benchmarks.motion_generation.planners.curobo import CuroboAdapter
from embodichain.benchmarks.motion_generation.planners.ik_interpolate import (
    IkInterpolateAdapter,
)
from embodichain.benchmarks.motion_generation.planners.nmg_onnx import NmgOnnxAdapter
from embodichain.benchmarks.motion_generation.planners.toppra import ToppraAdapter


def _robot() -> SimpleNamespace:
    """Provide reordered control joints and a speed-limited mimic follower."""
    caps = torch.tensor([[4.0, 3.0, 1.0], [2.0, 5.0, 0.6]])
    return SimpleNamespace(
        uid="robot",
        device=torch.device("cpu"),
        joint_names=["leader", "second", "follower"],
        active_joint_ids=[0, 1],
        mimic_ids=[2],
        mimic_parents=[0],
        mimic_multipliers=[-2.0],
        get_joint_ids=lambda name: [1, 0],
        get_qvel_limits=lambda *, name=None, joint_ids=None: caps[
            :, [1, 0] if name else joint_ids
        ],
    )


def _adapter(kind: type, *, velocity: object = None, config: dict | None = None):
    """Construct an adapter without building its optional backend."""
    return kind(
        PlannerSpecCfg(
            id="test", adapter="test", role="diagnostic_baseline", config=config or {}
        ),
        PlannerContext(
            robot=_robot(),
            control_part="arm",
            device=torch.device("cpu"),
            sample_interval=9,
            joint_velocity_limit_rad_s=velocity,
            joint_acceleration_limit_rad_s2=[8.0, 6.0],
        ),
    )


def test_velocity_request_cannot_loosen_valid_caps_and_fills_placeholders() -> None:
    """A deliberate suite request is capped where robot limits are usable."""
    actual = resolve_velocity_limits(
        torch.tensor([1.0, torch.inf, 0.0, -1.0]), 2.0, torch.empty(0)
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 2.0, 0.0, 2.0]))


def test_placeholder_without_budget_is_unavailable() -> None:
    """Unknown bounds cannot yield a claimed limit-respecting trajectory."""
    assert (
        resolve_velocity_limits(torch.tensor([2.0, 1e20]), None, torch.empty(0)) is None
    )


def test_effective_velocity_limits_respect_mimic_order_and_strictest_batch() -> None:
    """Follower caps project through multipliers before selecting control order."""
    actual = resolve_robot_velocity_limits(
        _robot(), "arm", [2.5, 1.0], torch.empty(0), dof=2
    )
    torch.testing.assert_close(actual, torch.tensor([2.5, 0.3]))


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), [1.0], True])
def test_dynamic_limits_reject_invalid_budgets(limit: object) -> None:
    """Malformed or incomplete constraints fail before backend construction."""
    with pytest.raises(ValueError):
        resolve_dynamic_limit(limit, 2, torch.empty(0))


def test_adapter_records_common_limits_and_rejects_private_budget() -> None:
    """Metadata states the same resolved constraints the planners consume."""
    adapter = _adapter(ToppraAdapter, velocity=2.0)
    actual = adapter.resolve_motion_limits(require_timing=True)
    assert actual["velocity"] == pytest.approx([2.0, 0.3])
    assert actual["acceleration"] == [8.0, 6.0]
    assert adapter.metadata.parameters["resolved_motion_limits"] == actual
    with pytest.raises(ValueError, match="differs from the suite"):
        _adapter(ToppraAdapter, config={"acceleration": 0.5}).resolve_motion_limits()


def test_nmg_retime_uses_conservative_common_budgets() -> None:
    """Retiming uses stricter batch caps instead of the first environment."""
    limits = _adapter(
        NmgOnnxAdapter, velocity=2.0, config={"num_arm_joints": 2}
    )._retiming_constraints()
    assert limits["velocity"] == pytest.approx([2.0, 0.3])
    assert limits["acceleration"] == [8.0, 6.0]


def test_curobo_maps_control_order_to_active_joint_order() -> None:
    """The selected control part may order joints differently from cuRobo."""
    limits = _adapter(CuroboAdapter, velocity=2.0)._backend_motion_limits()
    assert limits["max_velocity"] == pytest.approx([0.3, 2.0])
    assert limits["max_acceleration"] == [6.0, 8.0]


def test_joint_interpolation_uses_frozen_targets_without_ik() -> None:
    """The joint baseline consumes the same manifest targets as other planners."""
    adapter = _adapter(IkInterpolateAdapter, config={"interpolation_dt": 0.1})
    adapter.context.robot.compute_ik = Mock(
        side_effect=AssertionError("IK must not run")
    )
    case = BenchmarkCase(
        suite_version="2",
        track="motion",
        scenario_id="free_space",
        case_id="joint",
        seed=1,
        batch_size=1,
        num_waypoints=1,
        path_shape="joint",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 2),
        target_waypoints=torch.eye(4).view(1, 1, 4, 4),
        reference_qpos=torch.tensor([[[0.2, 0.4]]]),
        case_parameters={"motion_validity": "ordered_joint_waypoints"},
    )
    result = adapter.plan(case)
    assert adapter.supports_case(case) == (True, None)
    torch.testing.assert_close(result.positions[:, -1], case.reference_qpos[:, -1])
    assert result.success.tolist() == [True]


def test_toppra_keeps_common_batch_caps_for_smaller_planning_batch() -> None:
    """A smaller case must not discard stricter caps resolved for the suite."""
    from embodichain.lab.sim.motion.planners._joint_limits import (
        _resolve_joint_constraints,
    )

    adapter = _adapter(ToppraAdapter)
    options = adapter._plan_options()
    actual = _resolve_joint_constraints(
        adapter.context.robot,
        options.constraints,
        dof=2,
        control_part="arm",
        batch_size=1,
    )
    torch.testing.assert_close(
        actual["velocity"], torch.tensor([3.0, 0.3], dtype=torch.float64)
    )


def test_explicit_zero_speed_cap_is_preserved_and_moving_path_fails() -> None:
    """Stationary joints may hold, while a nonzero motion remains infeasible."""
    from embodichain.lab.sim.motion.planners._joint_limits import (
        _resolve_joint_constraints,
    )
    from embodichain.lab.sim.motion.planners.toppra_planner import retime_joint_paths
    from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod

    robot = _robot()
    constraints = _resolve_joint_constraints(
        robot,
        {"velocity": [0.0, 0.3], "acceleration": [8.0, 6.0]},
        dof=2,
        control_part="arm",
        batch_size=1,
    )
    assert constraints["velocity"][0] == 0.0
    result = retime_joint_paths(
        torch.tensor([[[0.0, 0.0], [0.2, 0.1]]]),
        constraints=constraints,
        sample_method=TrajectorySampleMethod.QUANTITY,
        sample_interval=10,
        device="cpu",
    )
    assert not bool(result.success[0])
    held = retime_joint_paths(
        torch.tensor([[[0.0, 0.0], [0.0, 0.1]]]),
        constraints=constraints,
        sample_method=TrajectorySampleMethod.QUANTITY,
        sample_interval=10,
        device="cpu",
    )
    assert bool(held.success[0])
    assert bool((held.positions[..., 0] == 0).all())


@pytest.mark.parametrize(
    "path",
    sorted(Path("embodichain/benchmarks/motion_generation/suites").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_shipped_suites_share_explicit_timed_planning_budgets(path: Path) -> None:
    """Every shipped timed adapter gets its budget from the protocol."""
    from embodichain.lab.sim.motion.planners import CuroboAutoGenCfg

    suite = load_suite(path)
    for spec in suite.planners:
        timed = spec.adapter in {"toppra", "curobo"} or (
            spec.adapter == "nmg_onnx" and spec.config.get("retime")
        )
        if timed:
            assert suite.protocol.joint_acceleration_limit_rad_s2 is not None
            assert "velocity" not in spec.config and "acceleration" not in spec.config
        if spec.adapter == "curobo":
            CuroboAutoGenCfg(**spec.config.get("auto_gen", {}))


def test_atomic_migration_preserves_existing_toppra_budgets() -> None:
    """Moving budgets to protocol must not accelerate existing Atomic tasks."""
    for path in Path("embodichain/benchmarks/motion_generation/suites").glob(
        "atomic*.yaml"
    ):
        suite = load_suite(path)
        if any(spec.adapter == "toppra" for spec in suite.planners):
            assert suite.protocol.joint_velocity_limit_rad_s == pytest.approx(0.2)
            assert suite.protocol.joint_acceleration_limit_rad_s2 == pytest.approx(0.5)


@pytest.mark.parametrize("waypoint_count", [1, 2])
def test_toppra_joint_cases_preserve_frozen_configurations_without_ik(
    waypoint_count: int,
) -> None:
    """Run the real generator and time parameterizer without Cartesian reprojection."""
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
    )

    adapter = _adapter(ToppraAdapter)
    adapter.context = replace(adapter.context, sample_interval=101)
    robot = adapter.context.robot
    for method in ("compute_fk", "compute_batch_fk", "compute_ik"):
        setattr(
            robot,
            method,
            Mock(
                side_effect=AssertionError(
                    "Joint targets must not be reprojected through FK/IK."
                )
            ),
        )
    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid=robot.uid)
    planner.robot = robot
    planner.device = torch.device("cpu")
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.robot = robot
    generator.device = robot.device
    adapter.motion_generator = generator
    targets = torch.tensor([[[0.2, -0.1], [-0.1, 0.1]]])[:, :waypoint_count]
    case = BenchmarkCase(
        suite_version="2",
        track="motion",
        scenario_id="free_space",
        case_id="joint",
        seed=1,
        batch_size=1,
        num_waypoints=waypoint_count,
        path_shape="joint",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 2),
        target_waypoints=torch.eye(4).expand(1, waypoint_count, 4, 4).clone(),
        reference_qpos=targets,
        case_parameters={"motion_validity": "ordered_joint_waypoints"},
    )
    result = adapter.plan(case)
    assert result.success.tolist() == [True]
    torch.testing.assert_close(result.positions[:, 0], case.start_qpos)
    torch.testing.assert_close(result.positions[:, -1], targets[:, -1])
    distances = torch.cdist(targets, result.positions).amin(dim=-1)
    assert bool((distances < 0.01).all())
