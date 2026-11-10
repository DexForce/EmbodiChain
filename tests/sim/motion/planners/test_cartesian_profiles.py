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
import torch

from cartesian_robot_support import CartesianRobot as _Robot

from embodichain.lab.sim.motion.planners._cartesian_profiles import (
    plan_cartesian_path,
    compute_eef_trajectory,
)
from embodichain.lab.sim.motion.planners import BezierPath, TrapezoidalPlanOptions

pytestmark = pytest.mark.no_sim


def _plan(robot, *, count=101, bezier=False):
    start = robot.compute_fk(robot.qpos, "left_arm", True)
    goal = start.clone()
    distance = 0.2 if bezier else 0.1
    goal[:, 2, 3] -= distance
    path = None
    if bezier:
        # Authored quintic translations; independent of the adapter implementation.
        offsets = robot.qpos.new_tensor(
            [
                [0.0, 0.0, 0.0],
                [0.02, 0.0, -0.02],
                [0.06, 0.0, -0.07],
                [0.06, 0.0, -0.13],
                [0.02, 0.0, -0.18],
                [0.0, 0.0, -0.2],
            ]
        )
        path = BezierPath(start[:, None, :3, 3] + offsets)
    options = TrapezoidalPlanOptions(
        profile="double_s" if bezier else "trapezoidal",
        sample_interval=count,
        constraints={"velocity": 0.15, "acceleration": 0.3, "jerk": 1.0},
        backend="torch",
    )
    return plan_cartesian_path(
        robot, "left_arm", robot.qpos, start, goal, options=options, path=path
    )


def test_cartesian_path_requires_three_samples_for_path_derivatives() -> None:
    robot = _Robot()

    with pytest.raises(ValueError, match="sample_count >= 3"):
        _plan(robot, count=2)


def test_cartesian_time_law_is_applied_before_ik() -> None:
    robot = _Robot()
    distance = 0.10
    velocity_limit = 0.15
    acceleration_limit = 0.30

    result, desired_poses, scalar_plan = _plan(robot)

    desired_xyz = desired_poses[0, :, :3, 3]
    assert torch.all(result.duration > 0.0)
    torch.testing.assert_close(desired_xyz[:, :2], torch.zeros_like(desired_xyz[:, :2]))
    assert scalar_plan.velocities.abs().max() <= velocity_limit + 1e-6
    assert scalar_plan.accelerations.abs().max() <= acceleration_limit + 1e-6
    assert torch.allclose(
        desired_xyz[-1] - desired_xyz[0], torch.tensor([0, 0, -distance])
    )
    assert torch.count_nonzero(result.velocities[:, 0]) == 0
    assert robot.path_ik_call_count == 1


def test_required_ik_failure_is_reported_without_dropping_waypoints() -> None:
    robot = _Robot()
    solve = robot.compute_batch_ik

    def fail_one(*args, **kwargs):
        success, positions = solve(*args, **kwargs)
        success[1, 3] = False
        return success, positions

    robot.compute_batch_ik = fail_one
    with pytest.raises(RuntimeError, match="env 1, sample 3"):
        _plan(robot)


@pytest.mark.parametrize("mismatch", ["orientation", "endpoints"])
def test_inconsistent_cartesian_geometry_is_rejected_before_ik(mismatch: str) -> None:
    robot = _Robot()
    start = torch.eye(4).repeat(2, 1, 1)
    goal = start.clone()
    goal[:, 2, 3] = -0.1
    path = None
    if mismatch == "orientation":
        goal[:, :2, :2] *= -1
    else:
        path = BezierPath(torch.zeros(2, 3, 3))
    with pytest.raises(ValueError, match="orientation|endpoints"):
        plan_cartesian_path(
            robot,
            "left_arm",
            robot.qpos,
            start,
            goal,
            options=TrapezoidalPlanOptions(),
            path=path,
        )
    assert robot.path_ik_call_count == 0


@pytest.mark.parametrize("count", [501, 2001])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_cartesian_bezier_derivatives_match_timed_positions(
    count: int, batch_size: int
) -> None:
    robot = _Robot(batch_size=batch_size, dtype=torch.float64)
    robot.qpos[:, :3] = torch.arange(batch_size)[:, None] * torch.tensor(
        [0.1, -0.2, 0.3]
    )
    result, _, _ = _plan(robot, count=count, bezier=True)

    for row in range(robot.qpos.shape[0]):
        time = result.dt[row].cumsum(0)
        velocity = torch.gradient(
            result.positions[row], spacing=(time,), dim=(0,), edge_order=2
        )[0]
        acceleration = torch.gradient(
            velocity, spacing=(time,), dim=(0,), edge_order=2
        )[0]
        # Interior central differences allow for finite spacing across jerk
        # switches and the approximate arc-length inverse lookup.
        torch.testing.assert_close(
            result.velocities[row, 2:-2], velocity[2:-2], atol=2e-5, rtol=1e-3
        )
        torch.testing.assert_close(
            result.accelerations[row, 2:-2], acceleration[2:-2], atol=2e-3, rtol=1e-3
        )


def test_eef_trajectory_uses_one_batched_fk_call() -> None:
    robot = _Robot(batch_size=2)
    sample_count = 101
    joint_positions = torch.zeros(2, sample_count, 6)
    joint_positions[1, :, 2] = torch.linspace(0.0, -0.1, sample_count)

    poses = compute_eef_trajectory(robot, "left_arm", joint_positions, env_index=1)

    assert poses.shape == (sample_count, 7)
    assert torch.equal(poses[:, 2], joint_positions[1, :, 2])
    assert torch.equal(
        poses[:, 3:],
        torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(sample_count, -1),
    )
    assert robot.fk_call_count == 1
