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

import argparse
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from matplotlib import pyplot as plt

from scripts.tutorials.sim.trapezoidal_planner import (
    build_cartesian_line_poses,
    build_demo_waypoints,
    compute_eef_trajectory,
    configure_plot_fonts,
    diagnostic_output_path,
    joint_derivatives_from_path_time_law,
    maximum_line_deviation,
    nearest_equivalent_joint_solution,
    plan_cartesian_line,
    plot_trajectory_diagnostics,
    positive_float,
    replay_plan,
    sample_count,
    set_equal_3d_limits,
)


class _Solver:
    """Identity kinematics used to validate path derivative propagation."""

    def get_jacobian(self, qpos: torch.Tensor, jac_type: str = "full") -> torch.Tensor:
        assert jac_type == "full"
        dof = qpos.shape[-1]
        return torch.eye(6, dof, dtype=qpos.dtype).expand(qpos.shape[0], -1, -1)


class _Robot:
    """Minimal robot interface used by the tutorial waypoint builders."""

    def __init__(self, batch_size: int = 2, dof: int = 6) -> None:
        self.qpos = torch.zeros(batch_size, dof)
        self.commands: list[torch.Tensor] = []
        self.solver = _Solver()
        self.fk_call_count = 0
        self.path_ik_call_count = 0

    def get_qpos(self, name: str) -> torch.Tensor:
        assert name == "left_arm"
        return self.qpos

    def get_joint_ids(self, name: str) -> list[int]:
        assert name == "left_arm"
        return list(range(self.qpos.shape[1]))

    def get_qpos_limits(self, joint_ids: list[int]) -> torch.Tensor:
        assert joint_ids == list(range(self.qpos.shape[1]))
        limits = torch.tensor((-1.0, 1.0)).repeat(self.qpos.shape[1], 1)
        return limits.unsqueeze(0).repeat(self.qpos.shape[0], 1, 1)

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str,
        to_matrix: bool,
        env_ids: list[int] | None = None,
    ) -> torch.Tensor:
        assert name == "left_arm"
        self.fk_call_count += 1
        if env_ids is not None:
            assert len(env_ids) == qpos.shape[0]
        if to_matrix:
            pose = torch.eye(4).repeat(qpos.shape[0], 1, 1)
            pose[:, :3, 3] = qpos[:, :3]
            return pose
        quaternion = torch.zeros(qpos.shape[0], 4)
        quaternion[:, 0] = 1.0
        return torch.cat((qpos[:, :3], quaternion), dim=-1)

    def compute_ik(
        self,
        pose: torch.Tensor,
        joint_seed: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert name == "left_arm"
        qpos = joint_seed.clone()
        qpos[:, :3] = pose[:, :3, 3]
        success = torch.ones(pose.shape[0], dtype=torch.bool)
        return success, qpos

    def compute_ik_path(
        self,
        pose: torch.Tensor,
        joint_seed: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert name == "left_arm"
        self.path_ik_call_count += 1
        qpos = joint_seed[:, None].expand(-1, pose.shape[1], -1).clone()
        qpos[:, :, :3] = pose[:, :, :3, 3]
        success = torch.ones(pose.shape[:2], dtype=torch.bool)
        return success, qpos

    def get_solver(self, name: str) -> _Solver:
        assert name == "left_arm"
        return self.solver

    def get_control_part_base_pose(self, name: str, to_matrix: bool) -> torch.Tensor:
        assert name == "left_arm"
        assert to_matrix
        return torch.eye(4).repeat(self.qpos.shape[0], 1, 1)

    def set_qpos(self, qpos: torch.Tensor, name: str) -> None:
        assert name == "left_arm"
        self.commands.append(qpos.clone())


class _Simulation:
    def __init__(self, physics_dt: float = 0.01) -> None:
        self.sim_config = SimpleNamespace(physics_dt=physics_dt)
        self.update_steps: list[int] = []

    def update(self, step: int) -> None:
        self.update_steps.append(step)


def test_plot_font_style_has_readable_cjk_fallback_and_hierarchy() -> None:
    configure_plot_fonts()

    assert plt.rcParams["font.sans-serif"][0] == "Noto Sans CJK SC"
    assert plt.rcParams["axes.titlesize"] > plt.rcParams["font.size"]
    assert plt.rcParams["font.size"] > plt.rcParams["legend.fontsize"]
    assert plt.rcParams["axes.unicode_minus"] is False


def test_joint_demo_moves_every_joint_within_limits() -> None:
    robot = _Robot()

    start, middle, goal = build_demo_waypoints(robot, "left_arm")

    assert torch.all(middle != start)
    assert torch.all(goal != start)
    assert torch.all(middle.abs() <= 1.0)
    assert torch.all(goal.abs() <= 1.0)


def test_six_axis_demo_uses_toppra_nonsingular_seed() -> None:
    robot = _Robot()

    start = build_demo_waypoints(robot, "left_arm")[0]

    assert torch.allclose(start[:, 1], torch.full((2,), torch.pi / 4.0))
    assert torch.allclose(start[:, 2], torch.full((2,), -torch.pi / 4.0))
    assert torch.allclose(start[:, 4], torch.full((2,), torch.pi / 4.0))


def test_cartesian_demo_preserves_orientation_and_requested_distance() -> None:
    robot = _Robot()
    requested_distance = 0.18

    start, goal = build_cartesian_line_poses(
        robot,
        "left_arm",
        robot.qpos,
        requested_distance,
    )

    displacement = goal[:, :3, 3] - start[:, :3, 3]
    assert torch.allclose(
        torch.linalg.vector_norm(displacement, dim=-1),
        torch.full((robot.qpos.shape[0],), requested_distance),
    )
    assert torch.equal(goal[:, :3, :3], start[:, :3, :3])


def test_cartesian_demo_rejects_nonpositive_distance() -> None:
    robot = _Robot()

    with pytest.raises(ValueError, match="greater than zero"):
        build_cartesian_line_poses(robot, "left_arm", robot.qpos, 0.0)


def test_cartesian_time_law_is_applied_before_ik() -> None:
    robot = _Robot()
    distance = 0.10
    velocity_limit = 0.15
    acceleration_limit = 0.30

    result, desired_poses, scalar_plan = plan_cartesian_line(
        robot,
        "left_arm",
        robot.qpos,
        distance=distance,
        profile="trapezoidal",
        sample_count=101,
        velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit,
        jerk_limit=1.0,
        backend="torch",
    )

    desired_xyz = desired_poses[0, :, :3, 3]
    assert torch.all(result.duration > 0.0)
    assert maximum_line_deviation(desired_xyz).item() < 1e-7
    assert scalar_plan.velocities.abs().max() <= velocity_limit + 1e-6
    assert scalar_plan.accelerations.abs().max() <= acceleration_limit + 1e-6
    assert torch.allclose(
        desired_xyz[-1] - desired_xyz[0], torch.tensor([0, 0, -distance])
    )
    assert torch.count_nonzero(result.velocities[:, 0]) == 0
    assert robot.path_ik_call_count == 1


def test_eef_trajectory_uses_one_batched_fk_call() -> None:
    robot = _Robot(batch_size=2)
    sample_count = 101
    joint_positions = torch.zeros(2, sample_count, 6)
    joint_positions[1, :, 2] = torch.linspace(0.0, -0.1, sample_count)

    poses = compute_eef_trajectory(robot, "left_arm", joint_positions, env_index=1)

    assert poses.shape == (sample_count, 7)
    assert torch.equal(poses[:, 2], joint_positions[1, :, 2])
    assert robot.fk_call_count == 1


def test_path_time_law_produces_exact_joint_derivatives_for_identity_model() -> None:
    sample_count = 101
    parameter = torch.linspace(0.0, torch.pi, sample_count, dtype=torch.float64)
    path_position = torch.linspace(
        0.0, 0.1, sample_count, dtype=torch.float64
    ).unsqueeze(0)
    path_velocity = torch.sin(parameter).unsqueeze(0)
    path_acceleration = torch.cos(parameter).unsqueeze(0)
    jacobians = (
        torch.eye(6, dtype=torch.float64)
        .reshape(1, 1, 6, 6)
        .expand(1, sample_count, -1, -1)
    )
    tangent = torch.tensor([[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    velocity, acceleration = joint_derivatives_from_path_time_law(
        jacobians,
        path_position,
        path_velocity,
        path_acceleration,
        tangent,
    )

    assert torch.allclose(velocity[..., 2], -path_velocity, atol=1e-7)
    assert torch.allclose(acceleration[..., 2], -path_acceleration, atol=1e-7)
    assert torch.count_nonzero(velocity[..., [0, 1, 3, 4, 5]]) == 0


def test_path_time_law_includes_jacobian_curvature_acceleration() -> None:
    sample_count = 51
    path_position = torch.linspace(
        0.0, 1.0, sample_count, dtype=torch.float64
    ).unsqueeze(0)
    path_velocity = torch.full_like(path_position, 0.5)
    path_acceleration = torch.zeros_like(path_position)
    jacobians = (
        torch.eye(6, dtype=torch.float64)
        .reshape(1, 1, 6, 6)
        .repeat(1, sample_count, 1, 1)
    )
    jacobians[0, :, 0, 0] = 1.0 + path_position[0]
    tangent = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    velocity, acceleration = joint_derivatives_from_path_time_law(
        jacobians,
        path_position,
        path_velocity,
        path_acceleration,
        tangent,
    )

    expected_q_s = 1.0 / (1.0 + path_position)
    expected_q_ss = -1.0 / (1.0 + path_position).square()
    assert torch.allclose(velocity[..., 0], expected_q_s * 0.5, atol=1e-12)
    assert torch.allclose(acceleration[..., 0], expected_q_ss * 0.25, atol=1e-12)


def test_nearest_equivalent_ik_solution_removes_angle_wrap() -> None:
    seed = torch.tensor([[3.13]])
    wrapped_solution = torch.tensor([[-3.13]])
    limits = torch.tensor([[[-2.0 * torch.pi, 2.0 * torch.pi]]])

    continuous = nearest_equivalent_joint_solution(wrapped_solution, seed, limits)

    assert (continuous - seed).abs().item() < 0.03


def test_line_deviation_detects_off_axis_samples() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [1.0, 0.0, 0.0]])

    deviation = maximum_line_deviation(xyz)

    assert deviation.item() == pytest.approx(0.1)


def test_equal_3d_limits_do_not_magnify_cross_axis_noise() -> None:
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    points = torch.tensor(
        [[1e-6, -2e-6, 0.0], [-1e-6, 2e-6, -0.1]], dtype=torch.float64
    )

    try:
        set_equal_3d_limits(axis, points)
        spans = torch.tensor(
            [
                axis.get_xlim()[1] - axis.get_xlim()[0],
                axis.get_ylim()[1] - axis.get_ylim()[0],
                axis.get_zlim()[1] - axis.get_zlim()[0],
            ]
        )
        assert torch.allclose(spans, torch.full_like(spans, spans[0]))
        assert spans[0].item() > 0.1
    finally:
        plt.close(figure)


def test_multiple_diagnostics_receive_distinct_output_names() -> None:
    base = Path("outputs/trajectory.png")

    joint = diagnostic_output_path(base, "joint_velocity_trapezoidal", True)
    cartesian = diagnostic_output_path(base, "cartesian_velocity_trapezoidal", True)

    assert joint.name == "trajectory_joint_velocity_trapezoidal.png"
    assert cartesian.name == "trajectory_cartesian_velocity_trapezoidal.png"


def test_diagnostics_do_not_save_without_explicit_output_path() -> None:
    assert diagnostic_output_path(None, "cartesian_velocity_trapezoidal", True) is None


def test_diagnostic_dashboard_contains_six_focused_panels() -> None:
    samples = 8
    dof = 6
    dt = torch.full((1, samples), 0.02)
    dt[:, 0] = 0.0
    poses = torch.zeros(samples, 7)
    poses[:, 2] = torch.linspace(0.0, -0.1, samples)
    poses[:, 3] = 1.0
    desired_poses = torch.eye(4).repeat(samples, 1, 1)
    desired_poses[:, 2, 3] = poses[:, 2]
    positions = torch.linspace(0.0, 0.2, samples).view(1, samples, 1)
    positions = positions.repeat(1, 1, dof)

    figure = plot_trajectory_diagnostics(
        dt=dt,
        eef_poses=poses,
        joint_positions=positions,
        joint_velocities=torch.full_like(positions, 0.1),
        joint_accelerations=torch.full_like(positions, 0.2),
        desired_eef_poses=desired_poses,
        env_index=0,
        output_path=None,
        profile_label="cartesian_acceleration_trapezoidal",
    )

    try:
        assert len(figure.axes) == 6
        assert "duration" in figure._suptitle.get_text()
        assert figure.axes[0].name == "3d"
        assert len(figure.axes[1].lines) == 6
    finally:
        plt.close(figure)


def test_replay_uses_explicit_dt_to_advance_physics() -> None:
    robot = _Robot(batch_size=1)
    simulation = _Simulation(physics_dt=0.01)
    positions = torch.zeros(1, 3, 6)
    dt = torch.tensor([[0.0, 0.02, 0.03]])

    replay_plan(
        simulation,
        robot,
        "left_arm",
        positions,
        dt,
        realtime=False,
    )

    assert simulation.update_steps == [1, 2, 3]
    assert len(robot.commands) == positions.shape[1]


@pytest.mark.parametrize(
    ("parser", "value"),
    [(positive_float, "0"), (positive_float, "nan"), (sample_count, "1")],
)
def test_cli_numeric_constraints_fail_before_simulation(
    parser: Callable[[str], float | int], value: str
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must be"):
        parser(value)
