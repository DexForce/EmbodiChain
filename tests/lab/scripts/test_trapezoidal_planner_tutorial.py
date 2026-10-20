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

from cartesian_robot_support import CartesianRobot as _Robot
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scripts.tutorials.sim.planner.trapezoidal_planner import (
    build_cartesian_line_poses,
    build_demo_waypoints,
    configure_plot_fonts,
    diagnostic_output_path,
    maximum_line_deviation,
    plot_trajectory_diagnostics,
    positive_float,
    replay_plan,
    sample_count,
    set_equal_3d_limits,
)


class _Simulation:
    def __init__(self, physics_dt: float = 0.01) -> None:
        self.sim_config = SimpleNamespace(physics_dt=physics_dt)
        self.update_steps: list[int] = []
        self.elapsed = 0.0
        self.arrival_times: list[float] = []

    def update(self, step: int, physics_dt: float | None = None) -> None:
        self.update_steps.append(step)
        self.elapsed += step * (
            self.sim_config.physics_dt if physics_dt is None else physics_dt
        )
        self.arrival_times.append(self.elapsed)


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

    assert simulation.elapsed == pytest.approx(dt.sum().item())
    torch.testing.assert_close(robot.commands[-1], positions[:, -1])


@pytest.mark.parametrize("samples", [3, 200, 1000])
@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_replay_preserves_duration_and_interpolates_physics_ticks(
    samples: int, speed: float
) -> None:
    robot = _Robot(batch_size=1, dtype=torch.float64)
    simulation = _Simulation()
    # The final tick is deliberately shorter than physics_dt.
    times = torch.linspace(0.0, 0.053, samples, dtype=torch.float64)
    positions = times[None, :, None].expand(1, -1, 6).clone()
    dt = torch.diff(times, prepend=times.new_zeros(1))[None]
    replay_plan(
        simulation, robot, "left_arm", positions, dt, replay_speed=speed, realtime=False
    )
    assert simulation.elapsed == pytest.approx(times[-1].item() / speed, abs=1e-12)
    assert len(simulation.update_steps) <= 11
    # One initial command, then the interpolated target of each physics tick.
    assert len(robot.commands) == len(simulation.arrival_times) + 1
    for arrival, command in zip(simulation.arrival_times, robot.commands[1:]):
        torch.testing.assert_close(command, torch.full_like(command, arrival * speed))
    torch.testing.assert_close(robot.commands[-1], positions[:, -1])


def test_replay_zero_duration_hold_does_not_advance_physics() -> None:
    robot = _Robot(batch_size=1)
    simulation = _Simulation()
    positions = torch.full((1, 10, 6), 0.25)
    replay_plan(
        simulation, robot, "left_arm", positions, torch.zeros(1, 10), realtime=False
    )
    assert simulation.update_steps == []
    torch.testing.assert_close(robot.commands[-1], positions[:, -1])


def test_replay_interpolates_nonuniform_samples_and_skips_zero_intervals() -> None:
    robot = _Robot(batch_size=1, dtype=torch.float64)
    simulation = _Simulation()
    positions = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0], dtype=torch.float64)
    positions = positions[None, :, None].expand(1, -1, 6)
    dt = torch.tensor([[0.0, 0.015, 0.0, 0.017, 0.0]], dtype=torch.float64)
    replay_plan(simulation, robot, "left_arm", positions, dt, realtime=False)
    expected = [0.0, 0.01 / 0.015, 1.0 + 0.005 / 0.017, 1.0 + 0.015 / 0.017, 2.0]
    assert simulation.elapsed == pytest.approx(0.032)
    assert len(robot.commands) == len(expected)
    for command, position in zip(robot.commands, expected):
        torch.testing.assert_close(command, torch.full_like(command, position))


@pytest.mark.parametrize("interval", [-0.01, float("nan"), float("inf")])
def test_replay_rejects_invalid_timing_before_commands(interval: float) -> None:
    robot = _Robot(batch_size=1)
    simulation = _Simulation()
    with pytest.raises(ValueError, match="finite non-negative"):
        replay_plan(
            simulation,
            robot,
            "left_arm",
            torch.zeros(1, 2, 6),
            torch.tensor([[0.0, interval]]),
            realtime=False,
        )
    assert robot.commands == []
    assert simulation.update_steps == []


def test_replay_rejects_per_environment_timing_mismatch() -> None:
    """Replay must not silently slow shorter batch rows to the longest row."""
    robot = _Robot(batch_size=2)
    simulation = _Simulation(physics_dt=0.01)
    positions = torch.zeros(2, 2, 6)
    dt = torch.tensor([[0.0, 0.02], [0.0, 0.03]])

    with pytest.raises(ValueError, match="identical dt rows"):
        replay_plan(simulation, robot, "left_arm", positions, dt, realtime=False)


@pytest.mark.parametrize(
    ("parser", "value"),
    [(positive_float, "0"), (positive_float, "nan"), (sample_count, "1")],
)
def test_cli_numeric_constraints_fail_before_simulation(
    parser: Callable[[str], float | int], value: str
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must be"):
        parser(value)
