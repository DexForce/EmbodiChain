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

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest
import torch

_EXAMPLE = (
    Path(__file__).parents[3] / "examples/sim/motion/trajectory_velocity_tracking.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "trajectory_velocity_tracking", _EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tutorial():
    path = Path(__file__).parents[3] / "scripts/tutorials/sim/motion_generator.py"
    spec = importlib.util.spec_from_file_location("motion_generator_tutorial", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_neural_example():
    path = Path(__file__).parents[3] / "examples/sim/motion/planners/neural_planner.py"
    spec = importlib.util.spec_from_file_location("neural_planner_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_starts_and_ends_at_rest() -> None:
    example = _load_example()

    positions, velocities, dt = example.generate_reference(
        torch.tensor([[1.0, -2.0]]),
        duration=1.0,
        control_dt=0.25,
        amplitude=0.2,
    )

    assert positions.shape == velocities.shape == (1, 5, 2)
    assert dt.shape == (1, 5)
    torch.testing.assert_close(positions[:, 0], torch.tensor([[1.0, -2.0]]))
    torch.testing.assert_close(positions[:, -1], torch.tensor([[1.0, -2.0]]))
    torch.testing.assert_close(velocities[:, [0, -1]], torch.zeros(1, 2, 2))
    torch.testing.assert_close(dt, torch.tensor([[0.0, 0.25, 0.25, 0.25, 0.25]]))


def test_tracking_metrics_use_joint_samples() -> None:
    example = _load_example()
    reference = torch.tensor([[[0.0, 0.0], [1.0, 2.0]]])
    measured = torch.tensor([[[0.0, 1.0], [2.0, 2.0]]])

    metrics = example.compute_tracking_metrics(reference, measured)

    assert metrics["rmse"] == pytest.approx(2**-0.5)
    assert metrics["p95"] == pytest.approx(1.0)
    assert metrics["max"] == pytest.approx(1.0)


def test_pose_errors_report_translation_norm_and_rotation_angle() -> None:
    example = _load_example()
    reference = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    measured = reference.clone()
    measured[0, 0, :3, 3] = torch.tensor([3.0, 4.0, 0.0])
    measured[0, 1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    translation_error, rotation_error = example.compute_pose_errors(reference, measured)

    torch.testing.assert_close(translation_error, torch.tensor([[5.0, 0.0]]))
    torch.testing.assert_close(
        rotation_error, torch.tensor([[0.0, torch.pi / 2]]), atol=1.0e-6, rtol=0.0
    )


def test_fk_trajectory_calls_robot_compute_fk_for_each_sample() -> None:
    example = _load_example()

    class FakeRobot:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, str, bool]] = []

        def compute_fk(self, qpos, name, to_matrix):
            self.calls.append((qpos.clone(), name, to_matrix))
            pose = torch.eye(4).repeat(qpos.shape[0], 1, 1)
            pose[:, 0, 3] = qpos.sum(dim=1)
            return pose

    robot = FakeRobot()
    qpos = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    poses = example._compute_fk_trajectory(robot, qpos, control_part="arm")

    assert len(robot.calls) == 2
    assert all(name == "arm" and to_matrix for _, name, to_matrix in robot.calls)
    torch.testing.assert_close(poses[0, :, 0, 3], torch.tensor([3.0, 7.0]))


def test_tutorial_resamples_fractional_intervals_and_holds_terminal_velocity() -> None:
    tutorial = _load_tutorial()

    class FakeRobot:
        def __init__(self) -> None:
            self.commands: list[tuple[str, float]] = []

        def get_joint_ids(self, _name):
            return [0]

        def set_qpos(self, qpos, **_kwargs):
            self.commands.append(("qpos", float(qpos[0, 0])))

        def set_qvel(self, qvel, **_kwargs):
            self.commands.append(("qvel", float(qvel[0, 0])))

    class FakeSim:
        class Config:
            physics_dt = 0.04

        sim_config = Config()

        def __init__(self) -> None:
            self.intervals: list[float] = []

        def update(self, physics_dt=None, step=1):
            assert step == 1
            self.intervals.append(physics_dt)

    robot, sim = FakeRobot(), FakeSim()
    tutorial.move_robot_along_trajectory(
        sim,
        robot,
        "arm",
        torch.tensor([[[0.0], [0.5], [1.0]]]),
        None,
        torch.tensor([[0.0, 0.06, 0.06]]),
    )

    assert sim.intervals == pytest.approx([0.04, 0.04, 0.04])
    assert sum(sim.intervals) == pytest.approx(0.12)
    assert robot.commands[-2:] == [("qpos", 1.0), ("qvel", 0.0)]
    assert [value for kind, value in robot.commands if kind == "qpos"] == pytest.approx(
        [0.0, 1 / 3, 2 / 3, 1.0]
    )


def test_neural_playback_preserves_native_velocity_and_physics_substeps() -> None:
    example = _load_neural_example()

    class FakeRobot:
        num_instances = 1

        def __init__(self) -> None:
            self.velocities: list[float] = []

        def get_joint_ids(self, _name):
            return [0]

        def set_qpos(self, **_kwargs):
            pass

        def set_qvel(self, qvel, **_kwargs):
            self.velocities.append(float(qvel[0, 0]))

    class FakeSim:
        class Config:
            physics_dt = 0.02

        sim_config = Config()

        def __init__(self) -> None:
            self.updates: list[tuple[float, int]] = []

        def update(self, physics_dt=None, step=1):
            self.updates.append((physics_dt, step))

    robot, sim = FakeRobot(), FakeSim()
    example.play_trajectory(
        sim,
        robot,
        "arm",
        torch.tensor([[[0.0], [1.0], [2.0]]]),
        torch.tensor([[[7.0], [8.0], [9.0]]]),
        torch.tensor([[0.0, 0.08, 0.08]]),
        step_repeat=4,
    )

    assert [update[0] for update in sim.updates] == pytest.approx([0.02, 0.02])
    assert [update[1] for update in sim.updates] == [4, 4]
    assert robot.velocities == [7.0, 8.0, 0.0]


def test_trial_explicitly_clears_current_and_target_velocity_after_reset() -> None:
    example = _load_example()

    class FakeRobot:
        def __init__(self) -> None:
            self.qvel_targets: list[bool] = []
            self.qpos = torch.tensor([[0.0]])
            self.current_velocity = 3.0

        def reset(self):
            self.qpos += self.current_velocity * 0.001

        def get_qpos(self):
            return self.qpos

        def get_joint_ids(self, _name):
            return [0]

        def compute_fk(self, qpos, name, to_matrix):
            assert name == "arm"
            assert to_matrix
            pose = torch.eye(4).repeat(qpos.shape[0], 1, 1)
            pose[:, 0, 3] = qpos[:, 0]
            return pose

        def set_qvel(self, value, *, target=True, **_kwargs):
            self.qvel_targets.append(target)
            if not target:
                self.current_velocity = float(value[0, 0])

        def set_qpos(self, qpos, **_kwargs):
            self.qpos = qpos.clone()

    class FakeSim:
        def update(self, **_kwargs):
            pass

    args = Namespace(
        physics_dt=0.1,
        control_dt=0.1,
        duration=0.2,
        amplitude=0.1,
        settle_seconds=0.0,
        terminal_settle_seconds=0.0,
    )
    robot = FakeRobot()
    common_initial = robot.get_qpos().clone()

    trial = example._run_trial(
        args, FakeSim(), robot, common_initial, use_velocity=False
    )

    assert robot.qvel_targets[:4] == [False, True, False, True]
    torch.testing.assert_close(trial["reference"][:, 0], common_initial)
