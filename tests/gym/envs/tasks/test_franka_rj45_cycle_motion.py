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
"""Verify that measured RJ45 corrections retain the authored command grid."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations.composition import (
    load_configured_task_program_deployment,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.utils.utility import load_config
from embodichain.extensions.tasks.manipulation.cable.rj45_integration import (
    FrankaRj45PhaseFactory,
    _FrankaRj45PhaseLowerer,
    _Rj45JointTrajectorySource,
)
from embodichain.extensions.tasks.manipulation.cable.rj45_motion import Rj45Feedback, Rj45Motion
from embodichain.extensions.tasks.manipulation.cable.rj45_scene import (
    load_config as load_rj45_config,
    task_directory,
)

pytestmark = pytest.mark.no_sim


def test_rj45_program_invokes_once_per_authored_phase() -> None:
    path = task_directory() / "task.franka.yaml"
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    deployment = load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
        evaluation=physical.evaluation,
    )
    catalog = deployment.integration.registration.catalog
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=catalog,
    )
    compiled = catalog.preflight(program)
    segments = tuple(compiled.iter_segments())
    assert tuple(segment.name for segment in segments) == (
        "home",
        "floor_pickup",
        "approach_socket",
        "insert_and_release",
        "seated_hold",
        "latch_grasp",
        "withdraw",
        "place_and_release",
        "return_home",
    )
    for segment in segments:
        (invocation,) = segment.calls
        assert invocation.call.semantic_id == FrankaRj45PhaseFactory.call_id
        assert invocation.call.arguments == {"phase": segment.name}


def test_rj45_chunks_preserve_the_per_tick_observations_and_commands(
    monkeypatch,
) -> None:
    phases = (
        ("home", 60),
        ("floor_pickup", 300),
        ("approach_socket", 240),
        ("insert_and_release", 240),
        ("seated_hold", 300),
        ("latch_grasp", 360),
        ("withdraw", 180),
        ("place_and_release", 360),
        ("return_home", 360),
    )
    nominal_phases = {"home", "seated_hold", "return_home"}
    positions = (
        np.arange(2401, dtype=np.float32)[:, None] / 4096
        + np.arange(9, dtype=np.float32)[None, :] / 32
    )
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, 0, 3] = np.linspace(0.02, 0.04, len(poses))
    motion = Rj45Motion(positions, poses, (), np.arange(len(positions)) / 60)

    def observed_poses(frame: int) -> tuple[np.ndarray, np.ndarray]:
        hand, plug = np.eye(4), np.eye(4)
        hand[:3, 3] = 0.02, frame / 10_000, 0.3
        plug[:3, 3] = 0.04, 0.1 + frame / 20_000, 0.2
        return hand, plug

    def solve_feedback(target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        return seed + np.resize(target[:3, 3], seed.shape).astype(seed.dtype)

    monkeypatch.setattr(
        "embodichain.extensions.tasks.manipulation.cable.rj45_motion._Rj45Kinematics",
        lambda *_: SimpleNamespace(solve_feedback=solve_feedback),
    )
    feedback_args = load_rj45_config(), Path("synthetic.urdf"), observed_poses(0)[1]
    feedback = Rj45Feedback(*feedback_args)
    legacy_feedback = Rj45Feedback(*feedback_args)

    robot, plug = Mock(), Mock()
    robot.get_qpos.return_value = torch.from_numpy(positions[0:1])
    context = SimpleNamespace(
        batch_size=1,
        require_control_dt=lambda: 1 / 60,
        is_preview=False,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(timestamp=0.0, qpos=torch.zeros(1, 9)),
    )
    read_frames = []

    def read_link_pose(index: int) -> torch.Tensor:
        frame = round(context.robot.timestamp * 60) + 1
        if index == 0:
            read_frames.append(frame)
        return torch.from_numpy(observed_poses(frame)[index][None])

    robot.get_link_pose.side_effect = lambda *_args, **_kw: read_link_pose(0)
    plug.get_link_pose.side_effect = lambda *_args, **_kw: read_link_pose(1)
    source = _Rj45JointTrajectorySource(robot, plug, motion, feedback, 60)
    lowerer = _FrankaRj45PhaseLowerer("rj45_mock", 60)
    active = torch.tensor([True])

    context.is_preview = True
    preview = source.resolve("place_and_release", 0, 360, context, active)
    assert preview.waypoint_count == 360
    robot.get_link_pose.assert_not_called()
    plug.get_link_pose.assert_not_called()
    context.is_preview = False

    expected_positions = positions.copy()
    actual_positions, actual_velocities = [], []
    elapsed = 0
    for phase, steps in phases:
        context.robot.timestamp = elapsed / 60
        goal = lowerer.lower(
            RegisteredSemanticCall(
                call_id=FrankaRj45PhaseFactory.call_id,
                arguments={"phase": phase},
            ),
            context=context,
            bound=None,
            option_template=None,
        ).goal.target
        assert sum(goal.chunk_lengths) == steps
        start = 0
        for duration in goal.chunk_lengths:
            context.robot.timestamp = (elapsed + start) / 60
            stop = start + duration
            chunk = source.resolve(phase, start, stop, context, active)
            if elapsed + start + 1 == 1860:
                # An echoed request must not apply the final placement filter twice.
                duplicate = source.resolve(phase, start, stop, context, active)
                torch.testing.assert_close(chunk.positions, duplicate.positions)
            assert chunk.waypoint_count == duration
            assert (chunk.dt == 1 / 60).all()
            robot.get_qpos.return_value = chunk.positions[:, -1]
            actual_positions.append(chunk.positions[0])
            actual_velocities.append(chunk.velocities[0])
            start = stop

        if phase not in nominal_phases:
            for frame in range(elapsed + 1, elapsed + steps + 1):
                expected_positions[frame] = legacy_feedback.correct(
                    frame / 60,
                    positions[frame],
                    poses[frame],
                    lambda: observed_poses(frame),
                )
        elapsed += steps

    assert elapsed == 2400
    assert read_frames == [
        *range(540, 601),
        1141,
        1488,
        *range(1860, 1981),
    ]
    expected = torch.from_numpy(expected_positions)
    torch.testing.assert_close(
        torch.cat(actual_positions), expected[1:], rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.cat(actual_velocities), expected.diff(dim=0) * 60, rtol=0, atol=0
    )
