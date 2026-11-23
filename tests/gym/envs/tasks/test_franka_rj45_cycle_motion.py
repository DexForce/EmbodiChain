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
"""Verify authored stage bounds and acknowledged feedback sampling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.extensions.tasks.configs import get_config_path
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.sim.atomic_skills import ObservedJointTrajectory
from embodichain.lab.sim.motion._grasp_trajectory_feedback import (
    _GraspTrajectoryFeedback,
)
from embodichain.lab.task_program.integrations._trajectory_phase import (
    _read_track,
    _TrajectoryPhaseFactory,
    _TrajectoryPhaseLowerer,
    _TrajectorySource,
)
from embodichain.lab.task_program.integrations.composition import (
    load_configured_task_program_deployment,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.utils.utility import load_config

pytestmark = pytest.mark.no_sim


def _deployment():
    path = get_config_path(
        "tasks/manipulation/cable/franka_rj45_cycle/task.franka.yaml"
    )
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    return load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
        evaluation=physical.evaluation,
    )


def test_rj45_program_invokes_once_per_authored_phase() -> None:
    deployment = _deployment()
    registration = deployment.integration.registration
    compiled = registration.catalog.preflight(deployment.load_program())
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
        assert invocation.call.semantic_id == _TrajectoryPhaseFactory.call_id
        assert invocation.call.arguments == {"phase": segment.name}


def test_rj45_feedback_samples_once_at_the_declared_arrivals() -> None:
    (factory,) = (
        _deployment().integration.registration.registered_semantic_lowerer_factories
    )
    positions, poses, _, dt = _read_track(Path(factory.trajectory).read_bytes())
    robot, plug = Mock(), Mock()
    robot.get_qpos.return_value = torch.from_numpy(positions[:1])
    context = SimpleNamespace(
        batch_size=1,
        require_control_dt=lambda: dt,
        is_preview=False,
        env_ids=torch.tensor([0]),
        planning_mask=torch.tensor([True]),
        robot=SimpleNamespace(timestamp=0.0, qpos=torch.zeros(1, 9)),
    )
    reads = []

    def observed_pose(index: int) -> torch.Tensor:
        frame = round(context.robot.timestamp / dt) + 1
        pose = np.eye(4)
        pose[0, 3] = 0.025 + index * 0.015
        pose[1, 3] = frame / (10_000 * (index + 1))
        if index == 0:
            reads.append(frame)
        return torch.from_numpy(pose[None])

    robot.get_link_pose.side_effect = lambda *_, **__: observed_pose(0)
    plug.get_link_pose.side_effect = lambda *_, **__: observed_pose(1)

    # The production feedback controller retains the measured transforms.
    # Use an analytic pose-to-joint projection instead of an external URDF.
    feedback = object.__new__(_GraspTrajectoryFeedback)
    feedback._initial_offset = factory.feedback.initial_offset
    feedback._initial_object = np.eye(4)
    feedback._tracks = factory.feedback.tracks
    feedback._grasps = [np.array(track.tool_in_object) for track in feedback._tracks]
    feedback._reference_objects = {}
    feedback._hz = 60
    feedback._joint_indices = list(range(7))
    feedback._solve = lambda target, seed: seed + target[0, 3]

    source = _TrajectorySource(robot, factory.phases, positions, poses, dt)
    source._feedback = feedback
    source._object = plug
    source._tool_link = factory.feedback.tool_link
    source._object_link = factory.feedback.link
    source.observation_steps = tuple(sorted(feedback.observation_steps()))
    lowerer = _TrajectoryPhaseLowerer("fixture", source)

    context.is_preview = True
    preview = source.resolve(
        "place_and_release", 0, 360, context, context.planning_mask
    )
    assert preview.waypoint_count == 360
    robot.get_link_pose.assert_not_called()
    plug.get_link_pose.assert_not_called()
    context.is_preview = False

    qpos, qvel = [], []
    for phase in factory.phases:
        context.robot.timestamp = phase.start * dt
        goal = lowerer.lower(
            RegisteredSemanticCall(
                call_id=factory.call_id,
                arguments={"phase": phase.id},
            ),
            context=context,
            bound=None,
            option_template=None,
        ).goal.target
        if isinstance(goal, ObservedJointTrajectory):
            chunks = []
            start = 0
            for count in goal.chunk_lengths:
                context.robot.timestamp = (phase.start + start) * dt
                chunk = source.resolve(
                    phase.id, start, start + count, context, context.planning_mask
                )
                if phase.start + start + 1 == 1860:
                    duplicate = source.resolve(
                        phase.id, start, start + count, context, context.planning_mask
                    )
                    assert duplicate is chunk
                chunks.append(chunk)
                robot.get_qpos.return_value = chunk.positions[:, -1]
                start += count
            assert start == phase.end - phase.start
        else:
            chunks = [goal]
            robot.get_qpos.return_value = goal.positions[:, -1]
        for chunk in chunks:
            qpos.append(chunk.positions[0])
            qvel.append(chunk.velocities[0])

    assert reads == [*range(540, 601), 1141, 1488, *range(1860, 1981)]
    issued = torch.cat(qpos)
    assert issued.shape == (2400, 9)
    expected_velocities = (
        torch.cat((torch.from_numpy(positions[:1]), issued)).diff(dim=0) * 60
    )
    torch.testing.assert_close(torch.cat(qvel), expected_velocities, rtol=0, atol=0)
    torch.testing.assert_close(issued[-360:], torch.from_numpy(positions[-360:]))
