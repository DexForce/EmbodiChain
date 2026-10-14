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

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions.state import SceneSnapshot, TaskState
from embodichain.lab.task_program.integrations.simulation.environment import (
    SimulationPlanningObservationProvider,
)


def _provider(enabled):
    provider = object.__new__(SimulationPlanningObservationProvider)
    root = torch.eye(4)[None]
    provider._robot = NS(
        get_qpos=lambda: torch.zeros(1, 2),
        get_qvel=lambda: torch.zeros(1, 2),
        get_qf=lambda: torch.zeros(1, 2),
        get_local_pose=Mock(return_value=root),
    )
    provider._env_ids = torch.tensor([0])
    provider._clock = NS(now=lambda: 1.0, step_dt=0.04)
    provider._scene_provider = NS(
        snapshot=lambda **kwargs: SceneSnapshot(
            timestamp=kwargs["timestamp"], version=1
        )
    )
    if enabled:
        provider._capture_root_pose = True
    return provider, root


def test_gensim_root_pose_is_captured_in_the_same_context_and_owned():
    provider, root = _provider(True)
    context = provider._capture(TaskState(batch_size=1, device="cpu"))
    assert context.robot.timestamp == context.scene.timestamp == 1.0
    assert torch.equal(context.robot.root_pose, root)
    root[:, 0, 3] = 9.0
    assert context.robot.root_pose[0, 0, 3] == 0
    provider._robot.get_local_pose.assert_called_once_with(to_matrix=True)


def test_other_simulation_profiles_do_not_acquire_new_root_pose_reads():
    provider, _ = _provider(False)
    provider._robot.get_local_pose.side_effect = AssertionError("unrelated capture")
    context = provider._capture(TaskState(batch_size=1, device="cpu"))
    assert context.robot.root_pose is None
    provider._robot.get_local_pose.assert_not_called()


@pytest.mark.parametrize(
    "value", [None, torch.eye(3)[None], torch.full((1, 4, 4), float("nan"))]
)
def test_missing_or_invalid_root_pose_is_not_invented(value):
    provider, _ = _provider(True)
    provider._robot.get_local_pose.return_value = value
    with pytest.raises((ValueError, TypeError)):
        provider._capture(TaskState(batch_size=1, device="cpu"))
