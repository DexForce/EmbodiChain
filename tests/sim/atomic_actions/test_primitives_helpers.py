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

"""Tests for atomic action primitive helpers."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    assemble_full_robot_trajectory,
    repeat_qpos,
    resolve_batched_pose,
    resolve_object_target,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import (
    PickUpOptions,
    _upright_yaw_pose_variants,
)

BATCH_SIZE = 2
ROBOT_DOF = 6
WAYPOINT_COUNT = 3


def test_resolve_batched_pose_broadcasts_and_owns_global_pose() -> None:
    source = torch.eye(4)

    result = resolve_batched_pose(
        source,
        num_envs=BATCH_SIZE,
        device=torch.device("cpu"),
        name="target_pose",
    )
    result[0, 0, 0] = 2.0

    assert result.shape == (BATCH_SIZE, 4, 4)
    assert source[0, 0] == 1.0


def test_assemble_full_robot_trajectory_overlays_control_parts() -> None:
    base = torch.zeros(BATCH_SIZE, ROBOT_DOF)
    first = torch.ones(BATCH_SIZE, WAYPOINT_COUNT, 2)
    second = torch.full((BATCH_SIZE, WAYPOINT_COUNT, 1), 2.0)

    result = assemble_full_robot_trajectory(
        base,
        (
            ((0, 2), first),
            ((5,), second),
        ),
    )

    expected = repeat_qpos(base, WAYPOINT_COUNT)
    expected[:, :, [0, 2]] = 1.0
    expected[:, :, 5] = 2.0
    assert torch.equal(result, expected)


def test_assemble_full_robot_trajectory_rejects_empty_parts() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        assemble_full_robot_trajectory(
            torch.zeros(BATCH_SIZE, ROBOT_DOF),
            (),
        )


def test_resolve_object_target_uses_custom_name_in_shape_error() -> None:
    with pytest.raises(ValueError, match="placing_object_target_pose"):
        resolve_object_target(
            torch.zeros(2, 4, 4),
            num_envs=3,
            device=torch.device("cpu"),
            name="placing_object_target_pose",
        )


def test_upright_yaw_pose_variants_preserve_translation() -> None:
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[:, :3, 3] = torch.tensor([[0.2, -0.1, 0.8], [-0.3, 0.4, 0.7]])

    variants = _upright_yaw_pose_variants(pose, 4)

    assert variants.shape == (2, 4, 4, 4)
    assert torch.allclose(variants[:, :, :3, 3], pose[:, None, :3, 3].expand(-1, 4, -1))
    assert torch.allclose(variants[:, 0], pose)


def test_upright_yaw_samples_must_be_positive() -> None:
    with pytest.raises(ValueError, match="upright_yaw_samples"):
        PickUpOptions(upright_yaw_samples=0)
