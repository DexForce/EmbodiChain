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

"""Tests for atomic-action tutorial helpers."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import pytest
import torch

from scripts.tutorials.atomic_action.tutorial_utils import (
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    should_wait_for_tutorial_input,
)


def test_should_wait_for_tutorial_input_is_disabled_for_headless_modes() -> None:
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=True,
                diagnose_plan=False,
                headless_play=False,
            )
        )
        is False
    )
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=False,
                diagnose_plan=True,
                headless_play=False,
            )
        )
        is False
    )
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=False,
                diagnose_plan=False,
                headless_play=True,
            )
        )
        is False
    )


def test_broadcast_pose_batch_repeats_single_pose_for_each_env() -> None:
    pose = torch.eye(4, dtype=torch.float32)

    batched = broadcast_pose_batch(pose, num_envs=3)

    assert batched.shape == (3, 4, 4)
    assert torch.allclose(batched[0], pose)
    assert torch.allclose(batched[1], pose)
    assert torch.allclose(batched[2], pose)


def test_broadcast_waypoint_pose_batch_repeats_waypoints_for_each_env() -> None:
    waypoints = torch.stack(
        [torch.eye(4, dtype=torch.float32), 2.0 * torch.eye(4, dtype=torch.float32)],
        dim=0,
    )

    batched = broadcast_waypoint_pose_batch(waypoints, num_envs=2)

    assert batched.shape == (2, 2, 4, 4)
    assert torch.allclose(batched[0], waypoints)
    assert torch.allclose(batched[1], waypoints)


def test_clone_local_pose_from_first_env_sets_shared_pose() -> None:
    first_pose = torch.eye(4, dtype=torch.float32)
    first_pose[0, 3] = 0.2
    poses = torch.stack(
        [
            first_pose,
            2.0 * torch.eye(4, dtype=torch.float32),
            3.0 * torch.eye(4, dtype=torch.float32),
        ],
        dim=0,
    )
    entity = MagicMock()
    entity.get_local_pose.return_value = poses

    shared = clone_local_pose_from_first_env(entity)

    expected = first_pose.unsqueeze(0).repeat(3, 1, 1)
    assert torch.allclose(shared, expected)
    entity.set_local_pose.assert_called_once()
    assert torch.allclose(entity.set_local_pose.call_args.args[0], expected)


def test_create_antipodal_semantics_keeps_mesh_data_on_affordance() -> None:
    vertices = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    triangles = torch.tensor([[0, 1, 1]])
    obj = MagicMock()
    obj.get_vertices.return_value = vertices.unsqueeze(0)
    obj.get_triangles.return_value = triangles.unsqueeze(0)

    semantics = create_antipodal_semantics(
        obj,
        label="cube",
        n_sample=64,
        force_reannotate=True,
    )

    assert semantics.entity is obj
    assert semantics.label == "cube"
    assert semantics.geometry == {}
    assert torch.equal(semantics.affordance.mesh_vertices, vertices)
    assert torch.equal(semantics.affordance.mesh_triangles, triangles)
    assert semantics.affordance.force_reannotate is True
    assert semantics.affordance.generator_cfg.antipodal_sampler_cfg.n_sample == 64


def test_broadcast_pose_batch_rejects_wrong_env_count() -> None:
    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

    with pytest.raises(ValueError, match="num_envs"):
        broadcast_pose_batch(poses, num_envs=3)
