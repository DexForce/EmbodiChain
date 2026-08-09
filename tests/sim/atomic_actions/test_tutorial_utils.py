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
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from scripts.tutorials.atomic_action.dynamic_obstacle_recovery import (
    _animate_obstacle_to_pose,
    _blocking_obstacle_pose,
    _maximum_path_deviation,
    _minimum_cuboid_clearance,
)
from scripts.tutorials.atomic_action.tutorial_utils import (
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    should_open_tutorial_window,
    should_wait_for_tutorial_input,
)

PHYSICS_DT = 0.1
MOVE_DURATION = 0.25
Y_OFFSET = 0.18
EXPECTED_STEP_COUNT = 3
CUBOID_SIZE = (0.2, 0.2, 0.2)


def _run_obstacle_animation(*, pace_wall_time: bool) -> tuple[MagicMock, MagicMock]:
    obstacle = MagicMock()
    adapter = MagicMock()
    adapter.physics_dt = PHYSICS_DT
    start_pose = torch.eye(4).unsqueeze(0)
    start_pose[:, 1, 3] = -0.2
    target_pose = start_pose.clone()
    target_pose[:, 1, 3] += Y_OFFSET

    result = _animate_obstacle_to_pose(
        obstacle,
        adapter,
        start_pose,
        target_pose=target_pose,
        duration=MOVE_DURATION,
        pace_wall_time=pace_wall_time,
    )

    assert torch.equal(result, target_pose)
    assert result.data_ptr() != target_pose.data_ptr()
    return obstacle, adapter


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
                viser=True,
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


def test_viser_does_not_open_native_tutorial_window() -> None:
    args = Namespace(
        headless=False,
        viser=True,
        diagnose_plan=False,
        headless_play=False,
    )

    assert not should_open_tutorial_window(args)


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


def test_obstacle_animation_interpolates_and_reaches_target_pose() -> None:
    obstacle, adapter = _run_obstacle_animation(pace_wall_time=False)

    poses = [entry.args[0] for entry in obstacle.set_local_pose.call_args_list]
    expected_y = torch.tensor([-0.14, -0.08, -0.02])
    actual_y = torch.stack([pose[0, 1, 3] for pose in poses])
    assert actual_y.tolist() == pytest.approx(expected_y.tolist())
    assert adapter.sleep.call_args_list == [call(PHYSICS_DT)] * EXPECTED_STEP_COUNT


def test_obstacle_animation_paces_live_viewer_in_wall_time() -> None:
    with patch(
        "scripts.tutorials.atomic_action.dynamic_obstacle_recovery.time.sleep"
    ) as sleep:
        _run_obstacle_animation(pace_wall_time=True)

    assert sleep.call_args_list == [call(PHYSICS_DT)] * EXPECTED_STEP_COUNT


def test_blocking_pose_targets_the_selected_initial_path_waypoint() -> None:
    start_pose = torch.eye(4).unsqueeze(0)
    path = torch.tensor([[[0.4, -0.1, 0.3], [0.5, 0.0, 0.4], [0.6, 0.1, 0.5]]])

    target_pose, waypoint_index = _blocking_obstacle_pose(
        start_pose,
        path,
        path_fraction=0.5,
    )

    assert waypoint_index == 1
    assert torch.equal(target_pose[:, :3, 3], path[:, waypoint_index])
    assert torch.equal(start_pose, torch.eye(4).unsqueeze(0))


def test_maximum_path_deviation_measures_detour_from_reference_polyline() -> None:
    reference_path = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    detour_path = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.2, 0.0], [1.0, 0.0, 0.0]]])

    deviation = _maximum_path_deviation(detour_path, reference_path)

    assert deviation.tolist() == pytest.approx([0.2])


def test_minimum_cuboid_clearance_is_positive_outside_cuboid() -> None:
    path = torch.tensor([[[0.25, 0.0, 0.0], [0.15, 0.0, 0.0]]])
    cuboid_pose = torch.eye(4).unsqueeze(0)

    clearance = _minimum_cuboid_clearance(
        path,
        cuboid_pose,
        size=CUBOID_SIZE,
    )

    assert clearance.tolist() == pytest.approx([0.05])


def test_minimum_cuboid_clearance_is_negative_inside_cuboid() -> None:
    path = torch.tensor([[[0.05, 0.0, 0.0]]])
    cuboid_pose = torch.eye(4).unsqueeze(0)

    clearance = _minimum_cuboid_clearance(
        path,
        cuboid_pose,
        size=CUBOID_SIZE,
    )

    assert clearance.tolist() == pytest.approx([-0.05])


def test_minimum_cuboid_clearance_uses_cuboid_orientation() -> None:
    path = torch.tensor([[[0.25, 0.0, 0.0]]])
    cuboid_pose = torch.eye(4).unsqueeze(0)
    cuboid_pose[0, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    clearance = _minimum_cuboid_clearance(
        path,
        cuboid_pose,
        size=(0.2, 0.4, 0.2),
    )

    assert clearance.tolist() == pytest.approx([0.05])
