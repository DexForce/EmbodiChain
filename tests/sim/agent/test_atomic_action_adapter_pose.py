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

import math
from types import SimpleNamespace

import torch

from embodichain.lab.sim.agent.atomic_action_adapter import (
    _is_public_grasp_thin_object,
    _object_geometry_stats_from_world_vertices,
    _object_pose,
    _public_grasp_effective_antipodal_min_open_length,
    _public_grasp_effective_open_check_margin,
    _public_grasp_lift_height,
    _with_effective_thin_object_public_grasp_kwargs,
)
from embodichain.lab.sim.atomic_actions.semantic_grasp import (
    rank_semantic_grasp_candidates,
)


def test_object_pose_prefers_cached_obj_info_pose():
    cached_pose = torch.eye(4)
    cached_pose[2, 3] = 1.23
    sim_pose = torch.eye(4)
    sim_pose[2, 3] = 9.87

    env = SimpleNamespace(
        obj_info={"bottle": {"pose": cached_pose}},
        sim=SimpleNamespace(
            get_rigid_object_uid_list=lambda: ["bottle"],
            get_rigid_object=lambda name: SimpleNamespace(
                get_local_pose=lambda to_matrix=False: sim_pose.unsqueeze(0)
            ),
        ),
    )

    result = _object_pose(env, "bottle")

    assert torch.allclose(result, cached_pose)


def test_object_pose_falls_back_to_sim_pose_when_cache_missing():
    sim_pose = torch.eye(4)
    sim_pose[2, 3] = 0.42

    env = SimpleNamespace(
        obj_info={},
        sim=SimpleNamespace(
            get_rigid_object_uid_list=lambda: ["bottle"],
            get_rigid_object=lambda name: SimpleNamespace(
                get_local_pose=lambda to_matrix=False: sim_pose.unsqueeze(0)
            ),
        ),
    )

    result = _object_pose(env, "bottle")

    assert torch.allclose(result, sim_pose)


def test_thin_object_uses_pca_xy_slenderness_and_relaxed_defaults():
    vertices = torch.tensor(
        [
            [-0.10, -0.01, 0.0],
            [0.10, -0.01, 0.0],
            [0.10, 0.01, 0.0],
            [-0.10, 0.01, 0.0],
        ],
        dtype=torch.float32,
    )
    angle = math.pi / 4
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    world_vertices = vertices @ rotation.T

    stats = _object_geometry_stats_from_world_vertices(world_vertices)

    assert stats["xy_slenderness"] > 3.0
    assert _is_public_grasp_thin_object(stats, {})
    assert _public_grasp_effective_antipodal_min_open_length(stats, {}) == 0.001
    assert _public_grasp_effective_open_check_margin(stats, {}) == 0.004


def test_thin_object_caps_effective_lift_height():
    vertices = torch.tensor(
        [
            [-0.10, -0.01, 0.0],
            [0.10, -0.01, 0.0],
            [0.10, 0.01, 0.0],
            [-0.10, 0.01, 0.0],
        ],
        dtype=torch.float32,
    )
    obj = SimpleNamespace(
        get_local_pose=lambda to_matrix=True: torch.eye(4).unsqueeze(0),
        get_vertices=lambda env_ids=None, scale=True: [vertices],
    )
    env = SimpleNamespace(
        robot=SimpleNamespace(device=torch.device("cpu")),
        sim=SimpleNamespace(get_rigid_object=lambda name: obj),
    )

    kwargs = _with_effective_thin_object_public_grasp_kwargs(
        env,
        "spoon",
        {"public_grasp_lift_height": 0.15},
    )

    assert _public_grasp_lift_height(kwargs) == 0.06


def test_thin_object_ranking_prefers_top_down_direction():
    lateral = SimpleNamespace(
        label="object_neg_x_down",
        direction=torch.tensor([-0.96, 0.05, -0.27]),
        qpos_score=0.1,
        geometry_score=0.1,
        roll_score=0.0,
        legacy_score=0.01,
        candidate_idx=0,
    )
    top_down = SimpleNamespace(
        label="top_down",
        direction=torch.tensor([0.0, 0.0, -1.0]),
        qpos_score=1.0,
        geometry_score=1.0,
        roll_score=0.0,
        legacy_score=0.1,
        candidate_idx=1,
    )

    ranked = rank_semantic_grasp_candidates(
        [lateral, top_down],
        {
            "public_grasp_is_thin_object": True,
            "public_grasp_rank_by_legacy_pose": True,
        },
    )

    assert ranked[0] is top_down
