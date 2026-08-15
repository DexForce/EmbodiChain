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

from types import SimpleNamespace

import pytest
import torch

from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
    antipodal_cache_key,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import (
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionChecker,
)
from embodichain.toolkits.graspkit.pg_grasp.profiles import (
    AntipodalGraspPolicy,
    ParallelJawEefProfile,
    get_parallel_jaw_eef_profile,
)


def _triangle_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )


def test_raw_antipodal_cache_key_tracks_only_sampling_stage_inputs() -> None:
    vertices, triangles = _triangle_mesh()
    base = AntipodalSamplerCfg(
        n_sample=1000,
        max_angle=0.2,
        min_length=0.003,
        max_length=0.1,
    )

    key = antipodal_cache_key(vertices, triangles, base)
    same = antipodal_cache_key(vertices.clone(), triangles.clone(), base)
    changed_policy = antipodal_cache_key(
        vertices,
        triangles,
        AntipodalSamplerCfg(
            n_sample=1000,
            max_angle=0.2,
            min_length=0.01,
            max_length=0.1,
        ),
    )
    changed_mesh = antipodal_cache_key(vertices + 0.01, triangles, base)

    assert same == key
    assert changed_policy != key
    assert changed_mesh != key


def test_filter_diagnostics_count_each_candidate_stage() -> None:
    generator = GraspGenerator.__new__(GraspGenerator)
    generator.device = torch.device("cpu")
    generator.cfg = GraspGeneratorCfg(
        n_deviated_approach_directions=1,
        n_top_grasps=10,
    )
    generator._last_filter_diagnostics = {}

    class _CollisionChecker:
        last_query_diagnostics = {
            "candidate_count": 2,
            "object_collision_count": 1,
            "support_collision_count": 0,
            "combined_collision_count": 1,
            "support_filter_enabled": False,
        }

        @staticmethod
        def query(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.tensor([True, False]), torch.tensor([-0.01, 0.02])

    generator._collision_checker = _CollisionChecker()
    origins = torch.tensor([[-0.01, 0.00, 0.0], [-0.01, 0.10, 0.0]])
    hits = torch.tensor([[0.01, 0.00, 0.0], [0.01, 0.10, 0.0]])

    success, poses, _, _ = generator._filter_valid_grasp_poses(
        origin_points_=origins,
        hit_points_=hits,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        mesh_vert_transformed=torch.tensor(
            [[-0.02, 0.0, 0.0], [0.02, 0.2, 0.0]],
        ),
        object_pose=torch.eye(4),
        stage_name="left",
    )

    assert success is True
    assert poses.shape[0] == 1
    assert generator.last_filter_diagnostics["left"] == {
        "input_pair_count": 2,
        "angle_valid_pair_count": 2,
        "pose_candidate_count": 2,
        "collision": _CollisionChecker.last_query_diagnostics,
        "collision_free_pose_count": 1,
        "returned_pose_count": 1,
    }


def _stub_collision_checker() -> GripperCollisionChecker:
    checker = GripperCollisionChecker.__new__(GripperCollisionChecker)
    checker._last_query_diagnostics = {}
    checker._checker = SimpleNamespace(
        query_batch_points=lambda points, **kwargs: (
            torch.zeros(points.shape[:2], dtype=torch.bool),
            torch.ones(points.shape[:2], dtype=torch.float32),
        )
    )
    checker.cfg = SimpleNamespace(contact_penetration_tolerance=0.0)
    checker._get_gripper_pc = lambda poses, lengths: torch.tensor(
        [
            [[0.0, 0.0, -0.01], [0.0, 0.0, 0.02]],
            [[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]],
        ],
        dtype=torch.float32,
    )
    checker.get_ground_height = lambda pose: 0.0
    return checker


def test_support_plane_collision_contributes_to_query_result() -> None:
    checker = _stub_collision_checker()
    poses = torch.eye(4).repeat(2, 1, 1)
    openings = torch.full((2,), 0.02)

    colliding, _ = checker.query(
        torch.eye(4),
        poses,
        openings,
        is_filter_ground_collision=True,
    )

    assert colliding.tolist() == [True, False]
    assert checker.last_query_diagnostics == {
        "candidate_count": 2,
        "object_collision_count": 0,
        "support_collision_count": 1,
        "combined_collision_count": 1,
        "support_filter_enabled": True,
    }


def test_support_plane_collision_can_be_disabled_explicitly() -> None:
    checker = _stub_collision_checker()

    colliding, _ = checker.query(
        torch.eye(4),
        torch.eye(4).repeat(2, 1, 1),
        torch.full((2,), 0.02),
        is_filter_ground_collision=False,
    )

    assert not colliding.any()
    assert checker.last_query_diagnostics["support_filter_enabled"] is False


def test_support_plane_height_validates_batch_shape() -> None:
    checker = _stub_collision_checker()

    with pytest.raises(ValueError, match="support_plane_height"):
        checker.query(
            torch.eye(4),
            torch.eye(4).repeat(2, 1, 1),
            torch.full((2,), 0.02),
            support_plane_height=torch.tensor([0.0, 0.0, 0.0]),
        )


def test_object_contact_tolerance_does_not_relax_support_plane() -> None:
    checker = _stub_collision_checker()
    thresholds: list[float] = []
    checker.cfg.contact_penetration_tolerance = 0.005
    checker._checker.query_batch_points = lambda points, **kwargs: (
        thresholds.append(float(kwargs["collision_threshold"]))
        or torch.zeros(points.shape[:2], dtype=torch.bool),
        torch.ones(points.shape[:2], dtype=torch.float32),
    )

    colliding, _ = checker.query(
        torch.eye(4),
        torch.eye(4).repeat(2, 1, 1),
        torch.full((2,), 0.02),
        is_filter_ground_collision=True,
    )

    assert thresholds == [-0.005]
    assert colliding.tolist() == [True, False]


def test_sampling_policy_intersects_contact_span_with_eef_limits() -> None:
    eef = get_parallel_jaw_eef_profile("robotiq_arg2f_140")
    policy = AntipodalGraspPolicy(
        min_contact_span=0.003,
        max_contact_span=0.2,
    )

    minimum, maximum = policy.resolved_opening_range(eef)
    generator_cfg = policy.generator_config(eef)

    assert minimum == pytest.approx(0.003)
    assert maximum == pytest.approx(eef.jaw_opening_max)
    assert generator_cfg.antipodal_sampler_cfg.min_length == pytest.approx(minimum)
    assert generator_cfg.antipodal_sampler_cfg.max_length == pytest.approx(maximum)


def test_eef_profile_round_trips_without_robot_specific_data() -> None:
    source = get_parallel_jaw_eef_profile("robotiq_arg2f_140")

    restored = ParallelJawEefProfile.from_mapping(source.as_mapping())

    assert restored == source
    assert "robot" not in restored.as_mapping()


def test_approach_schedule_is_reproducible_and_attempt_aware() -> None:
    direction = torch.tensor([0.0, 0.0, -1.0])

    first = GraspGenerator._deterministic_approach_directions(
        direction,
        count=4,
        max_angle=0.3,
        attempt_id=0,
    )
    repeated = GraspGenerator._deterministic_approach_directions(
        direction,
        count=4,
        max_angle=0.3,
        attempt_id=0,
    )
    retry = GraspGenerator._deterministic_approach_directions(
        direction,
        count=4,
        max_angle=0.3,
        attempt_id=1,
    )

    assert torch.allclose(torch.stack(first), torch.stack(repeated))
    assert not torch.allclose(first[0], retry[0])
    assert not torch.allclose(torch.stack(first), torch.stack(retry))
    assert torch.allclose(
        torch.linalg.vector_norm(torch.stack(retry), dim=1),
        torch.ones(4),
    )


def test_eef_collision_proxy_contains_all_calibrated_dimensions() -> None:
    profile = get_parallel_jaw_eef_profile("dh_pgi_140_80")

    collision = profile.collision_config(max_decomposition_hulls=8)

    assert collision.max_open_length == pytest.approx(0.1)
    assert collision.finger_length == pytest.approx(0.1)
    assert collision.y_thickness == pytest.approx(0.04)
    assert collision.root_z_width == pytest.approx(0.096)
    assert collision.open_check_margin == pytest.approx(0.03)
    assert collision.max_decomposition_hulls == 8
