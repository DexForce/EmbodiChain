# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Tests for reproducible simulation-level Affordance sampling values."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

import embodichain.lab.sim.atomic_actions.affordance_sampling as sampling_module
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
    AffordanceSample,
    AffordanceSamplingContext,
)
from embodichain.lab.sim.atomic_actions.affordance import (
    Affordance,
    AntipodalAffordance,
    AssembleAffordance,
    InteractionPoints,
    PressAffordance,
    TwistAffordance,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix


def _poses(batch_size: int, candidate_count: int | None = None) -> torch.Tensor:
    if candidate_count is None:
        return torch.eye(4).repeat(batch_size, 1, 1)
    return torch.eye(4).repeat(batch_size, candidate_count, 1, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("count", 0),
        ("count", True),
        ("seed", -1),
        ("episode_id", 1.5),
        ("attempt_id", -1),
        ("branch_override", 2),
    ],
)
def test_sampling_context_rejects_invalid_stream_identity(field: str, value: object):
    values = {"count": 2, "seed": 3, "episode_id": 4, "attempt_id": 5}
    values[field] = value
    with pytest.raises(ValueError):
        AffordanceSamplingContext(**values)


def test_sampling_context_uses_private_reproducible_generators():
    context = AffordanceSamplingContext(count=4, seed=3, episode_id=2, attempt_id=1)
    global_state = torch.random.get_rng_state().clone()

    first = torch.rand(5, generator=context.generator("grasp", group=7))
    second = torch.rand(5, generator=context.generator("grasp", group=7))
    changed = torch.rand(
        5,
        generator=replace(context, attempt_id=2).generator("grasp", group=7),
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_sampling_context_can_pin_a_logical_branch_across_environment_rows():
    context = AffordanceSamplingContext(
        count=4,
        seed=3,
        episode_id=2,
        branch_override=2,
    )

    assert context.branch_for(0) == 2
    assert context.branch_for(17) == 2
    values = context.sample_range(
        0.0,
        (-1.0, 1.0),
        env_ids=torch.tensor([0, 17]),
        key="pick",
    )
    assert torch.equal(values[0:1], values[1:2])


def test_pose_candidates_from_rows_preserves_empty_and_nonfinite_validity():
    finite = _poses(2)
    nonfinite = finite.clone()
    nonfinite[1, 0, 0] = torch.nan
    candidates = AffordancePoseCandidates.from_rows(
        [
            (nonfinite, torch.tensor([0.2, 0.1])),
            (torch.empty(0, 4, 4), torch.empty(0)),
        ],
        device="cpu",
    )

    assert candidates.poses.shape == (2, 2, 4, 4)
    assert candidates.valid.tolist() == [[True, False], [False, False]]
    assert torch.isinf(candidates.costs[1]).all()
    torch.testing.assert_close(candidates.poses[1], _poses(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "poses": torch.eye(4),
            "costs": torch.zeros(1, 1),
            "valid": torch.ones(1, 1, dtype=torch.bool),
        },
        {
            "poses": _poses(1, 2),
            "costs": torch.zeros(1, 1),
            "valid": torch.ones(1, 2, dtype=torch.bool),
        },
        {
            "poses": _poses(1, 2),
            "costs": torch.zeros(1, 2),
            "valid": torch.ones(1, 2),
        },
    ],
)
def test_pose_candidates_rejects_invalid_tensor_contract(kwargs: dict[str, object]):
    with pytest.raises((TypeError, ValueError)):
        AffordancePoseCandidates(**kwargs)


def test_affordance_sample_validates_and_owns_result_values():
    success = torch.tensor([True, False])
    poses = _poses(2)
    metadata = {
        "sampling": {"seed": 7},
        "candidate_ids": [0, -1],
    }
    sample = AffordanceSample(success=success, poses=poses, metadata=metadata)

    success[0] = False
    poses[0, 0, 0] = 7.0
    metadata["sampling"]["seed"] = 9
    metadata["candidate_ids"][0] = 9

    assert sample.success.tolist() == [True, False]
    assert sample.poses[0, 0, 0] == 1.0
    assert sample.metadata == {
        "sampling": {"seed": 7},
        "candidate_ids": [0, -1],
    }


@pytest.mark.parametrize(
    "success,poses,metadata",
    [
        (torch.ones(2), _poses(2), {}),
        (torch.ones(1, dtype=torch.bool), _poses(2), {}),
        (torch.ones(2, dtype=torch.bool), _poses(2).to(torch.int64), {}),
        (torch.ones(2, dtype=torch.bool), _poses(2), []),
    ],
)
def test_affordance_sample_rejects_invalid_result_contract(
    success: torch.Tensor,
    poses: torch.Tensor,
    metadata: object,
):
    with pytest.raises((TypeError, ValueError)):
        AffordanceSample(success=success, poses=poses, metadata=metadata)


def _sampling(count: int = 4) -> AffordanceSamplingContext:
    return AffordanceSamplingContext(count=count, seed=42, episode_id=3)


def test_affordance_is_the_only_public_candidate_sampling_entry_point():
    batch_size = 4
    local = _poses(batch_size)
    local[:, 0, 3] = torch.tensor([-0.04, -0.01, 0.02, 0.04])
    objects = _poses(batch_size)
    objects[:, 1, 3] = torch.arange(batch_size) * 2.5
    poses = objects[:, None] @ local[None]
    costs = torch.arange(batch_size, dtype=torch.float32).repeat(batch_size, 1)
    candidates = AffordancePoseCandidates(
        poses=poses,
        costs=costs,
        valid=torch.ones_like(costs, dtype=torch.bool),
    )
    global_state = torch.random.get_rng_state().clone()

    sample = Affordance().sample_candidates(
        candidates,
        sampling=_sampling(batch_size),
        env_ids=torch.arange(batch_size),
        key="pick",
        reference_poses=objects,
    )

    assert not hasattr(candidates, "select")
    assert isinstance(sample, AffordanceSample)
    assert sample.success.all()
    assert len(set(sample.metadata["candidate_ids"])) == batch_size
    assert sample.metadata["candidate_ids"][0] == 0
    torch.testing.assert_close(sample.poses[:, 1, 3], objects[:, 1, 3])
    assert sample.metadata["pose_frame"] == "world"
    assert sample.metadata["env_ids"] == list(range(batch_size))
    assert sample.metadata["success"] == [True] * batch_size
    assert torch.equal(torch.random.get_rng_state(), global_state)
    torch.testing.assert_close(
        torch.tensor(sample.metadata["selected_poses"]), sample.poses
    )
    torch.testing.assert_close(
        torch.tensor(sample.metadata["reference_poses"]), objects
    )
    assert json.loads(json.dumps(sample.metadata, allow_nan=False)) == sample.metadata
    objects.zero_()
    sample.poses.zero_()
    assert sample.metadata["selected_poses"][0][0][0] == 1.0
    assert sample.metadata["reference_poses"][0][0][0] == 1.0


def test_affordance_candidate_sampling_preserves_failed_rows_and_metadata():
    candidates = AffordancePoseCandidates.from_rows(
        [
            (_poses(1), torch.tensor([0.0])),
            (torch.empty(0, 4, 4), torch.empty(0)),
        ],
        device="cpu",
    )

    sample = Affordance().sample_candidates(
        candidates,
        sampling=_sampling(2),
        env_ids=torch.arange(2),
        key="grasp",
    )

    assert sample.success.tolist() == [True, False]
    assert sample.metadata["candidate_ids"] == [0, -1]
    assert sample.metadata["sampling"] == _sampling(2).metadata()
    assert sample.metadata["success"] == [True, False]
    assert sample.metadata["reference_poses"] is None
    assert sample.metadata["selected_poses"][1] == torch.eye(4).tolist()
    torch.testing.assert_close(sample.poses[1], torch.eye(4))


class _FakeGraspGenerator:
    def get_valid_grasp_poses(self, **_: object):
        first = _poses(2)
        first[:, 0, 3] = torch.tensor([0.0, 0.1])
        return [
            (first, torch.tensor([0.0, 1.0])),
            (torch.empty(0, 4, 4), torch.empty(0)),
        ]


def test_antipodal_affordance_builds_explicit_candidate_validity():
    affordance = AntipodalAffordance(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
    )

    candidates = affordance.get_grasp_candidates(
        _FakeGraspGenerator(),
        _poses(2),
        torch.tensor([0.0, 0.0, -1.0]),
    )

    assert candidates.valid.tolist() == [[True, True], [False, False]]


def test_press_sampling_rotates_only_contact_frame_roll():
    targets = _poses(4)
    targets[:, :3, :3] = axis_angle_to_rotation_matrix(torch.tensor([0.4, 0.2, -0.3]))
    targets[:, :3, 3] = torch.tensor([0.5, 0.1, 0.3])
    affordance = PressAffordance(
        press_axis=torch.tensor([1.0, 0.0, 0.0]),
        press_position=(0.01, 0.02, 0.03),
    )
    nominal = affordance.get_press_pose(targets)

    sample = affordance.sample_press_pose(
        targets,
        sampling=_sampling(4),
        env_ids=torch.arange(4),
        key="press",
    )

    assert sample.success.all()
    torch.testing.assert_close(sample.poses[:, :3, 3], nominal[:, :3, 3])
    torch.testing.assert_close(sample.poses[:, :3, 2], nominal[:, :3, 2])
    torch.testing.assert_close(torch.linalg.det(sample.poses[:, :3, :3]), torch.ones(4))
    assert sample.metadata["roll"][0] == 0.0
    assert len(torch.unique(sample.poses[:, :3, 0], dim=0)) == 4
    assert sample.metadata["pose_frame"] == "world"
    assert sample.metadata["env_ids"] == [0, 1, 2, 3]
    assert sample.metadata["success"] == [True] * 4
    torch.testing.assert_close(
        torch.tensor(sample.metadata["reference_poses"]), targets
    )
    torch.testing.assert_close(
        torch.tensor(sample.metadata["selected_poses"]), sample.poses
    )


@pytest.mark.parametrize("sampler", ["twist", "press"])
def test_nominal_sampling_rejects_misaligned_provenance_rows(sampler: str) -> None:
    targets = _poses(2)
    env_ids = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="matching rows"):
        if sampler == "twist":
            TwistAffordance(
                grasp_position=(0.0, 0.0, 0.0),
                axis_origin=(0.0, 0.0, 0.0),
            ).sample_grasp_pose(
                targets,
                sampling=None,
                env_ids=env_ids,
            )
        else:
            PressAffordance(
                press_axis=torch.tensor([0.0, 0.0, 1.0]),
                press_position=(0.0, 0.0, 0.0),
            ).sample_press_pose(
                targets,
                sampling=None,
                env_ids=env_ids,
            )


def test_pose_sampling_metadata_rejects_misaligned_reference_rows() -> None:
    with pytest.raises(ValueError, match="reference_poses must match"):
        sampling_module._pose_sampling_metadata(
            _poses(2),
            torch.ones(2, dtype=torch.bool),
            torch.arange(2),
            _poses(1),
        )


@pytest.mark.gpu
def test_pose_sampling_metadata_rejects_mismatched_devices() -> None:
    with pytest.raises(ValueError, match="share a device"):
        sampling_module._pose_sampling_metadata(
            _poses(2),
            torch.ones(2, dtype=torch.bool),
            torch.arange(2, device="cuda"),
            None,
        )


def test_twist_sampling_requires_explicit_roll_symmetry():
    targets = _poses(4)
    fixed = TwistAffordance(
        grasp_position=(0.1, 0.0, 0.0),
        axis_origin=(0.0, 0.0, 0.0),
    )
    nominal = fixed.get_grasp_pose(targets)

    fixed_sample = fixed.sample_grasp_pose(
        targets,
        sampling=_sampling(4),
        env_ids=torch.arange(4),
        key="twist",
    )
    torch.testing.assert_close(fixed_sample.poses, nominal)

    symmetric = TwistAffordance(
        grasp_position=(0.1, 0.0, 0.0),
        axis_origin=(0.0, 0.0, 0.0),
        grasp_roll_range=(-0.2, 0.2),
    )
    sample = symmetric.sample_grasp_pose(
        targets,
        sampling=_sampling(4),
        env_ids=torch.arange(4),
        key="twist",
    )
    torch.testing.assert_close(sample.poses[:, :3, 3], nominal[:, :3, 3])
    torch.testing.assert_close(sample.poses[:, :3, 2], nominal[:, :3, 2])
    assert not torch.equal(sample.poses, nominal)
    assert sample.metadata["pose_frame"] == "world"
    torch.testing.assert_close(
        torch.tensor(sample.metadata["reference_poses"]), targets
    )
    torch.testing.assert_close(
        torch.tensor(sample.metadata["selected_poses"]), sample.poses
    )
    torch.testing.assert_close(
        torch.tensor(sample.metadata["roll"]),
        _sampling(4).sample_range(
            0.0, (-0.2, 0.2), env_ids=torch.arange(4), key="twist"
        ),
    )


def test_assembly_sampling_uses_only_declared_rotational_symmetry():
    rotations = _poses(3)
    rotations[:, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor(
            [
                [0.0, 0.0, torch.pi / 2],
                [0.0, 0.0, torch.pi],
                [0.0, 0.0, -torch.pi / 2],
            ]
        )
    )
    relation = torch.eye(4)
    relation[2, 3] = 0.03
    affordance = AssembleAffordance(
        assemble_to_base_pose=relation,
        symmetry_transforms=rotations,
    )
    nominal = affordance.get_assemble_object_pose(_poses(4))

    sample = affordance.sample_assemble_object_pose(
        _poses(4),
        sampling=_sampling(4),
        env_ids=torch.arange(4),
        key="assemble",
    )

    torch.testing.assert_close(sample.poses[:, :3, 3], nominal[:, :3, 3])
    assert len(torch.unique(sample.poses.flatten(1), dim=0)) == 4
    assert sample.metadata["candidate_ids"][0] == 0
    assert set(sample.metadata["candidate_ids"]) == {0, 1, 2, 3}


def test_interaction_points_sampling_retains_success_and_metadata():
    affordance = InteractionPoints(
        points=torch.tensor([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        normals=torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1),
        point_types=["press", "press", "pull"],
    )

    sample = affordance.sample_poses(
        _poses(4),
        sampling=_sampling(4),
        env_ids=torch.arange(4),
        key="interaction",
        point_type="press",
    )

    assert sample.success.all()
    assert (sample.poses[:, 0, 3] <= 0.03).all()
    assert len(torch.unique(sample.poses[:, :3, 3], dim=0)) == 2
    assert sample.metadata["reused"] == [False, False, True, True]
    torch.testing.assert_close(
        sample.poses[:, :3, 2], torch.tensor([0.0, 0.0, 1.0]).expand(4, -1)
    )
