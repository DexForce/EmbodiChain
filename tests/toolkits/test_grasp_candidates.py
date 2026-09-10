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

"""Candidate validity, legacy adaptation, and deterministic grasp sampling."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from embodichain.toolkits.graspkit import (
    GraspCandidateBatch,
    GraspPoseGenerator,
    ParallelJawGripperModelCfg,
)
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalSampler,
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp import pose_generator as facade_module
from embodichain.toolkits.graspkit.pg_grasp._antipodal_backend import (
    _AntipodalMeshBackend,
)


def _poses(count: int) -> torch.Tensor:
    poses = torch.eye(4).repeat(count, 1, 1)
    poses[:, 0, 3] = torch.arange(count) * 0.01
    return poses


@pytest.mark.parametrize(
    "bad_kind", ["nan_pose", "bad_rotation", "nan_cost", "nan_width", "negative_width"]
)
def test_invalid_candidate_is_masked_without_hiding_valid_peer(bad_kind: str) -> None:
    poses = _poses(2).unsqueeze(0)
    costs = torch.zeros(1, 2)
    widths = torch.full((1, 2), 0.04)
    if bad_kind == "nan_pose":
        poses[0, 0, 0, 0] = torch.nan
    elif bad_kind == "bad_rotation":
        poses[0, 0, 0, 0] = -1
    elif bad_kind == "nan_cost":
        costs[0, 0] = torch.nan
    elif bad_kind == "nan_width":
        widths[0, 0] = torch.nan
    else:
        widths[0, 0] = -0.01
    batch = GraspCandidateBatch(
        poses, costs, torch.ones(1, 2, dtype=torch.bool), widths
    )
    assert batch.valid_mask.tolist() == [[False, True]]
    assert torch.isfinite(batch.poses).all()
    assert batch.costs[0, 0].isinf()
    assert batch.rejection_reasons[0][0] is not None
    assert batch.rejection_reasons[0][1] is None
    torch.testing.assert_close(batch.poses[0, 1], poses[0, 1])


def test_ragged_padding_and_owned_single_row() -> None:
    first = _poses(2)
    batch = GraspCandidateBatch.from_ragged(
        [(first, torch.zeros(2)), (_poses(0), torch.empty(0))]
    )
    assert batch.valid_mask.tolist() == [[True, True], [False, False]]
    snapshot = batch.row(0)
    first.zero_()
    snapshot.poses.zero_()
    torch.testing.assert_close(batch.poses[0, 0], torch.eye(4))
    assert batch.rejection_reasons[1] == ("PADDING", "PADDING")


def test_all_empty_candidate_rows_keep_source_count() -> None:
    batch = GraspCandidateBatch.from_ragged([(_poses(0), torch.empty(0))] * 3)
    assert batch.poses.shape == (3, 0, 4, 4)
    assert batch.valid_mask.shape == (3, 0)
    assert batch.grasp_ids == ((), (), ())
    assert GraspCandidateBatch.from_ragged([]).poses.shape == (0, 0, 4, 4)


def test_object_relative_ids_survive_reordering_and_replica_translation() -> None:
    source = _poses(2)
    original = GraspCandidateBatch.from_ragged(
        [(source, torch.zeros(2))], object_poses=torch.eye(4).unsqueeze(0)
    )
    translated_object = torch.eye(4)
    translated_object[:3, 3] = torch.tensor([2.0, 3.0, 1.0])
    moved = translated_object @ source.flip(0)
    reordered = GraspCandidateBatch.from_ragged(
        [(moved, torch.ones(2))], object_poses=translated_object.unsqueeze(0)
    )
    assert original.grasp_ids[0] == tuple(reversed(reordered.grasp_ids[0]))


def test_batch_rejects_shared_shape_and_frame_contract_errors() -> None:
    with pytest.raises(ValueError, match="costs"):
        GraspCandidateBatch(
            _poses(2).unsqueeze(0), torch.zeros(2), torch.ones(1, 2, dtype=torch.bool)
        )
    with pytest.raises(TypeError, match="boolean"):
        GraspCandidateBatch(_poses(2).unsqueeze(0), torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError, match="frame"):
        GraspCandidateBatch(
            _poses(0).unsqueeze(0),
            torch.empty(1, 0),
            torch.empty(1, 0, dtype=torch.bool),
            frame="",
        )


class _LegacyGenerator(GraspPoseGenerator):
    def get_valid_grasp_poses(
        self, **kwargs: Any
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(pose.unsqueeze(0), torch.zeros(1)) for pose in kwargs["obj_poses"]]

    def get_best_grasp_poses(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ones(1, dtype=torch.bool), kwargs["obj_poses"], torch.ones(1)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "mesh_vertices": torch.tensor(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]]
        ),
        "mesh_triangles": torch.tensor([[0, 1, 2]]),
        "obj_poses": torch.eye(4).repeat(2, 1, 1),
        "approach_direction": torch.tensor([0.0, 0.0, -1.0]),
    }


def test_old_generator_remains_instantiable_and_adapts_without_widths() -> None:
    generator = _LegacyGenerator()
    batch = generator.get_grasp_candidates(**_inputs())
    assert batch.valid_mask.tolist() == [[True], [True]]
    assert batch.opening_widths is None
    with pytest.raises(NotImplementedError, match="local RNG"):
        generator.get_grasp_candidates(**_inputs(), generator=torch.Generator())


class _RandomBackend:
    """Exercise facade stream isolation without collision geometry or simulation."""

    instances: list[_RandomBackend] = []

    def __init__(self, vertices: torch.Tensor, **kwargs: Any) -> None:
        self.device = vertices.device
        self.is_prepared = True
        self.sampling_seed = kwargs.get("sampling_seed")
        self.selected_seeds = [self.sampling_seed]
        self.calls = 0
        type(self).instances.append(self)

    def _select_sampling_seed(self, sampling_seed: int | None) -> None:
        self.sampling_seed = sampling_seed
        self.selected_seeds.append(sampling_seed)

    def get_valid_grasp_poses(
        self,
        object_pose: torch.Tensor,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls += 1
        poses = object_pose.repeat(2, 1, 1)
        poses[:, 0, 3] += torch.rand(2, generator=generator)
        return True, poses, torch.tensor([0.04, 0.2]), torch.tensor([0.1, 0.2])


def test_antipodal_preserves_widths_and_isolates_rng_on_cache_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RandomBackend.instances.clear()
    monkeypatch.setattr(facade_module, "_AntipodalMeshBackend", _RandomBackend)
    service = AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
    inputs = _inputs()
    global_state = torch.random.get_rng_state().clone()
    first = service.get_grasp_candidates(
        **inputs, generator=torch.Generator().manual_seed(42)
    )
    second = service.get_grasp_candidates(
        **inputs, generator=torch.Generator().manual_seed(42)
    )
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert len(_RandomBackend.instances) == 1
    assert _RandomBackend.instances[0].calls == 4
    torch.testing.assert_close(first.poses, second.poses)
    assert first.grasp_ids == second.grasp_ids
    assert first.valid_mask.tolist() == [[True, False], [True, False]]
    assert first.opening_widths is not None
    assert first.opening_widths[0, 0] == pytest.approx(0.04)
    assert first.rejection_reasons[0][1] == "INVALID_OPENING_WIDTH"


def test_antipodal_sampling_seeds_share_geometry_but_select_distinct_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RandomBackend.instances.clear()
    monkeypatch.setattr(facade_module, "_AntipodalMeshBackend", _RandomBackend)
    service = AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
    inputs = _inputs()
    service.get_grasp_candidates(**inputs, generator=torch.Generator().manual_seed(1))
    service.get_grasp_candidates(**inputs, generator=torch.Generator().manual_seed(2))
    assert len(_RandomBackend.instances) == 1
    assert len(set(_RandomBackend.instances[0].selected_seeds)) == 2


def test_antipodal_empty_backend_result_is_empty_only_in_rich_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyBackend(_RandomBackend):
        def get_valid_grasp_poses(
            self, **_: Any
        ) -> tuple[bool, torch.Tensor, float, torch.Tensor]:
            return False, torch.eye(4), 0.0, torch.zeros(1)

    monkeypatch.setattr(facade_module, "_AntipodalMeshBackend", EmptyBackend)
    service = AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
    rich = service.get_grasp_candidates(**_inputs())
    legacy = service.get_valid_grasp_poses(**_inputs())
    assert rich.poses.shape == (2, 0, 4, 4)
    assert legacy[0][0].shape == (1, 4, 4)
    assert legacy[0][1].isinf().all()


def test_antipodal_backend_error_is_not_reported_as_no_grasp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenBackend(_RandomBackend):
        def get_valid_grasp_poses(
            self, **_: Any
        ) -> tuple[bool, torch.Tensor, float, torch.Tensor]:
            raise RuntimeError("collision backend unavailable")

    monkeypatch.setattr(facade_module, "_AntipodalMeshBackend", BrokenBackend)
    service = AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
    with pytest.raises(RuntimeError, match="collision backend unavailable"):
        service.get_grasp_candidates(**_inputs())


def test_antipodal_row_permutation_keeps_frozen_grasp_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(facade_module, "_AntipodalMeshBackend", _RandomBackend)
    service = AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
    inputs = _inputs()
    inputs["obj_poses"][1, 1, 3] = 0.1
    inputs["approach_direction"] = torch.tensor([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
    original = service.get_grasp_candidates(
        **inputs, generator=torch.Generator().manual_seed(22)
    )
    inputs["obj_poses"] = inputs["obj_poses"].flip(0)
    inputs["approach_direction"] = inputs["approach_direction"].flip(0)
    permuted = service.get_grasp_candidates(
        **inputs, generator=torch.Generator().manual_seed(22)
    )
    assert original.grasp_ids[0][0] == permuted.grasp_ids[1][0]
    assert original.grasp_ids[1][0] == permuted.grasp_ids[0][0]


def test_disk_pair_cache_includes_local_sampling_seed() -> None:
    backend = object.__new__(_AntipodalMeshBackend)
    backend._sampler_cfg = AntipodalSamplerCfg()
    backend._interactive_annotation = False
    backend._use_largest_connected_component = False
    inputs = _inputs()
    backend._sampling_seed = 1
    first = backend._get_cache_dir(inputs["mesh_vertices"], inputs["mesh_triangles"])
    backend._sampling_seed = 2
    second = backend._get_cache_dir(inputs["mesh_vertices"], inputs["mesh_triangles"])
    assert first != second


def test_seed_sample_cache_is_bounded_and_reuses_existing_collision_geometry() -> None:
    backend = object.__new__(_AntipodalMeshBackend)
    backend.device = torch.device("cpu")
    backend._sampled_pairs = {}
    collision_geometry = object()
    backend._collision_checker = collision_geometry
    for seed in range(7):
        backend._sampling_seed = seed
        backend._hit_point_pairs = torch.full((1, 2, 3), float(seed))
        backend._remember_sample()
    assert len(backend._sampled_pairs) == 4
    backend._select_sampling_seed(4)
    torch.testing.assert_close(backend.antipodal_pairs, torch.full((1, 2, 3), 4.0))
    assert backend._collision_checker is collision_geometry


def test_approach_rotation_consumes_only_supplied_local_rng() -> None:
    vectors = torch.tensor([[0.0, 0.0, -1.0]]).repeat(8, 1)
    global_state = torch.random.get_rng_state().clone()
    first = AntipodalSampler._random_rotate_unit_vectors(
        vectors, 0.3, generator=torch.Generator().manual_seed(7)
    )
    second = AntipodalSampler._random_rotate_unit_vectors(
        vectors, 0.3, generator=torch.Generator().manual_seed(7)
    )
    torch.testing.assert_close(first, second)
    assert torch.equal(global_state, torch.random.get_rng_state())
    torch.testing.assert_close(torch.linalg.vector_norm(first, dim=-1), torch.ones(8))
