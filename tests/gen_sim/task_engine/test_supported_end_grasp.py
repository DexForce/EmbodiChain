# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

from __future__ import annotations

from unittest.mock import Mock
from types import SimpleNamespace

import pytest
import torch

from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
from embodichain.toolkits.graspkit.pg_grasp._antipodal_backend import (
    _AntipodalMeshBackend,
)
from embodichain.toolkits.graspkit.pg_grasp.pose_generator import (
    AntipodalGraspPoseGenerator,
)
from embodichain.gen_sim.task_engine._task_program.supported_end import (
    for_axis_region,
    select_supported_end,
)


def backend(*, center_mode="centroid", body=None):
    result = object.__new__(_AntipodalMeshBackend)
    result.device = torch.device("cpu")
    result.vertices = torch.tensor(
        [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 1.0], [0.1, -0.1, 1.0]]
    )
    result._center_mode = center_mode
    values = torch.linspace(0.4, 0.6, 9) if body is None else torch.tensor(body)
    centers = torch.zeros(len(values) + 2, 3)
    centers[:, 2] = torch.cat((values, torch.tensor([0.95, 0.95])))
    offset = torch.tensor([0.05, 0.0, 0.0])
    result._hit_point_pairs = torch.stack((centers - offset, centers + offset), 1)
    result._max_deviation_angle = 0.35
    result._approach_direction_samples = 4
    result._max_candidates = 500
    result._filter_ground_collision = False
    result._collision_checker = Mock()
    result._collision_checker.query.side_effect = (
        lambda _object, poses, _widths, **kwargs: (
            poses[:, 2, 3] > 0.9,
            torch.zeros(len(poses)),
        )
    )
    result._antipodal_sampler = Mock()
    result._antipodal_sampler.sample.side_effect = AssertionError(
        "Do not annotate again."
    )
    return result


def query(instance, policy="mesh_extent", *, positive=True, axis=True):
    return instance.get_valid_grasp_poses(
        torch.eye(4),
        torch.tensor([0.0, 0.0, -1.0]),
        obj_longest_axis=torch.tensor([0.0, 0.0, 1.0]) if axis else None,
        is_positive_part=positive,
        axis_region_selector=(
            select_supported_end if policy == "supported_end" else None
        ),
    )


@pytest.mark.parametrize("positive", [True, False])
def test_empty_mesh_end_can_use_only_static_qualified_cached_support(positive):
    instance = backend()
    before = instance._hit_point_pairs.clone()
    assert query(instance, positive=positive)[0] is False
    valid, poses, widths, costs = query(instance, "supported_end", positive=positive)
    assert valid is True and torch.isfinite(costs).all()
    assert (widths > 0).all()
    if positive:
        assert (poses[:, 2, 3] > 0.53).all() and (poses[:, 2, 3] <= 0.6).all()
    else:
        assert (poses[:, 2, 3] < 0.47).all() and (poses[:, 2, 3] >= 0.4).all()
    assert torch.equal(instance._hit_point_pairs, before)
    instance._antipodal_sampler.sample.assert_not_called()


@pytest.mark.parametrize("center_mode", ["bounds", "centroid"])
def test_existing_legal_region_is_exact_fast_return_with_same_rng(center_mode):
    instance = backend(center_mode=center_mode, body=[0.75, 0.8, 0.85])
    torch.manual_seed(19)
    initial = torch.random.get_rng_state().clone()
    original = query(instance)
    expected_rng = torch.random.get_rng_state().clone()
    instance._collision_checker.query.reset_mock()
    torch.random.set_rng_state(initial)
    selected = query(instance, "supported_end")
    assert original[0] is selected[0] is True
    for left, right in zip(original[1:], selected[1:], strict=True):
        assert torch.equal(left, right)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    instance._collision_checker.query.assert_called_once()
    assert not hasattr(instance, "last_diagnostics")


def test_collision_empty_original_reuses_exact_single_direction_draw_budget():
    instance = backend()
    torch.manual_seed(31)
    initial = torch.random.get_rng_state().clone()
    assert query(instance)[0] is False
    expected = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial)
    assert query(instance, "supported_end")[0] is True
    assert torch.equal(torch.random.get_rng_state(), expected)


def test_angle_empty_original_still_uses_at_most_one_declared_direction_group():
    instance = backend()
    torch.manual_seed(7)
    initial = torch.random.get_rng_state().clone()
    query(instance, axis=False)
    expected = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial)
    assert query(instance, "supported_end", positive=False)[0] is True
    assert torch.equal(torch.random.get_rng_state(), expected)


def test_support_domain_is_computed_before_top_k():
    instance = backend()
    instance._max_candidates = 1
    valid, poses, _, _ = query(instance, "supported_end")
    assert valid is True and len(poses) == 1
    assert poses[0, 2, 3] > 0.53


def test_all_static_rejected_never_reinterprets_invalid_pairs():
    instance = backend()
    instance._collision_checker.query.side_effect = (
        lambda _object, poses, _widths, **kwargs: (
            torch.ones(len(poses), dtype=torch.bool),
            torch.zeros(len(poses)),
        )
    )
    assert query(instance, "supported_end")[0] is False
    assert instance._collision_checker.query.call_count == 2


def test_multiple_direction_variants_of_one_support_center_are_degenerate():
    instance = backend(body=[0.5])
    assert query(instance, "supported_end")[0] is False


def test_unpartitioned_call_does_not_change_geometry_or_rng():
    instance = backend()
    torch.manual_seed(29)
    state = torch.random.get_rng_state().clone()
    original = query(instance, axis=False)
    expected = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(state)
    selected = query(instance, "supported_end", axis=False)
    assert selected[0] is original[0]
    assert all(
        torch.equal(x, y) for x, y in zip(original[1:], selected[1:], strict=True)
    )
    assert torch.equal(torch.random.get_rng_state(), expected)


@pytest.mark.parametrize("policy", ["", "whole_mesh", True, None])
def test_unknown_policy_fails_before_candidate_processing(policy):
    instance = backend()
    with pytest.raises(ValueError, match="axis_region_policy"):
        for_axis_region(Mock(), policy)
    instance._collision_checker.query.assert_not_called()


@pytest.mark.parametrize("method", ["get_valid_grasp_poses", "get_grasp_candidates"])
@pytest.mark.parametrize("policy", ["mesh_extent", "supported_end"])
def test_service_policy_is_per_call_and_default_preserves_legacy_keyword_shape(
    method, policy
):
    generator = AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
    prepared = Mock(device=torch.device("cpu"))
    prepared.get_valid_grasp_poses.return_value = (
        True,
        torch.eye(4)[None],
        torch.tensor([0.04]),
        torch.zeros(1),
    )
    generator._backend = Mock(return_value=prepared)
    getattr(generator, method)(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        obj_poses=torch.eye(4)[None],
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        axis_region_selector=(
            select_supported_end if policy == "supported_end" else None
        ),
    )
    forwarded = prepared.get_valid_grasp_poses.call_args.kwargs
    if policy == "mesh_extent":
        assert "axis_region_selector" not in forwarded
    else:
        assert forwarded["axis_region_selector"] is select_supported_end
    assert not hasattr(generator.algorithm_cfg, "axis_region_selector")


def test_region_selector_sees_only_collision_qualified_candidates_before_top_k():
    instance = backend()
    instance._max_candidates = 1
    observed = []

    def selector(**values):
        centers = values["centers"]
        assert len(centers) > 1
        assert (centers[:, 2] <= 0.6).all()
        observed.append(centers.clone())
        return select_supported_end(**values)

    success, poses, _, _ = instance.get_valid_grasp_poses(
        torch.eye(4),
        torch.tensor([0.0, 0.0, -1.0]),
        obj_longest_axis=torch.tensor([0.0, 0.0, 1.0]),
        axis_region_selector=selector,
    )
    assert success and len(poses) == 1
    assert len(observed) == 1


@pytest.mark.parametrize("invalid", ["mask_shape", "mask_dtype", "center_nonfinite"])
def test_region_selector_rejects_invalid_selection(invalid):
    instance = backend()

    def selector(**values):
        mask, center = select_supported_end(**values)
        if invalid == "mask_shape":
            mask = mask[:-1]
        elif invalid == "mask_dtype":
            mask = mask.float()
        else:
            center = center * float("nan")
        return mask, center

    with pytest.raises(ValueError, match="Region selector"):
        instance.get_valid_grasp_poses(
            torch.eye(4),
            torch.tensor([0.0, 0.0, -1.0]),
            obj_longest_axis=torch.tensor([0.0, 0.0, 1.0]),
            axis_region_selector=selector,
        )


@pytest.mark.parametrize("policy", ["mesh_extent", "supported_end"])
def test_adaptive_sampling_preserves_region_policy_diagnostics(policy):
    from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
        GraspStrategy,
        select_grasp,
    )
    from embodichain.lab.sim.atomic_actions import AntipodalAffordance

    prepared = backend()
    generator = AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
    generator._backend = Mock(return_value=prepared)
    affordance = AntipodalAffordance(
        mesh_vertices=prepared.vertices,
        mesh_triangles=torch.tensor([[0, 1, 2], [0, 2, 3]]),
    )
    choice = GraspStrategy(
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([True]),
    )
    sample, direction = select_grasp(
        affordance,
        generator,
        torch.eye(4)[None],
        (choice,),
        SimpleNamespace(affordance_sampling=None, env_ids=torch.tensor([0])),
        "region-test",
        axis_region_policy=policy,
    )
    assert sample.success.tolist() == [policy == "supported_end"]
    assert (
        sample.metadata["adaptive_grasp"].get("axis_region_policy", "mesh_extent")
        == policy
    )
    assert direction.shape == (1, 3)
