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

"""GenSim proposals preserve the former deferred pool and random stream."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.pick_proposals import (
    native_center_proposals,
)
from embodichain.gen_sim.task_engine._task_program.actions import (
    GenSimPickUp,
    _PickSelection,
    _PICK_SELECTION,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUpOptions
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp._antipodal_backend import (
    _AntipodalMeshBackend,
)


def _fixture(pair_count: int, center_mode: str, maximum: int):
    vertices = torch.tensor([[-0.1, -0.1, 0.0], [0.1, 0.1, 0.2], [0.0, 0.1, 0.1]])
    triangles = torch.tensor([[0, 1, 2]])
    with torch.random.fork_rng():
        torch.manual_seed(71)
        centers = torch.rand(pair_count, 3) * 0.1
        axis = torch.nn.functional.normalize(torch.randn(pair_count, 3), dim=-1)
        pairs = torch.stack((centers - axis * 0.02, centers + axis * 0.02), dim=1)
    generator = AntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(
            model_id="native-proposal-test",
            min_opening_width=0.002,
            max_opening_width=0.128,
            finger_length=0.13,
            finger_width=0.03,
            finger_thickness=0.01,
            palm_depth=0.08,
        ),
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(
            center_mode=center_mode, max_candidates=maximum
        ),
        collision_cfg=ParallelJawGraspCollisionCfg(filter_ground_collision=False),
    )
    generator.prepare_mesh = Mock(return_value=pairs.clone())
    reference = object.__new__(_AntipodalMeshBackend)
    reference.device = torch.device("cpu")
    reference.vertices = vertices
    reference._hit_point_pairs = pairs
    reference._center_mode = center_mode
    reference._max_deviation_angle = generator.algorithm_cfg.approach_deviation_angle
    reference._approach_direction_samples = (
        generator.algorithm_cfg.approach_direction_samples
    )
    reference._max_candidates = 100_000
    reference._filter_ground_collision = False
    # The former deferral erased this mask before ranking. Returning zero here
    # exercises the unchanged shared geometry/ranking without a deferral API.
    reference._collision_checker = SimpleNamespace(
        query=lambda _object, poses, widths, **kwargs: (
            torch.zeros(len(poses), dtype=torch.bool),
            torch.zeros(len(poses)),
        )
    )
    return generator, reference, vertices, triangles


@pytest.mark.parametrize("seed", [0, 19])
@pytest.mark.parametrize("center_mode", ["bounds", "centroid"])
@pytest.mark.parametrize("pair_count,maximum", [(0, 5), (16, 50), (80, 5)])
def test_raw_pose_cost_order_and_rng_match_previous_geometry_before_cap(
    seed, center_mode, pair_count, maximum
):
    generator, reference, vertices, triangles = _fixture(
        pair_count, center_mode, maximum
    )
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    poses[1, :3, 3] = torch.tensor([0.4, -0.2, 0.6])
    directions = torch.tensor([[0.0, 0.0, -1.0], [0.8, 0.0, -0.6]])
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        initial = torch.get_rng_state()
        expected = []
        for pose, direction in zip(poses, directions, strict=True):
            valid, selected, _, costs = reference.get_valid_grasp_poses(pose, direction)
            selected = selected.reshape(-1, 4, 4)
            costs = (
                costs.reshape(-1) if valid else torch.full((len(selected),), torch.inf)
            )
            expected.append((selected, costs))
        after_reference = torch.get_rng_state()
        torch.set_rng_state(initial)
        actual = native_center_proposals(
            generator,
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            obj_poses=poses,
            approach_direction=directions,
        )
        assert torch.equal(torch.get_rng_state(), after_reference)
    assert len(actual) == len(expected)
    for (poses_a, costs_a), (poses_b, costs_b) in zip(actual, expected, strict=True):
        torch.testing.assert_close(poses_a, poses_b, rtol=0, atol=0)
        torch.testing.assert_close(costs_a, costs_b, rtol=0, atol=0, equal_nan=True)
    generator.prepare_mesh.assert_called_once_with(
        mesh_vertices=vertices, mesh_triangles=triangles
    )


def test_native_proposals_cannot_bypass_an_existing_ground_policy():
    generator, _, vertices, triangles = _fixture(16, "bounds", 5)
    generator._collision_cfg.filter_ground_collision = True
    with pytest.raises(ValueError, match="ground"):
        native_center_proposals(
            generator,
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            obj_poses=torch.eye(4)[None],
            approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        )
    generator.prepare_mesh.assert_not_called()


@pytest.mark.parametrize("limit", [1, 3, 50])
def test_native_qualification_precedes_cap_and_uses_final_tcp_rolls(limit):
    action = GenSimPickUp()
    action._planning_services = SimpleNamespace(device=torch.device("cpu"))
    poses = torch.eye(4).repeat(1, 51, 1, 1)
    poses[:, :, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    poses[0, :, 0, 3] = torch.arange(51)
    options = PickUpOptions(approach_alignment_max_angle=0.1)
    options.grasp_frame_to_eef[0, 3] = 0.25
    seen = []

    def shared(values, *_args):
        seen.append(values.clone())
        return values @ options.grasp_frame_to_eef, torch.ones(
            values.shape[:2], dtype=torch.bool
        )

    action._shared_grasp_variants = shared
    screen = SimpleNamespace(mask=lambda values, _: values[..., 0, 3] > 50.1)
    selection = _PickSelection(closing_geometry=screen)
    token = _PICK_SELECTION.set(selection)
    try:
        selected, valid = action._select_feasible_grasp_variants(
            poses,
            torch.zeros(1, 1),
            torch.eye(4)[None],
            object(),
            options,
            torch.tensor([0.0, 0.0, -1.0]),
            candidate_costs=torch.arange(51, dtype=torch.float32)[None],
            candidate_limit=limit,
        )
    finally:
        _PICK_SELECTION.reset(token)
    assert len(seen) == 1 and seen[0].shape[1] == 1
    assert seen[0][0, 0, 0, 3] == 50
    assert valid.sum() == 1 and valid[0, 50]
    assert selected[0, 50, 0, 3] == 50.25
    assert selection.closing_scene_checks[-1]["qualified_before_limit"] == 1


def test_native_candidate_cap_keeps_configured_count_and_original_cost_order():
    action = GenSimPickUp()
    action._planning_services = SimpleNamespace(device=torch.device("cpu"))
    poses = torch.eye(4).repeat(1, 6, 1, 1)
    poses[:, :, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    poses[0, :, 0, 3] = torch.arange(6)
    seen = []

    def shared(values, *_args):
        seen.append(values.clone())
        return values, torch.ones(values.shape[:2], dtype=torch.bool)

    action._shared_grasp_variants = shared
    checked = []

    def mask(values, _):
        checked.extend(values[0, :, 0, 0, 3].tolist())
        return torch.ones(values.shape[:3], dtype=torch.bool)

    screen = SimpleNamespace(mask=mask)
    token = _PICK_SELECTION.set(_PickSelection(closing_geometry=screen))
    try:
        _, valid = action._select_feasible_grasp_variants(
            poses,
            torch.zeros(1, 1),
            torch.eye(4)[None],
            object(),
            PickUpOptions(approach_alignment_max_angle=0.1),
            torch.tensor([0.0, 0.0, -1.0]),
            candidate_costs=torch.tensor([[0.9, 0.1, 0.7, 0.5, 0.3, 0.8]]),
            candidate_limit=3,
        )
    finally:
        _PICK_SELECTION.reset(token)
    assert seen[0][0, :, 0, 3].tolist() == [1, 4, 3]
    assert valid.sum() == 3
    assert checked == [1, 4, 3]


def test_native_geometry_is_not_queried_for_direction_incompatible_columns():
    action = GenSimPickUp()
    action._planning_services = SimpleNamespace(device=torch.device("cpu"))
    poses = torch.eye(4).repeat(1, 3, 1, 1)
    poses[0, 0, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    seen = []

    def mask(values, _):
        assert (values[..., 2, 2] < -0.99).all()
        seen.append(values.clone())
        return torch.ones(values.shape[:3], dtype=torch.bool)

    action._shared_grasp_variants = lambda values, *_: (
        values,
        torch.ones(values.shape[:2], dtype=torch.bool),
    )
    token = _PICK_SELECTION.set(
        _PickSelection(closing_geometry=SimpleNamespace(mask=mask))
    )
    try:
        _, valid = action._select_feasible_grasp_variants(
            poses,
            torch.zeros(1, 1),
            torch.eye(4)[None],
            object(),
            PickUpOptions(approach_alignment_max_angle=0.1),
            torch.tensor([0.0, 0.0, -1.0]),
            candidate_costs=torch.tensor([[0.5, 0.0, 0.1]]),
            candidate_limit=3,
        )
    finally:
        _PICK_SELECTION.reset(token)
    assert valid.tolist() == [[True, False, False]]
    assert len(seen) == 1 and seen[0].shape == (1, 1, 2, 4, 4)


@pytest.mark.parametrize("limit", [1, 3, 20])
def test_lazy_native_limit_matches_exhaustive_ranking_with_rejected_prefix(limit):
    action = GenSimPickUp()
    action._planning_services = SimpleNamespace(device=torch.device("cpu"))
    poses = torch.eye(4).repeat(1, 10, 1, 1)
    poses[:, :, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    poses[0, :, 0, 3] = torch.arange(10)
    queried, retained = [], []

    def mask(values, _):
        queried.extend(values[0, :, 0, 0, 3].tolist())
        return values[..., 0, 3].remainder(2) == 1

    def shared(values, *_):
        retained.extend(values[0, :, 0, 3].tolist())
        return values, torch.ones(values.shape[:2], dtype=torch.bool)

    action._shared_grasp_variants = shared
    token = _PICK_SELECTION.set(
        _PickSelection(closing_geometry=SimpleNamespace(mask=mask))
    )
    try:
        action._select_feasible_grasp_variants(
            poses,
            torch.zeros(1, 1),
            torch.eye(4)[None],
            object(),
            PickUpOptions(approach_alignment_max_angle=0.1),
            torch.tensor([0.0, 0.0, -1.0]),
            candidate_costs=torch.arange(10, dtype=torch.float32)[None],
            candidate_limit=limit,
        )
    finally:
        _PICK_SELECTION.reset(token)
    assert retained == [1, 3, 5, 7, 9][:limit]
    assert queried == list(range(min(10, 2 * limit)))
