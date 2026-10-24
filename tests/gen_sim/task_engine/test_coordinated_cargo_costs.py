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

"""E5 cargo clearance changes finite proposal preference, never gate acceptance."""

from __future__ import annotations

from itertools import product
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import coordinated_grasp as grasp


def _case(rotated: bool = False):
    carrier = torch.eye(4)[None]
    if rotated:
        carrier[0, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        carrier[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    cargo = carrier.clone()
    cargo[0, :3, 3] += carrier[0, :3, :3] @ torch.tensor([0.0, 0.0, 0.04])
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        scene=NS(
            entities={
                "carrier": NS(pose=carrier, confidence=1),
                "cargo": NS(pose=cargo, confidence=1),
            }
        ),
    )
    meshes = {
        "carrier": torch.tensor(list(product((-0.2, 0.2), (-0.2, 0.2), (-0.01, 0.1))))[
            None
        ],
        "cargo": torch.tensor(list(product((-0.01, 0.01), repeat=3)))[None],
    }
    poses = carrier.expand(2, -1, -1).clone()
    poses[:, :3, 3] += (
        torch.tensor([[0.0, 0.0, 0.04], [0.1, 0.0, 0.04]]) @ carrier[0, :3, :3].T
    )
    geometry = grasp.HandGeometry(
        body=torch.zeros(2, 1, 3),
        pads=torch.zeros(2, 2, 1, 3),
        gaps=torch.tensor([0.1, 0.01]),
    )
    return context, meshes, poses, geometry


@pytest.mark.parametrize("rotated", [False, True])
def test_observed_contents_prefer_clear_hand_without_dropping_candidates(
    rotated,
) -> None:
    context, meshes, poses, geometry = _case(rotated)
    costs, widths = torch.tensor([0.0, 0.1]), torch.tensor([0.05, 0.05])
    with grasp.cargo_clearance_scope(context, meshes, "carrier"):
        ranked = grasp._cargo_costs(geometry, poses, widths, costs)
    assert ranked.shape == costs.shape and torch.isfinite(ranked).all()
    assert int(ranked.argmin()) == 1
    assert ranked[1] == costs[1]
    assert torch.equal(costs, torch.tensor([0.0, 0.1]))
    assert grasp._cargo_costs(geometry, poses, widths, costs) is costs


def test_contents_poses_are_fresh_per_scope_and_not_long_lived() -> None:
    context, meshes, poses, geometry = _case()
    costs, widths = torch.tensor([0.0, 0.1]), torch.tensor([0.05, 0.05])
    with grasp.cargo_clearance_scope(context, meshes, "carrier"):
        context.scene.entities["cargo"].pose[0, 0, 3] = 0.5
        assert int(grasp._cargo_costs(geometry, poses, widths, costs).argmin()) == 1
    with grasp.cargo_clearance_scope(context, meshes, "carrier"):
        assert grasp._cargo_costs(geometry, poses, widths, costs) is costs


def test_cargo_scope_cleans_up_after_planning_exception() -> None:
    context, meshes, poses, geometry = _case()
    costs, widths = torch.tensor([0.0, 0.1]), torch.tensor([0.05, 0.05])
    with pytest.raises(RuntimeError, match="cancelled"):
        with grasp.cargo_clearance_scope(context, meshes, "carrier"):
            raise RuntimeError("cancelled")
    assert grasp._cargo_costs(geometry, poses, widths, costs) is costs


def test_no_content_or_unsupported_batch_keeps_existing_preference() -> None:
    context, meshes, poses, geometry = _case()
    costs, widths = torch.tensor([0.0, 0.1]), torch.tensor([0.05, 0.05])
    for selected in ({"carrier": meshes["carrier"]}, meshes):
        if len(selected) > 1:
            context.batch_size = 2
        with grasp.cargo_clearance_scope(context, selected, "carrier"):
            assert grasp._cargo_costs(geometry, poses, widths, costs) is costs


def test_all_near_candidates_remain_finite_and_available() -> None:
    context, meshes, poses, geometry = _case()
    poses[1] = poses[0]
    costs, widths = torch.tensor([0.0, 0.1]), torch.tensor([0.05, 0.05])
    with grasp.cargo_clearance_scope(context, meshes, "carrier"):
        ranked = grasp._cargo_costs(geometry, poses, widths, costs)
    assert ranked.shape == (2,) and torch.isfinite(ranked).all()
    assert ranked[0] < ranked[1]


def test_overlapping_proxies_prefer_shallow_overlap_instead_of_uniform_cost() -> None:
    context, meshes, poses, geometry = _case()
    poses[1] = poses[0]
    poses[1, 0, 3] = 0.009
    costs, widths = torch.tensor([0.0, 0.1]), torch.tensor([0.05, 0.05])
    with grasp.cargo_clearance_scope(context, meshes, "carrier"):
        ranked = grasp._cargo_costs(geometry, poses, widths, costs)
    assert torch.isfinite(ranked).all()
    assert int(ranked.argmin()) == 1


def test_clear_candidates_and_unavailable_geometry_do_not_change_base_costs() -> None:
    context, meshes, poses, geometry = _case()
    poses[:, 0, 3] += 1
    costs, widths = torch.tensor([0.0, 0.1]), torch.tensor([0.05, 0.05])
    before = torch.get_rng_state().clone()
    with grasp.cargo_clearance_scope(context, meshes, "carrier"):
        assert grasp._cargo_costs(geometry, poses, widths, costs) is costs

        def broken(*args):
            raise RuntimeError("geometry unavailable")

        unavailable = NS(gaps=geometry.gaps, at_width=broken)
        assert grasp._cargo_costs(unavailable, poses, widths, costs) is costs
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("role", ["left", "right"])
def test_sampler_keeps_the_same_qualified_32_pose_width_pairs(
    monkeypatch, role
) -> None:
    _, _, _, geometry = _case()
    poses = torch.eye(4)[None].repeat(40, 1, 1)
    poses[:, 0, 3] = torch.arange(40)
    widths = 0.05 + torch.arange(40) * 0.00001
    costs = torch.arange(40, dtype=torch.float32)
    item = {
        "is_success": True,
        "grasp_poses": poses,
        "open_lengths": widths,
        "total_cost": costs,
    }
    monkeypatch.setattr(
        grasp.AntipodalGraspPoseGenerator,
        "get_dual_arm_valid_grasp_poses",
        lambda *args, **kwargs: [{"left": item, "right": item}],
    )
    sampler = object.__new__(grasp._HandSampler)
    sampler._role, sampler._geometry = role, geometry
    sampler._geometry_key = lambda *args: "mesh"
    sampler._collision_cfg = NS(filter_ground_collision=False)
    sampler._validators = {
        "mesh": NS(
            fit_contacts=lambda object_pose, trial, gaps: (trial, gaps),
            query=lambda object_pose, trial, gaps, **kwargs: (
                torch.zeros(len(gaps), dtype=torch.bool),
                gaps,
            ),
        )
    }
    seen = []

    def rerank(selected_geometry, selected_poses, selected_widths, selected_costs):
        seen.append(
            (selected_poses.clone(), selected_widths.clone(), selected_costs.clone())
        )
        return selected_costs.flip(0)

    monkeypatch.setattr(grasp, "_cargo_costs", rerank)
    result = sampler.get_dual_arm_valid_grasp_poses(
        mesh_vertices=torch.zeros(1, 4, 3),
        mesh_triangles=torch.zeros(1, 4, 3),
        obj_poses=torch.eye(4)[None],
    )[0][role]
    assert len(seen) == 1
    assert torch.equal(result["grasp_poses"], poses[:32])
    assert torch.equal(result["open_lengths"], widths[:32])
    assert torch.equal(seen[0][2], costs[:32])
    assert torch.equal(result["total_cost"], costs[:32].flip(0))


def test_nested_scope_and_budget_exception_restore_the_outer_snapshot() -> None:
    from embodichain.gen_sim.task_engine._task_program._search_cost import (
        PlanningBudgetExceeded,
    )

    context, meshes, poses, geometry = _case()
    costs, widths = torch.tensor([0.0, 0.1]), torch.tensor([0.05, 0.05])
    with grasp.cargo_clearance_scope(context, meshes, "carrier"):
        with pytest.raises(PlanningBudgetExceeded):
            with grasp.cargo_clearance_scope(
                context, {"carrier": meshes["carrier"]}, "carrier"
            ):
                assert grasp._cargo_costs(geometry, poses, widths, costs) is costs
                raise PlanningBudgetExceeded("test", 90.0, 91.0)
        assert int(grasp._cargo_costs(geometry, poses, widths, costs).argmin()) == 1
    assert grasp._cargo_costs(geometry, poses, widths, costs) is costs
