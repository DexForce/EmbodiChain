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

"""Task grasp filters preserve baseline candidate and failure contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.grasp_filter import (
    E6ApproachGraspPoseGenerator,
    GraspRule,
    TaskGraspPoseGenerator,
    accepted_candidates,
    geometry_key,
    opening_envelope_mask,
)
from embodichain.toolkits.graspkit import (
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)

__all__: list[str] = []

VERTICES = torch.tensor([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.20], [0.0, 0.01, 0.10]])
TRIANGLES = torch.tensor([[0, 1, 2]])


def test_opening_envelope_rejects_wide_or_off_center_object() -> None:
    vertices = torch.tensor([[-0.07, -0.02, 0.0], [0.07, 0.02, 0.0]])
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    poses[2] = poses[1]
    poses[2, 1, 3] = 0.05
    assert opening_envelope_mask(vertices, poses, torch.eye(4), 0.128).tolist() == [
        False,
        True,
        False,
    ]
    world = torch.eye(4)
    world[:3, 3] = torch.tensor([2.0, -3.0, 0.5])
    assert opening_envelope_mask(vertices, world @ poses, world, 0.128).tolist() == [
        False,
        True,
        False,
    ]


def test_unconstrained_pick_still_checks_opening_envelope() -> None:
    generator = TaskGraspPoseGenerator(CandidateGenerator(((0.1,),)), ())
    rows = generator.get_valid_grasp_poses(
        mesh_vertices=torch.tensor([[-0.08, 0.0, 0.0], [0.08, 0.0, 0.0]]),
        mesh_triangles=torch.empty(0, 3, dtype=torch.long),
        obj_poses=torch.eye(4).unsqueeze(0),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )
    assert torch.isinf(rows[0][1]).all()


def test_e6_clearance_shifts_only_within_pad_workspace() -> None:
    from embodichain.gen_sim.task_engine._task_program.e6_clearance import HandClearance

    clearance = HandClearance(
        points=torch.tensor([[0.0, 0.0, 0.02]]),
        aperture=0.017,
        max_retreat=0.028,
        table_z=0.0,
    )
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[:, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    poses[:, 2, 3] = torch.tensor([0.008, 0.003, -0.020, 0.1])
    widths = torch.tensor([0.006, 0.006, 0.006, 0.08])
    costs = torch.tensor([1.0, 2.0, 3.0, 0.0])
    shifted, actual_widths, filtered = clearance.filter(poses, widths, costs)
    assert torch.allclose(shifted[:2, 2, 3], torch.tensor([0.028, 0.028]))
    assert torch.equal(actual_widths, widths)
    assert torch.equal(filtered[:2], costs[:2])
    assert torch.isinf(filtered[2:]).all()
    assert torch.equal(poses[:, 2, 3], torch.tensor([0.008, 0.003, -0.020, 0.1]))
    assert torch.equal(costs, torch.tensor([1.0, 2.0, 3.0, 0.0]))


def test_e6_clearance_provider_fails_closed_and_preserves_unrelated_grasps() -> None:
    from embodichain.gen_sim.task_engine._task_program.e6_clearance import HandClearance

    delegate = CandidateGenerator(((0.02, 0.04),))
    provider = E6ApproachGraspPoseGenerator(
        delegate,
        frozenset({geometry_key(VERTICES, TRIANGLES)}),
        clearance=HandClearance(torch.zeros(1, 3), 0.001, 0.028, 0.0),
    )
    kwargs = dict(
        mesh_vertices=VERTICES,
        mesh_triangles=TRIANGLES,
        obj_poses=torch.eye(4)[None],
        approach_direction=torch.tensor([0.0, -1.0, 0.0]),
    )
    success, _, _ = provider.get_best_grasp_poses(**kwargs)
    assert success.tolist() == [False]
    success, _, width = provider.get_best_grasp_poses(
        **{**kwargs, "mesh_vertices": VERTICES + 1.0}
    )
    assert success.tolist() == [True]
    assert width.tolist() == pytest.approx([0.05])


def test_e6_hand_clearance_uses_named_fk_and_includes_fixed_descendants() -> None:
    from embodichain.gen_sim.task_engine._task_program.e6_clearance import (
        build_hand_clearance,
    )

    links = ["tool", "a_finger_pad", "b_finger_pad", "fixed_mount"]
    master = SimpleNamespace(name="gripper", parent_link_name="tool")
    fixed = SimpleNamespace(name="fixed", parent_link_name="tool")
    calls = []

    def fk(*, qpos, link_names, qpos_joint_names):
        calls.append((link_names, qpos_joint_names))
        poses = torch.eye(4).repeat(len(qpos), len(link_names), 1, 1)
        poses[:, :, 2, 3] = 1.0
        for index, link in enumerate(link_names):
            if link in links[1:3]:
                sign = -1 if link == links[1] else 1
                poses[:, index, 0, 3] = sign * (0.04 - 0.03 * qpos[:, 1])
        return poses

    robot = SimpleNamespace(
        cfg=SimpleNamespace(
            control_parts={"hand": ["gripper"]},
            solver_cfg={"arm": SimpleNamespace(end_link_name="tool", tcp=torch.eye(4))},
        ),
        link_names=links,
        joint_names=["arm_joint", "gripper"],
        get_parent_joint_chain=lambda link: (
            [] if link == "tool" else [fixed if link == "fixed_mount" else master]
        ),
        get_qpos=lambda: torch.zeros(1, 2),
        compute_fk=fk,
        get_link_vert_face=lambda link: (
            torch.tensor([[-0.001, 0.0, -0.03], [0.001, 0.0, 0.03]]),
            torch.empty(0, 3, dtype=torch.int64),
        ),
    )
    profile = build_hand_clearance(
        robot,
        motion_part="arm",
        hand_part="hand",
        commands={"open": [0.0], "grasp": [1.0]},
        table_z=0.5,
    )
    assert profile.aperture == pytest.approx(0.023, abs=1e-6)
    assert profile.max_retreat == pytest.approx(0.026, abs=1e-6)
    assert profile.points.shape == (7 * 4 * 2, 3)
    assert profile.points[:, 2].abs().max() == pytest.approx(0.03, abs=1e-6)
    assert calls == [(links + ["tool"], ["arm_joint", "gripper"])]


def test_e6_short_pads_reject_all_candidates_without_extrapolating() -> None:
    from embodichain.gen_sim.task_engine._task_program.e6_clearance import HandClearance

    clearance = HandClearance(torch.zeros(1, 3), 0.01, 0.005, 0.0)
    poses = torch.eye(4)[None]
    shifted, _, costs = clearance.filter(poses, torch.tensor([0.005]), torch.zeros(1))
    assert torch.equal(poses, shifted)
    assert torch.isinf(costs).all()


class CandidateGenerator(ParallelJawGraspPoseGenerator):
    def __init__(
        self,
        rows: tuple[tuple[float, ...], ...],
        *,
        opening_width: float = 0.05,
        candidate_rotation: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            ParallelJawGripperModelCfg(
                model_id="test",
                finger_length=0.10,
                finger_width=0.02,
                finger_thickness=0.01,
            )
        )
        self.rows = rows
        self.opening_width = opening_width
        self.candidate_rotation = candidate_rotation
        self.best_directions = []

    def get_valid_grasp_poses(self, **kwargs):
        poses = kwargs["obj_poses"]
        results = []
        for row, heights in enumerate(self.rows):
            local = torch.eye(4).repeat(len(heights), 1, 1)
            if self.candidate_rotation is not None:
                local[:, :3, :3] = self.candidate_rotation
            local[:, 2, 3] = torch.tensor(heights)
            results.append(
                (poses[row] @ local, torch.arange(len(heights), dtype=torch.float32))
            )
        return results

    def get_best_grasp_poses(self, **kwargs):
        self.best_directions.append(kwargs["approach_direction"].clone())
        poses = kwargs["obj_poses"]
        return (
            torch.ones(len(poses), dtype=torch.bool),
            poses.clone(),
            torch.full((len(poses),), self.opening_width),
        )

    def get_grasp_candidates(self, **kwargs):
        self.best_directions.append(kwargs["approach_direction"].clone())
        return [
            (poses, torch.full((len(poses),), self.opening_width), costs)
            for poses, costs in self.get_valid_grasp_poses(**kwargs)
        ]

    def get_dual_arm_valid_grasp_poses(self, **kwargs):
        return [None] * len(kwargs["obj_poses"])


def test_e6_candidate_metadata_preserves_widths_and_approach() -> None:
    delegate = CandidateGenerator(((0.02, 0.04),))
    provider = E6ApproachGraspPoseGenerator(
        delegate, frozenset({geometry_key(VERTICES, TRIANGLES)})
    )
    poses, widths, costs = provider.get_grasp_candidates(
        mesh_vertices=VERTICES,
        mesh_triangles=TRIANGLES,
        obj_poses=torch.eye(4)[None],
        approach_direction=torch.tensor([0.0, -1.0, 0.0]),
    )[0]
    assert torch.allclose(widths, torch.tensor([0.05, 0.05]))
    assert torch.equal(costs, torch.tensor([0.0, 1.0]))
    assert poses.shape == (2, 4, 4)
    assert torch.allclose(
        delegate.best_directions[0], torch.tensor([0.0, -(0.5**0.5), -(0.5**0.5)])
    )


def test_e6_clearance_adds_short_axis_variant_for_legacy_grip_bar() -> None:
    from embodichain.gen_sim.task_engine._task_program.e6_clearance import HandClearance

    vertices = torch.tensor(
        [
            [x, y, z]
            for x in (-0.05, 0.05)
            for y in (-0.005, 0.005)
            for z in (-0.005, 0.005)
        ]
    )
    triangles = torch.tensor([[0, 1, 2]])
    provider = E6ApproachGraspPoseGenerator(
        CandidateGenerator(((0.1,),), opening_width=0.1),
        frozenset({geometry_key(vertices, triangles)}),
        clearance=HandClearance(torch.zeros(1, 3), 0.02, 0.028, 0.0),
    )
    poses, widths, costs = provider.get_grasp_candidates(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        obj_poses=torch.eye(4)[None],
        approach_direction=torch.tensor([0.0, -1.0, 0.0]),
    )[0]
    assert poses.shape == (1, 4, 4)
    assert widths.tolist() == pytest.approx([0.01])
    assert torch.isfinite(costs).all()
    assert torch.allclose(poses[0, :3, 0], torch.tensor([0.0, 1.0, 0.0]))
    success, _, selected_width = provider.get_best_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        obj_poses=torch.eye(4)[None],
        approach_direction=torch.tensor([0.0, -1.0, 0.0]),
    )
    assert success.tolist() == [True]
    assert selected_width.tolist() == pytest.approx([0.01])


def test_e6_short_axis_variant_does_not_replace_valid_stock_candidate() -> None:
    from embodichain.gen_sim.task_engine._task_program.e6_clearance import HandClearance

    provider = E6ApproachGraspPoseGenerator(
        CandidateGenerator(((0.1,),), opening_width=0.005),
        frozenset({geometry_key(VERTICES, TRIANGLES)}),
        clearance=HandClearance(torch.zeros(1, 3), 0.2, 0.028, 0.0),
    )
    success, pose, width = provider.get_best_grasp_poses(
        mesh_vertices=VERTICES,
        mesh_triangles=TRIANGLES,
        obj_poses=torch.eye(4)[None],
        approach_direction=torch.tensor([0.0, -1.0, 0.0]),
    )
    assert success.tolist() == [True]
    assert width.tolist() == pytest.approx([0.005])
    assert torch.allclose(pose[0, :3, :3], torch.eye(3))
    assert pose[0, 2, 3].item() == pytest.approx(0.08)


def test_e6_short_axis_fallback_requires_stock_closing_along_long_axis() -> None:
    from embodichain.gen_sim.task_engine._task_program.e6_clearance import HandClearance

    vertices = torch.tensor(
        [
            [x, y, z]
            for x in (-0.05, 0.05)
            for y in (-0.005, 0.005)
            for z in (-0.005, 0.005)
        ]
    )
    triangles = torch.tensor([[0, 1, 2]])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    provider = E6ApproachGraspPoseGenerator(
        CandidateGenerator(((0.1,),), opening_width=0.1, candidate_rotation=rotation),
        frozenset({geometry_key(vertices, triangles)}),
        clearance=HandClearance(torch.zeros(1, 3), 0.02, 0.028, 0.0),
    )
    success, _, _ = provider.get_best_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        obj_poses=torch.eye(4)[None],
        approach_direction=torch.tensor([0.0, -1.0, 0.0]),
    )
    assert success.tolist() == [False]


def rule(*, upper_half=True, margin=0.02) -> GraspRule:
    return GraspRule(
        "can",
        "release",
        geometry_key(VERTICES, TRIANGLES),
        (0.0, 0.0, 1.0),
        0.10,
        upper_half,
        tuple(torch.eye(4).reshape(-1).tolist()),
        0.0,
        margin,
    )


def sample(provider, poses, **kwargs):
    return provider.get_valid_grasp_poses(
        mesh_vertices=VERTICES,
        mesh_triangles=TRIANGLES,
        obj_poses=poses,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        **kwargs,
    )


def test_filter_uses_object_local_region_independently_for_each_row() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    before = poses.clone()
    provider = TaskGraspPoseGenerator(
        CandidateGenerator(((0.08, 0.15), (0.08,))), (rule(),)
    ).for_pick("can", "release", VERTICES, TRIANGLES)
    rows = sample(provider, poses)
    assert rows[0][0].shape == (1, 4, 4)
    assert rows[0][0][0, 2, 3].item() == pytest.approx(0.15)
    assert torch.isfinite(rows[0][1]).all()
    assert rows[1][0].shape == (1, 4, 4)
    assert torch.isinf(rows[1][1]).all()
    torch.testing.assert_close(poses, before)


def test_release_filter_rejects_a_low_grasp_before_stock_ik() -> None:
    provider = TaskGraspPoseGenerator(
        CandidateGenerator(((0.06, 0.10),)), (rule(upper_half=False),)
    ).for_pick("can", "release", VERTICES, TRIANGLES)
    rows = sample(provider, torch.eye(4).unsqueeze(0))
    assert rows[0][0][:, 2, 3].tolist() == pytest.approx([0.10])


def test_symmetric_roll_variants_have_identical_release_clearance() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, 2, 3] = torch.tensor([0.06, 0.15])
    mirrored = poses.clone()
    mirrored[:, :3, :2] *= -1
    model = CandidateGenerator(((),)).gripper_model
    torch.testing.assert_close(
        accepted_candidates(poses, torch.eye(4), rule(), model),
        accepted_candidates(mirrored, torch.eye(4), rule(), model),
    )


def test_no_candidate_is_encoded_as_ineligible_not_as_a_fake_success() -> None:
    provider = TaskGraspPoseGenerator(
        CandidateGenerator(((), (0.05,))), (rule(),)
    ).for_pick("can", "release", VERTICES, TRIANGLES)
    rows = sample(provider, torch.eye(4).repeat(2, 1, 1))
    assert all(
        p.shape == (1, 4, 4) and torch.isfinite(p).all() and torch.isinf(c).all()
        for p, c in rows
    )


def test_end_specific_sampling_keeps_its_baseline_contract() -> None:
    provider = TaskGraspPoseGenerator(CandidateGenerator(((0.06, 0.10),)), (rule(),))
    rows = sample(
        provider,
        torch.eye(4).unsqueeze(0),
        obj_longest_axis=torch.tensor([0.0, 0.0, 1.0]),
        is_positive_part=False,
    )
    assert rows[0][0].shape[0] == 2
    assert torch.isfinite(rows[0][1]).all()


def test_rule_identity_must_match_the_bound_object_and_geometry() -> None:
    provider = TaskGraspPoseGenerator(CandidateGenerator(((0.15,),)), (rule(),))
    provider.require_rule("can", "release", VERTICES, TRIANGLES)
    with pytest.raises(ValueError, match="matching"):
        provider.require_rule("other", "release", VERTICES, TRIANGLES)
    with pytest.raises(ValueError, match="matching"):
        provider.require_rule("can", "release", VERTICES * 2, TRIANGLES)


def test_future_pick_rule_does_not_filter_handover_or_another_pick() -> None:
    provider = TaskGraspPoseGenerator(CandidateGenerator(((0.06, 0.15),)), (rule(),))
    poses = torch.eye(4).unsqueeze(0)
    assert sample(provider, poses)[0][0].shape[0] == 2
    scoped = provider.for_pick("can", "release", VERTICES, TRIANGLES)
    assert sample(scoped, poses)[0][0][:, 2, 3].tolist() == pytest.approx([0.15])
    assert sample(provider, poses)[0][0].shape[0] == 2


def test_pick_rule_does_not_intersect_other_targets_with_same_geometry() -> None:
    from dataclasses import replace

    provider = TaskGraspPoseGenerator(
        CandidateGenerator(((0.15,),)),
        (rule(), replace(rule(), target_id="later", midpoint=0.2)),
    )
    poses = torch.eye(4).unsqueeze(0)
    first = provider.for_pick("can", "release", VERTICES, TRIANGLES)
    later = provider.for_pick("can", "later", VERTICES, TRIANGLES)
    assert torch.isfinite(sample(first, poses)[0][1]).all()
    assert torch.isinf(sample(later, poses)[0][1]).all()


def test_e6_approach_is_scoped_to_bound_handle_geometry() -> None:
    delegate = CandidateGenerator(((),))
    provider = E6ApproachGraspPoseGenerator(
        delegate, frozenset({geometry_key(VERTICES, TRIANGLES)})
    )
    provider.get_best_grasp_poses(
        mesh_vertices=VERTICES,
        mesh_triangles=TRIANGLES,
        obj_poses=torch.eye(4).unsqueeze(0),
        approach_direction=torch.tensor([[0.0, -1.0, 0.0]]),
    )
    torch.testing.assert_close(
        delegate.best_directions[-1],
        torch.tensor([[0.0, -math.sqrt(0.5), -math.sqrt(0.5)]]),
    )

    provider.get_best_grasp_poses(
        mesh_vertices=VERTICES * 2,
        mesh_triangles=TRIANGLES,
        obj_poses=torch.eye(4).unsqueeze(0),
        approach_direction=torch.tensor([[0.0, -1.0, 0.0]]),
    )
    torch.testing.assert_close(
        delegate.best_directions[-1], torch.tensor([[0.0, -1.0, 0.0]])
    )
