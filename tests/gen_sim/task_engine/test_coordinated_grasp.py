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

"""E5 geometry checks must not change single-arm grasp services."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.coordinated_grasp import (
    CoordinatedGraspGenerator,
    HandGeometry,
    HandCollisionChecker,
    build_hand_geometry,
    TriangleTarget,
)
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg


def hand() -> HandGeometry:
    # Parallel pads close from 100 mm to zero; a body point stays above the target.
    pads = torch.tensor(
        [
            [
                [[-0.06, -0.01, -0.02], [-0.05, 0.01, 0.02]],
                [[0.05, -0.01, -0.02], [0.06, 0.01, 0.02]],
            ],
            [
                [[-0.01, -0.01, -0.02], [0.0, 0.01, 0.02]],
                [[0.0, -0.01, -0.02], [0.01, 0.01, 0.02]],
            ],
        ]
    )
    body = torch.tensor([[[0.0, 0.0, -0.10]], [[0.0, 0.0, -0.10]]])
    return HandGeometry(body, pads, torch.tensor([0.10, 0.0]))


class Slab:
    def query_batch_points(self, points, **kwargs):
        distance = points[..., 0].abs() - 0.01
        distance = torch.maximum(distance, points[..., 2].abs() - 0.03)
        return distance <= 0, distance


def test_contact_width_is_full_gap_and_geometry_uses_actual_open() -> None:
    geometry = hand()
    body, pads = geometry.at_width(torch.tensor([0.02]))
    assert pads[0, 1, :, 0].min() - pads[0, 0, :, 0].max() == pytest.approx(0.02)
    checker = HandCollisionChecker(geometry, Slab())
    rejected, _ = checker.query(torch.eye(4), torch.eye(4)[None], torch.tensor([0.02]))
    assert rejected.tolist() == [False]


def test_finite_pad_fit_corrects_sparse_pair_gap_and_center_without_mutation() -> None:
    poses = torch.eye(4)[None]
    poses[:, 0, 3] = 0.004
    widths = torch.tensor([0.014])
    before = poses.clone()
    checker = HandCollisionChecker(hand(), Slab())
    fitted, gaps = checker.fit_contacts(torch.eye(4), poses, widths)
    assert fitted[0, 0, 3] == pytest.approx(0.0, abs=2e-5)
    assert gaps[0] == pytest.approx(0.0202, abs=2e-5)
    rejected, _ = checker.query(torch.eye(4), fitted, gaps)
    assert rejected.tolist() == [False]
    assert torch.equal(poses, before)
    assert widths.tolist() == pytest.approx([0.014])


def test_clear_but_noncontacting_pads_are_not_a_grasp() -> None:
    poses = torch.eye(4)[None]
    poses[:, 0, 3] = 0.3
    rejected, _ = HandCollisionChecker(hand(), Slab()).query(
        torch.eye(4), poses, torch.tensor([0.02])
    )
    assert rejected.tolist() == [True]


def test_triangle_target_preserves_hollow_space_and_rejects_open_mesh() -> None:
    import trimesh

    mesh = trimesh.creation.annulus(r_min=0.03, r_max=0.05, height=0.02)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)
    target = TriangleTarget(vertices, faces)
    collided, distance = target.query_batch_points(
        torch.tensor([[[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.08, 0.0, 0.0]]])
    )
    assert collided.tolist() == [[False, True, False]]
    assert distance[0, 0] > 0.025
    with pytest.raises(ValueError, match="watertight"):
        TriangleTarget(vertices, faces[:-1])


def test_pad_contact_is_allowed_but_non_pad_interference_is_rejected() -> None:
    geometry = hand()
    geometry.body[-1, 0] = torch.tensor([0.0, 0.0, 0.0])
    rejected, _ = HandCollisionChecker(geometry, Slab()).query(
        torch.eye(4), torch.eye(4)[None], torch.tensor([0.02])
    )
    assert rejected.tolist() == [True]


def test_open_collision_and_excessive_width_fail_closed() -> None:
    geometry = hand()
    geometry.body[0, 0] = 0.0
    checker = HandCollisionChecker(geometry, Slab())
    rejected, _ = checker.query(
        torch.eye(4), torch.eye(4).repeat(2, 1, 1), torch.tensor([0.02, 0.11])
    )
    assert rejected.tolist() == [True, True]


def test_intermediate_closing_interference_is_rejected() -> None:
    geometry = hand()
    pads = torch.stack((geometry.pads[0], geometry.pads.mean(0), geometry.pads[1]))
    body = torch.tensor([[[0.0, 0.0, -0.10]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, -0.10]]])
    sweep = HandGeometry(body, pads, torch.tensor([0.10, 0.05, 0.0]))
    rejected, _ = HandCollisionChecker(sweep, Slab()).query(
        torch.eye(4), torch.eye(4)[None], torch.tensor([0.02])
    )
    assert rejected.tolist() == [True]


def test_invalid_candidates_remain_row_local_failures() -> None:
    rejected, _ = HandCollisionChecker(hand(), Slab()).query(
        torch.eye(4),
        torch.eye(4).repeat(4, 1, 1),
        torch.tensor([float("nan"), -0.01, 0.11, 0.02]),
    )
    assert rejected.tolist() == [True, True, True, False]


def test_pad_side_contact_is_not_treated_as_intended_contact() -> None:
    class Obstacle:
        def query_batch_points(self, points, **kwargs):
            distance = (
                torch.linalg.vector_norm(
                    points - points.new_tensor([-0.02, -0.01, -0.02]), dim=-1
                )
                - 0.001
            )
            return distance <= 0, distance

    rejected, _ = HandCollisionChecker(hand(), Obstacle()).query(
        torch.eye(4), torch.eye(4)[None], torch.tensor([0.02])
    )
    assert rejected.tolist() == [True]


def test_contact_outside_pad_height_is_rejected() -> None:
    geometry = hand()
    geometry.pads[..., 2] += 0.10
    rejected, _ = HandCollisionChecker(geometry, Slab()).query(
        torch.eye(4), torch.eye(4)[None], torch.tensor([0.02])
    )
    assert rejected.tolist() == [True]


def test_single_arm_calls_forward_unchanged_and_dual_is_lazy() -> None:
    delegate = Mock(gripper_model=ParallelJawGripperModelCfg())
    dual = Mock()
    factory = Mock(return_value=dual)
    wrapper = CoordinatedGraspGenerator(delegate, factory)
    for name in (
        "get_valid_grasp_poses",
        "get_best_grasp_poses",
        "get_grasp_candidates",
    ):
        value = getattr(wrapper, name)(marker="unchanged")
        assert value is getattr(delegate, name).return_value
        getattr(delegate, name).assert_called_once_with(marker="unchanged")
    factory.assert_not_called()
    assert wrapper.gripper_model == delegate.gripper_model
    wrapper.get_dual_arm_valid_grasp_poses(marker="dual")
    factory.assert_called_once_with()
    dual.get_dual_arm_valid_grasp_poses.assert_called_once_with(marker="dual")


def test_rigid_transform_does_not_change_collision_decision() -> None:
    checker = HandCollisionChecker(hand(), Slab())
    transform = torch.eye(4)
    transform[:3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = torch.tensor([0.8, -0.4, 0.7])
    rejected, _ = checker.query(transform, transform[None], torch.tensor([0.02]))
    assert rejected.tolist() == [False]


def test_urdf_collision_geometry_uses_deployed_commands_and_tcp(tmp_path) -> None:
    path = tmp_path / "hand.urdf"
    path.write_text("""<robot name="hand">
      <link name="tool"><visual><geometry><box size="1 1 1"/></geometry></visual>
        <collision><origin xyz="0 0 -0.1"/><geometry><box size="0.01 0.01 0.01"/></geometry></collision></link>
      <link name="a_finger_pad"><collision><geometry><box size="0.01 0.02 0.04"/></geometry></collision></link>
      <link name="b_finger_pad"><collision><geometry><box size="0.01 0.02 0.04"/></geometry></collision></link>
      <joint name="hand_q" type="revolute"><parent link="tool"/><child link="a_finger_pad"/></joint>
      <joint name="other" type="fixed"><parent link="tool"/><child link="b_finger_pad"/></joint>
    </robot>""")
    seen = []
    tcp = torch.eye(4)
    tcp[2, 3] = 0.02

    def fk(*, qpos, link_names, qpos_joint_names):
        seen.append(qpos.clone())
        poses = torch.eye(4).repeat(len(qpos), len(link_names), 1, 1)
        poses[:, :, 2, 3] = 0.5
        for i, name in enumerate(link_names):
            if name.endswith("finger_pad"):
                poses[:, i, 0, 3] = (-1 if name.startswith("a") else 1) * (
                    0.06 - 0.05 * qpos[:, 1]
                )
        return poses

    robot = SimpleNamespace(
        cfg=SimpleNamespace(
            fpath=path,
            control_parts={"hand": ["hand_q"]},
            solver_cfg={"arm": SimpleNamespace(end_link_name="tool", tcp=tcp)},
        ),
        joint_names=["arm_q", "hand_q"],
        get_qpos=lambda: torch.tensor([[0.7, 0.4]]),
        compute_fk=fk,
    )
    geometry = build_hand_geometry(
        robot,
        motion_part="arm",
        hand_part="hand",
        commands={"open": [0.2], "grasp": [0.8]},
    )
    assert geometry.gaps[[0, -1]].tolist() == pytest.approx([0.09, 0.03])
    assert seen[0][[0, -1], 1].tolist() == pytest.approx([0.2, 0.8])
    assert torch.all(seen[0][:, 0] == 0.7)
    assert geometry.body[..., 2].max() == pytest.approx(-0.115)
    assert geometry.pads[..., 2].min() == pytest.approx(-0.04)
    assert geometry.pads.shape[2] > 8


def test_unsupported_filter_policy_is_not_silently_ignored() -> None:
    checker = HandCollisionChecker(hand(), Slab())
    with pytest.raises(ValueError, match="ground"):
        checker.query(
            torch.eye(4),
            torch.eye(4)[None],
            torch.tensor([0.02]),
            is_filter_ground_collision=True,
        )
    with pytest.raises(ValueError, match="threshold"):
        checker.query(
            torch.eye(4),
            torch.eye(4)[None],
            torch.tensor([0.02]),
            collision_threshold=-0.01,
        )


def test_e5_installation_preserves_single_arm_services_in_mixed_program(
    monkeypatch,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import (
        coordinated_grasp as module,
    )
    from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

    def resource(role):
        return SimpleNamespace(
            resource_id=role,
            endpoints=(
                SimpleNamespace(endpoint_id="motion", control_part=role + "_arm"),
                SimpleNamespace(
                    endpoint_id="grasp", control_part=role + "_eef", command_preset=role
                ),
            ),
        )

    commands = {
        "left": {"open": [0.0], "grasp": [0.7]},
        "right": {"open": [0.1], "grasp": [0.6]},
    }
    registration = SimpleNamespace(
        robot_profile_binding=SimpleNamespace(
            resources=[resource("left"), resource("right")],
            command_presets=[
                SimpleNamespace(preset_id=role, commands=value)
                for role, value in commands.items()
            ],
        )
    )
    model = ParallelJawGripperModelCfg(model_id="robotiq_arg2f_140", finger_length=0.13)
    delegates = {role + "_eef": AntipodalGraspPoseGenerator(model) for role in commands}
    singles = {name: Mock(return_value=object()) for name in delegates}
    for name, delegate in delegates.items():
        monkeypatch.setattr(delegate, "get_valid_grasp_poses", singles[name])
    built = []

    def build(robot, **kwargs):
        built.append(kwargs)
        return kwargs["hand_part"]

    def sampler(delegate, geometry, role):
        return SimpleNamespace(
            get_dual_arm_valid_grasp_poses=lambda **kwargs: [
                {"left": {"geometry": geometry}, "right": {"geometry": geometry}}
            ]
        )

    monkeypatch.setattr(module, "build_hand_geometry", build)
    monkeypatch.setattr(module, "_HandSampler", sampler)
    installed = module.install_coordinated_grasps(registration, object(), delegates)
    for name, wrapper in installed.items():
        assert wrapper.gripper_model.finger_length == 0.13
        assert (
            wrapper.get_valid_grasp_poses(marker="before_e5")
            is singles[name].return_value
        )
    assert not built
    dual = installed["left_eef"].get_dual_arm_valid_grasp_poses(
        mesh_vertices=torch.zeros(1, 3)
    )
    assert dual == [
        {"left": {"geometry": "left_eef"}, "right": {"geometry": "right_eef"}}
    ]
    assert [item["commands"] for item in built] == list(commands.values())
    for name, wrapper in installed.items():
        assert (
            wrapper.get_valid_grasp_poses(marker="after_e5")
            is singles[name].return_value
        )
        assert delegates[name]._backends == {}
    installed["right_eef"].get_dual_arm_valid_grasp_poses(
        mesh_vertices=torch.zeros(1, 3)
    )
    assert len(built) == 2


def test_non_robotiq_e5_keeps_existing_generators() -> None:
    from embodichain.gen_sim.task_engine._task_program.coordinated_grasp import (
        install_coordinated_grasps,
    )
    from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

    resources = [
        SimpleNamespace(
            resource_id=role,
            endpoints=[
                SimpleNamespace(endpoint_id="motion", control_part=role + "_arm"),
                SimpleNamespace(endpoint_id="grasp", control_part=role + "_eef"),
            ],
        )
        for role in ("left", "right")
    ]
    registration = SimpleNamespace(
        robot_profile_binding=SimpleNamespace(resources=resources, command_presets=[])
    )
    generators = {
        role + "_eef": AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
        for role in ("left", "right")
    }
    result = install_coordinated_grasps(registration, object(), generators)
    assert all(result[name] is generator for name, generator in generators.items())


@pytest.mark.parametrize("accepted", [False, True])
def test_proposals_cannot_escape_without_role_specific_geometry_validation(
    monkeypatch, accepted
) -> None:
    from embodichain.gen_sim.task_engine._task_program.coordinated_grasp import (
        _HandSampler,
    )
    from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

    delegate = AntipodalGraspPoseGenerator(ParallelJawGripperModelCfg())
    sampler = _HandSampler(delegate, hand(), "right")
    poses, widths, costs = (
        torch.eye(4).repeat(2, 1, 1),
        torch.tensor([0.02, 0.02]),
        torch.tensor([1.0, 2.0]),
    )
    proposal = {
        "is_success": True,
        "grasp_poses": poses,
        "open_lengths": widths,
        "total_cost": costs,
    }
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_dual_arm_valid_grasp_poses",
        lambda *args, **kwargs: [{"left": proposal, "right": proposal}],
    )
    vertices, faces = torch.zeros(3, 3), torch.tensor([[0, 1, 2]])
    checker = Mock()
    checker.fit_contacts.side_effect = lambda obj, p, w: (p, w)
    checker.query.side_effect = lambda obj, p, w, **kwargs: (
        torch.full((len(p),), not accepted),
        torch.zeros(len(p)),
    )
    sampler._validators[sampler._geometry_key(vertices, faces)] = checker
    result = sampler.get_dual_arm_valid_grasp_poses(
        mesh_vertices=vertices, mesh_triangles=faces, obj_poses=torch.eye(4)[None]
    )
    assert result[0]["right"]["is_success"] is accepted
    assert result[0]["left"]["is_success"] is False
    assert checker.query.called
    assert torch.equal(poses, torch.eye(4).repeat(2, 1, 1))
    with pytest.raises(RuntimeError, match="single-arm"):
        sampler.get_valid_grasp_poses()
