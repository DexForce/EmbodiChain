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

from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import scene_collision as gate


def _pose(x=0.0, y=0.0):
    pose = torch.eye(4)[None]
    pose[:, 0, 3], pose[:, 1, 3] = x, y
    return pose


def _box(owner, entity=None, name=None, extents=(0.1, 0.1, 0.1)):
    import trimesh

    return gate._component(
        owner, entity or owner, name or owner, trimesh.creation.box(extents=extents)
    )


@pytest.fixture
def fixture(tmp_path):
    pytest.importorskip("fcl")
    pytest.importorskip("yourdfpy")
    collision = '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
    urdf = tmp_path / "snapshot.urdf"
    urdf.write_text(
        '<robot name="snapshot"><link name="world"/><link name="arm_root"/>'
        '<joint name="mount" type="fixed"><parent link="world"/><child link="arm_root"/></joint>'
        '<link name="tool"/>'
        '<joint name="slide" type="prismatic"><parent link="arm_root"/><child link="tool"/>'
        '<axis xyz="1 0 0"/><limit lower="0" upper="4" effort="1" velocity="1"/></joint>'
        f'<link name="left_finger_pad">{collision}</link>'
        '<joint name="left_pad" type="fixed"><parent link="tool"/><child link="left_finger_pad"/>'
        '<origin xyz="0 0.2 0"/></joint>'
        f'<link name="right_finger_pad">{collision}</link>'
        '<joint name="right_pad" type="fixed"><parent link="tool"/><child link="right_finger_pad"/>'
        '<origin xyz="0 -0.2 0"/></joint>'
        f'<link name="other_finger_pad">{collision}</link>'
        '<joint name="other" type="fixed"><parent link="world"/><child link="other_finger_pad"/>'
        '<origin xyz="3 0.2 0"/></joint></robot>'
    )

    def fk(qpos):
        result = torch.eye(4)[None].repeat(qpos.shape[0], 1, 1)
        result[:, 0, 3] = qpos[:, 0]
        return result

    def forbidden(*_args, **_kwargs):
        pytest.fail("Collision qualification read native state instead of context.")

    solver = NS(
        root_link_name="arm_root",
        end_link_name="tool",
        get_fk=fk,
        get_tcp=lambda: torch.eye(4),
    )
    robot = NS(
        cfg=NS(fpath=str(urdf)),
        joint_names=("slide",),
        get_solver=lambda _part: solver,
        get_joint_ids=lambda _part: [0],
        get_qpos=forbidden,
        get_link_pose=forbidden,
        get_local_pose=forbidden,
        compute_batch_fk=forbidden,
    )
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=NS(qpos=torch.zeros(1, 1), root_pose=_pose(), timestamp=0.0),
        scene=NS(
            entities={"obstacle": NS(pose=_pose(5), confidence=1.0)}, timestamp=0.0
        ),
        task=NS(held_objects={}),
    )
    meshes = gate.SceneCollisionMeshes((_box("obstacle"),))
    return NS(robot=robot, context=context, meshes=meshes, solver=solver)


def test_every_emitted_sample_not_only_endpoint_is_screened(fixture):
    fixture.context.scene.entities["obstacle"].pose = _pose(0.5, 0.2)
    positions = torch.tensor([[[0.0], [0.5], [1.0]]])
    report = gate.inspect_snapshot_motion(
        fixture.robot, fixture.context, positions, "arbitrary_part", fixture.meshes
    )
    assert not report["valid"] and report["frame"] == 1
    assert report["kind"] == "scene"


def test_context_pose_and_no_native_read_are_authoritative(fixture):
    positions = torch.tensor([[[0.0], [0.5], [1.0]]])
    original = positions.clone()
    report = gate.inspect_snapshot_motion(
        fixture.robot, fixture.context, positions, "arbitrary_part", fixture.meshes
    )
    assert report["valid"]
    torch.testing.assert_close(positions, original)


def test_passive_robot_geometry_is_not_exempt_from_scene_collision(fixture):
    fixture.context.scene.entities["obstacle"].pose = _pose(3, 0.2)
    report = gate.inspect_snapshot_motion(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        fixture.meshes,
    )
    assert not report["valid"]
    assert report["pairs"][0][0] == "other_finger_pad"


def test_articulated_walls_do_not_turn_open_cavity_into_whole_container_box(fixture):
    fixture.context.scene.entities = {
        "wall_left": NS(pose=_pose(0, 1)),
        "wall_right": NS(pose=_pose(0, -1)),
    }
    meshes = gate.SceneCollisionMeshes(
        (_box("cabinet", "wall_left", "left"), _box("cabinet", "wall_right", "right"))
    )
    assert gate.inspect_snapshot_motion(
        fixture.robot, fixture.context, torch.zeros(1, 1, 1), "arbitrary_part", meshes
    )["valid"]


@pytest.mark.parametrize("missing", ["root", "entity", "mesh", "fk", "timestamp"])
def test_missing_snapshot_geometry_pose_or_fk_fails_closed(fixture, missing):
    if missing == "root":
        fixture.context.robot.root_pose = None
    elif missing == "entity":
        fixture.context.scene.entities = {}
    elif missing == "mesh":
        fixture.meshes = gate.SceneCollisionMeshes(unavailable="missing_physical_mesh")
    elif missing == "fk":
        fixture.solver.get_fk = lambda _q: _pose(0.01)
    else:
        fixture.context.scene.timestamp = 1.0
    report = gate.inspect_snapshot_motion(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        fixture.meshes,
    )
    assert not report["valid"] and "unavailable" in report


def test_payload_hits_other_arm_without_whole_robot_whitelist(fixture):
    meshes = gate.SceneCollisionMeshes((*fixture.meshes.components, _box("payload")))
    report = gate.inspect_payload_robot(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        meshes,
        "payload",
        _pose(3, 0.2)[:, None],
    )
    assert not report["valid"] and report["kind"] == "payload_robot"
    assert report["pairs"][0][0] == "other_finger_pad"


def test_only_owned_two_pad_contact_is_allowed_in_explicit_phase_frames(fixture):
    meshes = gate.SceneCollisionMeshes(
        (*fixture.meshes.components, _box("payload", extents=(0.1, 0.5, 0.1)))
    )
    contacts = (
        gate.PadContact(
            "payload", ("left_finger_pad", "right_finger_pad"), "hold", 0, 1
        ),
    )
    report = gate.inspect_payload_robot(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 2, 1),
        "arbitrary_part",
        meshes,
        "payload",
        _pose()[:, None].repeat(1, 2, 1, 1),
        pad_contacts=contacts,
        phase_names=("hold", "hold"),
    )
    assert not report["valid"] and report["frame"] == 1
    report = gate.inspect_payload_robot(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        meshes,
        "payload",
        _pose()[:, None],
        pad_contacts=contacts,
        phase_names=("hold",),
    )
    assert report["valid"]


def test_declared_contact_cannot_override_actual_command_phase(fixture):
    meshes = gate.SceneCollisionMeshes(
        (*fixture.meshes.components, _box("payload", extents=(0.1, 0.5, 0.1)))
    )
    contacts = (
        gate.PadContact(
            "payload", ("left_finger_pad", "right_finger_pad"), "hold", 0, 1
        ),
    )
    report = gate.inspect_payload_robot(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        meshes,
        "payload",
        _pose()[:, None],
        pad_contacts=contacts,
        phase_names=("approach",),
    )
    assert not report["valid"] and report["kind"] == "payload_robot"


def test_non_pad_on_operated_arm_is_not_exempt_from_payload_collision(fixture):
    from xml.etree import ElementTree

    path = fixture.robot.cfg.fpath
    tree = ElementTree.parse(path)
    tool = next(
        link for link in tree.getroot().findall("link") if link.attrib["name"] == "tool"
    )
    tool.append(
        ElementTree.fromstring(
            '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
        )
    )
    tree.write(path)
    meshes = gate.SceneCollisionMeshes(
        (*fixture.meshes.components, _box("payload", extents=(0.1, 0.5, 0.1)))
    )
    contacts = (
        gate.PadContact(
            "payload", ("left_finger_pad", "right_finger_pad"), "hold", 0, 1
        ),
    )
    report = gate.inspect_payload_robot(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        meshes,
        "payload",
        _pose()[:, None],
        pad_contacts=contacts,
        phase_names=("hold",),
    )
    assert not report["valid"] and any(pair[0] == "tool" for pair in report["pairs"])


def test_carried_object_exclusion_requires_exclusive_context_state(fixture):
    report = gate.inspect_snapshot_motion(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        fixture.meshes,
        carried_object_id="obstacle",
    )
    assert not report["valid"] and "exclusive_carried_state" in report["unavailable"]


def test_articulation_contact_cannot_exempt_entire_cabinet_owner(fixture):
    fixture.context.scene.entities = {"wall": NS(pose=_pose(0, 0.2))}
    meshes = gate.SceneCollisionMeshes((_box("cabinet", "wall", "wall_mesh"),))
    contacts = (
        gate.PadContact(
            "cabinet", ("left_finger_pad", "right_finger_pad"), "hold", 0, 1
        ),
    )
    report = gate.inspect_snapshot_motion(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        meshes,
        pad_contacts=contacts,
        phase_names=("hold",),
    )
    assert not report["valid"] and "pad_contact_scope" in report["unavailable"]


@pytest.mark.parametrize(
    "pads,phase",
    [
        (("left_finger_pad", "other_finger_pad"), "hold"),
        (("tool", "right_finger_pad"), "hold"),
        (("left_finger_pad", "right_finger_pad"), "approach"),
    ],
)
def test_unknown_contact_or_other_hand_exemption_is_unavailable(fixture, pads, phase):
    contacts = (gate.PadContact("obstacle", pads, phase, 0, 1),)
    report = gate.inspect_snapshot_motion(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        fixture.meshes,
        pad_contacts=contacts,
    )
    assert not report["valid"] and "unavailable" in report


def test_capture_uses_physical_local_shape_not_visual_vertices_or_live_pose():
    pytest.importorskip("trimesh")
    shape = NS(
        vertices=None,
        triangles=None,
        half_extents=torch.tensor([0.05, 0.1, 0.15]),
        local_pose=_pose(0.4)[0],
    )
    native = NS(
        get_collision_shapes=lambda: [shape],
        get_vertices=lambda **_kw: pytest.fail(
            "Visual mesh is not collision geometry."
        ),
        get_local_pose=lambda **_kw: pytest.fail(
            "Geometry capture cannot freeze live poses."
        ),
    )
    bindings = NS(
        rigid_objects=(
            NS(entity_id="body", simulation_uid="native_body", collision_role="NONE"),
        ),
        articulations=(),
        links=(),
        rigidized_articulations=(),
    )
    meshes = gate.capture_collision_meshes(
        NS(get_rigid_object=lambda _uid: native), bindings
    )
    assert meshes.unavailable is None and len(meshes.components) == 1
    assert float(meshes.components[0].vertices[:, 0].min()) == pytest.approx(0.35)
    assert float(meshes.components[0].vertices[:, 0].max()) == pytest.approx(0.45)


def test_capture_unsupported_shape_never_publishes_partial_inventory():
    native = NS(
        get_collision_shapes=lambda: [
            NS(vertices=None, triangles=None, half_extents=None)
        ]
    )
    bindings = NS(
        rigid_objects=(NS(entity_id="body", simulation_uid="native_body"),),
        articulations=(),
        links=(),
        rigidized_articulations=(),
    )
    result = gate.capture_collision_meshes(
        NS(get_rigid_object=lambda _uid: native), bindings
    )
    assert result.unavailable and not result.components


@pytest.mark.parametrize(
    "unmapped,primitive", [(False, False), (True, False), (False, True)]
)
def test_articulation_capture_requires_complete_owned_enabled_collision_inventory(
    tmp_path, unmapped, primitive
):
    pytest.importorskip("pxr")
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    owner = UsdGeom.Xform.Define(stage, "/root")
    UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    mesh = UsdGeom.Mesh.Define(stage, "/root/mesh")
    mesh.CreatePointsAttr().Set(box.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(box.faces))
    mesh.CreateFaceVertexIndicesAttr().Set(box.faces.reshape(-1).tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    if primitive:
        cube = UsdGeom.Cube.Define(stage, "/root/unrepresented_cube")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(
            True
        )
    stage.GetRootLayer().Save()
    art = NS(cfg=NS(fpath=str(source), body_scale=[1, 1, 1]))
    bindings = NS(
        rigid_objects=(),
        articulations=(NS(entity_id="assembly", simulation_uid="native"),),
        links=(
            ()
            if unmapped
            else (
                NS(
                    articulation_id="assembly",
                    native_link_name="root",
                    entity_id="root_link",
                ),
            )
        ),
        rigidized_articulations=(),
    )
    result = gate.capture_collision_meshes(
        NS(get_articulation=lambda _uid: art), bindings
    )
    assert bool(result.unavailable) is (unmapped or primitive)
    if not unmapped and not primitive:
        assert result.components[0].pose_entity_id == "root_link"


def test_snapshot_fk_and_pad_helpers_use_context_root_and_exact_source_owners(fixture):
    fixture.context.robot.root_pose = _pose(2)
    positions = torch.tensor([[[0.0], [0.5]]])
    poses = gate.snapshot_eef_poses(
        fixture.robot, fixture.context, positions, "arbitrary_part"
    )
    torch.testing.assert_close(poses[0, :, 0, 3], torch.tensor([2.0, 2.5]))
    assert gate.snapshot_pad_links(
        fixture.robot, fixture.context, "arbitrary_part"
    ) == ("left_finger_pad", "right_finger_pad")


def test_snapshot_pad_helper_requires_two_colliding_owned_pads(fixture):
    from xml.etree import ElementTree

    tree = ElementTree.parse(fixture.robot.cfg.fpath)
    link = next(
        link
        for link in tree.getroot().findall("link")
        if link.attrib["name"] == "right_finger_pad"
    )
    link.remove(link.find("collision"))
    tree.write(fixture.robot.cfg.fpath)
    with pytest.raises(ValueError, match="two_owned_pads"):
        gate.snapshot_pad_links(fixture.robot, fixture.context, "arbitrary_part")


def test_robot_scene_screen_keeps_released_object_at_its_predicted_pose(fixture):
    positions = torch.tensor([[[0.0], [0.5], [1.0]]])
    poses = _pose(5)[:, None].repeat(1, 3, 1, 1)
    poses[:, 2] = _pose(1, 0.2)
    original = poses.clone()
    result = gate.inspect_snapshot_motion(
        fixture.robot,
        fixture.context,
        positions,
        "arbitrary_part",
        fixture.meshes,
        object_pose_samples={"obstacle": poses},
    )
    assert not result["valid"] and result["frame"] == 2 and result["kind"] == "scene"
    torch.testing.assert_close(poses, original)


@pytest.mark.parametrize("owner", ["unbound", "articulation"])
def test_robot_scene_prediction_requires_a_bound_rigid_owner(fixture, owner):
    meshes = fixture.meshes
    if owner == "articulation":
        meshes = gate.SceneCollisionMeshes((_box(owner, "link"),))
    result = gate.inspect_snapshot_motion(
        fixture.robot,
        fixture.context,
        torch.zeros(1, 1, 1),
        "arbitrary_part",
        meshes,
        object_pose_samples={owner: _pose()[:, None]},
    )
    assert not result["valid"] and "rigid_pose_schedule" in result["unavailable"]


def test_payload_scene_contact_permission_is_exact_component_and_frame(fixture):
    fixture.context.scene.entities["obstacle"].pose = _pose(1)
    meshes = gate.SceneCollisionMeshes(
        (_box("payload"), _box("obstacle", name="floor"))
    )
    poses = _pose(1)[:, None].repeat(1, 2, 1, 1)
    result = gate.inspect_payload_scene(
        fixture.context,
        meshes,
        "payload",
        poses,
        allowed_contacts={"floor": torch.tensor([[True, False]])},
    )
    assert (
        not result["valid"]
        and result["frame"] == 1
        and result["kind"] == "payload_scene"
    )
    assert result["pairs"] == [["payload", "obstacle", "floor"]]
    assert gate.inspect_payload_scene(
        fixture.context,
        meshes,
        "payload",
        poses,
        allowed_contacts={"floor": torch.ones(1, 2, dtype=torch.bool)},
    )["valid"]


def test_floor_allowance_cannot_skip_another_component_of_same_owner(fixture):
    fixture.context.scene.entities["obstacle"].pose = _pose(1)
    meshes = gate.SceneCollisionMeshes(
        (_box("payload"), _box("obstacle", name="floor"), _box("obstacle", name="wall"))
    )
    result = gate.inspect_payload_scene(
        fixture.context,
        meshes,
        "payload",
        _pose(1)[:, None],
        allowed_contacts={"floor": torch.ones(1, 1, dtype=torch.bool)},
    )
    assert not result["valid"] and result["pairs"] == [["payload", "obstacle", "wall"]]


@pytest.mark.parametrize(
    "name,mask",
    [
        ("obstacle", torch.ones(1, 1, dtype=torch.bool)),
        ("floor", torch.ones(1, 1)),
        ("floor", torch.ones(1, dtype=torch.bool)),
    ],
)
def test_payload_scene_unknown_owner_or_malformed_allowance_is_unavailable(
    fixture, name, mask
):
    meshes = gate.SceneCollisionMeshes(
        (_box("payload"), _box("obstacle", name="floor"))
    )
    result = gate.inspect_payload_scene(
        fixture.context,
        meshes,
        "payload",
        _pose()[:, None],
        allowed_contacts={name: mask},
    )
    assert not result["valid"] and "component_contact_scope" in result["unavailable"]


def test_payload_scene_missing_snapshot_pose_is_unavailable(fixture):
    meshes = gate.SceneCollisionMeshes((_box("payload"), _box("absent")))
    result = gate.inspect_payload_scene(
        fixture.context, meshes, "payload", _pose()[:, None]
    )
    assert not result["valid"] and "pose_unavailable" in result["unavailable"]
