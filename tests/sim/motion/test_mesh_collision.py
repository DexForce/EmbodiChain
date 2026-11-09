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

"""Native discrete mesh queries over an independently specified box scene."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

_requires_newton = pytest.mark.skipif(
    importlib.util.find_spec("newton") is None,
    reason="native collision queries require the supported Newton collision runtime",
)

from embodichain.lab.sim.motion.mesh_collision import (
    RobotMeshCollisionSnapshot,
    SimulationFreeMotionQuery,
)
from ._mesh_collision_fixture import mesh_query_fixture


@_requires_newton
@pytest.mark.parametrize("shape_kind", ["box", "convex"])
def test_cooked_complete_volume_is_not_downgraded_to_triangle_surface(shape_kind):
    from embodichain.lab.sim.objects.rigid_object import RigidBodyShape
    from embodichain.lab.sim.scene.feasibility.collision import (
        CollisionBody,
        NewtonCollisionChecker,
    )

    robot, simulation = mesh_query_fixture()
    if shape_kind == "convex":
        import trimesh

        shape = robot.get_link_collision_shapes("tool")[0]
        mesh = trimesh.creation.box((0.1, 0.1, 0.1))
        shape.shape_type = RigidBodyShape.CONVEX
        shape.vertices = torch.from_numpy(mesh.vertices.copy())
        shape.triangles = torch.from_numpy(mesh.faces.copy())
        shape.half_extents = None
    obstacle = simulation.get_rigid_object("obstacle")
    shape = obstacle.get_collision_shapes()[0]
    shape.half_extents[:] = 0.1875
    pose = torch.eye(4)[None]
    pose[:, :3, 3] = torch.tensor([0.55, 0.0, 0.3])
    obstacle.get_local_pose = lambda **kw: pose.clone()
    # Independent native primitive query establishes the contained-volume fact.
    native = NewtonCollisionChecker(
        [
            CollisionBody(
                "hand", "hand", "box", dimensions=(0.1, 0.1, 0.1), robot=True
            ),
            CollisionBody("world", "world", "box", dimensions=(0.375, 0.375, 0.375)),
        ],
        clearance_m=0.0,
        device="cpu",
        use_cuda_graph=False,
    )
    try:
        assert native.contact_pairs({"hand": pose[0].numpy(), "world": pose[0].numpy()})
    finally:
        native.close()
    result = SimulationFreeMotionQuery(robot, simulation).inspect(
        robot.get_qpos(), control_part="arm"
    )
    assert not result.valid and result.kind == "scene" and result.frame == 0
    assert result.geometry_source == "native_cooked_robot_and_typed_scene_shapes"
    assert result.metadata()["geometry_source"] == result.geometry_source


@pytest.fixture(autouse=True)
def forbid_fcl_queries(monkeypatch):
    import trimesh

    def forbidden(*args, **kwargs):
        pytest.fail("Motion collision queries must use the native pipeline")

    monkeypatch.setattr(trimesh.collision, "CollisionManager", forbidden)


def test_missing_newton_is_reported_before_any_robot_or_scene_capture(monkeypatch):
    from embodichain.lab.sim.scene import _collision_backends

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(_collision_backends, "import_module", missing)
    robot, simulation = mesh_query_fixture()
    with pytest.raises(ImportError, match="Newton planning queries require"):
        SimulationFreeMotionQuery(robot, simulation)
    with pytest.raises(ImportError, match="Newton planning queries require"):
        RobotMeshCollisionSnapshot(robot, control_part="arm")
    robot.get_qpos.assert_not_called()
    robot.get_local_pose.assert_not_called()


@pytest.mark.parametrize(
    ("stop", "kind", "pairs"),
    [
        ((0.125, 0.0), None, ()),
        (
            (0.25, 0.0),
            "scene",
            tuple(
                (name, "obstacle:0")
                for name in (
                    "pad_a/0:geometry_4",
                    "pad_b/0:geometry_5",
                    "tool/0:geometry_3",
                )
            ),
        ),
        (
            (0.6875, 0.0),
            "other_arm",
            tuple(
                (name, "passive/0:geometry_6")
                for name in (
                    "pad_a/0:geometry_4",
                    "pad_b/0:geometry_5",
                    "tool/0:geometry_3",
                )
            ),
        ),
        (
            (0.375, -0.5),
            "self",
            tuple(
                tuple(sorted((name, "shoulder/0:geometry_1")))
                for name in (
                    "pad_a/0:geometry_4",
                    "pad_b/0:geometry_5",
                    "tool/0:geometry_3",
                )
            ),
        ),
    ],
    ids=["clear", "scene", "other_arm", "self"],
)
@_requires_newton
def test_native_query_preserves_pair_scope_order_and_mount_exemptions(
    stop, kind, pairs
):
    # In the authored axis-aligned fixture, tool/pad centres are
    # .55/.59 + slide + fold, passive x=1.2 and obstacle x=.85.
    # These poses are away from touching boundaries and overlap precisely
    # the three listed boxes. The shoulder mounting contact at q=0 is exempt.
    robot, simulation = mesh_query_fixture()
    observed = robot.get_qpos().clone()
    query = SimulationFreeMotionQuery(robot, simulation)
    positions = torch.zeros(2, robot.dof)
    positions[1, :2] = torch.tensor(stop)
    report = query.inspect(positions, control_part="arm")
    assert report.backend == "newton.CollisionPipeline"
    assert (
        report.metadata()["geometry_source"]
        == "native_cooked_robot_and_typed_scene_shapes"
    )
    assert report.valid is (kind is None)
    assert report.kind == kind and report.pairs == pairs
    assert report.frame == (None if kind is None else 1)
    assert report.frames == (2 if kind is None else None)
    torch.testing.assert_close(robot.get_qpos(), observed)


@_requires_newton
def test_free_motion_query_reads_fresh_scene_and_rejects_bad_fk():
    robot, simulation = mesh_query_fixture()
    query = SimulationFreeMotionQuery(robot, simulation)
    positions = torch.tensor([[0.0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0]])
    assert query.inspect(positions, control_part="arm").valid
    obstacle = simulation.get_rigid_object("obstacle")
    pose = obstacle.get_local_pose(to_matrix=True)
    pose[:, 0, 3] = 0.72
    obstacle.get_local_pose = lambda **kw: pose.clone()
    rejected = query.inspect(positions, control_part="arm")
    assert not rejected.valid and rejected.kind == "scene" and rejected.frame == 1
    robot.get_link_pose.return_value = robot.get_link_pose.return_value.clone()
    robot.get_link_pose.return_value[:, 2, 3] += 0.01
    with pytest.raises(ValueError, match="FK does not match"):
        query.inspect(positions, control_part="arm")


@_requires_newton
def test_native_query_unavailable_is_not_clear_and_releases_its_owned_model(
    monkeypatch,
):
    from embodichain.lab.sim.scene.feasibility import collision as owner

    created = []

    class Unavailable:
        def __init__(self, bodies, **options):
            assert "allow_triangle_meshes" not in options
            assert all(body.kind == "box" for body in bodies)
            self.closed = False
            created.append(self)

        def contact_pairs(self, poses, **kwargs):
            raise owner.UnsupportedFeasibilityError("Controlled native uncertainty")

        def close(self):
            self.closed = True

    monkeypatch.setattr(owner, "NewtonCollisionChecker", Unavailable)
    robot, simulation = mesh_query_fixture()
    with pytest.raises(owner.UnsupportedFeasibilityError, match="native uncertainty"):
        SimulationFreeMotionQuery(robot, simulation).inspect(
            robot.get_qpos(), control_part="arm"
        )
    assert len(created) == 1 and created[0].closed


@_requires_newton
def test_cooked_shape_local_pose_and_last_component_are_applied_once():
    from embodichain.lab.sim.objects.rigid_object import (
        CollisionShapeDesc,
        RigidBodyShape,
    )

    robot, simulation = mesh_query_fixture()
    local = torch.eye(4)
    local[0, 3] = 0.25
    # This second native component has no authored URDF collision counterpart.
    robot.get_link_collision_shapes("tool").append(
        CollisionShapeDesc(
            "offset_part",
            RigidBodyShape.BOX,
            local,
            half_extents=torch.full((3,), 0.02),
        )
    )
    # Public descriptors already contain scale; applying cfg scale again would
    # incorrectly collide other original components before this offset part.
    robot.cfg.body_scale = [8.0, 4.0, 2.0]
    robot.get_link_physical_attr.return_value = None
    robot.get_link_physical_attr.side_effect = lambda **kw: [
        SimpleNamespace(enable_collision=True, rest_offset=0.1, contact_offset=0.2)
    ]
    obstacle = simulation.get_rigid_object("obstacle")
    pose = obstacle.get_local_pose(to_matrix=True)
    pose[:, 0, 3] = 0.8  # tool frame .55 + native local offset .25
    obstacle.get_local_pose = lambda **kw: pose.clone()
    report = SimulationFreeMotionQuery(robot, simulation).inspect(
        robot.get_qpos(), control_part="arm"
    )
    assert (report.kind, report.frame, report.pairs) == (
        "scene",
        0,
        (("tool/1:offset_part", "obstacle:0"),),
    )


@_requires_newton
@pytest.mark.parametrize("owner", ["pad_a", "passive"])
def test_each_native_collision_owner_requires_observed_fk_calibration(owner):
    robot, _ = mesh_query_fixture()
    original = robot.get_link_pose.side_effect

    def shifted(name, **kwargs):
        result = original(name, **kwargs)
        if name == owner:
            result[:, 2, 3] += 0.01
        return result

    robot.get_link_pose.side_effect = shifted
    with pytest.raises(ValueError, match="observed native link frame"):
        RobotMeshCollisionSnapshot(robot, control_part="arm")


@_requires_newton
def test_native_query_preserves_measured_mimic_state_and_applies_world_root_once(
    tmp_path,
):
    import xml.etree.ElementTree as ET
    import yourdfpy

    robot, _ = mesh_query_fixture()
    document = ET.parse(robot.cfg.fpath)
    slave = next(
        j for j in document.getroot().findall("joint") if j.attrib["name"] == "jaw_b"
    )
    ET.SubElement(slave, "mimic", joint="jaw_a", multiplier="1", offset="0")
    path = tmp_path / "measured_mimic.urdf"
    document.write(path)
    robot.cfg.fpath = str(path)
    # This is observed state, not a new desired mimic law. The native slave may
    # deviate from its leader; replacing it with leader qpos loses the real pose.
    qpos = robot.get_qpos().clone()
    qpos[0, 2:4] = torch.tensor([0.03, 0.031213412])
    robot.get_qpos.return_value = qpos
    root = torch.eye(4)[None]
    root[0, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    root[0, :3, 3] = torch.tensor([0.5, -0.25, 0.125])
    robot.get_local_pose.return_value = root
    local_tool = torch.eye(4)[None]
    local_tool[0, :3, 3] = torch.tensor([0.55, 0.0, 0.3])
    robot.get_link_pose.return_value = root @ local_tool
    projected = yourdfpy.URDF.load(path, load_meshes=False, load_collision_meshes=False)
    projected.update_cfg(dict(zip(robot.joint_names, qpos[0].tolist())))
    observed = robot.get_link_pose("pad_b", to_matrix=True)[0]
    assert not torch.allclose(
        torch.as_tensor(root[0].numpy() @ projected.get_transform("pad_b")),
        observed.double(),
        atol=2e-4,
        rtol=0,
    )
    before = qpos.clone()
    with RobotMeshCollisionSnapshot(robot, control_part="arm") as snapshot:
        report = snapshot.inspect(qpos)
    assert report.valid and report.frames == 1
    torch.testing.assert_close(robot.get_qpos(), before, rtol=0, atol=0)
    calls = robot.compute_fk.call_args_list
    assert len(calls) == 2  # one observed calibration, one complete sampled batch
    assert all(
        call.kwargs["name"] is None
        and call.kwargs["qpos_joint_names"] == tuple(robot.joint_names)
        for call in calls
    )


@_requires_newton
@pytest.mark.parametrize("damage", ["shape", "nonfinite"])
def test_native_query_rejects_invalid_public_full_fk_output(damage):
    robot, _ = mesh_query_fixture()
    original = robot.compute_fk.side_effect

    def damaged(**kwargs):
        value = original(**kwargs)
        if damage == "shape":
            return value[:, 0]
        value[0, 0, 0, 0] = float("nan")
        return value

    robot.compute_fk.side_effect = damaged
    with pytest.raises(ValueError, match="Public full-joint FK requires finite"):
        RobotMeshCollisionSnapshot(robot, control_part="arm")


@_requires_newton
@pytest.mark.parametrize(
    "missing", ["disabled", "empty", "shape_type", "physical_port"]
)
def test_unknown_or_disabled_native_shapes_do_not_certify_clearance(
    monkeypatch, missing
):
    from embodichain.lab.sim.objects.rigid_object import RigidBodyShape
    from embodichain.lab.sim.scene.feasibility import collision as owner

    robot, _ = mesh_query_fixture()
    if missing == "disabled":
        robot.get_link_physical_attr.side_effect = lambda **kw: [
            SimpleNamespace(enable_collision=False, rest_offset=0.0, contact_offset=0.0)
        ]
    elif missing == "empty":
        robot.get_link_collision_shapes("tool").clear()
    elif missing == "shape_type":
        robot.get_link_collision_shapes("tool")[0].shape_type = RigidBodyShape.SDF
    else:
        robot.get_link_physical_attr.side_effect = RuntimeError("unsupported backend")

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete native geometry must reject before building a query")

    monkeypatch.setattr(owner, "NewtonCollisionChecker", forbidden)
    with pytest.raises(owner.UnsupportedFeasibilityError):
        RobotMeshCollisionSnapshot(robot, control_part="arm")


@_requires_newton
@pytest.mark.parametrize("surface_owner", ["robot", "scene"])
def test_actual_concave_mesh_keeps_triangles_and_unavailable_clearance(
    monkeypatch, surface_owner
):
    import numpy as np
    import trimesh
    from embodichain.lab.sim.objects.rigid_object import RigidBodyShape
    from embodichain.lab.sim.scene.feasibility import collision as owner

    robot, simulation = mesh_query_fixture()
    components = []
    for y in (-0.3, 0.3):
        part = trimesh.creation.box((0.1, 0.1, 0.1))
        part.apply_translation((0.0, y, 0.0))
        components.append(part)
    mesh = trimesh.util.concatenate(components)
    assert not mesh.is_convex
    obstacle = simulation.get_rigid_object("obstacle")
    shape = (
        robot.get_link_collision_shapes("tool")[0]
        if surface_owner == "robot"
        else obstacle.get_collision_shapes()[0]
    )
    shape.shape_type = RigidBodyShape.MESH
    shape.half_extents = None
    shape.vertices = torch.from_numpy(mesh.vertices.copy())
    shape.triangles = torch.from_numpy(mesh.faces.copy())
    pose = obstacle.get_local_pose(to_matrix=True)
    pose[:, 0, 3] = 0.55  # Empty gap is around the tool, within the mesh AABB.
    obstacle.get_local_pose = lambda **kw: pose.clone()
    captured = []
    original = owner.NewtonCollisionChecker

    def capture(bodies, **kwargs):
        captured.extend(body for body in bodies if body.kind == "mesh")
        return original(bodies, **kwargs)

    monkeypatch.setattr(owner, "NewtonCollisionChecker", capture)
    with pytest.raises(
        owner.UnsupportedFeasibilityError, match="Triangle meshes require"
    ):
        SimulationFreeMotionQuery(robot, simulation).inspect(
            robot.get_qpos(), control_part="arm"
        )
    assert len(captured) == 1 and captured[0].kind == "mesh"
    np.testing.assert_array_equal(captured[0].vertices, mesh.vertices)
    np.testing.assert_array_equal(captured[0].triangles, mesh.faces)


@_requires_newton
def test_owned_robot_query_scope_reuses_only_current_robot_and_restores_nested_scope(
    monkeypatch,
):
    from embodichain.lab.sim.motion import mesh_collision as owner

    robot, _ = mesh_query_fixture()
    positions = robot.get_qpos().clone()
    created = []
    real = owner.RobotMeshCollisionSnapshot

    def capture(*args, **kwargs):
        result = real(*args, **kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(owner, "RobotMeshCollisionSnapshot", capture)
    owner.inspect_robot_collision(robot, positions, control_part="arm")
    owner.inspect_robot_collision(robot, positions, control_part="arm")
    assert len(created) == 2
    for query in created:
        with pytest.raises(RuntimeError, match="closed"):
            query.inspect(positions)
    with owner.robot_collision_query_scope():
        first = owner.inspect_robot_collision(robot, positions, control_part="arm")
        assert first.geometry_source == "native_cooked_robot_shapes"
        second = owner.inspect_robot_collision(robot, positions, control_part="arm")
        assert first == second and len(created) == 3
        outer = created[-1]
        with pytest.raises(KeyboardInterrupt):
            with owner.robot_collision_query_scope():
                owner.inspect_robot_collision(robot, positions, control_part="arm")
                assert len(created) == 4
                raise KeyboardInterrupt
        with pytest.raises(RuntimeError, match="closed"):
            created[-1].inspect(positions)
        assert outer.inspect(positions) == first
        owner.inspect_robot_collision(robot, positions, control_part="arm")
        assert len(created) == 4
    with pytest.raises(RuntimeError, match="closed"):
        outer.inspect(positions)
    owner.inspect_robot_collision(robot, positions, control_part="arm")
    assert len(created) == 5


@_requires_newton
def test_owned_robot_query_scope_rejects_model_drift_without_refresh(monkeypatch):
    from embodichain.lab.sim.motion import mesh_collision as owner

    robot, _ = mesh_query_fixture()
    positions = robot.get_qpos().clone()
    with owner.robot_collision_query_scope():
        owner.inspect_robot_collision(robot, positions, control_part="arm")
        monkeypatch.setattr(robot.cfg, "fpath", str(robot.cfg.fpath) + ".changed")
        with pytest.raises(ValueError, match="Observed robot changed"):
            owner.inspect_robot_collision(robot, positions, control_part="arm")


def test_free_motion_query_preserves_all_140_frames_and_one_owned_query(monkeypatch):
    """The adapter neither decimates the frozen workload nor retries a query."""
    from embodichain.lab.sim.scene.feasibility import collision as owner

    queries = []

    class RecordingQuery:
        def __init__(self, bodies, **options):
            self.bodies = tuple(bodies)
            self.frames = []
            self.closed = False
            queries.append(self)

        def contact_pairs(self, poses, *, all_pairs):
            assert all_pairs and set(poses) == {body.frame_uid for body in self.bodies}
            self.frames.append(poses)
            return ()

        def close(self):
            self.closed = True

    monkeypatch.setattr(owner, "NewtonCollisionChecker", RecordingQuery)
    robot, simulation = mesh_query_fixture()
    report = SimulationFreeMotionQuery(robot, simulation).inspect(
        torch.zeros(140, robot.dof), control_part="arm"
    )
    assert report.valid and report.frames == 140
    assert len(queries) == 1 and len(queries[0].frames) == 140 and queries[0].closed
    assert len(queries[0].bodies) == 8  # seven robot parts and the complete world box


@pytest.mark.parametrize("damage", ["disabled", "incomplete"])
def test_free_motion_query_rejects_unavailable_world_collision_before_native_build(
    monkeypatch, damage
):
    from embodichain.lab.sim.scene.feasibility import collision as owner

    robot, simulation = mesh_query_fixture()
    body = simulation.spawn_result.get("obstacle").native().get_physical_body()
    if damage == "disabled":
        body.is_collision_enable = lambda: False
    else:
        body.get_shape_count = lambda: 2

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete or disabled world geometry cannot build a query")

    monkeypatch.setattr(owner, "NewtonCollisionChecker", forbidden)
    with pytest.raises(owner.UnsupportedFeasibilityError):
        SimulationFreeMotionQuery(robot, simulation).inspect(
            robot.get_qpos(), control_part="arm"
        )
