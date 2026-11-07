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
    RobotArticulationCollisionSnapshot,
    SimulationFreeMotionQuery,
)
from _mesh_collision_fixture import mesh_query_fixture


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
    with pytest.raises(ImportError, match="Newton planning queries require"):
        RobotArticulationCollisionSnapshot(robot, simulation, control_part="arm")
    robot.get_qpos.assert_not_called()
    robot.get_local_pose.assert_not_called()


@pytest.mark.parametrize(
    ("stop", "kind", "pairs"),
    [
        ((0.125, 0.0), None, ()),
        (
            (0.25, 0.0),
            "scene",
            tuple((f"geometry_{index}", "obstacle:0") for index in (3, 4, 5)),
        ),
        (
            (0.6875, 0.0),
            "other_arm",
            tuple((f"geometry_{index}", "geometry_6") for index in (3, 4, 5)),
        ),
        (
            (0.375, -0.5),
            "self",
            tuple(("geometry_1", f"geometry_{index}") for index in (3, 4, 5)),
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
            assert options["allow_triangle_meshes"] is True
            assert any(body.kind == "mesh" for body in bodies)
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
def test_robot_articulation_snapshot_excludes_rigids_but_checks_articulations(
    monkeypatch,
):
    import trimesh
    from embodichain.lab.sim.objects.backends import _collision_inputs

    robot, simulation = mesh_query_fixture()
    positions = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0, 0.0]])
    assert (
        not SimulationFreeMotionQuery(robot, simulation)
        .inspect(positions, control_part="arm")
        .valid
    )
    with RobotArticulationCollisionSnapshot(
        robot, simulation, control_part="arm"
    ) as snapshot:
        clear = snapshot.inspect(positions)
        assert (
            clear.valid
            and clear.metadata()["scope"] == "discrete_robot_articulation_meshes"
        )
    with pytest.raises(RuntimeError, match="closed"):
        snapshot.inspect(positions)
    obstacle = simulation.get_rigid_object("obstacle")
    articulation = SimpleNamespace(
        cfg=SimpleNamespace(fpath="fixture.usda", body_scale=[1.0, 1.0, 1.0]),
        get_link_pose=lambda *args, **kw: obstacle.get_local_pose(to_matrix=True),
    )
    simulation.get_articulation_uid_list = lambda: ["drawer"]
    simulation.get_articulation = lambda uid: articulation
    monkeypatch.setattr(
        _collision_inputs,
        "_usd_collision_meshes",
        lambda cfg: {
            ("body", "mesh"): trimesh.creation.box(extents=[0.08, 0.08, 0.08])
        },
    )
    with RobotArticulationCollisionSnapshot(
        robot, simulation, control_part="arm"
    ) as snapshot:
        blocked = snapshot.inspect(positions)
        assert not blocked.valid and blocked.frame == 1 and blocked.kind == "scene"
        assert blocked.metadata()["scope"] == "discrete_robot_articulation_meshes"
        assert any("drawer:mesh" in pair for pair in blocked.pairs)


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


@_requires_newton
def test_precontact_scope_keeps_fixed_arm_and_explicit_exclusion_local():
    robot, simulation = mesh_query_fixture()
    query = SimulationFreeMotionQuery(robot, simulation)
    positions = torch.zeros(3, robot.dof)
    observed = robot.get_qpos().clone()
    report = query.inspect_precontact_closure(
        positions,
        control_part="arm",
        arm_joint_ids=(0, 1),
        source_uid="obstacle",
        contact_fraction=0.5,
        grid_samples=5,
    )
    assert report.valid and report.frames == 3
    assert report.scope == "discrete_precontact_meshes"
    assert report.excluded_rigid_uid == "obstacle"
    complete = query.inspect(positions, control_part="arm")
    assert complete.scope == "discrete_free_motion_meshes"
    assert complete.excluded_rigid_uid is None
    torch.testing.assert_close(robot.get_qpos(), observed)
    positions[1, 0] = 0.01
    with pytest.raises(ValueError, match="fixed arm"):
        query.inspect_precontact_closure(
            positions,
            control_part="arm",
            arm_joint_ids=(0, 1),
            source_uid="obstacle",
            contact_fraction=0.5,
            grid_samples=5,
        )
