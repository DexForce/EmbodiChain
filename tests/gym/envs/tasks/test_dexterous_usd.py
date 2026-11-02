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

import copy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from embodichain_tasks.manipulation.dexterous._audit_model import (
    audit_model,
    snapshot_execution_model,
)


def test_static_body_merge_preserves_world_geometry() -> None:
    source = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="ground" pos="1 2 0" quat=".7071067811865476 0 0 .7071067811865476"><geom name="floor" type="box" size="1 2 .1" pos=".2 0 0"/></body><body name="ball" pos="0 0 2"><freejoint/><geom name="ball_geom" size=".1"/></body></worldbody></mujoco>'
    )
    target = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="floor" type="box" size="1 2 .1" pos="1 2.2 0" quat=".7071067811865476 0 0 .7071067811865476"/><body name="ball" pos="0 0 2"><freejoint/><geom name="ball_geom" size=".1"/></body></worldbody></mujoco>'
    )
    arguments = ({0: 0}, [], {"ground": 0, "ball": 1})
    assert audit_model(source, target, *arguments)["status"] == "passed"
    target.geom_pos[0, 0] += 0.01
    result = audit_model(source, target, *arguments)
    assert any(item["field"] == "geom[floor].geom_pos" for item in result["mismatches"])


@pytest.mark.parametrize(
    "force_range,passed", [((-np.inf, np.inf), True), ((-1e6, 1e6), False)]
)
def test_unlimited_joint_rejects_any_finite_added_limit(force_range, passed) -> None:
    source = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="arm"><joint name="hinge"/><geom size=".1"/></body></worldbody></mujoco>'
    )
    target = copy.copy(source)
    target.jnt_actfrclimited[0] = True
    target.jnt_actfrcrange[0] = force_range
    result = audit_model(source, target, {0: 0}, [], {"arm": 1})
    assert (result["status"] == "passed") == passed


@pytest.mark.parametrize("exclude,passed", [(True, True), (False, False)])
def test_parent_filter_normalization_still_checks_every_collision(
    exclude, passed
) -> None:
    xml = '<mujoco><worldbody><body name="a"><joint name="a_joint"/><geom name="ga" size=".1"/><body name="b" pos="0 0 .4"><joint name="b_joint"/><geom name="gb" size=".1"/></body></body></worldbody>{}</mujoco>'
    source = mujoco.MjModel.from_xml_string(xml.format(""))
    target = mujoco.MjModel.from_xml_string(
        xml.format(
            '<contact><exclude body1="a" body2="b"/></contact>' if exclude else ""
        )
    )
    target.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    result = audit_model(
        source,
        target,
        {0: 0, 1: 1},
        [],
        {"a": 1, "b": 2},
        explicit_collision_filtering=True,
    )
    assert (result["status"] == "passed") == passed


def test_execution_snapshot_reads_device_parameters_without_mutation() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    values = np.array([[0.25]], dtype=np.float32)
    remote = SimpleNamespace(
        dof_damping=SimpleNamespace(numpy=lambda: values), opt=SimpleNamespace()
    )
    snapshot = snapshot_execution_model(
        SimpleNamespace(mj_model=model, mjw_model=remote)
    )
    assert snapshot.dof_damping[0] == 0.25
    assert model.dof_damping[0] == 0.0
    values[0, 0] = 0.5
    assert snapshot.dof_damping[0] == 0.25


def test_snapshot_checks_per_world_inputs_without_auditing_derived_acceleration() -> (
    None
):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="joint"/><geom size=".1"/></body></worldbody><actuator><motor joint="joint"/></actuator></mujoco>'
    )
    remote = SimpleNamespace(
        actuator_acc0=SimpleNamespace(numpy=lambda: np.array([[1.0], [1.0000001]])),
        dof_damping=SimpleNamespace(numpy=lambda: np.array([[0.25], [0.25]])),
        opt=SimpleNamespace(),
    )
    solver = SimpleNamespace(mj_model=model, mjw_model=remote)
    report = {}
    assert (
        snapshot_execution_model(solver, derived_cache_report=report).dof_damping[0]
        == 0.25
    )
    assert report["actuator_acc0"]["max_world_spread"] > 0
    remote.dof_damping = SimpleNamespace(numpy=lambda: np.array([[0.25], [0.5]]))
    with pytest.raises(ValueError, match="dof_damping"):
        snapshot_execution_model(solver)


def test_snapshot_rejects_per_world_timestep_changes() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    remote = SimpleNamespace(
        opt=SimpleNamespace(
            timestep=SimpleNamespace(numpy=lambda: np.array([0.002, 0.003]))
        )
    )
    with pytest.raises(ValueError, match="Nonuniform execution option: timestep"):
        snapshot_execution_model(SimpleNamespace(mj_model=model, mjw_model=remote))


def test_dexuni_ownership_includes_free_bodies_and_replicas() -> None:
    from dexsim.spawn import (
        ArticulationDesc,
        MeshObjectDesc,
        RigidBodyPhysicsDesc,
        SceneDesc,
        NewtonPhysicsDesc,
        CollisionDesc,
    )
    from dexsim.types import ActorType
    from embodichain_tasks.manipulation.dexterous._dexuni_adapter import (
        _mujoco_articulation_ids,
    )

    scene = SceneDesc(
        articulations=[ArticulationDesc(name="hand")],
        mesh_objects=[
            MeshObjectDesc(
                name="cube",
                physics=RigidBodyPhysicsDesc(actor_type=ActorType.DYNAMIC),
                collisions=[CollisionDesc()],
            ),
            MeshObjectDesc(
                name="anchor",
                physics=RigidBodyPhysicsDesc(
                    actor_type=ActorType.KINEMATIC, newton=NewtonPhysicsDesc(mocap=True)
                ),
            ),
            MeshObjectDesc(
                name="floor",
                physics=RigidBodyPhysicsDesc(actor_type=ActorType.STATIC),
                collisions=[CollisionDesc()],
            ),
        ],
    )
    assert _mujoco_articulation_ids(scene, 1) == (0, 1, 2)
    assert _mujoco_articulation_ids(scene, 4) == tuple(range(12))
