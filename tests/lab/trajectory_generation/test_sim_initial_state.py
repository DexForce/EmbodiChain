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

"""Physical initial-state contracts without creating a simulation world."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.trajectory_generation.integrations.sim import (
    SimInitialStateAdapter,
)

_BATCH = 2
_JOINTS = 3  # Include one mimic/gripper coordinate in addition to the arm.


class _Robot:
    """Mutable full-joint state with the production Robot getter/setter API."""

    def __init__(self) -> None:
        self.uid = "robot"
        self.device = torch.device("cpu")
        self.num_instances = _BATCH
        self.dof = _JOINTS
        self.joint_names = ["arm", "gripper", "mimic"]
        self.control_parts = {"arm": ["arm"], "gripper": ["gripper", "mimic"]}
        self.cfg = SimpleNamespace(fix_base=True, use_usd_properties=False)
        self.pose = torch.eye(4).repeat(_BATCH, 1, 1)
        self.pose[1, 0, 3] = 2.0
        self.qpos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.qvel = self.qpos / 10
        self.target_qpos = self.qpos + 0.01
        self.target_qvel = self.qpos / 20
        self.qf = self.qpos / 30
        self.limits = torch.tensor([-1.0, 1.0]).repeat(_BATCH, _JOINTS, 1)
        self.root_link_name = "base"
        velocities = torch.zeros(_BATCH, 2, 6)
        self.body_data = SimpleNamespace(
            link_names=["tip", "base"],  # Root selection must use identity.
            entities=[SimpleNamespace(get_root_link_name=lambda: "base")],
            body_link_vel=velocities,
            root_lin_vel=velocities[:, 1, :3],
            root_ang_vel=velocities[:, 1, 3:],
        )
        self.writes: list[str] = []
        self.after_root_write = lambda: None

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        assert to_matrix
        return self.pose

    def get_qpos(self, target: bool = False) -> torch.Tensor:
        return self.target_qpos if target else self.qpos

    def get_qvel(self, target: bool = False) -> torch.Tensor:
        return self.target_qvel if target else self.qvel

    def get_qf(self) -> torch.Tensor:
        return self.qf

    def get_qpos_limits(self) -> torch.Tensor:
        return self.limits

    def set_local_pose(self, value: torch.Tensor) -> None:
        self.writes.append("root_pose")
        self.pose.copy_(value)
        self.after_root_write()

    def set_qpos(self, value: torch.Tensor, target: bool = True) -> None:
        self.writes.append("target_qpos" if target else "qpos")
        self.get_qpos(target=target).copy_(value)

    def set_qvel(self, value: torch.Tensor, target: bool = True) -> None:
        self.writes.append("target_qvel" if target else "qvel")
        self.get_qvel(target=target).copy_(value)

    def set_qf(self, value: torch.Tensor) -> None:
        self.writes.append("qf")
        self.qf.copy_(value)


class _RigidObject:
    """Minimal rigid body retaining force state independently of velocity."""

    def __init__(self, uid: str, *, static: bool = False) -> None:
        self.uid = uid
        self.device = torch.device("cpu")
        self.num_instances = _BATCH
        self.is_static = self.is_non_dynamic = static
        self.pose = torch.eye(4).repeat(_BATCH, 1, 1)
        self.pose[:, 0, 3] = torch.tensor([0.2, 0.8])
        self.body_state = torch.zeros(_BATCH, 13)
        if not static:
            self.body_state[:, 7:] = torch.arange(12).reshape(_BATCH, 6) / 100
        self.pending_force = torch.zeros(_BATCH, 3)
        self.writes: list[str] = []

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        assert to_matrix
        return self.pose

    def set_local_pose(self, value: torch.Tensor) -> None:
        self.writes.append("pose")
        self.pose.copy_(value)

    def clear_dynamics(self) -> None:
        self.writes.append("clear_dynamics")
        self.body_state[:, 7:] = 0
        self.pending_force.zero_()

    def set_velocity(self, lin_vel: torch.Tensor, ang_vel: torch.Tensor) -> None:
        self.writes.append("velocity")
        self.body_state[:, 7:10] = lin_vel
        self.body_state[:, 10:13] = ang_vel


def _scene() -> tuple[SimpleNamespace, _Robot, SimInitialStateAdapter]:
    robot = _Robot()
    sim = SimpleNamespace(
        num_envs=_BATCH,
        _robots={robot.uid: robot},
        _rigid_objects={
            "cube": _RigidObject("cube"),
            "table": _RigidObject("table", static=True),
        },
        _articulations={},
        _rigid_object_groups={},
        _soft_objects={},
        _cloth_objects={},
        _constraints={},
    )
    return sim, robot, SimInitialStateAdapter(sim, robot)


def test_capture_owns_nested_tensor_state_and_complete_joint_order() -> None:
    sim, robot, adapter = _scene()
    state = adapter.capture()
    robot.qpos.zero_()
    sim._rigid_objects["cube"].body_state.zero_()
    assert state.joint_names == ("arm", "gripper", "mimic")
    assert state.robot["qpos"][1, 2] == pytest.approx(0.6)
    assert state.rigid_objects["cube"]["angular_velocity"][1, 2] == pytest.approx(0.11)
    with pytest.raises(TypeError):
        state.robot["qpos"] = torch.zeros(_BATCH, _JOINTS)
    with pytest.raises(TypeError):
        state.rigid_objects["unknown"] = {}


def test_restore_repeatedly_recovers_all_rows_targets_and_pending_forces() -> None:
    sim, robot, adapter = _scene()
    state = adapter.capture()
    cube = sim._rigid_objects["cube"]

    def advance_world_during_root_write() -> None:
        # The real root setter advances the world before remaining restoration.
        robot.qpos.add_(0.1)
        cube.pose[:, 0, 3] += 0.3
        cube.body_state[:, 7:] += 0.2

    robot.after_root_write = advance_world_during_root_write
    for _ in range(3):
        robot.qpos.zero_()
        robot.target_qpos.zero_()
        robot.qvel.zero_()
        robot.target_qvel.zero_()
        robot.qf.zero_()
        cube.pending_force.fill_(4)
        adapter.restore(state)
        assert adapter.verify(state).accepted
        assert not cube.pending_force.any()
        assert robot.writes[-6:] == [
            "root_pose",
            "qpos",
            "qvel",
            "target_qpos",
            "target_qvel",
            "qf",
        ]
        assert sim._rigid_objects["table"].writes[-1:] == ["pose"]


@pytest.mark.parametrize("field", ["qpos", "qvel", "target_qpos", "target_qvel", "qf"])
def test_verify_detects_joint_or_controller_drift(field: str) -> None:
    _, robot, adapter = _scene()
    state = adapter.capture()
    getattr(robot, field)[1, 2] += 0.1
    result = adapter.verify(state)
    assert not result.accepted
    assert f"robot.{field}" in result.checks[0].detail


def test_verify_detects_root_velocity_and_rigid_velocity_drift() -> None:
    sim, robot, adapter = _scene()
    state = adapter.capture()
    robot.body_data.root_lin_vel[0, 0] = 0.1
    assert not adapter.verify(state).accepted
    robot.body_data.root_lin_vel.zero_()
    sim._rigid_objects["cube"].body_state[1, 10] += 0.1
    assert not adapter.verify(state).accepted


def test_verify_nonfinite_live_state_fails_without_physical_write() -> None:
    _, robot, adapter = _scene()
    state = adapter.capture()
    robot.qvel[0, 0] = float("nan")
    assert not adapter.verify(state).accepted
    assert robot.writes == []


@pytest.mark.parametrize(
    "registry",
    [
        "_articulations",
        "_rigid_object_groups",
        "_soft_objects",
        "_cloth_objects",
        "_constraints",
    ],
)
def test_unsupported_entity_or_constraint_rejected_before_restore(
    registry: str,
) -> None:
    sim, robot, adapter = _scene()
    state = adapter.capture()
    getattr(sim, registry)["unsupported"] = object()
    with pytest.raises(ValueError, match="does not support"):
        adapter.restore(state)
    assert robot.writes == []


@pytest.mark.parametrize(
    "change",
    [
        "extra_robot",
        "partial_batch",
        "floating_base",
        "usd_base",
        "joint_order",
        "object_uid",
        "control_parts",
        "body_mode",
    ],
)
def test_topology_and_control_changes_fail_preflight(change: str) -> None:
    sim, robot, adapter = _scene()
    state = adapter.capture()
    if change == "extra_robot":
        sim._robots["other"] = _Robot()
    elif change == "partial_batch":
        robot.num_instances = 1
    elif change == "floating_base":
        robot.cfg.fix_base = False
    elif change == "usd_base":
        robot.cfg.use_usd_properties = True
    elif change == "joint_order":
        robot.joint_names.reverse()
    elif change == "object_uid":
        sim._rigid_objects["cube"].uid = "renamed"
    elif change == "control_parts":
        robot.control_parts["arm"] = ["gripper"]
    else:
        sim._rigid_objects["cube"].is_non_dynamic = True
    with pytest.raises(ValueError):
        adapter.restore(state)
    assert robot.writes == []


@pytest.mark.parametrize(
    "change",
    [
        "missing_field",
        "wrong_shape",
        "nonfinite",
        "nonrigid_pose",
        "root_velocity",
        "joint_limit",
        "object_missing",
    ],
)
def test_malformed_snapshot_rejected_before_any_write(change: str) -> None:
    _, robot, adapter = _scene()
    state = adapter.capture()
    if change == "missing_field":
        state = replace(
            state,
            robot={
                key: value for key, value in state.robot.items() if key != "target_qvel"
            },
        )
    elif change == "wrong_shape":
        state = replace(state, robot={**state.robot, "qpos": torch.zeros(1, _JOINTS)})
    elif change == "nonfinite":
        state.robot["qpos"][0, 0] = float("nan")
    elif change == "nonrigid_pose":
        state.robot["root_pose"][0, 0, 0] = -1
    elif change == "root_velocity":
        state.robot["root_linear_velocity"][0, 0] = 0.1
    elif change == "joint_limit":
        state.robot["target_qpos"][0, 0] = 2
    else:
        state = replace(state, rigid_objects={})
    with pytest.raises(ValueError):
        adapter.restore(state)
    assert robot.writes == []


def test_signature_tracks_structure_and_ignores_episode_motion() -> None:
    sim, robot, adapter = _scene()
    signature = adapter.signature()
    robot.qpos.add_(0.1)
    sim._rigid_objects["cube"].pose[:, 0, 3] += 1
    assert adapter.signature() == signature
    sim._rigid_objects = dict(reversed(tuple(sim._rigid_objects.items())))
    assert adapter.signature() == signature


def test_verification_uses_explicit_absolute_tolerance() -> None:
    sim, robot, _ = _scene()
    tolerance = 0.01
    adapter = SimInitialStateAdapter(sim, robot, atol=tolerance)
    state = adapter.capture()
    robot.qpos[0, 0] += tolerance / 2
    assert adapter.verify(state).accepted
    robot.qpos[0, 0] += tolerance
    assert not adapter.verify(state).accepted


@pytest.mark.parametrize("atol", [-1, float("nan"), float("inf"), True])
def test_invalid_tolerance_rejected(atol: float) -> None:
    sim, robot, _ = _scene()
    with pytest.raises((ValueError, TypeError)):
        SimInitialStateAdapter(sim, robot, atol=atol)


@pytest.mark.requires_sim
@pytest.mark.parametrize("physics_enabled", [False, True])
def test_real_cpu_robot_and_rigid_object_initial_state(
    tmp_path, physics_enabled
) -> None:
    """Exercise the actual CPU setters/getters without assets, IK, or cameras."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RigidObjectCfg, RobotCfg
    from embodichain.lab.sim.shapes import CubeCfg

    urdf = tmp_path / "initial_state_robot.urdf"
    urdf.write_text(
        '<robot name="initial_state_robot">'
        '<link name="base"><inertial><mass value="1"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>'
        "</inertial></link>"
        '<link name="tip"><inertial><mass value="0.1"/>'
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>'
        '</inertial><visual><geometry><box size="0.05 0.05 0.05"/></geometry></visual>'
        '<collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link>'
        '<joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>'
        '<origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>'
        '<limit lower="-1" upper="1" effort="10" velocity="2"/></joint></robot>',
        encoding="utf-8",
    )
    sim = SimulationManager(
        SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=1)
    )
    try:
        sim.enable_physics(physics_enabled)
        robot = sim.add_robot(
            RobotCfg(uid="initial_robot", fpath=str(urdf), fix_base=True)
        )
        cube = sim.add_rigid_object(
            RigidObjectCfg(
                uid="initial_cube",
                shape=CubeCfg(size=[0.05, 0.05, 0.05]),
                init_pos=[1, 0, 1],
            )
        )
        # Distinct nonzero values catch confusion between current and target APIs.
        robot.set_qpos(torch.tensor([[0.2]]), target=False)
        robot.set_qpos(torch.tensor([[0.3]]), target=True)
        robot.set_qvel(torch.tensor([[0.1]]), target=False)
        robot.set_qvel(torch.tensor([[0.15]]), target=True)
        robot.set_qf(torch.tensor([[0.05]]))
        cube.set_velocity(
            lin_vel=torch.tensor([[0.1, 0.2, 0.3]]),
            ang_vel=torch.tensor([[0.3, 0.2, 0.1]]),
        )
        adapter = SimInitialStateAdapter(sim, robot)
        state = adapter.capture()
        robot.set_qpos(torch.tensor([[-0.2]]), target=False)
        robot.set_qpos(torch.tensor([[-0.3]]), target=True)
        robot.set_qvel(torch.zeros(1, 1), target=False)
        robot.set_qvel(torch.zeros(1, 1), target=True)
        robot.set_qf(torch.zeros(1, 1))
        cube.clear_dynamics()
        pose = cube.get_local_pose(to_matrix=True)
        pose[:, 0, 3] += 0.2
        cube.set_local_pose(pose)
        adapter.restore(state)
        result = adapter.verify(state)
        assert result.accepted, result.checks
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


def test_host_planning_snapshots_own_state_and_keep_physical_case_rows() -> None:
    from embodichain.lab.sim.motion.expansion import (
        SceneCase,
        ValidationCheck,
        ValidationResult,
    )
    from embodichain.lab.trajectory_generation.initial_state import (
        FixedSceneHost,
        InitialStateProfile,
    )

    sim, robot, adapter = _scene()
    cases = tuple(
        SceneCase(f"case-{row}", f"initial-{row}", "scene", "move", "robot")
        for row in range(_BATCH)
    )
    profile = InitialStateProfile(
        profile_id="fixed_scene_initial_state",
        prepare=lambda: None,
        signature=lambda: "fixed-configuration",
        verify=lambda cases: ValidationResult(
            (ValidationCheck("task_initial", "passed"),)
        ),
    )
    with FixedSceneHost(adapter, profile) as host:
        binding = host.acquire_case(cases)
        snapshots = host.snapshots(binding)
        assert [value.scene_case for value in snapshots] == list(cases)
        assert torch.equal(snapshots[1].joint_positions, robot.qpos[1])
        assert torch.equal(snapshots[1].root_pose, robot.pose[1])
        expected = snapshots[0].joint_positions.clone()
        snapshots[0].joint_positions.zero_()
        snapshots[0].entity_poses["cube"].zero_()
        fresh = host.snapshots(binding)
        assert torch.equal(fresh[0].joint_positions, expected)
        assert torch.equal(
            fresh[0].entity_poses["cube"], sim._rigid_objects["cube"].pose[0]
        )
        assert host.initial_observation(binding) is None
        host.restore_initial()
        with pytest.raises(RuntimeError, match="obsolete"):
            host.snapshots(binding)
