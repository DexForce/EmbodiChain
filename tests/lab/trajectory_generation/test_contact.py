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

"""Physical contact evidence, phase permissions and complete collision geometry."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    MotionSnapshot,
    SceneCase,
    TrajectoryPhase,
)
from embodichain.lab.trajectory_generation.integrations.contact import (
    PickUpContactProfile,
    PickUpMotionValidator,
)


def _monitor():
    validator = object.__new__(PickUpMotionValidator)
    validator.profile = PickUpContactProfile()
    validator.world = SimpleNamespace(adjacent_pairs=set())
    validator.robot = SimpleNamespace(num_instances=2)
    validator._users = {
        1: (0, "cube"),
        2: (0, "gripper_finger1_link_1"),
        3: (0, "gripper_finger2_link_1"),
        4: (0, "bench"),
        5: (0, "forearm"),
        11: (1, "cube"),
        12: (1, "gripper_finger1_link_1"),
    }
    data = np.zeros((2, 11))
    data[:, 9] = 0.001
    native = {"data": data, "users": np.array([[1, 2], [3, 1]])}
    validator._physics = SimpleNamespace(
        get_cpu_contact_buffer=lambda: (native["data"], native["users"])
    )
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[:, 2, 3] = 0.505
    observed = {"object_pose": pose.clone(), "tcp_pose": pose.clone()}
    validator.observations = lambda: observed
    phases = (
        TrajectoryPhase("transit", 0, 1),
        TrajectoryPhase("approach", 1, 2, kind="contact"),
        TrajectoryPhase("close", 2, 3, kind="contact"),
        TrajectoryPhase("lift", 3, 4, kind="contact"),
        TrajectoryPhase("hold", 4, 6, kind="hold"),
    )
    cases = [
        SceneCase(f"case_{i}", "initial", "scene", "pickup", "robot") for i in range(2)
    ]
    initial = torch.eye(4)
    initial[2, 3] = 0.325
    snapshots = [
        MotionSnapshot(
            case,
            ("joint",),
            torch.zeros(1),
            torch.zeros(1),
            torch.eye(4),
            {"cube": initial},
        )
        for case in cases
    ]
    candidates = [
        CandidateTrajectoryBatch(
            torch.zeros(1, 6, 1),
            torch.tensor([[0.0, 0.05, 0.05, 0.05, 0.05, 0.05]]),
            torch.tensor([6]),
            (
                CandidateIdentity(
                    case.scene_case_id,
                    "initial",
                    f"candidate_{i}",
                    f"family_{i}",
                    "source",
                    "v1",
                    "template",
                ),
            ),
            ("joint",),
            (phases,),
            source_row_indices=torch.tensor([i]),
        )
        for i, case in enumerate(cases)
    ]
    validator.begin_rollout(candidates, snapshots)
    return validator, native, observed


def _hold(validator):
    for _ in range(5):
        validator.observe_substep(4, (True, False), physics_dt=0.25)
    return validator.rollout_validation(0)


def test_stable_lift_requires_both_real_finger_contacts():
    validator, native, _ = _monitor()
    assert _hold(validator).accepted
    validator, native, _ = _monitor()
    native["data"][1, 9] = 0.0  # Geometric proximity without normal impulse.
    checks = {c.check_id: c for c in _hold(validator).checks}
    assert checks["physical_contacts"].status == "passed"
    assert checks["held_object_stability"].status == "failed"
    assert checks["held_object_stability"].metrics["finger_1_contact_fraction"] == 0


@pytest.mark.parametrize(
    "failure", ["table", "robot", "unknown", "cross_row", "penetration"]
)
def test_forbidden_contact_rejects_even_a_stable_lift(failure):
    validator, native, _ = _monitor()
    if failure == "table":
        native["users"][0] = (1, 4)
    if failure == "robot":
        native["users"][0] = (1, 5)
    if failure == "unknown":
        native["users"][0] = (1, 9999)
    if failure == "cross_row":
        native["users"][0] = (1, 11)
    if failure == "penetration":
        native["data"][0, 10] = -0.003
    checks = {c.check_id: c for c in _hold(validator).checks}
    assert checks["physical_contacts"].status == "failed"


@pytest.mark.parametrize("failure", ["slip", "rotate", "fall", "empty", "short"])
def test_hold_rejects_slipping_rotating_falling_and_empty_grasps(failure):
    validator, _, observed = _monitor()
    validator.observe_substep(4, (True, False), physics_dt=0.25)
    if failure == "slip":
        observed["object_pose"][0, 0, 3] += 0.02
    if failure == "rotate":
        a = 0.25
        observed["object_pose"][0, :2, :2] = torch.tensor(
            [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        )
    if failure == "fall":
        observed["object_pose"][0, 2, 3] = 0.325
    if failure == "empty":
        observed["object_pose"][0, 0, 3] += 0.1
    result = validator.rollout_validation(0) if failure == "short" else _hold(validator)
    assert result.checks[1].status == "failed"


def test_contact_permissions_depend_on_phase_and_grasp_entry_distance():
    validator, _, _ = _monitor()
    finger = validator.profile.finger_links[0]
    assert not validator._allowed("cube", finger, "transit", 0.0, 0.01)
    assert not validator._allowed("cube", finger, "approach", 0.0, 0.05)
    assert validator._allowed("cube", finger, "approach", 0.0, 0.01)
    assert validator._allowed("cube", finger, "close", 0.0)
    assert not validator._allowed("cube", "forearm", "close", 0.0)
    assert validator._allowed("cube", "bench", "lift", 0.001)
    assert not validator._allowed("cube", "bench", "hold", 0.001)


@pytest.mark.parametrize("failure", ["overflow", "nan", "shape"])
def test_missing_or_unbounded_native_evidence_raises(failure):
    validator, native, _ = _monitor()
    if failure == "overflow":
        validator.profile.max_contacts_per_step = 1
    if failure == "nan":
        native["data"][0, 9] = np.nan
    if failure == "shape":
        native["users"] = np.zeros((1, 2))
    with pytest.raises(ValueError, match="Native contact evidence"):
        _hold(validator)


def test_unobserved_hold_cannot_pass():
    validator, _, _ = _monitor()
    assert not validator.rollout_validation(0).accepted


def test_full_state_collision_uses_collision_shapes_and_moving_fingers(tmp_path):
    pytest.importorskip("fcl")
    pytest.importorskip("yourdfpy")
    import pytorch_kinematics as pk
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.trajectory_generation.integrations._collision import (
        _FullStateCollisionWorld,
    )

    # Deliberately no visual mesh: a render-geometry fallback would miss this finger.
    urdf = """<robot name="test"><link name="base"/>
    <link name="finger"><collision><origin xyz="0.2 0 0"/><geometry><box size="0.02 0.02 0.02"/></geometry></collision></link>
    <joint name="slide" type="prismatic"><parent link="base"/><child link="finger"/><axis xyz="1 0 0"/><limit lower="0" upper="1" velocity="1" effort="1"/></joint></robot>"""
    path = tmp_path / "robot.urdf"
    path.write_text(urdf)
    obj = SimpleNamespace(
        cfg=SimpleNamespace(shape=CubeCfg(size=(0.05, 0.05, 0.05))),
        get_body_scale=lambda: torch.ones(1, 3),
    )
    sim = SimpleNamespace(
        get_rigid_object_uid_list=lambda: ("cube", "bench"),
        get_rigid_object=lambda uid: obj,
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(body_scale=(1.0, 1.0, 1.0), fpath=str(path)),
        device=torch.device("cpu"),
        joint_names=("slide",),
        pk_chain=pk.build_chain_from_urdf(urdf),
    )
    world = _FullStateCollisionWorld(sim, robot)
    root = torch.eye(4)
    root[2, 3] = 1.0
    poses = world.link_poses(torch.tensor([[0.0], [0.1]]), root)
    cube = np.eye(4)
    cube[:3, 3] = (0.2, 0.0, 1.0)
    bench = np.eye(4)
    bench[:3, 3] = (0.5, 0.0, 1.0)
    objects = {"cube": cube, "bench": bench}
    assert (
        world.collisions(
            {"finger": poses["finger"][0]}, objects, "cube", lambda a, b: False
        )
        == "finger / cube"
    )
    assert (
        world.collisions(
            {"finger": poses["finger"][1]}, objects, "cube", lambda a, b: False
        )
        is None
    )
    objects["cube"] = bench.copy()  # Held object can collide while the arm clears.
    assert (
        world.collisions(
            {"finger": poses["finger"][1]}, objects, "cube", lambda a, b: False
        )
        == "cube / bench"
    )


def test_cross_row_contact_rejects_active_second_body_when_first_row_is_idle():
    validator, native, _ = _monitor()
    native["users"][0] = (11, 1)
    assert _hold(validator).checks[0].status == "failed"


def test_slip_during_lift_is_not_hidden_by_a_stable_terminal_hold():
    validator, _, observed = _monitor()
    validator.observe_substep(3, (True, False), physics_dt=0.25)
    observed["object_pose"][0, 0, 3] += 0.02
    assert _hold(validator).checks[1].status == "failed"
