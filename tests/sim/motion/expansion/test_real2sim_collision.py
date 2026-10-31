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

"""Sampled collision evidence preserves phase-specific contact policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.data_pipeline.real2sim import (
    JointSpec,
    PhaseAnnotation,
    Real2SimTrajectory,
    build_motion_prototype,
    normalize_trajectory,
)
from embodichain.lab.sim.motion.expansion.real2sim_collision import (
    BoxCollisionBody,
    CollisionPhase,
    PlaneCollisionBody,
    SampledRobotCollisionChecker,
)
from embodichain.lab.task_program.integrations.simulation.real2sim_collision import (
    TabletopCollisionGuardFactory,
    TabletopPrototypeCollisionValidator,
    _support_errors,
)

_URDF = """\
<robot name="collision_fixture">
  <link name="base">
    <collision><origin xyz="-2 0 0"/><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="finger">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="finger"/><axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _pose(x: float) -> np.ndarray:
    result = np.eye(4)
    result[0, 3] = x
    return result


def _checker(tmp_path: Path) -> SampledRobotCollisionChecker:
    path = tmp_path / "fixture.urdf"
    path.write_text(_URDF)
    return SampledRobotCollisionChecker(
        path,
        joint_names=("slide",),
        bodies=(BoxCollisionBody("object", (0.1, 0.1, 0.1), np.eye(4)),),
        world_from_base=np.eye(4),
    )


def test_phase_contact_permissions_are_sampled_and_depth_limited(
    tmp_path: Path,
) -> None:
    checker = _checker(tmp_path)
    joints = np.zeros((2, 1))
    poses = {"object": np.stack((_pose(0.5), _pose(0.15)))}
    free = CollisionPhase("free", 0, 2, {})
    rejected = checker.validate(joints, body_poses=poses, phases=(free,))
    assert not rejected.accepted
    assert rejected.first_failure_sample == 1
    assert rejected.first_failure_pair == ("body:object", "robot:finger")

    pair = ("robot:finger", "body:object")
    phases = (
        CollisionPhase("approach", 0, 1, {}),
        CollisionPhase("grasp", 1, 2, {pair: 0.06}),
    )
    accepted = checker.validate(joints, body_poses=poses, phases=phases)
    assert accepted.accepted and accepted.allowed_contact_samples == 1
    shallow = phases[:1] + (CollisionPhase("grasp", 1, 2, {pair: 0.01}),)
    rejected = checker.validate(joints, body_poses=poses, phases=shallow)
    assert not rejected.accepted
    assert rejected.reason == "allowed contact exceeds its penetration limit"
    assert rejected.first_failure_penetration_m == pytest.approx(0.05, abs=1e-6)


def test_collision_requires_explicit_pose_and_complete_phase_mapping(
    tmp_path: Path,
) -> None:
    checker = _checker(tmp_path)
    joints = np.zeros((2, 1))
    with pytest.raises(ValueError, match="cover exactly"):
        checker.validate(
            joints, body_poses={}, phases=(CollisionPhase("all", 0, 2, {}),)
        )
    poses = {"object": np.stack((_pose(0.5), _pose(0.5)))}
    with pytest.raises(ValueError, match="cover every sample"):
        checker.validate(
            joints, body_poses=poses, phases=(CollisionPhase("part", 0, 1, {}),)
        )
    with pytest.raises(ValueError, match="unknown entity"):
        checker.validate(
            joints,
            body_poses=poses,
            phases=(
                CollisionPhase("all", 0, 2, {("body:object", "robot:missing"): 0.01}),
            ),
        )


def test_source_requires_matching_robot_and_object_geometry(tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    body = BoxCollisionBody("object", (0.1, 0.1, 0.1), np.eye(4))
    source = Real2SimTrajectory(
        source_id="source",
        source_revision="1",
        unit_id="pick",
        skill_id="pick_up",
        origin="synthetic_reference",
        joint_names=("slide",),
        joint_units=("m",),
        qpos=np.zeros((3, 1)),
        timestamps=np.array([0.0, 0.1, 0.2]),
        time_unit="s",
        phases=(
            PhaseAnnotation("approach", 0, 1, "free", "object"),
            PhaseAnnotation("close", 1, 2, "contact", "object"),
            PhaseAnnotation("lift", 2, 3, "hold", "gravity"),
        ),
        object_pose=np.eye(4),
        robot_geometry_sha256=checker.geometry_sha256,
        object_geometry_sha256=body.geometry_sha256,
    )
    checker.validate_source_geometry(source, object_uid="object")
    from dataclasses import replace

    with pytest.raises(ValueError, match="object_geometry_sha256 does not match"):
        checker.validate_source_geometry(
            replace(source, object_geometry_sha256="a" * 64), object_uid="object"
        )
    with pytest.raises(ValueError, match="robot_geometry_sha256 is unverified"):
        checker.validate_source_geometry(
            replace(source, robot_geometry_sha256=None), object_uid="object"
        )


def test_contact_sweep_finds_expected_box_contact(tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    checker.validate_prismatic_link("finger", "slide")
    # Two 0.2 m boxes touch when the finger has moved 0.3 m towards a body
    # whose center is 0.5 m away; the command endpoint is 0.4 m.
    fraction = checker.first_link_body_contact(
        np.array([0.0]),
        np.array([0.4]),
        link="finger",
        body_uid="object",
        body_pose=_pose(0.5),
    )
    # The backend's contact margin places the detected point within 2 µm.
    assert fraction == pytest.approx(0.75, abs=5e-6)
    with pytest.raises(ValueError, match="does not reach"):
        checker.first_link_body_contact(
            np.array([0.0]),
            np.array([0.1]),
            link="finger",
            body_uid="object",
            body_pose=_pose(0.5),
        )


def test_tabletop_guard_projects_command_to_geometric_contact(tmp_path: Path) -> None:
    path = tmp_path / "fixture.urdf"
    path.write_text(_URDF)
    box = BoxCollisionBody("object", (0.1, 0.1, 0.1), np.eye(4))
    support = BoxCollisionBody("support", (0.1, 0.1, 0.1), np.eye(4), True)
    base = np.eye(4)
    base[2, 3] = 0.5
    checker = SampledRobotCollisionChecker(
        path,
        joint_names=("slide",),
        bodies=(box, support, PlaneCollisionBody("ground", np.eye(4))),
        world_from_base=base,
    )
    obj = _pose(0.5)
    obj[2, 3] = 0.5
    source = Real2SimTrajectory(
        source_id="source",
        source_revision="1",
        unit_id="pick",
        skill_id="pick_up",
        origin="synthetic_reference",
        joint_names=("slide",),
        joint_units=("m",),
        qpos=np.array([[0.0], [0.0], [0.4], [0.4], [0.4]]),
        timestamps=np.arange(5) * 0.1,
        time_unit="s",
        phases=(
            PhaseAnnotation("approach", 0, 1, "free", "object"),
            PhaseAnnotation("close", 1, 3, "contact", "object"),
            PhaseAnnotation("lift", 3, 5, "hold", "gravity"),
        ),
        object_pose=obj,
        robot_geometry_sha256=checker.geometry_sha256,
        object_geometry_sha256=box.geometry_sha256,
    )
    prototype = build_motion_prototype(
        normalize_trajectory(
            source, joints=(JointSpec("slide", "prismatic"),), control_dt=0.1
        )
    )

    class Robot:
        joint_names = ("slide",)

        def compute_batch_fk(self, *, qpos, name, to_matrix):
            return (
                torch.tensor(base, dtype=qpos.dtype)
                .expand(1, qpos.shape[1], 4, 4)
                .clone()
            )

    validator = TabletopPrototypeCollisionValidator(
        robot=Robot(),
        checker=checker,
        prototypes=(prototype,),
        object_uid="object",
        target_uid="support",
        ground_uid="ground",
        finger_links=("finger",),
        finger_joints=("slide",),
        closing_directions=(1,),
        base_link="base",
        contact_depth_m=0.001,
        base_ground_depth_m=0.001,
    )
    pair = ("body:object", "body:support")
    assert (
        pair not in validator.phase_policy("pick_up", "approach", 0, 1).allowed_contacts
    )
    supported = validator.phase_policy(
        "pick_up", "approach", 0, 1, object_supported_by_target=True
    )
    assert supported.allowed_contacts[pair] == pytest.approx(0.001)
    request = SimpleNamespace(
        skill_id="pick_up",
        binding=SimpleNamespace(
            endpoint=lambda *_: SimpleNamespace(
                require_target=lambda _: SimpleNamespace(
                    joint_ids=(0,), control_part="arm"
                )
            )
        ),
    )
    target_pose = _pose(1.5)
    context = SimpleNamespace(
        batch_size=1,
        scene=SimpleNamespace(
            entities={
                "object": SimpleNamespace(pose=torch.tensor(obj)[None]),
                "support": SimpleNamespace(pose=torch.tensor(target_pose)[None]),
            }
        ),
    )
    plan = SimpleNamespace(
        skill_id="pick_up",
        joint_trajectory=SimpleNamespace(
            positions=torch.tensor([[[0.0], [0.0], [0.4], [0.4], [0.4]]]),
            waypoint_count=5,
        ),
        segments=(
            SimpleNamespace(name="approach", start=0, stop=1),
            SimpleNamespace(name="close", start=1, stop=3),
            SimpleNamespace(name="lift", start=3, stop=5),
        ),
        diagnostics=SimpleNamespace(
            metadata={"source_id": "source", "source_revision": "1", "unit_id": "pick"}
        ),
    )
    assert validator.validate(request, context, plan).accepted


def test_support_measurement_checks_height_and_rotated_footprint() -> None:
    obj = BoxCollisionBody("object", (0.02, 0.01, 0.02), np.eye(4))
    target = BoxCollisionBody("target", (0.1, 0.1, 0.05), np.eye(4), True)
    object_pose = np.eye(4)
    object_pose[:3, 3] = (0.0, 0.0, 0.07)
    target_pose = np.eye(4)
    height, overhang = _support_errors(object_pose, target_pose, obj, target)
    assert height == pytest.approx(0.0, abs=1e-12)
    assert overhang == pytest.approx(0.0, abs=1e-12)
    # A quarter turn swaps the 2 cm/1 cm footprint half-extents.
    object_pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    object_pose[0, 3] = 0.095
    object_pose[2, 3] += 0.01
    height, overhang = _support_errors(object_pose, target_pose, obj, target)
    assert height == pytest.approx(0.01)
    assert overhang == pytest.approx(0.005)


def test_refinement_support_checks_whole_footprint_and_signed_release_gap() -> None:
    from embodichain.lab.sim.motion.refinement import MotionRefinementCfg

    validator = object.__new__(TabletopPrototypeCollisionValidator)
    validator._object = BoxCollisionBody("object", (0.02, 0.01, 0.02), np.eye(4))
    validator._target = BoxCollisionBody("target", (0.1, 0.1, 0.05), np.eye(4), True)
    validator._target_uid = "target"
    validator._contact_depth = 0.001
    context = SimpleNamespace(
        scene=SimpleNamespace(
            entities={"target": SimpleNamespace(pose=torch.eye(4)[None])}
        )
    )
    pose = np.eye(4)
    pose[2, 3] = 0.08
    cfg = MotionRefinementCfg()
    evidence = validator.validate_refinement_support(context, pose, cfg)
    assert evidence["support_verified"] is True
    assert evidence["release_gap_m"] == pytest.approx(0.01)
    assert evidence["minimum_support_slack_m"] == pytest.approx(0.08)
    pose[0, 3] = 0.09  # Centre remains on support; the complete footprint does not.
    with pytest.raises(ValueError, match="footprint"):
        validator.validate_refinement_support(context, pose, cfg)
    pose[0, 3] = 0
    for height in (0.06, 0.15):
        pose[2, 3] = height
        with pytest.raises(ValueError, match="gap"):
            validator.validate_refinement_support(context, pose, cfg)


def test_plan_guard_rejects_nominal_velocity_before_collision_check() -> None:
    @dataclass(frozen=True)
    class Diagnostics:
        metadata: dict

    @dataclass(frozen=True)
    class Plan:
        joint_trajectory: object
        segments: tuple
        diagnostics: Diagnostics

    validator = object.__new__(TabletopPrototypeCollisionValidator)
    validator.validate = lambda *_: pytest.fail(
        "collision must not run for an over-limit plan"
    )
    guard = TabletopCollisionGuardFactory(
        validator, velocity_limits=(0.1,), acceleration_limits=(1.0,)
    )
    endpoint = SimpleNamespace(joint_ids=(0,), control_part="arm")
    request = SimpleNamespace(analysis_call_index=0)
    resolved = SimpleNamespace(
        binding=SimpleNamespace(
            endpoint=lambda *_: SimpleNamespace(require_target=lambda _: endpoint)
        ),
        motion_policy=SimpleNamespace(velocity_limit=None, acceleration_limit=None),
    )
    engine = SimpleNamespace(
        robot=SimpleNamespace(get_qpos_limits=lambda: torch.tensor([[[-1.0, 1.0]]]))
    )
    trajectory = SimpleNamespace(
        positions=torch.tensor([[[0.0], [0.02]]]),
        dt=torch.tensor([[0.0, 0.1]]),
        batch_size=1,
        robot_dof=1,
    )
    plan = Plan(trajectory, (), Diagnostics({}))
    with pytest.raises(ValueError, match="exceeds sampled motion limits"):
        guard.create_plan_transform(request, engine=engine)(resolved, None, plan)
