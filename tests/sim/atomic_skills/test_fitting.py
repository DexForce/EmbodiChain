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

"""Independent Cartesian robot oracle for the inverse/native/splice contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from embodichain.data_pipeline.skill_fitting import JointRecording
from embodichain.lab.sim.atomic_skills import (
    AtomicSkillEngine,
    ControlPartCommandProfile,
)
from embodichain.lab.sim.atomic_skills.fitting import fit_skill_segment, SkillFitCfg
from embodichain.lab.sim.motion.motion_generator import MotionGenerator, MotionGenCfg
from embodichain.lab.sim.motion.planners import (
    TrapezoidalPlanner,
    TrapezoidalPlannerCfg,
)


def test_annotation_choices_cover_builtin_skills() -> None:
    """Annotation-only users can choose every built-in skill, including DemoSkill."""
    from embodichain.lab.sim.atomic_skills.primitives import BUILTIN_SKILL_TYPES
    from embodichain.toolkits.trajectory_annotator._skills import SKILL_CHOICES

    assert {identifier for identifier, _ in SKILL_CHOICES} == {
        skill.skill_id for skill in BUILTIN_SKILL_TYPES
    }


class CartesianRobot:
    """Six independent pose coordinates and one hand, with analytic FK/IK."""

    device = torch.device("cpu")
    dof = 7
    num_instances = 1
    joint_names = tuple(f"coordinate{i}" for i in range(7))
    control_parts = {"tool_motion": range(6), "fingers": [6]}
    mimic_ids = ()
    mimic_parents = ()
    mimic_multipliers = ()

    def get_joint_ids(self, name=None):
        return list(range(self.dof)) if name is None else list(self.control_parts[name])

    def get_qpos(self, name=None):
        return torch.zeros(1, len(self.get_joint_ids(name)))

    def get_qpos_limits(self, name=None, joint_ids=None):
        ids = self.get_joint_ids(name) if joint_ids is None else joint_ids
        return torch.tensor([[-10.0, 10.0]] * len(ids))[None]

    def get_qvel_limits(self, name=None, joint_ids=None):
        ids = self.get_joint_ids(name) if joint_ids is None else joint_ids
        return torch.full((1, len(ids)), 100.0)

    def get_solver(self, name):
        return SimpleNamespace(supports_continuous_batch_ik=False)

    def compute_fk(self, qpos, name, to_matrix=True, **kwargs):
        result = torch.eye(4).repeat(len(qpos), 1, 1)
        result[:, :3, 3] = qpos[:, :3]
        result[:, :3, :3] = torch.tensor(
            Rotation.from_rotvec(qpos[:, 3:6].numpy()).as_matrix(), dtype=torch.float32
        )
        return result

    def compute_batch_fk(self, qpos, name, to_matrix=True, **kwargs):
        return self.compute_fk(qpos.reshape(-1, 6), name).reshape(
            *qpos.shape[:-1], 4, 4
        )

    def compute_ik(self, pose, name, joint_seed, **kwargs):
        qpos = torch.cat(
            (
                pose[:, :3, 3],
                torch.tensor(
                    Rotation.from_matrix(pose[:, :3, :3].numpy()).as_rotvec(),
                    dtype=torch.float32,
                ),
            ),
            dim=-1,
        )
        return torch.ones(len(pose), dtype=torch.bool), qpos

    def compute_batch_ik(self, pose, name, joint_seed, **kwargs):
        valid, q = self.compute_ik(pose.reshape(-1, 4, 4), name, joint_seed)
        return valid.reshape(pose.shape[:2]), q.reshape(*pose.shape[:2], 6)


def _engine(robot=None):
    cfg = TrapezoidalPlannerCfg(robot_uid="analytic_cartesian_oracle")
    planner = TrapezoidalPlanner(
        cfg, robot=CartesianRobot() if robot is None else robot
    )
    generator = MotionGenerator(MotionGenCfg(planner_cfg=cfg), planner=planner)
    return AtomicSkillEngine(
        generator,
        control_profiles={
            "fingers": ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.0]), grasp=torch.tensor([0.03])
            )
        },
    )


def _pickup():
    # Independent source: a vertical approach, linear closure, 12 cm lift.
    q = np.zeros((150, 7), dtype=np.float32)
    q[:, 0] = 0.2
    q[:, 2] = np.r_[
        np.full(6, 0.2),
        np.linspace(0.2, 0.05, 61),
        np.full(13, 0.05),
        np.linspace(0.05, 0.17, 61),
        np.full(9, 0.17),
    ]
    q[:, 6] = np.r_[np.zeros(67), np.linspace(0, 0.03, 13), np.full(70, 0.03)]
    return JointRecording(
        np.arange(150) * 0.01, q, CartesianRobot.joint_names, "design_reference"
    )


@pytest.mark.parametrize("accept", (True, False))
def test_pickup_inverse_native_and_splice_preserve_source_and_report_rejection(
    accept,
) -> None:
    recording = _pickup()
    before = recording.qpos.copy()
    arrays, report = fit_skill_segment(
        recording,
        _engine(),
        skill_id="pick_up",
        endpoints={"primary": {"motion": "tool_motion", "grasp": "fingers"}},
        start=6,
        stop=141,
        acceleration_limits=np.full(7, 1000.0 if accept else 0.001),
        cfg=SkillFitCfg(max_candidates=1),
    )
    assert report["parameters"]["lift_height"] == pytest.approx(0.12, abs=1e-6)
    assert report["accepted"] == accept
    assert report["physical_success"] == "unknown"
    np.testing.assert_array_equal(recording.qpos, before)
    np.testing.assert_array_equal(arrays["patched_qpos"][:9], before[:9])
    np.testing.assert_array_equal(arrays["patched_qpos"][138:], before[138:])
    assert "phase_progress" in arrays
    if not accept:
        np.testing.assert_array_equal(arrays["patched_qpos"], before)
        assert report["failures"]
        assert report["limits"]["acceleration_ratio"] > 1
        assert np.ptp(arrays["native_phase_qpos"][:, 2]) > 0.1


def test_scene_pickup_uses_object_grasp_preserves_boundary_and_unselected_joints() -> (
    None
):
    """A valid object grasp may differ from rotating/noisy real TCP samples."""
    from embodichain.lab.sim.atomic_skills import ObjectSemantics
    from embodichain.lab.sim.atomic_skills.affordance import AntipodalAffordance
    from embodichain.lab.sim.atomic_skills.primitives import GraspGoal

    class RobotWithPassiveAxis(CartesianRobot):
        dof = 8
        joint_names = tuple(f"coordinate{i}" for i in range(8))

    source = _pickup()
    qpos = np.column_stack((source.qpos, np.sin(source.timestamps) * 0.01)).astype(
        np.float32
    )
    # There is no stable pre-grasp orientation, and the valid grasp differs
    # from the real TCP by 3 cm (above the old 1.5 cm reconstruction gate).
    qpos[:, 3] = np.linspace(0, 0.2, len(qpos))
    recording = JointRecording(
        source.timestamps, qpos, RobotWithPassiveAxis.joint_names, "measured_state"
    )
    anchor = torch.eye(4)
    anchor[:3, 3] = torch.tensor([0.23, 0.0, 0.05])
    goal = GraspGoal(
        semantics=ObjectSemantics(AntipodalAffordance(), {}, "known_object"),
        object_pose=torch.eye(4),
        grasp_xpos=anchor,
    )
    arrays, report = fit_skill_segment(
        recording,
        _engine(RobotWithPassiveAxis()),
        skill_id="pick_up",
        endpoints={"primary": {"motion": "tool_motion", "grasp": "fingers"}},
        start=6,
        stop=141,
        acceleration_limits=np.full(8, 1000.0),
        cfg=SkillFitCfg(max_candidates=1),
        scene_grasps=(goal,),
    )
    assert report["accepted"], report["failures"]
    assert report["parameters"]["object_id"] == "known_object"
    assert report["parameters"]["lift_height"] == pytest.approx(0.12, abs=1e-6)
    close = next(p for p in report["phases"] if p["name"] == "close")
    a, b = report["start"] + close["start"], report["start"] + close["stop"]
    np.testing.assert_allclose(
        arrays["patched_qpos"][a:b, :3], np.tile([0.23, 0, 0.05], (b - a, 1)), atol=1e-6
    )
    np.testing.assert_array_equal(arrays["patched_qpos"][:9], qpos[:9])
    np.testing.assert_array_equal(arrays["patched_qpos"][138:], qpos[138:])
    np.testing.assert_array_equal(arrays["patched_qpos"][:, 7], qpos[:, 7])


@pytest.mark.parametrize("fixed_hints", [False, True])
def test_scene_phase_retiming_respects_explicit_hints(fixed_hints) -> None:
    """A late observed close leaves too little time for a changed grasp to exit."""
    from embodichain.lab.sim.atomic_skills import ObjectSemantics
    from embodichain.lab.sim.atomic_skills.affordance import AntipodalAffordance
    from embodichain.lab.sim.atomic_skills.primitives import GraspGoal

    class LimitedXSpeedRobot(CartesianRobot):
        def get_qvel_limits(self, name=None, joint_ids=None):
            ids = self.get_joint_ids(name) if joint_ids is None else joint_ids
            return torch.tensor([[1.0 if i == 0 else 100.0 for i in ids]])

    qpos = np.zeros((150, 7), dtype=np.float32)
    qpos[:, 0] = 0.2
    qpos[:, 2] = np.r_[
        np.linspace(0.2, 0.05, 100),
        np.full(13, 0.05),
        np.linspace(0.05, 0.17, 28),
        np.full(9, 0.17),
    ]
    qpos[:, 6] = np.r_[np.zeros(100), np.linspace(0, 0.03, 13), np.full(37, 0.03)]
    recording = JointRecording(
        np.arange(150) * 0.01, qpos, CartesianRobot.joint_names, "design_reference"
    )
    goal_pose = torch.eye(4)
    goal_pose[:3, 3] = torch.tensor([0.4, 0, 0.05])
    goal = GraspGoal(
        semantics=ObjectSemantics(AntipodalAffordance(), {}, "target"),
        grasp_xpos=goal_pose,
        object_pose=torch.eye(4),
    )
    hints = [
        dict(name="approach", start=0, stop=94),
        dict(name="close", start=94, stop=107),
        dict(name="lift", start=107, stop=135),
    ]
    arrays, report = fit_skill_segment(
        recording,
        _engine(LimitedXSpeedRobot()),
        skill_id="pick_up",
        endpoints={"primary": {"motion": "tool_motion", "grasp": "fingers"}},
        start=6,
        stop=141,
        acceleration_limits=np.array([10] + [1000] * 6),
        cfg=SkillFitCfg(max_candidates=1),
        scene_grasps=(goal,),
        phase_hints=hints if fixed_hints else None,
    )
    assert report["accepted"] == (not fixed_hints), report["failures"]
    assert report["source_phases"] == hints
    if fixed_hints:
        assert report["phases"] == hints
        np.testing.assert_array_equal(arrays["patched_qpos"], qpos)
    else:
        close = report["phases"][1]
        assert close["start"] < hints[1]["start"]
        assert close["stop"] - close["start"] == 13
        assert report["limits"]["velocity_ratio"] <= 1


@pytest.mark.parametrize("skill", ("slide", "twist"))
def test_axis_skills_identify_geometric_parameters_and_use_native_motion(skill) -> None:
    engine = _engine()
    # Known rigid source path, independent of the skill's planner code.
    grasp = np.array([0.2, 0, 0.05, 2.0, 0, 0])
    rotation = Rotation.from_rotvec(grasp[3:]).as_matrix()
    pre = grasp.copy()
    pre[:3] -= rotation[:, 2] * 0.1
    if skill == "slide":
        # Native pull approaches against the positive rail, then moves negative.
        pre = grasp.copy()
        pre[0] -= 0.1
        end = grasp.copy()
        end[0] -= 0.15
        contact = np.linspace(grasp, end, 61)
    else:
        angle = np.linspace(0, 0.6, 61)
        contact = np.repeat(grasp[None], 61, axis=0)
        contact[:, 3:] = Rotation.from_matrix(
            Rotation.from_rotvec(angle[:, None] * np.array([0, 0, 1])).as_matrix()
            @ rotation
        ).as_rotvec()
        end = contact[-1]
    arm = np.concatenate(
        (
            np.repeat(pre[None], 6, axis=0),
            np.linspace(pre, grasp, 61),
            np.repeat(grasp[None], 13, axis=0),
            contact,
            np.repeat(end[None], 13, axis=0),
        )
    )
    hand = np.r_[
        np.zeros(67),
        np.linspace(0, 0.03, 13),
        np.full(61, 0.03),
        np.linspace(0.03, 0, 13),
    ]
    if skill == "twist":
        arm = np.concatenate((arm, np.linspace(end, pre, 61)))
        hand = np.r_[hand, np.zeros(61)]
    arm = np.concatenate((arm, np.repeat(arm[-1:], 9, axis=0)))
    hand = np.r_[hand, np.zeros(9)]
    q = np.column_stack((arm, hand)).astype(np.float32)
    source = JointRecording(
        np.arange(len(q)) * 0.01, q, CartesianRobot.joint_names, "design_reference"
    )
    arrays, report = fit_skill_segment(
        source,
        engine,
        skill_id=skill,
        endpoints={"primary": {"motion": "tool_motion", "grasp": "fingers"}},
        start=0,
        stop=len(q),
        acceleration_limits=np.full(7, 1000.0),
        cfg=SkillFitCfg(max_candidates=1),
    )
    assert report["accepted"], report["failures"]
    parameter, expected = (
        ("translation_distance", 0.15) if skill == "slide" else ("twist_angle", 0.6)
    )
    assert report["parameters"][parameter] == pytest.approx(expected, abs=1e-5)
    assert report["native_joint_rmse"] < 0.01
    np.testing.assert_array_equal(arrays["patched_qpos"][:3], q[:3])


def test_handover_identifies_both_participants_without_object_attachment() -> None:
    class DualCartesianRobot(CartesianRobot):
        dof = 14
        joint_names = tuple(f"coordinate{i}" for i in range(14))
        control_parts = {
            "sender_motion": range(6),
            "sender_hand": [6],
            "receiver_motion": range(7, 13),
            "receiver_hand": [13],
        }

    robot = DualCartesianRobot()
    cfg = TrapezoidalPlannerCfg(robot_uid="dual_cartesian_oracle")
    planner = TrapezoidalPlanner(cfg, robot=robot)
    engine = AtomicSkillEngine(
        MotionGenerator(MotionGenCfg(planner_cfg=cfg), planner=planner),
        control_profiles={
            part: ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.0]), grasp=torch.tensor([0.03])
            )
            for part in ("sender_hand", "receiver_hand")
        },
    )
    state = np.zeros(14)
    state[:3], state[7:10], state[6] = [0.2, 0, 0.2], [0.2, -0.3, 0.2], 0.03
    parts = []

    def stage(indices, values, count):
        nonlocal state
        target = state.copy()
        target[indices] = values
        path = np.linspace(state, target, count)
        parts.append(path)
        state = target

    stage(slice(0, 3), [0.2, 0, 0.3], 40)
    stage(slice(7, 10), [0.2, 0, 0.2], 20)
    stage(slice(7, 10), [0.2, 0, 0.3], 20)
    stage(13, 0.03, 13)
    stage(13, 0.03, 6)
    stage(6, 0.0, 13)
    stage(slice(0, 3), [0.2, -0.2, 0.3], 40)
    stage(6, 0.0, 9)
    q = np.concatenate(parts).astype(np.float32)
    source = JointRecording(
        np.arange(len(q)) * 0.01, q, robot.joint_names, "design_reference"
    )
    _, report = fit_skill_segment(
        source,
        engine,
        skill_id="hand_over",
        endpoints={
            "source": {"motion": "sender_motion", "grasp": "sender_hand"},
            "destination": {"motion": "receiver_motion", "grasp": "receiver_hand"},
        },
        start=0,
        stop=len(q),
        acceleration_limits=np.full(14, 1000.0),
        cfg=SkillFitCfg(max_candidates=1),
    )
    assert report["accepted"], report["failures"]
    assert report["physical_success"] == "unknown"
    np.testing.assert_allclose(
        report["parameters"]["relative_exchange"], np.eye(4), atol=1e-6
    )


def test_moving_arm_during_closure_is_rejected_as_template_mismatch() -> None:
    original = _pickup()
    q = original.qpos.copy()
    q[70:77, 0] += 0.03
    source = JointRecording(
        original.timestamps, q, original.joint_names, original.semantics
    )
    with pytest.raises(ValueError, match="stationary arm during closure"):
        fit_skill_segment(
            source,
            _engine(),
            skill_id="pick_up",
            endpoints={"primary": {"motion": "tool_motion", "grasp": "fingers"}},
            start=6,
            stop=141,
            acceleration_limits=np.full(7, 1000.0),
        )


@pytest.mark.parametrize("recipe,operator", [(1, "joint_residual"), (2, "via_points")])
def test_sequence_expansion_preserves_gripper_contact_and_unbound_joints(
    tmp_path, recipe, operator
):
    from embodichain.lab.sim.atomic_skills import (
        PickUpMotionGoal,
        PickUpOptions,
        MotionPolicy,
    )
    from embodichain.lab.sim.motion.expansion import load_expansion_profile
    from embodichain.toolkits.skill_fitting._sequence_expansion import (
        _transform_factory,
        write_sequence_expansion_profile,
    )

    class WithPassiveJoint(CartesianRobot):
        uid = "independent_cartesian"
        dof = 8
        joint_names = tuple(f"coordinate{i}" for i in range(8))

    robot = WithPassiveJoint()
    robot.body_data = SimpleNamespace(qpos_limits=robot.get_qpos_limits())
    engine = _engine(robot)
    goal = torch.eye(4)
    goal[:3, 3] = torch.tensor([0.2, 0, 0.1])
    invocation = engine.make_invocation(
        "pick_up",
        PickUpMotionGoal(goal),
        control_parts={"primary": {"motion": "tool_motion", "grasp": "fingers"}},
        motion_policy=MotionPolicy(
            strategy="motion_gen", sample_count=80, acceleration_limit=0.5
        ),
        skill_options=PickUpOptions(
            lift_height=0.12, pre_grasp_distance=0.1, hand_interp_steps=20
        ),
    )
    context = engine.initial_context(control_dt=0.01)
    baseline = engine.plan(invocation, context)
    write_sequence_expansion_profile(tmp_path, [dict(kind="atomic_skill")])
    profile = load_expansion_profile(tmp_path / "expansion.yaml")
    limits = SimpleNamespace(
        joint_names=robot.joint_names,
        velocity_limits=[100.0] * 8,
        acceleration_limits=[1000.0] * 8,
    )
    transform = _transform_factory(profile, "a" * 64, recipe, 0, limits)(0, engine)
    expanded = engine.plan(invocation, context, plan_transform=transform)
    repeated = engine.plan(invocation, context, plan_transform=transform)
    original = baseline.joint_trajectory.positions
    actual = expanded.joint_trajectory.positions
    assert bool(expanded.plan_success.all())
    assert (actual - original).abs().max() > 1e-5
    torch.testing.assert_close(actual[:, :, 6:], original[:, :, 6:], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, [0, -1]], original[:, [0, -1]], rtol=0, atol=0)
    for segment in baseline.segments:
        if segment.name == "close":
            torch.testing.assert_close(
                actual[:, segment.start : segment.stop],
                original[:, segment.start : segment.stop],
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(
        expanded.joint_trajectory.dt, baseline.joint_trajectory.dt, rtol=0, atol=0
    )
    torch.testing.assert_close(
        repeated.joint_trajectory.positions, actual, rtol=0, atol=0
    )
    assert expanded.diagnostics.metadata["expansion"]["rows"][0]["operator"] == operator
