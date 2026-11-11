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


def _engine():
    cfg = TrapezoidalPlannerCfg(robot_uid="analytic_cartesian_oracle")
    planner = TrapezoidalPlanner(cfg, robot=CartesianRobot())
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
