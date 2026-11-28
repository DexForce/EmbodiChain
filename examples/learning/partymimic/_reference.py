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

"""Contact-timed RP1 reference poses for the reconstructed carrying demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.extensions.models.policy.partymimic import (
    PartyMimicPolicy,
    PartyMimicState,
)

__all__: list[str] = []

CONTROL_DT = 0.02
PHYSICS_DT = 0.005
# Box dimensions are reconstructed from the Carrying loads footage.
STEP_HEIGHT = 0.50
STEP_FRONT = 0.23
STEP_DEPTH = 0.54
STEP_WIDTH = 0.56
FOOT_SOLE_OFFSET = 0.045
CRATE_OFFSET = (0.13, 0.0, -0.055)


@dataclass(frozen=True)
class StepPlatform:
    """Platform geometry associated with the reference motion."""

    front: float
    size: tuple[float, float, float]


_BASELINE_PLATFORM = StepPlatform(STEP_FRONT, (STEP_DEPTH, STEP_WIDTH, STEP_HEIGHT))
# Reconstructed from the platform edges and RP1 foot scale in the payload clip.
_RP_HAND_PLATFORM = StepPlatform(0.26, (0.56, 0.65, 0.48))


@dataclass(frozen=True)
class Reference:
    """Sparse5 pose sequence and the scene dimensions used to generate it."""

    poses: np.ndarray
    initial_joints: np.ndarray
    control_steps: int
    payload_kg: float
    payload_size: tuple[float, float, float]
    platform: StepPlatform | None = None


def mirror_sparse_poses(poses: np.ndarray) -> np.ndarray:
    """Reflect sagittally and exchange the left and right wrists and ankles."""
    mirrored = poses[..., [0, 1, 3, 2, 5, 4], :].copy()
    mirrored[..., [1, 3, 5]] *= -1
    return mirrored


class SagittalMirror:
    """Transform RP1 state and joint targets between sagittally reflected frames."""

    def __init__(self, model: mujoco.MjModel, joint_names: tuple[str, ...]) -> None:
        indices = {name: index for index, name in enumerate(joint_names)}
        self._partners = np.asarray(
            [
                indices[
                    name.replace("left_", "@")
                    .replace("right_", "left_")
                    .replace("@", "right_")
                ]
                for name in joint_names
            ]
        )
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        axes = np.asarray([data.xaxis[model.joint(name).id] for name in joint_names])
        # Angular vectors reflect as -M w, with M = diag(1, -1, 1).
        reflected_axes = axes * [-1, 1, -1]
        alignment = np.sum(axes[self._partners] * reflected_axes, axis=1)
        if not np.allclose(np.abs(alignment), 1, atol=1e-6):
            raise ValueError("Expected bilateral RP1 joint axes.")
        self._signs = np.sign(alignment).astype(np.float32)

    def joints(self, values: np.ndarray) -> np.ndarray:
        """Reflect positions, velocities or position targets in policy joint order."""
        return values[self._partners] * self._signs

    def state(self, state: PartyMimicState) -> PartyMimicState:
        """Reflect an observed RP1 sample before inference."""
        return PartyMimicState(
            self.joints(np.asarray(state.joint_position)),
            self.joints(np.asarray(state.joint_velocity)),
            np.asarray(state.root_quaternion) * [-1, 1, -1, 1],
            np.asarray(state.root_angular_velocity) * [-1, 1, -1],
        )


def _smooth(x: np.ndarray) -> np.ndarray:
    u = np.clip(x, 0, 1)
    return u * u * (3 - 2 * u)


def _move(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return _smooth((t - start) / (end - start))


def make_reference(
    scenario: str,
    model_path: Path,
    policy: PartyMimicPolicy,
    *,
    video_step: bool = False,
    rp_hands: bool = False,
) -> Reference:
    """Build named reference poses from the calibrated RP1 standing kinematics."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joints = dict(zip(policy.joint_names, policy.default_joint_position, strict=True))
    # Calibrated chest-height grip with both native wrists outside the crate.
    arm_pose = {
        "shoulder_pitch": -0.5292,
        "shoulder_roll": 0.54225,
        "shoulder_yaw": 0.27654,
        "elbow": -0.08246,
        "wrist_roll": -0.50299,
        "wrist_yaw": -0.18068,
        "wrist_pitch": 0.30428,
    }
    for side, sign in (("left", 1), ("right", -1)):
        for joint, position in arm_pose.items():
            joints[f"{side}_{joint}_joint"] = sign * position
    qpos = np.asarray(list(joints.values()), dtype=np.float32)
    for name, position in zip(joints, qpos, strict=True):
        data.joint(name).qpos[:] = position
    mujoco.mj_forward(model, data)
    ids = [model.body(name).id for name in policy.keypoint_names]
    data.qpos[2] += FOOT_SOLE_OFFSET - data.xpos[ids[4:], 2].min()
    mujoco.mj_forward(model, data)
    pose = np.column_stack((data.xpos[ids], data.xquat[ids][:, [1, 2, 3, 0]]))
    platform = None
    payload_width = 0.33
    if scenario == "walk":
        frames = _walk(pose)
        mass = 2.0
    elif scenario == "step-up":
        initial = pose.astype(np.float32)
        platform = _BASELINE_PLATFORM
        if video_step and rp_hands:
            platform = _RP_HAND_PLATFORM
            payload_width = 0.315
            frames = _rp_hand_step_up(mirror_sparse_poses(initial), platform)
        elif video_step:
            frames = _video_step_up(initial)
        else:
            frames = _nominal_step_up(initial)
        mass = 1.5
    else:
        raise ValueError(f"Unknown carrying scenario {scenario!r}.")
    return Reference(
        frames.astype(np.float32),
        qpos,
        len(frames) - 1,
        mass,
        (0.29, payload_width, 0.30),
        platform,
    )


def _upper_body(initial: np.ndarray, poses: np.ndarray, pitch: np.ndarray) -> None:
    """Place the torso and both wrists as a rigid carrying posture over the pelvis."""
    rotation_vectors = np.zeros((len(pitch), 3))
    rotation_vectors[:, 1] = pitch
    tilt = Rotation.from_rotvec(rotation_vectors)
    positions = tilt.as_matrix() @ (initial[:4, :3] - initial[0, :3]).T
    root = poses[:, 0, :3].copy()
    poses[:, :4, :3] = root[:, None] + positions.transpose(0, 2, 1)
    for body in range(4):
        poses[:, body, 3:] = (tilt * Rotation.from_quat(initial[body, 3:])).as_quat()


def _walk(initial: np.ndarray) -> np.ndarray:
    duration = 8.0
    t = np.arange(round(duration / CONTROL_DT) + 1) * CONTROL_DT
    poses = np.repeat(initial[None], len(t), axis=0)
    foothold = initial[4:, :3].copy()
    period = 0.52
    first = 0.36
    count = 14
    distance = 4.03
    stride = distance / (count - 1)

    # A small weight transfer precedes the first swing. The final two
    # footholds are aligned so the terminal pose does not require a recovery step.
    active = (t - first) / period
    phase = np.clip(active, 0, count)
    root_x = stride * np.maximum(phase - 0.5, 0)
    root_x = np.minimum(root_x, distance)
    lateral = -0.050 * np.sin(np.pi * (phase + 0.13))
    envelope = _move(t, 0, first) * (1 - _move(t, 7.30, duration))
    poses[:, 0, 0] += root_x
    poses[:, 0, 1] += envelope * lateral
    poses[:, 0, 2] -= envelope * (0.027 + 0.006 * np.cos(2 * np.pi * phase))
    _upper_body(initial, poses, envelope * 0.055)

    for i in range(count):
        leg = i % 2
        start = first + i * period
        swing = np.clip((t - start) / (period - 0.07), 0, 1)
        end_x = initial[4 + leg, 0] + min((i + 0.8) * stride, distance)
        if i >= count - 2:
            end_x = initial[4 + leg, 0] + distance
        poses[:, 4 + leg, 0] += (end_x - foothold[leg, 0]) * _smooth(swing)
        poses[:, 4 + leg, 2] += 0.065 * np.sin(np.pi * swing) ** 2
        foothold[leg, 0] = end_x
    return poses


def _nominal_step_up(initial: np.ndarray) -> np.ndarray:
    t = np.arange(round(8.0 / CONTROL_DT) + 1) * CONTROL_DT
    poses = np.repeat(initial[None], len(t), axis=0)
    foothold_x = STEP_FRONT + 0.2938
    swings = ((0.62, 1.60, 0.51, 0.26), (1.86, 2.74, 0.48, 0.22))
    for leg, (start, plant, lift_time, forward_delay) in enumerate(swings, 4):
        # Lift the foot above the face before translating over the edge.
        lift = _move(t, start, start + lift_time)
        forward = _move(t, start + forward_delay, plant - 0.1645)
        settle = _move(t, plant - 0.30, plant)
        clearance = lift - settle
        poses[:, leg, 0] += (foothold_x - initial[leg, 0]) * forward
        poses[:, leg, 2] += STEP_HEIGHT * lift + 0.025 * clearance
        toe_up = Rotation.from_rotvec(np.outer(-0.12 * clearance, [0, 1, 0]))
        poses[:, leg, 3:] = (toe_up * Rotation.from_quat(initial[leg, 3:])).as_quat()
        # The wider grip changes the balance during swing and transfer.
        poses[:, leg, 2] += 0.046 * (
            _move(t, start, start + 0.41) - _move(t, plant - 0.31, plant)
        )
    for axis in (0, 2):
        poses[:, 5, axis] = np.interp(t - 0.0429, t, poses[:, 5, axis])

    # Hold the first foothold before transferring weight onto the platform.
    transfer = _move(t, 1.62, 2.18)
    poses[:, 0, 0] += (
        0.05 * _move(t, 0.15, 1.60) + 0.405 * transfer + 0.035 * _move(t, 2.21, 2.84)
    )
    poses[:, 0, 1] += (
        -0.065 * _move(t, 0.08, 0.66) + 0.16 * transfer - 0.095 * _move(t, 2.68, 3.19)
    )
    poses[:, 0, 2] += (
        -0.075 * _move(t, 0.10, 0.74)
        + 0.31 * _move(t, 1.57, 2.18)
        + (STEP_HEIGHT - 0.235) * _move(t, 2.12, 3.02)
    )
    poses[:, 0, 0] += 0.0038 * _move(t, 1.62, 2.84)
    poses[:, 0, 1] += 0.0152 * (_move(t, 0.08, 0.66) - _move(t, 1.57, 2.35))
    pitch = 0.3785 * (_move(t, 0.27, 1.62) - _move(t, 2.21, 2.94))
    _upper_body(initial, poses, pitch)
    return poses


def _video_step_up(initial: np.ndarray) -> np.ndarray:
    # Lead with the right foot and turn the torso over the loaded leg.
    left_lead = mirror_sparse_poses(initial)
    poses = mirror_sparse_poses(_nominal_step_up(left_lead))
    t = np.arange(len(poses)) * CONTROL_DT
    yaw_pulse = _move(t, 1.67, 2.26) - _move(t, 2.80, 3.50)
    _turn_carrying_torso(poses, 0.40 * yaw_pulse)
    return poses


def _rp_hand_step_up(left_lead: np.ndarray, platform: StepPlatform) -> np.ndarray:
    """Clear the edge with the lead foot before transferring weight onto it."""
    t = np.arange(round(8.0 / CONTROL_DT) + 1) * CONTROL_DT
    poses = np.repeat(left_lead[None], len(t), axis=0)
    height = platform.size[2]
    lead_start, lead_plant = 0.7546, 1.4621
    trail_start, trail_plant = 2.2902, 3.1505
    lead_x = platform.front + 0.1104
    trail_x = platform.front + 0.1807
    swings = (
        (lead_start, lead_plant, 0.5199, 0.2219, 0.0605, lead_x),
        (trail_start, trail_plant, 0.34, 0.18, 0.1337, trail_x),
    )
    for leg, (start, plant, lift_time, forward_delay, clearance, x) in enumerate(
        swings, 4
    ):
        lift = _move(t, start, start + lift_time)
        forward = _move(t, start + forward_delay, plant - 0.035)
        swing = lift - _move(t, plant - 0.22, plant)
        poses[:, leg, 0] += (x - left_lead[leg, 0]) * forward
        poses[:, leg, 2] += height * lift + clearance * swing
        toe_up = Rotation.from_rotvec(np.outer(-0.10 * swing, [0, 1, 0]))
        poses[:, leg, 3:] = (toe_up * Rotation.from_quat(left_lead[leg, 3:])).as_quat()
    poses[:, 4, 1] -= 0.0525 * _move(t, lead_start, lead_plant)

    transfer_start, transfer_end = 1.5657, 2.3193
    transfer = _move(t, transfer_start, transfer_end)
    preload = _move(t, 0.12, lead_start + 0.06)
    straighten = _move(t, trail_plant - 0.02, trail_plant + 0.36)
    approach = 0.0635
    stance_x = lead_x - 0.04
    finish_x = 0.5 * (lead_x + trail_x) - 0.04
    poses[:, 0, 0] += (
        approach * _move(t, 0.13, lead_plant)
        + (stance_x - approach) * transfer
        + (finish_x - stance_x) * straighten
    )
    preload_shift, transfer_shift = 0.0875, 0.0859
    poses[:, 0, 1] += (
        -preload_shift * preload
        + (preload_shift + transfer_shift) * transfer
        - transfer_shift * _move(t, trail_plant - 0.10, trail_plant + 0.46)
    )
    # Rise gradually over the lead foot and straighten after the trailing foot clears.
    preload_rise, transfer_rise = 0.0065, 0.0855
    poses[:, 0, 2] += (
        preload_rise * preload
        + (height * transfer_rise - preload_rise)
        * _move(t, transfer_start - 0.10, transfer_end)
        + height * (1 - transfer_rise) * straighten
    )
    pitch = 0.131 * _move(t, 0.14, lead_start + 0.06) - 0.131 * _move(
        t, lead_plant, transfer_start + 0.18
    )
    pitch += 0.0388 * (
        _move(t, transfer_start - 0.15, transfer_end)
        - _move(t, trail_plant, trail_plant + 0.45)
    )
    _upper_body(left_lead, poses, pitch)
    poses = mirror_sparse_poses(poses)
    yaw = 0.7641 * (
        _move(t, lead_plant - 0.25, transfer_end)
        - _move(t, trail_plant - 0.08, trail_plant + 0.58)
    )
    _turn_carrying_torso(poses, yaw)
    return poses


def _turn_carrying_torso(poses: np.ndarray, yaw: np.ndarray) -> None:
    chest = Rotation.from_rotvec(np.outer(yaw, [0, 0, 1]))
    pivot = poses[:, 1, :3].copy()
    for body in (1, 2, 3):
        poses[:, body, :3] = pivot + chest.apply(poses[:, body, :3] - pivot)
        poses[:, body, 3:] = (chest * Rotation.from_quat(poses[:, body, 3:])).as_quat()
