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
"""Observe the Franka wrap using the Gym-owned contact sampling clock."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .wrap_measurements import _capsule_pad_gap, _rope_geometry, evaluate_wrap
from .wrap_motion import PHASE_END_S

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from .wrap_motion import WrapMotion

__all__ = ["attach_wrap_monitor"]


class _WrapMonitor:
    def __init__(
        self,
        env: EmbodiedEnv,
        config: dict[str, Any],
        motion: WrapMotion,
        robot_path: Path,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("The Franka rope wrap deployment requires one arena.")
        self.config = config
        self.robot = env.robot
        self.cable = env.sim.get_cable_object("cable")
        self.sensor = env.sim.get_sensor("contacts")
        self.motion = motion
        self.lengths = np.linalg.norm(np.diff(self.cable.cfg.points, axis=0), axis=1)
        material_arc = self.lengths.sum() - config["grasp_inset_m"]
        self.material_index = int(
            np.searchsorted(np.cumsum(self.lengths), material_arc)
        )
        self.material_offset = (
            material_arc
            - self.lengths[: self.material_index].sum()
            - self.lengths[self.material_index] / 2
        )
        tree = ET.parse(robot_path)
        limits = [
            tree.find(f".//joint[@name='{name}']/limit")
            for name in self.robot.joint_names
        ]
        self.lo = np.array([float(limit.get("lower")) for limit in limits])
        self.hi = np.array([float(limit.get("upper")) for limit in limits])
        self.actor_paths = {}
        self.active = True
        self.rows = []
        self.baseline = self.cable.data.segment_state_w[0, 0, :3].cpu().numpy().copy()
        self.material_baseline = None
        self.terminal_report = None
        self.begin_control_step()
        self._capture()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or 0 in env_ids:
            self.active = False

    def begin_control_step(self) -> None:
        self.substep = 0
        self.finger_contacts = [False, False]
        self.bilateral_contact = False
        self.robot_post_peak = self.penetration = 0.0
        self.dropped = 0
        self.worst = None

    def before_sample(self) -> None:
        return None

    def after_sample(self, dt: float, token: object) -> None:
        if not self.active or self.terminal_report is not None:
            return
        self.substep += 1
        if self.substep % (self.config["substeps"] // 4) == 0:
            self._contacts()
        if self.substep == self.config["substeps"]:
            self._capture()

    def _contacts(self) -> None:
        contact = self.sensor
        robot, cable = self.robot, self.cable
        config, lengths = self.config, self.lengths
        actor_paths = self.actor_paths
        data = contact.get_data()
        valid = data["is_valid"]
        candidate_positions = [[], []]
        for ids, separation, position in zip(
            data["user_ids"][valid].cpu().tolist(),
            data["distance"][valid].cpu().tolist(),
            data["position"][valid].cpu().tolist(),
        ):
            pair = []
            for actor_id in ids:
                if actor_id < 0:
                    continue
                if actor_id not in actor_paths:
                    info = contact.get_actor_info(actor_id)
                    actor_paths[actor_id] = f"{info.path}/{info.link_name or ''}"
                pair.append(actor_paths[actor_id])
            # DexSim currently registers rigid/articulation actors,
            # leaving cable counterparts anonymous. Unknown identity
            # is only a candidate: match the live cable capsules and
            # finger gap below before accepting contact evidence.
            is_rope = any(actor_id < 0 for actor_id in ids) or any(
                "/cable/" in p for p in pair
            )
            for k, name in enumerate(("fr3_leftfinger", "fr3_rightfinger")):
                if is_rope and any(name in p for p in pair) and separation <= 1e-5:
                    candidate_positions[k].append(position)
            if any("/post/" in p for p in pair) and any("/franka/" in p for p in pair):
                self.robot_post_peak = max(self.robot_post_peak, -separation)
            if -separation > self.penetration:
                self.penetration = -separation
                self.worst = pair
        matched = [False, False]
        if any(candidate_positions):
            sampled_state = cable.data.segment_state_w[0, -8:].cpu().numpy()
            axes = Rotation.from_quat(sampled_state[:, 3:7]).as_matrix()[:, :, 2]
            starts = sampled_state[:, :3] - axes * lengths[-8:, None] / 2
            delta = axes * lengths[-8:, None]
            for k, name in enumerate(("fr3_leftfinger", "fr3_rightfinger")):
                if not candidate_positions[k]:
                    continue
                contact_points = np.asarray(candidate_positions[k])
                fraction = np.clip(
                    np.sum(
                        (contact_points[:, None] - starts) * delta,
                        axis=-1,
                    )
                    / np.sum(delta * delta, axis=-1),
                    0.0,
                    1.0,
                )
                closest = starts + fraction[..., None] * delta
                point_gap = (
                    np.linalg.norm(contact_points[:, None] - closest, axis=-1).min()
                    - config["cable_radius_m"]
                )
                pose = robot.get_link_pose(name, to_matrix=True)[0].cpu().numpy()
                pad_gap = _capsule_pad_gap(sampled_state, lengths[-8:], pose, config)
                matched[k] = bool(point_gap <= 0.0003 and pad_gap <= 0.0002)
                self.finger_contacts[k] |= matched[k]
        self.bilateral_contact |= all(matched)
        self.dropped += contact.latest_sample_dropped_contacts

    def _capture(self) -> None:
        config, lengths = self.config, self.lengths
        robot, cable = self.robot, self.cable
        frame = len(self.rows)
        t = frame / config["control_hz"]
        targets, labels = self.motion.poses, self.motion.labels
        material_index, material_offset = self.material_index, self.material_offset
        lo, hi, baseline = self.lo, self.hi, self.baseline
        finger_contacts = self.finger_contacts
        state = cable.data.segment_state_w[0].cpu().numpy()
        qpos = robot.get_qpos()[0].cpu().numpy()
        tcp = robot.get_link_pose("fr3_hand_tcp", to_matrix=True)[0].cpu().numpy()
        command_tcp = targets[frame]
        label = labels[frame]
        scale = config["duration_s"] / PHASE_END_S["hold_closed"]
        if PHASE_END_S["home"] * scale < t <= PHASE_END_S["approach_post"] * scale:
            command_tcp = (
                robot.compute_fk(
                    qpos=robot.get_qpos(name="arm", target=True),
                    name="arm",
                    to_matrix=True,
                )[0]
                .cpu()
                .numpy()
            )
            if t <= config["lift_complete_s"]:
                label = "CABLE PICKUP"
        pad_gaps = []
        for name in ("fr3_leftfinger", "fr3_rightfinger"):
            pose = robot.get_link_pose(name, to_matrix=True)[0].cpu().numpy()
            pad_gaps.append(_capsule_pad_gap(state[-8:], lengths[-8:], pose, config))
        finger_contacts = [
            present and gap <= 0.0002 for present, gap in zip(finger_contacts, pad_gaps)
        ]
        material_axis = Rotation.from_quat(state[material_index, 3:7]).as_matrix()[:, 2]
        material_world = state[material_index, :3] + material_offset * material_axis
        material_local = tcp[:3, :3].T @ (material_world - tcp[:3, 3])
        if (
            self.material_baseline is None
            and t >= config["grasp_complete_s"]
            and self.bilateral_contact
        ):
            self.material_baseline = material_local.copy()
        material_slip = (
            float(np.linalg.norm(material_local - self.material_baseline))
            if self.material_baseline is not None
            else 0.0
        )
        geometry = _rope_geometry(state, lengths, config)
        nodes = geometry.pop("nodes")
        row = {
            "time_s": t,
            "phase": label,
            **geometry,
            "actual_tcp_pose": tcp.tolist(),
            "command_tcp_pose": command_tcp.tolist(),
            "tcp_error_m": float(np.linalg.norm(tcp[:3, 3] - command_tcp[:3, 3])),
            "tail_tcp_distance_m": float(np.linalg.norm(nodes[-1] - tcp[:3, 3])),
            "grasp_region_tcp_distance_m": float(
                np.linalg.norm(state[-6:, :3] - tcp[:3, 3], axis=1).min()
            ),
            "grasp_confirmed": self.material_baseline is not None,
            "bilateral_rope_contact": self.bilateral_contact,
            "material_point_tcp_m": material_local.tolist(),
            "material_slip_m": material_slip,
            "finger_pad_gaps_m": pad_gaps,
            "left_rope_contact": finger_contacts[0],
            "right_rope_contact": finger_contacts[1],
            "finger_gap_m": float(qpos[7] + qpos[8]),
            "anchor_error_m": float(np.linalg.norm(state[0, :3] - baseline)),
            "arm_joint_limit_violation_rad": float(
                max(0.0, (lo[:7] - qpos[:7]).max(), (qpos[:7] - hi[:7]).max())
            ),
            "finger_joint_limit_violation_m": float(
                max(0.0, (lo[7:] - qpos[7:]).max(), (qpos[7:] - hi[7:]).max())
            ),
            "robot_post_penetration_m": self.robot_post_peak,
            "penetration_m": self.penetration,
            "worst_contact": self.worst,
            "dropped_contacts": self.dropped,
            "finite": bool(np.isfinite(state).all() and np.isfinite(qpos).all()),
        }
        self.rows.append(row)
        if not row["finite"]:
            raise RuntimeError(f"Nonfinite physical state at {t}")

        if len(self.rows) == len(self.motion.times):
            self.terminal_report = evaluate_wrap(self.rows, self.config)


def attach_wrap_monitor(
    env: EmbodiedEnv, config: dict[str, Any], motion: WrapMotion, robot_path: Path
) -> _WrapMonitor:
    """Start the episode at the settled state before its first command."""
    previous = getattr(env, "_wrap_monitor", None)
    if previous is not None:
        if previous.active:
            return previous
        previous.sensor.remove_sampling_observer("franka_rope_wrap", expected=previous)
    monitor = _WrapMonitor(env, config, motion, robot_path)
    monitor.sensor.add_sampling_observer("franka_rope_wrap", monitor)
    env._wrap_monitor = monitor
    return monitor
