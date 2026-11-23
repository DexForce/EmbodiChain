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
"""Measure the RJ45 deployment on the Gym contact sampling clock."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import numpy as np

from .rj45_measurements import evaluate_rj45, measure_rj45_frame

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = ["attach_rj45_monitor"]

_FINGERS = {"arena_0/franka/fr3_leftfinger", "arena_0/franka/fr3_rightfinger"}
_LATCH = "arena_0/plug/latch"
_SOCKET = "arena_0/socket/socket"
_LATCH_LOWER = -np.deg2rad(11)


class _Rj45Monitor:
    def __init__(
        self,
        env: EmbodiedEnv,
        config: dict[str, Any],
        anchor_baseline: np.ndarray,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("The Franka RJ45 deployment requires one arena.")
        self.config = config
        self.robot = env.robot
        self.plug = env.sim.get_articulation("plug")
        self.cable = env.sim.get_cable_object("cable")
        self.sensor = env.sim.get_sensor("contacts")
        self.lengths = np.linalg.norm(np.diff(self.cable.cfg.points, axis=0), axis=1)
        self.anchor_baseline = anchor_baseline
        self.control_steps = round(config["duration_s"] * config["control_hz"])
        self.actor_paths: dict[int, str] = {}
        self.rows: list[dict[str, Any]] = []
        self.active = True
        self.terminal_report: dict[str, Any] | None = None
        self.begin_control_step()
        self._capture()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or 0 in env_ids:
            self.active = False

    def begin_control_step(self) -> None:
        self.substep = 0
        self.finger_latch_contact = self.latch_socket_contact = False
        self.penetration = self.latch_limit_violation = 0.0
        self.dropped_contacts = 0

    def before_sample(self) -> None:
        return None

    def after_sample(self, dt: float, token: object) -> None:
        if not self.active or self.terminal_report is not None:
            return
        self.substep += 1
        if self.substep % (self.config["substeps"] // 4) == 0:
            self._sample_contacts()
        if self.substep == self.config["substeps"]:
            self._capture()

    def _sample_contacts(self) -> None:
        data = self.sensor.get_data()
        valid = data["is_valid"]
        for ids, separation in zip(
            data["user_ids"][valid].cpu().tolist(),
            data["distance"][valid].cpu().tolist(),
        ):
            self.penetration = max(self.penetration, -separation)
            if separation > 0:
                continue
            pair = set()
            for actor_id in ids:
                if actor_id >= 0:
                    if actor_id not in self.actor_paths:
                        actor = self.sensor.get_actor_info(actor_id)
                        self.actor_paths[actor_id] = (
                            f"{actor.path}/{actor.link_name or ''}"
                        )
                    pair.add(self.actor_paths[actor_id])
            if _LATCH in pair:
                self.finger_latch_contact |= bool(pair & _FINGERS)
                self.latch_socket_contact |= _SOCKET in pair
        angle = float(self.plug.get_qpos()[0, 0])
        self.latch_limit_violation = max(
            self.latch_limit_violation, angle, _LATCH_LOWER - angle
        )
        self.dropped_contacts += self.sensor.latest_sample_dropped_contacts

    def _capture(self) -> None:
        row = measure_rj45_frame(
            self.config,
            frame=len(self.rows),
            plug_pose=self.plug.get_link_pose("plug", to_matrix=True)[0].cpu().numpy(),
            tcp_pose=self.robot.get_link_pose("fr3_hand_tcp", to_matrix=True)[0]
            .cpu()
            .numpy(),
            robot_qpos=self.robot.get_qpos()[0].cpu().numpy(),
            latch_angle_rad=float(self.plug.get_qpos()[0, 0]),
            cable_states=self.cable.data.segment_state_w[0].cpu().numpy(),
            cable_lengths=self.lengths,
            anchor_baseline=self.anchor_baseline,
            finger_latch_contact=self.finger_latch_contact,
            latch_socket_contact=self.latch_socket_contact,
            latch_limit_violation_rad=float(self.latch_limit_violation),
            penetration_m=self.penetration,
            dropped_contacts=self.dropped_contacts,
        )
        self.rows.append(row)
        if not row["finite"]:
            raise RuntimeError(f"Nonfinite RJ45 physical state at {row['time_s']} s.")
        if len(self.rows) == self.control_steps + 1:
            self.terminal_report = evaluate_rj45(self.rows, self.config)


def attach_rj45_monitor(
    env: EmbodiedEnv, config: dict[str, Any], *, anchor_baseline: np.ndarray
) -> None:
    """Capture frame zero at the settled state and observe subsequent control ticks."""
    previous = getattr(env, "_rj45_monitor", None)
    if previous is not None:
        previous.sensor.remove_sampling_observer("franka_rj45_cycle", expected=previous)
    monitor = _Rj45Monitor(env, config, anchor_baseline)
    monitor.sensor.add_sampling_observer("franka_rj45_cycle", monitor)
    env._rj45_monitor = monitor
