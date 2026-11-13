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
"""Shared S03 controller and immutable input loading; no simulator imports."""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np

__all__ = ["CornerLiftController", "read_inputs"]


def read_inputs(args, root: Path) -> dict:
    """Load authored cloth nodes, material corner ids and the common IK path.

    Args:
        args: Worker command-line options.
        root: Input snapshot directory.

    Returns:
        Shared scene/control inputs preserving physical node order.
    """
    config = json.loads((root / "s03_config.json").read_text())
    for item in args.set or []:
        key, value = item.split("=", 1)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = json.loads(value)
    # Keep the recorded config immutable while resolving relocated snapshots.
    robot_path = Path(config["robot"]["urdf"])
    local_robot = root / "franka_emika_panda" / robot_path.name
    protocol_path = root.parent.parent / "protocol.json"
    if not robot_path.is_file() and local_robot.is_file() and protocol_path.is_file():
        identity = json.loads(protocol_path.read_text())["inputs_sha256"]
        key = str(local_robot.relative_to(root.parent))
        expected = identity.get(key)
        if (
            expected is None
            or hashlib.sha256(local_robot.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Relocated S03 robot does not match its frozen asset hash")
        config["robot"]["urdf"] = str(local_robot.resolve())
    mesh = np.load(root / "s03_cloth_mesh.npz")
    ik = np.load(root / "s03_ik_path.npz")
    visual = config["cloth"].get("visual", {})
    uv = None
    texture = None
    if visual:
        xy = mesh["vertices"][:, :2]
        uv = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0)
        uv[:, 1] = 1.0 - uv[:, 1]
        uv *= visual.get("uv_repeat", 2.0)
        texture = root / visual["texture_file"]
        if not texture.is_file():
            raise FileNotFoundError(texture)
    if args.state != "s00" or args.variant != "nominal":
        raise ValueError("S03 has one frozen state and variant")
    return {
        "config": config,
        "initial": mesh["vertices"],
        "surface": mesh["triangles"],
        "tets": np.empty((0, 4), np.int32),
        "patches": mesh["corner_ids"],
        "corner_ids": mesh["corner_ids"],
        "cloth_uv": uv,
        "cloth_texture": None if texture is None else str(texture),
        "state": {"id": "s00", "block_center": config["cloth"]["center"]},
        "ik_s": ik["s"],
        "ik_q": ik["arm_q"],
        "waypoint_s": ik["waypoint_s"],
    }


class CornerLiftController:
    """Fixed physical finger closure and timed lift/hold/translate/release."""

    PHASES = (
        "settle",
        "close",
        "lift",
        "hold",
        "transport",
        "lower",
        "release",
        "retreat",
        "observe",
        "done",
    )
    MOVES = {
        "lift": (0, 1, "lift_duration"),
        "transport": (1, 2, "transport_duration"),
        "lower": (2, 3, "lower_duration"),
        "retreat": (3, 4, "retreat_duration"),
    }

    def __init__(self, config: dict, waypoint_s: np.ndarray) -> None:
        self.config = config
        self.waypoint_s = waypoint_s
        initialization = config.get("preview_initialization", {})
        self.standoff = float(initialization.get("gripper_standoff_m", 0.0))
        self.free_settle = float(initialization.get("free_settle_duration_s", 0.0))
        if not 0.0 <= self.free_settle < config["trajectory"]["settle_duration"]:
            raise ValueError(
                "Free settling must leave time for the open-gripper approach"
            )
        self.s = self.standoff
        self.finger = config["robot"]["finger_initial_q"]
        self.finger_closed = config["grip"]["finger_closed"]
        self.phase = "settle"
        self.phase_start = 0.0
        self.events = {"settle": 0.0}
        self.reference_height = None

    def _duration(self, phase: str) -> float:
        if phase in self.MOVES:
            return self.config["trajectory"][self.MOVES[phase][2]]
        if phase == "close":
            return self.config["grip"]["close_duration"]
        if phase == "release":
            return self.config["grip"]["release_duration"]
        return self.config["trajectory"][phase + "_duration"]

    def step(self, now: float, height: float) -> tuple[float, float, str]:
        if self.finished:
            return self.s, self.finger, self.phase
        if self.phase == "settle":
            self.reference_height = height
            approach = np.clip(
                (now - self.free_settle)
                / (self.config["trajectory"]["settle_duration"] - self.free_settle),
                0.0,
                1.0,
            )
            self.s = float(
                self.standoff
                * (1.0 - approach**3 * (10.0 + approach * (-15.0 + 6.0 * approach)))
            )
        duration = self._duration(self.phase)
        u = np.clip((now - self.phase_start) / duration, 0.0, 1.0)
        v = float(u**3 * (10 + u * (-15 + 6 * u)))
        if self.phase in self.MOVES:
            a, b, _ = self.MOVES[self.phase]
            self.s = float((1 - v) * self.waypoint_s[a] + v * self.waypoint_s[b])
        if self.phase == "close":
            close_motion = self.config["grip"].get("close_motion_duration", duration)
            close_u = float(np.clip((now - self.phase_start) / close_motion, 0.0, 1.0))
            v = close_u**3 * (10.0 + close_u * (-15.0 + 6.0 * close_u))
            self.finger = float(
                (1 - v) * self.config["robot"]["finger_initial_q"]
                + v * self.finger_closed
            )
        if self.phase == "release":
            self.finger = float(
                (1 - v) * self.finger_closed + v * self.config["grip"]["finger_open"]
            )
        if now - self.phase_start >= duration - 1e-9:
            self.phase = self.PHASES[self.PHASES.index(self.phase) + 1]
            self.phase_start = now
            self.events[self.phase] = now
        return self.s, self.finger, self.phase

    def compression(self) -> None:
        return None

    @property
    def finished(self) -> bool:
        return self.phase == "done"

    def max_frames(self) -> int:
        return (
            int(
                sum(self._duration(p) for p in self.PHASES[:-1])
                * self.config["trajectory"]["frame_rate"]
            )
            + 20
        )
