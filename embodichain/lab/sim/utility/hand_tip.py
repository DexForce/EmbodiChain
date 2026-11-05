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

"""Named-FK adapters for declared authored-collision and native-mesh hand tips."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import math

import numpy as np
import torch

from .grasp_geometry import RobotHandGeometryCfg, _owned_cfg, _resolve_hand_frame
from .urdf_collision import load_urdf_collision_inputs

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["HandCollisionTipCalibration", "closed_native_hand_tip_offset"]


class HandCollisionTipCalibration:
    """Calibrate a declared hand from complete authored collision vertices.

    This single-environment adapter uses named FK without changing robot state.
    It deliberately does not use cooked colliders, surface samples or a
    precomputed aperture table. Each fit preserves the twelve-step closed-side
    bisection and TCP-X pad ordering of the migrated control path.

    Args:
        robot: Robot supplying named FK and the authored URDF.
        cfg: Explicit hand joints, pad links, open/closed commands and fixed TCP.
        qpos: Owned full-robot B=1 observation used for every FK evaluation.
        finger_link_names: Complete calibrated finger collider roster, including
            non-pad finger links. Its model owner supplies exact names; this
            adapter never infers fingers from prefixes or substrings.
    """

    def __init__(
        self,
        robot: Robot,
        cfg: RobotHandGeometryCfg,
        qpos: torch.Tensor,
        *,
        finger_link_names: Sequence[str],
    ) -> None:
        cfg = _owned_cfg(cfg)
        path, hand_links, hand_joints, solver = _resolve_hand_frame(robot, cfg)
        if (
            not isinstance(qpos, torch.Tensor)
            or not qpos.is_floating_point()
            or qpos.shape != (1, len(robot.joint_names))
            or not torch.isfinite(qpos).all()
        ):
            raise ValueError("Hand tip calibration requires a finite full B=1 qpos.")
        if isinstance(finger_link_names, str):
            raise ValueError("Finger colliders require an explicit link sequence.")
        names = tuple(finger_link_names)
        if (
            not names
            or any(type(name) is not str or not name for name in names)
            or len(set(names)) != len(names)
            or not set(cfg.pad_link_names) <= set(names)
            or not set(names) <= set(hand_links)
        ):
            raise ValueError("Finger collider roster must include both declared pads.")
        self._robot, self._cfg = robot, cfg
        self._end_link = solver.end_link_name
        self._tcp = tuple(tuple(row) for row in solver.tcp)
        self._qpos = qpos.detach().clone()
        self._names = names
        self._joint_names = tuple(robot.joint_names)
        self._joint_ids = tuple(self._joint_names.index(name) for name in hand_joints)
        self._commands = tuple(
            qpos.new_tensor(command).reshape(1, -1)
            for command in (cfg.open_qpos, cfg.closed_qpos)
        )
        meshes = load_urdf_collision_inputs(
            path, names, body_scale=robot.cfg.body_scale, vertices_only=True
        )
        self._vertices = {
            name: qpos.new_tensor(np.concatenate([mesh.vertices for mesh in values]))
            for name, values in meshes.items()
        }

    def _tcp_points(
        self, fraction: float, names: tuple[str, ...], *, closed: bool = False
    ) -> list[torch.Tensor]:
        qpos = self._qpos.clone()
        qpos[:, list(self._joint_ids)] = (
            self._commands[1]
            if closed
            else self._commands[0] + fraction * (self._commands[1] - self._commands[0])
        )
        poses = self._robot.compute_fk(
            qpos=qpos,
            link_names=[self._end_link, *names],
            qpos_joint_names=self._joint_names,
        )
        inverse = torch.linalg.inv(poses[0, 0] @ poses.new_tensor(self._tcp))
        result = []
        for index, name in enumerate(names, start=1):
            transform = inverse @ poses[0, index]
            vertices = self._vertices[name].to(transform)
            result.append(vertices @ transform[:3, :3].T + transform[:3, 3])
        return result

    def fit_opening(self, width: float) -> tuple[float, torch.Tensor]:
        """Return TCP-Z tip offset and B=1 hand command at the closed-side bound.

        Exactly fifteen FK evaluations are performed for an in-range opening:
        two endpoints, twelve bisections and the final tip measurement. Endpoint
        ``math.isclose`` snapping and float operation order remain unchanged.
        Width must be finite and positive; out-of-range gaps are rejected.

        Args:
            width: Requested TCP-X aperture in metres within the measured stroke.

        Returns:
            Leading TCP-Z offset and closed-side hand joint command.
        """
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("Opening width must be finite and positive.")

        def measure(fraction: float) -> tuple[float, float]:
            clouds = self._tcp_points(fraction, self._cfg.pad_link_names)
            clouds.sort(key=lambda points: float(points[:, 0].mean()))
            return float(clouds[1][:, 0].min() - clouds[0][:, 0].max()), max(
                float(points[:, 2].max()) for points in clouds
            )

        open_gap, closed_gap = measure(0.0)[0], measure(1.0)[0]
        if math.isclose(width, open_gap):
            width = open_gap
        elif math.isclose(width, closed_gap):
            width = closed_gap
        if (
            not math.isfinite(open_gap)
            or not math.isfinite(closed_gap)
            or open_gap <= closed_gap
            or not closed_gap <= width <= open_gap
        ):
            raise ValueError("Opening width is outside the measured hand aperture.")
        low, high = 0.0, 1.0
        for _ in range(12):
            mid = (low + high) / 2
            if measure(mid)[0] > width:
                low = mid
            else:
                high = mid
        offset = measure(high)[1]
        command = self._commands[0] + high * (self._commands[1] - self._commands[0])
        return offset, command

    def closed_tip_offset(self) -> float:
        """Measure the leading TCP-Z vertex of the complete closed finger roster.

        The migrated collision-tip qualification requires a finite offset in
        [0, 0.15] metres. This is not a contact or collision-free certificate.
        """
        offset = max(
            float(points[:, 2].max())
            for points in self._tcp_points(1.0, self._names, closed=True)
        )
        if not math.isfinite(offset) or not 0 <= offset <= 0.15:
            raise ValueError("Closed-finger collision tip calibration is unavailable.")
        return offset


def closed_native_hand_tip_offset(
    robot: Robot,
    cfg: RobotHandGeometryCfg,
    qpos: torch.Tensor,
    *,
    finger_link_names: Sequence[str],
) -> float:
    """Retain Press's native link-mesh closed-tip calibration in its declared TCP.

    The explicit complete finger roster replaces name inference. This B=1 adapter
    never substitutes authored collider vertices for the original native meshes.

    Args:
        robot: Existing Robot supplying named FK and link-mesh vertices.
        cfg: Explicit hand joints, commands and fixed TCP calibration.
        qpos: Finite full-robot B=1 reference posture; the input remains unchanged.
        finger_link_names: Complete explicit finger link roster from the model owner.

    Returns:
        Leading closed-finger TCP-Z offset in metres, in the original [0, 0.15] band.
    """
    cfg = _owned_cfg(cfg)
    _, hand_links, joints, solver = _resolve_hand_frame(robot, cfg)
    if isinstance(finger_link_names, str):
        raise ValueError("Finger geometry requires an explicit link sequence.")
    names = tuple(finger_link_names)
    if not names or len(set(names)) != len(names) or not set(names) <= set(hand_links):
        raise ValueError(
            "Native tip calibration requires an exact complete finger roster."
        )
    if (
        qpos.shape != (1, len(robot.joint_names))
        or not qpos.is_floating_point()
        or not torch.isfinite(qpos).all()
    ):
        raise ValueError("Native tip calibration requires finite full B=1 qpos.")
    closed = qpos.clone()
    closed[:, [robot.joint_names.index(name) for name in joints]] = closed.new_tensor(
        cfg.closed_qpos
    )
    poses = robot.compute_fk(
        qpos=closed,
        link_names=[solver.end_link_name, *names],
        qpos_joint_names=robot.joint_names,
    )
    inverse = torch.linalg.inv(poses[0, 0] @ poses.new_tensor(solver.tcp))
    leading = []
    for index, name in enumerate(names, start=1):
        vertices, _ = robot.get_link_vert_face(name)
        transform = inverse @ poses[0, index]
        points = vertices.to(transform) @ transform[:3, :3].T + transform[:3, 3]
        leading.append(points[:, 2].max())
    offset = float(torch.stack(leading).max())
    if not math.isfinite(offset) or not 0 <= offset <= 0.15:
        raise ValueError("Closed-finger native press-tip calibration is unavailable.")
    return offset
