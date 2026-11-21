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

"""Measured Cartesian motions for the single-arm Tool Hang demonstration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from embodichain.lab.gym.envs.types import ControllerAction

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__: list[str] = []

_UP = np.array([0.0, 0.0, 1.0])
_GRIP = np.array([0.04375, 0.0, 0.068])
_TIP = np.array([0.04375, 0.0, -0.13445])


def _limit(vector: np.ndarray, maximum: float) -> np.ndarray:
    return vector * min(1.0, maximum / max(np.linalg.norm(vector), 1e-12))


class _UnreachableWaypoint(RuntimeError):
    """No joint seed solved a requested Cartesian target."""


class ToolHangMotion:
    """Plan from measured actor and hand poses, emitting native controller actions."""

    def __init__(self, env: EmbodiedEnv) -> None:
        if env.num_envs != 1:
            raise ValueError("The Tool Hang demonstration requires one arena.")
        self.env = env
        self.robot = env.robot
        self.grip_effort = 12.0
        self.position_bias = np.zeros(3)
        self.rotation_bias = np.eye(3)

    def hand(self) -> np.ndarray:
        return self.robot.get_link_pose("fr3_hand_tcp", to_matrix=True)[0].cpu().numpy()

    def pose(self, name: str) -> np.ndarray:
        return (
            self.env.sim.get_articulation(name)
            .get_local_pose(to_matrix=True)[0]
            .cpu()
            .numpy()
        )

    def point(self, name: str, offset: np.ndarray) -> np.ndarray:
        pose = self.pose(name)
        return pose[:3, 3] + pose[:3, :3] @ offset

    def _solve(
        self, xyz: np.ndarray, rotation: np.ndarray, seed: torch.Tensor
    ) -> torch.Tensor:
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, xyz
        goal = torch.as_tensor(pose[None], device=self.env.device, dtype=seed.dtype)
        ok, qpos = self.robot.compute_ik(goal, joint_seed=seed, name="arm")
        if ok.item():
            return qpos.reshape(1, 7)

        # The horizontal insertion can leave joint 7 on the opposite IK branch
        # from the next top-down pinch. Try redundant wrist seeds before moving.
        home = seed.new_tensor(self.robot.cfg.init_qpos[:7]).reshape_as(seed)
        candidates = []
        for wrist_offset in (0.0, -1.5, 1.5, -2.5):
            candidate = home.clone()
            candidate[:, 6] += wrist_offset
            ok, solution = self.robot.compute_ik(goal, joint_seed=candidate, name="arm")
            if ok.item():
                solution = solution.reshape_as(seed)
                candidates.append(((solution - seed).abs().amax().item(), solution))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
        raise _UnreachableWaypoint(
            f"Panda could not reach the Tool Hang waypoint at {xyz.tolist()}."
        )

    def _command(
        self, qpos: torch.Tensor, jaw: float, qvel: torch.Tensor | None = None
    ) -> ControllerAction:
        qpos[:, 7:] = jaw
        effort = torch.zeros_like(qpos)
        if jaw < 0.014:
            effort[:, 7:] = -self.grip_effort
        command = {"qpos": qpos, "qf": effort}
        if qvel is not None:
            command["qvel"] = qvel
        return ControllerAction(TensorDict(command, batch_size=(1,)))

    def hold(
        self, jaw: float, count: int, *, target: bool = False
    ) -> Iterator[ControllerAction]:
        qpos = self.robot.get_qpos(target=target).clone()
        for _ in range(count):
            yield self._command(qpos, jaw)

    def move(
        self,
        xyz: np.ndarray,
        rotation: np.ndarray,
        jaw: float,
        *,
        rate: float = 1.0,
        steps: int | None = None,
        settle: int = 5,
    ) -> Iterator[ControllerAction]:
        """Interpolate an IK target, then measure the residual for the next waypoint."""
        start = self.robot.get_qpos().clone()
        xyz, rotation = xyz + self.position_bias, self.rotation_bias @ rotation
        delta = self._solve(xyz, rotation, start[:, :7]) - start[:, :7]
        duration = steps or max(
            5, int(1.7 * delta.abs().max().item() / (rate * self.env.step_dt))
        )
        for i in range(1, duration + settle + 1):
            t = min(i / duration, 1.0)
            qpos = start.clone()
            qpos[:, :7] += (3 - 2 * t) * t**2 * delta
            qvel = torch.zeros_like(qpos)
            qvel[:, :7] = (6 * t - 6 * t**2) / (duration * self.env.step_dt) * delta
            yield self._command(qpos, jaw, qvel)
        result = self.hand()
        self.position_bias = _limit(xyz - result[:3, 3], 0.015)
        residual = Rotation.from_matrix(rotation @ result[:3, :3].T).as_rotvec()
        self.rotation_bias = Rotation.from_rotvec(_limit(residual, 0.07)).as_matrix()

    def _grasp_frame(self) -> Iterator[ControllerAction]:
        yield from self.hold(0.04, 4)
        closing = -self.pose("frame")[:3, 0].copy()
        closing[2] = 0.0
        closing /= np.linalg.norm(closing)
        top_down = np.column_stack((np.cross(closing, -_UP), closing, -_UP))
        yield from self.move(self.point("frame", _GRIP) + 0.07 * _UP, top_down, 0.04)
        yield from self.move(
            self.point("frame", _GRIP) - 0.003 * _UP, top_down, 0.04, rate=0.45
        )
        yield from self.hold(0.0, 16)
        start = self.hand()[:3, 3].copy()
        for rise in (0.018, 0.060, 0.14):
            yield from self.move(start + rise * _UP, top_down, 0.0, rate=0.55)
            if self.point("frame", _GRIP)[2] < start[2] + 0.5 * rise:
                raise RuntimeError("The frame did not rise with the fingers.")

    def lift_frame(self) -> Iterator[ControllerAction]:
        """Raise the frame, turn its rod down and carry it to the socket."""
        yield from self._grasp_frame()
        hand = self.hand()
        elevated = hand[:3, 3].copy()
        elevated[2] = 1.08
        yield from self.move(elevated, hand[:3, :3], 0.0, rate=0.7)
        hand = self.hand()
        turn_at = hand[:3, 3] + np.array([-0.02, 0.04, 0.03])
        for angle in (35, 70, 88):
            orientation = (
                hand[:3, :3]
                @ Rotation.from_euler("y", -angle, degrees=True).as_matrix()
            )
            yield from self.move(turn_at, orientation, 0.0, rate=0.8)
        yield from self.hold(0.0, 30, target=True)
        socket = self.point("stand", np.array([0, 0.045, 0]))
        yield from self.move(
            socket + [-0.02, -0.045, 0.255], self.hand()[:3, :3], 0.0, rate=0.7
        )

    def _center_frame_site(
        self,
        site: np.ndarray,
        target: np.ndarray,
        count: int,
        speed: float,
        tolerance: float,
    ) -> Iterator[ControllerAction]:
        hand = self.hand()
        rotation = self.rotation_bias @ hand[:3, :3]
        goal = hand[:3, 3] + self.position_bias
        joints = self.robot.get_qpos().clone()
        for i in range(count):
            position = self.point("frame", site)
            err = target - position
            if np.linalg.norm(err) < tolerance and i >= 2:
                break
            hand = self.hand()
            if np.linalg.norm(position - hand[:3, 3]) > 0.24:
                raise RuntimeError("The Panda lost the frame while centering it.")
            goal += _limit(3 * err, speed) * self.env.step_dt
            goal = hand[:3, 3] + _limit(goal - hand[:3, 3], 0.033)
            joints[:, :7] = self._solve(goal, rotation, joints[:, :7])
            yield self._command(joints, 0.0)

    def seat_frame(self) -> Iterator[ControllerAction]:
        """Center the measured rod tip between the walls and release the seated frame."""
        socket = self.point("stand", np.array([0, 0.045, 0]))
        for height in (
            0.113,
            0.084,
            0.076,
            0.065,
            0.048,
            0.032,
            0.007,
            -0.022,
            -0.048,
            -0.066,
        ):
            yield from self._center_frame_site(
                _TIP, socket + height * _UP, 55, 0.13, 0.0019
            )
            if height < 0.07:
                grip_target = socket.copy()
                grip_target[2] = self.point("frame", _GRIP)[2]
                yield from self._center_frame_site(_GRIP, grip_target, 13, 0.09, 0.002)
        hand = self.hand()
        yield from self.move(hand[:3, 3], hand[:3, :3], 0.04, steps=15, settle=7)
        withdrawal = -0.09 * hand[:3, 2] + 0.06 * _UP
        try:
            yield from self.move(
                hand[:3, 3] + withdrawal, hand[:3, :3], 0.04, rate=0.65, settle=20
            )
        except _UnreachableWaypoint:
            # A redundant wrist can finish insertion near a joint limit. Half
            # the unobstructed withdrawal still clears the fully open pads.
            yield from self.move(
                hand[:3, 3] + 0.5 * withdrawal,
                hand[:3, :3],
                0.04,
                rate=0.65,
                settle=20,
            )

        # Clear the installed crossbar before changing from the horizontal
        # insertion grasp to the top-down wrench grasp. Interpolating those
        # wrist poses directly can sweep the open hand through the stand.
        hand = self.hand()
        lift = hand[:3, 3].copy()
        lift[2] = self.point("frame", _GRIP)[2] + 0.14
        yield from self.move(lift, hand[:3, :3], 0.04, rate=0.75)
        # Cross at that height on the wrench side of the stand, then descend.
        wrench = self.pose("tool")[:3, 3]
        across = lift.copy()
        across[0] = socket[0]
        across[1] = 0.35 * socket[1] + 0.65 * wrench[1]
        yield from self.move(across, hand[:3, :3], 0.04, rate=0.75, settle=8)
