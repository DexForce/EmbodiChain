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

"""Closed-loop, contact-mediated DROID pick-and-place demonstration segments."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from embodichain.lab.gym.envs import DemoSegment
from embodichain.utils.math import pose_inv

if TYPE_CHECKING:
    from embodichain.extensions.tasks.manipulation.polaris._env import PolarisDroidEnv

__all__: list[str] = []


@dataclass(frozen=True)
class _Grasp:
    """Define a physical grasp and release for one named object.

    XY offsets are metres from the current collision AABB centre, rotated by
    the object's or target's yaw. The pad midpoint aims ``grasp_z`` metres
    above the object bottom. ``yaw`` is the closing-axis heading in degrees
    relative to the object's yaw. ``place_yaw_offset`` adds degrees during
    transport. ``surface_z`` sets release clearance above the target bottom;
    its default uses the top. ``scrub_distance`` is travel along arena X in
    metres. ``linear`` subdivides descent into Cartesian waypoints.
    """

    object: str
    target: str
    grasp_xy: Sequence[float]
    grasp_z: float
    yaw: float
    place_xy: Sequence[float] = (0, 0)
    surface_z: float | None = None
    scrub_distance: float = 0.0
    place_yaw_offset: float = 0.0
    linear: bool = False


class DroidExpert:
    """Plan native joint targets from measured gripper and object geometry."""

    _instructions = {
        "approach": "Align the open gripper above the {item}.",
        "descend": "Lower the finger pads around the {item}.",
        "close": "Close the gripper on the {item}.",
        "lift": "Lift the {item} clear of the table.",
        "transport": "Carry the {item} above the {target}.",
        "lower": "Lower the {item} to its release height at the {target}.",
        "scrub_out": "Move the {item} across the {target}.",
        "scrub_back": "Move the {item} back across the {target}.",
        "release": "Open the gripper to release the {item} at the {target}.",
        "withdraw": "Withdraw the open gripper above the {target}.",
    }

    def __init__(self, env: PolarisDroidEnv) -> None:
        self.env = env
        self.robot = env.robot
        self.arm = self.robot.get_joint_ids("arm", remove_mimic=True)
        rows = (env.cfg.extensions or {}).get("expert")
        if not rows:
            raise ValueError(
                f"The {env.scene} deployment needs an env.extensions.expert placement list."
            )
        self.placements = [_Grasp(**row) for row in rows]
        self.clearance = {name: height for name, _, height in env.rubric.placements}
        if sorted(row.object for row in self.placements) != sorted(self.clearance):
            raise ValueError(
                f"Expert placements must cover each rubric object once: {list(self.clearance)}."
            )
        for row in self.placements:
            if row.target not in env._objects:
                raise ValueError(
                    f"Unknown expert target {row.target!r}. Available objects: {list(env._objects)}."
                )
            if len(row.grasp_xy) != 2 or len(row.place_xy) != 2:
                raise ValueError(
                    f"Expert XY offsets for {row.object!r} must have two components."
                )

    def _hand(self) -> torch.Tensor:
        """Return the measured pad midpoint and the current closing and approach axes."""
        left = self.robot.get_link_pose("left_inner_finger_pad")[:, :3]
        right = self.robot.get_link_pose("right_inner_finger_pad")[:, :3]
        base = self.robot.get_link_pose("robotiq_arg2f_base_link", to_matrix=True)
        x = torch.nn.functional.normalize(right - left)
        z = base[:, :3, 2]
        y = torch.nn.functional.normalize(torch.linalg.cross(z, x))
        mat = base.clone()
        mat[:, :3, :3] = torch.stack((x, y, torch.linalg.cross(x, y)), dim=-1)
        mat[:, :3, 3] = 0.5 * (left + right)
        return mat

    def _bounds(self, name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obj = self.env._objects[name]
        pose = obj.get_local_pose(to_matrix=True)
        vertices = self.env._vertices[name] @ pose[:, :3, :3].transpose(1, 2)
        vertices += pose[:, None, :3, 3]
        return (
            vertices.amin(1),
            vertices.amax(1),
            torch.atan2(pose[:, 1, 0], pose[:, 0, 0]),
        )

    @staticmethod
    def _xy(
        low: torch.Tensor,
        high: torch.Tensor,
        yaw: torch.Tensor,
        offset: Sequence[float],
    ) -> torch.Tensor:
        local = low.new_tensor(offset)
        dx = local[0] * yaw.cos() - local[1] * yaw.sin()
        dy = local[0] * yaw.sin() + local[1] * yaw.cos()
        return 0.5 * (low + high)[:, :2] + torch.stack((dx, dy), dim=1)

    def _motion(
        self,
        point: torch.Tensor,
        yaw: torch.Tensor,
        grip: float,
        *,
        speed: float = 0.010,
        dwell: int = 3,
        linear: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Yield absolute-arm and normalized-gripper actions for a measured pad target.

        Args:
            point: Arena-frame pad-midpoint positions ``(N, 3)``, in metres.
            yaw: World closing-axis headings ``(N,)``, in radians.
            grip: Normalized gripper command, -1 open or +1 closed.
            speed: Maximum nominal translation (metres per control tick).
            dwell: Extra ticks to hold the final target.
            linear: Re-solve IK along a straight Cartesian approach.
        """
        hand = self._hand()
        arm = self.robot.get_qpos()[:, self.arm]
        fk = self.robot.compute_fk(arm, name="arm", to_matrix=True)
        hand_from_tcp = pose_inv(hand) @ fk
        x = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), dim=-1)
        x = torch.where((x * hand[:, :3, 0]).sum(1, keepdim=True) < 0, -x, x)
        z = torch.zeros_like(x)
        z[:, 2] = -1
        target = hand.clone()
        target[:, :3, :3] = torch.stack((x, torch.linalg.cross(z, x), z), dim=-1)
        target[:, :3, 3] = point
        distance = float((hand[:, :3, 3] - point).norm(dim=1).max())
        count = max(1, math.ceil(distance / 0.015)) if linear else 1
        for index in range(1, count + 1):
            waypoint = target.clone()
            waypoint[:, :3, 3] = hand[:, :3, 3].lerp(point, index / count)
            arm = self.robot.get_qpos()[:, self.arm].clone()
            goal = arm.clone()
            solved = torch.zeros(len(arm), device=arm.device, dtype=torch.bool)
            for elbow in (0.0, 0.8, -0.8, 1.5, -1.5):
                seed = arm.clone()
                seed[:, 2] += elbow
                seed[:, 4] -= elbow
                result = self.robot.compute_ik(
                    waypoint @ hand_from_tcp, joint_seed=seed, name="arm"
                )
                if result is None:
                    raise RuntimeError(
                        "PolaRiS requires a native IK solver on the arm control part."
                    )
                valid, candidate = result
                take = valid.reshape(-1) & ~solved
                goal[take] = candidate.reshape_as(arm)[take]
                solved |= take
                if solved.all():
                    break
            if not solved.all():
                raise RuntimeError(
                    f"PolaRiS IK failed for pad target {waypoint[:,:3,3].tolist()}."
                )
            steps = max(
                2,
                math.ceil(distance / (count * speed)),
                math.ceil(float((goal - arm).abs().max()) / 0.095),
            )
            for step in range(1, steps + (dwell if index == count else 0) + 1):
                fraction = min(1.0, step / steps)
                # Smoothstep starts and ends this joint interpolation at rest.
                joints = arm.lerp(goal, fraction**2 * (3 - 2 * fraction))
                yield torch.cat(
                    (joints, joints.new_full((len(joints), 1), grip)), dim=1
                )

    def _hold(self, grip: float, steps: int) -> Iterator[torch.Tensor]:
        joints = self.robot.get_qpos()[:, self.arm].clone()
        action = torch.cat((joints, joints.new_full((len(joints), 1), grip)), dim=1)
        for _ in range(steps):
            yield action

    def segments(self) -> Iterator[DemoSegment]:
        """Plan each phase against the state left by the previous phase."""
        for grasp in self.placements:
            name, target = grasp.object, grasp.target

            def phase(
                label: str,
                actions: Iterator[torch.Tensor],
                validator: Callable[[], torch.Tensor] | None = None,
            ) -> DemoSegment:
                return DemoSegment(
                    name=f"{name}_{label}",
                    target_uid=name,
                    instruction=self._instructions[label].format(
                        item=name.replace("_", " "), target=target.replace("_", " ")
                    ),
                    actions=actions,
                    validator=validator,
                )

            low, high, yaw = self._bounds(name)
            point = torch.cat(
                (self._xy(low, high, yaw, grasp.grasp_xy), low[:, 2:3] + grasp.grasp_z),
                dim=1,
            )
            yaw = yaw + math.radians(grasp.yaw)
            approach = point.clone()
            approach[:, 2] = torch.maximum(high[:, 2] + 0.09, point[:, 2] + 0.10)
            yield phase("approach", self._motion(approach, yaw, -1))
            yield phase(
                "descend",
                self._motion(point, yaw, -1, speed=0.005, linear=grasp.linear),
            )
            yield phase("close", self._hold(1, 12))
            lift = self._hand()[:, :3, 3].clone()
            lift[:, 2] += 0.14
            yield phase(
                "lift",
                self._motion(lift, yaw, 1, speed=0.004),
                lambda name=name: self._bounds(name)[0][:, 2] > self.clearance[name],
            )
            low, high, _ = self._bounds(name)
            goal_lo, goal_hi, target_yaw = self._bounds(target)
            hand_pos = self._hand()[:, :3, 3]
            drop = hand_pos.clone()
            drop[:, :2] += (
                self._xy(goal_lo, goal_hi, target_yaw, grasp.place_xy)
                - (low + high)[:, :2] * 0.5
            )
            top = (
                goal_hi[:, 2]
                if grasp.surface_z is None
                else goal_lo[:, 2] + grasp.surface_z
            )
            drop[:, 2] += top - low[:, 2] + 0.002
            yaw = yaw + math.radians(grasp.place_yaw_offset)
            transit = drop.clone()
            transit[:, 2] = torch.maximum(
                hand_pos[:, 2], goal_hi[:, 2] + (high - low)[:, 2] + 0.08
            )
            yield phase("transport", self._motion(transit, yaw, 1, speed=0.006))
            yield phase("lower", self._motion(drop, yaw, 1, speed=0.004))
            distance = grasp.scrub_distance
            if distance:
                across = drop.clone()
                across[:, 0] += distance
                yield phase("scrub_out", self._motion(across, yaw, 1, speed=0.004))
                yield phase("scrub_back", self._motion(drop, yaw, 1, speed=0.004))
            yield phase("release", self._hold(-1, 12))
            exit_point = drop.clone()
            exit_point[:, 2] += 0.12
            yield phase("withdraw", self._motion(exit_point, yaw, -1, speed=0.010))
