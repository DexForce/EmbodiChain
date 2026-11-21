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

"""Physical acceptance for the robosuite Tool Hang composite objects."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers.cfg import ConditionCfg, RLTaskCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs.managers.manager_base import Functor
from embodichain.lab.gym.envs.objectives.conditions import ConditionResult
from embodichain.utils.math import quat_from_euler_xyz, quat_mul

__all__ = [
    "ToolHangExpertEnv",
    "ToolHangSuccess",
    "reset_tool_hang",
    "tool_hang_fallen",
]


def _transform_sites(pose: torch.Tensor, sites: torch.Tensor) -> torch.Tensor:
    """Transform fixed object-frame landmarks into the arena frame."""
    return sites @ pose[:, :3, :3].transpose(1, 2) + pose[:, None, :3, 3]


def _straddles(a: torch.Tensor, b: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return (torch.linalg.cross(a, axis) * torch.linalg.cross(b, axis)).sum(-1) < 0


class ToolHangSuccess(Functor):
    """Require simultaneous geometric insertion, threading, support and release.

    Both ring and bar have dedicated contact actors in their fixed-link URDFs.
    The dimensions below are the upstream object-frame sites, in metres.

    Args:
        cfg: Native condition declaration.
        env: Environment containing the three objects and the contact sensor.
    """

    def __init__(self, cfg: ConditionCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._stand = env.sim.get_articulation("stand")
        self._frame = env.sim.get_articulation("frame")
        self._tool = env.sim.get_articulation("tool")
        self._sensor = env.get_sensor("tool_contacts")
        self._bar = self._sensor.get_actor_ids("frame", ["horizontal_bar"])
        self._tool_ids = self._sensor.get_actor_ids("tool", ["tool", "large_ring"])
        self._hand = self._sensor.get_actor_ids(
            "FrankaPanda",
            [
                "fr3_hand",
                "fr3_leftfinger",
                "fr3_rightfinger",
                "fr3_leftpad",
                "fr3_rightpad",
            ],
        )
        self._stand_sites = torch.tensor(
            [
                (0, 0, -0.075),
                (0, 0, 0.08),
                (0, 0.0395, 0.005),
                (-0.0055, 0.045, 0.005),
                (0, 0.0505, 0.005),
                (0.0055, 0.045, 0.005),
            ],
            device=env.device,
        )
        self._frame_sites = torch.tensor(
            [
                (0.04375, 0, -0.13445),
                (0.04375, 0, -0.09),
                (0.04375, 0, 0.08625),
                (-0.0475, 0, 0.08625),
            ],
            device=env.device,
        )
        self._tool_sites = torch.tensor(
            [
                (-0.0935328, 0, 0),
                (-0.0935328 - 0.012934, 0, 0),
                (-0.0935328 + 0.012934, 0, 0),
            ],
            device=env.device,
        )

    def __call__(self, env: EmbodiedEnv) -> ConditionResult:
        """Measure the assembled frame and a physically supported, released ring.

        Args:
            env: Environment after the current physics step.

        Returns:
            Simultaneous success with independent insertion and hanging diagnostics.
        """
        stand = self._stand.get_local_pose(to_matrix=True)
        frame = self._frame.get_local_pose(to_matrix=True)
        tool = self._tool.get_local_pose(to_matrix=True)
        base, top, w0, w1, w2, w3 = _transform_sites(stand, self._stand_sites).unbind(1)
        tip, mount, elbow, end = _transform_sites(frame, self._frame_sites).unbind(1)
        center, side0, side4 = _transform_sites(tool, self._tool_sites).unbind(1)
        vertical = torch.nn.functional.normalize(top - base, dim=-1)
        rod = torch.nn.functional.normalize(elbow - mount, dim=-1)
        assembled = (
            (vertical[:, 2] > math.cos(math.radians(10)))
            & (torch.linalg.vector_norm(tip - base, dim=-1) < 0.05)
            & _straddles(w0 - mount, w2 - mount, rod)
            & _straddles(w1 - mount, w3 - mount, rod)
        )
        bar = elbow - end
        axis = torch.nn.functional.normalize(bar, dim=-1)
        delta = center - end
        projection = (delta * axis).sum(-1)
        along = projection / torch.linalg.vector_norm(bar, dim=-1)
        radial = torch.linalg.vector_norm(delta - projection[:, None] * axis, dim=-1)
        threaded = (
            (radial < 0.00675)
            & (along > 0.05)
            & (along < 1.0)
            & _straddles(side0 - end, side4 - end, axis)
        )

        self._sensor.update()
        report = self._sensor.get_data()
        pairs = report["user_ids"]
        # Force-reporting solvers can support a resting object at positive
        # separation; geometry-only solvers require touching surfaces.
        active = report["is_valid"] & (
            (report["distance"] <= 0)
            | report["impulse_valid"] & (report["impulse"] > 0)
        )

        def touches(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            first = (pairs[..., None] == a[:, None, None, :]).any(-1)
            second = (pairs[..., None] == b[:, None, None, :]).any(-1)
            either_order = (first[..., 0] & second[..., 1]) | (
                first[..., 1] & second[..., 0]
            )
            return (either_order & active).any(-1)

        ring_contact = touches(self._bar, self._tool_ids[:, 1:])
        hand_contact = touches(self._hand, self._tool_ids)
        on_frame = threaded & ring_contact & ~hand_contact
        valid = ~self._sensor.dropped_contacts_mask
        for pose in (stand, frame, tool):
            valid = valid & torch.isfinite(pose).all(dim=(1, 2))
        return ConditionResult(
            assembled & on_frame,
            valid,
            {
                "frame_assembled": assembled,
                "hole_threaded": threaded,
                "ring_bar_contact": ring_contact,
                "hand_tool_contact": hand_contact,
                "tool_on_frame": on_frame,
                "radial_distance": radial,
                "along_fraction": along,
            },
        )


def tool_hang_fallen(env: EmbodiedEnv, height: float = 0.7) -> ConditionResult:
    """Detect a frame or wrench falling below the table.

    Args:
        env: Environment with free frame and wrench articulations.
        height: Minimum arena-local root height in metres.

    Returns:
        Failure and the lower of the two measured heights.
    """
    elevations = torch.stack(
        [
            env.sim.get_articulation(uid).get_local_pose()[:, 2]
            for uid in ("frame", "tool")
        ],
        dim=1,
    )
    lowest = elevations.amin(1)
    return ConditionResult(
        lowest < height, torch.isfinite(lowest), {"min_height": lowest}
    )


def reset_tool_hang(
    env: EmbodiedEnv,
    env_ids: torch.Tensor,
    xy_jitter: float,
    yaw_jitter_deg: float,
    joint_jitter: float,
) -> None:
    """Jitter the already restored objects in arena XY/yaw and the arm at home.

    The host restores solver state for the selected worlds before this event.
    These writes preserve the other worlds and the restored velocities.

    Args:
        env: Environment owning the frame, wrench and Panda.
        env_ids: Selected environment rows.
        xy_jitter: Maximum translation on each horizontal axis in metres.
        yaw_jitter_deg: Maximum rotation about arena vertical in degrees.
        joint_jitter: Maximum independent arm joint displacement in radians.
    """
    for uid in ("frame", "tool"):
        obj = env.sim.get_articulation(uid)
        pose = obj.get_local_pose()[env_ids].clone()
        noise = torch.empty_like(pose[:, :3]).uniform_(-1, 1)
        pose[:, :2] += noise[:, :2] * xy_jitter
        yaw = noise[:, 2] * math.radians(yaw_jitter_deg)
        zero = torch.zeros_like(yaw)
        pose[:, 3:] = quat_mul(quat_from_euler_xyz(zero, zero, yaw), pose[:, 3:])
        obj.set_local_pose(pose, env_ids=env_ids)
    arm_ids = env.robot.get_joint_ids("arm")
    qpos = env.robot.get_qpos()[env_ids][:, arm_ids]
    qpos += torch.empty_like(qpos).uniform_(-joint_jitter, joint_jitter)
    env.robot.set_state(env_ids=env_ids, joint_ids=arm_ids, qpos=qpos, target_qpos=qpos)


@register_env(
    "RobosuiteToolHangExpert-v1", max_episode_steps=2000, supports_handwritten_demo=True
)
class ToolHangExpertEnv(EmbodiedEnv):
    """Use measured Panda motions with the same insertion and contact evaluator as the policy."""

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        cfg.rl_task = RLTaskCfg(
            success={"hanging": ConditionCfg(func=ToolHangSuccess)},
            failure={"fallen": ConditionCfg(func=tool_hang_fallen)},
        )
        # Tune the expert position and velocity servos to the Panda torque limits.
        # The finger drives, arm effort limits and policy controller are independent.
        drives = cfg.robot.joint_drive_props
        for key, values in (
            ("stiffness", (4500, 4500, 3500, 2500, 500, 400, 250)),
            ("damping", (130, 130, 100, 80, 22, 18, 10)),
        ):
            field = {
                k: v
                for k, v in getattr(drives, key).items()
                if not k.startswith("Joint")
            }
            field.update({f"Joint{i}": value for i, value in enumerate(values, 1)})
            setattr(drives, key, field)
        super().__init__(cfg=cfg, **kwargs)

    def create_demo_segments(self, **kwargs: Any) -> Iterator[DemoSegment]:
        """Plan each stage against the physical result of the preceding one."""
        from embodichain.extensions.tasks.manipulation.assembly._tool_hang_expert import (
            ToolHangMotion,
        )

        motion = ToolHangMotion(self)
        yield DemoSegment(
            name="lift_frame",
            target_uid="frame",
            instruction="Grasp the frame and raise its rod over the stand.",
            actions=motion.lift_frame(),
        )
        yield DemoSegment(
            name="seat_frame",
            target_uid="frame",
            instruction="Guide the rod into the socket and release the frame.",
            actions=motion.seat_frame(),
            validator=lambda: self.compute_task_state()[2][
                "success.hanging.frame_assembled"
            ],
        )
        from embodichain.extensions.tasks.manipulation.assembly._tool_hang_wrench import (
            hang_wrench,
        )

        yield DemoSegment(
            name="hang_wrench",
            target_uid="tool",
            instruction="Pick up the wrench, thread its large ring over the hook and release it.",
            actions=hang_wrench(motion),
        )
