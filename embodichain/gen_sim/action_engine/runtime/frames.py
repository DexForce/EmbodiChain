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

"""Resolve directional relations in live robot and world frames."""

from __future__ import annotations

from typing import Any

import torch

from .robot_parts import arm_control_part

__all__ = [
    "DIRECTIONAL_RELATIONS",
    "arm_base_poses",
    "relation_axes",
    "relation_offset",
    "robot_frame_axes",
]


DIRECTIONAL_RELATIONS = frozenset(
    {
        "left",
        "left_of",
        "right",
        "right_of",
        "front",
        "front_of",
        "in_front_of",
        "behind",
        "back",
        "front_left",
        "front_left_of",
        "front_right",
        "front_right_of",
        "back_left",
        "back_left_of",
        "back_right",
        "back_right_of",
    }
)


def arm_base_poses(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return live world poses of the left and right arm bases."""
    left_part = arm_control_part(env, "left_arm")
    right_part = arm_control_part(env, "right_arm")
    robot = env.robot
    if hasattr(robot, "get_solver") and hasattr(robot, "get_link_pose"):
        left_solver = robot.get_solver(name=left_part)
        right_solver = robot.get_solver(name=right_part)
        left_root = getattr(left_solver, "root_link_name", None)
        right_root = getattr(right_solver, "root_link_name", None)
        if left_root is None or right_root is None:
            raise ValueError("Directional grounding requires both arm root links.")
        left = robot.get_link_pose(link_name=left_root, to_matrix=True)
        right = robot.get_link_pose(link_name=right_root, to_matrix=True)
    elif hasattr(robot, "get_control_part_base_pose"):
        left = robot.get_control_part_base_pose(name=left_part, to_matrix=True)
        right = robot.get_control_part_base_pose(name=right_part, to_matrix=True)
    elif hasattr(env, "get_current_xpos_agent"):
        left, right = env.get_current_xpos_agent()
    else:
        raise ValueError(
            "Directional grounding requires live left/right arm-base or TCP poses."
        )

    left = _batched_pose(left, env)
    right = _batched_pose(right, env)
    return left, right


def robot_frame_axes(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized world-space forward and left axes for a dual-arm robot."""
    left, right = arm_base_poses(env)
    lateral = left[:, :2, 3] - right[:, :2, 3]
    norm = torch.linalg.vector_norm(lateral, dim=1, keepdim=True)
    if bool((norm <= 1.0e-6).any()):
        raise ValueError("Left and right arm bases must have distinct XY positions.")
    lateral = lateral / norm
    forward = torch.stack((lateral[:, 1], -lateral[:, 0]), dim=1)
    return forward, lateral


def relation_axes(
    env: Any,
    relation: str,
    *,
    frame: str,
) -> tuple[torch.Tensor, ...]:
    """Return signed world-space axes whose projections define a relation."""
    relation = str(relation)
    if relation not in DIRECTIONAL_RELATIONS:
        return ()
    if frame == "robot":
        forward, lateral = robot_frame_axes(env)
    elif frame == "world":
        count = int(env.num_envs)
        forward = torch.tensor(
            [1.0, 0.0], dtype=torch.float32, device=env.device
        ).repeat(count, 1)
        lateral = torch.tensor(
            [0.0, 1.0], dtype=torch.float32, device=env.device
        ).repeat(count, 1)
    else:
        raise ValueError(f"Unsupported directional relation frame {frame!r}.")

    components: list[torch.Tensor] = []
    if relation.startswith("front") or relation in {"front", "front_of", "in_front_of"}:
        components.append(forward)
    elif relation.startswith("back") or relation in {"behind", "back"}:
        components.append(-forward)
    if "left" in relation or relation in {"left", "left_of"}:
        components.append(lateral)
    elif "right" in relation or relation in {"right", "right_of"}:
        components.append(-lateral)
    return tuple(components)


def relation_offset(
    env: Any,
    relation: str,
    *,
    frame: str,
    forward_distance: float,
    lateral_distance: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Resolve one directional relation into a batched world-space offset."""
    axes = relation_axes(env, relation, frame=frame)
    if not axes:
        return None
    offset = torch.zeros((int(env.num_envs), 3), dtype=dtype, device=device)
    has_forward = relation.startswith(("front", "back")) or relation in {
        "front",
        "front_of",
        "in_front_of",
        "behind",
        "back",
    }
    for index, axis in enumerate(axes):
        axis = axis.to(dtype=dtype, device=device)
        distance = forward_distance if has_forward and index == 0 else lateral_distance
        offset[:, :2] += axis * float(distance)
    return offset


def _batched_pose(value: Any, env: Any) -> torch.Tensor:
    pose = torch.as_tensor(value, dtype=torch.float32, device=env.device)
    if pose.shape == (4, 4):
        pose = pose.unsqueeze(0).repeat(int(env.num_envs), 1, 1)
    if pose.shape != (int(env.num_envs), 4, 4):
        raise ValueError(
            "Frame pose must have shape (4, 4) or "
            f"({int(env.num_envs)}, 4, 4), got {tuple(pose.shape)}."
        )
    return pose
