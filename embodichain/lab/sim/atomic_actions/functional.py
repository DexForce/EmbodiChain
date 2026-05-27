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

"""Functional public API for atomic actions.

These helpers are thin wrappers over the existing cfg/action classes. They keep
the lower-level ``Action.execute()`` return contract:
``(is_success, trajectory, joint_ids)``.
"""

from __future__ import annotations

from typing import Any

import torch

from .actions import (
    MoveAction,
    MoveActionCfg,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
)
from .core import ObjectSemantics


def move(
    *,
    motion_generator,
    target: torch.Tensor,
    start_qpos: torch.Tensor | None = None,
    control_part: str = "arm",
    sample_interval: int = 50,
    cfg: MoveActionCfg | None = None,
    **cfg_kwargs: Any,
) -> tuple[bool, torch.Tensor, list[int]]:
    """Move an end effector to a target pose without changing the gripper."""
    action_cfg = cfg or MoveActionCfg(
        control_part=control_part,
        sample_interval=sample_interval,
        **cfg_kwargs,
    )
    return MoveAction(motion_generator, cfg=action_cfg).execute(
        target=target,
        start_qpos=start_qpos,
    )


def pick_up(
    *,
    motion_generator,
    target: ObjectSemantics | torch.Tensor,
    start_qpos: torch.Tensor | None = None,
    control_part: str = "arm",
    hand_control_part: str = "hand",
    hand_open_qpos: torch.Tensor | None = None,
    hand_close_qpos: torch.Tensor | None = None,
    pre_grasp_distance: float = 0.15,
    lift_height: float = 0.1,
    approach_direction: torch.Tensor | None = None,
    ranked_grasp_selection: bool = False,
    grasp_approach_directions: list[tuple[str, torch.Tensor]] | None = None,
    grasp_rank_options: dict[str, Any] | None = None,
    sample_interval: int = 80,
    hand_interp_steps: int = 5,
    cfg: PickUpActionCfg | None = None,
    **cfg_kwargs: Any,
) -> tuple[bool, torch.Tensor, list[int]]:
    """Plan approach, grasp, gripper close, and lift for an object or grasp pose."""
    action_cfg = cfg or PickUpActionCfg(
        control_part=control_part,
        hand_control_part=hand_control_part,
        hand_open_qpos=hand_open_qpos,
        hand_close_qpos=hand_close_qpos,
        pre_grasp_distance=pre_grasp_distance,
        lift_height=lift_height,
        approach_direction=(
            approach_direction
            if approach_direction is not None
            else torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        ),
        ranked_grasp_selection=ranked_grasp_selection,
        grasp_approach_directions=grasp_approach_directions,
        grasp_rank_options=grasp_rank_options or {},
        sample_interval=sample_interval,
        hand_interp_steps=hand_interp_steps,
        **cfg_kwargs,
    )
    return PickUpAction(motion_generator, cfg=action_cfg).execute(
        target=target,
        start_qpos=start_qpos,
    )


def place(
    *,
    motion_generator,
    target: torch.Tensor,
    start_qpos: torch.Tensor | None = None,
    control_part: str = "arm",
    hand_control_part: str = "hand",
    hand_open_qpos: torch.Tensor | None = None,
    hand_close_qpos: torch.Tensor | None = None,
    lift_height: float = 0.1,
    sample_interval: int = 80,
    hand_interp_steps: int = 5,
    cfg: PlaceActionCfg | None = None,
    **cfg_kwargs: Any,
) -> tuple[bool, torch.Tensor, list[int]]:
    """Plan place, gripper open, and retract for a held object."""
    action_cfg = cfg or PlaceActionCfg(
        control_part=control_part,
        hand_control_part=hand_control_part,
        hand_open_qpos=hand_open_qpos,
        hand_close_qpos=hand_close_qpos,
        lift_height=lift_height,
        sample_interval=sample_interval,
        hand_interp_steps=hand_interp_steps,
        **cfg_kwargs,
    )
    return PlaceAction(motion_generator, cfg=action_cfg).execute(
        target=target,
        start_qpos=start_qpos,
    )


def gripper_open(
    *,
    motion_generator,
    open_qpos: torch.Tensor | None = None,
    start_qpos: torch.Tensor | None = None,
    control_part: str = "hand",
    sample_interval: int = 15,
    cfg: MoveActionCfg | None = None,
    **cfg_kwargs: Any,
) -> tuple[bool, torch.Tensor, list[int]]:
    """Open a gripper through MoveAction joint interpolation."""
    target_qpos = open_qpos
    action_cfg = cfg or MoveActionCfg(
        control_part=control_part,
        sample_interval=sample_interval,
        **cfg_kwargs,
    )
    return MoveAction(motion_generator, cfg=action_cfg).execute(
        target=target_qpos,
        start_qpos=start_qpos,
    )


def gripper_close(
    *,
    motion_generator,
    close_qpos: torch.Tensor | None = None,
    start_qpos: torch.Tensor | None = None,
    control_part: str = "hand",
    sample_interval: int = 15,
    cfg: MoveActionCfg | None = None,
    **cfg_kwargs: Any,
) -> tuple[bool, torch.Tensor, list[int]]:
    """Close a gripper through MoveAction joint interpolation."""
    target_qpos = close_qpos
    action_cfg = cfg or MoveActionCfg(
        control_part=control_part,
        sample_interval=sample_interval,
        **cfg_kwargs,
    )
    return MoveAction(motion_generator, cfg=action_cfg).execute(
        target=target_qpos,
        start_qpos=start_qpos,
    )
