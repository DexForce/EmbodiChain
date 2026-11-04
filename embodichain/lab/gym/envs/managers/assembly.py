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
"""Reset and reward functors for blind assembly tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import numpy as np
from embodichain.lab.gym.envs.objectives.conditions import ConditionResult

from .observations import (
    AssemblyProprioception,
    assembly_contact_labels,
    assembly_privileged_state,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = [
    "AssemblyProprioception",
    "assembly_privileged_state",
    "assembly_contact_labels",
    "reset_assembly",
    "assembly_success",
    "assembly_failure",
    "assembly_milestone_reward",
]


def reset_assembly(env: EmbodiedEnv, env_ids: torch.Tensor) -> None:
    """Initialize selected physical grasps without stepping other arenas."""
    state = env.assembly_state
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=state.env.device)
    n = len(ids)
    if n == 0:
        return
    q = state.env.robot.get_qpos(name="arm")[ids]
    tcp = (
        state._assembly_robot_base[None]
        @ state._assembly_chain.forward_kinematics(q).get_matrix()
    )
    turn = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=state.env.device))
    rotation = tcp[:, :3, :3] @ turn
    # Keep the upper grip section between the physical pads. Gripper
    # closure remains a force-limited drive; no fixed grasp constraint.
    grip_z = (
        (float(state.scene.assembly_asset.config["face_width"]) / 2 + 0.0085 + 0.01875)
        if state.scene.assembly_kind == "gear"
        else state.scene._moving_zmax - 0.012
    )
    part = torch.eye(4, device=state.env.device).repeat(n, 1, 1)
    part[:, :3, :3] = rotation
    part[:, :3, 3] = tcp[:, :3, 3] - (
        rotation
        @ torch.tensor([0.0, 0.0, grip_z], device=state.env.device)
        .expand(n, -1)
        .unsqueeze(-1)
    ).squeeze(-1)
    grasp_jitter = float(getattr(state.env, "grasp_position_range", 0.0025))
    # Along-pad displacement is along z; lateral x does not change the
    # closing aperture. A y displacement would create an invalid initial
    # one-sided clamp rather than the paper's along-pad randomization.
    part[:, :3, 3] += (
        rotation
        @ (
            (torch.rand((n, 3), device=state.env.device) - 0.5)
            * torch.tensor([0.0, 0.0, 2 * grasp_jitter], device=state.env.device)
        ).unsqueeze(-1)
    ).squeeze(-1)
    fixture = part.clone()
    initial_offset = float(
        getattr(
            state.env,
            "initial_position_range",
            {"peg": 0.0125, "gear": 0.0075, "nut": 0.015}[state.scene.assembly_kind],
        )
    )
    offset = (torch.rand((n, 3), device=state.env.device) - 0.5) * torch.tensor(
        [2 * initial_offset, 2 * initial_offset, 0.0], device=state.env.device
    )
    offset[:, 2] = state.scene._moving_zmin - state.scene._fixture_zmax
    fixture[:, :3, 3] += (rotation @ offset.unsqueeze(-1)).squeeze(-1)
    state.env.sim.get_rigid_object(state.scene._moving_uid).set_local_pose(
        part, env_ids=ids
    )
    state.env.sim.get_rigid_object(state.scene._moving_uid).set_velocity(
        lin_vel=torch.zeros((n, 3), device=state.env.device),
        ang_vel=torch.zeros((n, 3), device=state.env.device),
        env_ids=ids,
    )
    state.env.sim.get_rigid_object(state.scene._fixture_uid).set_local_pose(
        fixture, env_ids=ids
    )
    if state.scene.assembly_kind == "gear":
        # Every fixture entity is transformed rigidly with the shaft.
        anchor = state.env.sim.get_articulation("gear_a")
        source_shaft = np.asarray(state.scene._fixture_definition.position)
        source_anchor = next(
            b for b in state.scene.assembly_asset.bodies if b.name == "gear_a"
        )
        local = torch.as_tensor(
            np.asarray(source_anchor.position) - source_shaft,
            dtype=torch.float32,
            device=state.env.device,
        )
        anchor_pose = fixture.clone()
        anchor_pose[:, :3, 3] += (rotation @ local.expand(n, -1).unsqueeze(-1)).squeeze(
            -1
        )
        anchor.set_local_pose(anchor_pose, env_ids=ids)
        anchor.set_qpos(
            torch.rand((n, 1), device=state.env.device) * 2 * torch.pi,
            env_ids=ids,
            target=False,
        )
    grip_ids, grip_values = _grasp_joint_reset(
        state.env.robot, state.scene._grasp_reset_joint, n, state.env.device
    )
    state.env.robot.set_qpos(
        grip_values,
        joint_ids=grip_ids,
        env_ids=ids,
        target=False,
    )
    state.env.robot.set_qpos(
        torch.full(
            (n, len(state.env.robot.get_joint_ids(name="eef"))),
            state.scene._grasp_target_joint,
            device=state.env.device,
        ),
        name="eef",
        env_ids=ids,
    )
    state._assembly_reset_rows[ids] = True
    state._assembly_success_hold[ids] = 0
    state._assembly_milestones[ids] = False
    state._assembly_new_milestones[ids] = False
    state._assembly_angle[ids] = 0
    state._assembly_previous_yaw[ids] = 0
    state._assembly_capture_angle[ids] = float("nan")
    state._assembly_capture_depth[ids] = float("nan")
    for field in ("_assembly_pending_reset_ids", "_assembly_frame_reset_ids"):
        previous = getattr(state, field, None)
        setattr(
            state,
            field,
            (
                ids.clone()
                if previous is None
                else torch.unique(torch.cat((previous, ids)))
            ),
        )
    state._assembly_revision += 1
    state.refresh_assembly_state()


def assembly_milestone_reward(
    env: EmbodiedEnv, obs: Any, action: torch.Tensor, info: dict[str, Any]
) -> torch.Tensor:
    """Pay each capture/mesh/seat milestone once in the current episode."""
    env.assembly_state.refresh_assembly_state()
    weights = (
        [1.0, 0.0, 1.0, 0.0]
        if env.assembly_state.scene.assembly_kind == "peg"
        else [1.0, 1.0, 1.0, 0.0]
    )
    return (
        env.assembly_state._assembly_new_milestones.float()
        * torch.tensor(weights, device=env.device)
    ).sum(dim=-1)


def _grasp_joint_reset(
    robot: Any, value: float, rows: int, device: torch.device
) -> tuple[list[int], torch.Tensor]:
    """Expand the EEF leaders through source mimic relations without name rules."""
    values = {int(i): value for i in robot.get_joint_ids(name="eef")}
    relations = list(
        zip(
            robot.mimic_ids,
            robot.mimic_parents,
            robot.mimic_multipliers,
            robot.mimic_offsets,
        )
    )
    for _ in range(len(relations) + 1):
        for child, parent, multiplier, offset in relations:
            if parent in values:
                values[int(child)] = values[parent] * multiplier + offset
    indices = sorted(values)
    return indices, torch.tensor(
        [values[i] for i in indices], dtype=torch.float32, device=device
    ).expand(rows, -1)


def assembly_success(env: EmbodiedEnv) -> ConditionResult:
    """Advance once per new control state and measure physical success.

    Args:
        env: Environment with startup-bound assembly state.

    Returns:
        Physical success, validity and shared geometric diagnostics.
    """
    state = env.assembly_state
    state.advance()
    values = state._assembly_metrics
    metrics = {
        k: v for k, v in values.items() if k not in {"success", "fail", "relative_pose"}
    }
    valid = torch.isfinite(values["relative_pose"]).all(dim=-1)
    return ConditionResult(values["success"], valid, metrics)


def assembly_failure(env: EmbodiedEnv) -> ConditionResult:
    """Read failure from the snapshot consumed by success and rewards.

    Args:
        env: Environment with startup-bound assembly state.

    Returns:
        Failure and measurement-validity masks.
    """
    state = env.assembly_state
    state.refresh_assembly_state()
    values = state._assembly_metrics
    return ConditionResult(
        values["fail"], torch.isfinite(values["relative_pose"]).all(dim=-1), {}
    )
