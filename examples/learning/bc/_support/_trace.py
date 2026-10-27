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

"""Read-only physical snapshots for the optional PickPlace diagnostic trace."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def task_snapshot(raw: Any, geometry: torch.Tensor) -> dict[str, Any]:
    """Read one measured state and cached outcome without advancing counters.

    The necessary conditions mirror the showcase's physical predicate for
    diagnosis only. The environment remains the authority for success, and this
    helper never invokes its update, reset, expert or control methods.
    """
    cube = raw.sim.get_rigid_object("cube")
    pose, velocity = cube.get_local_pose(), cube.body_data.vel
    qpos = raw.robot.get_qpos()
    tcp = raw.robot.compute_fk(
        qpos=qpos[:, raw.robot.get_joint_ids("arm")], name="arm", to_matrix=True
    )[:, :3, 3]
    goal = raw.goal_pose
    distance = torch.linalg.vector_norm(tcp - pose[:, :3], dim=-1)
    goal_error = torch.linalg.vector_norm(pose[:, :3] - goal, dim=-1)
    linear_speed = torch.linalg.vector_norm(velocity[:, :3], dim=-1)
    angular_speed = torch.linalg.vector_norm(velocity[:, 3:], dim=-1)
    conditions = {
        "lifted": raw._lifted,
        "near_goal": goal_error <= 0.03,
        "goal_height": (pose[:, 2] - goal[:, 2]).abs() <= 0.012,
        "opened": qpos[:, 7:].amin(dim=-1) >= 0.035,
        "withdrawn": distance >= 0.10,
        "linear_stable": linear_speed < 0.05,
        "angular_stable": angular_speed < 0.2,
        "finite": torch.isfinite(pose).all(-1) & torch.isfinite(velocity).all(-1),
    }
    measured = {name: bool(value[0]) for name, value in conditions.items()}
    stable_steps = int(raw._stable_steps[0])
    stable_seconds = stable_steps * raw.step_dt
    cached_success = bool(raw._success[0])
    if cached_success:
        phase = "success"
    elif measured["near_goal"]:
        if measured["opened"] and measured["withdrawn"]:
            phase = "stabilize"
        elif measured["opened"]:
            phase = "withdraw"
        else:
            phase = "release"
    elif measured["lifted"]:
        phase = "transport"
    else:
        phase = "grasp" if float(distance[0]) <= 0.08 else "approach"
    return {
        "geometry": geometry[0].tolist(),
        "proprio": geometry[0, :27].tolist(),
        "cube_pose": pose[0].tolist(),
        "tcp_xyz": tcp[0].tolist(),
        "gripper_qpos": qpos[0, 7:].tolist(),
        "goal_xyz": goal[0].tolist(),
        "cube_linear_velocity": velocity[0, :3].tolist(),
        "cube_angular_velocity": velocity[0, 3:].tolist(),
        "cube_linear_speed": float(linear_speed[0]),
        "cube_angular_speed": float(angular_speed[0]),
        "tcp_cube_distance": float(distance[0]),
        "goal_error": float(goal_error[0]),
        "conditions": measured,
        "instantaneously_accepted": all(measured.values()),
        "stable_steps": stable_steps,
        "stable_seconds": stable_seconds,
        "cached_success": cached_success,
        "success_counter_consistent": cached_success == (stable_seconds >= 0.3 - 1e-8),
        "diagnostic_phase": phase,
        "observation_tcp_offset": float(
            torch.linalg.vector_norm(geometry[0, 18:21].to(tcp) - tcp[0])
        ),
        "observation_cube_offset": float(
            torch.linalg.vector_norm(geometry[0, 27:30].to(pose) - pose[0, :3])
        ),
    }


def save_replan_inputs(
    executor: Any, destination: Path, features: torch.Tensor | None
) -> dict[str, str]:
    """Save a bounded sample of the actual measured input history.

    Byte RGB is recovered from the executor's already scaled uint8 inputs;
    diagnostics do not introduce another encoder forward or policy history.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "history": executor._history.detach().cpu().clone(),
            "images_uint8": {
                name: value.detach().cpu().mul(255).round().to(torch.uint8)
                for name, value in executor._images.items()
            },
            "encoder_features": features,
        },
        destination,
    )
    return {
        "path": str(destination.resolve()),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
