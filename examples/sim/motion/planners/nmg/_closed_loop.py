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

"""Simulation feedback and an authored joint disturbance for the NMG demo."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.motion.planners import NeuralPlanner, PlanState
    from examples.sim.motion.planners.nmg._visuals import TrajectoryMarkers


MAX_STEPS_PER_TARGET = 120
FINAL_HOLD_SECONDS = 0.5
_PERTURBATION_WAYPOINT = 1
_PERTURB_AFTER_STEPS = 4
_PERTURBATION_RAD = -0.55
_ARRIVAL_HOLD_STEPS = 3


@torch.no_grad()
def play_closed_loop(
    sim: SimulationManager,
    planner: NeuralPlanner,
    target_states: list[PlanState],
    modalities: list[str],
    joint_ids: Sequence[int],
    control_dt: float,
    markers: TrajectoryMarkers | None,
) -> dict[str, object]:
    """Drive the arm from measured joint positions and record strict arrivals."""
    robot = planner.robot
    physics_dt = sim.sim_config.physics_dt
    substeps = round(control_dt / physics_dt)
    started = sim.simulation_time

    # Reuse the planner's encoded targets and acceptance test. Each policy step
    # receives a new measurement from the simulator and carries the last action.
    (
        positions,
        orientations,
        joints,
        valid_mask,
        position_mask,
        orientation_mask,
        joint_mask,
        target_count,
    ) = planner._parse_waypoints(target_states, modalities, control_part="arm")
    targets = (positions, orientations, joints)
    constraint_masks = (position_mask, orientation_mask, joint_mask)
    lower, upper = robot.get_qpos_limits(name="arm").unbind(-1)
    max_delta = robot.get_qvel_limits(name="arm").clamp(max=1.2) * control_dt
    action_scale = float(planner.cfg.action_scale)
    qpos = robot.get_qpos(name="arm")
    pose = planner._fk_matrix(qpos, "arm")
    last_action = torch.zeros_like(qpos)
    zero_qvel = torch.zeros_like(qpos)
    robot.set_qvel(zero_qvel, joint_ids=joint_ids)
    active = torch.zeros(1, dtype=torch.long, device=qpos.device)

    arrivals: list[dict[str, object]] = []
    disturbance: dict[str, object] | None = None
    feedback_steps = 0
    injected_after_step = 0
    for slot in range(target_count):
        active.fill_(slot)
        if markers is not None:
            markers.show_targets(slot, active=slot)
        settled_steps = 0
        policy_pose = planner._policy_pose_xyzw(pose, "arm")
        for step in range(MAX_STEPS_PER_TARGET):
            if slot == _PERTURBATION_WAYPOINT and step == _PERTURB_AFTER_STEPS:
                before_joint = float(qpos[0, 0])
                before_tip = pose[0, :3, 3].clone()
                displaced = qpos.clone()
                displaced[:, 0] += _PERTURBATION_RAD
                displaced.clamp_(lower, upper)
                robot.set_state(
                    joint_ids=joint_ids,
                    qpos=displaced,
                    qvel=zero_qvel,
                    target_qpos=displaced,
                    target_qvel=zero_qvel,
                )
                qpos = robot.get_qpos(name="arm")
                pose = planner._fk_matrix(qpos, "arm")
                after_tip = pose[0, :3, 3]
                disturbance = {
                    "waypoint": slot,
                    "time_s": sim.simulation_time - started,
                    "joint": robot.control_parts["arm"][0],
                    "joint_delta_rad": float(qpos[0, 0]) - before_joint,
                    "tcp_displacement_m": float((after_tip - before_tip).norm()),
                    "recovered": False,
                }
                injected_after_step = feedback_steps
                if markers is not None:
                    markers.show_perturbation(before_tip, after_tip)
                # Give the recording one frame of the displaced state. Replace
                # the drive's old setpoint so it cannot pull the arm back.
                sim.update(physics_dt, substeps)
                qpos = robot.get_qpos(name="arm")
                pose = planner._fk_matrix(qpos, "arm")
                policy_pose = planner._policy_pose_xyzw(pose, "arm")
                settled_steps = 0

            obs = planner._build_obs(
                qpos,
                policy_pose,
                *targets,
                valid_mask,
                *constraint_masks,
                active,
                last_action,
            )
            last_action = planner._policy(obs.to(dtype=torch.float32)).clamp(-1.0, 1.0)
            delta = (last_action * action_scale).clamp(-max_delta, max_delta)
            robot.set_qpos((qpos + delta).clamp(lower, upper), joint_ids=joint_ids)
            sim.update(physics_dt, substeps)
            feedback_steps += 1

            qpos = robot.get_qpos(name="arm")
            pose = planner._fk_matrix(qpos, "arm")
            policy_pose = planner._policy_pose_xyzw(pose, "arm")
            if markers is not None:
                markers.record_tcp(pose[0, :3, 3])
            reached = bool(
                planner._is_active_reached(
                    qpos, policy_pose, *targets, *constraint_masks, active
                )[0]
            )
            settled_steps = settled_steps + 1 if reached else 0
            if settled_steps < _ARRIVAL_HOLD_STEPS:
                continue
            if slot == _PERTURBATION_WAYPOINT and disturbance is None:
                continue
            arrivals.append(
                {
                    "waypoint": slot,
                    "modality": modalities[slot],
                    "feedback_steps": step + 1,
                    "time_s": sim.simulation_time - started,
                }
            )
            if slot == _PERTURBATION_WAYPOINT:
                assert disturbance is not None
                disturbance["recovered"] = True
                disturbance["recovery_steps"] = feedback_steps - injected_after_step
            if markers is not None:
                markers.show_targets(slot + 1)
            break
        else:
            break

    completed = len(arrivals) == target_count
    if completed:
        robot.set_qpos(qpos, joint_ids=joint_ids)
        for _ in range(round(FINAL_HOLD_SECONDS / physics_dt)):
            sim.update(step=1)
    return {
        "simulation_playback": "completed" if completed else "failed",
        "playback_duration_s": sim.simulation_time - started,
        "feedback_steps": feedback_steps,
        "feedback_arrivals": arrivals,
        "perturbation": disturbance,
    }
