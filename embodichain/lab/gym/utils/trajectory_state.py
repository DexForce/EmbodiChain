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

"""Shared simulation-state capture and restore helpers for trajectories.

All seven-element root and rigid-object poses use EmbodiChain's
``(x, y, z, qx, qy, qz, qw)`` convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, TYPE_CHECKING
from dataclasses import asdict

import torch
from tensordict import TensorDictBase

from embodichain.lab.evaluation.observations import _raw_field_kind
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.managers.cfg import ObservationCfg
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.task_program.integrations import (
        SimulationTaskProgramRegistration,
    )

__all__ = [
    "capture_trajectory_state",
    "restore_trajectory_state",
    "trajectory_observation_cfgs",
    "export_observation_history",
]


class _TrajectoryEnv(Protocol):
    """Environment surface required by trajectory state helpers."""

    robot: Any
    sim: Any


_TrajectoryState: TypeAlias = Mapping[str, Any] | TensorDictBase


def _assign_state(
    target: torch.Tensor,
    source: torch.Tensor,
    env_ids: torch.Tensor,
    step_ids: torch.Tensor | None,
) -> None:
    """Assign selected environment rows, optionally at per-row time indices."""
    if step_ids is None:
        target[env_ids] = source[env_ids]
    else:
        target[env_ids, step_ids] = source[env_ids]


def capture_trajectory_state(
    env: _TrajectoryEnv,
    states: TensorDictBase,
    env_ids: torch.Tensor,
    step_ids: torch.Tensor | None = None,
) -> None:
    """Capture restorable simulation state into a trajectory state view.

    Args:
        env: Environment containing the robot and simulation scene entities.
        states: Destination trajectory state buffer.
        env_ids: Environment rows to capture.
        step_ids: Optional per-environment destination time indices.
    """
    robot_states = states["robot"]
    _assign_state(
        robot_states["root_pose"], env.robot.get_local_pose(), env_ids, step_ids
    )
    _assign_state(robot_states["qpos"], env.robot.get_qpos(), env_ids, step_ids)
    if "qvel" in robot_states.keys():
        _assign_state(robot_states["qvel"], env.robot.get_qvel(), env_ids, step_ids)

    if "articulations" in states.keys():
        articulation_states = states["articulations"]
        for uid, articulation in env.sim._articulations.items():
            if uid not in articulation_states.keys():
                continue
            entity_states = articulation_states[uid]
            _assign_state(
                entity_states["root_pose"],
                articulation.get_local_pose(),
                env_ids,
                step_ids,
            )
            _assign_state(
                entity_states["qpos"], articulation.get_qpos(), env_ids, step_ids
            )
            if "qvel" in entity_states.keys():
                _assign_state(
                    entity_states["qvel"],
                    articulation.get_qvel(),
                    env_ids,
                    step_ids,
                )

    if "rigid_objects" in states.keys():
        rigid_states = states["rigid_objects"]
        for uid, rigid_object in env.sim._rigid_objects.items():
            if uid not in rigid_states.keys():
                continue
            entity_states = rigid_states[uid]
            _assign_state(
                entity_states["pose"],
                rigid_object.get_local_pose(),
                env_ids,
                step_ids,
            )
            if "lin_vel" in entity_states.keys():
                body_state = rigid_object.body_state
                _assign_state(
                    entity_states["lin_vel"],
                    body_state[:, 7:10],
                    env_ids,
                    step_ids,
                )
                _assign_state(
                    entity_states["ang_vel"],
                    body_state[:, 10:13],
                    env_ids,
                    step_ids,
                )


def restore_trajectory_state(
    env: _TrajectoryEnv,
    states: _TrajectoryState,
) -> None:
    """Restore one trajectory timestep into an environment scene.

    Args:
        env: Environment containing the robot and simulation scene entities.
        states: One timestep of recorded robot and scene-object states.
    """
    robot_states = states["robot"]
    env.robot.set_local_pose(robot_states["root_pose"])
    # Restore complete qpos, including mimic joints. Kinematic replay disables
    # physics, so active-joint writes alone cannot propagate to mimic children.
    env.robot.set_qpos(robot_states["qpos"], target=False)
    if "qvel" in robot_states.keys():
        env.robot.set_qvel(robot_states["qvel"], target=False)

    if "articulations" in states.keys():
        articulation_states = states["articulations"]
        trajectory_uids = set(articulation_states.keys())
        scene_uids = set(env.sim._articulations.keys())
        for uid in trajectory_uids - scene_uids:
            logger.log_warning(
                f"Trajectory articulation '{uid}' is not present in the scene; skipping."
            )
        for uid in trajectory_uids & scene_uids:
            articulation = env.sim._articulations[uid]
            entity_states = articulation_states[uid]
            articulation.set_local_pose(entity_states["root_pose"])
            articulation.set_qpos(entity_states["qpos"], target=False)
            if "qvel" in entity_states.keys():
                articulation.set_qvel(entity_states["qvel"], target=False)
        for uid in scene_uids - trajectory_uids:
            logger.log_warning(
                f"Scene articulation '{uid}' is not in the trajectory; leaving initial state."
            )

    if "rigid_objects" in states.keys():
        rigid_states = states["rigid_objects"]
        trajectory_uids = set(rigid_states.keys())
        scene_uids = set(env.sim._rigid_objects.keys())
        for uid in trajectory_uids - scene_uids:
            logger.log_warning(
                f"Trajectory rigid object '{uid}' is not present in the scene; skipping."
            )
        for uid in trajectory_uids & scene_uids:
            rigid_object = env.sim._rigid_objects[uid]
            entity_states = rigid_states[uid]
            rigid_object.set_local_pose(entity_states["pose"])
            if "lin_vel" in entity_states.keys() and not rigid_object.is_non_dynamic:
                rigid_object.set_velocity(
                    lin_vel=entity_states["lin_vel"],
                    ang_vel=entity_states["ang_vel"],
                )
        for uid in scene_uids - trajectory_uids:
            logger.log_warning(
                f"Scene rigid object '{uid}' is not in the trajectory; leaving initial state."
            )


def _raw_entity_observation_cfg(name: str, uid: str, quantity: str) -> ObservationCfg:
    from embodichain.lab.gym.envs.managers.cfg import ObservationCfg, SceneEntityCfg
    from embodichain.lab.gym.envs.managers.observations import get_entity_state

    return ObservationCfg(
        func=get_entity_state,
        mode="add",
        name=name,
        params={"entity_cfg": SceneEntityCfg(uid=uid), "quantity": quantity},
        consumers=("recording",),
    )


def _evaluation_observation_cfgs(
    registration: SimulationTaskProgramRegistration,
) -> dict[str, ObservationCfg]:
    from embodichain.lab.gym.envs.managers.cfg import ObservationCfg
    from embodichain.lab.gym.envs.managers.observations import (
        _get_evaluation_state,
        _get_segment_start_state,
    )

    fields = {
        field
        for spec in registration.evaluation_conditions.values()
        for field in spec.required_fields
    }
    result = {}
    for binding in registration.scene_binding.rigid_objects:
        uid = binding.simulation_uid
        for quantity in ("pose", "lin_vel", "ang_vel"):
            name = f"rigid_objects/{uid}/{quantity}"
            if name in fields:
                result[f"recording:{name}"] = _raw_entity_observation_cfg(
                    name, uid, quantity
                )
        name = f"rigid_objects/{uid}/frame"
        if name in fields:
            result[f"recording:{name}"] = _raw_entity_observation_cfg(name, uid, "pose")
        name = f"rigid_objects/{uid}/segment_start_frame"
        if name in fields:
            result[f"recording:{name}"] = ObservationCfg(
                func=_get_segment_start_state,
                mode="add",
                name=name,
                params={"uid": uid, "field": name},
                consumers=("recording",),
            )
    for binding in getattr(registration.scene_binding, "articulations", ()):
        uid = binding.simulation_uid
        for name in sorted(fields):
            if name.startswith(f"articulations/{uid}/qpos/"):
                result[f"recording:{name}"] = _raw_entity_observation_cfg(
                    name, uid, "qpos"
                )
    for name in sorted(fields):
        if name.startswith(("links/", "robot/", "robot_links/")):
            result[f"recording:{name}"] = ObservationCfg(
                func=_get_evaluation_state,
                mode="add",
                name=name,
                params={"field": name},
                consumers=("recording",),
            )
    if {cfg.name for cfg in result.values()} != fields:
        raise ValueError("Evaluation requires registered raw physical fields.")
    kinds = {name: _raw_field_kind(name) for name in fields}
    layouts = {
        name: name.rsplit("/", 1)[0] for name in fields if name.startswith("robot/")
    }
    for spec in registration.evaluation_conditions.values():
        spec._check_input_contract(kinds, layouts)
    return result


def _validate_evaluation_observation_bindings(env: _TrajectoryEnv, configs) -> None:
    from embodichain.lab.task_program.integrations.simulation.bindings import (
        _require_control_part_dof,
    )

    for cfg in configs.values():
        parts = cfg.name.split("/")
        if parts[0] == "articulations":
            _, uid, _, joint = parts
            asset = env.sim.get_articulation(uid)
            if asset is None or joint not in asset.joint_names:
                raise ValueError(f"Evaluation field {cfg.name!r} has no native joint.")
        elif parts[0] == "links":
            _, uid, link, _ = parts
            asset = env.sim.get_articulation(uid)
            if asset is None or link not in asset.link_names:
                raise ValueError(f"Evaluation field {cfg.name!r} has no native link.")
        elif parts[0] == "robot_links":
            if parts[1] not in env.robot.link_names:
                raise ValueError(
                    f"Evaluation field {cfg.name!r} has no native robot link."
                )
        elif parts[0] == "robot":
            _require_control_part_dof(env.robot, parts[1])


def trajectory_observation_cfgs(
    env: _TrajectoryEnv, uids: list[str] | None = None
) -> dict[str, ObservationCfg]:
    """Declare replay fields through the same raw observation functors as policy.

    Args:
        env: Host exposing robot and scene registries for schema assembly.
        uids: Optional allow-list for non-robot entities.

    Returns:
        Recording-only add functors. No physical state is acquired here; legacy
        trajectory field paths are export names, not another acquisition path.
    """
    result = {}

    def add(prefix: str, uid: str, quantities: tuple[str, ...]) -> None:
        for quantity in quantities:
            field = (
                "root_pose"
                if quantity == "pose" and prefix != "rigid_objects"
                else quantity
            )
            name = f"{prefix}/{uid}/{field}" if prefix != "robot" else f"robot/{field}"
            if prefix == "rigid_objects" and quantity == "pose":
                name = f"rigid_objects/{uid}/pose"
            result[f"recording:{name}"] = _raw_entity_observation_cfg(
                name, uid, quantity
            )

    add("robot", env.robot.uid, ("pose", "qpos", "qvel"))
    for uid in env.sim._articulations:
        if uids is None or uid in uids:
            add("articulations", uid, ("pose", "qpos", "qvel"))
    for uid in env.sim._rigid_objects:
        if uids is None or uid in uids:
            add("rigid_objects", uid, ("pose", "lin_vel", "ang_vel"))
    return result


def export_observation_history(
    observations: ObservationManager,
    env_ids: list[int] | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Project numeric history through its single observation schema for export.

    Args:
        observations: Manager owning the field meanings and canonical IDs.
        env_ids: Selected original environment rows, or all rows.
        device: Snapshot destination device.

    Returns:
        Owned history with source descriptors and alias views. Existing
        trajectory state projections can share the same snapshot tensors.
    """
    snapshot = observations.snapshot_history(env_ids, device=device)
    fields = snapshot["fields"]
    for name, values in fields.items():
        metadata = asdict(observations.schema[name])
        metadata["dtype"] = str(observations.schema[name].dtype)
        values["descriptor"] = metadata
    snapshot["fields"] = {
        name: fields[field_id] for name, field_id in observations.field_ids.items()
    }
    return snapshot
