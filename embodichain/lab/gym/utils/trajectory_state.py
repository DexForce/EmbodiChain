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

from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
from uuid import uuid4
from typing import Any, Protocol, TypeAlias, TYPE_CHECKING
from dataclasses import asdict

import numpy as np
import torch
from tensordict import TensorDictBase

from embodichain.lab.evaluation.observations import _raw_field_kind
from embodichain.utils import logger
from embodichain.data_pipeline.recording.media import (
    MediaFrameReader,
    MediaSpool,
    verify_media_manifest,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.managers.cfg import ObservationCfg
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.task_program.integrations import (
        SimulationTaskProgramRegistration,
    )

__all__ = [
    "FluidTrajectoryRecorder",
    "FluidTrajectoryReader",
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


_MAX_BUFFER_BYTES = 32 * 1024 * 1024


class FluidTrajectoryRecorder:
    """Spool one particle-state frame per recorded control boundary.

    Args:
        directory: Parent directory for this attempt's unique sidecars.
        num_envs: Source environment row count.
        step_dt: Control period in seconds.
        enabled_env_ids: Optional rows eligible for recording.

    Each row uses the shared media spool with a 32 MiB buffer. Particle positions
    are stored in arena-local coordinates; publication and commit are separate.
    """

    def __init__(
        self,
        directory: Path,
        num_envs: int,
        step_dt: float,
        *,
        enabled_env_ids: Sequence[int] | None = None,
    ) -> None:
        self.directory = directory / f"fluid-state-{uuid4().hex}"
        self.num_envs = num_envs
        self.step_dt = step_dt
        enabled = (
            tuple(range(num_envs))
            if enabled_env_ids is None
            else tuple(enabled_env_ids)
        )
        if (
            not enabled
            or any(type(index) is not int for index in enabled)
            or len(set(enabled)) != len(enabled)
            or any(not 0 <= index < num_envs for index in enabled)
        ):
            raise ValueError("Fluid recording must select unique valid rows.")
        self._enabled = frozenset(enabled)
        self._spools: list[MediaSpool | None] = [None] * num_envs
        self._parent_ids = [str(uuid4()) for _ in range(num_envs)]
        self._counts = [0] * num_envs
        self._published = False
        self._committed = False

    def capture(self, env: Any, env_ids: Sequence[int], *, reset: bool = False) -> None:
        """Capture selected rows, including the reset frame when requested."""
        if self._published:
            raise RuntimeError(
                "A finalized fluid trajectory cannot capture more frames."
            )
        ids = tuple(index for index in env_ids if index in self._enabled)
        if not ids:
            return
        if reset:
            for index in ids:
                if self._spools[index] is not None:
                    shutil.rmtree(self._spools[index].directory)
                    self._spools[index] = None
                self._parent_ids[index] = str(uuid4())
                self._counts[index] = 0
        elif any(self._spools[index] is None for index in ids):
            raise RuntimeError("Fluid trajectory has no initial control frame.")
        offsets = env.sim.arena_offsets[list(ids)].detach().cpu().numpy()
        frames = {}
        for uid in sorted(fluid.uid for fluid in env.cfg.fluid_object):
            state = env.sim.get_fluid_object(uid).data.particle_state_w
            if state.dtype != torch.float32 or state.ndim != 3 or state.shape[-1] != 6:
                raise ValueError("Recorded fluid state must be float32 (E, N, 6).")
            values = state[list(ids)].detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Recorded fluid state must be finite.")
            frames[uid] = values
        for row_position, index in enumerate(ids):
            if self._spools[index] is None:
                self._spools[index] = MediaSpool(
                    self.directory / f"env_{index:03d}",
                    max_buffer_bytes=_MAX_BUFFER_BYTES,
                    parent_episode_id=self._parent_ids[index],
                    calibration={"coordinate_frame": "arena_local", "channels": 6},
                )
            row = {}
            for uid, values in frames.items():
                local = values[row_position].copy()
                local[:, :3] -= offsets[row_position]
                row[uid] = local
            self._spools[index].append(self._counts[index] * self.step_dt, row)
            self._counts[index] += 1

    def finish(
        self,
        trajectory_path: Path,
        lengths: Sequence[int],
        *,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Publish selected, verified sidecars before the trajectory references them."""
        if self._published:
            raise RuntimeError("Fluid trajectory was already published.")
        if len(lengths) != self.num_envs:
            raise ValueError("Fluid trajectory row count differs from source.")
        selected = tuple(range(self.num_envs)) if env_ids is None else tuple(env_ids)
        if (
            not selected
            or any(type(index) is not int for index in selected)
            or len(set(selected)) != len(selected)
            or any(not 0 <= index < self.num_envs for index in selected)
        ):
            raise ValueError("Fluid trajectory must select unique valid rows.")
        manifests = []
        for index in selected:
            spool = self._spools[index]
            length = lengths[index]
            if spool is None or self._counts[index] != int(length) + 1:
                raise ValueError("Fluid frames do not align with trajectory lengths.")
            path = spool.finish()
            verify_media_manifest(
                path,
                expected_frames=self._counts[index],
                parent_episode_id=self._parent_ids[index],
                expected_timestamps=np.arange(self._counts[index]) * self.step_dt,
            )
            manifests.append(str(path.relative_to(trajectory_path.parent)))
        for index in set(range(self.num_envs)) - set(selected):
            spool = self._spools[index]
            if spool is not None:
                shutil.rmtree(spool.directory)
                self._spools[index] = None
        self._published = True
        return {
            "schema_version": 1,
            "coordinate_frame": "arena_local",
            "manifests": manifests,
            "parent_ids": [self._parent_ids[index] for index in selected],
        }

    def commit(self) -> None:
        """Retain sidecars after their referring trajectory is durable."""
        if not self._published:
            raise RuntimeError("Fluid sidecars must be finalized before commit.")
        self._committed = True

    def abort(self) -> None:
        """Discard uncommitted particle chunks after failure or interruption."""
        if not self._committed:
            shutil.rmtree(self.directory, ignore_errors=True)


class FluidTrajectoryReader:
    """Validate and stream particle frames independently of collection policy.

    Args:
        path: Referring trajectory file, which anchors relative sidecar paths.
        trajectory: Loaded trajectory metadata with particle manifests.
        cfg: Target configuration defining environment count and fluid sources.

    Content, timing, shape and frame conventions are checked before replay.
    The host separately decides which recording purposes or task outcomes it accepts.
    """

    def __init__(self, path: str | Path, trajectory: Mapping, cfg: Any) -> None:
        self.path = Path(path).resolve()
        meta = trajectory["meta"]
        source_envs = int(meta["num_envs"])
        target_envs = int(cfg.num_envs)
        if source_envs not in (1, target_envs):
            raise ValueError(
                "Fluid replay source must match env count or contain one row."
            )
        sidecar = meta.get("fluid_state")
        if (
            not isinstance(sidecar, Mapping)
            or sidecar.get("schema_version") != 1
            or sidecar.get("coordinate_frame") != "arena_local"
        ):
            raise ValueError("Fluid replay requires aligned particle-state sidecars.")
        paths = sidecar.get("manifests")
        parent_ids = sidecar.get("parent_ids")
        if not isinstance(paths, list) or not isinstance(parent_ids, list):
            raise ValueError("Fluid replay sidecar manifest is malformed.")
        if len(paths) != source_envs or len(parent_ids) != source_envs:
            raise ValueError("Fluid replay sidecar rows do not match the trajectory.")
        self._uids = tuple(sorted(fluid.uid for fluid in cfg.fluid_object))
        if not self._uids:
            raise ValueError("Fluid replay requires configured fluid objects.")
        expected_counts = {
            fluid.uid: int(fluid.spawn.sample(fluid.particle_spacing).shape[0])
            for fluid in cfg.fluid_object
        }
        self._readers = []
        try:
            for index, (name, parent_id) in enumerate(zip(paths, parent_ids)):
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(parent_id, str)
                    or not parent_id
                ):
                    raise ValueError(
                        "Fluid replay manifest paths and IDs must be strings."
                    )
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        "Fluid replay manifest escapes its artifact directory."
                    )
                candidate = self.path.parent / relative
                if any(
                    part.is_symlink()
                    for part in (candidate, *candidate.parents)
                    if part.is_relative_to(self.path.parent)
                ):
                    raise ValueError("Fluid replay manifest cannot use symlinks.")
                manifest_path = candidate.resolve()
                if not manifest_path.is_relative_to(self.path.parent):
                    raise ValueError(
                        "Fluid replay manifest escapes its artifact directory."
                    )
                count = int(meta["lengths"][index]) + 1
                declared = verify_media_manifest(
                    manifest_path,
                    expected_frames=count,
                    parent_episode_id=parent_id,
                    expected_timestamps=np.arange(count) * float(meta["dt"]),
                )
                if declared.get("calibration") != {
                    "coordinate_frame": "arena_local",
                    "channels": 6,
                }:
                    raise ValueError("Fluid replay particle frame convention differs.")
                if tuple(sorted(declared["features"])) != self._uids:
                    raise ValueError(
                        "Fluid replay particle UIDs differ from the scene."
                    )
                for uid, schema in declared["features"].items():
                    if (
                        len(schema["shape"]) != 2
                        or schema["shape"][0] != expected_counts[uid]
                        or schema["shape"][1] != 6
                        or np.dtype(schema["dtype"]) != np.dtype("float32")
                    ):
                        raise ValueError("Fluid replay requires float32 (N, 6) frames.")
                self._readers.append(MediaFrameReader(manifest_path))
        except Exception:
            self.close()
            raise
        self._source_envs = source_envs
        self._target_envs = target_envs

    def apply(self, env: Any, steps: torch.Tensor) -> None:
        """Write selected control-boundary frames without advancing physics."""
        if tuple(steps.shape) != (self._target_envs,):
            raise ValueError("Fluid replay step indices do not match target rows.")
        indices = [int(value) for value in steps.detach().cpu().tolist()]
        offsets = env.sim.arena_offsets
        selected: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for row, step in enumerate(indices):
            source = 0 if self._source_envs == 1 else row
            key = source, step
            if key not in selected:
                _, selected[key] = self._readers[source].read(step)
        for uid in self._uids:
            fluid = env.sim.get_fluid_object(uid)
            values = []
            for row, step in enumerate(indices):
                source = 0 if self._source_envs == 1 else row
                state = selected[source, step][uid]
                if (
                    state.shape != (fluid.num_particles, 6)
                    or not np.isfinite(state).all()
                ):
                    raise ValueError("Fluid replay particle count or state is invalid.")
                values.append(state)
            world = torch.as_tensor(np.stack(values), device=env.device).clone()
            world[:, :, :3] += offsets[:, None, :]
            fluid.write_particle_state_to_sim(world)

    def close(self) -> None:
        for reader in getattr(self, "_readers", ()):
            reader.close()
