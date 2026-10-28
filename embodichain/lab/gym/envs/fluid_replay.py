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

"""One-way fluid Gym replay of an accepted Task Program recording."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
import math
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import load_trajectory
from embodichain.lab.sim.cfg import NewtonPhysicsCfg

__all__ = ["FluidReplayEnv"]


class FluidReplayEnv(EmbodiedEnv):
    """Replay measured rigid states while continuing to solve fluid physics.

    The source must have been generated and accepted by Task Program. This is
    a replay/evaluation environment: policy actions do not control the robot.
    Gym reset/step, sensors and configured observation/reward managers still
    execute normally. No rigid-fluid force feedback is modeled.

    Args:
        cfg: Physical environment configuration, including fluid declarations.
        trajectory: Accepted source recording (path or loaded trajectory dict).
        speed: Positive source-time multiplier.
        hold_seconds: Initial and final stationary intervals in simulation time.
        kwargs: Forwarded environment constructor options.
    """

    def __init__(
        self,
        cfg: EmbodiedEnvCfg,
        *,
        trajectory: str | dict,
        speed: float = 2.0,
        hold_seconds: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if (
            not math.isfinite(speed)
            or speed <= 0
            or not math.isfinite(hold_seconds)
            or hold_seconds < 0
        ):
            raise ValueError(
                "Replay speed must be positive and hold_seconds nonnegative."
            )
        if cfg.record_trajectory or cfg.actions is not None:
            raise ValueError(
                "Fluid replay does not support action managers or trajectory/action recording."
            )
        if (
            not cfg.filter_dataset_saving
            and cfg.dataset is not None
            and any(
                not key.startswith("_")
                for key in (
                    cfg.dataset
                    if isinstance(cfg.dataset, Mapping)
                    else vars(cfg.dataset)
                )
            )
        ):
            raise ValueError(
                "Fluid replay does not support policy dataset recording; record measured fluid diagnostics separately."
            )
        self._source = load_trajectory(
            deepcopy(trajectory) if isinstance(trajectory, dict) else trajectory
        )
        meta = self._source["meta"]
        qualification = meta.get("fluid_source", {})
        if (
            qualification.get("kind") != "task_program"
            or qualification.get("accepted") is not True
            or not isinstance(qualification.get("program_hash"), str)
            or not qualification["program_hash"]
        ):
            raise ValueError(
                "Fluid replay requires a Task Program recording with accepted fluid_source qualification."
            )
        if int(meta["num_envs"]) != 1:
            raise ValueError(
                "The initial fluid replay adapter broadcasts one qualified source row."
            )
        count = int(meta["lengths"][0])
        dt = float(meta["dt"])
        if count < 2 or not math.isfinite(dt) or dt <= 0:
            raise ValueError(
                "Replay needs at least two source frames at a finite positive cadence."
            )
        self._source_times = np.arange(count, dtype=float) * dt
        self._speed = speed
        self._hold_seconds = hold_seconds
        self.replay_duration = 2 * hold_seconds + self._source_times[-1] / speed
        cfg = deepcopy(cfg)
        physics = cfg.sim_cfg.physics_cfg
        solver = getattr(physics, "solver_cfg", None)
        solver_type = (
            solver.get("solver_type")
            if isinstance(solver, dict)
            else getattr(solver, "solver_type", None)
        )
        if not isinstance(physics, NewtonPhysicsCfg) or solver_type != "dexuni":
            raise ValueError("Fluid replay requires Newton DexUni with DFSPH.")
        if cfg.sim_cfg.physics_cfg.num_substeps != 1:
            raise ValueError(
                "Fluid Gym replay requires num_substeps=1; choose physics_dt for boundary cadence."
            )
        states = self._source["states"]
        joints = states["robot", "qpos"][0, :count].cpu().numpy()
        self._joint_curve = PchipInterpolator(self._source_times, joints)
        cfg.robot.init_qpos = joints[0].tolist()
        roots = states["robot", "root_pose"][0, :count].cpu().numpy()
        root_rotations = Rotation.from_quat(roots[:, 3:]).as_matrix()
        if not np.allclose(roots[:, :3], roots[0, :3], atol=1e-5) or not np.allclose(
            root_rotations, root_rotations[0], atol=1e-5
        ):
            raise ValueError(
                "Fluid replay currently supports fixed-root robot recordings only."
            )
        root_pose = np.eye(4, dtype=np.float32)
        root_pose[:3, :3] = root_rotations[0]
        root_pose[:3, 3] = roots[0, :3]
        cfg.robot.init_local_pose = root_pose

        self._rigid_curves = {}
        objects = {item.uid: item for item in cfg.background + cfg.rigid_object}
        for uid in meta.get("rigid_object_uids", []):
            if uid not in objects:
                continue
            if objects[uid].body_type != "kinematic":
                raise ValueError(f"Replayed rigid object {uid!r} must be kinematic.")
            poses = states["rigid_objects", uid, "pose"][0, :count].cpu().numpy()
            self._rigid_curves[uid] = (
                PchipInterpolator(self._source_times, poses[:, :3]),
                Slerp(self._source_times, Rotation.from_quat(poses[:, 3:])),
            )
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = Rotation.from_quat(poses[0, 3:]).as_matrix()
            pose[:3, 3] = poses[0, :3]
            objects[uid].init_local_pose = pose
        if not self._rigid_curves:
            raise ValueError("No recorded rigid objects match the replay scene.")
        self._replay_time = np.zeros(cfg.num_envs, dtype=np.float64)
        self._previous_rigid = {}
        super().__init__(cfg, **kwargs)
        try:
            if joints.shape[1] != self.robot.dof:
                raise ValueError(
                    "Source robot DOF count does not match the replay embodiment."
                )
            if "active_joint_ids" in meta and list(meta["active_joint_ids"]) != list(
                self.active_joint_ids
            ):
                raise ValueError(
                    "Source active joint order does not match the replay embodiment."
                )
            if "joint_names" in meta:
                expected = [self.robot.joint_names[i] for i in self.active_joint_ids]
                if list(meta["joint_names"]) != expected:
                    raise ValueError(
                        "Source joint names do not match the replay embodiment."
                    )
        except Exception:
            self._cleanup_failed_initialization()
            raise
        self._replay_no_auto_reset = True

    def step(
        self, action: Any, **kwargs: Any
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance replay; reset is required after every row is exhausted."""
        if np.all(self._replay_time >= self.replay_duration):
            raise RuntimeError(
                "Fluid replay is exhausted; reset before stepping again."
            )
        return super().step(action, **kwargs)

    def check_truncated(self, obs: Any, info: dict) -> torch.Tensor:
        """Signal each row's replay exhaustion independently."""
        return super().check_truncated(obs, info) | torch.as_tensor(
            self._replay_time >= self.replay_duration, device=self.device
        )

    def _step_action(self, action: Any) -> Any:
        """Preserve the Gym lifecycle while motion comes from recorded states."""
        return action

    def _apply_replay_state(self, ids: list[int], dt: float) -> None:
        if not ids:
            return
        source_times = np.clip(
            (self._replay_time[ids] - self._hold_seconds) * self._speed,
            0.0,
            self._source_times[-1],
        )
        qpos = torch.as_tensor(
            self._joint_curve(source_times), dtype=torch.float32, device=self.device
        )
        self.robot.set_qpos(qpos, env_ids=ids, target=False)
        active = (self._replay_time[ids] >= self._hold_seconds) & (
            self._replay_time[ids]
            < self._hold_seconds + self._source_times[-1] / self._speed
        )
        qvel = (
            self._joint_curve.derivative()(source_times) * self._speed * active[:, None]
        )
        self.robot.set_qvel(
            torch.as_tensor(qvel, dtype=torch.float32, device=self.device),
            env_ids=ids,
            target=False,
        )
        if not hasattr(self, "_replay_offsets"):
            self._replay_offsets = self.sim.arena_offsets.detach().cpu().numpy().copy()
        offsets = self._replay_offsets[ids]
        for uid, (positions, rotations) in self._rigid_curves.items():
            poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(ids), axis=0)
            poses[:, :3, :3] = rotations(source_times).as_matrix()
            poses[:, :3, 3] = positions(source_times) + offsets
            previous = self._previous_rigid.get(uid)
            if previous is None:
                previous = np.repeat(
                    np.eye(4, dtype=np.float32)[None], self.num_envs, axis=0
                )
                self._previous_rigid[uid] = previous
                previous[ids] = poses
            old = previous[ids].copy()
            linear = (
                (poses[:, :3, 3] - old[:, :3, 3]) / dt
                if dt > 0
                else np.zeros((len(ids), 3))
            )
            angular = (
                Rotation.from_matrix(
                    poses[:, :3, :3] @ old[:, :3, :3].transpose(0, 2, 1)
                ).as_rotvec()
                / dt
                if dt > 0
                else np.zeros((len(ids), 3))
            )
            self.sim.get_rigid_object(uid).set_kinematic_state(
                torch.from_numpy(poses),
                torch.from_numpy(linear),
                torch.from_numpy(angular),
                env_ids=ids,
            )
            previous[ids] = poses

    def _advance_physics(self) -> None:
        sensors = [
            sensor for sensor in self.sensors.values() if sensor.requires_substep_update
        ]
        for sensor in sensors:
            sensor.begin_control_step()
        ids = list(range(self.num_envs))
        for _ in range(self.cfg.sim_steps_per_control):
            self._apply_replay_state(ids, self.physics_dt)
            self.sim.update(step=1, render_final_step=False)
            self._replay_time += self.physics_dt
            for sensor in sensors:
                sensor.update_physics_step(self.physics_dt)

    def _initialize_episode(self, env_ids=None, **kwargs: Any) -> None:
        super()._initialize_episode(env_ids=env_ids, **kwargs)
        ids = (
            list(range(self.num_envs))
            if env_ids is None
            else torch.as_tensor(env_ids).cpu().tolist()
        )
        self._replay_time[ids] = 0.0
        self._apply_replay_state(ids, 0.0)
        # Reset-mode events may have randomized the container; refill only the
        # selected rows from its measured world pose and the local fill template.
        for cfg in self.cfg.fluid_object:
            if cfg.spawn.reference_uid is None:
                continue
            from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
            from embodichain.lab.gym.envs.managers.events import (
                reset_fluid_in_container,
            )

            reset_fluid_in_container(
                self,
                ids,
                [
                    SceneEntityCfg(uid=cfg.uid),
                    SceneEntityCfg(uid=cfg.spawn.reference_uid),
                ],
            )

    @property
    def source_qualification(self) -> dict:
        """Return independent Task Program provenance and source acceptance."""
        return deepcopy(self._source["meta"]["fluid_source"])

    def get_info(self, **kwargs: Any) -> dict:
        """Return normal Gym info with row-local replay time and completion."""
        info = super().get_info(**kwargs)
        info["fluid_replay_time_s"] = torch.as_tensor(
            self._replay_time.copy(), device=self.device
        )
        info["fluid_replay_finished"] = torch.as_tensor(
            self._replay_time >= self.replay_duration, device=self.device
        )
        info["fluid_source_accepted"] = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return info
