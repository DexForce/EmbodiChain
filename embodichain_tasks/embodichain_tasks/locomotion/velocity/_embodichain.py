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

"""EmbodiChain runtime adapter for backend-independent velocity tasks."""

from __future__ import annotations

import math
from dataclasses import MISSING, fields
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.utils.math import quat_apply, quat_apply_inverse

from .._robot import apply_task_joint_drive_properties

__all__ = ["EmbodiChainVelocityEnv"]


def _aggregate_contacts(
    user_ids: torch.Tensor,
    valid: torch.Tensor,
    normal: torch.Tensor,
    tangential_impulse: torch.Tensor,
    normal_impulse: torch.Tensor,
    target_user_ids: torch.Tensor,
    physics_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate signed contact force for selected articulation links."""
    impulse = normal * normal_impulse.unsqueeze(-1) + tangential_impulse
    force = impulse / physics_dt
    actor0 = user_ids[:, :, 0].unsqueeze(-1) == target_user_ids.unsqueeze(1)
    actor1 = user_ids[:, :, 1].unsqueeze(-1) == target_user_ids.unsqueeze(1)
    matches = valid.unsqueeze(-1) & (actor0 | actor1)
    sign = actor1.to(force.dtype) - actor0.to(force.dtype)
    selected_force = (
        force.unsqueeze(2) * sign.unsqueeze(-1) * valid[:, :, None, None]
    ).sum(dim=1)
    return matches.any(dim=1), selected_force


def _count_self_collisions(
    user_ids: torch.Tensor,
    valid: torch.Tensor,
    normal: torch.Tensor,
    tangential_impulse: torch.Tensor,
    normal_impulse: torch.Tensor,
    robot_user_ids: torch.Tensor,
    force_threshold: float,
    physics_dt: float,
) -> torch.Tensor:
    """Mark environments with a force-qualified articulation self-contact."""
    actor0_is_robot = (
        user_ids[:, :, 0].unsqueeze(-1) == robot_user_ids.unsqueeze(1)
    ).any(dim=-1)
    actor1_is_robot = (
        user_ids[:, :, 1].unsqueeze(-1) == robot_user_ids.unsqueeze(1)
    ).any(dim=-1)
    impulse = normal * normal_impulse.unsqueeze(-1) + tangential_impulse
    force = torch.linalg.vector_norm(impulse, dim=-1) / physics_dt
    return (
        (valid & actor0_is_robot & actor1_is_robot & (force > force_threshold))
        .any(dim=-1)
        .to(dtype=torch.float32)
    )


def _site_velocity_b(
    link_pose: torch.Tensor,
    link_velocity: torch.Tensor,
    local_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one link-attached site's linear and angular velocity."""
    world_offset = quat_apply(link_pose[:, 3:7], local_offset)
    linear_velocity_w = link_velocity[:, :3] + torch.cross(
        link_velocity[:, 3:], world_offset, dim=-1
    )
    return (
        quat_apply_inverse(link_pose[:, 3:7], linear_velocity_w),
        quat_apply_inverse(link_pose[:, 3:7], link_velocity[:, 3:]),
    )


def _select_single_command_axis(
    command: torch.Tensor,
    env_ids: torch.Tensor,
    axis: torch.Tensor,
) -> None:
    """Keep one planar command axis for each selected environment."""
    selected = command[env_ids]
    axis_mask = torch.nn.functional.one_hot(axis, num_classes=3).to(selected.dtype)
    command[env_ids] = selected * axis_mask


class EmbodiChainVelocityEnv(EmbodiedEnv):
    """Connect one pure Torch velocity task to EmbodiChain and DexSim state."""

    velocity_task_config: Any = None
    state_type: type[Any]
    build_observations_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    compute_rewards_fn: Callable[..., Any]
    compute_termination_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    corrupt_actor_fn: Callable[..., torch.Tensor] | None = None
    foot_link_names: tuple[str, ...] = ()
    foot_offsets: tuple[tuple[float, float, float], ...] = ()
    illegal_contact_link_names: tuple[str, ...] = ()
    orientation_link_name: str | None = None
    imu_offset: tuple[float, float, float] | None = None
    explicit_pd_effort_control = False

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs) -> None:
        cfg = EmbodiedEnvCfg() if cfg is None else cfg
        if cfg.robot is not MISSING and self.velocity_task_config is not None:
            apply_task_joint_drive_properties(cfg.robot, self.velocity_task_config)
            if self.explicit_pd_effort_control:
                cfg.robot.joint_drive_props.stiffness = 0.0
                cfg.robot.joint_drive_props.damping = 0.0
        super().__init__(cfg, **kwargs)

    def _init_sim_state(self, **kwargs) -> None:
        config = self.velocity_task_config
        if config is None:
            raise RuntimeError("A velocity task configuration is required.")
        missing_joints = [
            name for name in config.joint_names if name not in self.robot.joint_names
        ]
        missing_links = [
            name
            for name in (*self.foot_link_names, *self.illegal_contact_link_names)
            if name not in self.robot.link_names
        ]
        if missing_joints or missing_links:
            raise RuntimeError(
                f"Asset mapping failed; missing joints={missing_joints}, "
                f"missing links={missing_links}."
            )
        if not math.isclose(self.physics_dt, config.physics_dt, abs_tol=1.0e-9):
            raise RuntimeError(
                f"Task physics_dt={config.physics_dt} differs from "
                f"environment physics_dt={self.physics_dt}."
            )
        if not math.isclose(self.step_dt, config.control_dt, abs_tol=1.0e-9):
            raise RuntimeError(
                f"Task control_dt={config.control_dt} differs from "
                f"environment control_dt={self.step_dt}."
            )

        super()._init_sim_state(**kwargs)
        self.add_detached_uids_for_reset([self.robot.uid])
        self.policy_joint_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in config.joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.foot_link_ids = torch.tensor(
            [self.robot.link_names.index(name) for name in self.foot_link_names],
            dtype=torch.long,
            device=self.device,
        )
        orientation_name = self.orientation_link_name or self.robot.link_names[0]
        self.orientation_link_id = self.robot.link_names.index(orientation_name)
        self._imu_offset = (
            None
            if self.imu_offset is None
            else torch.tensor(
                self.imu_offset,
                dtype=torch.float32,
                device=self.device,
            ).expand(self.num_envs, -1)
        )
        self._foot_offsets = torch.tensor(
            self.foot_offsets,
            dtype=torch.float32,
            device=self.device,
        )
        sensor = self.get_sensor("locomotion_contacts")
        actor_ids = {}
        for actor_id in sensor.item_user_ids.cpu().tolist():
            info = sensor.get_actor_info(actor_id)
            actor_ids[info.env_id, info.link_name] = actor_id

        def contact_ids(names: Sequence[str]) -> torch.Tensor:
            return torch.tensor(
                [
                    [actor_ids[env_id, name] for name in names]
                    for env_id in range(self.num_envs)
                ],
                dtype=torch.int32,
                device=self.device,
            ).reshape(self.num_envs, len(names))

        self._foot_user_ids = contact_ids(self.foot_link_names)
        self._illegal_contact_user_ids = contact_ids(self.illegal_contact_link_names)
        self._robot_user_ids = contact_ids(
            tuple(name for name in self.robot.link_names if (0, name) in actor_ids)
        )

        seed = int(self.cfg.seed if self.cfg.seed is not None else 0)
        self._locomotion_generator = torch.Generator(device=self.device)
        self._locomotion_generator.manual_seed(seed)
        action_shape = (self.num_envs, config.action_dim)
        foot_shape = (self.num_envs, len(self.foot_link_names))
        illegal_shape = (self.num_envs, len(self.illegal_contact_link_names))
        self.command = torch.zeros((self.num_envs, 3), device=self.device)
        self.reward_command = torch.zeros_like(self.command)
        self._heading_target = torch.zeros(self.num_envs, device=self.device)
        self._gait_frequency = torch.zeros(self.num_envs, device=self.device)
        self._gait_process = torch.zeros(self.num_envs, device=self.device)
        self._heading_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._standing_env = torch.zeros_like(self._heading_env)
        self._forward_env = torch.zeros_like(self._heading_env)
        self._command_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.locomotion_action = torch.zeros(action_shape, device=self.device)
        self.last_locomotion_action = torch.zeros_like(self.locomotion_action)
        self.encoder_bias = torch.zeros_like(self.locomotion_action)
        bias = config.data.get("events", {}).get("encoder_bias")
        if bias is not None:
            minimum, maximum = bias["params"]["bias_range"]
            self.encoder_bias.uniform_(
                float(minimum),
                float(maximum),
                generator=self._locomotion_generator,
            )
        self._episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._previous_joint_velocity = torch.zeros(action_shape, device=self.device)
        self._joint_acceleration = torch.zeros_like(self._previous_joint_velocity)
        self._previous_root_velocity_w = torch.zeros(
            (self.num_envs, 6), device=self.device
        )
        self._root_acceleration_w = torch.zeros_like(self._previous_root_velocity_w)
        self._filtered_base_lin_vel_b = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._filtered_base_ang_vel_b = torch.zeros_like(self._filtered_base_lin_vel_b)
        self._foot_air_time = torch.zeros(foot_shape, device=self.device)
        self._last_foot_contact = torch.zeros(
            foot_shape, dtype=torch.bool, device=self.device
        )
        self._first_foot_contact = torch.zeros_like(
            self._foot_air_time, dtype=torch.float32
        )
        self._contact_history_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._foot_peak_height = torch.zeros(foot_shape, device=self.device)
        self._substep_illegal_contact_force = torch.zeros(
            illegal_shape, device=self.device
        )
        self._substep_self_collision_count = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_illegal_contact_force_by_body = torch.zeros(
            illegal_shape, device=self.device
        )
        self._global_control_step = 0
        self._state_cache: Any | None = None
        self._reward_cache: Any | None = None
        self._last_termination = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_commands(env_ids)
        self._update_heading_commands()
        self.reward_command.copy_(self.command)
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.action_dim,),
            dtype=np.float32,
        )

    def record_locomotion_action(self, action: torch.Tensor) -> None:
        """Record policy actions before converting them to joint targets."""
        self.last_locomotion_action.copy_(self.locomotion_action)
        self.locomotion_action.copy_(action)

    def velocity_command_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return configured planar and yaw command bounds in SI units."""
        ranges = self.velocity_task_config.data["commands"]["twist"]["ranges"]
        values = np.asarray(
            [ranges["lin_vel_x"], ranges["lin_vel_y"], ranges["ang_vel_z"]],
            dtype=np.float32,
        )
        return values[:, 0], values[:, 1]

    def set_velocity_command(
        self, command: Sequence[float], *, gait_frequency: float | None = None
    ) -> None:
        """Hold one velocity command until reset or the next explicit command.

        Zero velocity defaults to static standing. For a phase-conditioned task,
        an explicit positive ``gait_frequency`` requests stepping in place.
        Moving commands preserve each active gait frequency; stopped rows resume
        at the midpoint of the configured range without resetting gait phase.

        Args:
            command: Target ``(vx, vy, yaw_rate)`` in m/s and rad/s.
            gait_frequency: Optional gait frequency in Hz. Positive values must
                lie in the task's gait range. Zero requires a zero velocity.

        Raises:
            ValueError: The command or gait frequency is invalid for this task.
        """
        values = np.asarray(tuple(command), dtype=np.float32)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Velocity command must contain finite vx, vy, yaw_rate.")
        stopped = bool(np.linalg.norm(values) < 1.0e-5)
        frequency_range = self.velocity_task_config.data["commands"]["twist"].get(
            "gait_frequency_range"
        )
        frequency = None
        if gait_frequency is not None:
            frequency = float(gait_frequency)
            if frequency_range is None:
                raise ValueError("This task does not accept a gait frequency.")
            if not math.isfinite(frequency) or frequency < 0.0:
                raise ValueError("Gait frequency must be finite and nonnegative.")
            if frequency == 0.0 and not stopped:
                raise ValueError("A moving command requires a positive gait frequency.")
            if frequency > 0.0 and not (
                frequency_range[0] <= frequency <= frequency_range[1]
            ):
                raise ValueError("Gait frequency is outside the configured range.")

        self.command.copy_(torch.as_tensor(values, device=self.device).unsqueeze(0))
        self.reward_command.copy_(self.command)
        self._command_steps_remaining.fill_(torch.iinfo(torch.long).max)
        self._heading_env.fill_(False)
        self._forward_env.fill_(False)
        static_standing = stopped and (frequency is None or frequency == 0.0)
        self._standing_env.fill_(static_standing)
        if frequency_range is not None:
            if static_standing:
                self._gait_frequency.zero_()
                self._gait_process.zero_()
            elif frequency is not None:
                self._gait_frequency.fill_(frequency)
            else:
                midpoint = 0.5 * (frequency_range[0] + frequency_range[1])
                self._gait_frequency.copy_(
                    torch.where(
                        self._gait_frequency > 0.0,
                        self._gait_frequency,
                        midpoint,
                    )
                )
        self._invalidate_task_cache()

    def _active_command_ranges(self) -> torch.Tensor:
        command = self.velocity_task_config.data["commands"]["twist"]
        ranges = command["ranges"]
        active = (
            tuple(ranges["lin_vel_x"]),
            tuple(ranges["lin_vel_y"]),
            tuple(ranges["ang_vel_z"]),
        )
        curriculum = self.velocity_task_config.data.get("curriculum", {})
        stages = (
            curriculum.get("command_vel", {})
            .get("params", {})
            .get("velocity_stages", ())
        )
        for stage in sorted(stages, key=lambda item: int(item["step"])):
            if self._global_control_step < int(stage["step"]):
                break
            active = tuple(
                tuple(stage.get(name, ranges[name]))
                for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
            )
        return torch.tensor(active, dtype=torch.float32, device=self.device)

    def _active_single_axis_fraction(self) -> float:
        command = self.velocity_task_config.data["commands"]["twist"]
        fraction = float(command.get("rel_single_axis_envs", 0.0))
        curriculum = self.velocity_task_config.data.get("curriculum", {})
        stages = (
            curriculum.get("single_axis", {})
            .get("params", {})
            .get("fraction_stages", ())
        )
        for stage in sorted(stages, key=lambda item: int(item["step"])):
            if self._global_control_step < int(stage["step"]):
                break
            fraction = float(stage["fraction"])
        return fraction

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = int(env_ids.numel())
        if count == 0:
            return
        ranges = self._active_command_ranges()
        random = torch.rand(
            (count, 3), device=self.device, generator=self._locomotion_generator
        )
        self.command[env_ids] = ranges[:, 0] + random * (ranges[:, 1] - ranges[:, 0])
        command = self.velocity_task_config.data["commands"]["twist"]
        single_axis_fraction = self._active_single_axis_fraction()
        single_axis_mask = (
            torch.rand(count, device=self.device, generator=self._locomotion_generator)
            <= single_axis_fraction
        )
        single_axis_ids = env_ids[single_axis_mask]
        if single_axis_ids.numel() > 0:
            single_axis = torch.randint(
                0,
                3,
                (single_axis_ids.numel(),),
                device=self.device,
                generator=self._locomotion_generator,
            )
            _select_single_command_axis(self.command, single_axis_ids, single_axis)
        minimum, maximum = command["resampling_time_range"]
        seconds = torch.empty(count, device=self.device).uniform_(
            float(minimum),
            float(maximum),
            generator=self._locomotion_generator,
        )
        self._command_steps_remaining[env_ids] = (
            torch.ceil(seconds / self.step_dt).to(torch.long).clamp_min(1)
        )
        heading_range = command["ranges"].get("heading")
        if bool(command.get("heading_command")) and heading_range is not None:
            self._heading_target[env_ids] = torch.empty(
                count, device=self.device
            ).uniform_(
                float(heading_range[0]),
                float(heading_range[1]),
                generator=self._locomotion_generator,
            )
            self._heading_env[env_ids] = torch.rand(
                count, device=self.device, generator=self._locomotion_generator
            ) <= float(command.get("rel_heading_envs", 0.0))
        else:
            self._heading_env[env_ids] = False
        self._standing_env[env_ids] = torch.rand(
            count, device=self.device, generator=self._locomotion_generator
        ) <= float(command.get("rel_standing_envs", 0.0))
        gait_frequency_range = command.get("gait_frequency_range")
        if gait_frequency_range is None:
            self._gait_frequency[env_ids] = 0.0
        else:
            self._gait_frequency[env_ids] = torch.empty(
                count, device=self.device
            ).uniform_(
                float(gait_frequency_range[0]),
                float(gait_frequency_range[1]),
                generator=self._locomotion_generator,
            )
            self._gait_frequency[env_ids[self._standing_env[env_ids]]] = 0.0
        self._forward_env[env_ids] = torch.rand(
            count, device=self.device, generator=self._locomotion_generator
        ) <= float(command.get("rel_forward_envs", 0.0))
        forward_ids = env_ids[self._forward_env[env_ids]]
        if forward_ids.numel() > 0:
            self.command[forward_ids, 0] = (
                self.command[forward_ids, 0].abs().clamp_min(0.3)
            )
            self.command[forward_ids, 1:] = 0.0

    def _update_heading_commands(self) -> None:
        command = self.velocity_task_config.data["commands"]["twist"]
        quaternion = self.robot.body_data.root_pose[:, 3:7]
        x, y, z, w = quaternion.unbind(dim=-1)
        heading = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y.square() + z.square()),
        )
        error = self._heading_target - heading
        error = torch.atan2(torch.sin(error), torch.cos(error))
        yaw_range = self._active_command_ranges()[2]
        yaw = (float(command.get("heading_control_stiffness", 0.0)) * error).clamp(
            min=float(yaw_range[0]), max=float(yaw_range[1])
        )
        self.command[self._heading_env, 2] = yaw[self._heading_env]
        self.command[self._standing_env] = 0.0

    def _update_sim_state(self, **kwargs) -> None:
        super()._update_sim_state(**kwargs)
        velocity = self.robot.get_qvel()[:, self.policy_joint_ids]
        self._joint_acceleration.copy_(
            (velocity - self._previous_joint_velocity) / self.step_dt
        )
        self._previous_joint_velocity.copy_(velocity)
        root_quaternion = self.robot.body_data.root_pose[:, 3:7]
        root_lin_vel_w = self.robot.body_data.root_lin_vel
        root_ang_vel_w = self.robot.body_data.root_ang_vel
        root_velocity_w = torch.cat((root_lin_vel_w, root_ang_vel_w), dim=-1)
        self._root_acceleration_w.copy_(
            (root_velocity_w - self._previous_root_velocity_w) / self.step_dt
        )
        self._previous_root_velocity_w.copy_(root_velocity_w)
        base_lin_vel_b = quat_apply_inverse(root_quaternion, root_lin_vel_w)
        base_ang_vel_b = quat_apply_inverse(root_quaternion, root_ang_vel_w)
        filter_weight = float(
            self.velocity_task_config.data.get("velocity_filter_weight", 1.0)
        )
        self._filtered_base_lin_vel_b.lerp_(base_lin_vel_b, filter_weight)
        self._filtered_base_ang_vel_b.lerp_(base_ang_vel_b, filter_weight)
        self._episode_step += 1
        self._global_control_step += 1
        self._gait_process.add_(self.step_dt * self._gait_frequency).remainder_(1.0)
        self.reward_command.copy_(self.command)
        self._command_steps_remaining.sub_(1)
        resample_ids = (
            (self._command_steps_remaining <= 0).nonzero(as_tuple=False).squeeze(-1)
        )
        self._resample_commands(resample_ids)
        self._update_heading_commands()
        self._invalidate_task_cache()

    def _initialize_episode(
        self,
        env_ids: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        super()._initialize_episode(env_ids=env_ids, **kwargs)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        count = int(ids.numel())
        if count == 0:
            return
        data = self.velocity_task_config.data
        root = data["robot"]
        pose_range = data["events"]["reset_base"]["params"]["pose_range"]
        pose = torch.zeros((count, 7), dtype=torch.float32, device=self.device)
        pose[:, 0].uniform_(
            *map(float, pose_range["x"]), generator=self._locomotion_generator
        )
        pose[:, 1].uniform_(
            *map(float, pose_range["y"]), generator=self._locomotion_generator
        )
        root_height = float(
            root["root_height"] if "root_height" in root else root["root_position"][2]
        )
        pose[:, 2].uniform_(
            root_height + float(pose_range["z"][0]),
            root_height + float(pose_range["z"][1]),
            generator=self._locomotion_generator,
        )
        yaw = torch.empty(count, device=self.device).uniform_(
            *map(float, pose_range["yaw"]), generator=self._locomotion_generator
        )
        pose[:, 6] = torch.cos(0.5 * yaw)
        pose[:, 5] = torch.sin(0.5 * yaw)
        self.robot.clear_dynamics(env_ids=ids)
        self.robot.set_local_pose(pose, env_ids=ids)
        joint_position = torch.as_tensor(
            self.velocity_task_config.default_joint_position,
            dtype=torch.float32,
            device=self.device,
        ).expand(count, -1)
        self.robot.set_qpos(
            joint_position,
            joint_ids=self.policy_joint_ids,
            env_ids=ids,
            target=False,
        )
        self.robot.set_qpos(
            joint_position,
            joint_ids=self.policy_joint_ids,
            env_ids=ids,
            target=True,
        )
        self.locomotion_action[ids] = 0.0
        self.last_locomotion_action[ids] = 0.0
        self._episode_step[ids] = 0
        self._previous_joint_velocity[ids] = 0.0
        self._joint_acceleration[ids] = 0.0
        self._previous_root_velocity_w[ids] = 0.0
        self._root_acceleration_w[ids] = 0.0
        self._filtered_base_lin_vel_b[ids] = 0.0
        self._filtered_base_ang_vel_b[ids] = 0.0
        self._foot_air_time[ids] = 0.0
        self._last_foot_contact[ids] = False
        self._first_foot_contact[ids] = 0.0
        self._contact_history_step[ids] = 0
        self._foot_peak_height[ids] = 0.0
        self._substep_illegal_contact_force[ids] = 0.0
        self._substep_self_collision_count[ids] = 0.0
        self._last_illegal_contact_force_by_body[ids] = 0.0
        self._resample_commands(ids)
        self._update_heading_commands()
        self.reward_command[ids] = self.command[ids]
        self._invalidate_task_cache()

    def _invalidate_task_cache(self) -> None:
        self._state_cache = None
        self._reward_cache = None

    def _read_contacts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor = self.get_sensor("locomotion_contacts")
        if getattr(sensor, "supports_body_contact_reduction", False):
            foot_contact = sensor.get_body_contact_found(
                self._foot_user_ids, ground_only=True
            )
            foot_force = sensor.get_body_contact_force(
                self._foot_user_ids, ground_only=True
            )
            illegal_force_by_body = sensor.get_body_contact_force(
                self._illegal_contact_user_ids, peak=True
            )
            self._last_illegal_contact_force_by_body.copy_(illegal_force_by_body)
            return foot_contact, foot_force, illegal_force_by_body

        contact = sensor.get_data()
        args = (
            contact["user_ids"],
            contact["is_valid"],
            contact["normal"],
            contact["friction"],
            contact["impulse"],
        )
        foot_contact, foot_force = _aggregate_contacts(
            *args,
            self._foot_user_ids,
            self.physics_dt,
        )
        _, illegal_force = _aggregate_contacts(
            *args,
            self._illegal_contact_user_ids,
            self.physics_dt,
        )
        illegal_force_by_body = torch.linalg.vector_norm(illegal_force, dim=-1)
        illegal_force_by_body = torch.maximum(
            illegal_force_by_body, self._substep_illegal_contact_force
        )
        self._last_illegal_contact_force_by_body.copy_(illegal_force_by_body)
        return foot_contact, foot_force, illegal_force_by_body

    def _advance_physics(self) -> None:
        """Collect contacts after each substep of the control interval."""
        sensor = self.get_sensor("locomotion_contacts")
        if not sensor.cfg.track_substeps:
            super()._advance_physics()
            return
        for substep_index in range(self.cfg.sim_steps_per_control):
            self.sim.update(self.physics_dt, 1)
            self._capture_contact_substep(substep_index)

    def _capture_contact_substep(self, substep_index: int) -> None:
        sensor = self.get_sensor("locomotion_contacts")
        if getattr(sensor, "supports_body_contact_reduction", False):
            if substep_index == 0:
                sensor.begin_contact_history()
            sensor.update_body_contacts()
            return

        if substep_index == 0:
            self._substep_illegal_contact_force.zero_()
            self._substep_self_collision_count.zero_()
        sensor.update()
        contact = sensor.get_data()
        _, illegal_force = _aggregate_contacts(
            contact["user_ids"],
            contact["is_valid"],
            contact["normal"],
            contact["friction"],
            contact["impulse"],
            self._illegal_contact_user_ids,
            self.physics_dt,
        )
        torch.maximum(
            self._substep_illegal_contact_force,
            torch.linalg.vector_norm(illegal_force, dim=-1),
            out=self._substep_illegal_contact_force,
        )
        threshold = sensor.cfg.self_collision_force_threshold
        if threshold is not None:
            self_collision_count = _count_self_collisions(
                contact["user_ids"],
                contact["is_valid"],
                contact["normal"],
                contact["friction"],
                contact["impulse"],
                self._robot_user_ids,
                threshold,
                self.physics_dt,
            )
            self._substep_self_collision_count.add_(self_collision_count)

    def _read_self_collision_count(self) -> torch.Tensor:
        sensor = self.get_sensor("locomotion_contacts")
        if getattr(sensor, "supports_self_collision_reduction", False):
            return sensor.get_self_collision_count()
        return self._substep_self_collision_count

    def _advance_contact_history(self, foot_contact: torch.Tensor) -> None:
        advance = self._episode_step > self._contact_history_step
        self._first_foot_contact.zero_()
        if not advance.any():
            return
        first = foot_contact & ~self._last_foot_contact
        self._first_foot_contact[advance] = first[advance].to(
            self._first_foot_contact.dtype
        )
        self._foot_air_time[advance] = torch.where(
            foot_contact[advance],
            torch.zeros_like(self._foot_air_time[advance]),
            self._foot_air_time[advance] + self.step_dt,
        )
        self._last_foot_contact[advance] = foot_contact[advance]
        self._contact_history_step[advance] = self._episode_step[advance]

    def _orbital_angular_momentum(
        self,
        link_position: torch.Tensor,
        link_linear_velocity: torch.Tensor,
    ) -> torch.Tensor:
        masses = self.robot.default_link_masses.to(self.device)
        total_mass = masses.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        center = (link_position * masses.unsqueeze(-1)).sum(dim=1) / total_mass
        center_velocity = (link_linear_velocity * masses.unsqueeze(-1)).sum(
            dim=1
        ) / total_mass
        relative_position = link_position - center.unsqueeze(1)
        linear_momentum = masses.unsqueeze(-1) * (
            link_linear_velocity - center_velocity.unsqueeze(1)
        )
        return torch.cross(relative_position, linear_momentum, dim=-1).sum(dim=1)

    def _common_state(self) -> dict[str, torch.Tensor]:
        root_pose = self.robot.body_data.root_pose
        root_quaternion = root_pose[:, 3:7]
        root_lin_vel_w = self.robot.body_data.root_lin_vel
        root_ang_vel_w = self.robot.body_data.root_ang_vel
        gravity_w = torch.zeros_like(root_lin_vel_w)
        gravity_w[:, 2] = -1.0
        link_pose = self.robot.body_data.body_link_pose
        link_velocity = self.robot.body_data.body_link_vel
        selected_pose = link_pose[:, self.foot_link_ids]
        selected_velocity = link_velocity[:, self.foot_link_ids]
        local_offset = self._foot_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        world_offset = quat_apply(selected_pose[:, :, 3:7], local_offset)
        foot_position = selected_pose[:, :, :3] + world_offset
        foot_velocity = selected_velocity[:, :, :3] + torch.cross(
            selected_velocity[:, :, 3:], world_offset, dim=-1
        )
        foot_contact, foot_force, illegal_force_by_body = self._read_contacts()
        self._advance_contact_history(foot_contact)
        self._foot_peak_height.copy_(
            torch.where(
                ~foot_contact,
                torch.maximum(self._foot_peak_height, foot_position[:, :, 2]),
                self._foot_peak_height,
            )
        )
        target_height = float(
            self.velocity_task_config.data.get("rewards", {})
            .get("foot_swing_height", {})
            .get("params", {})
            .get("target_height", 1.0)
        )
        foot_swing_height_cost = (
            torch.square(self._foot_peak_height / target_height - 1.0)
            * self._first_foot_contact
        )
        self._foot_peak_height.copy_(
            torch.where(
                self._first_foot_contact.bool(),
                torch.zeros_like(self._foot_peak_height),
                self._foot_peak_height,
            )
        )
        limits = self.robot.body_data.qpos_limits[0, self.policy_joint_ids]
        limit_factor = float(
            self.velocity_task_config.data["robot"].get(
                "soft_joint_position_limit_factor", 1.0
            )
        )
        midpoint = limits.mean(dim=-1)
        half_range = 0.5 * (limits[:, 1] - limits[:, 0]) * limit_factor
        orientation_pose = link_pose[:, self.orientation_link_id]
        orientation_velocity = link_velocity[:, self.orientation_link_id]
        base_lin_vel_b = quat_apply_inverse(root_quaternion, root_lin_vel_w)
        base_ang_vel_b = quat_apply_inverse(root_quaternion, root_ang_vel_w)
        if self._imu_offset is not None:
            base_lin_vel_b, base_ang_vel_b = _site_velocity_b(
                orientation_pose,
                orientation_velocity,
                self._imu_offset,
            )
        return {
            "base_lin_vel_b": base_lin_vel_b,
            "base_ang_vel_b": base_ang_vel_b,
            "reward_base_lin_vel_b": self._filtered_base_lin_vel_b,
            "reward_base_ang_vel_b": self._filtered_base_ang_vel_b,
            "projected_gravity_b": quat_apply_inverse(root_quaternion, gravity_w),
            "command": self.command,
            "reward_command": self.reward_command,
            "gait_frequency": self._gait_frequency,
            "gait_process": self._gait_process,
            "joint_pos": self.robot.get_qpos()[:, self.policy_joint_ids],
            "joint_vel": self.robot.get_qvel()[:, self.policy_joint_ids],
            "joint_acc": self._joint_acceleration,
            "joint_torque": self.robot.get_qf()[:, self.policy_joint_ids],
            "action": self.locomotion_action,
            "last_action": self.last_locomotion_action,
            "episode_step": self._episode_step,
            "root_height": root_pose[:, 2],
            "root_acceleration_w": self._root_acceleration_w,
            "base_quat_w": root_quaternion,
            "foot_pos_w": foot_position,
            "foot_quat_w": selected_pose[:, :, 3:7],
            "foot_height": foot_position[:, :, 2],
            "foot_vel_w": foot_velocity,
            "foot_contact": foot_contact,
            "foot_force_w": foot_force,
            "foot_air_time": self._foot_air_time,
            "first_foot_contact": self._first_foot_contact,
            "soft_joint_lower": (midpoint - half_range).unsqueeze(0),
            "soft_joint_upper": (midpoint + half_range).unsqueeze(0),
            "reward_body_ang_vel_w": orientation_velocity[:, 3:],
            "angular_momentum_w": self._orbital_angular_momentum(
                link_pose[:, :, :3], link_velocity[:, :, :3]
            ),
            "illegal_contact_force": (
                illegal_force_by_body.amax(dim=-1)
                if illegal_force_by_body.shape[-1]
                else torch.zeros(self.num_envs, device=self.device)
            ),
            "illegal_contact_force_by_body": illegal_force_by_body,
            "orientation_projected_gravity_b": quat_apply_inverse(
                orientation_pose[:, 3:7], gravity_w
            ),
            "self_collision_count": self._read_self_collision_count(),
            "encoder_bias": self.encoder_bias,
            "foot_swing_height_cost": foot_swing_height_cost,
        }

    def _make_task_state(self, common: dict[str, torch.Tensor]) -> Any:
        values: dict[str, Any] = {}
        for field in fields(self.state_type):
            if field.name in common:
                values[field.name] = common[field.name]
            elif field.default is not MISSING:
                values[field.name] = field.default
            else:
                raise KeyError(f"Task state field {field.name!r} has no adapter value.")
        return self.state_type(**values)

    def get_velocity_locomotion_state(self) -> Any:
        """Return the current physical state in the task's tensor schema."""
        if self._state_cache is None:
            self._state_cache = self._make_task_state(self._common_state())
        return self._state_cache

    def build_velocity_locomotion_observations(
        self,
        *,
        enable_corruption: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build actor and asymmetric critic observations."""
        actor, critic = self.build_observations_fn(
            self.velocity_task_config, self.get_velocity_locomotion_state()
        )
        if enable_corruption and self.corrupt_actor_fn is not None:
            actor = actor.clone()
            actor = self.corrupt_actor_fn(actor, self._locomotion_generator)
        return actor, critic

    def get_velocity_locomotion_reward(self, info: dict[str, Any]) -> Any:
        """Return raw, weighted, and total task rewards for this transition."""
        if self._reward_cache is None:
            terminated = torch.as_tensor(
                info["fail"], dtype=torch.bool, device=self.device
            )
            self._reward_cache = self.compute_rewards_fn(
                self.velocity_task_config,
                self.get_velocity_locomotion_state(),
                terminated,
            )
        return self._reward_cache

    def get_velocity_locomotion_reward_terms(
        self, info: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Return raw task reward terms for diagnostics."""
        return self.get_velocity_locomotion_reward(info).raw_terms

    def compute_task_state(
        self, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute task failures and compact velocity diagnostics."""
        del kwargs
        state = self.get_velocity_locomotion_state()
        terminated, _ = self.compute_termination_fn(self.velocity_task_config, state)
        self._last_termination.copy_(terminated)
        success = torch.zeros_like(terminated)
        tracked_command = getattr(state, "reward_command", None)
        if tracked_command is None:
            tracked_command = state.command
        tracked_lin_vel_b = getattr(state, "reward_base_lin_vel_b", None)
        if tracked_lin_vel_b is None:
            tracked_lin_vel_b = state.base_lin_vel_b
        tracked_ang_vel_b = getattr(state, "reward_base_ang_vel_b", None)
        if tracked_ang_vel_b is None:
            tracked_ang_vel_b = state.base_ang_vel_b
        metrics = {
            "linear_velocity_error": torch.linalg.vector_norm(
                tracked_command[:, :2] - tracked_lin_vel_b[:, :2], dim=-1
            ),
            "yaw_rate_error": torch.abs(
                tracked_command[:, 2] - tracked_ang_vel_b[:, 2]
            ),
            "root_height": state.root_height.clone(),
        }
        if hasattr(state, "illegal_contact_force"):
            metrics["maximum_illegal_contact_force"] = (
                state.illegal_contact_force.clone()
            )
        return success, terminated, metrics
