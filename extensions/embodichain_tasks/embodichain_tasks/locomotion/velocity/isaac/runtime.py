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

"""EmbodiChain and DexSim adapter for fixed Isaac velocity definitions."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from embodichain.lab.sim.types import EnvAction, EnvObs
from embodichain.utils.math import quat_apply_inverse

from .._embodichain import EmbodiChainVelocityEnv
from .mdp import (
    IsaacVelocityState,
    IsaacVelocityConstants,
    build_observation,
    observation_terms,
    action_target,
    compute_rewards,
    compute_termination,
    dc_motor_effort,
    update_contact_times,
)
from .terrain import ReferenceHeightScanner
from .curriculum import ReferenceTerrainCurriculum

__all__ = ["IsaacVelocityEnv"]


class IsaacVelocityEnv(EmbodiChainVelocityEnv):
    """Run the reference policy interface on shared training or evaluation terrain.

    The benchmark disables startup mass/COM and interval pushes on both stacks.
    ``reference_randomization=True`` enables those reference training events.
    Training loads the complete reference Rough terrain and curriculum; fixed
    scenes load evaluation origins. Both use the shared triangle mesh for
    rays and Newton rasterization for collision.
    """

    state_type = IsaacVelocityState
    compute_rewards_fn = staticmethod(compute_rewards)
    compute_termination_fn = staticmethod(compute_termination)

    def _initialize_contact_histories(self) -> None:
        """Use this task's control-frame histories and normal-force sampling."""

    def _invalidate_task_cache(self) -> None:
        super()._invalidate_task_cache()
        self._observation_cache = None

    def _prepare_scene(self, **kwargs: Any) -> None:
        from dexsim.spawn import NewtonCollisionDesc, GeometryDesc

        rough = self.velocity_task_config.action_dim == 37
        self._reference_terrain = None
        geometry = None
        contact = NewtonCollisionDesc(
            mu=1.0, margin=0.0, gap=0.01, ke=160000.0, kd=1100.0
        )
        if rough:
            from embodichain.data import get_data_path
            import json

            path = Path(get_data_path(self.cfg.reference_terrain))
            if getattr(self.cfg, "reference_randomization", False) and not getattr(
                self.cfg, "reference_terrain_course", None
            ):
                self.cfg.reference_terrain_course = json.loads(
                    path.with_suffix(".json").read_text()
                )
            terrain = np.load(path)
            self._reference_terrain = (terrain["vertices"], terrain["triangles"])
            geometry = GeometryDesc.mesh(
                vertices=terrain["vertices"], triangles=terrain["triangles"]
            )
            contact.heightfield_resolution = float(terrain["resolution"])
        self.sim.configure_ground_plane_collision(newton=contact, geometry=geometry)
        self._graph_motor = None
        if self.explicit_pd_effort_control and self.sim.is_newton_backend:
            self._graph_motor = self.sim._register_joint_dc_motor_control(
                self.robot.uid, self.velocity_task_config.data["robot"]
            )
        super()._prepare_scene(**kwargs)

    def _init_sim_state(self, **kwargs: Any) -> None:
        self._reference_events_deferred = True
        self._reference_scan_reuse = None
        super()._init_sim_state(**kwargs)
        from ...managers.observations import velocity_locomotion_observation
        from ...managers.rewards import velocity_locomotion_total_reward

        expected = {
            id(self.cfg.observations): velocity_locomotion_observation,
            id(self.cfg.rewards): velocity_locomotion_total_reward,
        }
        self._reference_task_managers_only = set(self.sensors) == {
            "locomotion_contacts"
        } and all(
            self._configured_functor_callable(term) is expected[id(collection)]
            for collection, _, term in self._iter_configured_functors()
            if id(collection) in expected
        )
        config = self.velocity_task_config
        self._reference_constants = IsaacVelocityConstants.from_config(
            config, self.locomotion_action
        )
        self._reference_origin = self.locomotion_action.new_tensor(
            getattr(self.cfg, "reference_origin", (0.0, 0.0, 0.0))
        )
        self.compute_rewards_fn = partial(
            compute_rewards, constants=self._reference_constants
        )
        self.single_action_space = gym.spaces.Box(
            -np.inf, np.inf, (config.action_dim,), dtype=np.float32
        )
        self._reference_sensor = self.get_sensor("locomotion_contacts")
        names = (*self.foot_link_names, *self.illegal_contact_link_names)
        self._tracked_actor_ids = self._reference_sensor.get_actor_ids(
            self.robot.uid, names
        ).to(torch.long)
        self._tracked_keys = (
            torch.arange(self.num_envs, device=self.device)[:, None] * (2**32)
            + self._tracked_actor_ids
        ).flatten()
        self._sorted_keys, self._sorted_rows = self._tracked_keys.sort()
        bodies = len(names)
        self._normal_force = torch.zeros((self.num_envs, bodies, 3), device=self.device)
        self._normal_history = torch.zeros(
            (self.num_envs, 3, bodies, 3), device=self.device
        )
        self._air_time = torch.zeros((self.num_envs, bodies), device=self.device)
        self._contact_time = torch.zeros_like(self._air_time)
        self._last_air_time = torch.zeros_like(self._air_time)
        self._last_contact_time = torch.zeros_like(self._air_time)
        self._applied_effort = torch.zeros_like(self.locomotion_action)
        self._previous_physics_joint_velocity = torch.zeros_like(self.locomotion_action)
        self._position_target = self._applied_effort.clone()
        self._kp = torch.tensor(config.data["robot"]["stiffness"], device=self.device)
        self._kd = torch.tensor(config.data["robot"]["damping"], device=self.device)
        self._effort_limit = torch.tensor(
            config.data["robot"]["effort_limit"], device=self.device
        )
        self._velocity_limit = torch.tensor(
            config.data["robot"]["velocity_limit"], device=self.device
        )
        if self._graph_motor is not None:
            import warp as wp

            target = wp.to_torch(self._graph_motor.targets)
            target.copy_(self._position_target)
            self._position_target = target
            self._applied_effort = wp.to_torch(self._graph_motor.effort)
            self._previous_physics_joint_velocity = wp.to_torch(
                self._graph_motor.velocity
            )
        self._push_remaining = torch.empty(self.num_envs, device=self.device).uniform_(
            10.0, 15.0, generator=self._locomotion_generator
        )
        self._height_scanner = None
        if self._reference_terrain is not None:
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend

            backend = get_newton_backend(self.sim.spawn_result.world)
            self._height_scanner = ReferenceHeightScanner(
                *self._reference_terrain,
                self.num_envs,
                self.device,
                physics_model=backend.model,
                physics_state=backend.state_0,
            )
        self._reference_initial_states = (
            {
                name: torch.tensor(
                    [row[name] for row in self.cfg.reference_initial_states],
                    device=self.device,
                    dtype=torch.float32,
                )
                for name in ("root_pose", "root_velocity", "qpos", "qvel")
            }
            if getattr(self.cfg, "reference_initial_states", None) is not None
            else None
        )
        self._observation_cache: torch.Tensor | None = None
        self._reference_terrain_course = None
        terrain_course = getattr(self.cfg, "reference_terrain_course", None)
        if terrain_course is not None:
            origins = torch.tensor(
                terrain_course["terrain_origins"],
                device=self.device,
                dtype=torch.float32,
            )
            self._reference_terrain_course = ReferenceTerrainCurriculum(
                origins,
                self.num_envs,
                terrain_course["max_init_terrain_level"],
                terrain_course["generator"]["size"][0],
                self.max_episode_steps * self.step_dt,
                self._locomotion_generator,
            )
        if getattr(self.cfg, "reference_randomization", False):
            self._randomize_startup()
        self._reference_events_deferred = False

    def _randomize_startup(self) -> None:
        """Apply reference log-uniform mass/inertia and additive COM events."""
        spec = self.velocity_task_config.data["events"]["add_base_mass"]
        name = spec["body_names"]
        minimum, maximum = spec["params"]["mass_distribution_params"]
        scale = (
            torch.empty((self.num_envs, 1), device=self.device)
            .uniform_(
                np.log(minimum), np.log(maximum), generator=self._locomotion_generator
            )
            .exp_()
        )
        self.robot.set_inertia(
            self.robot.get_inertia(link_names=[name]) * scale[:, :, None],
            link_names=[name],
        )
        self.robot.set_mass(
            self.robot.get_mass(link_names=[name]) * scale, link_names=[name]
        )
        com_spec = self.velocity_task_config.data["events"].get("base_com")
        if com_spec is not None:
            pose = self.robot.get_com_pose(link_names=[name]).clone()
            for axis, key in enumerate(("x", "y", "z")):
                pose[:, :, axis] += torch.empty_like(pose[:, :, axis]).uniform_(
                    *com_spec["params"]["com_range"][key],
                    generator=self._locomotion_generator,
                )
            self.robot.set_com_pose(pose, link_names=[name])

    def _step_action(self, action: torch.Tensor) -> torch.Tensor:
        """Write unbounded reference targets through the Spawn batch state API."""
        self._position_target.copy_(
            action_target(
                self.velocity_task_config,
                self.locomotion_action,
                constants=self._reference_constants,
            )
        )
        if not self.explicit_pd_effort_control:
            self.robot.body_data.articulation_view.apply_state(
                None, self.policy_joint_ids, target_qpos=self._position_target
            )
        self._last_action_trace = self.action_manager.action_trace()
        return action

    def _advance_physics(self) -> None:
        """Update actuator effort at 200 Hz and sample lazy contacts at 50 Hz."""
        if getattr(self, "_graph_motor", None) is not None:
            for _ in range(self.cfg.sim_steps_per_control):
                self.sim.update(self.physics_dt, 1, render_final_step=False)
            self._sample_reference_contacts()
            return
        for step in range(self.cfg.sim_steps_per_control):
            if (
                not self.explicit_pd_effort_control
                and step < self.cfg.sim_steps_per_control - 1
            ):
                # Implicit drives run in the solver. Only the final pre-step
                # effort and velocity are consumed by control-step rewards.
                self.sim.update(self.physics_dt, 1, render_final_step=False)
                continue
            state = self.robot.body_data.fetch_state()
            velocity = state["qvel"][:, self.policy_joint_ids]
            if step == self.cfg.sim_steps_per_control - 1:
                self._previous_physics_joint_velocity.copy_(velocity)
            effort = (
                self._kp
                * (self._position_target - state["qpos"][:, self.policy_joint_ids])
                - self._kd * velocity
            )
            if self.explicit_pd_effort_control:
                effort = dc_motor_effort(
                    effort,
                    velocity,
                    self._effort_limit,
                    self._velocity_limit,
                    self._effort_limit,
                )
                self.robot.set_qf(effort, joint_ids=self.policy_joint_ids)
            else:
                effort = effort.clamp(-self._effort_limit, self._effort_limit)
            self._applied_effort.copy_(effort)
            self.sim.update(self.physics_dt, 1, render_final_step=False)
        self._sample_reference_contacts()

    def _sample_reference_contacts(self) -> None:
        """Reduce all counterpart normal impulses to tracked body net forces."""
        if getattr(self, "_reference_defer_observations", False) and not (
            self._reference_sensor.requires_substep_update
        ):
            solver_dt = self.physics_dt / self.cfg.sim_cfg.physics_cfg.num_substeps
            self._reference_sensor._reduce_normal_impulses(
                self._sorted_keys, self._sorted_rows, self._normal_force, solver_dt
            )
        else:
            self._reference_sensor.update()
            data = self._reference_sensor.get_data()
            ids = data["user_ids"].to(torch.long)
            valid = data["is_valid"]
            solver_dt = self.physics_dt / self.cfg.sim_cfg.physics_cfg.num_substeps
            force = data["normal"] * data["impulse"][:, :, None] / solver_dt
            env = torch.arange(self.num_envs, device=self.device)[:, None, None]
            keys = (env * (2**32) + ids).flatten()
            index = torch.searchsorted(self._sorted_keys, keys).clamp(
                max=self._sorted_keys.numel() - 1
            )
            match = (self._sorted_keys[index] == keys) & valid[:, :, None].expand_as(
                ids
            ).flatten()
            forces = torch.stack((-force, force), dim=2).flatten(0, 2)
            self._normal_force.zero_()
            self._normal_force.view(-1, 3).index_add_(
                0, self._sorted_rows[index], torch.where(match[:, None], forces, 0.0)
            )
        self._normal_history[:, 1:] = self._normal_history[:, :-1].clone()
        self._normal_history[:, 0].copy_(self._normal_force)
        update_contact_times(
            self._normal_force,
            self.step_dt,
            self._air_time,
            self._contact_time,
            self._last_air_time,
            self._last_contact_time,
        )

    def _update_sim_state(self, **kwargs: Any) -> None:
        data = super()._update_sim_state(**kwargs)
        # The EA native Go2 actuator path updates articulation data once per
        # decimation block. G1 updates it after each individual physics step.
        if not self.explicit_pd_effort_control:
            self._joint_acceleration.copy_(
                (self._previous_joint_velocity - self._previous_physics_joint_velocity)
                / self.physics_dt
            )
        if data is not None:
            self._state_cache = self._make_reference_state(data)
        self._observation_cache = None

    def _advance_command_timer(self) -> None:
        if not getattr(self, "_reference_events_deferred", False):
            super()._advance_command_timer()

    def _update_heading_commands(self) -> None:
        if not getattr(self, "_reference_events_deferred", False):
            super()._update_heading_commands()

    def _apply_reference_interval_events(self) -> None:
        """Apply reference velocity pushes after rewards and episode resets."""
        if getattr(self.cfg, "reference_randomization", False):
            self._push_remaining.sub_(self.step_dt)
            ids = (self._push_remaining <= 0.0).nonzero().flatten()
            if ids.numel():
                velocity = self.robot.body_data.root_vel[ids].clone()
                velocity[:, :2].uniform_(
                    -0.5, 0.5, generator=self._locomotion_generator
                )
                heights = self.get_velocity_locomotion_state().height_scan
                self.robot.set_state(env_ids=ids, root_velocity=velocity)
                # Velocity writes preserve the ray origins and terrain geometry.
                self._state_cache = None
                self._reference_scan_reuse = heights
                self._observation_cache = None
                self._push_remaining[ids] = torch.empty(
                    ids.numel(), device=self.device
                ).uniform_(10.0, 15.0, generator=self._locomotion_generator)

    def step(
        self,
        action: EnvAction,
        *,
        policy_observations_only: bool = False,
        **kwargs: Any,
    ) -> tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Compute rewards and resets before commands, pushes and observations."""
        self._reference_defer_observations = (
            policy_observations_only
            and getattr(self, "_reference_task_managers_only", True)
            and not any(
                (
                    getattr(self, "rollout_buffer", None) is not None,
                    getattr(self, "dataset_manager", None) is not None,
                    getattr(self, "_traj_buffer", None) is not None,
                    bool(
                        getattr(
                            getattr(self, "event_manager", None), "available_modes", ()
                        )
                    ),
                )
            )
        )
        self._reference_policy_observations_only = (
            self._reference_defer_observations
            and getattr(self, "_reference_task_managers_only", False)
        )
        self._reference_events_deferred = True
        try:
            try:
                _, rewards, terminated, truncated, info = super().step(action, **kwargs)
            finally:
                self._reference_events_deferred = False
                self._reference_defer_observations = False
            self._advance_command_timer()
            self._update_heading_commands()
            self._apply_reference_interval_events()
            return self.get_obs(**kwargs), rewards, terminated, truncated, info
        finally:
            self._reference_policy_observations_only = False

    def get_obs(self, **kwargs: Any) -> EnvObs:
        """Defer unused training observations until commands and resets finish."""
        if getattr(self, "_reference_defer_observations", False):
            return {}
        if getattr(self, "_reference_policy_observations_only", False):
            obs = TensorDict({}, batch_size=[self.num_envs], device=self.device)
            return self._extend_obs(obs, **kwargs)
        return super().get_obs(**kwargs)

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs: Any
    ) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        if ids.numel() == 0:
            return
        if self._reference_terrain_course is not None:
            self._reference_terrain_course.update(
                ids, self.robot.body_data.root_pose[:, :3], self.reward_command
            )
        previous_deferred = getattr(self, "_reference_events_deferred", False)
        self._reference_events_deferred = True
        try:
            super()._initialize_episode(env_ids=env_ids, **kwargs)
        finally:
            self._reference_events_deferred = previous_deferred
        deterministic = getattr(self.cfg, "reference_deterministic_reset", True)
        pose = self._reference_reset_pose
        self._reference_reset_pose = None
        if deterministic:
            pose.zero_()
            pose[:, 6] = 1.0
            pose[:, 2] = self.velocity_task_config.data["robot"]["root_height"]
        origin = (
            self._reference_terrain_course.env_origins[ids]
            if self._reference_terrain_course is not None
            else getattr(self, "_reference_origin", None)
        )
        if origin is None:
            origin = pose.new_tensor(
                getattr(self.cfg, "reference_origin", (0.0, 0.0, 0.0))
            )
        pose[:, :3] += origin
        constants = getattr(self, "_reference_constants", None)
        default_position = (
            constants.default_joint_position
            if constants is not None
            else pose.new_tensor(self.velocity_task_config.default_joint_position)
        )
        joint = default_position.expand(ids.numel(), -1).clone()
        velocity = torch.zeros((ids.numel(), 6), device=self.device)
        if not deterministic:
            bounds = self.velocity_task_config.data["events"]["reset_robot_joints"][
                "params"
            ]["position_range"]
            joint *= torch.empty_like(joint).uniform_(
                *bounds, generator=self._locomotion_generator
            )
            velocity.uniform_(-0.5, 0.5, generator=self._locomotion_generator)
        limits = (
            constants.joint_limits
            if constants is not None
            else joint.new_tensor(
                self.velocity_task_config.data["robot"]["soft_joint_pos_limits"]
            )
        )
        joint = joint.clamp(limits[:, 0], limits[:, 1])
        joint_velocity = torch.zeros_like(joint)
        if self._reference_initial_states is not None:
            selected = ids % self._reference_initial_states["root_pose"].shape[0]
            pose = self._reference_initial_states["root_pose"][selected]
            velocity = self._reference_initial_states["root_velocity"][selected]
            joint = self._reference_initial_states["qpos"][selected]
            joint_velocity = self._reference_initial_states["qvel"][selected]
        self.robot.set_state(
            env_ids=ids,
            joint_ids=self.policy_joint_ids,
            root_pose=pose,
            qpos=joint,
            qvel=joint_velocity,
            target_qpos=joint,
            root_velocity=velocity,
            clear_dynamics=True,
        )
        self._position_target[ids] = joint
        for buffer in (
            self._normal_force,
            self._normal_history,
            self._air_time,
            self._contact_time,
            self._last_air_time,
            self._last_contact_time,
            self._applied_effort,
            self._previous_physics_joint_velocity,
        ):
            buffer.index_fill_(0, ids, 0.0)
        self._push_remaining[ids] = torch.empty(
            ids.numel(), device=self.device
        ).uniform_(10.0, 15.0, generator=self._locomotion_generator)
        self._observation_cache = None
        self._invalidate_task_cache()

    def _write_episode_initial_state(
        self,
        env_ids: torch.Tensor,
        root_pose: torch.Tensor,
        joint_position: torch.Tensor,
    ) -> None:
        """Keep sampled RNG values until the reference reset writes final state."""
        self._reference_reset_pose = root_pose.clone()

    def get_velocity_locomotion_state(self) -> IsaacVelocityState:
        """Fetch policy-ordered reference state and actual torso height rays."""
        if self._state_cache is not None:
            if (
                self._height_scanner is not None
                and self._state_cache.height_scan.shape[-1] == 0
                and not getattr(self, "_reference_defer_observations", False)
            ):
                self._state_cache = replace(
                    self._state_cache, height_scan=self._compute_reference_height_scan()
                )
            return self._state_cache
        self._state_cache = self._make_reference_state(
            self.robot.body_data.fetch_state()
        )
        return self._state_cache

    def _make_reference_state(
        self, data: dict[str, torch.Tensor]
    ) -> IsaacVelocityState:
        """Build task state from the current control frame's articulation read."""
        quat = data["root_pose"][:, 3:7]
        if self.device.type == "cuda" and not any(
            value.requires_grad
            for value in (quat, data["root_lin_vel"], data["root_ang_vel"])
        ):
            from ._frame import _reference_frame_vectors

            linear_b, angular_b, gravity_b, linear_yaw = _reference_frame_vectors(
                quat, data["root_lin_vel"], data["root_ang_vel"]
            )
        else:
            gravity = torch.zeros_like(data["root_lin_vel"])
            gravity[:, 2] = -1.0
            x, y, z, w = quat.unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
            yaw_quat = torch.zeros_like(quat)
            yaw_quat[:, 2] = torch.sin(yaw / 2)
            yaw_quat[:, 3] = torch.cos(yaw / 2)
            linear_b = quat_apply_inverse(quat, data["root_lin_vel"])
            angular_b = quat_apply_inverse(quat, data["root_ang_vel"])
            gravity_b = quat_apply_inverse(quat, gravity)
            linear_yaw = quat_apply_inverse(yaw_quat, data["root_lin_vel"])
        count = len(self.foot_link_names)
        heights = data["qpos"].new_empty((self.num_envs, 0))
        if self._height_scanner is not None and not getattr(
            self, "_reference_defer_observations", False
        ):
            heights = self._compute_reference_height_scan()
        return IsaacVelocityState(
            base_lin_vel_b=linear_b,
            base_ang_vel_b=angular_b,
            projected_gravity_b=gravity_b,
            base_lin_vel_yaw=linear_yaw,
            base_ang_vel_w=data["root_ang_vel"],
            command=self.reward_command,
            joint_pos=data["qpos"][:, self.policy_joint_ids],
            joint_vel=data["qvel"][:, self.policy_joint_ids],
            joint_acc=self._joint_acceleration,
            applied_torque=self._applied_effort,
            action=self.locomotion_action,
            previous_action=self.last_locomotion_action,
            foot_vel_w=self.robot.body_data.body_link_vel[:, self.foot_link_ids, :3],
            foot_force_history=self._normal_history[:, :, :count],
            current_air_time=self._air_time[:, :count],
            current_contact_time=self._contact_time[:, :count],
            last_air_time=self._last_air_time[:, :count],
            illegal_force_history=self._normal_history[:, :, count:],
            height_scan=heights,
            episode_step=self._episode_step,
            root_height=data["root_pose"][:, 2],
        )

    def _compute_reference_height_scan(self) -> torch.Tensor:
        """Reuse height rays across velocity-only writes."""
        reused = getattr(self, "_reference_scan_reuse", None)
        self._reference_scan_reuse = None
        return (
            reused
            if reused is not None
            else self._height_scanner.compute(
                self.robot.body_data.body_link_pose[:, self.orientation_link_id]
            )
        )

    def build_velocity_locomotion_observations(
        self, *, enable_corruption: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the same reference policy tensor to actor and critic."""
        if self._observation_cache is None:
            state = replace(self.get_velocity_locomotion_state(), command=self.command)
            noise = {}
            if enable_corruption and not getattr(
                self, "_reference_events_deferred", False
            ):
                for name, term in observation_terms(
                    self.velocity_task_config,
                    state,
                    constants=self._reference_constants,
                ).items():
                    spec = self.velocity_task_config.data["observations"][name]["noise"]
                    if spec:
                        noise[name] = torch.empty_like(term).uniform_(
                            spec["n_min"],
                            spec["n_max"],
                            generator=self._locomotion_generator,
                        )
            self._observation_cache = build_observation(
                self.velocity_task_config,
                state,
                noise=noise,
                constants=self._reference_constants,
            )
        return self._observation_cache, self._observation_cache

    def _extend_obs(self, obs: EnvObs, **kwargs: Any) -> EnvObs:
        """Generate fresh noise per observation pass and share it with the critic."""
        self._observation_cache = None
        return super()._extend_obs(obs, **kwargs)

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return body-contact failures and reference frame tracking errors."""
        state = self.get_velocity_locomotion_state()
        terminated, _ = compute_termination(self.velocity_task_config, state)
        velocity = (
            state.base_lin_vel_yaw
            if self.velocity_task_config.action_dim == 37
            else state.base_lin_vel_b
        )
        angular = (
            state.base_ang_vel_w
            if self.velocity_task_config.action_dim == 37
            else state.base_ang_vel_b
        )
        return (
            torch.zeros_like(terminated),
            terminated,
            {
                "linear_velocity_error": (state.command[:, :2] - velocity[:, :2]).norm(
                    dim=-1
                ),
                "yaw_rate_error": (state.command[:, 2] - angular[:, 2]).abs(),
                "root_height": state.root_height,
                "maximum_illegal_contact_force": state.illegal_force_history.norm(
                    dim=-1
                ).amax(dim=(1, 2)),
            },
        )
