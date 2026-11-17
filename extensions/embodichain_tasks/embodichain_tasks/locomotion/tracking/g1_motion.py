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

"""G1 reference-motion tracking using simulator state and shared tensor objectives."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import torch
from tensordict import TensorDict

from embodichain.compute.kinematics.tree import TreeKinematics
from embodichain.data_pipeline.datasets.motion import MotionLibrary
from embodichain.data_pipeline.datasets.motion_sampling import (
    MotionStartSampler,
    MotionStartSamplerCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg, EventCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    RigidBodyPhysicsCfg,
    RigidBodyMaterialCfg,
)
from embodichain.utils import configclass
from embodichain.utils.math import matrix_from_quat
from embodichain_robots.locomotion_profiles import load_locomotion_robot_profile
from embodichain_tasks.locomotion.managers.actions import ReferenceJointPositionAction
from ._task import MotionTrackingCfg, MotionTrackingTask
from embodichain_tasks.locomotion.managers.events import randomize_motion_context

__all__ = ["G1MotionCfg", "UnitreeG1MotionEnv"]

_TRACKED_LINKS = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


@configclass
class G1MotionCfg:
    """Motion source and episode options for the 29-joint G1 tracking task."""

    motion_path: str = "LAFAN1G1/motion.npz"
    quaternion_order: str = "xyzw"
    clips: list[str] | None = None
    observation_noise: bool = True
    random_start: bool = True
    fixed_clip_assignment: bool = False
    adaptive_sampling: bool = False
    sampler: MotionStartSamplerCfg = MotionStartSamplerCfg()
    reset_noise: bool = False
    domain_randomization: bool = False
    compile_mode: str | None = None
    gamma: float = 0.99
    task: MotionTrackingCfg = MotionTrackingCfg()


@register_env("UnitreeG1MotionRL-v1", max_episode_steps=500, supports_rl=True)
class UnitreeG1MotionEnv(EmbodiedEnv):
    """Track reference clips with absolute leg commands and upper-body residuals.

    ``policy`` contains the ten-frame proprioceptive history followed by future
    goals. ``critic`` contains state features, the same goal, physical parameters and
    remaining time capped and normalized by 100 reference frames. Terminal observations and state are saved
    before automatic resets in ``info['final_observation']`` and
    ``info['transition']``. Their tensors own storage.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        cfg = copy.deepcopy(cfg)
        options = (cfg.extensions or {}).get("motion_tracking", {})
        self.motion_cfg = G1MotionCfg(**options)
        if isinstance(self.motion_cfg.task, dict):
            self.motion_cfg.task = MotionTrackingCfg(**self.motion_cfg.task)
        if isinstance(self.motion_cfg.sampler, dict):
            self.motion_cfg.sampler = MotionStartSamplerCfg(**self.motion_cfg.sampler)
        if not 0 < self.motion_cfg.gamma < 1:
            raise ValueError("Tracking discount must lie in (0, 1).")
        if self.motion_cfg.domain_randomization:
            events = (
                cfg.events
                if isinstance(cfg.events, dict)
                else vars(cfg.events) if cfg.events is not None else {}
            )
            cfg.events = {
                **events,
                "motion_context": EventCfg(func=randomize_motion_context, mode="reset"),
            }
        profile = load_locomotion_robot_profile("UnitreeG1")["robot"]
        self.motion_joint_names = tuple(profile["joint_names"])
        resting = {
            f"{side}_{joint}_joint": value
            for side in ("left", "right")
            for joint, value in (
                ("hip_pitch", -0.312),
                ("knee", 0.669),
                ("ankle_pitch", -0.363),
                ("shoulder_pitch", 0.2),
                ("elbow", 0.6),
            )
        }
        resting.update(left_shoulder_roll_joint=0.2, right_shoulder_roll_joint=-0.2)
        self._resting = [resting.get(n, 0.0) for n in self.motion_joint_names]
        self._scales = [
            0.25 * e / k for e, k in zip(profile["effort_limit"], profile["stiffness"])
        ]
        self._reference_weights = [0.0] * 12 + [1.0] * 17
        robot = cfg.robot
        self._urdf = Path(robot.fpath).expanduser().read_text()
        robot.init_qpos = self._resting
        robot.build_pk_chain = False
        robot.root_props = ArticulationRootPropertiesCfg(
            fixed_base=False, self_collision_enabled=False
        )
        robot.attrs = RigidBodyPhysicsCfg(
            material_props=RigidBodyMaterialCfg(dynamic_friction=0.3)
        )
        no_collision = [
            re.escape(link.attrib["name"])
            for link in ElementTree.fromstring(self._urdf).findall("link")
            if link.find("collision") is None
        ]
        robot.link_attrs = {
            "source_without_collision": LinkPhysicsOverrideCfg(
                link_names_expr=no_collision,
                attrs=RigidBodyPhysicsCfg(
                    collision_props=CollisionPropertiesCfg(collision_enabled=False)
                ),
            )
        }
        drive = {
            key: dict(zip(self.motion_joint_names, profile[value]))
            for key, value in (
                ("stiffness", "stiffness"),
                ("damping", "damping"),
                ("armature", "armature"),
                ("max_effort", "effort_limit"),
                ("max_velocity", "velocity_limit"),
            )
        }
        robot.joint_drive_props = JointDrivePropertiesCfg(
            target_mode="position",
            friction=0.0,
            limit_stiffness=0.0,
            limit_damping=0.0,
            **drive,
        )
        cfg.actions = {
            "joint_position": ActionTermCfg(
                func=ReferenceJointPositionAction,
                params={
                    "joint_names": list(self.motion_joint_names),
                    "preserve_order": True,
                    "offset": self._resting,
                    "scale": self._scales,
                    "clip": None,
                },
            )
        }
        super().__init__(cfg, **kwargs)

    def _init_sim_state(self, **kwargs: Any) -> None:
        c = self.motion_cfg
        if not math.isclose(self.step_dt, c.task.control_dt, abs_tol=1e-9):
            raise ValueError(
                "Motion task control interval differs from simulation timing"
            )
        self.motion = MotionLibrary.from_npz(
            get_data_path(c.motion_path),
            self.motion_joint_names,
            quaternion_order=c.quaternion_order,
            device=self.device,
            clips=c.clips,
        )
        if c.fixed_clip_assignment and (
            self.num_envs != self.motion.num_clips
            or c.random_start
            or c.adaptive_sampling
        ):
            raise ValueError(
                "Fixed clip assignment requires one environment per clip and "
                "disables random starts and adaptive sampling."
            )
        if not math.isclose(self.motion.dt, self.step_dt, abs_tol=1e-9):
            raise ValueError(
                "Motion frequency differs from environment control frequency"
            )
        kin = TreeKinematics(self._urdf, self.motion_joint_names, device=self.device)
        limits = kin.position_limits
        mid = limits.mean(-1)
        half = (limits[:, 1] - limits[:, 0]) * 0.45
        tensor = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)
        self.task = MotionTrackingTask(
            kin,
            tensor(self._resting),
            tensor(self._scales),
            torch.stack((mid - half, mid + half), -1),
            tensor(self._reference_weights),
            _TRACKED_LINKS,
            c.task,
        )
        self._task_features = self.task.features
        self._task_analytics = self._compute_transition
        if c.compile_mode is not None:
            # Partial resets have variable row counts. Fuse their features without
            # capturing a separate CUDA graph for every reset size.
            self._task_features = torch.compile(
                self.task.features,
                fullgraph=True,
                dynamic=True,
                options={"triton.cudagraphs": False},
            )
            self._task_analytics = torch.compile(
                self._task_analytics, mode=c.compile_mode, fullgraph=True
            )
        self.start_sampler = MotionStartSampler(self.motion, c.sampler)
        self._episode_start = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._episode_steps = torch.zeros_like(self._episode_start)
        self._episode_discounted_return = torch.zeros(self.num_envs, device=self.device)
        self.motion_frame = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._history = torch.zeros(
            self.num_envs,
            c.task.history_steps,
            self.task.observation_dim,
            device=self.device,
        )
        self._goal = torch.zeros(self.num_envs, self.task.goal_dim, device=self.device)
        self._features = torch.zeros(self.num_envs, 245, device=self.device)
        self._reward = torch.zeros(self.num_envs, device=self.device)
        self._failed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._continuation = torch.ones(self.num_envs, device=self.device)
        self._reward_terms = {}
        self._friction = torch.ones(self.num_envs, 1, device=self.device)
        self._com_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.policy_joint_ids = torch.tensor(
            [self.robot.joint_names.index(n) for n in self.motion_joint_names],
            device=self.device,
        )
        self._root_com = self.robot.get_com_pose(["pelvis"])[:, 0, :3].clone()
        self._default_torso_com = self.robot.get_com_pose(["torso_link"]).clone()
        super()._init_sim_state(**kwargs)
        self.position_action = self.action_manager.get_term("joint_position")
        self.add_detached_uids_for_reset([self.robot.uid, "floor"])
        # The per-world plane supplies independently configurable ground friction.
        for handle in self.sim.spawn_result.handles.values():
            if handle.get_name() == "default_plane":
                handle.enable_collision(False)
        self.sim.prepare()
        self._refresh(
            torch.arange(self.num_envs, device=self.device), reset_history=True
        )

    def read_state(self) -> dict[str, torch.Tensor]:
        """Return owned root-origin world state in the motion joint order."""
        data = self.robot.body_data.fetch_state()
        pose = data["root_pose"]
        rotation = matrix_from_quat(pose[:, 3:7])
        angular = data["root_ang_vel"]
        offset = (rotation @ self._root_com.unsqueeze(-1)).squeeze(-1)
        return {
            "pos": pose[:, :3].clone(),
            "rot": rotation,
            "quat": pose[:, 3:7].clone(),
            "q": data["qpos"].index_select(1, self.policy_joint_ids),
            "v": data["root_lin_vel"] - torch.cross(angular, offset, dim=-1),
            "w": angular.clone(),
            "qd": data["qvel"].index_select(1, self.policy_joint_ids),
        }

    def capture_transition_start(self) -> dict[str, Any]:
        """Capture owned pre-action state, reference progress and physical context.

        Pair this snapshot with ``info['transition']`` after stepping. The latter
        holds the successor before auto-reset, even on a terminal transition.
        """
        return {
            "state": self.read_state(),
            "frame": self.motion_frame.clone(),
            "context": torch.cat(
                (self._friction, self.position_action.position_bias, self._com_offset),
                -1,
            ),
            "previous_action": self.position_action.raw_actions.clone(),
        }

    def write_state(
        self, env_ids: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> None:
        """Reset selected physical rows from world root-origin state."""
        from embodichain.utils.math import quat_from_matrix

        if env_ids.numel() == 0:
            return
        rotation = state["rot"]
        angular = state["w"]
        offset = (rotation @ self._root_com[env_ids].unsqueeze(-1)).squeeze(-1)
        velocity = state["v"] + torch.cross(angular, offset, dim=-1)
        self.robot.set_state(
            env_ids=env_ids,
            joint_ids=self.policy_joint_ids.tolist(),
            root_pose=torch.cat((state["pos"], quat_from_matrix(rotation)), -1),
            qpos=state["q"],
            target_qpos=state["q"],
            qvel=state["qd"],
            root_velocity=torch.cat((velocity, angular), -1),
            clear_dynamics=True,
        )

    def _initialize_episode(
        self, env_ids: torch.Tensor | None = None, **kwargs: Any
    ) -> None:
        super()._initialize_episode(env_ids, **kwargs)
        ids = self.episode_context._ids(env_ids)
        frames = kwargs.get("motion_frames")
        if frames is None and self.motion_cfg.fixed_clip_assignment:
            frames = self.motion.starts[ids]
        if (
            frames is None
            and self.motion_cfg.adaptive_sampling
            and self.motion_cfg.random_start
        ):
            frames = self.start_sampler.sample(
                len(ids), self.get_generator("motion_reset")
            )
        if frames is None:
            generator = self.get_generator("motion_reset")
            clips = torch.randint(
                self.motion.num_clips,
                (len(ids),),
                device=self.device,
                generator=generator,
            )
            frames = self.motion.starts[clips].clone()
            if self.motion_cfg.random_start:
                frames += (
                    torch.rand(len(ids), device=self.device, generator=generator)
                    * (self.motion.lengths[clips] - 1)
                ).long()
        frames = torch.as_tensor(frames, device=self.device, dtype=torch.long)
        if frames.shape != ids.shape or bool(
            ((frames < 0) | (frames >= self.motion.num_frames)).any()
        ):
            raise ValueError("motion_frames must provide one valid frame per reset row")
        self.motion_frame[ids] = frames
        self._episode_start[ids] = frames
        self._episode_steps[ids] = 0
        self._episode_discounted_return[ids] = 0
        state = self.motion.frame(frames)
        if self.motion_cfg.reset_noise:
            from embodichain.utils.math import matrix_from_euler

            generator = self.get_generator("motion_reset_noise")
            draw = (
                lambda width: torch.rand(
                    (len(ids), width), device=self.device, generator=generator
                )
                * 2
                - 1
            )
            pose = draw(6) * state["pos"].new_tensor([0.05, 0.05, 0.01, 0.1, 0.1, 0.2])
            velocity = draw(6) * state["pos"].new_tensor(
                [0.5, 0.5, 0.2, 0.52, 0.52, 0.78]
            )
            state["pos"] = state["pos"] + pose[:, :3]
            state["rot"] = matrix_from_euler(pose[:, 3:].flip(-1), "ZYX") @ state["rot"]
            state["q"] = (state["q"] + draw(self.task.dof) * 0.1).clamp(
                self.task.soft_limits[:, 0], self.task.soft_limits[:, 1]
            )
            state["v"] = state["v"] + velocity[:, :3]
            state["w"] = state["w"] + velocity[:, 3:]
        self.write_state(ids, state)
        self._refresh(ids, reset_history=True)
        self._reward[ids] = 0
        self._failed[ids] = False
        self._continuation[ids] = 1

    def _refresh(
        self,
        ids: torch.Tensor | None,
        *,
        reset_history: bool,
        state: dict[str, torch.Tensor] | None = None,
        analytics: dict[str, Any] | None = None,
    ) -> None:
        rows = slice(None) if ids is None else ids
        if analytics is None:
            state = {
                k: v[rows]
                for k, v in (self.read_state() if state is None else state).items()
            }
        obs = (
            self.task.observation(
                state,
                self.position_action.raw_actions[rows],
                self.position_action.position_bias[rows],
            )
            if analytics is None
            else analytics["observation"][rows]
        )
        if self.motion_cfg.observation_noise:
            obs = (
                obs
                + (
                    torch.rand(
                        obs.shape,
                        device=self.device,
                        generator=self.get_generator("motion_observation"),
                    )
                    * 2
                    - 1
                )
                * self.task.noise_scale
            )
        if reset_history:
            self._history[rows] = obs.unsqueeze(1)
        else:
            self._history[rows] = torch.cat(
                (self._history[rows, 1:], obs.unsqueeze(1)), 1
            )
        self._goal[rows] = (
            self.task.goal(
                state,
                self.motion.future(self.motion_frame[rows], self.task.future_offsets),
            )
            if analytics is None
            else analytics["goal"][rows]
        )
        self._features[rows] = (
            self._task_features(state)
            if analytics is None
            else analytics["features"][rows]
        )

    def _compute_transition(
        self,
        state: dict[str, torch.Tensor],
        reference: dict[str, torch.Tensor],
        future: dict[str, torch.Tensor],
        action: torch.Tensor,
        previous_action: torch.Tensor,
        bias: torch.Tensor,
    ) -> dict[str, Any]:
        """Evaluate finite state rows and task outputs in one compiled function."""
        valid = torch.stack(
            [torch.isfinite(value).flatten(1).all(1) for value in state.values()]
        ).all(0)
        # Invalid rows use finite reference values for task arithmetic and are
        # excluded from dynamics supervision by the returned validity mask.
        state = {
            key: torch.where(
                valid.reshape((-1,) + (1,) * (value.ndim - 1)), value, reference[key]
            )
            for key, value in state.items()
        }
        result = self.task.analytics(
            state, reference, future, action, previous_action, bias
        )
        result["state"], result["valid"] = state, valid
        result["failed"] = result["failed"] | ~valid
        result["reward"] = torch.where(valid, result["reward"], 0)
        result["reward_terms"] = {
            name: torch.where(valid, value, 0)
            for name, value in result["reward_terms"].items()
        }
        result["continuation"] = torch.where(valid, result["continuation"], 0)
        return result

    def _update_sim_state(self, **kwargs: Any) -> None:
        super()._update_sim_state(**kwargs)
        self.motion_frame = torch.minimum(
            self.motion_frame + 1,
            self.motion.ends[self.motion.clip_ids(self.motion_frame)] - 1,
        )
        state = self.read_state()
        reference = self.motion.frame(self.motion_frame)
        analytics = self._task_analytics(
            state,
            reference,
            self.motion.future(self.motion_frame, self.task.future_offsets),
            self.position_action.raw_actions,
            self.position_action.previous_raw_actions,
            self.position_action.position_bias,
        )
        self._transition_valid = analytics["valid"]
        self._transition_state = analytics["state"]
        # Reset invokes another compiled function. Keep mutable episode fields
        # in owned buffers rather than retaining CUDA graph output storage.
        self._reward.copy_(analytics["reward"])
        self._reward_terms = analytics["reward_terms"]
        self._episode_discounted_return += (
            self.motion_cfg.gamma**self._episode_steps * self._reward
        )
        self._episode_steps += 1
        self._failed.copy_(analytics["failed"])
        self._continuation.copy_(analytics["continuation"])
        self._refresh(
            None,
            reset_history=False,
            analytics=analytics,
        )

    def get_obs(self, **kwargs: Any) -> TensorDict:
        remaining = (
            self.motion.remaining(self.motion_frame)
            .float()
            .clamp(max=100)
            .unsqueeze(-1)
            / 100.0
        )
        parameters = torch.cat(
            (self._friction, self.position_action.position_bias, self._com_offset), -1
        )
        return TensorDict(
            {
                "policy": torch.cat((self._history.flatten(1), self._goal), -1),
                "critic": torch.cat(
                    (self._features, self._goal, parameters, remaining), -1
                ),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Expose tracking failure through the simulator RL termination contract."""
        return (
            torch.zeros_like(self._failed),
            self._failed.clone(),
            {"tracking_reward": self._reward.clone()},
        )

    def check_truncated(self, obs: TensorDict, info: dict[str, Any]) -> torch.Tensor:
        return self.motion.remaining(self.motion_frame) == 0

    def get_reward(
        self, obs: TensorDict, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        return self._reward.clone()

    def _hook_after_sim_step(
        self,
        obs: TensorDict,
        action: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        kwargs["truncateds"].logical_and_(~kwargs["terminateds"])
        if self.motion_cfg.adaptive_sampling:
            ids = torch.nonzero(dones, as_tuple=False).flatten()
            if len(ids):
                horizon = self.motion.remaining(self._episode_start[ids]).clamp_max(
                    self.max_episode_steps
                )
                maximum = (
                    self.task.max_step_reward
                    * (1 - self.motion_cfg.gamma**horizon)
                    / (1 - self.motion_cfg.gamma)
                )
                ratio = self._episode_discounted_return[ids] / maximum
                self.start_sampler.update(self._episode_start[ids], ratio)
                info["completed_normalized_return"] = ratio.clone()
        info["final_observation"] = obs.clone()
        info["_final_observation"] = dones.clone()
        info["transition"] = {
            "state": {k: v.clone() for k, v in self._transition_state.items()},
            "usable": self._transition_valid.clone(),
            "frame": self.motion_frame.clone(),
            "history": self._history.clone(),
            "goal": self._goal.clone(),
            "action": self.position_action.raw_actions.clone(),
            "setpoint_action": self.position_action.setpoint_actions.clone(),
            "target_qpos": self.position_action.processed_actions.clone(),
            "continuation": self._continuation.clone(),
        }
        info["reward_terms"] = {k: v.clone() for k, v in self._reward_terms.items()}
        super()._hook_after_sim_step(obs, action, rewards, dones, info, **kwargs)
