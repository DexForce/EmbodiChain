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

"""Differentiable reference-motion rollouts from measured replay starts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from tensordict import TensorDict

from embodichain.compute.kinematics import LinkKinematics
from embodichain.data_pipeline.datasets.motion import MotionLibrary
from embodichain.learning.rl.env import DifferentiableRolloutSpec, register_learning_env
from embodichain.utils import configclass
from embodichain_models.dynamics.rwm import KinematicWorldModel, TransitionReplay
from embodichain_models.dynamics.rwm.robot_state import unpack_robot_state
from ._task import MotionTrackingTask

__all__ = [
    "MotionModelCfg",
    "MotionModelStart",
    "MotionModelEnv",
    "replay_motion_starts",
]


@dataclass(frozen=True)
class MotionModelStart:
    """Owned, detached measured state and clean oldest-first observation history."""

    state: torch.Tensor
    context: torch.Tensor
    frame: torch.Tensor
    history: torch.Tensor
    previous_action: torch.Tensor
    valid: torch.Tensor


@torch.no_grad()
def replay_motion_starts(
    replay: TransitionReplay, batch_size: int, task: MotionTrackingTask
) -> MotionModelStart:
    """Sample usable starts with episode-padded history and preserved encoder bias."""
    for _ in range(4):
        data = replay.sample_starts(batch_size * 4, task.cfg.history_steps)
        mask = (
            data["history_complete"]
            & data["start"]["usable"]
            & data["history"]["usable"].all(-1)
        )
        if int(mask.sum()) >= batch_size:
            start = {k: v[mask][:batch_size] for k, v in data["start"].items()}
            history = {k: v[mask][:batch_size] for k, v in data["history"].items()}
            observations = task.observation(
                unpack_robot_state(history["state"]),
                history["previous_action"],
                history["context"][..., 1 : 1 + task.dof],
            )
            return MotionModelStart(
                start["state"],
                start["context"],
                start["frame"],
                observations,
                start["previous_action"],
                start["usable"],
            )
    raise RuntimeError("Replay has too few usable starts with complete history.")


@configclass
class MotionModelCfg:
    """Short-horizon prediction and training-time perturbations."""

    horizon: int = 32
    observation_noise: bool = True
    goal_position_noise: float = 0.0
    action_noise: float = 0.0
    clean_previous_action: bool = False
    compile_mode: str | None = None


class MotionModelEnv:
    """Run a fixed-horizon model trajectory using the physical task's tensor rules.

    A workflow injects measured starts or a provider that samples replay at reset.
    Motion-clip suffixes are marked invalid; soft continuation remains differentiable.
    Finite-horizon bootstrap is handled by APG. Each complete rollout is reset by
    the trainer. WM parameters/statistics are frozen by the surrounding workflow
    until all dependent backward work finishes.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        model: KinematicWorldModel,
        motion: MotionLibrary,
        task: MotionTrackingTask,
        cfg: MotionModelCfg | None = None,
        start_provider: Callable[[], MotionModelStart] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.empty(0, device=device).device
        self.model, self.motion, self.task = model, motion, task
        self.cfg = cfg or MotionModelCfg()
        if (
            num_envs < 1
            or self.cfg.horizon < 1
            or min(self.cfg.action_noise, self.cfg.goal_position_noise) < 0
        ):
            raise ValueError("Invalid motion-model batch, horizon or noise.")
        if tuple(task.kinematics.joint_names) != tuple(model.cfg.joint_names) or tuple(
            motion.joint_names
        ) != tuple(model.cfg.joint_names):
            raise ValueError("Model, motion and task joint order must match.")
        if (
            motion.device != self.device
            or task.kinematics.device != self.device
            or not abs(motion.dt - model.cfg.step_dt) < 1e-8
        ):
            raise ValueError(
                "Model environment device/control period differs from motion data."
            )
        if (
            not torch.equal(task.resting_position, model.resting_position)
            or task.kinematics.urdf != model.urdf
        ):
            raise ValueError(
                "Task FK features must use the model geometry and resting pose."
            )
        self._transition_fn = (
            self._transition
            if self.cfg.compile_mode is None
            else torch.compile(
                self._transition, mode=self.cfg.compile_mode, fullgraph=True
            )
        )
        self._reference_links_fn = (
            self.task.links
            if self.cfg.compile_mode is None
            else torch.compile(
                self.task.links, mode=self.cfg.compile_mode, fullgraph=True
            )
        )
        self.start_provider = start_provider
        self._start: MotionModelStart | None = None
        self._steps = 0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(torch.initial_seed())
        policy_dim = task.observation_dim * task.cfg.history_steps + task.goal_dim
        critic_dim = model.feature_dim + task.goal_dim + model.cfg.context_dim + 1
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(-float("inf"), float("inf"), (policy_dim,)),
                "critic": gym.spaces.Box(-float("inf"), float("inf"), (critic_dim,)),
            }
        )
        self.single_action_space = gym.spaces.Box(
            -float("inf"), float("inf"), (task.dof,)
        )

    def set_starts(self, starts: MotionModelStart) -> None:
        """Copy a batch so replay replacement cannot change an outstanding rollout."""
        expected = {
            "state": (self.num_envs, self.model.state_dim),
            "context": (self.num_envs, self.model.cfg.context_dim),
            "frame": (self.num_envs,),
            "history": (
                self.num_envs,
                self.task.cfg.history_steps,
                self.task.observation_dim,
            ),
            "previous_action": (self.num_envs, self.task.dof),
            "valid": (self.num_envs,),
        }
        values = {}
        for name, shape in expected.items():
            value = getattr(starts, name)
            if value.shape != shape or value.device != self.device:
                raise ValueError(
                    f"Invalid imagination start field {name}; expected {shape} on {self.device}."
                )
            values[name] = value.detach().clone()
        self._start = MotionModelStart(**values)

    def prepare_differentiable_rollout(
        self, rollout_index: int
    ) -> DifferentiableRolloutSpec:
        """Request one independent, complete imagination horizon."""
        return DifferentiableRolloutSpec(self.cfg.horizon)

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[TensorDict, dict[str, Any]]:
        """Start from a replay sample, adding fresh noise to its clean history."""
        if seed is not None:
            self._generator.manual_seed(seed)
        if options and "starts" in options:
            self.set_starts(options["starts"])
        elif self.start_provider is not None:
            self.set_starts(self.start_provider())
        if self._start is None:
            raise RuntimeError(
                "Inject measured starts before resetting the model environment."
            )
        s = self._start
        self.state = s.state.clone()
        self.context, self.frame = s.context.clone(), s.frame.clone()
        self.previous_action = s.previous_action.clone()
        self.history = s.history.clone() + self._observation_noise(s.history.shape)
        self._steps = 0
        with torch.no_grad():
            offsets = torch.arange(
                self.cfg.horizon + int(self.task.future_offsets[-1]) + 1,
                device=self.device,
            )
            self._reference = self.motion.future(self.frame, offsets)
            raw = self.frame[:, None] + offsets[None]
            self._valid = (
                raw < self.motion.ends[self.motion.clip_ids(self.frame), None]
            ) & s.valid[:, None]
            self._remaining = (
                self.motion.ends[self.motion.clip_ids(self.frame), None] - raw - 1
            ).clamp(0, 100).float() / 100
            ref = {k: v[:, : self.cfg.horizon + 1] for k, v in self._reference.items()}
            self._reference_links = self._reference_links_fn(ref)
        return self._refresh(), {}

    def _observation_noise(self, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        if not self.cfg.observation_noise:
            return torch.zeros(shape, device=self.device)
        return (
            torch.rand(shape, device=self.device, generator=self._generator) * 2 - 1
        ) * self.task.noise_scale

    def _refresh(self, *, noise_goal: bool = True) -> TensorDict:
        state = unpack_robot_state(self.state)
        links = self.task.links(state)
        self._features = self.task.features(state, links)
        future = {
            k: v[:, self._steps + self.task.future_offsets]
            for k, v in self._reference.items()
        }
        clean_goal = self.task.goal(state, future)
        goal = clean_goal
        if noise_goal and self.cfg.goal_position_noise:
            offsets = len(self.task.cfg.future_offsets)
            split = goal.reshape(*goal.shape[:-1], offsets, goal.shape[-1] // offsets)
            noise = (
                torch.rand(
                    split[..., :3].shape, device=self.device, generator=self._generator
                )
                * 2
                - 1
            ) * self.cfg.goal_position_noise
            goal = torch.cat((split[..., :3] + noise, split[..., 3:]), -1).flatten(-2)
        self._observation = TensorDict(
            {
                "policy": torch.cat((self.history.flatten(1), goal), -1),
                "critic": torch.cat(
                    (
                        self._features,
                        clean_goal,
                        self.context,
                        self._remaining[:, self._steps, None],
                    ),
                    -1,
                ),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
        return self._observation

    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance state/history and expose graph-connected reward and continuation."""
        if self._steps >= self.cfg.horizon:
            raise RuntimeError(
                "Reset before collecting another complete imagination rollout."
            )
        index = self._steps
        executed = action
        if self.cfg.action_noise:
            executed = (
                action
                + torch.randn(
                    action.shape, device=self.device, generator=self._generator
                )
                * self.cfg.action_noise
            )
        reference = {k: v[:, index + 1] for k, v in self._reference.items()}
        future = {
            k: v[:, index + 1 + self.task.future_offsets]
            for k, v in self._reference.items()
        }
        ref_links = LinkKinematics(
            **{
                name: getattr(self._reference_links, name)[:, index + 1]
                for name in (
                    "position",
                    "rotation",
                    "linear_velocity",
                    "angular_velocity",
                )
            }
        )
        previous_action = action if self.cfg.clean_previous_action else executed
        noise = self._observation_noise((self.num_envs, self.task.observation_dim))
        goal_noise = None
        if self.cfg.goal_position_noise:
            goal_noise = (
                torch.rand(
                    (self.num_envs, len(self.task.cfg.future_offsets), 3),
                    device=self.device,
                    generator=self._generator,
                )
                * 2
                - 1
            ) * self.cfg.goal_position_noise
        (
            self.state,
            self.history,
            self._features,
            policy_observation,
            critic_observation,
            reward,
            terms,
            hard,
            continuation,
        ) = self._transition_fn(
            self.state,
            self.history,
            self._features,
            self.context,
            executed,
            previous_action,
            self.previous_action,
            reference,
            future,
            ref_links,
            self._remaining[:, index + 1, None],
            noise,
            goal_noise,
        )
        self.previous_action = previous_action
        self._steps += 1
        self._observation = output = TensorDict(
            {"policy": policy_observation, "critic": critic_observation},
            batch_size=[self.num_envs],
            device=self.device,
        )
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (
            output,
            reward,
            done,
            done.clone(),
            {
                "continuation": continuation,
                "valid": self._valid[:, index],
                "next_valid": self._valid[:, index + 1],
                "hard_failure": hard.detach(),
                "reward_terms": {k: v.detach() for k, v in terms.items()},
            },
        )

    def _transition(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        features: torch.Tensor,
        context: torch.Tensor,
        executed: torch.Tensor,
        action: torch.Tensor,
        previous: torch.Tensor,
        reference: Mapping[str, torch.Tensor],
        future: Mapping[str, torch.Tensor],
        ref_links: LinkKinematics,
        remaining: torch.Tensor,
        noise: torch.Tensor,
        goal_noise: torch.Tensor | None,
    ) -> tuple:
        """Evaluate one tensor transition; RNG and episode bookkeeping stay outside."""
        bias = context[:, 1 : 1 + self.task.dof]
        setpoint, _ = self.task.control(executed, reference["q"], bias)
        predicted = self.model(state, setpoint, context, features=features)
        state_dict = unpack_robot_state(predicted)
        observation = self.task.observation(state_dict, action, bias) + noise
        history = torch.cat((history[:, 1:], observation[:, None]), 1)
        links = self.task.links(state_dict)
        features = self.task.features(state_dict, links)
        goal = self.task.goal(state_dict, future)
        policy_goal = goal
        if goal_noise is not None:
            offsets = len(self.task.cfg.future_offsets)
            split = goal.reshape(*goal.shape[:-1], offsets, goal.shape[-1] // offsets)
            policy_goal = torch.cat(
                (split[..., :3] + goal_noise, split[..., 3:]), -1
            ).flatten(-2)
        policy_observation = torch.cat((history.flatten(1), policy_goal), -1)
        critic_observation = torch.cat((features, goal, context, remaining), -1)
        reward, terms = self.task.reward(
            state_dict, reference, action, previous, links, ref_links
        )
        hard, continuation = self.task.termination(state_dict, reference)
        return (
            predicted,
            history,
            features,
            policy_observation,
            critic_observation,
            reward,
            {name: value.detach() for name, value in terms.items()},
            hard,
            continuation,
        )

    def detach_state(self) -> TensorDict:
        """Cut recurrent gradients while preserving state, reference progress and noise."""
        self.state, self.history, self.previous_action = (
            self.state.detach(),
            self.history.detach(),
            self.previous_action.detach(),
        )
        self._features = self._features.detach()
        self._observation = self._observation.detach()
        return self._observation

    def close(self) -> None:
        """Release no external resources; this environment owns tensor state only."""


@register_learning_env("MotionTrackingModelRL")
def _build_motion_model_environment(
    num_envs: int,
    device: torch.device,
    *,
    model: KinematicWorldModel,
    motion: MotionLibrary,
    task: MotionTrackingTask,
    start_provider: Callable[[], MotionModelStart],
    **cfg: Any,
) -> MotionModelEnv:
    """Combine scalar configuration with workflow-owned model and replay resources."""
    return MotionModelEnv(
        num_envs,
        device,
        model=model,
        motion=motion,
        task=task,
        cfg=MotionModelCfg(**cfg),
        start_provider=start_provider,
    )
