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

"""Privileged-state Sharpa cylinder rotation through a learned world model."""

from __future__ import annotations

from collections.abc import Mapping
from math import radians
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from embodichain_models.dynamics.rwm import (
    RobotWorldModel,
    RobotWorldModelCfg,
    frozen_world_model,
)
from embodichain_models.dynamics.rwm.robot_state import rotation_log

from embodichain.learning.rl.env import register_learning_env
from embodichain.learning.sl import LossResult, SupervisedBatch
from embodichain.utils import configclass

__all__ = ["SharpaInHandCfg", "SharpaInHandModelEnv", "SharpaInHandWorldModel"]

_JOINT_NAMES = tuple(
    [
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
    ]
    + [
        f"right_{finger}_{joint}"
        for finger in ("index", "middle", "ring")
        for joint in ("MCP_FE", "MCP_AA", "PIP", "DIP")
    ]
    + [
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ]
)
_STATE_DIM = 62
_CONTEXT_DIM = 47  # current target(22), reset object anchor(3), reset hand pose(22)
_REFERENCE_OBJECT_POSITION = (-0.09559, -0.00517, 0.61906)
_REFERENCE_HAND_POSE = tuple(
    radians(value)
    for value in (
        95.12771,
        -3.11244,
        14.81626,
        -1.03493,
        12.23986,
        65.21091,
        6.1133,
        15.58495,
        5.90325,
        31.74149,
        -0.95812,
        41.88173,
        12.844,
        31.72383,
        9.84458,
        35.22366,
        18.02839,
        10.9712,
        68.30895,
        7.99151,
        5.89626,
        5.89875,
    )
)


@configclass
class SharpaInHandCfg:
    """Shared measured/imagined task parameters.

    Args:
        step_dt: Fixed control period in seconds.
        action_scale: Maximum incremental joint target per control step, radians.
        max_episode_steps: Maximum real episode or imagined rollout length.
        drop_height_band: Maximum absolute height deviation from the reset anchor.
        stiffness: PD gain used by the estimated torque/work reward.
        damping: PD damping used by the estimated torque/work reward.
        rotation_weight: Rotation reward coefficient; zero supports hold training.
        height_penalty: Optional smooth height-deviation penalty coefficient.
    """

    step_dt: float = 0.05
    action_scale: float = 1.0 / 24.0
    max_episode_steps: int = 400
    drop_height_band: float = 0.02
    stiffness: float = 3.0
    damping: float = 0.1
    rotation_weight: float = 2.5
    height_penalty: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid periods, dimensions and reward/control constants."""
        for name in ("step_dt", "action_scale", "drop_height_band", "stiffness"):
            if not 0 < float(getattr(self, name)) < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.damping < float("inf"):
            raise ValueError("damping must be finite and nonnegative")
        for name in ("rotation_weight", "height_penalty"):
            if not 0 <= float(getattr(self, name)) < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.max_episode_steps) is not int or self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")


class SharpaInHandWorldModel(RobotWorldModel):
    """RWM transition conditioned on object pose relative to a fixed palm.

    Args:
        hidden_dims: World-model MLP hidden widths.
        servo_context: Include commanded-minus-current joint positions as context.
        world_frame: Predict fixed-world increments rather than body increments.
        axial_symmetry: Use only the native cylinder's axis, removing its
            physically irrelevant rotation about that axis. Requires world_frame.

    The state uses the cylinder as root body and the22 ordered hand joints.
    Default body-frame increments require full predicted object pose as context
    relative to the fixed palm. World-frame increments avoid that moving basis.
    For an analytic cylinder, axial symmetry preserves its world axis while
    excluding its physically irrelevant body phase from the predictor's input.
    Future measured poses never enter an autoregressive prediction.
    Actions are the executed absolute joint-position targets, in radians.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 128),
        *,
        servo_context: bool = False,
        world_frame: bool = False,
        axial_symmetry: bool = False,
    ) -> None:
        if axial_symmetry and not world_frame:
            raise ValueError("Native axial symmetry requires world-frame increments")
        context_dim = (6 if axial_symmetry else 12) + (22 if servo_context else 0)
        semantics = (
            "sharpa-native-cylinder-world-axis-v5"
            if axial_symmetry
            else "sharpa-fixed-palm-world-v4" if world_frame else None
        )
        super().__init__(
            RobotWorldModelCfg(
                num_joints=22,
                action_dim=22,
                context_dim=context_dim,
                hidden_dims=tuple(hidden_dims),
                step_dt=0.05,
                joint_names=_JOINT_NAMES,
                action_semantics=(
                    f"{semantics}-{'servo' if servo_context else 'target'}-rad"
                    if semantics is not None
                    else (
                        "sharpa-fixed-palm-servo-delta-rad-v3"
                        if servo_context
                        else "sharpa-fixed-palm-absolute-joint-target-rad-v2"
                    )
                ),
            )
        )
        self.servo_context = servo_context
        self.world_frame = world_frame
        self.axial_symmetry = axial_symmetry

    def _context(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pose = state[..., :12]
        if self.axial_symmetry:
            axis = state[..., 3:12].reshape(*state.shape[:-1], 3, 3)[..., :, 2]
            pose = torch.cat((state[..., :3], axis), -1)
        if self.servo_context:
            return torch.cat((pose, action - state[..., 18:40]), -1)
        return pose

    def _aligned(self, state: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(3, device=state.device, dtype=state.dtype).flatten()
        return torch.cat(
            (state[..., :3], identity.expand(*state.shape[:-1], 9), state[..., 12:]),
            -1,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict from current state; external future-state context is forbidden.

        Args:
            state: Canonical cylinder/hand state.
            action: Absolute commanded hand positions.
            context: Must be omitted; context is derived from current prediction.

        Returns:
            Graph-connected next state.
        """
        if context is not None:
            raise ValueError("Fixed-palm context comes from the current prediction")
        physical_context = self._context(state, action)
        if not self.world_frame:
            return super().forward(state, action, physical_context)
        predicted = super().forward(self._aligned(state), action, physical_context)
        rotation = predicted[..., 3:12].reshape(*state.shape[:-1], 3, 3)
        rotation = rotation @ state[..., 3:12].reshape(*state.shape[:-1], 3, 3)
        return torch.cat(
            (predicted[..., :3], rotation.flatten(-2), predicted[..., 12:]), -1
        )

    @torch.no_grad()
    def update_normalization(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> None:
        """Update statistics once from newly measured transitions.

        Args:
            state: Measured current state.
            action: Executed absolute targets.
            next_state: Measured next state captured before reset.
            context: Must be omitted.
        """
        if context is not None:
            raise ValueError("Fixed-palm context comes from measured current state")
        physical_context = self._context(state, action)
        if self.world_frame:
            rotation = next_state[..., 3:12].reshape(*state.shape[:-1], 3, 3)
            rotation = rotation @ state[..., 3:12].reshape(
                *state.shape[:-1], 3, 3
            ).transpose(-1, -2)
            next_state = torch.cat(
                (next_state[..., :3], rotation.flatten(-2), next_state[..., 12:]), -1
            )
            state = self._aligned(state)
        super().update_normalization(state, action, next_state, physical_context)


class _WorldObjective:
    """Autoregressive task fitting with error axes matching the model's frame."""

    def compute_loss(
        self, model: SharpaInHandWorldModel, batch: SupervisedBatch
    ) -> LossResult:
        from embodichain_models.dynamics.rwm.robot_state import robot_state_error

        state = batch.inputs["initial_state"]
        actions, valid = batch.inputs["actions"], batch.masks["valid"]
        if (
            actions.ndim != 3
            or actions.shape[1] < 1
            or valid.shape != actions.shape[:2]
            or valid.dtype != torch.bool
            or not valid[:, 0].all()
            or (valid[:, 1:] & ~valid[:, :-1]).any()
        ):
            raise ValueError("World fitting requires nonempty aligned prefix windows")
        predictions = []
        for index in range(actions.shape[1]):
            active = valid[:, index, None]
            predicted = model(state, torch.where(active, actions[:, index], 0))
            state = torch.where(active, predicted, state)
            predictions.append(state)
        predicted, target = torch.stack(predictions, 1)[valid], batch.targets[valid]
        if model.world_frame:
            rotation = target[:, 3:12].reshape(-1, 3, 3)
            rotation = rotation @ predicted[:, 3:12].reshape(-1, 3, 3).transpose(1, 2)
            delta = target - predicted
            error = torch.cat((delta[:, :3], rotation_log(rotation), delta[:, 12:]), -1)
        else:
            error = robot_state_error(predicted, target)
        scale = model.output_normalizer.standard_deviation.clamp_min(0.01).clone()
        scale[..., :3] = 0.002
        scale[..., 2:3] = 0.001
        weights = error.new_tensor([8.0] * 12 + [1.0] * 44)
        loss = ((error / scale).square() * weights).sum(-1).mean() / weights.sum()
        return LossResult(
            loss,
            metrics={
                "position_error_m": (predicted[:, :3] - target[:, :3])
                .norm(dim=-1)
                .mean()
                .detach()
            },
            weight=float(valid.sum()),
        )


def _target(
    context: torch.Tensor,
    action: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    cfg: SharpaInHandCfg,
) -> torch.Tensor:
    return (context[..., :22] + cfg.action_scale * action.clamp(-1, 1)).clamp(
        lower, upper
    )


def _cylinder_axis_observation(observation: torch.Tensor) -> torch.Tensor:
    """Retain the native cylinder's world axis without its irrelevant phase."""
    rotation = observation[..., 3:12]
    mask = rotation.new_tensor((0, 0, 1) * 3)
    return torch.cat((observation[..., :3], rotation * mask, observation[..., 12:]), -1)


def _joint_limit_aware_action(
    proposal: torch.Tensor,
    observation: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    action_scale: float = 1 / 24,
    softness: float = 0.02,
) -> torch.Tensor:
    """Map actor proposals to feasible Sharpa deltas with exact smooth gradients.

    This changes only the policy parameterization. The simulation still applies
    the original incremental action and hard joint-limit projection. A smooth
    feasible target and rate interpolation keep that physical projection from
    creating a flat outward-command region at a joint bound.
    """
    current = lower + (observation[..., 62:84] + 1) * (upper - lower) / 2
    requested = current + 2 * action_scale * proposal
    desired = (
        lower
        + torch.nn.functional.softplus(requested - lower, beta=1 / softness)
        - torch.nn.functional.softplus(requested - upper, beta=1 / softness)
    )
    return torch.tanh((desired - current) / action_scale)


def _observation(
    state: torch.Tensor,
    context: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Complete privileged state and reset references, in fixed-palm axes."""
    span = (upper - lower).clamp_min(1e-6)
    q = 2 * (state[..., 18:40] - lower) / span - 1
    target = 2 * (context[..., :22] - lower) / span - 1
    home = 2 * (context[..., 25:47] - lower) / span - 1
    palm = state.new_tensor((0.0, 0.0, 0.5))
    return torch.cat(
        (
            (state[..., :3] - palm) * 10,
            state[..., 3:18],
            q,
            state[..., 40:62] * 0.1,
            target,
            home,
            (context[..., 22:25] - palm) * 10,
        ),
        -1,
    )


def _transition_rates(
    state: torch.Tensor,
    next_state: torch.Tensor,
    step_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Control-period linear, world angular and joint velocities."""
    rotation = state[..., 3:12].reshape(*state.shape[:-1], 3, 3)
    next_rotation = next_state[..., 3:12].reshape(*state.shape[:-1], 3, 3)
    return (
        (next_state[..., :3] - state[..., :3]) / step_dt,
        rotation_log(next_rotation @ rotation.transpose(-1, -2)) / step_dt,
        (next_state[..., 18:40] - state[..., 18:40]) / step_dt,
    )


def _stable_rotation_mask(
    alive: torch.Tensor,
    rotation: torch.Tensor,
    positive_steps: torch.Tensor,
    final_positive_steps: torch.Tensor,
    *,
    steps: int,
    final_window_steps: int,
) -> torch.Tensor:
    """Require survival, one revolution and inclusive 80% positive-step counts.

    Compare counts before displaying FP32 fractions so an exact 80/100 pass
    cannot become a failure through rounding of a normalized GPU fraction.
    """
    return (
        alive
        & (rotation >= 2 * torch.pi)
        & (5 * positive_steps >= 4 * steps)
        & (5 * final_positive_steps >= 4 * final_window_steps)
    )


def _reward(
    state: torch.Tensor,
    context: torch.Tensor,
    cfg: SharpaInHandCfg,
    *,
    previous_state: torch.Tensor,
) -> torch.Tensor:
    """Fixed source anchors and control-period PD torque/work estimates.

    Reset references still own termination and optional height shaping. They
    do not replace the source task's fixed object and hand reward references.
    Velocities come from the completed transition, as in the reference task,
    rather than sampling instantaneous simulator velocities at control ticks.
    """
    linear, angular, qvel = _transition_rates(previous_state, state, cfg.step_dt)
    q = state[..., 18:40]
    reference = state.new_tensor(_REFERENCE_OBJECT_POSITION + _REFERENCE_HAND_POSE)
    torque = cfg.stiffness * (context[..., :22] - q) - cfg.damping * qvel
    displacement = state[..., :3] - context[..., 22:25]
    return cfg.step_dt * (
        cfg.rotation_weight * angular[..., 2].clamp(-0.5, 0.5)
        - 0.3 * linear.abs().sum(-1)
        - 0.4 * (q - reference[3:]).square().sum(-1)
        - 0.1 * torque.square().sum(-1)
        - 0.5 * (torque * qvel).sum(-1).square()
        + 0.003 / ((state[..., :3] - reference[:3]).norm(dim=-1) + 0.001)
        - cfg.height_penalty * (displacement[..., 2] / cfg.drop_height_band).square()
    )


@register_learning_env("SharpaInHandModel")
class SharpaInHandModelEnv:
    """Optimize a privileged actor through a fitted Sharpa world model.

    Args:
        num_envs: Number of independent imagined trajectories.
        device: Model, state and sampling device.
        model_checkpoint: World-model export from the Sharpa online workflow.
        max_episode_steps: Imagined rollout horizon; physical evaluation uses 400.
        model: Explicit caller-owned model instead of a checkpoint, for composition.
        initial_states: Measured start states when passing an explicit model.
        initial_contexts: Targets and reset anchors aligned with start states.
        joint_limits: Ordered lower/upper positions, shape (22, 2).
        height_penalty: Optional smooth holding penalty in the imagined objective.
        rotation_weight: Rotation coefficient for a holding/spinning curriculum.

    Checkpoints contain measured starts, not fabricated initial conditions.
    The environment freezes its model parameters/statistics while preserving
    action and recurrent-state gradients. Physical policy evaluation belongs
    to the simulation acquisition host, never this learned environment.
    """

    observation_dim = 109
    action_dim = 22

    def __init__(
        self,
        num_envs: int = 64,
        device: str | torch.device = "cpu",
        *,
        model_checkpoint: str | None = None,
        max_episode_steps: int = 16,
        model: SharpaInHandWorldModel | None = None,
        initial_states: torch.Tensor | None = None,
        initial_contexts: torch.Tensor | None = None,
        joint_limits: torch.Tensor | None = None,
        height_penalty: float = 0.0,
        rotation_weight: float = 2.5,
    ) -> None:
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs, self.device = num_envs, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.cfg = SharpaInHandCfg(
            max_episode_steps=max_episode_steps,
            height_penalty=height_penalty,
            rotation_weight=rotation_weight,
        )
        if model_checkpoint is not None:
            if model is not None:
                raise ValueError("Select a checkpoint or an explicit model")
            data = torch.load(
                Path(model_checkpoint), map_location=self.device, weights_only=True
            )
            if data.get("format") != "sharpa-inhand-world-model-v1":
                raise ValueError("Unknown Sharpa world-model checkpoint format")
            model = SharpaInHandWorldModel(
                tuple(data["hidden_dims"]),
                servo_context=data.get("servo_context", False),
                world_frame=data.get("world_frame", False),
                axial_symmetry=data.get("axial_symmetry", False),
            )
            model.load_state_dict(data["model"])
            initial_states, initial_contexts, joint_limits = (
                data["initial_states"],
                data["initial_contexts"],
                data["joint_limits"],
            )
        if model is None or any(
            x is None for x in (initial_states, initial_contexts, joint_limits)
        ):
            raise ValueError(
                "A fitted model, measured start bank and joint limits are required"
            )
        self.model = model.to(self.device)
        if (
            self.model.cfg.joint_names != _JOINT_NAMES
            or self.model.cfg.step_dt != self.cfg.step_dt
            or self.model.cfg.action_dim != 22
            or self.model.cfg.context_dim
            != (6 if self.model.axial_symmetry else 12)
            + (22 if self.model.servo_context else 0)
        ):
            raise ValueError(
                "World-model joint/action/period layout differs from the Sharpa contract"
            )
        self._starts = initial_states.detach().to(self.device, torch.float32).clone()
        self._contexts = (
            initial_contexts.detach().to(self.device, torch.float32).clone()
        )
        limits = joint_limits.detach().to(self.device, torch.float32).clone()
        if (
            self._starts.ndim != 2
            or self._starts.shape[1] != _STATE_DIM
            or self._starts.shape[0] < 1
            or self._contexts.shape != (len(self._starts), _CONTEXT_DIM)
            or limits.shape != (22, 2)
            or not torch.isfinite(limits).all()
            or not (limits[:, 1] > limits[:, 0]).all()
            or not torch.isfinite(self._starts).all()
            or not torch.isfinite(self._contexts).all()
        ):
            raise ValueError("Invalid measured starts, contexts or joint limits")
        self.lower, self.upper = limits[:, 0], limits[:, 1]
        self._generator = torch.Generator(device=self.device).manual_seed(0)
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (109,), np.float32
        )
        self.single_action_space = gym.spaces.Box(-1, 1, (22,), np.float32)
        self.observation_space, self.action_space = (
            self.single_observation_space,
            self.single_action_space,
        )
        self.state = torch.empty(num_envs, 62, device=self.device)
        self.context = torch.empty(num_envs, 47, device=self.device)
        self.episode_step = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.episode_return = torch.zeros(num_envs, device=self.device)
        self.rotation = torch.zeros(num_envs, device=self.device)
        self._freeze = frozen_world_model(self.model)
        self._freeze.__enter__()
        self.reset()

    def _samples(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.randint(
            len(self._starts), (count,), device=self.device, generator=self._generator
        )
        return self._starts[ids], self._contexts[ids]

    def replace_start_bank(
        self,
        initial_states: torch.Tensor,
        initial_contexts: torch.Tensor,
    ) -> None:
        """Install measured replay starts between completed optimizer updates.

        Args:
            initial_states: Owned-by-caller canonical state bank.
            initial_contexts: Aligned current targets and original reset anchors.

        Existing state and outstanding graph tensors remain unchanged. Reset
        after replacement before starting the next independent rollout.
        """
        if (
            initial_states.ndim != 2
            or initial_states.shape[1] != 62
            or len(initial_states) < 1
            or initial_contexts.shape != (len(initial_states), 47)
            or not torch.isfinite(initial_states).all()
            or not torch.isfinite(initial_contexts).all()
        ):
            raise ValueError("Invalid measured start bank")
        self._starts = initial_states.detach().to(self.device, torch.float32).clone()
        self._contexts = (
            initial_contexts.detach().to(self.device, torch.float32).clone()
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Replace selected rows without changing outstanding graph tensors.

        Args:
            seed: Optional sampling seed.
            options: Optional unique int64 `env_ids` for selective reset.

        Returns:
            Current privileged observations and empty auxiliary info.
        """
        if seed is not None:
            self._generator.manual_seed(seed)
        ids = None if options is None else options.get("env_ids")
        if ids is None:
            self.state, self.context = self._samples(self.num_envs)
            self.episode_step = torch.zeros_like(self.episode_step)
            self.episode_return = torch.zeros_like(self.episode_return)
            self.rotation = torch.zeros_like(self.rotation)
        else:
            ids = torch.as_tensor(ids, device=self.device)
            if (
                ids.ndim != 1
                or ids.dtype != torch.int64
                or ids.unique().numel() != ids.numel()
                or (ids < 0).any()
                or (ids >= self.num_envs).any()
            ):
                raise ValueError("env_ids must be unique in-range int64 indices")
            state, context = self._samples(len(ids))
            self.state, self.context = self.state.index_copy(
                0, ids, state
            ), self.context.index_copy(0, ids, context)
            self.episode_step = self.episode_step.index_fill(0, ids, 0)
            self.episode_return = self.episode_return.index_fill(0, ids, 0)
            self.rotation = self.rotation.index_fill(0, ids, 0)
        return _observation(self.state, self.context, self.lower, self.upper), {}

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Predict one graph-connected step and auto-reset only completed rows.

        Args:
            action: Finite FP32 batch of 22 incremental joint commands.

        Returns:
            Next observation, terminal reward, dropped/time-limit masks and
            detached terminal metrics. Metrics measure imagined outcomes.
        """
        if (
            action.shape != (self.num_envs, 22)
            or action.device != self.device
            or action.dtype != torch.float32
        ):
            raise ValueError(
                "actions must be aligned FP32 tensors on the environment device"
            )
        if not torch.isfinite(action).all():
            raise ValueError("actions must be finite")
        target = _target(self.context, action, self.lower, self.upper, self.cfg)
        context = torch.cat((target, self.context[:, 22:]), -1)
        state = self.model(self.state, target)
        reward = _reward(state, context, self.cfg, previous_state=self.state)
        terminated = (state[:, 2] - context[:, 24]).abs() > self.cfg.drop_height_band
        steps = self.episode_step + 1
        truncated = steps >= self.cfg.max_episode_steps
        done = terminated | truncated
        returns = self.episode_return + reward.detach()
        _, angular, _ = _transition_rates(self.state, state, self.cfg.step_dt)
        rotation = self.rotation + angular[:, 2].detach() * self.cfg.step_dt
        info = {
            "episode": {
                "r": returns,
                "l": steps,
                "rotation_rad": rotation,
                "survival_seconds": steps * self.cfg.step_dt,
                "dropped": terminated.float(),
            },
            "_episode": done,
        }
        reset_state, reset_context = self._samples(self.num_envs)
        self.state = torch.where(done[:, None], reset_state, state)
        self.context = torch.where(done[:, None], reset_context, context)
        self.episode_step = torch.where(done, 0, steps)
        self.episode_return = torch.where(done, 0, returns)
        self.rotation = torch.where(done, 0, rotation)
        return (
            _observation(self.state, self.context, self.lower, self.upper),
            reward,
            terminated,
            truncated,
            info,
        )

    def detach_state(self) -> torch.Tensor:
        """Cut recurrent gradients without resetting values or sampling RNG.

        Returns:
            Detached current observations.
        """
        self.state, self.context = self.state.detach(), self.context.detach()
        return _observation(self.state, self.context, self.lower, self.upper)

    def close(self) -> None:
        """Restore caller model flags after dependent backward work completes."""
        if self._freeze is not None:
            self._freeze.__exit__(None, None, None)
            self._freeze = None
