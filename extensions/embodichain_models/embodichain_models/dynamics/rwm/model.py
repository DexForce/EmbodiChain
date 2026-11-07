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

"""Configurable robot world model with frame-local, FP32 state integration.

The state-increment recipe is inspired by SuperMimic's robot world model:
https://github.com/Motphys/Uni-SuperMimic/tree/ceafbc863c52b3d613d0e19976794c5a31b9b402.
This module provides a standalone supervised model; it contains no reward,
motion reference, policy learner or simulation dependency.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from .robot_state import (
    integrate_robot_state,
    robot_state_features,
    robot_state_increment,
)
from embodichain.learning.models.mlp import MLP
from embodichain.learning.models.normalizer import EmpiricalNormalizer
from embodichain.utils import configclass

__all__ = ["RobotWorldModelCfg", "RobotWorldModel", "frozen_world_model"]


@configclass
class RobotWorldModelCfg:
    """Configure a fixed-control-period robot state increment model.

    Args:
        num_joints: Number of ordered joint position/velocity entries.
        action_dim: Width of the explicitly defined control vector.
        context_dim: Width of recorded physical context, such as friction.
        hidden_dims: Hidden MLP widths.
        step_dt: Fixed control period represented by each prediction, seconds.
        joint_names: Optional ordered identifiers included in checkpoint identity.
        action_semantics: Meaning of controls, retained in checkpoint identity.
        amp_bfloat16: Use BF16 network computation on CUDA; state math is FP32.
        normalization_epsilon: Input standard-deviation stabilizer.
    """

    num_joints: int = 1
    action_dim: int = 1
    context_dim: int = 0
    hidden_dims: tuple[int, ...] = (256, 256)
    step_dt: float = 0.02
    joint_names: tuple[str, ...] = ()
    action_semantics: str = "generalized_effort"
    amp_bfloat16: bool = False
    normalization_epsilon: float = 1e-2


class RobotWorldModel(nn.Module):
    """Predict body-frame increments and integrate a canonical robot state.

    Args:
        cfg: Model dimensions and physical/control identity. Canonical states
            have width ``18 + 2*num_joints``; see ``embodichain_models.dynamics.rwm.robot_state``.

    Statistics update only through :meth:`update_normalization`, never during
    :meth:`forward`. The model starts as an identity transition because its
    final layer is zero-initialized. This does not imply a physics prior.
    """

    def __init__(self, cfg: RobotWorldModelCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else RobotWorldModelCfg()
        self.cfg.validate()
        if (
            type(self.cfg.num_joints) is not int
            or self.cfg.num_joints < 0
            or type(self.cfg.action_dim) is not int
            or self.cfg.action_dim < 1
            or type(self.cfg.context_dim) is not int
            or self.cfg.context_dim < 0
            or not self.cfg.hidden_dims
            or any(
                type(width) is not int or width < 1 for width in self.cfg.hidden_dims
            )
            or not 0 < self.cfg.step_dt < float("inf")
            or not self.cfg.action_semantics
        ):
            raise ValueError("Invalid robot world model dimensions or control period")
        if self.cfg.joint_names and (
            len(self.cfg.joint_names) != self.cfg.num_joints
            or len(set(self.cfg.joint_names)) != self.cfg.num_joints
            or any(not name for name in self.cfg.joint_names)
        ):
            raise ValueError(
                "joint_names must identify every ordered joint exactly once"
            )
        self.state_dim = 18 + 2 * self.cfg.num_joints
        self.increment_dim = 12 + 2 * self.cfg.num_joints
        self.input_dim = (
            10 + 2 * self.cfg.num_joints + self.cfg.action_dim + self.cfg.context_dim
        )
        self.input_normalizer = EmpiricalNormalizer(
            self.input_dim, self.cfg.normalization_epsilon
        )
        self.output_normalizer = EmpiricalNormalizer(self.increment_dim, 1e-4)
        self.net = MLP(
            self.input_dim,
            self.increment_dim,
            self.cfg.hidden_dims,
            activation="silu",
            use_layernorm=True,
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self._statistics_frozen = False

    def _features(
        self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor | None
    ) -> torch.Tensor:
        if (
            state.ndim < 2
            or state.shape[-1:] != (self.state_dim,)
            or state.dtype != torch.float32
        ):
            raise ValueError(f"state must be FP32 with width {self.state_dim}")
        if action.shape != state.shape[:-1] + (self.cfg.action_dim,):
            raise ValueError(
                "action shape does not match the state batch and configured width"
            )
        if action.dtype != torch.float32 or action.device != state.device:
            raise ValueError("actions must be FP32 on the state device")
        if context is None:
            if self.cfg.context_dim:
                raise ValueError(
                    "Configured physical context must be provided explicitly"
                )
            context = state.new_empty(state.shape[:-1] + (0,))
        if context.shape != state.shape[:-1] + (self.cfg.context_dim,):
            raise ValueError(
                "context shape does not match the state batch and configured width"
            )
        if context.dtype != torch.float32 or context.device != state.device:
            raise ValueError("context must be FP32 on the state device")
        return torch.cat((robot_state_features(state), action, context), -1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one fixed-period transition without detaching inputs.

        Args:
            state: Canonical FP32 robot states.
            action: Aligned FP32 controls with configured semantics.
            context: Aligned physical context, required when its width is nonzero.

        Returns:
            FP32 next states retaining state/action gradients even when every
            model parameter has ``requires_grad=False``.
        """
        # An enclosing learner autocast may cover the whole objective. Keep
        # normalization and physical integration in FP32 independently of it.
        with torch.autocast(state.device.type, enabled=False):
            features = self._features(state, action, context)
            normalized = self.input_normalizer(features)
        enabled = self.cfg.amp_bfloat16 and state.device.type == "cuda"
        network_context = (
            torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()
        )
        with network_context:
            prediction = self.net(normalized)
        with torch.autocast(state.device.type, enabled=False):
            increment = (
                prediction.float()
                * (
                    self.output_normalizer.standard_deviation
                    + self.output_normalizer.epsilon
                )
                + self.output_normalizer.mean
            )
            return integrate_robot_state(state, increment)

    @torch.no_grad()
    def update_normalization(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> None:
        """Merge an explicitly selected batch of real training transitions.

        Args:
            state: Current states from the training distribution.
            action: Controls that produced ``next_state``.
            next_state: Physical next states captured before reset.
            context: Recorded physical context.
        """
        if self._statistics_frozen:
            raise RuntimeError("World model normalization is frozen during prediction")
        self.input_normalizer.update(
            self._features(state, action, context).reshape(-1, self.input_dim)
        )
        self.output_normalizer.update(
            robot_state_increment(state, next_state).reshape(-1, self.increment_dim)
        )

    def get_extra_state(self) -> dict[str, Any]:
        """Return model layout and action identity saved beside tensor weights.

        Returns:
            Versioned configuration including state, joint and action semantics.
        """
        return {"format": "robot-world-model-v1", "config": self.cfg.to_dict()}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Reject weights belonging to a different state or action contract.

        Args:
            state: Identity returned by :meth:`get_extra_state`.
        """
        if state != self.get_extra_state():
            raise ValueError("Robot world model checkpoint contract differs")


@contextmanager
def frozen_world_model(model: RobotWorldModel) -> Iterator[None]:
    """Freeze parameters and statistics while retaining state/action autograd.

    Args:
        model: Model to use for a dependent rollout and backward pass.

    Yields:
        None. Complete dependent backward work before leaving the context and
        before changing model parameters; this context does not use no_grad.
    """
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    frozen = model._statistics_frozen
    try:
        model._statistics_frozen = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        model._statistics_frozen = frozen
        for parameter, enabled in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(enabled)
