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

"""Kinematic-feature dynamics with heading-frame state increments.

The architecture follows SuperMimic's fixed-period state prediction recipe.
Geometry and feature evaluation use EmbodiChain's differentiable tree kinematics.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from embodichain.compute.kinematics import (
    TreeKinematics,
    matrix_product,
    matrix_vector_product,
    heading_rotation,
    heading_kinematic_features,
)
from embodichain.learning.models.mlp import MLP
from embodichain.learning.models.normalizer import EmpiricalNormalizer
from embodichain.utils import configclass
from .robot_state import _parts, _body, rotation_exp, rotation_log

__all__ = [
    "KinematicWorldModelCfg",
    "KinematicWorldModel",
    "heading_state_increment",
    "integrate_heading_state",
]


def heading_state_increment(
    state: torch.Tensor, next_state: torch.Tensor
) -> torch.Tensor:
    """Encode displacement/velocity in heading axes and angular velocity in body axes.

    Args:
        state: Canonical matrix-based robot states, width ``18 + 2*J``.
        next_state: Aligned states with the same shape.

    Returns:
        Increments of width ``12 + 2*J``. Rotation is a right-multiplied
        axis-angle increment; angular velocity compares each state's own body frame.
    """
    if state.shape != next_state.shape:
        raise ValueError("Current and next state shapes must match")
    p, r, v, w, q, qd = _parts(state)
    p1, r1, v1, w1, q1, qd1 = _parts(next_state)
    h = heading_rotation(r)
    return torch.cat(
        (
            _body(h, p1 - p),
            rotation_log(matrix_product(r.transpose(-1, -2), r1)),
            _body(h, v1 - v),
            _body(r1, w1) - _body(r, w),
            q1 - q,
            qd1 - qd,
        ),
        -1,
    )


def integrate_heading_state(
    state: torch.Tensor, increment: torch.Tensor
) -> torch.Tensor:
    """Integrate heading-frame increments without detaching either input.

    Args:
        state: Canonical states with world root-origin velocity.
        increment: Aligned increments of width ``state.shape[-1] - 6``.

    Returns:
        Newly allocated canonical states with proper rotation matrices.
    """
    if increment.shape != state.shape[:-1] + (state.shape[-1] - 6,):
        raise ValueError("Increment shape does not match robot state")
    p, r, v, w, q, qd = _parts(state)
    dp, dr, dv, dw, dq, dqd = increment.split(
        (3, 3, 3, 3, q.shape[-1], q.shape[-1]), -1
    )
    h = heading_rotation(r)
    r1 = matrix_product(r, rotation_exp(dr))
    world = lambda rotation, value: matrix_vector_product(rotation, value)
    return torch.cat(
        (
            p + world(h, dp),
            r1.flatten(-2),
            v + world(h, dv),
            world(r1, _body(r, w) + dw),
            q + dq,
            qd + dqd,
        ),
        -1,
    )


@configclass
class KinematicWorldModelCfg:
    """Configure a fixed-period kinematic-feature dynamics network.

    Joint order, resting pose and action semantics form checkpoint identity.
    The default three 2048-wide SiLU layers match the G1 integration baseline.
    """

    joint_names: tuple[str, ...] = ()
    resting_position: tuple[float, ...] = ()
    action_dim: int = 29
    context_dim: int = 33
    hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    step_dt: float = 0.02
    action_semantics: str = "normalized_joint_position_setpoint"
    normalization_epsilon: float = 1e-3
    amp_bfloat16: bool = False


class KinematicWorldModel(nn.Module):
    """Predict next states from full-body features, controls and physical context.

    Args:
        urdf: Robot geometry XML. Only kinematic fields enter computation.
        cfg: Ordered joints, control identity and network configuration.

    Input statistics use a quadratic epsilon and clipping at ten standard
    deviations. Output statistics scale the predicted 70-dimensional increment
    for a 29-joint robot. Statistics change only through explicit measured-data
    updates. FK geometry is cached once per device, outside repeated tensor work.
    """

    def __init__(self, urdf: str, cfg: KinematicWorldModelCfg) -> None:
        super().__init__()
        cfg.validate()
        if (
            not cfg.joint_names
            or len(set(cfg.joint_names)) != len(cfg.joint_names)
            or len(cfg.resting_position) != len(cfg.joint_names)
            or type(cfg.action_dim) is not int
            or cfg.action_dim < 1
            or type(cfg.context_dim) is not int
            or cfg.context_dim < 0
            or not cfg.hidden_dims
            or any(type(w) is not int or w < 1 for w in cfg.hidden_dims)
            or not 0 < cfg.step_dt < float("inf")
            or not cfg.action_semantics
        ):
            raise ValueError("Invalid kinematic world model configuration")
        self.cfg, self.urdf = cfg, urdf
        self._trees: dict[torch.device, tuple[TreeKinematics, torch.Tensor]] = {}
        tree, _ = self._geometry(torch.device("cpu"))
        self.state_dim = 18 + 2 * len(cfg.joint_names)
        self.increment_dim = self.state_dim - 6
        self.feature_dim = (
            13 + 2 * len(cfg.joint_names) + 6 * (len(tree.moving_link_names) - 1)
        )
        self.input_dim = self.feature_dim + cfg.action_dim + cfg.context_dim
        self.register_buffer(
            "resting_position", torch.tensor(cfg.resting_position, dtype=torch.float32)
        )
        self.input_normalizer = EmpiricalNormalizer(
            self.input_dim, cfg.normalization_epsilon, variance_epsilon=True, clip=10.0
        )
        self.output_normalizer = EmpiricalNormalizer(
            self.increment_dim, 1e-4, variance_epsilon=True
        )
        self.net = MLP(
            self.input_dim, self.increment_dim, cfg.hidden_dims, activation="silu"
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)
        self._statistics_frozen = False

    def _geometry(self, device: torch.device) -> tuple[TreeKinematics, torch.Tensor]:
        if device not in self._trees:
            tree = TreeKinematics(self.urdf, self.cfg.joint_names, device=device)
            self._trees[device] = tree, tree.link_indices(tree.moving_link_names[1:])
        return self._trees[device]

    def features(self, state: torch.Tensor) -> torch.Tensor:
        """Return differentiable heading-frame root/joint/link features."""
        if (
            state.ndim < 2
            or state.shape[-1] != self.state_dim
            or state.dtype != torch.float32
        ):
            raise ValueError(f"State must be FP32 with width {self.state_dim}")
        p, r, v, w, q, qd = _parts(state)
        tree, moving = self._geometry(state.device)
        links = tree.evaluate(p, r, q, v, w, qd)
        return heading_kinematic_features(
            p,
            r,
            v,
            w,
            q,
            qd,
            self.resting_position,
            links.position.index_select(-2, moving),
            links.linear_velocity.index_select(-2, moving),
        )

    def inputs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode aligned states, executed setpoints and recorded physical context."""
        if context is None and self.cfg.context_dim == 0:
            context = state.new_empty(state.shape[:-1] + (0,))
        for name, value, width in (
            ("action", action, self.cfg.action_dim),
            ("context", context, self.cfg.context_dim),
        ):
            if (
                value is None
                or value.shape != state.shape[:-1] + (width,)
                or value.dtype != torch.float32
                or value.device != state.device
            ):
                raise ValueError(f"{name} must be aligned FP32 data with width {width}")
        features = self.features(state) if features is None else features
        if (
            features.shape != state.shape[:-1] + (self.feature_dim,)
            or features.dtype != torch.float32
            or features.device != state.device
        ):
            raise ValueError(
                "Precomputed kinematic features must be aligned FP32 tensors."
            )
        return torch.cat((features, action, context), -1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one period; optionally reuse graph-connected task FK features.

        Precomputed features must use this model's geometry, joint order and
        resting pose. Normalization and integration retain FP32 arithmetic.
        """
        with torch.autocast(state.device.type, enabled=False):
            normalized = self.input_normalizer(
                self.inputs(state, action, context, features=features)
            )
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.cfg.amp_bfloat16 and state.is_cuda
            else nullcontext()
        )
        with amp:
            prediction = self.net(normalized)
        with torch.autocast(state.device.type, enabled=False):
            increment = (
                prediction.float() * self.output_normalizer.scale
                + self.output_normalizer.mean
            )
            return integrate_heading_state(state, increment)

    @torch.no_grad()
    def update_normalization(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> None:
        """Merge measured training transitions; validation and imagined data stay excluded."""
        if self._statistics_frozen:
            raise RuntimeError("World model normalization is frozen during prediction")
        self.input_normalizer.update(
            self.inputs(state, action, context).reshape(-1, self.input_dim)
        )
        self.output_normalizer.update(
            heading_state_increment(state, next_state).reshape(-1, self.increment_dim)
        )

    def state_error(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Measure error in target heading axes and each state's body angular axes."""
        return heading_state_increment(target, predicted)

    def error_scale(self) -> torch.Tensor:
        """Return measured increment scales including the quadratic epsilon."""
        return self.output_normalizer.scale

    def get_extra_state(self) -> dict[str, Any]:
        """Save geometry and state/control identity beside trainable weights."""
        return {
            "format": "kinematic-world-model-v1",
            "config": self.cfg.to_dict(),
            "urdf": self.urdf,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Require the same geometry, state layout and control semantics on load."""
        if state != self.get_extra_state():
            raise ValueError("Kinematic world model checkpoint contract differs")
