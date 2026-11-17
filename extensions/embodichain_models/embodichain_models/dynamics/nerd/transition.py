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

"""Canonical, graph-preserving NeRD rollout independent of a simulation host."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .codec import NeRDStateCodec, append_history
from .geometry import NeRDGeometry
from .model import NeRDModel

__all__ = ["NeRDState", "NeRDTransition"]


@dataclass(frozen=True)
class NeRDState:
    """Owned canonical coordinates and history; callers treat tensors as immutable.

    Args:
        q: Canonical positions, ``(batch, q_dim)``.
        qd: Canonical COM velocities, ``(batch, qd_dim)``.
        history: Ordered feature tensors, ``(batch, capacity, width)``.
        valid_lengths: Per-row valid history lengths, ``(batch,)`` int64.
    """

    q: Tensor
    qd: Tensor
    history: Mapping[str, Tensor]
    valid_lengths: Tensor

    def detach(self) -> NeRDState:
        """Return the same values at an explicit TBPTT boundary, without reset.

        Returns:
            A detached state. Its tensors remain read-only; no history is cleared.
        """
        return NeRDState(
            self.q.detach(),
            self.qd.detach(),
            {key: value.detach() for key, value in self.history.items()},
            self.valid_lengths.detach(),
        )


class NeRDTransition:
    """Compose current-state geometry, codec, history and a resident network.

    Args:
        network: Existing NeRD network; parameters/statistics must stay unchanged
            until backward on dependent rollouts has completed.
        codec: Physical coordinate and target conventions of the checkpoint.
        geometry: Fixed geometry with an injected graph-preserving FK callable.
        dt: Positive timestep represented by one network prediction.

    The transition owns no evolving state, native buffers, resets or renderer.
    Each call computes contacts from its input state and returns new history.
    Body poses of the successor are computed only by consumers that need them.
    Offline supervised training may continue using measured contact features.
    """

    def __init__(
        self,
        network: NeRDModel,
        codec: NeRDStateCodec,
        geometry: NeRDGeometry,
        dt: float,
    ) -> None:
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("NeRD prediction dt must be positive and finite")
        self.network = network
        self.codec = codec
        self.geometry = geometry
        self.dt = float(dt)
        if codec.robot == "ant" and geometry.contact_bodies.shape[1] != 25:
            raise ValueError("NeRD Ant requires 25 ordered contact slots")
        if codec.robot == "cartpole" and geometry.contact_bodies.shape[1]:
            raise ValueError("NeRD Cartpole is contact-free")

    def initial_state(self, q: Tensor, qd: Tensor) -> NeRDState:
        """Create empty feature histories without detaching the supplied state.

        Args:
            q: Canonical batched positions, optionally graph-connected leaves.
            qd: Canonical batched COM velocities.

        Returns:
            Explicit state with zero valid history lengths.
        """
        self._validate_coordinates(q, qd)
        history = {
            key: q.new_zeros((q.shape[0], self.network.cfg.history_length, width))
            for key, width in self.network.cfg.input_dims.items()
        }
        return NeRDState(
            q,
            qd,
            history,
            torch.zeros(q.shape[0], dtype=torch.long, device=q.device),
        )

    def step(
        self,
        state: NeRDState,
        effort: Tensor,
        *,
        dt: float | None = None,
        parameters: Mapping[str, Tensor] | None = None,
    ) -> NeRDState:
        """Return one complete state/FK/contact/network transition without mutation.

        Args:
            state: Canonical state and explicit history from the preceding step.
            effort: Model-order controls, ``(batch, 2)`` for Cartpole or
                ``(batch, 8)`` actuated efforts for Ant. Free-base forces cannot
                be represented by this interface.
            dt: Optional confirmation of the checkpoint prediction interval.
            parameters: Optional explicit named network parameter tensors.

        Returns:
            New state retaining state, effort, history and network gradients.
        """
        if dt is not None and (
            isinstance(dt, bool)
            or not math.isclose(float(dt), self.dt, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"NeRD timestep must equal trained dt={self.dt}, got {dt}")
        self._validate_coordinates(state.q, state.qd)
        controls = 2 if self.codec.robot == "cartpole" else 8
        if (
            effort.shape != (state.q.shape[0], controls)
            or effort.dtype != state.q.dtype
            or effort.device != state.q.device
        ):
            raise ValueError(
                "Effort must match model-order controls and state dtype/device"
            )
        if (
            state.valid_lengths.shape != (state.q.shape[0],)
            or state.valid_lengths.dtype != torch.long
            or state.valid_lengths.device != state.q.device
            or set(state.history) != set(self.network.cfg.input_dims)
        ):
            raise ValueError("History layout/lengths must match the state and network")
        for key, width in self.network.cfg.input_dims.items():
            value = state.history[key]
            if (
                value.shape
                != (state.q.shape[0], self.network.cfg.history_length, width)
                or value.dtype != state.q.dtype
                or value.device != state.q.device
            ):
                raise ValueError(f"History field {key!r} has an incompatible layout")
        contacts = self.geometry.contacts(state.q, state.qd)
        features = self.codec.encode(
            state.q, state.qd, effort, contacts, self.geometry.gravity_dir
        )
        history, lengths = append_history(state.history, state.valid_lengths, features)
        if parameters is None:
            prediction = self.network.evaluate(history, lengths).squeeze(1)
        else:
            output = torch.func.functional_call(
                self.network, parameters, (history,), strict=False
            )
            prediction = output.gather(
                1, (lengths - 1)[:, None, None].expand(-1, 1, output.shape[-1])
            ).squeeze(1)
        q, qd = self.codec.decode(
            state.q,
            state.qd,
            prediction,
            current_embedding=features["states_embedding"],
        )
        return NeRDState(q, qd, history, lengths)

    def _validate_coordinates(self, q: Tensor, qd: Tensor) -> None:
        if (
            q.shape != (self.geometry.gravity.shape[0], self.codec.q_dim)
            or qd.shape != (q.shape[0], self.codec.qd_dim)
            or q.dtype != self.geometry.gravity.dtype
            or qd.dtype != q.dtype
            or q.device != self.geometry.gravity.device
            or qd.device != q.device
        ):
            raise ValueError(
                "Canonical q/qd must match codec dimensions and geometry dtype/device"
            )
