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

"""Graph-preserving robot world model prediction state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .model import RobotWorldModel

__all__ = ["RWMPredictor"]


class RWMPredictor:
    """Retain a batched prediction state without native simulation buffers.

    Args:
        model: Caller-owned world model.
        initial_state: Initial canonical FP32 state batch.

    Calls allocate new states, preserving outstanding results and unselected
    environment graphs. A caller performing policy optimization should use
    ``frozen_world_model`` around the rollout and dependent backward work.
    """

    def __init__(self, model: RobotWorldModel, initial_state: torch.Tensor) -> None:
        self.model = model
        self._state = initial_state.clone()
        if self._state.ndim != 2 or self._state.shape[-1] != model.state_dim:
            raise ValueError("Predictor initial state must be a canonical state batch")

    @property
    def state(self) -> torch.Tensor:
        """Current graph-connected state; callers must not mutate its storage."""
        return self._state

    def step(
        self, action: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Advance one transition and retain the returned graph-connected state.

        Args:
            action: One aligned control per environment.
            context: Optional physical context.

        Returns:
            Newly allocated canonical states.
        """
        self._state = self.model(self._state, action, context)
        return self._state

    def reset(
        self, initial_state: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Replace all or selected states without overwriting old graph tensors.

        Args:
            initial_state: Full replacement batch, or one state per selected id.
            env_ids: Optional unique one-dimensional environment indices.

        Returns:
            Current state after replacement. Unselected rows retain their graph.
        """
        if (
            initial_state.dtype != self._state.dtype
            or initial_state.device != self._state.device
        ):
            raise ValueError("Predictor reset state dtype and device must match")
        if env_ids is None:
            if initial_state.shape != self._state.shape:
                raise ValueError("Full predictor reset must preserve batch layout")
            self._state = initial_state.clone()
        else:
            if (
                env_ids.ndim != 1
                or env_ids.dtype != torch.int64
                or initial_state.shape != (len(env_ids), self.model.state_dim)
            ):
                raise ValueError(
                    "Selective predictor reset requires aligned int64 indices"
                )
            if env_ids.unique().numel() != env_ids.numel():
                raise ValueError("Selective predictor reset indices must be unique")
            self._state = self._state.index_copy(0, env_ids, initial_state)
        return self._state

    def detach_state(self) -> torch.Tensor:
        """Cut the current prediction graph while preserving every state value.

        Returns:
            The detached state. No reset, resampling or statistics update occurs.
        """
        self._state = self._state.detach()
        return self._state

    def state_dict(self) -> dict[str, Any]:
        """Return an owned graph-free prediction-state snapshot.

        Returns:
            Canonical state and model identity; model weights remain owned by
            the learning checkpoint.
        """
        return {
            "format": "rwm-predictor-v1",
            "contract": self.model.get_extra_state(),
            "state": self._state.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore prediction state under the same physical/action contract.

        Args:
            state: Snapshot from :meth:`state_dict`.
        """
        value = state["state"]
        if (
            state.get("format") != "rwm-predictor-v1"
            or state.get("contract") != self.model.get_extra_state()
            or value.shape != self._state.shape
            or value.dtype != self._state.dtype
        ):
            raise ValueError("Predictor snapshot contract or state layout differs")
        self._state = value.detach().to(self._state.device).clone()
