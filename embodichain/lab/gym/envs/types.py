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

"""Shared action types for the Gym environment boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from tensordict import TensorDict

from embodichain.lab.sim.types import EnvAction

from ._json import json_safe_copy as _json_safe_copy

__all__ = ["ControllerAction"]


@dataclass(frozen=True, slots=True, eq=False)
class ControllerAction:
    """Owned controller-ready action that must still pass through ``env.step``.

    The action has already completed the raw-policy preprocessing stage. An
    :class:`~embodichain.lab.gym.envs.embodied_env.EmbodiedEnv` therefore skips
    ``ActionManager`` terms in ``pre`` mode, validates the controller command,
    and continues through the normal simulation step. Terms in ``post`` mode
    still run after the command has been applied.

    Args:
        value: Controller-ready tensor or ``TensorDict``.
        metadata: JSON-compatible producer provenance. The environment does not
            interpret this mapping.
    """

    value: EnvAction
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, (torch.Tensor, TensorDict)):
            raise TypeError("value must be a torch.Tensor or TensorDict.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        owned_value = self.value.clone()
        owned_metadata = _json_safe_copy(self.metadata, field_name="metadata")
        object.__setattr__(self, "value", owned_value)
        object.__setattr__(self, "metadata", MappingProxyType(owned_metadata))

    def snapshot(self) -> ControllerAction:
        """Return an independently owned controller-action envelope."""
        return ControllerAction(value=self.value, metadata=self.metadata)
