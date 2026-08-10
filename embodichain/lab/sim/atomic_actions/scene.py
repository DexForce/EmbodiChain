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

"""Scene-observation boundary for dynamic atomic-action execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .state import SceneSnapshot


@runtime_checkable
class SceneProvider(Protocol):
    """Produce scene snapshots correlated with execution environments.

    Implementations own scene-change detection and revision advancement. A
    snapshot's entity rows must follow the supplied ``env_ids`` order. Scene
    and collision-world revisions must never regress for a stable environment.
    """

    def snapshot(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> SceneSnapshot:
        """Capture the latest versioned scene state.

        Args:
            timestamp: Observation timestamp supplied by the execution backend.
            env_ids: Stable ordered environment correlation IDs.

        Returns:
            Scene snapshot whose batched entities follow ``env_ids`` order.
        """


__all__ = ["SceneProvider"]
