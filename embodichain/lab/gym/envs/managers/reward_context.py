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

"""Qualified reward inputs for one physical transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .observation_types import ObservationFrame

__all__ = ["RewardTransitionContext"]


@dataclass(frozen=True)
class RewardTransitionContext:
    """Context supplied by the host after measurement and termination evaluation.

    Args:
        frame: Canonical physical acquisition, before reset.
        terminated: Actual task terminal mask; skill completion is independent.
    """

    frame: ObservationFrame
    terminated: torch.Tensor

    def __post_init__(self) -> None:
        if self.terminated.dtype != torch.bool or self.terminated.shape != (
            len(self.frame.episode_ids),
        ):
            raise ValueError(
                "Reward terminal flags must match the observation episode rows"
            )
        object.__setattr__(self, "terminated", self.terminated.clone())

    @property
    def identity(self) -> tuple[int, tuple[int, ...]]:
        """Return acquisition and episode identity for idempotent reward reads."""
        return self.frame.revision, self.frame.episode_ids
