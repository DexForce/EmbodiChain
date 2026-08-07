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

"""Backend-independent planner adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.planners.utils import PlanResult

from ..config import PlannerSpecCfg, stable_hash
from ..models import AlgorithmRole, BenchmarkCase, PlannerMetadata

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["PlannerAdapter", "PlannerContext"]


@dataclass(frozen=True)
class PlannerContext:
    """Runtime objects shared with a planner adapter."""

    robot: "Robot"
    control_part: str
    device: torch.device
    sample_interval: int


class PlannerAdapter(ABC):
    """Uniform lifecycle around a motion-planning implementation."""

    capabilities: frozenset[str] = frozenset()
    model_revision: str = "N/A"
    separate_prepare: bool = False
    """Whether this backend exposes a distinct lazy preparation phase."""

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        self.spec = spec
        self.context = context

    @property
    def metadata(self) -> PlannerMetadata:
        """Return stable identity, role, configuration, and capabilities."""
        return PlannerMetadata(
            algorithm_id=self.spec.id,
            algorithm_role=AlgorithmRole(self.spec.role),
            adapter=self.spec.adapter,
            config_hash=stable_hash(self.spec.config),
            capabilities=self.capabilities,
            model_revision=str(
                self.spec.config.get("model_revision", self.model_revision)
            ),
            parameters=dict(self.spec.config),
        )

    def availability(self) -> tuple[bool, str | None]:
        """Return whether this adapter can run in the current process."""
        return True, None

    @abstractmethod
    def build(self) -> None:
        """Construct the underlying planner without preparing lazy backends."""

    def prepare(self, case: BenchmarkCase) -> dict[str, object] | None:
        """Prepare a lazy backend, or return ``None`` when not applicable."""
        return None

    @abstractmethod
    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Plan one env-batched benchmark case."""

    def _close_motion_generator(self) -> None:
        """Release ``motion_generator.planner`` when adapters own one."""
        motion_generator = getattr(self, "motion_generator", None)
        if motion_generator is None:
            return
        close_fn = getattr(getattr(motion_generator, "planner", None), "close", None)
        if close_fn is not None:
            close_fn()
        self.motion_generator = None

    def close(self) -> None:
        """Release backend resources when the implementation exposes them."""
