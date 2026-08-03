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

"""Capability-aware NMG adapter stub reserved for a future checkpoint."""

from __future__ import annotations

from embodichain.lab.sim.planners import PlanResult

from ..models import BenchmarkCase
from ..registry import register_planner_adapter
from .base import PlannerAdapter

__all__ = ["NeuralAdapterStub"]


class NeuralAdapterStub(PlannerAdapter):
    """Expose configurable NMG precision without initializing an unavailable model."""

    capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})
    model_revision = "not-ready"

    def availability(self) -> tuple[bool, str | None]:
        """Mark the placeholder unsupported until the checkpoint contract lands."""
        pos_eps = float(self.spec.config.get("pos_eps", 0.05))
        rot_eps = float(self.spec.config.get("rot_eps", 0.3))
        return (
            False,
            "NMG adapter is a stub pending the production checkpoint; "
            f"configured pos_eps={pos_eps} m, rot_eps={rot_eps} rad.",
        )

    def build(self) -> None:
        """Reject accidental construction of the explicit placeholder."""
        raise RuntimeError("The NMG adapter is not implemented yet.")

    def plan(self, case: BenchmarkCase) -> PlanResult:  # noqa: ARG002
        """Reject accidental execution of the explicit placeholder."""
        raise RuntimeError("The NMG adapter is not implemented yet.")


register_planner_adapter("neural_stub", NeuralAdapterStub)
