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

"""Atomic evidence provider port and shared physical observation contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from embodichain.lab.sim.evidence import (
    PhysicalEvidenceBatch,
    PhysicalEvidenceFrame,
    PhysicalEvidenceRequest,
)

if TYPE_CHECKING:
    from .state import PlanningContext

__all__ = [
    "PhysicalEvidenceBatch",
    "PhysicalEvidenceFrame",
    "PhysicalEvidenceProvider",
    "PhysicalEvidenceRequest",
]


@runtime_checkable
class PhysicalEvidenceProvider(Protocol):
    """Acquire synchronized physical evidence for an execution observation."""

    def collect(
        self,
        context: "PlanningContext",
        request: PhysicalEvidenceRequest,
    ) -> PhysicalEvidenceFrame:
        """Return evidence aligned with the supplied planning context."""
        ...
