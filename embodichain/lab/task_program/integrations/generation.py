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

"""Source-neutral generation adapters for grounded Task Program calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from embodichain.lab.sim.atomic_actions import ActionPlan, ActionPlanTemplateAdapter
from embodichain.lab.sim.motion.expansion import (
    SourceContext,
    TrajectoryTemplate,
)

__all__ = ["TaskProgramSourceAdapter"]


@dataclass(frozen=True)
class TaskProgramSourceAdapter:
    """Expose a Task Program's grounded Atomic plan through SourceAdapter."""

    atomic_adapter: ActionPlanTemplateAdapter
    kind: ClassVar[str] = "task_program"

    def __post_init__(self) -> None:
        if not isinstance(self.atomic_adapter, ActionPlanTemplateAdapter):
            raise TypeError("atomic_adapter must be an ActionPlanTemplateAdapter")

    def export_template(
        self,
        source: ActionPlan,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        """Delegate one grounded Atomic plan to the common template adapter.

        Args:
            source: Grounded Atomic plan produced for one Task Program call.
            context: Coordinator-owned source and scene identity.

        Returns:
            Complete phase-annotated trajectory template.
        """
        return self.atomic_adapter.export_template(source, context=context)
