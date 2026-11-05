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

"""Explicit planning inputs and scoped geometry services for bounded Twist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING, runtime_checkable
import torch

if TYPE_CHECKING:
    from .primitives.twist_recovery import TwistRecoverySnapshot
    from embodichain.lab.sim.objects import Robot
    from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
    from embodichain.lab.sim.motion.revolute_query import (
        RevolutePathReport,
        RevoluteDependencyReport,
    )
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext

__all__ = ["TwistGeometryQuery", "TwistPlanningSession", "TwistPlanningProvider"]


@runtime_checkable
class TwistGeometryQuery(Protocol):
    """Numeric query lifetime borrowed only within one planning attempt."""

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed robot whose samples this query accepts."""
        ...

    @property
    def scope(self) -> NativeQueryScope:
        """Return the query identity and same-observation timestamp/version."""
        ...

    @property
    def source_axis(self) -> tuple[float, float, float]:
        """Return the original source-precision signed local contact axis."""
        ...

    def inspect_free_path(
        self, positions: torch.Tensor, frame_indices: tuple[int, ...]
    ) -> RevolutePathReport:
        """Measure full-robot vertex envelopes for the supplied frame indices.

        Args:
            positions: Complete B1 full-robot trajectory samples.
            frame_indices: Explicit ordered free-motion samples to inspect.

        Returns:
            Numeric geometry qualification, never physical contact evidence.
        """
        ...

    def contact_metrics(
        self, full_qpos: torch.Tensor, start_qpos: torch.Tensor
    ) -> tuple[float, float, float]:
        """Measure source pad/parent distances without selecting a candidate.

        Args:
            full_qpos: B1 complete contact posture.
            start_qpos: Same-observation full start posture.

        Returns:
            Parent-near fraction, maximum pad distance and motion norm.
        """
        ...

    def qualify_root_dependency(
        self,
        positions: torch.Tensor,
        *,
        grasp_point: tuple[float, float, float],
        axis: torch.Tensor,
        roll: float,
    ) -> RevoluteDependencyReport:
        """Qualify a declared invariant grasp and its complete collider sweep.

        Args:
            positions: B1 complete trajectory samples.
            grasp_point: Selected link-local contact point.
            axis: Selected local contact axis.
            roll: Selected contact-frame roll.

        Returns:
            Numeric proof allowing the caller to retain or remap its dependency.
        """
        ...

    def close(self) -> None:
        """Close owned query resources, leaving borrowed native objects alive."""
        ...


@dataclass(frozen=True, slots=True)
class TwistPlanningSession:
    """Owned cursor snapshot and explicit bindings for one query lifetime.

    Args:
        state: Immutable logical cursor snapshot, never a live cursor.
        query: Newly prepared geometry query owned by this session.
        parent_entity_id: Canonical parent dependency after successful geometric proof.
        invocation_id: Exact execution occurrence that requested this preparation.
        invocation_revision: Exact request revision for that occurrence.
        completed_hold_qpos: Observed drive targets for an already-complete operation;
            absent for ordinary candidate planning. It is not physical evidence.
    """

    state: TwistRecoverySnapshot
    query: TwistGeometryQuery
    parent_entity_id: str
    invocation_id: str
    invocation_revision: int
    completed_hold_qpos: torch.Tensor | None = None

    def __post_init__(self) -> None:
        from .primitives.twist_recovery import TwistRecoverySnapshot

        if type(self.state) is not TwistRecoverySnapshot:
            raise TypeError("Twist planning requires an immutable cursor snapshot.")
        if not isinstance(self.query, TwistGeometryQuery):
            raise TypeError("Twist planning requires a numeric geometry query.")
        if (
            type(self.parent_entity_id) is not str
            or not self.parent_entity_id
            or self.parent_entity_id != self.parent_entity_id.strip()
        ):
            raise ValueError("Twist parent dependency must be canonical.")
        if (
            type(self.invocation_id) is not str
            or not self.invocation_id
            or self.invocation_id != self.invocation_id.strip()
        ):
            raise ValueError("Twist session requires its exact invocation identity.")
        if type(self.invocation_revision) is not int or self.invocation_revision < 0:
            raise ValueError("Twist invocation revision must be non-negative.")
        if self.completed_hold_qpos is not None:
            value = self.completed_hold_qpos
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[0] != 1
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Completed Twist requires finite B1 drive targets.")
            object.__setattr__(self, "completed_hold_qpos", value.detach().clone())


@runtime_checkable
class TwistPlanningProvider(Protocol):
    """Host-owned occurrence/episode binding; Atomic never selects it by names."""

    @property
    def robot(self) -> Robot:
        """Return the exact robot served by this provider."""
        ...

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> TwistPlanningSession:
        """Prepare a fresh scoped query from the exact invocation/observation.

        The provider owns cleanup if construction fails. The Atomic consumer
        always closes a returned session's query. A host commits the returned
        plan's immutable outcome separately, then adopts it before feedback.
        Preview hosts use a forked cursor; this call never advances live state.

        Args:
            request: Exact resolved Atomic invocation with its explicit policies.
            context: Same-observation robot/scene/task inputs.

        Returns:
            Independently owned planning inputs for this occurrence and episode.
        """
        ...
