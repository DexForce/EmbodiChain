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

"""Route-explicit recovery and suffix-replanning coordinator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .recovery import RuntimeGraph

__all__ = ["DynamicRecoveryController", "RecoveryDirective"]

Replanner = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class RecoveryDirective:
    """A graph revision that must execute before suffix replanning."""

    failure_type: str
    failed_node_id: str
    recovery_group_id: str | None
    graph: dict[str, Any]
    requires_recovery_execution: bool
    active_env_ids: tuple[int, ...]


class DynamicRecoveryController:
    """Keep offline and online dynamic replanning as separately testable modes."""

    def __init__(
        self,
        runtime_graph: RuntimeGraph,
        *,
        mode: str,
        offline_replanner: Replanner | None = None,
        online_replanner: Replanner | None = None,
    ) -> None:
        if mode not in {"offline_dynamic", "online_dynamic"}:
            raise ValueError("Dynamic mode must be offline_dynamic or online_dynamic.")
        selected = offline_replanner if mode == "offline_dynamic" else online_replanner
        if not callable(selected):
            raise ValueError(f"{mode} requires its matching replanner callback.")
        self.runtime_graph = runtime_graph
        self.mode = mode
        self._replanner = selected

    def handle_failure(
        self,
        *,
        failed_node_id: str,
        failure_type: str,
        active_env_ids: Sequence[int] | None = None,
    ) -> RecoveryDirective:
        """Insert recovery when known; otherwise request immediate full replanning."""
        env_ids = tuple(
            sorted(
                set(
                    range(self.runtime_graph.num_envs)
                    if active_env_ids is None
                    else (int(env_id) for env_id in active_env_ids)
                )
            )
        )
        if not env_ids or env_ids[0] < 0 or env_ids[-1] >= self.runtime_graph.num_envs:
            raise ValueError(
                "Recovery active_env_ids are outside the environment range."
            )
        if failure_type == "object_fallen":
            graph = self.runtime_graph.insert_default_recovery(
                failed_node_id=failed_node_id,
                failure_type=failure_type,
                active_env_ids=env_ids,
            )
            group_id = self.runtime_graph.revisions[-1].inserted_group_ids[0]
            return RecoveryDirective(
                failure_type,
                failed_node_id,
                group_id,
                graph,
                True,
                self.runtime_graph.revisions[-1].active_env_ids,
            )
        return RecoveryDirective(
            failure_type,
            failed_node_id,
            None,
            self.runtime_graph.graph,
            False,
            env_ids,
        )

    def handle_execution_result(self, result: Any) -> RecoveryDirective:
        """Create a directive from the first actionable runtime failure event."""
        events = getattr(result, "failure_events", None)
        if not isinstance(events, Sequence) or not events:
            raise ValueError("Execution result contains no recoverable failure event.")
        event = next(
            (
                item
                for item in events
                if isinstance(item, Mapping) and bool(item.get("fatal", True))
            ),
            None,
        )
        if event is None:
            raise ValueError(
                "Execution result contains no fatal recoverable failure event."
            )
        if not isinstance(event, Mapping):
            raise ValueError("Execution failure events must be mappings.")
        node_id = event.get("node_id")
        failure_type = event.get("failure_type")
        env_ids = event.get("env_ids", ())
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Dynamic recovery requires a v3 SeedGraph node_id.")
        if not isinstance(failure_type, str):
            raise ValueError("Execution failure event requires failure_type.")
        return self.handle_failure(
            failed_node_id=node_id,
            failure_type=failure_type,
            active_env_ids=env_ids,
        )

    def replan(
        self,
        directive: RecoveryDirective,
        *,
        completed_group_ids: Sequence[str],
        recovery_succeeded: bool,
        observations: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Replace only the unfinished suffix after recovery or escalation."""
        if directive.requires_recovery_execution and not recovery_succeeded:
            reason = f"{directive.failure_type}:recovery_failed"
        else:
            reason = f"{directive.failure_type}:state_restored"
        replacement = self._replanner(
            graph=self.runtime_graph.graph,
            completed_group_ids=tuple(completed_group_ids),
            failure_type=directive.failure_type,
            observations=dict(observations or {}),
        )
        return self.runtime_graph.replace_unfinished_suffix(
            replacement,
            completed_group_ids=completed_group_ids,
            reason=reason,
        )
