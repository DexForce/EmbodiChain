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

"""Small typed runtime views over the serialized execution program."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import StateDelta

from .state import ExecutionState

__all__ = [
    "ActionOutcome",
    "ExecutionEdge",
    "ExecutionProgram",
    "ExecutionReport",
    "ExecutionResult",
    "GroundedAction",
    "SemanticStep",
]


@dataclass(frozen=True)
class ExecutionEdge:
    """One executable DAG edge containing symbolic atomic actions."""

    id: str
    source: str
    target: str
    actions: tuple[dict[str, Any], ...]
    depends_on: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticStep:
    """One closed-loop intent expanded into one or more execution edges."""

    id: str
    parent_step_id: str
    operator: str
    object_uid: str
    actor: dict[str, Any]
    goal: dict[str, Any]
    depends_on: tuple[str, ...]
    postcondition: dict[str, Any]
    edge_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionProgram:
    """Validated in-memory form of ``action_engine_execution_program_v1``."""

    raw: dict[str, Any]
    task: str
    start: str
    goal: str
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[ExecutionEdge, ...]
    semantic_steps: tuple[SemanticStep, ...]
    allocation_groups: tuple[dict[str, Any], ...]
    seed_graph: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionProgram":
        """Construct an immutable runtime view from a validated mapping."""
        raw = deepcopy(dict(value))
        edges = tuple(
            ExecutionEdge(
                id=str(edge["id"]),
                source=str(edge["source"]),
                target=str(edge["target"]),
                actions=tuple(
                    deepcopy(dict(action))
                    for action in edge.get("actions", edge.get("symbolic_actions", ()))
                ),
                depends_on=tuple(str(item) for item in edge.get("depends_on", ())),
                resources=tuple(str(item) for item in edge.get("resources", ())),
            )
            for edge in raw["edges"]
        )
        steps = tuple(
            SemanticStep(
                id=str(step["id"]),
                parent_step_id=str(step["parent_step_id"]),
                operator=str(step["operator"]),
                object_uid=str(step.get("object", step.get("object_uid", ""))),
                actor=deepcopy(dict(step["actor"])),
                goal=deepcopy(dict(step.get("goal", {}))),
                depends_on=tuple(str(item) for item in step.get("depends_on", ())),
                postcondition=deepcopy(dict(step.get("postcondition", {}))),
                edge_ids=tuple(str(item) for item in step["edge_ids"]),
            )
            for step in raw["semantic_steps"]
        )
        return cls(
            raw=raw,
            task=str(raw.get("task", raw.get("task_name", "task"))),
            start=str(raw["start"]),
            goal=str(raw["goal"]),
            nodes=tuple(deepcopy(raw["nodes"])),
            edges=edges,
            semantic_steps=steps,
            allocation_groups=tuple(
                deepcopy(dict(group)) for group in raw.get("allocation_groups", ())
            ),
            seed_graph=None,
        )


@dataclass(frozen=True)
class GroundedAction:
    """A public atomic-action target resolved from the current simulator state."""

    action_class: str
    arm: str
    control: str
    target: Any
    cfg: dict[str, Any]
    object_pose: torch.Tensor | None = None
    reference_pose: torch.Tensor | None = None
    target_object_pose: torch.Tensor | None = None
    motion_policy: dict[str, Any] = field(default_factory=dict)
    object_uid: str | None = None
    """Scene UID of the object whose semantic step produced this action."""


@dataclass
class ActionOutcome:
    """Planning output kept in full-robot coordinates."""

    trajectory: torch.Tensor
    success: torch.Tensor
    next_state: ExecutionState
    grounded: GroundedAction
    prior_state: ExecutionState | None = None
    expected_effects: StateDelta | None = None
    planner_trace: dict[str, Any] = field(default_factory=dict)

    def state_after(self, verified: torch.Tensor) -> ExecutionState:
        """Commit expected effects only for physically verified rows."""
        if self.prior_state is None or self.expected_effects is None:
            return self.next_state
        mask = torch.as_tensor(
            verified,
            dtype=torch.bool,
            device=self.trajectory.device,
        ).reshape(-1)
        if mask.numel() != self.trajectory.shape[0]:
            raise ValueError("Verified mask must match the ActionOutcome batch.")
        terminal_qpos = (
            self.trajectory[:, -1]
            if self.trajectory.shape[1]
            else self.prior_state.last_qpos
        )
        qpos = torch.where(
            mask[:, None],
            terminal_qpos,
            self.prior_state.last_qpos,
        )
        task = self.expected_effects.apply(
            self.prior_state.to_task_state(),
            mask,
        )
        return ExecutionState.from_task_state(task, last_qpos=qpos)

    @property
    def cost(self) -> torch.Tensor:
        """Return joint-path length for each vectorized environment."""
        if self.trajectory.shape[1] < 2:
            return torch.zeros(
                self.trajectory.shape[0],
                dtype=torch.float32,
                device=self.trajectory.device,
            )
        return torch.linalg.vector_norm(
            torch.diff(self.trajectory, dim=1),
            dim=-1,
        ).sum(dim=1)


@dataclass
class ExecutionResult(Sequence[torch.Tensor]):
    """Result marker used by the existing demonstration-runner contract."""

    actions: list[torch.Tensor]
    success: torch.Tensor
    semantic_success: dict[str, torch.Tensor]
    record_dir: str | None = None
    already_executed: bool = True
    retry_count: int = 0
    recovery_count: int = 0
    revision_count: int = 0
    failure_events: list[dict[str, Any]] = field(default_factory=list)
    runtime_revisions: list[dict[str, Any]] = field(default_factory=list)
    retry_counts: list[int] = field(default_factory=list)

    @property
    def runtime_success(self) -> torch.Tensor:
        return self.success

    @property
    def runtime_graph_output_dir(self) -> str | None:
        return self.record_dir

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __getitem__(self, index):
        return self.actions[index]


@dataclass(frozen=True)
class ExecutionReport:
    """JSON-safe collaboration result built from an ``ExecutionResult``.

    The runtime result deliberately keeps tensors because the legacy demo
    runner consumes them.  The collaboration boundary instead exposes only a
    compact, serializable audit view and never retains the action tensors.
    """

    task_id: str
    plan_hash: str
    action_graph_hash: str
    status: str
    run_id: str
    episode_id: str
    environments: tuple[dict[str, Any], ...] = ()
    action_count: int = 0
    retry_count: int = 0
    recovery_count: int = 0
    revision_count: int = 0
    failure_events: tuple[dict[str, Any], ...] = ()
    graph_revisions: tuple[dict[str, Any], ...] = ()
    record_dir: str | None = None
    error: str | None = None
    schema_version: str = "action_engine_execution_report_v1"

    def as_mapping(self) -> dict[str, Any]:
        """Return a detached mapping suitable for strict JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "plan_hash": self.plan_hash,
            "action_graph_hash": self.action_graph_hash,
            "status": self.status,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "environments": deepcopy(list(self.environments)),
            "action_count": self.action_count,
            "retry_count": self.retry_count,
            "recovery_count": self.recovery_count,
            "revision_count": self.revision_count,
            "failure_events": deepcopy(list(self.failure_events)),
            "graph_revisions": deepcopy(list(self.graph_revisions)),
            "record_dir": self.record_dir,
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        """Compatibility spelling for artifact and CLI publishers."""
        return self.as_mapping()


def success_mask(value: bool | torch.Tensor, count: int, device: Any) -> torch.Tensor:
    """Normalize a primitive's scalar or batched success result."""
    mask = torch.as_tensor(value, dtype=torch.bool, device=device).reshape(-1)
    if mask.numel() == 1:
        return mask.repeat(count)
    if mask.numel() != count:
        raise ValueError(
            f"Atomic action success has {mask.numel()} values; expected {count}."
        )
    return mask


def trajectory_cost_numpy(value: torch.Tensor) -> np.ndarray:
    """Expose trajectory costs to assignment solvers without retaining gradients."""
    if value.shape[1] < 2:
        return np.zeros(value.shape[0], dtype=np.float64)
    diffs = torch.diff(value.detach(), dim=1)
    return (
        torch.linalg.vector_norm(diffs, dim=-1)
        .sum(dim=1)
        .cpu()
        .numpy()
        .astype(np.float64)
    )
