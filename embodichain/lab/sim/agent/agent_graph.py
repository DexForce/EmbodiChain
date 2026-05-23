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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from embodichain.lab.sim.agent.edge_action_executor import EdgeActionExecutor

__all__ = [
    "AgentGraphEdge",
    "AgentGraphNode",
    "AgentRecoveryBranch",
    "AgentTaskGraph",
    "ExecutedActionList",
]


@dataclass
class AgentGraphNode:
    """Semantic keyframe in an atomic-action task graph."""

    id: str
    semantic: str = ""


@dataclass
class AgentGraphEdge:
    """Executable transition between two graph nodes."""

    id: str
    source: str
    target: str
    left_arm_action: Any = None
    right_arm_action: Any = None
    monitor_sequences: list[list[Any]] | None = None
    monitor_labels: list[str] = field(default_factory=list)
    is_recovery: bool = False


@dataclass
class AgentRecoveryBranch:
    """Mapping from one edge monitor trigger to recovery edge transitions."""

    edge_id: str
    monitor_index: int
    recovery_edges: list[str]


class ExecutedActionList:
    """Action sequence already executed online by the graph runtime."""

    already_executed = True

    def __init__(self, actions: list[Any]) -> None:
        self.actions = actions

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __getitem__(self, index):
        return self.actions[index]


class AgentTaskGraph:
    """Atomic-action graph with monitor-triggered recovery edge switching."""

    def __init__(self, start: str, goal: str, max_transitions: int = 1000) -> None:
        self.start = start
        self.goal = goal
        self.max_transitions = max_transitions
        self.nodes: dict[str, AgentGraphNode] = {}
        self.edges: dict[str, AgentGraphEdge] = {}
        self.outgoing: dict[str, list[str]] = defaultdict(list)
        self.recovery_branches: dict[tuple[str, int], AgentRecoveryBranch] = {}

    def add_node(self, node_id: str, semantic: str = "") -> "AgentTaskGraph":
        self.nodes[node_id] = AgentGraphNode(node_id, semantic)
        return self

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        *,
        left_arm_action=None,
        right_arm_action=None,
        monitor_sequences=None,
        monitor_labels=None,
        is_recovery: bool = False,
    ) -> "AgentTaskGraph":
        self.edges[edge_id] = AgentGraphEdge(
            id=edge_id,
            source=source,
            target=target,
            left_arm_action=left_arm_action,
            right_arm_action=right_arm_action,
            monitor_sequences=monitor_sequences,
            monitor_labels=list(monitor_labels or []),
            is_recovery=is_recovery,
        )
        self.outgoing[source].append(edge_id)
        return self

    def add_recovery(
        self,
        edge_id: str,
        monitor_index: int,
        recovery_edges: list[str],
    ) -> "AgentTaskGraph":
        self.recovery_branches[(edge_id, monitor_index)] = AgentRecoveryBranch(
            edge_id=edge_id,
            monitor_index=monitor_index,
            recovery_edges=list(recovery_edges),
        )
        return self

    def run(self, env=None, **kwargs) -> ExecutedActionList:
        current = self.start
        pending_edges: list[Any] = []
        continuation_stack: list[list[Any]] = []
        executed_actions: list[Any] = []
        transitions = 0
        disable_recovery_branches = bool(kwargs.get("disable_recovery_branches", False))
        runtime_recovery_planner = kwargs.get("runtime_recovery_planner")
        prefer_runtime_recovery = bool(kwargs.get("prefer_runtime_llm_recovery", False))
        edge_executor = kwargs.get("edge_executor")
        if edge_executor is None:
            edge_executor = EdgeActionExecutor()
        executor_kwargs = dict(kwargs)
        for internal_key in (
            "edge_executor",
            "runtime_recovery_planner",
            "runtime_state_collector",
            "runtime_recovery_agent",
        ):
            executor_kwargs.pop(internal_key, None)

        while current != self.goal or pending_edges or continuation_stack:
            transitions += 1
            if transitions > self.max_transitions:
                raise RuntimeError("Agent task graph exceeded max_transitions.")

            if not pending_edges and continuation_stack:
                pending_edges = continuation_stack.pop()

            edge_ref = (
                pending_edges.pop(0) if pending_edges else self._next_edge(current)
            )
            edge = self._resolve_edge(edge_ref)
            edge_id = edge.id
            try:
                result = edge_executor.execute(
                    edge=edge,
                    env=env,
                    **executor_kwargs,
                )
            except Exception as exc:
                runtime_edges = None
                if callable(runtime_recovery_planner) and not disable_recovery_branches:
                    attempts = kwargs.setdefault(
                        "_runtime_recovery_exception_attempts", {}
                    )
                    attempt_key = edge.id
                    attempts[attempt_key] = int(attempts.get(attempt_key, 0)) + 1
                    if attempts[attempt_key] <= int(
                        kwargs.get("runtime_recovery_max_exception_attempts", 1)
                    ):
                        runtime_edges = self._plan_runtime_recovery(
                            runtime_recovery_planner,
                            edge=edge,
                            monitor_index=-1,
                            result={
                                "monitor_name": "edge_exception",
                                "step_index": None,
                                "exception": exc,
                            },
                            env=env,
                            kwargs=kwargs,
                        )
                if runtime_edges is not None:
                    branch_final_target = runtime_edges[-1].target
                    continuation_edges = list(pending_edges)
                    if (
                        branch_final_target == edge.source
                        and edge.source != edge.target
                    ):
                        continuation_edges = [edge_ref, *continuation_edges]
                    elif branch_final_target != edge.target:
                        raise RuntimeError(
                            f"Runtime recovery branch for failed edge '{edge.id}' "
                            "must merge to its source or target."
                        ) from exc
                    if continuation_edges:
                        continuation_stack.append(continuation_edges)
                    self._record_recovery_history(
                        edge=edge,
                        monitor_index=-1,
                        result={
                            "monitor_name": "edge_exception",
                            "step_index": None,
                            "exception": exc,
                        },
                        recovery_edges=runtime_edges,
                        strategy="runtime",
                        kwargs=kwargs,
                    )
                    pending_edges = list(runtime_edges)
                    current = edge.source
                    continue
                raise RuntimeError(
                    f"Agent task graph edge '{edge.id}' "
                    f"({edge.source}->{edge.target}) failed: {exc}"
                ) from exc
            executed_actions.extend(result["actions"])

            monitor_index = result["monitor_index"]
            if monitor_index is not None:
                if disable_recovery_branches:
                    raise RuntimeError(
                        f"Monitor '{result['monitor_name']}' triggered on edge "
                        f"'{edge.id}', but recovery branch execution is disabled."
                    )
                branch = None
                runtime_edges = None
                if callable(runtime_recovery_planner) and prefer_runtime_recovery:
                    runtime_edges = self._plan_runtime_recovery_if_allowed(
                        runtime_recovery_planner,
                        edge=edge,
                        monitor_index=monitor_index,
                        result=result,
                        env=env,
                        kwargs=kwargs,
                    )
                if runtime_edges is None:
                    branch = self.recovery_branches.get((edge.id, monitor_index))
                if (
                    branch is None
                    and runtime_edges is None
                    and callable(runtime_recovery_planner)
                ):
                    runtime_edges = self._plan_runtime_recovery_if_allowed(
                        runtime_recovery_planner,
                        edge=edge,
                        monitor_index=monitor_index,
                        result=result,
                        env=env,
                        kwargs=kwargs,
                    )
                if branch is None:
                    if runtime_edges is None:
                        raise RuntimeError(
                            f"No recovery branch for edge '{edge.id}' monitor {monitor_index}."
                        )
                    recovery_edges = list(runtime_edges)
                    branch_final_target = recovery_edges[-1].target
                    recovery_strategy = "runtime"
                else:
                    self._record_static_recovery_attempt(
                        edge=edge,
                        monitor_index=monitor_index,
                        kwargs=kwargs,
                    )
                    recovery_edges = list(branch.recovery_edges)
                    branch_final_target = self._resolve_edge(recovery_edges[-1]).target
                    recovery_strategy = "static"
                continuation_edges = list(pending_edges)
                if branch_final_target == edge.source and edge.source != edge.target:
                    continuation_edges = [edge_ref, *continuation_edges]
                elif branch_final_target != edge.target:
                    raise RuntimeError(
                        f"Recovery branch for edge '{edge.id}' must merge to its source or target."
                    )

                if continuation_edges:
                    continuation_stack.append(continuation_edges)
                self._record_recovery_history(
                    edge=edge,
                    monitor_index=monitor_index,
                    result=result,
                    recovery_edges=recovery_edges,
                    strategy=recovery_strategy,
                    kwargs=kwargs,
                )
                pending_edges = recovery_edges
                current = edge.source
                continue

            current = edge.target

        return ExecutedActionList(executed_actions)

    def _record_recovery_history(
        self,
        *,
        edge: AgentGraphEdge,
        monitor_index: int,
        result: dict[str, Any],
        recovery_edges: list[Any],
        strategy: str,
        kwargs: dict[str, Any],
    ) -> None:
        history = kwargs.setdefault("_recovery_attempt_history", [])
        origin_edge_id = _origin_edge_id(edge, history)
        failure_class = _failure_class(monitor_index, result)
        recovery_edge_ids = [
            self._resolve_edge(edge_ref).id for edge_ref in recovery_edges
        ]
        recovery_signatures = [
            getattr(self._resolve_edge(edge_ref), "runtime_recovery_signature", None)
            for edge_ref in recovery_edges
        ]
        recovery_signature = next(
            (signature for signature in recovery_signatures if signature),
            None,
        )
        history.append(
            {
                "strategy": strategy,
                "edge_id": edge.id,
                "origin_edge_id": origin_edge_id,
                "edge_source": edge.source,
                "edge_target": edge.target,
                "edge_is_recovery": edge.is_recovery,
                "monitor_index": monitor_index,
                "monitor_name": result.get("monitor_name"),
                "step_index": result.get("step_index"),
                "failure_class": failure_class,
                "failure_reason": _failure_reason(result),
                "attempt_key": _attempt_key(origin_edge_id, failure_class),
                "recovery_edge_ids": recovery_edge_ids,
                "recovery_signature": recovery_signature,
            }
        )

    @staticmethod
    def _record_static_recovery_attempt(
        *,
        edge: AgentGraphEdge,
        monitor_index: int,
        kwargs: dict[str, Any],
    ) -> None:
        max_total_attempts = _optional_positive_int(
            kwargs.get("recovery_max_total_attempts")
        )
        if max_total_attempts is not None:
            total_attempts = int(kwargs.get("_static_recovery_total_attempts", 0)) + 1
            kwargs["_static_recovery_total_attempts"] = total_attempts
            if total_attempts > max_total_attempts:
                raise RuntimeError(
                    "Static recovery exceeded total retry limit "
                    f"({max_total_attempts}). Last edge='{edge.id}', "
                    f"monitor_index={monitor_index}."
                )

        max_monitor_attempts = _optional_positive_int(
            kwargs.get("recovery_max_monitor_attempts")
        )
        if max_monitor_attempts is None:
            return

        attempts = kwargs.setdefault("_static_recovery_monitor_attempts", {})
        attempt_key = f"{edge.id}:{monitor_index}"
        attempts[attempt_key] = int(attempts.get(attempt_key, 0)) + 1
        if attempts[attempt_key] > max_monitor_attempts:
            raise RuntimeError(
                "Static recovery exceeded monitor retry limit "
                f"({max_monitor_attempts}) for edge '{edge.id}' "
                f"monitor {monitor_index}."
            )

    def _next_edge(self, node_id: str) -> str:
        for edge_id in self.outgoing[node_id]:
            if not self.edges[edge_id].is_recovery:
                return edge_id
        raise RuntimeError(f"No nominal outgoing edge from node '{node_id}'.")

    def _resolve_edge(self, edge_ref: Any) -> AgentGraphEdge:
        if isinstance(edge_ref, AgentGraphEdge):
            return edge_ref
        if isinstance(edge_ref, dict):
            return AgentGraphEdge(**edge_ref)
        return self.edges[edge_ref]

    def _plan_runtime_recovery(
        self,
        planner,
        *,
        edge: AgentGraphEdge,
        monitor_index: int,
        result: dict[str, Any],
        env,
        kwargs: dict[str, Any],
    ) -> list[AgentGraphEdge] | None:
        total_attempts = int(kwargs.get("_runtime_recovery_total_attempts", 0)) + 1
        kwargs["_runtime_recovery_total_attempts"] = total_attempts
        max_total_attempts = int(kwargs.get("runtime_recovery_max_total_attempts", 4))
        if total_attempts > max_total_attempts:
            raise RuntimeError(
                "Runtime recovery exceeded total retry limit "
                f"({max_total_attempts}). Last edge='{edge.id}', "
                f"monitor_index={monitor_index}."
            )
        history = list(kwargs.get("_recovery_attempt_history", []))
        origin_edge_id = _origin_edge_id(edge, history)
        failure_class = _failure_class(monitor_index, result)
        recovery_edges = planner(
            graph=self,
            edge=edge,
            monitor_index=monitor_index,
            monitor_name=result.get("monitor_name"),
            step_index=result.get("step_index"),
            env=env,
            runtime_kwargs=kwargs,
            failure_context=_failure_context(
                edge,
                monitor_index,
                result,
                origin_edge_id=origin_edge_id,
                failure_class=failure_class,
            ),
            runtime_state=_collect_runtime_state(kwargs, env),
            recovery_history=history,
        )
        if not recovery_edges:
            return None
        return [self._resolve_edge(edge_ref) for edge_ref in recovery_edges]

    def _plan_runtime_recovery_if_allowed(
        self,
        planner,
        *,
        edge: AgentGraphEdge,
        monitor_index: int,
        result: dict[str, Any],
        env,
        kwargs: dict[str, Any],
    ) -> list[AgentGraphEdge] | None:
        attempts = kwargs.setdefault("_runtime_recovery_monitor_attempts", {})
        history = list(kwargs.get("_recovery_attempt_history", []))
        attempt_key = _attempt_key(
            _origin_edge_id(edge, history),
            _failure_class(monitor_index, result),
        )
        attempts[attempt_key] = int(attempts.get(attempt_key, 0)) + 1
        if attempts[attempt_key] > int(
            kwargs.get("runtime_recovery_max_monitor_attempts", 2)
        ):
            raise RuntimeError(
                f"Runtime recovery exceeded monitor retry limit for " f"{attempt_key}."
            )
        return self._plan_runtime_recovery(
            planner,
            edge=edge,
            monitor_index=monitor_index,
            result=result,
            env=env,
            kwargs=kwargs,
        )


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def _failure_context(
    edge: AgentGraphEdge,
    monitor_index: int,
    result: dict[str, Any],
    *,
    origin_edge_id: str | None = None,
    failure_class: str | None = None,
) -> dict[str, Any]:
    return {
        "edge_id": edge.id,
        "origin_edge_id": origin_edge_id or edge.id,
        "edge_source": edge.source,
        "edge_target": edge.target,
        "edge_is_recovery": edge.is_recovery,
        "monitor_index": monitor_index,
        "monitor_name": result.get("monitor_name"),
        "step_index": result.get("step_index"),
        "failure_class": failure_class or _failure_class(monitor_index, result),
        "failure_reason": _failure_reason(result),
    }


def _origin_edge_id(edge: AgentGraphEdge, history: list[dict[str, Any]]) -> str:
    if not edge.is_recovery:
        return edge.id
    for attempt in reversed(history):
        if edge.id in attempt.get("recovery_edge_ids", []):
            return str(attempt.get("origin_edge_id") or attempt.get("edge_id"))
    return edge.id


def _failure_class(monitor_index: int, result: dict[str, Any]) -> str:
    exception = result.get("exception")
    if exception is not None:
        return f"exception:{type(exception).__name__}"
    monitor_name = result.get("monitor_name") or "unknown_monitor"
    return f"monitor:{monitor_name}:{monitor_index}"


def _attempt_key(origin_edge_id: str, failure_class: str) -> str:
    return f"{origin_edge_id}:{failure_class}"


def _failure_reason(result: dict[str, Any]) -> str:
    exception = result.get("exception")
    if exception is not None:
        return f"{type(exception).__name__}: {exception}"
    monitor_name = result.get("monitor_name")
    if monitor_name:
        return f"Monitor '{monitor_name}' triggered."
    return "Unknown runtime failure."


def _collect_runtime_state(kwargs: dict[str, Any], env) -> dict[str, Any] | None:
    collector = kwargs.get("runtime_state_collector")
    if not callable(collector):
        return None
    return collector(env)
