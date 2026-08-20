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

"""Grounded-plan compilation and compact execution reporting."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, TypeAlias

import numpy as np
import torch

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import (
    seed_graph_hash,
    validate_seed_graph,
)
from embodichain.gen_sim.action_engine.planning.linker import (
    validate_persisted_contracts,
)
from embodichain.gen_sim.action_engine.runtime import (
    ExecutionProgram,
    ExecutionReport,
    ExecutionResult,
    ProgramExecutor,
    build_execution_provenance,
    load_execution_program,
    validate_execution_report,
    write_execution_report,
)
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph
from embodichain.gen_sim.action_engine.unbound import (
    UnboundActionPlan,
    build_unbound_action_plan,
)

__all__ = ["ActionAgent", "ActionGraph"]

ExecutorFactory = Callable[..., ProgramExecutor]
ActionGraph: TypeAlias = dict[str, Any]


class ActionAgent:
    """Compile, preflight, execute, and report one grounded task plan."""

    def __init__(
        self,
        *,
        registry: AtomicCapabilityRegistry | None = None,
        executor_factory: ExecutorFactory = ProgramExecutor,
    ) -> None:
        self.registry = registry or build_atomic_capability_registry()
        self.executor_factory = executor_factory

    def plan(self, grounded_plan: Mapping[str, Any]) -> ActionGraph:
        """Compile a validated GroundedTaskPlan to the public SeedGraph v3."""
        plan = _validate_grounded_plan(grounded_plan)
        task_spec = _mapping(plan.get("task_spec"), "GroundedTaskPlan.task_spec")
        bindings = _role_binding_map(plan.get("role_bindings"))
        graph = instantiate_seed_graph(
            task_spec,
            bindings,
            registry=self.registry,
        )
        known_uids = _known_uids(
            plan.get("scene_manifest"),
            bindings=bindings,
        )
        known_uids.add("table")
        graph = validate_seed_graph(
            graph,
            known_objects=known_uids or None,
            known_actions=self.registry.names(),
            executable_actions=self.registry.executable_names(),
            require_executable=False,
        )
        validate_persisted_contracts(graph, self.registry)
        return graph

    def draft(self, candidate: Mapping[str, Any]) -> UnboundActionPlan:
        """Create an Action-owned draft before final scene UID binding.

        Args:
            candidate: One validated Task Engine candidate.

        Returns:
            A scene-independent action plan whose selectors contain no UIDs.
        """
        return build_unbound_action_plan(candidate)

    def preflight(
        self,
        action_graph: Mapping[str, Any] | str | Path,
        *,
        scene_manifest: Mapping[str, Any] | None = None,
        known_uids: Collection[str] | None = None,
    ) -> ExecutionProgram:
        """Reject invalid and planning-only graphs before simulator motion."""
        known = set(str(uid) for uid in (known_uids or ()) if str(uid))
        known.update(_known_uids(scene_manifest))
        if isinstance(action_graph, Mapping):
            metadata = action_graph.get("metadata", {})
            if isinstance(metadata, Mapping):
                bindings = metadata.get("role_bindings", {})
                if isinstance(bindings, Mapping):
                    known.update(str(uid) for uid in bindings.values() if str(uid))
        if known:
            known.add("table")
        return load_execution_program(
            action_graph,
            known_objects=known or None,
            registry=self.registry,
            require_executable=True,
        )

    def execute(
        self,
        action_graph: Mapping[str, Any] | str | Path,
        env: Any,
        *,
        grounded_plan: Mapping[str, Any] | None = None,
        scene_manifest: Mapping[str, Any] | None = None,
        known_uids: Collection[str] | None = None,
        run_id: str | None = None,
        episode_index: int = 0,
        episode_seed: int | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
        executor_kwargs: Mapping[str, Any] | None = None,
    ) -> ExecutionReport:
        """Preflight and execute a graph, converting all outcomes to a report."""
        task_id = _task_id(grounded_plan, action_graph)
        plan_hash = _plan_hash(grounded_plan)
        graph_hash = _action_graph_hash(action_graph)
        effective_run_id = run_id or _new_run_id()
        provenance = build_execution_provenance(
            episode_seed=episode_seed,
            runtime_arguments=runtime_arguments,
        )
        effective_manifest = scene_manifest
        if effective_manifest is None and grounded_plan is not None:
            value = grounded_plan.get("scene_manifest")
            if isinstance(value, Mapping):
                effective_manifest = value

        try:
            program = self.preflight(
                action_graph,
                scene_manifest=effective_manifest,
                known_uids=known_uids,
            )
        except (TypeError, ValueError, OSError) as exc:
            return self._empty_report(
                env,
                task_id=task_id,
                plan_hash=plan_hash,
                graph_hash=graph_hash,
                status="rejected",
                run_id=effective_run_id,
                episode_index=episode_index,
                provenance=provenance,
                error=_error_message(exc),
            )

        kwargs = dict(executor_kwargs or {})
        kwargs.setdefault("capability_registry", self.registry)
        try:
            executor = self.executor_factory(program, env, **kwargs)
            result = executor.run(
                run_id=effective_run_id,
                episode_index=episode_index,
            )
        except Exception as exc:
            return self._empty_report(
                env,
                task_id=task_id,
                plan_hash=plan_hash,
                graph_hash=graph_hash,
                status="aborted",
                run_id=effective_run_id,
                episode_index=episode_index,
                provenance=provenance,
                error=_error_message(exc),
            )
        if not isinstance(result, ExecutionResult):
            return self._empty_report(
                env,
                task_id=task_id,
                plan_hash=plan_hash,
                graph_hash=graph_hash,
                status="aborted",
                run_id=effective_run_id,
                episode_index=episode_index,
                provenance=provenance,
                error="TypeError: ProgramExecutor.run must return ExecutionResult.",
            )
        try:
            return self.report_execution_result(
                result,
                action_graph=action_graph,
                grounded_plan=grounded_plan,
                run_id=effective_run_id,
                episode_index=episode_index,
                episode_seed=episode_seed,
                runtime_arguments=runtime_arguments,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return self._empty_report(
                env,
                task_id=task_id,
                plan_hash=plan_hash,
                graph_hash=graph_hash,
                status="aborted",
                run_id=effective_run_id,
                episode_index=episode_index,
                provenance=provenance,
                error=_error_message(exc),
            )

    def run(
        self,
        grounded_plan: Mapping[str, Any],
        env: Any,
        *,
        run_id: str | None = None,
        episode_index: int = 0,
        episode_seed: int | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
        executor_kwargs: Mapping[str, Any] | None = None,
    ) -> ExecutionReport:
        """Compile and execute one GroundedTaskPlan through the full pipeline."""
        effective_run_id = run_id or _new_run_id()
        try:
            graph = self.plan(grounded_plan)
        except (TypeError, ValueError, OSError) as exc:
            return self._empty_report(
                env,
                task_id=_task_id(grounded_plan, {}),
                plan_hash=_plan_hash(grounded_plan),
                graph_hash=_document_hash({}),
                status="rejected",
                run_id=effective_run_id,
                episode_index=episode_index,
                provenance=build_execution_provenance(
                    episode_seed=episode_seed,
                    runtime_arguments=runtime_arguments,
                ),
                error=_error_message(exc),
            )
        return self.execute(
            graph,
            env,
            grounded_plan=grounded_plan,
            run_id=effective_run_id,
            episode_index=episode_index,
            episode_seed=episode_seed,
            runtime_arguments=runtime_arguments,
            executor_kwargs=executor_kwargs,
        )

    def report_execution_result(
        self,
        result: ExecutionResult,
        *,
        action_graph: Mapping[str, Any] | str | Path,
        grounded_plan: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        episode_index: int = 0,
        episode_seed: int | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> ExecutionReport:
        """Convert a result already executed by the legacy runner to a report."""
        if not isinstance(result, ExecutionResult):
            raise TypeError("result must be an ExecutionResult.")
        return self._result_report(
            result,
            task_id=_task_id(grounded_plan, action_graph),
            plan_hash=_plan_hash(grounded_plan),
            graph_hash=_action_graph_hash(action_graph),
            run_id=run_id or _new_run_id(),
            episode_index=episode_index,
            provenance=build_execution_provenance(
                episode_seed=episode_seed,
                runtime_arguments=runtime_arguments,
            ),
        )

    def rejection_report(
        self,
        action_graph: Mapping[str, Any] | str | Path,
        error: BaseException | str,
        *,
        grounded_plan: Mapping[str, Any] | None = None,
        environment_count: int = 1,
        run_id: str | None = None,
        episode_index: int = 0,
        episode_seed: int | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> ExecutionReport:
        """Build a zero-action report for a preflight rejection."""
        message = error if isinstance(error, str) else _error_message(error)
        return self._empty_report(
            SimpleNamespace(num_envs=max(1, int(environment_count))),
            task_id=_task_id(grounded_plan, action_graph),
            plan_hash=_plan_hash(grounded_plan),
            graph_hash=_action_graph_hash(action_graph),
            status="rejected",
            run_id=run_id or _new_run_id(),
            episode_index=episode_index,
            provenance=build_execution_provenance(
                episode_seed=episode_seed,
                runtime_arguments=runtime_arguments,
            ),
            error=str(message),
        )

    def abortion_report(
        self,
        action_graph: Mapping[str, Any] | str | Path,
        error: BaseException | str,
        *,
        grounded_plan: Mapping[str, Any] | None = None,
        environment_count: int = 1,
        run_id: str | None = None,
        episode_index: int = 0,
        episode_seed: int | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> ExecutionReport:
        """Build a zero-action report for an unexpected runtime exception."""
        message = error if isinstance(error, str) else _error_message(error)
        return self._empty_report(
            SimpleNamespace(num_envs=max(1, int(environment_count))),
            task_id=_task_id(grounded_plan, action_graph),
            plan_hash=_plan_hash(grounded_plan),
            graph_hash=_action_graph_hash(action_graph),
            status="aborted",
            run_id=run_id or _new_run_id(),
            episode_index=episode_index,
            provenance=build_execution_provenance(
                episode_seed=episode_seed,
                runtime_arguments=runtime_arguments,
            ),
            error=str(message),
        )

    def _result_report(
        self,
        result: ExecutionResult,
        *,
        task_id: str,
        plan_hash: str,
        graph_hash: str,
        run_id: str,
        episode_index: int,
        provenance: Mapping[str, Any],
    ) -> ExecutionReport:
        success = _bool_vector(result.success)
        semantics = {
            str(step_id): _bool_vector(mask)
            for step_id, mask in result.semantic_success.items()
        }
        failures = tuple(_json_safe(item) for item in result.failure_events)
        revisions = tuple(_json_safe(item) for item in result.runtime_revisions)
        action_count = len(result.actions)
        environments = tuple(
            {
                "env_id": str(env_id),
                "success": value,
                "semantic_success": {
                    step_id: values[env_id]
                    for step_id, values in semantics.items()
                    if env_id < len(values)
                },
                "action_count": action_count,
                "retry_count": _retry_count_for_env(result, env_id),
                "recovery_count": _revision_count_for_env(
                    revisions, env_id, kind="insert_recovery"
                ),
                "revision_count": _revision_count_for_env(revisions, env_id),
                "failures": _events_for_env(failures, env_id),
            }
            for env_id, value in enumerate(success)
        )
        report = ExecutionReport(
            task_id=task_id,
            plan_hash=plan_hash,
            action_graph_hash=graph_hash,
            status="succeeded" if all(success) else "failed",
            run_id=run_id,
            episode_id=str(episode_index),
            provenance=deepcopy(dict(provenance)),
            environments=environments,
            action_count=action_count,
            retry_count=int(result.retry_count),
            recovery_count=int(result.recovery_count),
            revision_count=int(result.revision_count),
            failure_events=failures,
            graph_revisions=revisions,
            record_dir=result.record_dir,
            error=None,
        )
        validated = _validated_report(report)
        _publish_execution_report(validated)
        return validated

    def _empty_report(
        self,
        env: Any,
        *,
        task_id: str,
        plan_hash: str,
        graph_hash: str,
        status: str,
        run_id: str,
        episode_index: int,
        provenance: Mapping[str, Any],
        error: str,
    ) -> ExecutionReport:
        count = _environment_count(env)
        report = ExecutionReport(
            task_id=task_id,
            plan_hash=plan_hash,
            action_graph_hash=graph_hash,
            status=status,
            run_id=run_id,
            episode_id=str(episode_index),
            provenance=deepcopy(dict(provenance)),
            environments=tuple(
                {
                    "env_id": str(env_id),
                    "success": False,
                    "semantic_success": {},
                    "action_count": 0,
                    "retry_count": 0,
                    "recovery_count": 0,
                    "revision_count": 0,
                    "failures": [],
                }
                for env_id in range(count)
            ),
            error=error,
        )
        return _validated_report(report)


def _validate_grounded_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("GroundedTaskPlan must be a mapping.")
    # GroundedTaskPlan is a cross-engine protocol owned by Task Engine.
    # Import lazily so Action Engine remains importable without initializing
    # the coordinator or Scene Adapter.
    try:
        from embodichain.gen_sim.task_engine.orchestration.contracts import (
            validate_grounded_task_plan,
        )
    except (ImportError, AttributeError):
        return deepcopy(dict(value))
    return validate_grounded_task_plan(value)


def _validated_report(report: ExecutionReport) -> ExecutionReport:
    payload = report.as_mapping()
    validate_execution_report(payload)
    return report


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return deepcopy(dict(value))


def _role_binding_map(value: Any) -> dict[str, str]:
    source = _mapping(value, "GroundedTaskPlan.role_bindings")
    nested = source.get("role_bindings")
    if isinstance(nested, Mapping):
        source = dict(nested)
    result = {str(role): str(uid) for role, uid in source.items()}
    if not result or any(not role or not uid for role, uid in result.items()):
        raise ValueError("GroundedTaskPlan role bindings must not be empty.")
    return result


def _known_uids(
    manifest: Any,
    *,
    bindings: Mapping[str, str] | None = None,
) -> set[str]:
    result = {str(uid) for uid in (bindings or {}).values() if str(uid)}
    if not isinstance(manifest, Mapping):
        return result
    objects = manifest.get("objects", ())
    if isinstance(objects, Sequence) and not isinstance(
        objects, (str, bytes, bytearray)
    ):
        for item in objects:
            if isinstance(item, Mapping):
                uid = item.get("uid", item.get("runtime_uid"))
                if isinstance(uid, str) and uid:
                    result.add(uid)
    return result


def _task_id(
    plan: Mapping[str, Any] | None,
    graph: Mapping[str, Any] | str | Path,
) -> str:
    if isinstance(plan, Mapping):
        value = plan.get("task_id")
        if isinstance(value, str) and value:
            return value
    if isinstance(graph, Mapping):
        value = graph.get("task_id")
        if isinstance(value, str) and value:
            return value
    return "unknown_task"


def _plan_hash(plan: Mapping[str, Any] | None) -> str:
    if isinstance(plan, Mapping):
        hashes = plan.get("hashes", {})
        if isinstance(hashes, Mapping):
            value = hashes.get("plan")
            if isinstance(value, str) and value:
                return value
        return _safe_document_hash(plan)
    return _document_hash({})


def _action_graph_hash(value: Mapping[str, Any] | str | Path) -> str:
    if isinstance(value, Mapping):
        try:
            return seed_graph_hash(value)
        except (TypeError, ValueError):
            return _safe_document_hash(value)
    path = Path(value).expanduser()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return (
        _action_graph_hash(loaded)
        if isinstance(loaded, Mapping)
        else _safe_document_hash(loaded)
    )


def _safe_document_hash(value: Any) -> str:
    try:
        return _document_hash(value)
    except (TypeError, ValueError, OverflowError):
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _document_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bool_vector(value: Any) -> list[bool]:
    if isinstance(value, torch.Tensor):
        return [bool(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [bool(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [bool(item) for item in value]
    return [bool(value)]


def _events_for_env(
    events: Sequence[Mapping[str, Any]], env_id: int
) -> list[dict[str, Any]]:
    result = []
    for event in events:
        env_ids = event.get("env_ids")
        if isinstance(env_ids, Sequence) and not isinstance(
            env_ids, (str, bytes, bytearray)
        ):
            if env_id not in env_ids:
                continue
            item = deepcopy(dict(event))
            item["env_ids"] = [env_id]
            result.append(item)
        else:
            result.append(deepcopy(dict(event)))
    return result


def _revision_count_for_env(
    revisions: Sequence[Mapping[str, Any]],
    env_id: int,
    *,
    kind: str | None = None,
) -> int:
    count = 0
    for revision in revisions:
        if kind is not None and revision.get("kind") != kind:
            continue
        active = revision.get("active_env_ids")
        if (
            isinstance(active, Sequence)
            and not isinstance(active, (str, bytes, bytearray))
            and env_id not in active
        ):
            continue
        count += 1
    return count


def _retry_count_for_env(result: ExecutionResult, env_id: int) -> int:
    counts = result.retry_counts
    if env_id < len(counts):
        return int(counts[env_id])
    return int(result.retry_count)


def _environment_count(env: Any) -> int:
    value = getattr(env, "num_envs", 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _error_message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _publish_execution_report(report: ExecutionReport) -> None:
    """Atomically publish the compact report beside runtime episode records."""
    if not report.record_dir:
        return
    write_execution_report(report.record_dir, report)
