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

"""Execute offline and online SeedGraphs from strictly identical resets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_engine.domain import (
    seed_graph_hash,
    validate_seed_graph,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.protocol import (
    COMPARISON_FILENAME,
    EXECUTION_PROGRAM_FILENAME,
)
from embodichain.gen_sim.action_engine.runtime.motion_policy import (
    resolve_motion_policy,
)

__all__ = ["ABExecutionResult", "run_strict_ab", "state_digest"]

EnvFactory = Callable[..., Any]
ExecutorFactory = Callable[[Mapping[str, Any], Any], Any]
SnapshotReader = Callable[[Any], Mapping[str, Any]]
SuccessEvaluator = Callable[..., Any]
BranchFinalizer = Callable[..., list[str]]

_FULL_SNAPSHOT_KEYS = frozenset(
    {"robot_qpos", "object_poses", "articulation_state", "camera_calibration"}
)
_PRIVATE_OR_LIVE_KEYS = frozenset(
    {
        "absolute_position",
        "coordinates",
        "grasp_pose",
        "joint_positions",
        "live_pose",
        "live_transform",
        "oracle",
        "pose",
        "positions",
        "qpos",
        "target_pose",
        "trajectory",
        "waypoints",
        "xpos",
    }
)


@dataclass(frozen=True)
class ABExecutionResult:
    """Paths and summaries from one strict A/B run."""

    comparison_path: Path
    offline_dir: Path
    online_dir: Path
    initial_state_digest: str
    comparison: dict[str, Any]


def state_digest(snapshot: Mapping[str, Any]) -> str:
    """Hash nested tensors/arrays/mappings without lossy JSON conversion."""
    digest = hashlib.sha256()
    _update_digest(digest, snapshot)
    return digest.hexdigest()


def run_strict_ab(
    task_spec: Mapping[str, Any],
    offline_graph: Mapping[str, Any],
    online_graph: Mapping[str, Any],
    *,
    env_factory: EnvFactory | None = None,
    executor_factory: ExecutorFactory,
    snapshot_reader: SnapshotReader,
    output_dir: str | Path,
    seed: int,
    shared_config: Mapping[str, Any] | None = None,
    planning_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    success_evaluator: SuccessEvaluator | None = None,
    known_objects: set[str] | None = None,
    expected_initial_state_digest: str | None = None,
    branch_finalizer: BranchFinalizer | None = None,
    episode_index: int = 0,
    strict_state_digest: bool | None = None,
    prepared_environments: Mapping[str, Any] | None = None,
    prepared_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    require_branch_videos: bool = False,
) -> ABExecutionResult:
    """Run both planners in isolated environments after exact state checks.

    Callers that need visual observations before planning may supply two already
    reset environments and their snapshots.  The environments remain owned by
    this function once supplied and are closed on every exit path.

    Set ``require_branch_videos`` for production A/B runs.  In that mode each
    finalizer must publish one non-empty ``video.mp4`` in its branch directory.
    """
    supplied_environments = (
        tuple(prepared_environments.values())
        if prepared_environments is not None
        else ()
    )
    try:
        task, config, offline, online = _validate_ab_inputs(
            task_spec,
            offline_graph,
            online_graph,
            shared_config=shared_config,
            success_evaluator=success_evaluator,
            known_objects=known_objects,
        )
    except BaseException:
        # Prepared environments are already live before graph validation.  The
        # caller transfers ownership at function entry, including invalid-input
        # paths that return before the normal environment scope below.
        _close_environments(supplied_environments)
        raise

    metrics = dict(planning_metrics or {})
    environments: dict[str, Any] = {}
    try:
        routes = ("offline", "online")
        if prepared_environments is not None:
            # Keep every supplied object in ``environments`` until after shape
            # validation so the finally block closes extras on this error path.
            environments = dict(prepared_environments)
            if set(environments) != set(routes):
                raise ValueError(
                    "prepared_environments must contain exactly offline and online."
                )
            environments = {route: prepared_environments[route] for route in routes}
        else:
            if not callable(env_factory):
                raise TypeError(
                    "env_factory is required when prepared_environments is not supplied."
                )
            for route in routes:
                environments[route] = env_factory(
                    route=route,
                    seed=int(seed),
                    config=config,
                )
        if id(environments["offline"]) == id(environments["online"]):
            raise RuntimeError(
                "Strict A/B requires two isolated environment instances; "
                "env_factory returned the same object twice."
            )
        for route, env in environments.items():
            marker = getattr(env, "action_engine_ab_route", None)
            if marker is not None and str(marker) != route:
                raise RuntimeError(
                    f"A/B environment route marker {marker!r} does not match {route!r}."
                )
        snapshots: dict[str, Mapping[str, Any]] = {}
        if prepared_snapshots is not None and prepared_environments is None:
            raise ValueError(
                "prepared_snapshots requires prepared_environments so the state "
                "being compared is unambiguous."
            )
        if prepared_snapshots is not None:
            if set(prepared_snapshots) != set(routes):
                raise ValueError(
                    "prepared_snapshots must contain exactly offline and online."
                )
            snapshots = {route: prepared_snapshots[route] for route in routes}
        digests = {}
        for route, env in environments.items():
            if prepared_snapshots is None:
                env.reset(seed=int(seed))
                snapshots[route] = snapshot_reader(env)
            _validate_snapshot(snapshots[route], route=route, require_full=False)
        if strict_state_digest is None:
            strict_state_digest = bool(config.get("strict_state_digest", False))
            # Automatically enforce the expanded contract whenever a caller
            # supplies any of the new state components, while retaining the
            # two-field v1 test helper compatibility.
            strict_state_digest = strict_state_digest or any(
                set(snapshot) & (_FULL_SNAPSHOT_KEYS - {"robot_qpos", "object_poses"})
                for snapshot in snapshots.values()
            )
        if strict_state_digest:
            for route, snapshot in snapshots.items():
                _validate_snapshot(snapshot, route=route, require_full=True)
        for route, snapshot in snapshots.items():
            digests[route] = state_digest(snapshots[route])
        if digests["offline"] != digests["online"]:
            raise RuntimeError(
                "Strict A/B initial state mismatch: "
                f"offline={digests['offline']}, online={digests['online']}."
            )
        if (
            expected_initial_state_digest is not None
            and digests["offline"] != expected_initial_state_digest
        ):
            raise RuntimeError(
                "Strict A/B execution state does not match the online-planning "
                f"snapshot: planning={expected_initial_state_digest}, "
                f"execution={digests['offline']}."
            )

        root = Path(output_dir).expanduser().resolve()
        branch_dirs = {route: root / route for route in environments}
        for branch_dir in branch_dirs.values():
            branch_dir.mkdir(parents=True, exist_ok=True)
        # Construct and preflight both executors before invoking either run.
        # A route-specific executor may perform capability/robot checks that
        # cannot be expressed in the serializable SeedGraph validator.
        executors: dict[str, Any] = {}
        for route, graph in (("offline", offline), ("online", online)):
            _write_json(branch_dirs[route] / EXECUTION_PROGRAM_FILENAME, graph)
            executors[route] = executor_factory(graph, environments[route])
        preflight_errors: dict[str, Exception] = {}
        for route, executor in executors.items():
            preflight = getattr(executor, "preflight", None)
            if not callable(preflight):
                preflight = getattr(executor, "validate", None)
            if not callable(preflight):
                continue
            try:
                outcome = _call_preflight(
                    preflight,
                    route=route,
                    graph=(offline if route == "offline" else online),
                    env=environments[route],
                )
                if outcome is not None:
                    try:
                        preflight_ok = bool(outcome)
                    except (TypeError, ValueError, RuntimeError) as exc:
                        raise RuntimeError(
                            f"{route} executor preflight returned a non-scalar result."
                        ) from exc
                    if not preflight_ok:
                        raise RuntimeError(
                            f"{route} executor preflight returned false."
                        )
            except Exception as exc:
                preflight_errors[route] = exc
        if preflight_errors:
            detail = "; ".join(
                f"{route}: {type(error).__name__}: {error}"
                for route, error in sorted(preflight_errors.items())
            )
            raise RuntimeError(
                "Strict A/B preflight failed; no branch was allowed to move. " + detail
            )
        results = {}
        finalization_errors: dict[str, Exception] = {}
        for route, graph in (("offline", offline), ("online", online)):
            result = None
            started = perf_counter()
            try:
                executor = executors[route]
                result = executor.run(
                    run_id=f"ab-{seed}-{route}",
                    episode_index=episode_index,
                )
                elapsed = perf_counter() - started
                success_override = (
                    success_evaluator(
                        task_spec=task,
                        graph=graph,
                        env=environments[route],
                        result=result,
                        route=route,
                    )
                    if success_evaluator is not None
                    else None
                )
                results[route] = _result_summary(
                    result,
                    elapsed,
                    graph,
                    metrics.get(route, {}),
                    success_override=success_override,
                )
            except Exception as exc:
                elapsed = perf_counter() - started
                results[route] = _error_result_summary(
                    exc,
                    elapsed,
                    graph,
                    metrics.get(route, {}),
                )
            try:
                raw_video_paths = (
                    branch_finalizer(
                        route=route,
                        env=environments[route],
                        result=result,
                        branch_dir=branch_dirs[route],
                        episode_index=episode_index,
                    )
                    if branch_finalizer is not None
                    else list(getattr(result, "video_paths", ()))
                )
                video_paths = [str(path) for path in raw_video_paths]
                if require_branch_videos:
                    _validate_branch_video_paths(
                        video_paths,
                        route=route,
                        branch_dir=branch_dirs[route],
                    )
            except Exception as exc:
                video_paths = []
                results[route]["video_error"] = f"{type(exc).__name__}: {exc}"
                finalization_errors[route] = exc
            results[route]["video_paths"] = video_paths
            results[route]["initial_state_digest"] = digests[route]
            results[route]["seed_graph_hash"] = seed_graph_hash(graph)
            _write_json(
                branch_dirs[route] / "runtime_revisions.json",
                {
                    "schema_version": "action_engine_runtime_revisions_v1",
                    "task_id": task["task_id"],
                    "route": route,
                    "revisions": list(getattr(result, "runtime_revisions", ())),
                },
            )
            _write_json(branch_dirs[route] / "result.json", results[route])

        comparison = {
            "schema_version": "action_engine_ab_comparison_v1",
            "task_id": task["task_id"],
            "seed": int(seed),
            "shared_config": config,
            "initial_state_digest": digests["offline"],
            "initial_state_digests": dict(digests),
            "strict_state_digest": bool(strict_state_digest),
            "graph_hashes": {
                "offline": seed_graph_hash(offline),
                "online": seed_graph_hash(online),
            },
            "branches": {
                "offline": {
                    **results["offline"],
                },
                "online": {
                    **results["online"],
                },
            },
            "graph_difference": _graph_difference(offline, online),
            "video_finalization_errors": {
                route: f"{type(error).__name__}: {error}"
                for route, error in sorted(finalization_errors.items())
            },
        }
        comparison_path = root / COMPARISON_FILENAME
        _write_json(comparison_path, comparison)
        if finalization_errors:
            detail = "; ".join(
                f"{route}: {type(error).__name__}: {error}"
                for route, error in sorted(finalization_errors.items())
            )
            first_error = next(iter(finalization_errors.values()))
            raise RuntimeError(
                "Strict A/B branch video finalization failed; comparison report "
                "was written with artifact errors. " + detail
            ) from first_error
        return ABExecutionResult(
            comparison_path,
            branch_dirs["offline"],
            branch_dirs["online"],
            digests["offline"],
            comparison,
        )
    finally:
        _close_environments(environments.values())


def _validate_ab_inputs(
    task_spec: Mapping[str, Any],
    offline_graph: Mapping[str, Any],
    online_graph: Mapping[str, Any],
    *,
    shared_config: Mapping[str, Any] | None,
    success_evaluator: SuccessEvaluator | None,
    known_objects: set[str] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate every serializable input before either branch may move."""
    task = validate_task_spec(task_spec)
    if task["level"] == "L4" and not callable(success_evaluator):
        raise ValueError(
            "Strict L4 A/B requires a path-independent private-oracle "
            "success_evaluator."
        )
    config = dict(shared_config or {})
    capabilities = build_atomic_capability_registry()
    offline = validate_seed_graph(
        offline_graph,
        known_objects=known_objects,
        known_actions=capabilities.names(),
        executable_actions=capabilities.executable_names(),
        require_executable=True,
    )
    online = validate_seed_graph(
        online_graph,
        known_objects=known_objects,
        known_actions=capabilities.names(),
        executable_actions=capabilities.executable_names(),
        require_executable=True,
    )
    robot_profile = str(config.get("robot_profile", "dual_ur10"))
    from embodichain.gen_sim.action_engine.planning.linker import (
        validate_persisted_contracts,
    )

    for graph in (offline, online):
        if graph["capability_catalog_hash"] != capabilities.catalog_hash():
            raise ValueError("A/B SeedGraph capability catalog does not match runtime.")
        validate_persisted_contracts(graph, capabilities)
        for node in graph["nodes"]:
            capabilities.validate_binding(node)
            resolve_motion_policy(
                robot_profile,
                node["atomic_action"],
                node["motion_policy"],
            )
        _reject_private_or_live_fields(graph, "A/B SeedGraph")
    if offline["task_id"] != task["task_id"] or online["task_id"] != task["task_id"]:
        raise ValueError("A/B graphs and TaskSpec must have the same task_id.")
    for route, graph in (("offline", offline), ("online", online)):
        if (
            graph["level"] != task["level"]
            or graph["reasoning_type"] != task["reasoning_type"]
        ):
            raise ValueError(
                f"A/B {route} SeedGraph level/reasoning does not match TaskSpec."
            )
        _validate_task_group_coverage(task, graph, route=route)
    if offline["planner_route"] != "offline" or online["planner_route"] != "online":
        raise ValueError(
            "Strict A/B requires explicit offline and online graph routes."
        )
    return task, config, offline, online


def _validate_branch_video_paths(
    video_paths: list[str], *, route: str, branch_dir: Path
) -> None:
    """Require the normalized video artifact used by strict production A/B."""
    expected = (branch_dir / "video.mp4").resolve()
    if len(video_paths) != 1:
        raise RuntimeError(
            f"Strict A/B {route} branch must publish exactly one video.mp4."
        )
    published = Path(video_paths[0]).expanduser().resolve()
    if published != expected:
        raise RuntimeError(
            f"Strict A/B {route} video must be published as {expected.as_posix()}."
        )
    if not expected.is_file() or expected.stat().st_size <= 0:
        raise RuntimeError(f"Strict A/B {route} video.mp4 is missing or empty.")


def _close_environments(environments: Any) -> None:
    """Best-effort close every distinct supplied environment exactly once."""
    seen: set[int] = set()
    for env in environments:
        if id(env) in seen:
            continue
        seen.add(id(env))
        close = getattr(env, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            # Preserve the validation/execution failure that triggered cleanup,
            # but continue closing the other independent branch.
            continue


def _result_summary(
    result: Any,
    elapsed: float,
    graph: Mapping[str, Any],
    planning_metrics: Mapping[str, Any],
    *,
    success_override: Any | None,
) -> dict[str, Any]:
    success = torch.as_tensor(
        (
            getattr(result, "success", False)
            if success_override is None
            else success_override
        ),
        dtype=torch.bool,
    )
    actions = list(getattr(result, "actions", ()))
    retries = int(getattr(result, "retry_count", 0))
    recoveries = int(getattr(result, "recovery_count", 0))
    revisions = int(getattr(result, "revision_count", 0))
    metadata = graph.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "route": str(graph.get("planner_route", "")),
        "seed_graph_hash": seed_graph_hash(graph),
        "planning_seconds": float(
            planning_metrics.get(
                "planning_seconds",
                metadata.get("planning_latency_seconds", 0.0),
            )
        ),
        "execution_seconds": float(elapsed),
        "vlm_call_count": int(
            planning_metrics.get(
                "vlm_call_count",
                metadata.get("vlm_call_count", 0),
            )
        ),
        "success": success.tolist(),
        "success_source": (
            "runtime_postconditions" if success_override is None else "private_oracle"
        ),
        "success_rate": float(success.float().mean()) if success.numel() else 0.0,
        "action_command_count": len(actions),
        "path_length": _path_length(actions),
        "retry_count": retries,
        "recovery_count": recoveries,
        "revision_count": revisions,
        "failure_events": list(getattr(result, "failure_events", ())),
        "ik_failure_count": sum(
            item.get("failure_type") in {"plan_failed", "search_exhausted"}
            for item in getattr(result, "failure_events", ())
        ),
        "record_dir": getattr(result, "record_dir", None),
        "video_paths": list(getattr(result, "video_paths", ())),
    }


def _error_result_summary(
    error: Exception,
    elapsed: float,
    graph: Mapping[str, Any],
    planning_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = graph.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "route": str(graph.get("planner_route", "")),
        "seed_graph_hash": seed_graph_hash(graph),
        "planning_seconds": float(
            planning_metrics.get(
                "planning_seconds",
                metadata.get("planning_latency_seconds", 0.0),
            )
        ),
        "execution_seconds": float(elapsed),
        "vlm_call_count": int(
            planning_metrics.get("vlm_call_count", metadata.get("vlm_call_count", 0))
        ),
        "success": [False],
        "success_source": "runtime_exception",
        "success_rate": 0.0,
        "action_command_count": 0,
        "path_length": 0.0,
        "retry_count": 0,
        "recovery_count": 0,
        "revision_count": 0,
        "failure_events": [],
        "ik_failure_count": 0,
        "record_dir": None,
        "video_paths": [],
        "error": f"{type(error).__name__}: {error}",
    }


def _path_length(actions: list[Any]) -> float:
    if len(actions) < 2:
        return 0.0
    tensors = [torch.as_tensor(action, dtype=torch.float32) for action in actions]
    return float(
        sum(
            torch.linalg.vector_norm(current - previous, dim=-1).sum()
            for previous, current in zip(tensors, tensors[1:])
        )
    )


def _graph_difference(
    offline: Mapping[str, Any], online: Mapping[str, Any]
) -> dict[str, Any]:
    offline_nodes = {str(node["id"]): node for node in offline["nodes"]}
    online_nodes = {str(node["id"]): node for node in online["nodes"]}
    offline_actions = [node["atomic_action"] for node in offline["nodes"]]
    online_actions = [node["atomic_action"] for node in online["nodes"]]
    common_ids = sorted(set(offline_nodes) & set(online_nodes))
    node_changes = []
    for node_id in common_ids:
        left = offline_nodes[node_id]
        right = online_nodes[node_id]
        changed_fields = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        if changed_fields:
            node_changes.append({"id": node_id, "changed_fields": changed_fields})
    offline_groups = {str(group["id"]): group for group in offline["task_groups"]}
    online_groups = {str(group["id"]): group for group in online["task_groups"]}
    common_group_ids = sorted(set(offline_groups) & set(online_groups))
    group_changes = []
    for group_id in common_group_ids:
        left = offline_groups[group_id]
        right = online_groups[group_id]
        changed_fields = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        if changed_fields:
            group_changes.append({"id": group_id, "changed_fields": changed_fields})
    atomic_action_difference = {
        "offline": offline_actions,
        "online": online_actions,
        "same_sequence": offline_actions == online_actions,
        "added_node_ids": sorted(set(online_nodes) - set(offline_nodes)),
        "removed_node_ids": sorted(set(offline_nodes) - set(online_nodes)),
        "changed_nodes": node_changes,
    }
    task_group_difference = {
        "offline_ids": sorted(offline_groups),
        "online_ids": sorted(online_groups),
        "added_ids": sorted(set(online_groups) - set(offline_groups)),
        "removed_ids": sorted(set(offline_groups) - set(online_groups)),
        "same_ids": set(offline_groups) == set(online_groups),
        "changed_groups": group_changes,
    }
    return {
        "offline_node_count": len(offline_actions),
        "online_node_count": len(online_actions),
        "same_action_sequence": offline_actions == online_actions,
        "offline_actions": offline_actions,
        "online_actions": online_actions,
        "added_node_ids": sorted(set(online_nodes) - set(offline_nodes)),
        "removed_node_ids": sorted(set(offline_nodes) - set(online_nodes)),
        "changed_nodes": node_changes,
        "offline_task_group_ids": sorted(offline_groups),
        "online_task_group_ids": sorted(online_groups),
        "added_task_group_ids": sorted(set(online_groups) - set(offline_groups)),
        "removed_task_group_ids": sorted(set(offline_groups) - set(online_groups)),
        "same_task_group_ids": set(offline_groups) == set(online_groups),
        "changed_task_groups": group_changes,
        "atomic_action_difference": atomic_action_difference,
        "task_group_difference": task_group_difference,
    }


def _validate_task_group_coverage(
    task: Mapping[str, Any], graph: Mapping[str, Any], *, route: str
) -> None:
    """Check explicit TaskSpec instances are neither dropped nor duplicated."""
    if task.get("level") == "L4":
        return
    expected = {
        str(item["id"])
        for item in task.get("task_instances", ())
        if isinstance(item, Mapping)
    }
    actual = {str(group["id"]) for group in graph.get("task_groups", ())}
    if expected != actual:
        raise ValueError(
            f"A/B {route} TaskGroup coverage mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


def _validate_snapshot(
    snapshot: Mapping[str, Any], *, route: str, require_full: bool = False
) -> None:
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"A/B {route} snapshot must be a mapping.")
    required = {"robot_qpos", "object_poses"}
    if require_full:
        required = set(_FULL_SNAPSHOT_KEYS)
    missing = required - set(snapshot)
    if missing:
        raise ValueError(
            f"A/B {route} snapshot is missing required state {sorted(missing)}."
        )
    qpos = torch.as_tensor(snapshot["robot_qpos"])
    if qpos.numel() == 0 or not bool(torch.isfinite(qpos).all()):
        raise ValueError(f"A/B {route} robot_qpos must be finite and non-empty.")
    object_poses = snapshot["object_poses"]
    if not isinstance(object_poses, Mapping) or not object_poses:
        raise ValueError(f"A/B {route} object_poses must be a non-empty mapping.")
    for uid, pose in object_poses.items():
        tensor = torch.as_tensor(pose)
        if not isinstance(uid, str) or not uid or tensor.numel() == 0:
            raise ValueError(f"A/B {route} object_poses contains an invalid entry.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"A/B {route} pose for {uid!r} must be finite.")
    if "articulation_state" in snapshot:
        articulation_state = snapshot["articulation_state"]
        if not isinstance(articulation_state, Mapping):
            raise ValueError(f"A/B {route} articulation_state must be a mapping.")
        for uid, state in articulation_state.items():
            if not isinstance(uid, str) or not uid:
                raise ValueError(
                    f"A/B {route} articulation_state contains invalid UID."
                )
            if not isinstance(state, Mapping):
                raise ValueError(
                    f"A/B {route} articulation state for {uid!r} must be a mapping."
                )
            if not state:
                raise ValueError(
                    f"A/B {route} articulation state for {uid!r} is empty."
                )
            for name, value in state.items():
                tensor = torch.as_tensor(value)
                if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(
                        f"A/B {route} articulation {uid!r}.{name} must be finite."
                    )
    if "camera_calibration" in snapshot:
        calibrations = snapshot["camera_calibration"]
        if not isinstance(calibrations, Mapping):
            raise ValueError(f"A/B {route} camera_calibration must be a mapping.")
        if require_full and not calibrations:
            raise ValueError(f"A/B {route} camera_calibration must not be empty.")
        for uid, calibration in calibrations.items():
            if not isinstance(uid, str) or not uid:
                raise ValueError(
                    f"A/B {route} camera_calibration contains invalid UID."
                )
            if not isinstance(calibration, Mapping):
                raise ValueError(
                    f"A/B {route} calibration for {uid!r} must be a mapping."
                )
            for name in ("intrinsics", "extrinsics"):
                if name not in calibration:
                    raise ValueError(
                        f"A/B {route} calibration for {uid!r} is missing {name}."
                    )
                tensor = torch.as_tensor(calibration[name])
                if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(
                        f"A/B {route} calibration {uid!r}.{name} must be finite."
                    )


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=str):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(repr(value).encode("utf-8"))


def _reject_private_or_live_fields(value: Any, context: str) -> None:
    """Reject oracle/grounded fields before either branch can execute."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _PRIVATE_OR_LIVE_KEYS:
                raise ValueError(f"{context} contains private/live field {key!r}.")
            _reject_private_or_live_fields(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_or_live_fields(child, f"{context}[{index}]")


def _call_preflight(
    callback: Callable[..., Any], *, route: str, graph: Mapping[str, Any], env: Any
) -> Any:
    """Call executor preflight hooks across the small supported API variants."""
    try:
        return callback()
    except TypeError as first_error:
        # Third-party branch executors often expose contextual keyword-only
        # arguments.  Retry only for an argument-binding TypeError; if the
        # callback itself raised TypeError, preserve that original failure.
        try:
            return callback(route=route, graph=graph, env=env)
        except TypeError:
            raise first_error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
