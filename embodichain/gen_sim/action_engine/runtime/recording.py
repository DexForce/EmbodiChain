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

"""Append compact per-environment execution events and a final summary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch

from embodichain.gen_sim.action_engine.domain import (
    execution_program_hash,
    seed_graph_hash,
)
from embodichain.utils.logger import log_warning

from .models import ExecutionProgram, GroundedAction, SemanticStep

__all__ = ["RuntimeRecorder"]

_SAFE_NAME = re.compile(r"[^0-9A-Za-z._-]+")


def _safe_name(value: str) -> str:
    result = _SAFE_NAME.sub("_", value).strip("._")
    if not result:
        raise ValueError("Runtime record path component must not be empty.")
    return result


def _default_output_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent / "outputs" / "action_engine"
    return Path.cwd() / "outputs" / "action_engine"


def _jsonable(value: Any, env_id: int | None = None) -> Any:
    if isinstance(value, torch.Tensor):
        item = value
        if env_id is not None and item.ndim > 0 and item.shape[0] > env_id:
            item = item[env_id]
        return item.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item, env_id) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, env_id) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


class RuntimeRecorder:
    """Record execution decisions without copying the whole program per step."""

    def __init__(
        self,
        program: ExecutionProgram,
        *,
        num_envs: int,
        run_id: str | None = None,
        episode_index: int = 0,
        output_root: str | Path | None = None,
        enabled: bool = True,
        runtime_policy: Mapping[str, Any] | None = None,
        runtime_policy_hash: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.num_envs = int(num_envs)
        self.run_id = _safe_name(
            run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else _default_output_root()
        )
        self.output_dir = (
            root
            / _safe_name(program.task)
            / self.run_id
            / f"episode_{int(episode_index):04d}"
        )
        # The validated source graph remains untouched. Runtime documents are
        # built from a detached copy and extend it only with a runtime envelope.
        self.seed_topology = deepcopy(program.seed_graph or program.raw)
        self.program_hash = (
            seed_graph_hash(program.seed_graph)
            if program.seed_graph is not None
            else execution_program_hash(program.raw)
        )
        self.step_specs = {
            str(step["id"]): deepcopy(step) for step in program.raw["semantic_steps"]
        }
        self.step_ordinals = {
            step.id: index for index, step in enumerate(program.semantic_steps, start=1)
        }
        self.events: list[list[dict[str, Any]]] = [[] for _ in range(self.num_envs)]
        self.program_metadata = {
            "schema_version": "action_engine_runtime_record_v2",
            "task": program.task,
            "run_id": self.run_id,
            "episode_index": int(episode_index),
            "program_schema_version": self.seed_topology.get("schema_version"),
            "seed_graph_hash": self.program_hash,
        }
        if runtime_policy is not None:
            if not isinstance(runtime_policy_hash, str) or not runtime_policy_hash:
                raise ValueError("Recorded runtime policy requires a non-empty hash.")
            self.program_metadata["runtime_policy"] = deepcopy(dict(runtime_policy))
            self.program_metadata["runtime_policy_hash"] = runtime_policy_hash

    def register_step(
        self,
        step: SemanticStep,
        spec: Mapping[str, Any],
    ) -> None:
        """Register a semantic step inserted by a runtime graph revision."""
        if not self.enabled:
            return
        raw = deepcopy(dict(spec))
        if str(raw.get("id")) != step.id:
            raise ValueError("Runtime step spec ID must match the semantic step ID.")
        existing = self.step_specs.get(step.id)
        if existing is not None:
            if existing != raw:
                raise ValueError(
                    f"Runtime step {step.id!r} was registered with a different spec."
                )
            return
        self.step_specs[step.id] = raw
        self.step_ordinals[step.id] = max(self.step_ordinals.values(), default=0) + 1

    def edge(
        self,
        edge_id: str,
        step: SemanticStep,
        *,
        assignments: list[str | None],
        grounded: list[GroundedAction],
        active: torch.Tensor,
        failed: torch.Tensor,
        action_steps: int,
        planner_traces: Sequence[Mapping[str, Any]] = (),
        diagnostics: Sequence[str] = (),
        phase: str = "primary",
    ) -> None:
        if not self.enabled:
            return
        if phase not in {"primary", "recovery", "replay", "final_revalidation"}:
            raise ValueError(f"Unknown execution phase {phase!r}.")
        for env_id in range(self.num_envs):
            event = {
                "event": "edge",
                "phase": phase,
                "edge_id": edge_id,
                "semantic_step_id": step.id,
                "operator": step.operator,
                "object": step.object_uid,
                "arm": assignments[env_id],
                "status": (
                    "skipped"
                    if not bool(active[env_id])
                    else ("failed" if bool(failed[env_id]) else "executed")
                ),
                "actions": [
                    {
                        "class": item.action_class,
                        "control": item.control,
                        "target_object_pose": _jsonable(
                            item.target_object_pose, env_id
                        ),
                        "motion_policy": _jsonable(item.motion_policy),
                    }
                    for item in grounded
                ],
                "trajectory_steps": (int(action_steps) if bool(active[env_id]) else 0),
                "time_utc": datetime.now(timezone.utc).isoformat(),
            }
            if diagnostics:
                event["diagnostics"] = [str(item) for item in diagnostics]
            if planner_traces:
                event["planner_attempts"] = _jsonable(planner_traces, env_id)
            self.events[env_id].append(event)

    def step(
        self,
        step: SemanticStep,
        success: torch.Tensor,
        *,
        observed: torch.Tensor | None,
        target: torch.Tensor | None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        phase: str = "primary",
    ) -> None:
        if not self.enabled:
            return
        if phase not in {"primary", "recovery", "replay", "final_revalidation"}:
            raise ValueError(f"Unknown execution phase {phase!r}.")
        if metadata is not None and len(metadata) != self.num_envs:
            raise ValueError("Runtime step metadata must match num_envs.")
        for env_id in range(self.num_envs):
            event = {
                "event": "semantic_step",
                "phase": phase,
                "semantic_step_id": step.id,
                "status": "success" if bool(success[env_id]) else "failed",
                "observed_position": _jsonable(observed, env_id),
                "target_position": _jsonable(target, env_id),
                "time_utc": datetime.now(timezone.utc).isoformat(),
            }
            if metadata is not None:
                event.update(_jsonable(dict(metadata[env_id])))
            self.events[env_id].append(event)
            self._write_step_checkpoint(env_id, step, event)

    def recovery(
        self,
        *,
        failure_type: str,
        failed_node_id: str,
        active: torch.Tensor,
        status: str,
        recovery_group_id: str | None = None,
        error: str | None = None,
        semantic_step_id: str | None = None,
    ) -> None:
        """Record one bounded local-recovery phase for the selected rows."""
        if not self.enabled:
            return
        if status not in {"started", "succeeded", "failed", "rejected"}:
            raise ValueError(f"Unknown recovery status {status!r}.")
        for env_id in range(self.num_envs):
            if not bool(active[env_id]):
                continue
            event = {
                "event": "local_recovery",
                "phase": "recovery",
                "failure_type": str(failure_type),
                "failed_node_id": str(failed_node_id),
                "recovery_group_id": recovery_group_id,
                "status": status,
                "error": error,
                "time_utc": datetime.now(timezone.utc).isoformat(),
            }
            if semantic_step_id is not None:
                event["semantic_step_id"] = str(semantic_step_id)
            self.events[env_id].append(event)

    def _env_dir(self, env_id: int) -> Path:
        return self.output_dir / f"env_{env_id:04d}"

    def _write_step_checkpoint(
        self,
        env_id: int,
        step: SemanticStep,
        event: dict[str, Any],
    ) -> None:
        """Atomically publish one closed-loop semantic-step checkpoint."""
        related_events = [
            deepcopy(item)
            for item in self.events[env_id]
            if item.get("semantic_step_id") == step.id
        ]
        checkpoint = {
            "schema_version": "action_engine_semantic_checkpoint_v2",
            "seed_graph_hash": self.program_hash,
            "task": self.program_metadata["task"],
            "run_id": self.run_id,
            "episode_index": self.program_metadata["episode_index"],
            "env_id": env_id,
            "semantic_step": deepcopy(self.step_specs[step.id]),
            "status": event["status"],
            "events": related_events,
            "checkpointed_at_utc": event["time_utc"],
        }
        ordinal = self.step_ordinals[step.id]
        filename = f"step_{ordinal:04d}_{_safe_name(step.id)}.json"
        _write_json_atomic(
            self._env_dir(env_id) / "checkpoints" / filename,
            checkpoint,
        )

    def finalize(
        self,
        success: torch.Tensor,
        *,
        error: str | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        from embodichain.gen_sim.action_engine.graph_visualization import (
            render_task_graph_png,
        )

        finished_at = datetime.now(timezone.utc).isoformat()
        for env_id in range(self.num_envs):
            runtime = {
                **self.program_metadata,
                "env_id": env_id,
                "status": (
                    "aborted"
                    if error is not None
                    else ("success" if bool(success[env_id]) else "failed")
                ),
                "error": error,
                "events": self.events[env_id],
                "finished_at_utc": finished_at,
            }
            document = deepcopy(self.seed_topology)
            document["runtime"] = runtime
            env_dir = self._env_dir(env_id)
            _write_json_atomic(
                env_dir / "task_graph.json",
                document,
            )
            try:
                png = render_task_graph_png(document)
                if not isinstance(png, bytes):
                    raise TypeError("render_task_graph_png must return bytes.")
                _write_bytes_atomic(env_dir / "task_graph.png", png)
            except Exception as exc:
                runtime["visualization_error"] = f"{type(exc).__name__}: {exc}"
                document["runtime"] = runtime
                _write_json_atomic(env_dir / "task_graph.json", document)
                log_warning(
                    "Unable to render Action Engine runtime graph for "
                    f"env {env_id}: {exc}"
                )
        return self.output_dir.as_posix()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    """Write one binary artifact without exposing a partial destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
