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

"""Load or compile execution programs without publishing intermediate copies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_engine.protocol import (
    ACTION_ENGINE_CONFIG_SCHEMA,
    EXECUTION_PROGRAM_SCHEMA,
    SEED_GRAPH_SCHEMA,
)

from .models import ExecutionProgram

__all__ = [
    "load_agent_execution_program",
    "load_execution_program",
]


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read {label} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} at {path} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object.")
    return dict(value)


def load_execution_program(
    source: Mapping[str, Any] | str | Path,
    *,
    known_objects: set[str] | None = None,
    registry: Any | None = None,
    require_executable: bool = True,
) -> ExecutionProgram:
    """Load a v3 SeedGraph and reject every legacy execution schema."""
    value = (
        dict(source)
        if isinstance(source, Mapping)
        else _read_json(Path(source).expanduser().resolve(), label="execution program")
    )
    schema = value.get("schema_version")
    if schema == SEED_GRAPH_SCHEMA:
        from embodichain.gen_sim.action_engine.capabilities import (
            build_atomic_capability_registry,
        )
        from embodichain.gen_sim.action_engine.compiler import (
            seed_graph_to_execution_program,
        )
        from embodichain.gen_sim.action_engine.domain import validate_seed_graph
        from embodichain.gen_sim.action_engine.planning.linker import (
            validate_persisted_contracts,
        )

        registry = registry or build_atomic_capability_registry()
        seed = validate_seed_graph(
            value,
            known_objects=known_objects,
            known_actions=registry.names(),
            executable_actions=registry.executable_names(),
            require_executable=require_executable,
        )
        validate_persisted_contracts(seed, registry)
        internal = seed_graph_to_execution_program(
            seed,
            known_objects=known_objects,
            registry=registry,
            require_executable=require_executable,
        )
        return replace(ExecutionProgram.from_mapping(internal), seed_graph=seed)
    if schema == "action_engine_seed_graph_v2":
        raise ValueError(
            "SeedGraph v2 lacks persisted Action Contracts and cannot be loaded; "
            "regenerate seed_task_graph.json and agent_config.json with the current "
            "generator to produce action_engine_seed_graph_v3."
        )
    if schema == EXECUTION_PROGRAM_SCHEMA:
        raise ValueError(
            "Action Engine v1 execution programs are no longer accepted; "
            "regenerate the task to produce action_engine_seed_graph_v3."
        )
    raise ValueError(f"Unsupported Action Engine graph schema {schema!r}.")


def _resolve_config_path(
    config: Mapping[str, Any],
    config_path: str | Path,
    *keys: str,
) -> Path | None:
    base = Path(config_path).expanduser().resolve().parent
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"agent_config.{key} must be a non-empty path string.")
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()
    return None


def load_agent_execution_program(
    agent_config: Mapping[str, Any],
    *,
    agent_config_path: str | Path,
    regenerate: bool = False,
    require_executable: bool = True,
) -> ExecutionProgram:
    """Resolve an agent config and optionally rebuild its SeedGraph in memory.

    ``--regenerate`` intentionally does not write a second graph artifact. The
    deterministic compiler result is validated and handed directly to runtime.
    """
    if agent_config.get("schema_version") != ACTION_ENGINE_CONFIG_SCHEMA:
        raise ValueError(
            "This Action Engine runtime accepts only v2 bundles. Regenerate "
            "task_spec.json, scene_requirements.json, seed_task_graph.json, "
            "and agent_config.json "
            "with the current generator."
        )
    known_objects = _known_objects(agent_config)
    task_path = _resolve_config_path(
        agent_config,
        agent_config_path,
        "task_spec",
        "task_spec_path",
    )
    execution_path = _resolve_config_path(
        agent_config,
        agent_config_path,
        "seed_task_graph",
        "seed_task_graph_path",
        "offline_seed_task_graph",
        "offline_seed_task_graph_path",
    )
    if regenerate:
        if task_path is None:
            raise ValueError("--regenerate requires agent_config.task_spec.")
        task_spec = _read_json(task_path, label="task specification")
        reference_graph = (
            _read_json(execution_path, label="SeedGraph")
            if execution_path is not None and execution_path.is_file()
            else None
        )
        program = load_execution_program(
            _regenerate_seed_graph(task_spec, reference_graph=reference_graph),
            known_objects=known_objects,
            require_executable=require_executable,
        )
    elif execution_path is None:
        if task_path is None:
            raise ValueError("agent_config requires seed_task_graph or task_spec.")
        task_spec = _read_json(task_path, label="task specification")
        program = load_execution_program(
            _regenerate_seed_graph(task_spec),
            known_objects=known_objects,
            require_executable=require_executable,
        )
    else:
        program = load_execution_program(
            execution_path,
            known_objects=known_objects,
            require_executable=require_executable,
        )
    _verify_agent_program(agent_config, program)
    _verify_program_objects(agent_config, program)
    return program


def _known_objects(agent_config: Mapping[str, Any]) -> set[str] | None:
    source = agent_config.get("source")
    if not isinstance(source, Mapping):
        return None
    uid_map = source.get("uid_map")
    if not isinstance(uid_map, Mapping):
        return None
    values = {str(uid) for uid in uid_map.values() if str(uid)}
    return values or None


def _verify_agent_program(
    agent_config: Mapping[str, Any],
    program: ExecutionProgram,
) -> None:
    """Reject a valid program that belongs to a different generated bundle."""
    configured_task = agent_config.get("task_name")
    if configured_task is not None and configured_task != program.task:
        raise ValueError(
            f"agent_config.task_name {configured_task!r} does not match "
            f"execution program task {program.task!r}."
        )
    expected_hash = agent_config.get("seed_task_graph_hash")
    if expected_hash is None:
        return
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "agent_config.seed_task_graph_hash must be a non-empty string."
        )
    if program.seed_graph is not None:
        from embodichain.gen_sim.action_engine.domain import seed_graph_hash

        actual_hash = seed_graph_hash(program.seed_graph)
    else:
        from embodichain.gen_sim.action_engine.domain import execution_program_hash

        actual_hash = execution_program_hash(program.raw)
    if actual_hash != expected_hash:
        raise ValueError(
            "SeedGraph hash does not match agent_config; regenerate the "
            "configuration bundle before running it."
        )


def _regenerate_seed_graph(
    task_spec: Mapping[str, Any],
    *,
    reference_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from embodichain.gen_sim.action_engine.domain import validate_task_spec

    task = validate_task_spec(task_spec)
    oracle = task.get("oracle", {})
    reference = oracle.get("reference_seed_graph")
    if isinstance(reference, Mapping):
        return dict(reference)
    metadata = task.get("metadata", {})
    bindings = metadata.get("role_bindings", {})
    if not isinstance(bindings, Mapping):
        raise ValueError("TaskSpec.metadata.role_bindings must be a mapping.")
    if not bindings and reference_graph is not None:
        graph_metadata = reference_graph.get("metadata", {})
        if not isinstance(graph_metadata, Mapping):
            raise ValueError("SeedGraph.metadata must be a mapping.")
        bindings = graph_metadata.get("role_bindings", {})
        if not isinstance(bindings, Mapping):
            raise ValueError("SeedGraph.metadata.role_bindings must be a mapping.")
    from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph

    return instantiate_seed_graph(task, bindings)


def _verify_program_objects(
    agent_config: Mapping[str, Any],
    program: ExecutionProgram,
) -> None:
    known = _known_objects(agent_config)
    if known is None:
        return
    references = {step.object_uid for step in program.semantic_steps}
    for step in program.semantic_steps:
        for key in (
            "reference_object",
            "support_object",
            "orientation_reference_object",
        ):
            value = step.goal.get(key)
            if isinstance(value, str):
                references.add(value)
        for payload in step.goal.get("payloads", []):
            value = payload.get("object") if isinstance(payload, Mapping) else payload
            if isinstance(value, str):
                references.add(value)
    unknown = references - known - {"self", "table", "table_center"}
    if unknown:
        raise ValueError(
            "Execution Program references objects not present in the scene: "
            f"{sorted(unknown)}. Regenerate the configuration bundle."
        )
