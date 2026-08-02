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
import json
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_engine.protocol import ACTION_ENGINE_CONFIG_SCHEMA

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


def _validated_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    # Import lazily so pure loader tests can install a lightweight domain module.
    from embodichain.gen_sim.action_engine.domain import validate_execution_program

    candidate = dict(value)
    validated = validate_execution_program(candidate)
    if validated is None:
        return candidate
    if not isinstance(validated, Mapping):
        raise TypeError("validate_execution_program must return a mapping or None.")
    return dict(validated)


def load_execution_program(
    source: Mapping[str, Any] | str | Path,
) -> ExecutionProgram:
    """Load and validate one execution program from memory or disk."""
    value = (
        dict(source)
        if isinstance(source, Mapping)
        else _read_json(Path(source).expanduser().resolve(), label="execution program")
    )
    return ExecutionProgram.from_mapping(_validated_mapping(value))


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
) -> ExecutionProgram:
    """Resolve an agent config and optionally recompile its task agent in memory.

    ``--regenerate`` intentionally does not write a second graph artifact. The
    deterministic compiler result is validated and handed directly to runtime.
    """
    if agent_config.get("schema_version") != ACTION_ENGINE_CONFIG_SCHEMA:
        raise ValueError(
            "This Action Engine runtime accepts only v1 bundles. Regenerate "
            "task_agent.json, seed_task_graph.json, and agent_config.json "
            "with the current generator."
        )
    known_objects = _known_objects(agent_config)
    task_path = _resolve_config_path(
        agent_config,
        agent_config_path,
        "task_agent",
        "task_agent_path",
    )
    execution_path = _resolve_config_path(
        agent_config,
        agent_config_path,
        "execution_program",
        "seed_task_graph",
        "execution_program_path",
    )
    if regenerate:
        if task_path is None:
            raise ValueError("--regenerate requires agent_config.task_agent.")
        task_agent = _read_json(task_path, label="task agent")
        from embodichain.gen_sim.action_engine.compiler import compile_task_agent

        compiled = compile_task_agent(task_agent, known_objects=known_objects)
        if not isinstance(compiled, Mapping):
            raise TypeError("compile_task_agent must return a mapping.")
        program = load_execution_program(compiled)
    elif execution_path is None:
        if task_path is None:
            raise ValueError("agent_config requires execution_program or task_agent.")
        task_agent = _read_json(task_path, label="task agent")
        from embodichain.gen_sim.action_engine.compiler import compile_task_agent

        compiled = compile_task_agent(task_agent, known_objects=known_objects)
        if not isinstance(compiled, Mapping):
            raise TypeError("compile_task_agent must return a mapping.")
        program = load_execution_program(compiled)
    else:
        program = load_execution_program(execution_path)
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
    expected_hash = agent_config.get("execution_program_hash")
    if expected_hash is None:
        return
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "agent_config.execution_program_hash must be a non-empty string."
        )
    from embodichain.gen_sim.action_engine.domain import execution_program_hash

    actual_hash = execution_program_hash(program.raw)
    if actual_hash != expected_hash:
        raise ValueError(
            "Execution program hash does not match agent_config; regenerate the "
            "configuration bundle before running it."
        )


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
