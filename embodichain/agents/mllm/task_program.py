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

"""Strict MLLM frontend for declarative Task Program JSON responses."""

from __future__ import annotations

from collections.abc import Iterator

from embodichain.lab.task_program.language.schema import (
    TaskProgramCfg,
    TaskProgramIntegrationCfg,
    HandOverCfg,
    InvokeCfg,
    PickCfg,
    PlaceCfg,
    ProgramNodeCfg,
    RepeatCfg,
    SegmentCfg,
    SemanticCallCfg,
    SequenceCfg,
)
from embodichain.lab.task_program.compiler import CompiledTaskProgram
from embodichain.lab.task_program.language.decoder import (
    ConfigPath,
    TaskProgramDecodeError,
    TaskProgramValidationContext,
    decode_task_program,
    validate_task_program,
)
from embodichain.lab.task_program.integrations.environment import (
    TaskProgramEnvironmentAdapter,
)
from embodichain.lab.task_program.language.loader import (
    MAX_TASK_PROGRAM_BYTES,
    parse_task_program_json,
)

__all__ = [
    "compile_mllm_task_program",
    "decode_mllm_task_program",
]

_CURATED_CALL_TYPES = (
    PickCfg,
    PlaceCfg,
    HandOverCfg,
)


def _iter_calls(
    node: ProgramNodeCfg,
    *,
    path: ConfigPath,
) -> Iterator[tuple[SemanticCallCfg, ConfigPath]]:
    """Yield every semantic call and its decoder-compatible source path."""
    if type(node) is InvokeCfg:
        yield node.call, (*path, "call")
        return
    if type(node) is SequenceCfg:
        for index, child in enumerate(node.items):
            yield from _iter_calls(child, path=(*path, "items", index))
        return
    if type(node) is RepeatCfg:
        yield from _iter_calls(node.body, path=(*path, "body"))
        return
    if type(node) is SegmentCfg:
        yield from _iter_calls(node.steps, path=(*path, "steps"))
        return
    raise TaskProgramDecodeError(
        "mllm_program_node_not_allowed",
        (*path, "kind"),
        "The MLLM frontend permits only sequential program nodes.",
    )


def _value_at_path(value: object, path: ConfigPath) -> object:
    """Return a raw decoded JSON value at one already validated config path."""
    current = value
    for part in path:
        if type(part) is int:
            if type(current) is not list or not 0 <= part < len(current):
                raise TaskProgramDecodeError(
                    "mllm_payload_mismatch",
                    path,
                    "Decoded model payload no longer matches the canonical program.",
                )
            current = current[part]
        else:
            if type(current) is not dict or part not in current:
                raise TaskProgramDecodeError(
                    "mllm_payload_mismatch",
                    path,
                    "Decoded model payload no longer matches the canonical program.",
                )
            current = current[part]
    return current


def _validate_mllm_policy(
    config: TaskProgramCfg,
    *,
    raw_payload: dict[str, object],
) -> None:
    """Apply the narrow agent-facing policy after canonical decoding."""
    for call, path in _iter_calls(config.program, path=("program",)):
        if type(call) not in _CURATED_CALL_TYPES:
            raise TaskProgramDecodeError(
                "mllm_call_not_allowed",
                (*path, "kind"),
                "The MLLM frontend permits only curated pick, place, and "
                "hand_over calls.",
            )
        raw_call = _value_at_path(raw_payload, path)
        if type(raw_call) is not dict:
            raise TaskProgramDecodeError(
                "mllm_payload_mismatch",
                path,
                "Decoded model payload no longer matches the canonical program.",
            )
        raw_resources = raw_call.get("resources", {})
        if type(raw_resources) is dict and raw_resources:
            raise TaskProgramDecodeError(
                "mllm_resource_override_not_allowed",
                (*path, "resources"),
                "MLLM responses cannot override robot resource bindings.",
            )
        if call.resources:
            raise TaskProgramDecodeError(
                "mllm_resource_override_not_allowed",
                (*path, "resources"),
                "MLLM responses cannot override robot resource bindings.",
            )


def decode_mllm_task_program(
    response: str,
    *,
    integration: TaskProgramIntegrationCfg,
    validation_context: TaskProgramValidationContext | None = None,
    max_bytes: int = MAX_TASK_PROGRAM_BYTES,
) -> TaskProgramCfg:
    """Decode one untrusted model response into the canonical program config.

    The model response is a single plain JSON object containing ``program_id``,
    ``targets``, and ``program``. The trusted host supplies ``integration``; a
    response attempting to select its own integration is rejected rather than
    silently overwritten. Only the sequential subset and curated built-in calls
    are admitted, and robot resource overrides are forbidden.

    Args:
        response: Untrusted model response containing one plain JSON document.
        integration: Host-owned scene, robot-profile, and runtime-preset choice.
        validation_context: Optional provider-free static reference validator.
        max_bytes: Maximum UTF-8 encoded response size.

    Returns:
        An owned canonical :class:`TaskProgramCfg`.

    Raises:
        TypeError: If ``integration`` is not an exact integration config.
        TaskProgramDecodeError: If JSON, schema, or MLLM policy validation
            fails.
    """
    if type(integration) is not TaskProgramIntegrationCfg:
        raise TypeError("integration must be exactly TaskProgramIntegrationCfg.")
    data = parse_task_program_json(response, max_bytes=max_bytes)
    if "integration" in data:
        raise TaskProgramDecodeError(
            "model_controlled_integration",
            ("integration",),
            "MLLM responses cannot select an integration; the host injects it.",
        )
    payload = dict(data)
    payload["integration"] = {
        "robot_profile": integration.robot_profile,
        "scene_registry": integration.scene_registry,
        "runtime_preset": integration.runtime_preset,
    }
    config = decode_task_program(payload)
    _validate_mllm_policy(config, raw_payload=payload)
    if validation_context is not None:
        validate_task_program(config, validation_context)
    return config


def compile_mllm_task_program(
    response: str,
    *,
    adapter: TaskProgramEnvironmentAdapter,
    integration: TaskProgramIntegrationCfg,
    validation_context: TaskProgramValidationContext | None = None,
    max_bytes: int = MAX_TASK_PROGRAM_BYTES,
) -> CompiledTaskProgram:
    """Decode and compile a model response through the existing environment path.

    This function introduces no MLLM-specific compiler. It delegates the owned
    config to :meth:`TaskProgramEnvironmentAdapter.compile`, which performs the
    canonical scene resolution and Task Program lowering used by every other
    frontend.

    Args:
        response: Untrusted model response containing one plain JSON document.
        adapter: Existing trusted Task Program environment adapter.
        integration: Host-owned scene, robot-profile, and runtime-preset choice.
        validation_context: Optional provider-free static reference validator.
        max_bytes: Maximum UTF-8 encoded response size.

    Returns:
        Provider-free program produced by the existing Task Program compiler.

    Raises:
        TypeError: If ``adapter`` or ``integration`` has the wrong exact type.
        TaskProgramDecodeError: If JSON, schema, or MLLM policy validation
            fails.
    """
    if type(adapter) is not TaskProgramEnvironmentAdapter:
        raise TypeError("adapter must be exactly TaskProgramEnvironmentAdapter.")
    config = decode_mllm_task_program(
        response,
        integration=integration,
        validation_context=validation_context,
        max_bytes=max_bytes,
    )
    return adapter.compile(config)
