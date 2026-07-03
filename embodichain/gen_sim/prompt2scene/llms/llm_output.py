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

import json
from typing import Any, Callable

__all__ = [
    "bind_structured_output",
    "coerce_json_object_output",
    "is_model_output_error",
    "call_structured_json_model_step",
    "StructuredModelCallError",
    "validate_json_schema",
]


class StructuredModelCallError(Exception):
    """Retryable structured-model call failure."""

    def __init__(
        self,
        *,
        context: str,
        attempt_count: int,
        original_exc: Exception,
    ) -> None:
        self.context = context
        self.attempt_count = attempt_count
        self.original_exc = original_exc
        super().__init__(str(original_exc))


def bind_structured_output(llm: Any, schema: dict[str, Any]) -> Any:
    """Bind a JSON schema to an LLM when the model wrapper supports it."""
    if hasattr(llm, "with_structured_output"):
        return llm.with_structured_output(schema)
    return llm


def coerce_json_object_output(response: Any, *, context: str) -> dict[str, Any]:
    """Coerce a model response into a JSON object."""
    if isinstance(response, dict):
        return response

    content = getattr(response, "content", response)
    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = "\n".join(text_parts)

    if isinstance(content, str):
        return _parse_json_text(content, context=context)

    raise ValueError(f"{context} model output has unsupported type: {type(response)!r}")


def is_model_output_error(exc: Exception) -> bool:
    """Return whether an exception is a retryable model output formatting error."""
    class_name = exc.__class__.__name__
    module_name = exc.__class__.__module__
    return class_name in {
        "JSONDecodeError",
        "OutputParserException",
        "SchemaValidationError",
        "ValidationError",
        "StructuredModelCallError",
    } or module_name.startswith("pydantic")


def validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    context: str,
) -> None:
    """Validate model output against the subset of JSON Schema used locally."""
    _validate_schema_value(value, schema, path=context)


def call_structured_json_model_step(
    *,
    llm: Any,
    schema: dict[str, Any],
    messages: list[dict[str, Any]],
    context: str,
    attempt_count: int,
    raw_output_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Call a structured-output model, validate JSON, and persist raw output."""
    model = bind_structured_output(llm, schema)
    try:
        response = model.invoke(messages)
        raw_model_output = coerce_json_object_output(response, context=context)
        validate_json_schema(
            raw_model_output,
            schema,
            context=f"{context} output",
        )
    except Exception as exc:
        if is_model_output_error(exc) or isinstance(exc, ValueError):
            raise StructuredModelCallError(
                context=context,
                attempt_count=attempt_count,
                original_exc=exc,
            ) from exc
        raise

    if raw_output_writer is not None:
        raw_output_writer(raw_model_output)
    return raw_model_output


def _parse_json_text(content: str, *, context: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError(f"{context} model output must be a JSON object.")
    return parsed


def _validate_schema_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None:
        _validate_type(value, expected_type, path=path)

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        raise ValueError(f"{path} must be one of {enum_values}.")

    if expected_type == "object" or isinstance(value, dict):
        _validate_object(value, schema, path=path)
    elif expected_type == "array" or isinstance(value, list):
        _validate_array(value, schema, path=path)
    elif expected_type == "string" or isinstance(value, str):
        _validate_string(value, schema, path=path)
    elif expected_type in {"integer", "number"}:
        _validate_number(value, schema, path=path)


def _validate_type(value: Any, expected_type: Any, *, path: str) -> None:
    if isinstance(expected_type, list):
        if any(_matches_type(value, item) for item in expected_type):
            return
        raise ValueError(f"{path} must match one of these types: {expected_type}.")

    if not _matches_type(value, expected_type):
        raise ValueError(f"{path} must be {expected_type}.")


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _validate_object(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(value, dict):
        return

    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}

    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} missing required keys: {missing}.")

    if schema.get("additionalProperties") is False:
        extra = sorted(set(value) - set(properties))
        if extra:
            raise ValueError(f"{path} has unexpected keys: {extra}.")

    for key, child_schema in properties.items():
        if key not in value or not isinstance(child_schema, dict):
            continue
        _validate_schema_value(value[key], child_schema, path=f"{path}.{key}")


def _validate_array(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(value, list):
        return

    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        raise ValueError(f"{path} must contain at least {min_items} items.")

    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        raise ValueError(f"{path} must contain at most {max_items} items.")

    items_schema = schema.get("items")
    if not isinstance(items_schema, dict):
        return

    for index, item in enumerate(value):
        _validate_schema_value(item, items_schema, path=f"{path}[{index}]")


def _validate_string(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(value, str):
        return

    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        raise ValueError(f"{path} must contain at least {min_length} characters.")

    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(value) > max_length:
        raise ValueError(f"{path} must contain at most {max_length} characters.")


def _validate_number(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return

    minimum = schema.get("minimum")
    if isinstance(minimum, int | float) and value < minimum:
        raise ValueError(f"{path} must be greater than or equal to {minimum}.")
