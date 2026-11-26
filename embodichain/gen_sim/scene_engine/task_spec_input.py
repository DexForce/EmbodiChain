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

"""Read task instructions and explicit entities for Scene Engine matching."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

__all__: list[str] = []

_ENTITY_ID_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


@dataclass(frozen=True, slots=True)
class TaskEntity:
    """One distinct task entity and its authoritative articulation flag.

    Attributes:
        id: TaskSpec entity identifier, safe to reuse in exported filenames.
        description: TaskSpec descriptions used to find the visible entity.
        is_articulated: Explicit TaskSpec value; never inferred from semantics.
        task_references: Distinct task-reference descriptions mentioning the entity.
    """

    id: str
    description: str
    is_articulated: bool
    task_references: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskSpecInput:
    """Task context used to bind task entities to observed scene instances.

    Attributes:
        instruction: Original task instruction, or empty when not supplied.
        entities: Distinct entities in order of first appearance.
    """

    instruction: str
    entities: tuple[TaskEntity, ...]


def load_task_spec_input(task_spec: str | Path) -> TaskSpecInput:
    """Read the matching projection of a ``task_spec/v1`` JSON document.

    The hybrid route publishes entities in the ``scene_objects`` scene requirement.
    Its reserved ``table`` entry is handled by ordinary support-surface analysis.
    When it is absent, use the legacy interpretation references and preserve
    their task-reference descriptions alongside the visual descriptions.
    Deduplicate entities in order of first appearance and join distinct descriptions.
    Missing, null or blank task instructions become an empty string; entity
    descriptions remain available for binding without an instruction.
    Articulation flags must be explicit and consistent for each entity.
    Unrelated TaskSpec fields, including bounding boxes, are not consumed.

    Args:
        task_spec: Path to the TaskSpec JSON corresponding to the source image.

    Returns:
        The task instruction and distinct entities for matching scene instances.

    Raises:
        ValueError: The JSON or the consumed TaskSpec entity fields are invalid.
        OSError: The input file cannot be read.
    """
    path = Path(task_spec)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid TaskSpec JSON in {path}: {exc.msg}") from exc

    document = _require_object(document, "TaskSpec")
    if document.get("schema_version") != "task_spec/v1":
        raise ValueError("TaskSpec.schema_version must be 'task_spec/v1'.")
    inputs = _require_object(document.get("inputs"), "TaskSpec.inputs")
    instruction = inputs.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        raise ValueError("TaskSpec.inputs.instruction must be text or null.")
    if instruction is None or not instruction.strip():
        instruction = ""
    descriptions: dict[str, list[str]] = {}
    task_references: dict[str, list[str]] = {}
    articulation_flags: dict[str, bool] = {}
    for entity_field, entity, reference_description in _entity_rows(document):
        entity_id = entity.get("id")
        if (
            not isinstance(entity_id, str)
            or _ENTITY_ID_PATTERN.fullmatch(entity_id) is None
            or entity_id == "table"
        ):
            raise ValueError(
                f"{entity_field}.id must be a safe nonempty asset identifier "
                "using letters, digits, underscores, dots or hyphens; "
                "'table' is reserved."
            )
        description = entity.get("description", reference_description)
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{entity_field}.description must be nonempty text.")
        description = description.strip()
        is_articulated = entity.get("is_articulated")
        if type(is_articulated) is not bool:
            raise ValueError(
                f"{entity_field}.is_articulated must be an explicit boolean."
            )
        if entity_id in articulation_flags:
            if articulation_flags[entity_id] != is_articulated:
                raise ValueError(
                    f"Conflicting is_articulated values for entity {entity_id!r}."
                )
        else:
            descriptions[entity_id] = []
            task_references[entity_id] = []
            articulation_flags[entity_id] = is_articulated
        if description not in descriptions[entity_id]:
            descriptions[entity_id].append(description)
        if isinstance(reference_description, str) and reference_description.strip():
            normalized_reference = reference_description.strip()
            if normalized_reference not in task_references[entity_id]:
                task_references[entity_id].append(normalized_reference)

    return TaskSpecInput(
        instruction=instruction,
        entities=tuple(
            TaskEntity(
                id=entity_id,
                description="\n".join(entity_descriptions),
                is_articulated=articulation_flags[entity_id],
                task_references=tuple(task_references[entity_id]),
            )
            for entity_id, entity_descriptions in descriptions.items()
        ),
    )


def _entity_rows(
    document: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any], object]]:
    """Read the authoritative scene requirement or the legacy reference rows."""
    configuration = document.get("configuration")
    requirements = (
        configuration.get("requirements") if isinstance(configuration, dict) else None
    )
    if isinstance(requirements, list):
        scene_object_requirements = [
            (index, row)
            for index, row in enumerate(requirements)
            if isinstance(row, dict)
            and row.get("consumer") == "scene"
            and row.get("key") == "scene_objects"
        ]
        if len(scene_object_requirements) > 1:
            raise ValueError("TaskSpec declares multiple scene_objects requirements.")
        if scene_object_requirements:
            index, requirement = scene_object_requirements[0]
            value_field = f"TaskSpec.configuration.requirements[{index}].value"
            for entity_index, raw_entity in enumerate(
                _require_list(requirement.get("value"), value_field)
            ):
                entity_field = f"{value_field}[{entity_index}]"
                entity = _require_object(raw_entity, entity_field)
                if entity.get("id") != "table":
                    yield entity_field, entity, None
            return

    interpretation = _require_object(
        document.get("interpretation"), "TaskSpec.interpretation"
    )
    references = _require_list(
        interpretation.get("references"), "TaskSpec.interpretation.references"
    )
    for reference_index, raw_reference in enumerate(references):
        reference_field = f"TaskSpec.interpretation.references[{reference_index}]"
        reference = _require_object(raw_reference, reference_field)
        entities = _require_list(
            reference.get("entities"), f"{reference_field}.entities"
        )
        for entity_index, raw_entity in enumerate(entities):
            entity_field = f"{reference_field}.entities[{entity_index}]"
            yield (
                entity_field,
                _require_object(raw_entity, entity_field),
                reference.get("description"),
            )


def _require_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object.")
    return value


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array.")
    return value
