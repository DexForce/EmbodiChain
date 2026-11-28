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
import hashlib
import json
from pathlib import Path
import re
from typing import Any

__all__: list[str] = []

_ENTITY_ID_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class TaskEntity:
    """One distinct task entity and its authoritative articulation flag.

    Attributes:
        id: TaskSpec entity identifier, safe to reuse in exported filenames.
        description: TaskSpec descriptions used to find the visible entity.
        is_articulated: Explicit TaskSpec value; never inferred from semantics.
        task_references: Distinct task-reference descriptions mentioning the entity.
        aliases: TaskSpec IDs of named parts owned by this entity.
    """

    id: str
    description: str
    is_articulated: bool
    task_references: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskSpecInput:
    """Task context used to bind task entities to observed scene instances.

    Attributes:
        instruction: Original task instruction, or empty when not supplied.
        entities: Distinct entities in order of first appearance.
    """

    instruction: str
    entities: tuple[TaskEntity, ...]


def load_task_spec_input(
    task_spec: str | Path, *, image_path: str | Path | None = None
) -> TaskSpecInput:
    """Read the matching projection of a ``task_spec/v1`` JSON document.

    The hybrid route publishes entities in the ``scene_objects`` scene requirement.
    Its reserved ``table`` entry is handled by ordinary support-surface analysis.
    When it is absent, use the legacy interpretation references and preserve
    their task-reference descriptions alongside the visual descriptions.
    Deduplicate entities in order of first appearance and join distinct descriptions.
    Use named parts as additional evidence for their owning scene object.
    Missing, null or blank task instructions become an empty string; entity
    descriptions remain available for binding without an instruction.
    Articulation flags must be explicit and consistent for each entity.
    Unrelated TaskSpec fields, including bounding boxes, are not consumed.

    Args:
        task_spec: Path to the TaskSpec JSON corresponding to the source image.
        image_path: Source image to check against a declared image SHA-256.

    Returns:
        The task instruction and distinct entities for matching scene instances.

    Raises:
        ValueError: The consumed TaskSpec fields are invalid or the image differs.
        OSError: The input file cannot be read.
    """
    path = Path(task_spec)
    return _parse_task_spec_input(path.read_bytes(), path=path, image_path=image_path)


def _parse_task_spec_input(
    data: bytes,
    *,
    path: Path,
    image_path: str | Path | None,
    image_sha256: str | None = None,
) -> TaskSpecInput:
    """Parse an already-read TaskSpec, optionally using a captured image digest."""
    try:
        document = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid TaskSpec JSON in {path}: {exc.msg}") from exc

    document = _require_object(document, "TaskSpec")
    if document.get("schema_version") != "task_spec/v1":
        raise ValueError("TaskSpec.schema_version must be 'task_spec/v1'.")
    inputs = _require_object(document.get("inputs"), "TaskSpec.inputs")
    _verify_source_image(inputs, image_path, image_sha256=image_sha256)
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

    configuration = document.get("configuration")
    part_owners = _part_owners(configuration)
    aliases_by_owner: dict[str, list[str]] = {}
    role_descriptions = (
        configuration.get("descriptions") if isinstance(configuration, dict) else None
    )
    for part_id, owner_id in part_owners.items():
        if owner_id not in descriptions:
            continue
        aliases_by_owner.setdefault(owner_id, []).append(part_id)
        part_references = descriptions.get(part_id, []) + task_references.get(
            part_id, []
        )
        if isinstance(role_descriptions, dict):
            role_description = role_descriptions.get(part_id)
            if isinstance(role_description, str) and role_description.strip():
                part_references.append(role_description.strip())
        task_references[owner_id].extend(part_references)

    return TaskSpecInput(
        instruction=instruction,
        entities=tuple(
            TaskEntity(
                id=entity_id,
                description="\n".join(entity_descriptions),
                is_articulated=articulation_flags[entity_id],
                task_references=tuple(dict.fromkeys(task_references[entity_id])),
                aliases=tuple(aliases_by_owner.get(entity_id, ())),
            )
            for entity_id, entity_descriptions in descriptions.items()
            if entity_id not in part_owners
        ),
    )


def _verify_source_image(
    inputs: dict[str, Any],
    image_path: str | Path | None,
    *,
    image_sha256: str | None,
) -> None:
    if "image" not in inputs:
        return
    image = _require_object(inputs["image"], "TaskSpec.inputs.image")
    if "sha256" not in image:
        return
    expected = image["sha256"]
    if not isinstance(expected, str) or _SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError("TaskSpec.inputs.image.sha256 must be a SHA-256 digest.")
    if image_path is not None:
        path = Path(image_path).expanduser()
        actual = image_sha256
        if actual is None:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(
                f"TaskSpec source image SHA-256 does not match the supplied image: {path}"
            )


def _part_owners(configuration: object) -> dict[str, str]:
    """Resolve each named part to its outermost owner."""
    roles = configuration.get("roles") if isinstance(configuration, dict) else None
    if not isinstance(roles, list):
        return {}

    parents: dict[str, str] = {}
    for role in roles:
        properties = role.get("properties") if isinstance(role, dict) else None
        if not isinstance(properties, dict) or "part_of" not in properties:
            continue
        part_id, owner_id = role.get("id"), properties["part_of"]
        if any(
            not isinstance(uid, str) or _ENTITY_ID_PATTERN.fullmatch(uid) is None
            for uid in (part_id, owner_id)
        ):
            raise ValueError("TaskSpec part_of declarations require valid role IDs.")
        if part_id in parents and parents[part_id] != owner_id:
            raise ValueError(
                f"Conflicting TaskSpec part_of declarations for {part_id!r}."
            )
        parents[part_id] = owner_id

    owners: dict[str, str] = {}
    for part_id in parents:
        owner_id = part_id
        visited: set[str] = set()
        while owner_id in parents:
            if owner_id in visited:
                raise ValueError(
                    f"Cyclic TaskSpec part_of declarations for {part_id!r}."
                )
            visited.add(owner_id)
            owner_id = parents[owner_id]
        owners[part_id] = owner_id
    return owners


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
