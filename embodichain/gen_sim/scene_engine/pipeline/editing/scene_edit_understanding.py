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

from embodichain.gen_sim.scene_engine.core.scene_edit_plan import (
    SceneEditOperation,
    SceneEditPlan,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraph
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)

_EDIT_SYSTEM_PROMPT = """You convert one user instruction into edits for an existing tabletop scene.

Use an existing object ID only when it appears in the supplied Existing object
IDs list. IDs identify existing objects exactly; never invent, correct, or
renumber them. The table ID is "table" and cannot be moved or deleted.

Each operation is one of:
1. move: move one existing object. object_id identifies it. target_id and
   relation are either both provided or both null.
2. delete: delete one existing object. Only object_id is provided.
3. add: create one new object. object_id must be null. Provide a lower-case
   singular snake_case category, name, and description. Multiple add operations
   may have the same category and name; their final IDs are assigned by the
   program in operation order. target_id and relation are either both provided
   or both null.

For a positioned move or add, target_id must be an Existing object ID and
relation must be one of on, left_of, right_of, in_front_of, or behind. Do not
position a new object relative to another newly added object.

Each existing object's center_xy is its center position [x, y] in the
table-frame Z-up world coordinate system. Smaller x is left, larger x is right,
larger y is in front, and smaller y is behind. Use center_xy only to disambiguate
references such as "the bottle on the left"; do not output coordinates. Express
the requested position using target_id and one allowed relation instead.

For every newly added object, category is its lower-case singular snake_case
class. name contains only color, material, texture, shape, and object details.
description contains only visible category, material, color, texture, shape,
and structural details. name and description must not mention position, the
table, or relations to any object.

Return JSON only: no Markdown, comments, or prose. Every operation must contain
exactly these fields: op, object_id, target_id, relation, category, name, and
description. Use null for every field that does not apply to an operation:
{
  "operations": [
    {
      "op": "move",
      "object_id": "bottle_001",
      "target_id": "book_001",
      "relation": "right_of",
      "category": null,
      "name": null,
      "description": null
    },
    {
      "op": "delete",
      "object_id": "cup_001",
      "target_id": null,
      "relation": null,
      "category": null,
      "name": null,
      "description": null
    },
    {
      "op": "add",
      "object_id": null,
      "target_id": null,
      "relation": null,
      "category": "orange",
      "name": "small orange",
      "description": "small round orange with a textured peel"
    },
    {
      "op": "add",
      "object_id": null,
      "target_id": "book_001",
      "relation": "right_of",
      "category": "orange",
      "name": "small orange",
      "description": "small round orange with a textured peel"
    }
  ]
}
The two orange additions intentionally share category and name. Do not add
fields beyond the required schema."""


def understand_scene_edit(
    *,
    scene: Scene,
    scene_graph: SceneGraph,
    edit_prompt: str,
    vlm_client: OpenAICompatibleVLM,
    json_max_attempts: int = 3,
) -> SceneEditPlan:
    """Understand one text edit instruction for an existing scene."""
    edit_prompt = edit_prompt.strip()
    if not edit_prompt:
        raise ValueError("Edit prompt must not be empty.")
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")
    # Give the VLM only the scene metadata needed to identify existing objects.
    simplified_scene_info = _simplify_scene_info(scene=scene)
    operations = _vlm_understand_scene_edit(
        scene=scene,
        edit_prompt=edit_prompt,
        simplified_scene_info=simplified_scene_info,
        vlm_client=vlm_client,
        json_max_attempts=json_max_attempts,
    )

    # SceneEditPlan validates all references against the immutable input scene graph.
    return SceneEditPlan(
        scene=scene,
        scene_graph=scene_graph,
        operations=operations,
    )


def _simplify_scene_info(scene: Scene) -> dict[str, object]:
    """Return the object metadata needed for edit instruction resolution."""
    return {
        "existing_object_ids": [scene_object.id for scene_object in scene.objects],
        "objects": [
            {
                "id": scene_object.id,
                "category": scene_object.category,
                "name": scene_object.name,
                "description": scene_object.description,
                "center_xy": scene_object.center_xy,
            }
            for scene_object in scene.objects
        ],
    }


def _vlm_understand_scene_edit(
    *,
    scene: Scene,
    edit_prompt: str,
    simplified_scene_info: dict[str, object],
    vlm_client: OpenAICompatibleVLM,
    json_max_attempts: int,
) -> list[SceneEditOperation]:
    """Return parsed edit operations from the VLM with assigned add IDs."""
    # Construct user prompt.
    user_prompt = (
        f"User edit instruction:\n{edit_prompt}\n\n"
        "Existing scene metadata:\n"
        f"{json.dumps(simplified_scene_info, indent=2, ensure_ascii=False)}"
    )
    last_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vlm_client.complete(
            system_prompt=_EDIT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            value = json.loads(_strip_json_code_fence(response_text))
            return _parse_scene_edit_operations(value, scene=scene)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = ValueError(f"VLM returned invalid scene edit JSON: {exc}")
            continue

    assert last_error is not None
    raise ValueError(
        "VLM returned invalid scene edit JSON after "
        f"{json_max_attempts} attempts: {last_error}"
    ) from last_error


def _strip_json_code_fence(response_text: str) -> str:
    """Remove one optional Markdown JSON fence from a VLM response."""
    stripped = response_text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("VLM response contains an incomplete JSON code fence.")
    return "\n".join(lines[1:-1]).strip()


def _parse_scene_edit_operations(
    value: object,
    *,
    scene: Scene,
) -> list[SceneEditOperation]:
    """Parse the strict VLM edit-draft schema into typed operations with add IDs."""
    if not isinstance(value, dict) or set(value) != {"operations"}:
        raise ValueError("Scene edit draft must contain exactly operations.")
    # Get and validate a list operation value.
    operations_value = value["operations"]
    if not isinstance(operations_value, list):
        raise ValueError("Scene edit draft operations must be a list.")

    expected_keys = {
        "op",
        "object_id",
        "target_id",
        "relation",
        "category",
        "name",
        "description",
    }
    # Get ids and counts of existing objects to assign new add IDs.
    assigned_object_ids = {scene_object.id for scene_object in scene.objects}
    category_counts = {
        category: sum(
            scene_object.category == category for scene_object in scene.objects
        )
        for category in {scene_object.category for scene_object in scene.objects}
    }
    operations: list[SceneEditOperation] = []
    for value in operations_value:
        if not isinstance(value, dict) or not isinstance(value.get("op"), str):
            raise ValueError("Scene edit operations must contain a string op.")
        op = value["op"]
        if op not in {"add", "move", "delete"}:
            raise ValueError("Scene edit operation op is invalid.")
        if set(value) != expected_keys:
            raise ValueError("Scene edit operations must use the required schema.")
        object_id = _optional_string(value.get("object_id"), field_name="object_id")
        category = _optional_string(value.get("category"), field_name="category")
        if op == "add":
            if object_id is not None:
                raise ValueError("VLM add operations must set object_id to null.")
            if category is None:
                raise ValueError("VLM add operations must provide a category.")
            # Add operation should generate new id here.
            # Never believe the LLM could always generate a valid id.
            object_id = _next_add_object_id(
                category=category,
                category_counts=category_counts,
                assigned_object_ids=assigned_object_ids,
            )
        operations.append(
            SceneEditOperation(
                op=op,
                object_id=object_id,
                target_id=_optional_string(
                    value.get("target_id"), field_name="target_id"
                ),
                relation=_optional_relation(value.get("relation")),
                category=category,
                name=_optional_string(value.get("name"), field_name="name"),
                description=_optional_string(
                    value.get("description"), field_name="description"
                ),
            )
        )
    return operations


def _next_add_object_id(
    *,
    category: str,
    category_counts: dict[str, int],
    assigned_object_ids: set[str],
) -> str:
    """Assign the next available ID for one new object category."""
    index = category_counts.get(category, 0) + 1
    object_id = f"{category}_{index:03d}"
    # In case the scene have orange_001 and orange_003.
    while object_id in assigned_object_ids:
        index += 1
        object_id = f"{category}_{index:03d}"
    category_counts[category] = index
    assigned_object_ids.add(object_id)
    return object_id


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Scene edit operation {field_name} must be a string or null.")
    return value.strip()


def _optional_relation(value: object) -> str | None:
    if value is None:
        return None
    if value not in {"on", "left_of", "right_of", "in_front_of", "behind"}:
        raise ValueError("Scene edit operation relation is invalid.")
    return value
