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
from pathlib import Path
import re
import shutil

from embodichain.gen_sim.scene_engine.core.asset import Asset
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.table import Table
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)


_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_LOCATION_WORD_PATTERN = re.compile(
    r"\b(?:left|right|front|back|center|middle|top|bottom|upper|lower|"
    r"foreground|background|near|next|beside|behind|between|on|in|inside|"
    r"under|above|below|against)\b",
    flags=re.IGNORECASE,
)
_SYSTEM_PROMPT = """You inspect one tabletop-scene image.
Identify the main table and every visible, physically distinct object that should
be segmented and later generated as an independent 3D asset.

Rules:
1. Ignore people, floor, carpet, walls, ceiling, doors, tiny incidental items,
   and objects cut off by the image border.
2. Merge visually or functionally unified units, such as a potted plant, a vase
   with flowers, or one built-in cabinet system.
3. Do not merge objects merely resting on another object. A mug on a table and
   the table are separate entries.
4. List every visible physical instance separately. If two objects look alike,
   keep the same category and name, but distinguish them in description using
   location. Do not add location to name.
5. category is a lower-case singular snake_case class, such as mug, book,
   potted_plant, or coffee_table. It must not contain color or material.
6. name contains only color, material, texture, shape, and object description.
   It must not contain position or relations, such as left, right, on, in, or
   near.
7. For table, description contains only its category, material, color, texture,
   shape, and visible structural details. Do not mention image coverage, image
   position, camera framing, or viewpoint. For example, do not write "occupying
   most of the image" or "at the center of the image".
8. For assets, description may include all visible details, including location
   and spatial context.

Return JSON only: no Markdown, comments, or prose outside this exact schema:
{
  "table": {
    "category": "coffee_table",
    "name": "light wood coffee table",
    "description": "low rectangular light wood coffee table with a smooth wood surface"
  },
  "assets": [
    {
      "category": "mug",
      "name": "blue ceramic mug",
      "description": "small blue ceramic mug on the left side of the table"
    }
  ]
}
For two identical blue mugs, output two asset entries with the same category and
name, and use their descriptions to state left/right or front/back. Do not
infer objects that are not visible. Use an empty assets array when no objects
are visible. Every field must be a non-empty string."""

_USER_PROMPT = (
      "Analyze the provided image and return only the required JSON object."
)


def understand_scene(
    scene: Scene,
    image_path: str | Path,
    output_root: str | Path,
    *,
    vlm_client: OpenAICompatibleVLM,
    json_max_attempts: int = 3,
) -> Scene:

    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")

    resolved_image_path = _validate_image_path(image_path)
    # The output in this stage will keep a JSON which contains 
    # the Scene data structure for debugging.
    stage_output_root = Path(output_root).expanduser().resolve() / "scene_understanding"
    if stage_output_root.exists():
        shutil.rmtree(stage_output_root)
    stage_output_root.mkdir(parents=True, exist_ok=True)

    last_validation_error: ValueError | None = None
    for attempt in range(1, json_max_attempts + 1):
        response_text = vlm_client.complete(
            image_path=resolved_image_path,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_USER_PROMPT,
        )
        try:
            understood_scene = validate_scene_understanding_json(response_text)
            scene.table = understood_scene.table
            scene.assets = understood_scene.assets
            validate_scene_understanding(scene)
        except ValueError as exc:
            last_validation_error = exc
            continue

        (stage_output_root / "scene.json").write_text(
            json.dumps(scene.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return scene

    assert last_validation_error is not None
    raise ValueError(
        "VLM returned invalid scene-understanding JSON after "
        f"{json_max_attempts} attempts: {last_validation_error}"
    ) from last_validation_error


def validate_scene_understanding_json(response_text: str) -> Scene:
    """Parse a VLM response and create a core ``Scene`` with generated IDs."""
    json_text = _strip_json_code_fence(response_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM response is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict) or set(payload) != {"table", "assets"}:
        raise ValueError("VLM JSON must contain exactly the keys: table and assets.")

    id_counters: dict[str, int] = {}
    table_fields = _parse_scene_object_fields(payload["table"], field_name="table")
    table = Table(
        # id=_next_id(table_fields["category"], id_counters)
        # Use a fixed ID for the table.
        id="table",
        **table_fields,
    )
    assets_value = payload["assets"]
    if not isinstance(assets_value, list):
        raise ValueError("VLM JSON key assets must be an array.")
    assets: list[Asset] = []
    for index, asset in enumerate(assets_value):
        fields = _parse_scene_object_fields(asset, field_name=f"assets[{index}]")
        assets.append(
            Asset(
                id=_next_id(fields["category"], id_counters),
                **fields,
            )
        )

    return Scene(table=table, assets=assets)


def validate_scene_understanding(scene: Scene) -> None:
    """Validate that scene understanding produced a complete semantic scene."""
    if scene.table is None:
        raise ValueError("Scene understanding must identify a table.")
    if scene.table.id != "table": # Currently it will always return true. For we hardcode the table id to "table".
        raise ValueError("Scene table id must be 'table'.")

    asset_ids = [asset.id for asset in scene.assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Scene asset ids must be unique.")

    for obj in [scene.table, *scene.assets]:
        if not obj.category or not obj.name or not obj.description:
            raise ValueError(
                "Every scene object must contain category, name, and description."
            )


def _strip_json_code_fence(response_text: str) -> str:
    stripped = response_text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("VLM response contains an incomplete JSON code fence.")
    return "\n".join(lines[1:-1]).strip()


def _validate_image_path(image_path: str | Path) -> Path:
    resolved_image_path = Path(image_path).expanduser().resolve()
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"Image input not found: {resolved_image_path}")
    if resolved_image_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("Image input must be a .jpg, .jpeg, or .png file.")
    return resolved_image_path


def _parse_scene_object_fields(
    value: object,
    *,
    field_name: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "category",
        "name",
        "description",
    }:
        raise ValueError(
            f"VLM JSON key {field_name} must contain exactly category, name, and "
            "description."
        )

    fields = {}
    for key in ("category", "name", "description"):
        raw_value = value[key]
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(
                f"VLM JSON key {field_name}.{key} must be a non-empty string."
            )
        fields[key] = raw_value.strip()

    if not _CATEGORY_PATTERN.fullmatch(fields["category"]):
        raise ValueError(
            f"VLM JSON key {field_name}.category must be a lower-case snake_case "
            "class name."
        )
    if _LOCATION_WORD_PATTERN.search(fields["name"]): # Check whether the name contains location.
        raise ValueError(
            f"VLM JSON key {field_name}.name must not contain location or "
            "relationship words."
        )
    return fields


def _next_id(category: str, counters: dict[str, int]) -> str:
    """Auto increment an ID for the same category, e.g. mug_001, mug_002, etc."""
    counters[category] = counters.get(category, 0) + 1
    return f"{category}_{counters[category]:03d}"
