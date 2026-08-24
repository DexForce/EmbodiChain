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
from typing import Any

from PIL import Image

from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
    TABLE_OBJECT_ID,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    MaskCandidate,
    build_mask_candidates,
    render_asset_mask_id_overlay,
    render_image_without_masks,
    render_numbered_mask_candidates,
    save_binary_mask,
    union_overlapping_mask_candidates,
)

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
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
   keep the same category and name. Do not encode location or spatial context
   in any semantic field.
5. category is a lower-case singular snake_case class, such as mug, book,
   potted_plant, or coffee_table. It must not contain color or material.
6. name is a concise human-readable phrase containing only color, material,
   texture, shape, and object details. It may contain spaces, but must not
   contain position or relations, such as left, right, on, in, or near.
7. For table, description contains only its category, material, color, texture,
   shape, and visible structural details. Do not mention image coverage, image
   position, camera framing, or viewpoint. For example, do not write "occupying
   most of the image" or "at the center of the image".
8. For assets, description contains only visible category, material, color,
   texture, shape, and structural details. Do not mention location, the table,
   or any relationship to another object.
   Structural direction words are allowed when they describe the object itself:
   "bottle with a black cap on top" is valid, while "bottle on the left of the
   table" is not.

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
      "description": "small blue ceramic mug with a curved handle"
    }
  ]
}
For two identical blue mugs, output two asset entries with the same category,
name, and description. Do not infer objects that are not visible. Use an empty
assets array when no objects are visible. Every field must be a non-empty
string."""

_USER_PROMPT = "Analyze the provided image and return only the required JSON object."

_TABLE_VALIDATION_SYSTEM_PROMPT = """You select the best table mask candidate.
The image contains table-mask candidates overlaid semi-transparently on the
scene. Gray regions are already-segmented non-table assets that were
intentionally removed for this validation; ignore them. Candidate numbers only
identify masks; do not treat the number or its background as scene content.

Choose the candidate covering the main visible table. A table candidate is
acceptable when it covers the visible tabletop and/or legs, even if some edges
are incomplete, objects on the table occlude parts of it, or it slightly
overlaps those objects. Return null only when no candidate depicts the main
table. If there is one plausible candidate, select it rather than returning
null.

Examples:
- Candidate 1 covers the tabletop and legs but misses a narrow edge:
  {"selected_mask_index": 1}
- Candidate 1 is a cup and candidate 2 covers the main table:
  {"selected_mask_index": 2}
- Every candidate is an object resting on the table, not the table itself:
  {"selected_mask_index": null}

Return JSON only, with exactly one key: selected_mask_index. Use a one-based
candidate index or null. Do not include Markdown or any other text."""
_ASSET_ASSIGNMENT_SYSTEM_PROMPT = """You assign outlined mask candidates to a group of scene assets.
The image is the original scene with numbered candidate mask outlines. The
number labels identify candidates only; they are not scene content. Use the
provided category, name, and description of every asset to match each asset to
exactly one candidate. Descriptions can distinguish visually similar assets by
location.

Extra candidate masks are normal and may be ignored. Never force a candidate
onto an asset. If any listed asset has no correct candidate, return
{"assignments": null}.

Examples:
- Two listed paper cups match candidate 1 and candidate 3:
  {"assignments": [{"asset_id": "paper_cup_001", "mask_index": 1}, {"asset_id": "paper_cup_002", "mask_index": 3}]}
- A listed asset is absent from every candidate:
  {"assignments": null}

Return JSON only, with exactly one key: assignments. It must be null or an
array of asset_id and mask_index objects. Do not include Markdown or any other
text."""
_INITIAL_SCENE_GRAPH_SYSTEM_PROMPT = """You inspect an outlined tabletop-scene image.
Each visible asset has an outline and an ID label. Build a support graph for the
listed assets and determine each asset's image-observed orientation state.

Every asset must have exactly one direct support parent with relation "on".
Use "table" when the asset directly rests on the table. Use another supplied
asset ID only when the image clearly shows the asset resting directly on that
asset's top surface. Do not infer an on relationship from 2D overlap alone.
When the direct support parent is uncertain, use "table". The table is a fixed
support ID, not an output node: never include it in nodes.

Use a non-null orientation_state only for an elongated object with a clear
primary long axis. Use "standing" when that axis is approximately vertical to
the tabletop, and "lying" when it is approximately parallel to the tabletop.
Use null for every object without a clear primary long axis or when uncertain.
The orientation state describes the asset itself and is independent of its
support parent.

Examples:
- A bottle directly on the table is upright:
  {"nodes": [{"object_id": "bottle_001", "parent_id": "table", "parent_relation": "on", "orientation_state": "standing"}]}
- A pen lies flat on a book, and the book is on the table:
  {"nodes": [{"object_id": "book_001", "parent_id": "table", "parent_relation": "on", "orientation_state": null}, {"object_id": "pen_001", "parent_id": "book_001", "parent_relation": "on", "orientation_state": "lying"}]}
- A round cup directly on the table has no reliable long axis:
  {"nodes": [{"object_id": "cup_001", "parent_id": "table", "parent_relation": "on", "orientation_state": null}]}

Return JSON only, with exactly one key: nodes. Include every supplied asset ID
exactly once and no unknown IDs. Do not include Markdown or any other text."""


def understand_scene(
    scene: Scene,
    image_path: str | Path,
    output_root: str | Path,
    *,
    vlm_client: OpenAICompatibleVLM,
    image_segmentation_client: ImageSegmentationClient,
    json_max_attempts: int = 3,
) -> tuple[Scene, SceneGraph]:

    resolved_image_path = _validate_image_path(image_path)
    # The output in this stage will keep a JSON which contains
    # the Scene data structure for debugging.
    stage_output_root = Path(output_root).expanduser().resolve() / "scene_understanding"
    if stage_output_root.exists():
        shutil.rmtree(stage_output_root)
    stage_output_root.mkdir(parents=True, exist_ok=True)

    _analyze_image_objects(  # Update the scene data structure internally.
        scene=scene,
        image_path=resolved_image_path,
        vlm_client=vlm_client,
        json_max_attempts=json_max_attempts,
    )

    # Receive the validated whole-scene mask, for VLM output the scene graph.
    asset_mask_id_overlay_path = _segment_scene(
        image_path=resolved_image_path,
        stage_output_root=stage_output_root,
        scene=scene,
        vlm_client=vlm_client,
        image_segmentation_client=image_segmentation_client,
    )

    # Use the segmented image to initialize the scene graph
    # with the help of the VLM client.
    # But at here, we do with the simplest way (hard code).
    scene_graph = _initialize_scene_graph_from_segmented_scene(
        scene,
        asset_mask_id_overlay_path=asset_mask_id_overlay_path,
        vlm_client=vlm_client,
        json_max_attempts=json_max_attempts,
    )

    # Write the Updated scene JSON for debugging.
    (stage_output_root / "scene.json").write_text(
        json.dumps(scene.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (stage_output_root / "scene_graph.json").write_text(
        json.dumps(scene_graph.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return scene, scene_graph


def _initialize_scene_graph_from_segmented_scene(
    scene: Scene,
    *,
    asset_mask_id_overlay_path: str | Path,
    vlm_client: OpenAICompatibleVLM,
    json_max_attempts: int = 3,
) -> SceneGraph:
    """Build the initial image-observed support graph for segmented assets."""
    # Get simplified scene info for VLM.
    scene_info = _simplify_scene_info_for_graph_initialization(scene=scene)
    resolved_asset_mask_id_overlay_path = _validate_image_path(
        asset_mask_id_overlay_path
    )
    if scene.table is None:
        raise ValueError("Cannot initialize a scene graph without a table.")
    # Return a validated scene graph.
    return _query_initial_scene_graph(
        scene_info=scene_info,
        asset_mask_id_overlay_path=resolved_asset_mask_id_overlay_path,
        vlm_client=vlm_client,
        json_max_attempts=json_max_attempts,
    )


def _simplify_scene_info_for_graph_initialization(
    *,
    scene: Scene,
) -> dict[str, object]:
    """Return the object metadata needed to initialize an image-based graph."""
    return {
        "assets": [
            {
                "id": asset.id,
                "category": asset.category,
                "name": asset.name,
                "description": asset.description,
            }
            for asset in scene.assets
        ],
    }


def _query_initial_scene_graph(
    *,
    scene_info: dict[str, object],
    asset_mask_id_overlay_path: Path,
    vlm_client: OpenAICompatibleVLM,
    json_max_attempts: int,
) -> SceneGraph:
    """Return one validated image-observed support graph."""
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")
    last_validation_error: ValueError | None = None
    for _ in range(json_max_attempts):
        # Get response.
        response_text = vlm_client.complete(
            image_path=asset_mask_id_overlay_path,
            system_prompt=_INITIAL_SCENE_GRAPH_SYSTEM_PROMPT,
            user_prompt=json.dumps(scene_info, ensure_ascii=False),
        )
        try:
            # Validate.
            return _parse_initial_scene_graph_response(
                response_text=response_text,
                assets=scene_info["assets"],
            )
        except ValueError as exc:
            last_validation_error = exc
    assert last_validation_error is not None
    raise ValueError(
        "VLM returned invalid initial scene-graph JSON after "
        f"{json_max_attempts} attempts: {last_validation_error}"
    ) from last_validation_error


def _parse_initial_scene_graph_response(
    *,
    response_text: str,
    assets: object,
) -> SceneGraph:
    """Parse a complete VLM support graph response for known scene assets."""
    if not isinstance(assets, list) or not all(
        isinstance(asset, dict) and isinstance(asset.get("id"), str) for asset in assets
    ):
        raise ValueError("Scene graph initialization requires assets with string ids.")
    asset_ids = [asset["id"] for asset in assets]
    json_text = _strip_json_code_fence(response_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"nodes"}:
        raise ValueError("VLM JSON must contain exactly the key: nodes.")
    nodes_value = payload["nodes"]
    if not isinstance(nodes_value, list):
        raise ValueError("VLM JSON key nodes must be an array.")

    nodes_by_id: dict[str, SceneGraphNode] = {}
    for index, node_value in enumerate(nodes_value):
        if not isinstance(node_value, dict) or set(node_value) != {
            "object_id",
            "parent_id",
            "parent_relation",
            "orientation_state",
        }:
            raise ValueError(
                "VLM JSON nodes["
                f"{index}] must contain exactly object_id, parent_id, "
                "parent_relation, and orientation_state."
            )
        object_id = node_value["object_id"]
        parent_id = node_value["parent_id"]
        parent_relation = node_value["parent_relation"]
        orientation_state = node_value["orientation_state"]
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"VLM JSON nodes[{index}].object_id is invalid.")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError(f"VLM JSON nodes[{index}].parent_id is invalid.")
        if parent_relation != "on":
            raise ValueError(f"VLM JSON nodes[{index}].parent_relation is invalid.")
        if orientation_state not in {None, "standing", "lying"}:
            raise ValueError(f"VLM JSON nodes[{index}].orientation_state is invalid.")
        if object_id in nodes_by_id:
            raise ValueError(f"VLM JSON repeats scene graph node for {object_id!r}.")
        nodes_by_id[object_id] = SceneGraphNode(
            object_id=object_id,
            parent_id=parent_id,
            parent_relation=parent_relation,
            orientation_state=orientation_state,
        )

    if set(nodes_by_id) != set(asset_ids):
        raise ValueError(
            "VLM JSON scene graph nodes must match all supplied asset IDs."
        )
    return SceneGraph(
        nodes=[
            SceneGraphNode(object_id=TABLE_OBJECT_ID, parent_id=None),
            *(nodes_by_id[asset_id] for asset_id in asset_ids),
        ]
    )


def _analyze_image_objects(
    *,
    scene: Scene,
    image_path: str | Path,
    vlm_client: OpenAICompatibleVLM,
    json_max_attempts: int = 3,
) -> None:
    """Analyze one image and update ``scene`` with validated semantic objects."""
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")

    resolved_image_path = _validate_image_path(image_path)
    last_validation_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vlm_client.complete(
            image_path=resolved_image_path,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_USER_PROMPT,
        )
        try:
            analyzed_scene = _parse_image_object_analysis_response(response_text)
            validate_scene_understanding(analyzed_scene)
        except ValueError as exc:
            last_validation_error = exc
            continue

        scene.objects = analyzed_scene.objects
        return None

    assert last_validation_error is not None
    raise ValueError(
        "VLM returned invalid image-object analysis JSON after "
        f"{json_max_attempts} attempts: {last_validation_error}"
    ) from last_validation_error


def _parse_image_object_analysis_response(response_text: str) -> Scene:
    """Parse one VLM image-object analysis response into a semantic ``Scene``."""
    json_text = _strip_json_code_fence(response_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM response is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict) or set(payload) != {"table", "assets"}:
        raise ValueError("VLM JSON must contain exactly the keys: table and assets.")

    id_counters: dict[str, int] = {}
    table_fields = _parse_scene_object_fields(payload["table"], field_name="table")
    table = SceneObject(
        # id=_next_id(table_fields["category"], id_counters)
        # Use a fixed ID for the table.
        id="table",
        kind="table",
        **table_fields,
    )
    assets_value = payload["assets"]
    if not isinstance(assets_value, list):
        raise ValueError("VLM JSON key assets must be an array.")
    assets: list[SceneObject] = []
    for index, asset in enumerate(assets_value):
        fields = _parse_scene_object_fields(asset, field_name=f"assets[{index}]")
        assets.append(
            SceneObject(
                id=_next_id(fields["category"], id_counters),
                kind="asset",
                **fields,
            )
        )

    return Scene(objects=[table, *assets])


def validate_scene_understanding(scene: Scene) -> None:
    """Validate that scene understanding produced a complete semantic scene."""
    if scene.table is None:
        raise ValueError("Scene understanding must identify a table.")
    if (
        scene.table.id != "table"
    ):  # Currently it will always return false. For we hardcode the table id to "table".
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
    return fields


def _next_id(category: str, counters: dict[str, int]) -> str:
    """Auto increment an ID for the same category, e.g. mug_001, mug_002, etc."""
    counters[category] = counters.get(category, 0) + 1
    return f"{category}_{counters[category]:03d}"


def _segment_scene(
    *,
    image_path: str | Path,
    stage_output_root: str | Path,
    scene: Scene,
    vlm_client: OpenAICompatibleVLM,
    image_segmentation_client: ImageSegmentationClient,
) -> Path:
    """Add validated masks and return an asset-only ID overlay image."""
    debug_output_root = (
        Path(stage_output_root) / "debug"
    )  # Keeps the mask debug images.
    masks_output_root = (
        Path(stage_output_root) / "masks"
    )  # Keeps the validated masked images of each assets (include the table)
    debug_output_root.mkdir()
    masks_output_root.mkdir()

    # Segment the table and assets with VLM validation separately.
    _segment_assets(
        image_path=image_path,
        debug_output_root=debug_output_root,
        masks_output_root=masks_output_root,
        scene=scene,
        vlm_client=vlm_client,
        image_segmentation_client=image_segmentation_client,
    )
    # Prepare an image which do not contains any asset, for the VLM validation of the table
    # segmentation more easily.
    asset_mask_paths: list[str] = []
    for asset in scene.assets:
        if asset.mask_path is None:
            raise ValueError(f"Asset {asset.id!r} has no validated mask path.")
        asset_mask_paths.append(asset.mask_path)
    table_validation_image_path, asset_union_mask = render_image_without_masks(
        image_path=image_path,
        mask_paths=asset_mask_paths,
        output_path=Path(debug_output_root) / "table_validation_base.png",
    )
    # Segment the table.
    _segment_table(
        image_path=image_path,
        validation_image_path=table_validation_image_path,
        label_avoid_mask=asset_union_mask,
        debug_output_root=debug_output_root,
        masks_output_root=masks_output_root,
        scene=scene,
        vlm_client=vlm_client,
        image_segmentation_client=image_segmentation_client,
    )
    asset_masks: list[tuple[str, str]] = []
    for asset in scene.assets:
        if asset.mask_path is None:
            raise ValueError(f"Asset {asset.id!r} has no validated mask path.")
        asset_masks.append((asset.id, asset.mask_path))
    return render_asset_mask_id_overlay(
        image_path=image_path,
        asset_masks=asset_masks,
        output_path=Path(masks_output_root) / "asset_masks_with_ids.png",
    )


def _segment_table(
    image_path: str | Path,
    validation_image_path: str | Path,
    label_avoid_mask: Image.Image,
    debug_output_root: str | Path,
    masks_output_root: str | Path,
    scene: Scene,
    *,
    vlm_client: OpenAICompatibleVLM,
    image_segmentation_client: ImageSegmentationClient,
) -> None:
    """Segment the table. (Now it only supports segment the complete tabletop)"""
    if scene.table is None:
        raise ValueError("Cannot segment a scene without a table.")

    table = scene.table
    # Build the segmentation prompts for table.
    for prompt_label, prompt in (
        ("name", table.name),
        ("description", table.description),
        ("table", "table"),
        ("plane", "plane"),
    ):
        candidates = union_overlapping_mask_candidates(
            build_mask_candidates(
                image_segmentation_client.segment_single_object(
                    image_path=image_path,
                    prompt=prompt,
                )
            ),
            min_iou=0.8,  # Union masks who have iou > 0.8
        )
        # If do not have candidate, then try segment the table with description, "table", "plane"...
        # Notice that, this part could be extended with other segmentation prompt like
        # a board, or newly-generated prompt from another VLM-calling etc.
        if not candidates:
            continue

        # Maybe the mask count = 1, but not correct;
        # Maybe the mask count > 1;
        # Thus, we need to validate with an VLM.
        candidates_image_path = render_numbered_mask_candidates(
            image_path=validation_image_path,
            candidates=candidates,
            label_avoid_mask=label_avoid_mask,
            output_path=(
                Path(debug_output_root)
                / f"table_candidates_{prompt_label}.png"  # Render with prompt label, for easily debug.
            ),
        )
        selected_mask_index = _validate_table_candidates_with_vlm(
            table=table,
            candidates=candidates,
            candidates_image_path=candidates_image_path,
            vlm_client=vlm_client,
        )
        if selected_mask_index is None:
            continue

        # Save result.
        candidate = _candidate_by_index(candidates, selected_mask_index)
        table.mask_path = str(
            save_binary_mask(
                candidate,
                image_size=_image_size(image_path),
                output_path=Path(masks_output_root) / "table_mask.png",
            )
        )
        return

    raise ValueError("Unable to find a VLM-validated segmentation mask for the table.")


def _validate_table_candidates_with_vlm(
    *,
    table: SceneObject,
    candidates: list[MaskCandidate],
    candidates_image_path: Path,
    vlm_client: OpenAICompatibleVLM,
    json_max_attempts: int = 3,
) -> int | None:

    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")

    user_prompt = (
        "Table category: "
        f"{table.category}\n"
        f"Table name: {table.name}\n"
        f"Table description: {table.description}\n"
        f"Candidate indices range from 1 to {len(candidates)}."
    )
    last_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vlm_client.complete(
            image_path=candidates_image_path,
            system_prompt=_TABLE_VALIDATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            return _parse_table_validation_response(response_text, candidates)
        except ValueError as exc:
            last_error = exc

    assert last_error is not None
    raise ValueError(
        "VLM returned invalid table-segmentation validation JSON after "
        f"{json_max_attempts} attempts: {last_error}"
    ) from last_error


def _parse_table_validation_response(
    response_text: str,
    candidates: list[MaskCandidate],
) -> int | None:
    """Validate the strict VLM response schema for table candidate selection."""
    try:
        payload = json.loads(_strip_json_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("VLM table validation response is not valid JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"selected_mask_index"}:
        raise ValueError(
            "VLM table validation JSON must contain only selected_mask_index."
        )

    selected_mask_index = payload["selected_mask_index"]
    if selected_mask_index is None:
        return None
    if isinstance(selected_mask_index, bool) or not isinstance(
        selected_mask_index, int
    ):
        raise ValueError("selected_mask_index must be an integer or null.")
    _candidate_by_index(candidates, selected_mask_index)
    return selected_mask_index


def _candidate_by_index(
    candidates: list[MaskCandidate],
    index: int,
) -> MaskCandidate:
    for candidate in candidates:
        if candidate.index == index:
            return candidate
    raise ValueError(f"VLM selected a nonexistent mask candidate: {index}.")


def _image_size(image_path: str | Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _segment_assets(
    image_path: str | Path,
    debug_output_root: str | Path,
    masks_output_root: str | Path,
    scene: Scene,
    *,
    vlm_client: OpenAICompatibleVLM,
    image_segmentation_client: ImageSegmentationClient,
) -> None:

    # Group the assets by their categories.
    assets_by_category: dict[str, list[SceneObject]] = {}
    for asset in scene.assets:
        assets_by_category.setdefault(asset.category, []).append(asset)

    image_size = _image_size(image_path)
    for category, assets in assets_by_category.items():
        mask_rles: list[dict[str, Any]] = []
        # Use categories and names as segmentation prompt.
        # Use category to segment first, then use each assets' name to segment.
        prompts = [category, *dict.fromkeys(asset.name for asset in assets)]
        for prompt in prompts:
            mask_rles.extend(
                image_segmentation_client.segment_single_object(
                    image_path=image_path,
                    prompt=prompt,
                )
            )
        # Union duplicated mask candidates.
        candidates = union_overlapping_mask_candidates(
            build_mask_candidates(mask_rles),
            min_iou=0.8,
        )
        # If the number of candidate is less than the grouped assets,
        # raise error directly.
        if len(candidates) < len(assets):
            raise ValueError(
                f"Asset category {category!r} has {len(assets)} assets but only "
                f"{len(candidates)} segmentation candidates."
            )

        candidates_image_path = render_numbered_mask_candidates(
            image_path=image_path,
            candidates=candidates,
            output_path=Path(debug_output_root) / f"asset_candidates_{category}.png",
            mask_style="outline",
        )
        assignments = _validate_asset_candidates_with_vlm(
            assets=assets,
            candidates=candidates,
            candidates_image_path=candidates_image_path,
            vlm_client=vlm_client,
        )
        if assignments is None:
            raise ValueError(
                f"VLM could not assign every {category!r} asset to a segmentation candidate."
            )
        # Save results.
        for asset in assets:
            asset.mask_path = str(
                save_binary_mask(
                    _candidate_by_index(candidates, assignments[asset.id]),
                    image_size=image_size,
                    output_path=Path(masks_output_root) / f"{asset.id}_mask.png",
                )
            )


def _validate_asset_candidates_with_vlm(
    *,
    assets: list[SceneObject],
    candidates: list[MaskCandidate],
    candidates_image_path: Path,
    vlm_client: OpenAICompatibleVLM,
    json_max_attempts: int = 3,
) -> dict[str, int] | None:
    """Ask the VLM for a complete one-to-one asset-to-candidate assignment."""
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")

    assets_text = "\n".join(
        "- "
        f"id: {asset.id}; category: {asset.category}; name: {asset.name}; "
        f"description: {asset.description}"
        for asset in assets
    )
    user_prompt = (
        "Asset group:\n"
        f"{assets_text}\n\n"
        f"Candidate indices range from 1 to {len(candidates)}."
    )
    last_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vlm_client.complete(
            image_path=candidates_image_path,
            system_prompt=_ASSET_ASSIGNMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            return _parse_asset_assignment_response(response_text, assets, candidates)
        except ValueError as exc:
            last_error = exc

    assert last_error is not None
    raise ValueError(
        "VLM returned invalid asset-segmentation assignment JSON after "
        f"{json_max_attempts} attempts: {last_error}"
    ) from last_error


def _parse_asset_assignment_response(
    response_text: str,
    assets: list[SceneObject],
    candidates: list[MaskCandidate],
) -> dict[str, int] | None:
    """Parse a strict complete assignment, or a valid missing-asset result."""
    try:
        payload = json.loads(_strip_json_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("VLM asset assignment response is not valid JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"assignments"}:
        raise ValueError("VLM asset assignment JSON must contain only assignments.")

    assignment_values = payload["assignments"]
    if assignment_values is None:
        return None
    if not isinstance(assignment_values, list):
        raise ValueError("assignments must be an array or null.")

    expected_asset_ids = {asset.id for asset in assets}
    assignments: dict[str, int] = {}
    assigned_mask_indices: set[int] = set()
    for assignment in assignment_values:
        if not isinstance(assignment, dict) or set(assignment) != {
            "asset_id",
            "mask_index",
        }:
            raise ValueError(
                "Each assignment must contain only asset_id and mask_index."
            )
        asset_id = assignment["asset_id"]
        mask_index = assignment["mask_index"]
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("assignment asset_id must be a non-empty string.")
        if isinstance(mask_index, bool) or not isinstance(mask_index, int):
            raise ValueError("assignment mask_index must be an integer.")
        if asset_id in assignments:
            raise ValueError(f"VLM assigned asset {asset_id!r} more than once.")
        if mask_index in assigned_mask_indices:
            raise ValueError(
                f"VLM assigned candidate {mask_index} to more than one asset."
            )
        _candidate_by_index(candidates, mask_index)
        assignments[asset_id] = mask_index
        assigned_mask_indices.add(mask_index)

    if set(assignments) != expected_asset_ids:
        raise ValueError("VLM assignments must cover every asset in the group.")
    return assignments
