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
import shutil
from typing import Any

from embodichain.gen_sim.scene_engine.core.asset import Asset
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.table import Table
from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
)
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_segmentation_utils import (
    MaskCandidate,
    build_mask_candidates,
    render_image_without_masks,
    render_numbered_mask_candidates,
    save_binary_mask,
    union_overlapping_mask_candidates,
)

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
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


def segment_scene(
    image_path: str | Path,
    output_root: str | Path,
    scene: Scene,
    *,
    vlm_client: OpenAICompatibleVLM,
    image_segmentation_client: ImageSegmentationClient,
) -> Scene:

    resolved_image_path = _validate_image_path(image_path)
    # The output in this stage will keep a JSON which contains
    # the Scene data structure for debugging.
    stage_output_root = Path(output_root).expanduser().resolve() / "scene_segmentation"
    if stage_output_root.exists():
        shutil.rmtree(stage_output_root)
    stage_output_root.mkdir(parents=True, exist_ok=True)
    debug_output_root = stage_output_root / "debug"  # Keeps the mask debug images.
    masks_output_root = (
        stage_output_root / "masks"
    )  # Keeps the validated masked images of each assets (include the table)
    debug_output_root.mkdir()
    masks_output_root.mkdir()

    # Segment the table and assets with VLM validation separately.
    _segment_assets(
        image_path=resolved_image_path,
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
    table_validation_image_path = render_image_without_masks(
        image_path=resolved_image_path,
        mask_paths=asset_mask_paths,
        output_path=Path(debug_output_root) / "table_validation_base.png",
    )
    # Segment the table.
    _segment_table(
        image_path=resolved_image_path,
        validation_image_path=table_validation_image_path,
        debug_output_root=debug_output_root,
        masks_output_root=masks_output_root,
        scene=scene,
        vlm_client=vlm_client,
        image_segmentation_client=image_segmentation_client,
    )
    # Write the Updated scene JSON for debugging.
    (stage_output_root / "scene.json").write_text(
        json.dumps(scene.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return scene


def _segment_table(
    image_path: str | Path,
    validation_image_path: str | Path,
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
    table: Table,
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


def _strip_json_code_fence(response_text: str) -> str:
    stripped = response_text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("VLM table validation response has an incomplete code fence.")
    return "\n".join(lines[1:-1]).strip()


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
    assets_by_category: dict[str, list[Asset]] = {}
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
    assets: list[Asset],
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
    assets: list[Asset],
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


def _validate_image_path(image_path: str | Path) -> Path:
    resolved_image_path = Path(image_path).expanduser().resolve()
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"Image input not found: {resolved_image_path}")
    if resolved_image_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("Image input must be a .jpg, .jpeg, or .png file.")
    return resolved_image_path
