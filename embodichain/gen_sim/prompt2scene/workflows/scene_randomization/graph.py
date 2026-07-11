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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager import (
    LayoutManager,
)
from embodichain.gen_sim.prompt2scene.llms import build_chat_model
from embodichain.gen_sim.prompt2scene.llms.llm_output import (
    call_structured_json_model_step,
)
from embodichain.gen_sim.prompt2scene.prompts.builders import (
    build_scene_randomization_intent_messages,
)
from embodichain.gen_sim.prompt2scene.prompts.schemas import (
    SCENE_RANDOMIZATION_INTENT_JSON_SCHEMA,
)
from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    WorkflowArtifactWriter,
)
from embodichain.gen_sim.prompt2scene.workflows.paths import SCENE_RANDOMIZATION_STEP
from embodichain.gen_sim.prompt2scene.workflows.scene_edit.utils import (
    export_scene_edit_gym_state,
    extract_scene_edit_support_region,
    extract_scene_object_footprints,
    extract_scene_objects,
    load_json_object,
    scene_state_path,
)

if TYPE_CHECKING:
    from embodichain.gen_sim.prompt2scene.llms import OpenAICompatibleLLMCfg

_DIRECTION_VECTORS: dict[str, np.ndarray] = {
    "left": np.array([-1.0, 0.0], dtype=np.float64),
    "right": np.array([1.0, 0.0], dtype=np.float64),
    "front": np.array([0.0, -1.0], dtype=np.float64),
    "back": np.array([0.0, 1.0], dtype=np.float64),
}


def run_scene_randomization(
    *,
    output_root: Path,
    prompt: str,
    llm_cfg: OpenAICompatibleLLMCfg,
    route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Randomize existing object positions with directional 2D AABB moves."""
    output_root = output_root.expanduser().resolve()
    state_path = scene_state_path(output_root)
    if not state_path.is_file():
        raise FileNotFoundError(f"Scene state not found: {state_path}")
    scene_state = load_json_object(state_path)

    output_dir = output_root / SCENE_RANDOMIZATION_STEP
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = WorkflowArtifactWriter(output_root, SCENE_RANDOMIZATION_STEP)
    intent = _analyze_randomization_intent(
        prompt=prompt,
        scene_state=scene_state,
        llm_cfg=llm_cfg,
        writer=writer,
    )
    layout_result = _build_randomized_layout(scene_state=scene_state, intent=intent)
    export_result = export_scene_edit_gym_state(
        output_root=output_root,
        scene_state=scene_state,
        generated_assets=[],
        layout_updates=layout_result["layout_updates"],
        output_dir=output_dir,
    )
    result = {
        "status": "ok" if export_result.get("status") == "ok" else "partial",
        "prompt": prompt,
        "scene_state_path": str(state_path),
        "route": route or {},
        "resolved_intent": intent,
        "layout_randomization": layout_result,
        "file_updates": export_result,
        "reason": "Scene randomization intent, layout update, and file export completed.",
    }
    writer.write_step_result(result)
    return result


def _analyze_randomization_intent(
    *,
    prompt: str,
    scene_state: dict[str, Any],
    llm_cfg: OpenAICompatibleLLMCfg,
    writer: WorkflowArtifactWriter,
) -> dict[str, Any]:
    scene_objects = extract_scene_objects(scene_state)
    llm = build_chat_model(llm_cfg)
    round_name = writer.next_debug_round_name("intent")
    intent = call_structured_json_model_step(
        llm=llm,
        schema=SCENE_RANDOMIZATION_INTENT_JSON_SCHEMA,
        messages=build_scene_randomization_intent_messages(
            prompt=prompt,
            scene_objects=scene_objects,
        ),
        context="scene_randomization_intent",
        attempt_count=1,
        raw_output_writer=lambda payload: writer.write_raw_model_output(
            round_name=round_name,
            payload=payload,
        ),
    )
    _validate_randomization_intent(intent=intent, scene_objects=scene_objects)
    return intent


def _validate_randomization_intent(
    *,
    intent: dict[str, Any],
    scene_objects: list[dict[str, str]],
) -> None:
    known_ids = {str(item.get("id", "")).strip() for item in scene_objects}
    for op in intent.get("operations", []) or []:
        object_id = str(op.get("target_object_id", "")).strip()
        direction = str(op.get("direction", "")).strip()
        if object_id not in known_ids:
            raise ValueError(f"Unknown randomization target object: {object_id}")
        if direction not in _DIRECTION_VECTORS:
            raise ValueError(f"Unsupported randomization direction: {direction}")


def _build_randomized_layout(
    *,
    scene_state: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    support_region = extract_scene_edit_support_region(scene_state)
    support_min, support_max = _support_aabb(support_region)
    support_size = np.maximum(support_max - support_min, 1.0e-6)
    short_side = float(np.min(support_size))
    move_step = float(np.clip(short_side * 0.12, 0.02, 0.15))
    push_step = float(np.clip(short_side * 0.03, 0.01, 0.05))

    footprints = extract_scene_object_footprints(scene_state)
    scene_by_id = {
        str(obj.get("id", "")).strip(): obj
        for obj in scene_state.get("objects", []) or []
        if isinstance(obj, dict) and str(obj.get("id", "")).strip()
    }
    items: dict[str, dict[str, Any]] = {}
    for object_id, obj in scene_by_id.items():
        footprint = footprints.get(object_id)
        if footprint is None:
            continue
        center = np.asarray(footprint["center_xy"], dtype=np.float64)
        size = np.asarray(footprint["size_xy"], dtype=np.float64)
        items[object_id] = {
            "id": object_id,
            "name": str(obj.get("name", "")).strip(),
            "description": str(obj.get("description", "")).strip(),
            "action": "keep",
            "center_xy": center.tolist(),
            "size_xy": size.tolist(),
            "footprint_2d": footprint,
            "source": "previous_scene",
        }

    moved_dirs: dict[str, np.ndarray] = {}
    applied_operations: list[dict[str, Any]] = []
    for op in intent.get("operations", []) or []:
        object_id = str(op.get("target_object_id", "")).strip()
        direction = str(op.get("direction", "")).strip()
        if object_id not in items or direction not in _DIRECTION_VECTORS:
            continue
        vec = _DIRECTION_VECTORS[direction]
        moved_dirs[object_id] = vec
        center = np.asarray(items[object_id]["center_xy"], dtype=np.float64)
        size = np.asarray(items[object_id]["size_xy"], dtype=np.float64)
        center = _clamp_center(center + vec * move_step, size, support_min, support_max)
        items[object_id]["center_xy"] = center.tolist()
        applied_operations.append({"target_object_id": object_id, "direction": direction})

    overlap_history = _resolve_overlaps(
        items=items,
        moved_dirs=moved_dirs,
        support_min=support_min,
        support_max=support_max,
        push_step=push_step,
    )
    final_overlaps = _overlap_pairs(items)
    for item in items.values():
        item["footprint_2d"] = LayoutManager.build_xy_footprint(
            center_xy=item["center_xy"],
            size_xy=item["size_xy"],
        )

    return {
        "status": "ok" if not final_overlaps else "partial",
        "support_region": support_region,
        "layout_updates": sorted(items.values(), key=lambda item: item["id"]),
        "optimization": {
            "method": "directional_2d_aabb_randomization",
            "move_step_m": move_step,
            "push_step_m": push_step,
            "max_overlap_push_rounds": 30,
            "applied_operations": applied_operations,
            "overlap_resolution_rounds": len(overlap_history),
            "remaining_overlaps": final_overlaps,
        },
    }


def _resolve_overlaps(
    *,
    items: dict[str, dict[str, Any]],
    moved_dirs: dict[str, np.ndarray],
    support_min: np.ndarray,
    support_max: np.ndarray,
    push_step: float,
    max_rounds: int = 30,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for round_idx in range(max_rounds):
        overlaps = _overlap_pairs(items)
        actionable = [
            pair for pair in overlaps
            if pair["a"] in moved_dirs or pair["b"] in moved_dirs
        ]
        if not actionable:
            break
        changed = False
        for pair in actionable:
            for object_id in (pair["a"], pair["b"]):
                vec = moved_dirs.get(object_id)
                if vec is None:
                    continue
                size = np.asarray(items[object_id]["size_xy"], dtype=np.float64)
                old = np.asarray(items[object_id]["center_xy"], dtype=np.float64)
                new = _clamp_center(old + vec * push_step, size, support_min, support_max)
                if not np.allclose(new, old):
                    items[object_id]["center_xy"] = new.tolist()
                    changed = True
        history.append({"round": round_idx + 1, "overlap_count": len(overlaps)})
        if not changed:
            break
    return history


def _overlap_pairs(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ids = sorted(items)
    overlaps: list[dict[str, Any]] = []
    for index, a in enumerate(ids):
        for b in ids[index + 1:]:
            a_min, a_max = _item_aabb(items[a])
            b_min, b_max = _item_aabb(items[b])
            overlap = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
            if float(overlap[0]) > 0.0 and float(overlap[1]) > 0.0:
                overlaps.append(
                    {
                        "a": a,
                        "b": b,
                        "overlap_xy": [float(overlap[0]), float(overlap[1])],
                    }
                )
    return overlaps


def _item_aabb(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(item["center_xy"], dtype=np.float64)
    size = np.asarray(item["size_xy"], dtype=np.float64)
    half = 0.5 * np.maximum(size, 0.0)
    return center - half, center + half


def _clamp_center(
    center: np.ndarray,
    size: np.ndarray,
    support_min: np.ndarray,
    support_max: np.ndarray,
) -> np.ndarray:
    half = 0.5 * np.maximum(size, 0.0)
    lo = support_min + half
    hi = support_max - half
    if np.any(hi < lo):
        return 0.5 * (support_min + support_max)
    return np.minimum(np.maximum(center, lo), hi)


def _support_aabb(support_region: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    aabb = support_region.get("aabb_xy")
    if not isinstance(aabb, list) or len(aabb) != 2:
        raise ValueError("Scene randomization requires support_region.aabb_xy.")
    support_min = np.asarray(aabb[0], dtype=np.float64)
    support_max = np.asarray(aabb[1], dtype=np.float64)
    if support_min.shape != (2,) or support_max.shape != (2,):
        raise ValueError("support_region.aabb_xy must contain two XY points.")
    return np.minimum(support_min, support_max), np.maximum(support_min, support_max)
