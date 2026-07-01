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
from typing import Any

from embodichain.gen_sim.prompt2scene.prompts import render_prompt
from embodichain.gen_sim.prompt2scene.utils.io import image_to_data_url

__all__ = [
    "build_filter_extra_instances_messages",
    "build_image_metric_scale_messages",
    "build_scene_intake_messages",
    "build_scene_intake_verifier_messages",
    "build_spatial_layout_messages",
    "build_text_metric_scale_messages",
    "build_text_relation_messages",
    "build_up_down_flip_check_messages",
]


SCENE_INTAKE_PROMPT = "scene_intake.yaml"
IMAGE_RELATIONS_PROMPT = "image_relations.yaml"
TEXT_RELATIONS_PROMPT = "text_relations.yaml"
UNIFIED_SCENE_GEN_PROMPT = "unified_scene_gen.yaml"



def build_scene_intake_messages(request: Prompt2SceneInput) -> list[dict[str, Any]]:
    """Build LangChain-compatible messages for scene intake."""

    from embodichain.gen_sim.prompt2scene.workflows.request import InputKind

    if request.input_kind == InputKind.TEXT:
        return [
            {
                "role": "system",
                "content": render_prompt(
                    SCENE_INTAKE_PROMPT, prompt_key="text_system"
                ),
            },
            {
                "role": "user",
                "content": render_prompt(
                    SCENE_INTAKE_PROMPT,
                    {"text": request.text or ""},
                    prompt_key="text_user",
                ),
            },
        ]
    return [
        {
            "role": "system",
            "content": render_prompt(
                SCENE_INTAKE_PROMPT, prompt_key="image_system"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        SCENE_INTAKE_PROMPT, prompt_key="image_user"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(request.image_path)},
                },
            ],
        },
    ]


def build_scene_intake_verifier_messages(
    *,
    request: Prompt2SceneInput,
    scene_intake: SceneIntakeSpec,
) -> list[dict[str, Any]]:
    """Build messages for scene-intake group and count verification."""

    from embodichain.gen_sim.prompt2scene.workflows.request import InputKind

    table_draft: dict[str, object] = {
        "name": scene_intake.table.name,
        "description": scene_intake.table.description,
        "complete_table_description": (
            scene_intake.table.complete_table_description
        ),
        "is_complete_visible_table": scene_intake.table.is_complete_visible_table,
        "class_candidate": list(scene_intake.table.class_candidate),
    }
    if scene_intake.table.object_coverage_percent is not None:
        table_draft["object_coverage_percent"] = (
            scene_intake.table.object_coverage_percent
        )
    scene_intake_json = json.dumps(
        {
            "table": table_draft,
            "assets": [
                {
                    "name": asset.name,
                    "description": asset.description,
                    "class_candidate": list(asset.class_candidate),
                    "count": asset.count,
                }
                for asset in scene_intake.assets
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

    if request.input_kind == InputKind.TEXT:
        return [
            {
                "role": "system",
                "content": render_prompt(
                    SCENE_INTAKE_PROMPT, prompt_key="verifier_system"
                ),
            },
            {
                "role": "user",
                "content": render_prompt(
                    SCENE_INTAKE_PROMPT,
                    {
                        "text": request.text or "",
                        "scene_intake_json": scene_intake_json,
                    },
                    prompt_key="verifier_text_user",
                ),
            },
        ]

    image_path = request.image_path
    if image_path is None:
        raise ValueError("Image input requires image_path.")
    return [
        {
            "role": "system",
            "content": render_prompt(
                SCENE_INTAKE_PROMPT, prompt_key="verifier_system"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        SCENE_INTAKE_PROMPT,
                        {"scene_intake_json": scene_intake_json},
                        prompt_key="verifier_image_user",
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image_path)},
                },
            ],
        },
    ]




def build_filter_extra_instances_messages(
    *,
    debug_image_path: Path,
    name: str,
    description: str,
    expected_count: int,
    class_candidate: list[str],
) -> list[dict[str, Any]]:
    """Build LangChain-compatible messages for VLM extra-mask filtering."""
    return [
        {
            "role": "system",
            "content": render_prompt(
                IMAGE_RELATIONS_PROMPT, prompt_key="filter_extra_instances_system"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        IMAGE_RELATIONS_PROMPT,
                        {
                            "name": name.replace("_", " "),
                            "description": description,
                            "expected_count": str(expected_count),
                            "class_candidate": ", ".join(
                                candidate.replace("_", " ")
                                for candidate in class_candidate
                            ),
                        },
                        prompt_key="filter_extra_instances_user",
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(debug_image_path)},
                },
            ],
        },
    ]


def build_spatial_layout_messages(
    *,
    bbox_name_image_path: Path,
    asset_ids: list[str],
) -> list[dict[str, Any]]:
    """Build messages for VLM spatial ordering and object-state extraction."""
    return [
        {
            "role": "system",
            "content": render_prompt(
                IMAGE_RELATIONS_PROMPT, prompt_key="spatial_layout_system"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        IMAGE_RELATIONS_PROMPT,
                        {
                            "asset_ids": "\n".join(
                                f"- {asset_id}" for asset_id in asset_ids
                            ),
                        },
                        prompt_key="spatial_layout_user",
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(bbox_name_image_path)},
                },
            ],
        },
    ]




def build_text_relation_messages(
    *,
    request: Prompt2SceneInput,
    scene_intake: SceneIntakeSpec,
) -> list[dict[str, Any]]:
    """Build messages for explicit text spatial-relation extraction."""
    asset_names = "\n".join(f"- {asset.name}" for asset in scene_intake.assets)
    return [
        {
            "role": "system",
            "content": render_prompt(TEXT_RELATIONS_PROMPT, prompt_key="system"),
        },
        {
            "role": "user",
            "content": render_prompt(
                TEXT_RELATIONS_PROMPT,
                {
                    "asset_names": asset_names,
                    "text": request.text or "",
                },
                prompt_key="user",
            ),
        },
    ]




def build_image_metric_scale_messages(
    *,
    bbox_name_image_path: Path,
    objects_json: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build messages for image-scene object metric scale estimation."""
    return [
        {
            "role": "system",
            "content": render_prompt(
                UNIFIED_SCENE_GEN_PROMPT, prompt_key="image_metric_scale_system"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        UNIFIED_SCENE_GEN_PROMPT,
                        {
                            "objects_json": json.dumps(
                                objects_json, ensure_ascii=False, indent=2
                            ),
                        },
                        prompt_key="image_metric_scale_user",
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(bbox_name_image_path)},
                },
            ],
        },
    ]


def build_text_metric_scale_messages(
    *,
    user_text: str,
    objects_json: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build messages for text-scene object metric scale estimation."""
    return [
        {
            "role": "system",
            "content": render_prompt(
                UNIFIED_SCENE_GEN_PROMPT, prompt_key="text_metric_scale_system"
            ),
        },
        {
            "role": "user",
            "content": render_prompt(
                UNIFIED_SCENE_GEN_PROMPT,
                {
                    "user_text": user_text,
                    "objects_json": json.dumps(
                        objects_json, ensure_ascii=False, indent=2
                    ),
                },
                prompt_key="text_metric_scale_user",
            ),
        },
    ]


def build_up_down_flip_check_messages(
    *,
    original_image_path: Path,
    comparison_image_path: Path,
) -> list[dict[str, Any]]:
    """Build messages for VLM support-normal up/down flip verification."""
    return [
        {
            "role": "system",
            "content": render_prompt(
                UNIFIED_SCENE_GEN_PROMPT, prompt_key="up_down_flip_check_system"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        UNIFIED_SCENE_GEN_PROMPT,
                        prompt_key="up_down_flip_check_user",
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(original_image_path)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(comparison_image_path)},
                },
            ],
        },
    ]
