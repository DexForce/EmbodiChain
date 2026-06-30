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
from typing import Any

from embodichain.gen_sim.prompt2scene.prompts import render_prompt
from embodichain.gen_sim.prompt2scene.workflows.request import (
    InputKind,
    Prompt2SceneInput,
)
from embodichain.gen_sim.prompt2scene.utils.io import image_to_data_url
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeSpec,
)

__all__ = [
    "build_scene_intake_messages",
    "build_scene_intake_verifier_messages",
]

SCENE_INTAKE_PROMPT_NAME = "scene_intake.yaml"


def build_scene_intake_messages(request: Prompt2SceneInput) -> list[dict[str, Any]]:
    """Build LangChain-compatible messages for scene intake."""
    if request.input_kind == InputKind.TEXT:
        return _build_text_messages(request)
    return _build_image_messages(request)


def build_scene_intake_verifier_messages(
    *,
    request: Prompt2SceneInput,
    scene_intake: SceneIntakeSpec,
) -> list[dict[str, Any]]:
    """Build messages for scene-intake group and count verification."""
    table_draft: dict[str, object] = {
        "name": scene_intake.table.name,
        "description": scene_intake.table.description,
        "complete_table_description": (
            scene_intake.table.complete_table_description
        ),
        "is_complete_visible_table": (
            scene_intake.table.is_complete_visible_table
        ),
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
        return _build_text_verifier_messages(
            request=request,
            scene_intake_json=scene_intake_json,
        )
    return _build_image_verifier_messages(
        request=request,
        scene_intake_json=scene_intake_json,
    )


def _build_text_messages(request: Prompt2SceneInput) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": render_prompt(SCENE_INTAKE_PROMPT_NAME, prompt_key="text_system"),
        },
        {
            "role": "user",
            "content": render_prompt(
                SCENE_INTAKE_PROMPT_NAME,
                {"text": request.text or ""},
                prompt_key="text_user",
            ),
        },
    ]


def _build_image_messages(request: Prompt2SceneInput) -> list[dict[str, Any]]:
    image_path = request.image_path
    if image_path is None:
        raise ValueError("Image input requires image_path.")

    return [
        {
            "role": "system",
            "content": render_prompt(SCENE_INTAKE_PROMPT_NAME, prompt_key="image_system"),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        SCENE_INTAKE_PROMPT_NAME,
                        prompt_key="image_user",
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image_path)},
                },
            ],
        },
    ]


def _build_text_verifier_messages(
    *,
    request: Prompt2SceneInput,
    scene_intake_json: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": render_prompt(
                SCENE_INTAKE_PROMPT_NAME,
                prompt_key="verifier_system",
            ),
        },
        {
            "role": "user",
            "content": render_prompt(
                SCENE_INTAKE_PROMPT_NAME,
                {
                    "text": request.text or "",
                    "scene_intake_json": scene_intake_json,
                },
                prompt_key="verifier_text_user",
            ),
        },
    ]


def _build_image_verifier_messages(
    *,
    request: Prompt2SceneInput,
    scene_intake_json: str,
) -> list[dict[str, Any]]:
    image_path = request.image_path
    if image_path is None:
        raise ValueError("Image input requires image_path.")

    return [
        {
            "role": "system",
            "content": render_prompt(
                SCENE_INTAKE_PROMPT_NAME,
                prompt_key="verifier_system",
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        SCENE_INTAKE_PROMPT_NAME,
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
