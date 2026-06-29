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
    "build_image_metric_scale_messages",
    "build_text_metric_scale_messages",
    "build_up_down_flip_check_messages",
]

UNIFIED_SCENE_GEN_PROMPT_NAME = "unified_scene_gen.yaml"


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
                UNIFIED_SCENE_GEN_PROMPT_NAME,
                prompt_key="image_metric_scale_system",
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        UNIFIED_SCENE_GEN_PROMPT_NAME,
                        {
                            "objects_json": json.dumps(
                                objects_json,
                                ensure_ascii=False,
                                indent=2,
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
                UNIFIED_SCENE_GEN_PROMPT_NAME,
                prompt_key="text_metric_scale_system",
            ),
        },
        {
            "role": "user",
            "content": render_prompt(
                UNIFIED_SCENE_GEN_PROMPT_NAME,
                {
                    "user_text": user_text,
                    "objects_json": json.dumps(
                        objects_json,
                        ensure_ascii=False,
                        indent=2,
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
                UNIFIED_SCENE_GEN_PROMPT_NAME,
                prompt_key="up_down_flip_check_system",
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        UNIFIED_SCENE_GEN_PROMPT_NAME,
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
