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
from typing import Any

from embodichain.gen_sim.prompt2scene.prompts import render_prompt
from embodichain.gen_sim.prompt2scene.utils.io import image_to_data_url

__all__ = [
    "build_filter_extra_instances_messages",
    "build_spatial_layout_messages",
]

IMAGE_RELATIONS_PROMPT_NAME = "image_relations.yaml"


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
                IMAGE_RELATIONS_PROMPT_NAME,
                prompt_key="filter_extra_instances_system",
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        IMAGE_RELATIONS_PROMPT_NAME,
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
                IMAGE_RELATIONS_PROMPT_NAME,
                prompt_key="spatial_layout_system",
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": render_prompt(
                        IMAGE_RELATIONS_PROMPT_NAME,
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
