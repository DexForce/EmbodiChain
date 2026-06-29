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

from typing import Any

from embodichain.gen_sim.prompt2scene.prompts import render_prompt
from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeSpec,
)

__all__ = ["build_text_relation_messages"]

TEXT_RELATIONS_PROMPT_NAME = "text_relations.yaml"


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
            "content": render_prompt(TEXT_RELATIONS_PROMPT_NAME, prompt_key="system"),
        },
        {
            "role": "user",
            "content": render_prompt(
                TEXT_RELATIONS_PROMPT_NAME,
                {
                    "asset_names": asset_names,
                    "text": request.text or "",
                },
                prompt_key="user",
            ),
        },
    ]
