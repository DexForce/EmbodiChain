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

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
    TABLE_OBJECT_ID,
)
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)


def understand_scene_edit(
    *,
    scene: Scene,
    edit_prompt: str,
    output_root: str | Path,
    vlm_client: OpenAICompatibleVLM,
) -> dict[str, object]:
    """Understand one text edit instruction for an existing scene."""
    edit_prompt = edit_prompt.strip()
    if not edit_prompt:
        raise ValueError("Edit prompt must not be empty.")

    return {
        "edit_prompt": edit_prompt,
        "operations": [],
    }
