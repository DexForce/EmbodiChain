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
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.scene_understanding import (
    understand_scene,
)


def generate_scene_from_image(
    image_path: str | Path,
    output_root: str | Path,
    *,
    llm_config_path: str | Path | None = None,
) -> Scene:
    """Generate the initial core scene state from an input image."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    # Initialize the VLM client.
    vlm_client = OpenAICompatibleVLM.from_config(llm_config_path)

    # 1. Scene Understanding
    scene = understand_scene(
        image_path=image_path,
        output_root=resolved_output_root,
        vlm_client=vlm_client,
    )

    # 2. Segmentation
    # 3. Objects + Coarse Layout Generation
    # 4. Geometry + Layout Refinement
    # 5. Scene Export

    return scene
