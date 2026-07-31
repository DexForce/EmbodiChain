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
from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
)

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)

from embodichain.gen_sim.scene_engine.pipeline.scene_understanding import (
    understand_scene,
)
from embodichain.gen_sim.scene_engine.pipeline.scene_segmentation import (
    segment_scene,
)
from embodichain.utils.logger import log_info

from embodichain.gen_sim.scene_engine.pipeline.scene_generation import (
    generate_scene_and_refine,
)
from embodichain.gen_sim.scene_engine.pipeline.scene_export import export_scene


def generate_scene_from_image(
    image_path: str | Path,
    output_root: str | Path,
    *,
    llm_config_path: str | Path | None = None,
    image_segmentation_config_path: str | Path | None = None,
    geometry_generation_config_path: str | Path | None = None,
) -> Scene:
    """Generate the initial core scene state from an input image."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    # Initialize the VLM client and the Scene data structure.
    vlm_client = OpenAICompatibleVLM.from_config(llm_config_path)
    scene = Scene()

    # 1. Scene Understanding
    log_info("Starting Scene Understanding")
    scene = understand_scene(
        scene=scene,
        image_path=image_path,
        output_root=resolved_output_root,
        vlm_client=vlm_client,
    )
    log_info("Completed Scene Understanding")

    # 2. Scene Segmentation
    log_info("Starting Scene Segmentation")
    # Load the config and fail if the Image Segmentation Server is unavailable.
    image_segmentation_client = ImageSegmentationClient.from_config(
        image_segmentation_config_path
    )
    try:
        image_segmentation_client.check_health()  # Error raising will happen internally.
        scene = segment_scene(
            image_path=image_path,
            output_root=resolved_output_root,
            scene=scene,
            vlm_client=vlm_client,
            image_segmentation_client=image_segmentation_client,
        )
    finally:
        image_segmentation_client.close()  # Kill the session to avoid resource leaks.
    log_info("Completed Scene Segmentation")

    # 3. Objects + Coarse Layout Generation
    log_info("Starting Objects + Coarse Layout Generation")
    # Load the config and fail if the Geometry Generation Server is unavailable.
    geometry_generation_client = GeometryGenerationClient.from_config(
        geometry_generation_config_path
    )
    try:
        geometry_generation_client.check_health()  # Error raising will happen internally.
        scene = generate_scene_and_refine(
            image_path=image_path,
            output_root=resolved_output_root,
            scene=scene,
            vlm_client=vlm_client,
            geometry_generation_client=geometry_generation_client,
        )
    finally:
        geometry_generation_client.close()  # Kill the session to avoid resource leaks.
    log_info("Completed Objects + Coarse Layout Generation")

    # 4. Scene Export
    log_info("Starting Scene Export")
    export_scene(
        scene=scene,
        output_root=resolved_output_root,
        table_max_convex_hull_num=16,
        asset_max_convex_hull_num=16,
    )
    log_info("Completed Scene Export")

    return scene
