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
from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.articulated_generation import (
    ArticulatedGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
)

from embodichain.gen_sim.scene_engine.pipeline.generation.scene_understanding import (
    understand_scene,
)
from embodichain.utils.logger import log_info

from embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation import (
    generate_scene_and_refine,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import SceneExporter


def generate_scene_from_image(
    image_path: str | Path,
    output_root: str | Path,
) -> Scene:
    """Generate the initial core scene state from an input image."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    # Initialize the VLM client and the Scene data structure.
    vlm_client = OpenAICompatibleVLM.from_dotenv()
    scene = Scene()

    # 1. Scene Understanding
    log_info("Starting Scene Understanding")
    # Load .env settings and fail if the Image Segmentation Server is unavailable.
    image_segmentation_client = ImageSegmentationClient.from_dotenv()
    try:
        image_segmentation_client.check_health()
        scene, scene_graph = understand_scene(
            scene=scene,
            image_path=image_path,
            output_root=resolved_output_root,
            vlm_client=vlm_client,
            image_segmentation_client=image_segmentation_client,
        )
    finally:
        image_segmentation_client.close()  # Close the session after scene understanding.
    log_info("Completed Scene Understanding")

    # 2. Objects + Coarse Layout Generation
    log_info("Starting Objects + Coarse Layout Generation")
    # Load .env settings and fail if the Geometry Generation Server is unavailable.
    geometry_generation_client = GeometryGenerationClient.from_dotenv()
    articulated_generation_client: ArticulatedGenerationClient | None = None
    if any(scene_object.is_articulated for scene_object in scene.objects):
        articulated_generation_client = ArticulatedGenerationClient.from_dotenv()
    try:
        geometry_generation_client.check_health()  # Error raising will happen internally.
        if articulated_generation_client is not None:
            articulated_generation_client.check_health()
        scene = generate_scene_and_refine(
            image_path=image_path,
            output_root=resolved_output_root,
            scene=scene,
            scene_graph=scene_graph,
            geometry_generation_client=geometry_generation_client,
            articulated_generation_client=articulated_generation_client,
            vlm_client=vlm_client,
        )
    finally:
        geometry_generation_client.close()  # Kill the session to avoid resource leaks.
        if articulated_generation_client is not None:
            articulated_generation_client.close()
    log_info("Completed Objects + Coarse Layout Generation")

    # 3. Scene Export
    log_info("Starting Scene Export")
    scene_exporter = SceneExporter(
        scene=scene,
        scene_graph=scene_graph,
        output_root=resolved_output_root,
    )
    scene_exporter.export()
    log_info("Completed Scene Export")

    return scene
