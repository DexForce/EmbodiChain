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

__all__: list[str] = []

from embodichain.gen_sim.services.runtime import GenSimServices, service_entry
from embodichain.gen_sim.services.compat import (
    VisionFacade,
    GeometryFacade,
    ImageFacade,
    SegmentationFacade,
    ArticulationFacade,
)


from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim.scene.expansion import SceneRandomizationJobCfg

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.services.compat import VisionClient
from embodichain.gen_sim.services.compat import GeometryClient
from embodichain.gen_sim.services.compat import ArticulationClient
from embodichain.gen_sim.services.compat import SegmentationClient

from embodichain.gen_sim.scene_engine.pipeline.generation.scene_understanding import (
    understand_scene,
)
from embodichain.utils.logger import log_info

from embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation import (
    generate_scene_and_refine,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import SceneExporter
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_utils import (
    rotate_scene_z_up_world,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
    build_scene_usd,
    build_scene_usdz,
)


@service_entry
def generate_scene_from_image(
    image_path: str | Path,
    output_root: str | Path,
    *,
    scene_z_rotation_degrees: float = 0.0,
    scene_randomization: SceneRandomizationJobCfg | dict | str | Path | None = None,
    services: GenSimServices | None = None,
) -> Scene:
    """Generate the initial core scene state from an input image.

    Args:
        image_path: Source tabletop image.
        output_root: Directory receiving intermediate and exported artifacts.
        scene_z_rotation_degrees: Final counterclockwise world-z rotation applied
            rigidly to the complete scene before export.
        scene_randomization: Optional grouped layout policy applied to the final
            exported workspace without changing the generated source scene.

    Returns:
        The final scene in the rotated export frame.
    """
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    # Initialize the VLM client and the Scene data structure.
    vlm_client = VisionFacade(services, "scene_understanding")
    scene = Scene()

    # 1. Scene Understanding
    log_info("Starting Scene Understanding")
    # Load .env settings and fail if the Image Segmentation Server is unavailable.
    image_segmentation_client = SegmentationFacade(services)
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
    geometry_generation_client = GeometryFacade(services)
    articulated_generation_client: ArticulationClient | None = None
    if any(scene_object.is_articulated for scene_object in scene.objects):
        articulated_generation_client = ArticulationFacade(services)
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
    rotate_scene_z_up_world(
        scene=scene,
        scene_graph=scene_graph,
        rotation_degrees=scene_z_rotation_degrees,
    )
    scene_exporter = SceneExporter(
        scene=scene,
        scene_graph=scene_graph,
        output_root=resolved_output_root,
    )
    scene_exporter.export()
    scene_usd_path = build_scene_usd(output_root=resolved_output_root)
    build_scene_usdz(scene_usd_path=scene_usd_path)
    log_info("Completed Scene Export")

    if scene_randomization is not None:
        from embodichain.gen_sim.scene_engine.randomization.adapter import (
            randomize_scene_export,
        )

        randomize_scene_export(resolved_output_root, scene_randomization)
    return scene
