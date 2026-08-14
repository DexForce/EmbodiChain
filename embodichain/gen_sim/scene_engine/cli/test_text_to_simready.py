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

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
import shutil

from PIL import Image

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.image_generation import (
    ImageGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    build_mask_candidates,
    invert_mask_if_foreground_is_off_center,
    save_binary_mask,
    union_overlapping_mask_candidates,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor import (
    SimReadyProcessor,
    SimReadyProcessorConfig,
)

__all__ = ["main", "run_text_to_simready"]


def run_text_to_simready(*, text: str, output_root: str | Path) -> SceneObject:
    """Run one manual text-to-SimReady asset pipeline for debugging."""
    text = text.strip()
    if not text:
        raise ValueError("Text prompt must not be empty.")

    root = Path(output_root).expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    debug_root = root / "debug"
    image_root = root / "generated_images"
    mask_root = root / "masks"
    coarse_root = root / "coarse_geometry"
    simready_root = root / "simready_geometry"
    for directory in (debug_root, image_root, mask_root, coarse_root, simready_root):
        directory.mkdir(parents=True, exist_ok=True)

    object_id = "asset_001"
    scene_object = SceneObject(
        id=object_id,
        kind="asset",
        category="asset",
        name=text,
        description=text,
    )

    image_generation_client = ImageGenerationClient.from_dotenv()
    image_segmentation_client = ImageSegmentationClient.from_dotenv()
    geometry_generation_client = GeometryGenerationClient.from_dotenv()
    vlm_client = OpenAICompatibleVLM.from_dotenv()
    try:
        image_generation_client.check_health()
        image_segmentation_client.check_health()
        geometry_generation_client.check_health()

        # Generate a centered single-object image from the semantic text prompt.
        image_path = image_generation_client.generate_image_by_prompt(
            prompt=text,
            output_path=image_root / f"{object_id}.png",
        )
        with Image.open(image_path) as image:
            image_size = image.size

        # Segment the generated object and apply the single-object foreground heuristic.
        candidates = union_overlapping_mask_candidates(
            build_mask_candidates(
                image_segmentation_client.segment_single_object(
                    image_path=image_path,
                    prompt=text,
                )
            ),
            min_iou=0.8,
        )
        if not candidates:
            raise ValueError("Image segmentation returned no mask candidates.")
        mask_path = save_binary_mask(
            invert_mask_if_foreground_is_off_center(candidates[0]),
            image_size=image_size,
            output_path=mask_root / f"{object_id}.png",
        )

        # Generate one coarse GLB using the generated image and its binary mask.
        geometry_generation_client.generate_objects(
            image_path=image_path,
            object_masks=[(object_id, mask_path)],
            output_root=coarse_root,
        )
        coarse_glb_path = coarse_root / f"{object_id}.glb"
        if not coarse_glb_path.is_file():
            raise FileNotFoundError(f"Coarse GLB was not generated: {coarse_glb_path}")

        # Use identity coarse layout; VLM determines rotation and real-world size.
        processor = SimReadyProcessor(
            scene=Scene(objects=[scene_object]),
            coarse_layout_by_id={
                object_id: {
                    "rot": [0.0, 0.0, 0.0],
                    "pos": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            },
            coarse_geometry_root=coarse_root,
            simready_geometry_root=simready_root,
            config=SimReadyProcessorConfig(
                use_vlm_scale=True,
                use_vlm_rotation=True,
            ),
            vlm_client=vlm_client,
        )
        simready_layout = processor.process_assets()
        (root / "result.json").write_text(
            json.dumps(
                {
                    "input_text": text,
                    "scene_object": scene_object.to_dict(),
                    "simready_layout": simready_layout,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return scene_object
    finally:
        image_generation_client.close()
        image_segmentation_client.close()
        geometry_generation_client.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the manual text-to-SimReady CLI."""
    parser = argparse.ArgumentParser(
        prog="embodichain test-text-to-simready",
        description="Debug text-to-image-to-segmentation-to-SimReady generation.",
    )
    parser.add_argument("--text", required=True, help="Description of one object.")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory for all intermediate and final artifacts.",
    )
    args = parser.parse_args(argv)
    scene_object = run_text_to_simready(text=args.text, output_root=args.output_root)
    print(f"Generated SimReady asset: {scene_object.simready_glb_path}")


if __name__ == "__main__":
    main()
