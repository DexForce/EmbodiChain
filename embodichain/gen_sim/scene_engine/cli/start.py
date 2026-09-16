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
from collections.abc import Sequence
import math
from pathlib import Path

from embodichain.gen_sim.scene_engine.pipeline.generate import generate_scene_from_image
from embodichain.gen_sim.scene_engine.pipeline.edit import edit_scene

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def cli_scene_engine(
    image: str | Path | None,
    output_root: str | Path,
    *,
    edit_prompt: str | None = None,
    scene_z_rotation_degrees: float = 0.0,
) -> None:
    """Generate a scene from an image, edit an export, or do both in sequence."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    if not math.isfinite(scene_z_rotation_degrees):
        raise ValueError("scene_z_rotation_degrees must be finite.")
    if edit_prompt is not None:
        edit_prompt = edit_prompt.strip()
        if not edit_prompt:
            raise ValueError("Edit prompt must not be empty.")

    if image is None:
        if edit_prompt is None:
            raise ValueError("Provide --image, --edit_prompt, or both.")
        edit_scene(output_root=resolved_output_root, edit_prompt=edit_prompt)
        print("Successfully completed!")
        return

    resolved_image_path = Path(image).expanduser().resolve()
    if not resolved_image_path.exists():
        raise FileNotFoundError(f"Image input not found: {resolved_image_path}")
    if not resolved_image_path.is_file():
        raise ValueError(f"Image input is not a file: {resolved_image_path}")
    if resolved_image_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            "Image input must have one of these extensions: .jpg, .jpeg, .png"
        )

    resolved_output_root.mkdir(parents=True, exist_ok=True)
    generate_scene_from_image(
        image_path=resolved_image_path,
        output_root=resolved_output_root,
        scene_z_rotation_degrees=scene_z_rotation_degrees,
    )
    if edit_prompt is not None:
        edit_scene(
            output_root=resolved_output_root,
            edit_prompt=edit_prompt,
        )
    print("Successfully completed!")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="embodichain scene-engine",
        description="Generate a Scene Engine export, edit one, or do both.",
        epilog="Service settings are read from embodichain/gen_sim/.env.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="Optional input image file (.jpg, .jpeg, or .png)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--edit_prompt",
        type=str,
        default=None,
        help="Text instruction for editing an existing or newly generated output root",
    )
    parser.add_argument(
        "--scene-z-rotation-degrees",
        "--scene_z_rotation_degrees",
        "--prompt2scene-scene-z-rotation-degrees",
        "--prompt2scene_scene_z_rotation_degrees",
        dest="scene_z_rotation_degrees",
        type=float,
        default=0.0,
        help=(
            "Final counterclockwise world-Z rotation applied to the complete "
            "generated scene. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--target-body-scale-mode",
        "--target_body_scale_mode",
        choices=("preserve",),
        default="preserve",
        help=(
            "Compatibility option for the direct-GLB Scene Engine path; source "
            "scale is always preserved."
        ),
    )
    parser.add_argument(
        "--prompt2scene-mesh-x-rotation-degrees",
        "--prompt2scene_mesh_x_rotation_degrees",
        type=_zero_mesh_x_rotation,
        default=0.0,
        help=(
            "Compatibility option for direct GLB loading. It must remain 0; "
            "DexSim performs the GLTF y-up conversion without a baked 90-degree fix."
        ),
    )
    args = parser.parse_args(argv)

    cli_scene_engine(
        args.image,
        args.output_root,
        edit_prompt=args.edit_prompt,
        scene_z_rotation_degrees=args.scene_z_rotation_degrees,
    )


def _zero_mesh_x_rotation(value: str) -> float:
    rotation = float(value)
    if not math.isfinite(rotation) or rotation != 0.0:
        raise argparse.ArgumentTypeError(
            "prompt2scene mesh X rotation must be 0 for direct GLB loading."
        )
    return rotation


if __name__ == "__main__":
    main()
