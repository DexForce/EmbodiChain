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
from pathlib import Path

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def cli_scene_engine(
    image: str | Path,
    output_root: str | Path,
) -> None:
    """Generate one scene using the shared GenSim ``.env`` configuration."""
    resolved_image_path = Path(image).expanduser().resolve()
    if not resolved_image_path.exists():
        raise FileNotFoundError(f"Image input not found: {resolved_image_path}")
    if not resolved_image_path.is_file():
        raise ValueError(f"Image input is not a file: {resolved_image_path}")
    if resolved_image_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            "Image input must have one of these extensions: .jpg, .jpeg, .png"
        )

    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    # Importing the generation pipeline initializes simulation dependencies.
    # Keeping it here lets ``embodichain scene-engine --help`` work in a
    # lightweight CLI-only environment.
    from embodichain.gen_sim.scene_engine.pipeline.generate import (
        generate_scene_from_image,
    )

    generate_scene_from_image(
        image_path=resolved_image_path,
        output_root=resolved_output_root,
    )
    print("Successfully completed!")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="embodichain scene-engine",
        description="embodichain.gen_sim.scene_engine Scene Engine Pipeline",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the required input image file (.jpg, .jpeg, or .png)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    args = parser.parse_args(argv)

    cli_scene_engine(args.image, args.output_root)


if __name__ == "__main__":
    main()
