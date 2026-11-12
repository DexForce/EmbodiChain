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

__all__ = ["cli_scene_engine", "main"]

from embodichain.gen_sim.services.runtime import GenSimServices, service_entry


import argparse
from collections.abc import Sequence
from pathlib import Path

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@service_entry
def cli_scene_engine(
    image: str | Path | None,
    output_root: str | Path,
    *,
    edit_prompt: str | None = None,
    scene_z_rotation_degrees: float = 180.0,
    scene_randomization: dict | str | Path | None = None,
    services: GenSimServices | None = None,
) -> None:
    """Generate a scene from an image, edit an export, or do both in sequence."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    if edit_prompt is not None:
        edit_prompt = edit_prompt.strip()
        if not edit_prompt:
            raise ValueError("Edit prompt must not be empty.")

    if image is None:
        if edit_prompt is None:
            raise ValueError("Provide --image, --edit_prompt, or both.")
        from embodichain.gen_sim.scene_engine.pipeline.edit import edit_scene

        edit_scene(
            output_root=resolved_output_root, edit_prompt=edit_prompt, services=services
        )
        if scene_randomization is not None:
            from embodichain.gen_sim.scene_engine.randomization.adapter import (
                randomize_scene_export,
            )

            randomize_scene_export(resolved_output_root, scene_randomization)
        services.write_records(resolved_output_root / "service_calls.json")
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
    try:
        from embodichain.gen_sim.scene_engine.pipeline.generate import (
            generate_scene_from_image,
        )

        generate_scene_from_image(
            image_path=resolved_image_path,
            output_root=resolved_output_root,
            scene_z_rotation_degrees=scene_z_rotation_degrees,
            services=services,
        )
        if edit_prompt is not None:
            from embodichain.gen_sim.scene_engine.pipeline.edit import edit_scene

            edit_scene(
                output_root=resolved_output_root,
                edit_prompt=edit_prompt,
                services=services,
            )
        if scene_randomization is not None:
            from embodichain.gen_sim.scene_engine.randomization.adapter import (
                randomize_scene_export,
            )

            randomize_scene_export(resolved_output_root, scene_randomization)
    finally:
        services.write_records(resolved_output_root / "service_calls.json")
    print("Successfully completed!")


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "embodichain scene-engine",
    operation: str | None = None,
) -> None:
    """Parse a scene operation while retaining the combined legacy entry.

    Args:
        argv: Arguments after the operation or legacy command name.
        prog: Display name supplied by the unified GenSim dispatcher.
        operation: Generate/edit selection; None preserves the combined CLI.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate a Scene Engine export, edit one, or do both.",
        epilog="Service settings are read from embodichain/gen_sim/.env.",
    )
    if operation != "edit":
        parser.add_argument(
            "--image",
            type=str,
            required=operation == "generate",
            help="Input image file (.jpg, .jpeg, or .png)",
        )
    else:
        parser.set_defaults(image=None)
    parser.add_argument(
        "--scene" if operation == "edit" else "--output",
        "--output-root",
        "--output_root",
        dest="output_root",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    if operation != "generate":
        parser.add_argument(
            "--prompt",
            "--edit-prompt",
            "--edit_prompt",
            dest="edit_prompt",
            type=str,
            required=operation == "edit",
            help="Text instruction for editing the scene workspace in place",
        )
    else:
        parser.set_defaults(edit_prompt=None)
    parser.add_argument(
        "--expansion-config",
        "--scene-randomization-config",
        dest="scene_randomization_config",
        type=Path,
        default=None,
        help="Optional post-export layout policy YAML",
    )
    args = parser.parse_args(argv)

    cli_scene_engine(
        args.image,
        args.output_root,
        edit_prompt=args.edit_prompt,
        scene_z_rotation_degrees=180.0,
        scene_randomization=args.scene_randomization_config,
    )


if __name__ == "__main__":
    main()
