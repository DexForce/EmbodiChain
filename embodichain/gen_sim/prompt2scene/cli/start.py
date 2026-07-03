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
from pathlib import Path

from embodichain.gen_sim.prompt2scene.pipeline.runner import run_prompt2scene
from embodichain.gen_sim.prompt2scene.llms import load_llm_config
from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput

__all__ = ["cli_prompt2scene", "main"]


def cli_prompt2scene(
    image_path: str | None,
    text: str | None,
    prompt: str | None,
    output_root: str,
    llm_config_path: str | None = None,
) -> None:
    """Run prompt2scene from normalized CLI argument values.

    Args:
        image_path: Path to an input image, if image mode is used.
        text: Text prompt, if text mode is used.
        prompt: Optional edit prompt.
        output_root: Directory where prompt2scene outputs are written.
        llm_config_path: Optional path to the LLM config JSON file.
    """
    request = Prompt2SceneInput.from_cli_args(
        image_path=Path(image_path) if image_path is not None else None,
        text=text,
        prompt=prompt,
        output_root=Path(output_root),
    )
    llm_cfg = load_llm_config(
        Path(llm_config_path) if llm_config_path is not None else None
    )
    run_prompt2scene(request, llm_cfg=llm_cfg)


def main() -> None:
    """Parse command line arguments and launch the prompt2scene pipeline."""
    parser = argparse.ArgumentParser(
        description="embodichain.gen_sim.prompt2scene Prompt-to-Scene Pipeline"
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to the input image file (.jpg, .jpeg, or .png)",
    )
    input_group.add_argument(
        "--text",
        type=str,
        help="Text prompt describing the target scene",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Optional edit instruction. Use with --image/--text to edit after "
            "generation, or with only --output_root to edit an existing scene."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--llm_config",
        type=str,
        default=None,
        help="Path to the LLM config JSON file",
    )

    args = parser.parse_args()

    cli_prompt2scene(
        args.image,
        args.text,
        args.prompt,
        args.output_root,
        args.llm_config,
    )


if __name__ == "__main__":
    main()
