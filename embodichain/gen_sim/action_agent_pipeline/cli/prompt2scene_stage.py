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
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_defaults import (
    DEFAULT_IMAGE,
    DEFAULT_IMAGE_DIR,
    IMAGE_SUFFIXES,
)

__all__ = ["resolve_prompt2scene_image", "run_prompt2scene_stage"]


def run_prompt2scene_stage(args: argparse.Namespace) -> Path:
    """Run the in-repo prompt2scene stage and return its exported gym config."""
    load_llm_config, run_prompt2scene, prompt2scene_input_cls = (
        _load_prompt2scene_components()
    )
    output_root = Path(args.prompt2scene_output_root).expanduser().resolve()
    text = _resolve_prompt2scene_text(args)
    image_path = None if text is not None else resolve_prompt2scene_image(args)
    request = prompt2scene_input_cls.from_cli_args(
        image_path=image_path,
        text=text,
        output_root=output_root,
    )
    llm_cfg = load_llm_config(
        Path(args.prompt2scene_llm_config).expanduser()
        if args.prompt2scene_llm_config
        else None
    )

    print("Running prompt2scene pipeline:", flush=True)
    if request.image_path is not None:
        print(f"  image: {request.image_path}", flush=True)
    if request.text is not None:
        print(f"  text: {request.text}", flush=True)
    print(f"  output_root: {request.output_root}", flush=True)

    result = run_prompt2scene(request, llm_cfg=llm_cfg)
    if result.gym_config_path is None or not result.gym_config_path.is_file():
        raise FileNotFoundError(
            "prompt2scene did not produce an exported gym_config.json under "
            f"{request.output_root}"
        )

    print(f"Using prompt2scene gym config: {result.gym_config_path}", flush=True)
    return result.gym_config_path


def _load_prompt2scene_components() -> tuple[Any, Any, Any]:
    from embodichain.gen_sim.prompt2scene.llms import load_llm_config
    from embodichain.gen_sim.prompt2scene.pipeline.runner import run_prompt2scene
    from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput

    return load_llm_config, run_prompt2scene, Prompt2SceneInput


def resolve_prompt2scene_image(args: argparse.Namespace) -> Path:
    """Resolve prompt2scene image input from shared action-agent CLI options."""
    if args.image_name:
        return _resolve_image_name(args.image_name)
    image_input = args.image or DEFAULT_IMAGE
    return Path(image_input).expanduser().resolve()


def _resolve_prompt2scene_text(args: argparse.Namespace) -> str | None:
    text = str(getattr(args, "prompt2scene_text", "") or "").strip()
    if text:
        if args.image or args.image_name:
            raise ValueError(
                "--prompt2scene-text cannot be combined with --image or --image-name."
            )
        return text
    return None


def _resolve_image_name(image_name: str) -> Path:
    image_path = Path(image_name).expanduser()
    if image_path.parent != Path("."):
        raise ValueError(
            "--image-name only accepts a file name under "
            f"{DEFAULT_IMAGE_DIR.as_posix()}. Use --image for a full path."
        )
    if image_path.suffix:
        return (DEFAULT_IMAGE_DIR / image_path).resolve()

    candidates = [
        DEFAULT_IMAGE_DIR / f"{image_name}{suffix}" for suffix in IMAGE_SUFFIXES
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0].resolve()
    if not existing:
        searched = ", ".join(path.as_posix() for path in candidates)
        raise FileNotFoundError(
            f"Image name '{image_name}' was not found. Tried: {searched}"
        )

    matched = ", ".join(path.name for path in existing)
    raise ValueError(
        f"Image name '{image_name}' is ambiguous. Use --image-name with a suffix: "
        f"{matched}"
    )
