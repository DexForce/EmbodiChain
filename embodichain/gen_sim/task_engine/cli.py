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

"""Unified CLI for complete Task Engine workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Final, Sequence

from .orchestration.scene_adapter import SceneAdapter
from .run_directory import reserve_run_directory
from .workflow import TaskEngineWorkflow
from .workflow_contracts import (
    TASK_RUN_REQUEST_SCHEMA,
    validate_scene_history_root,
    validate_scene_output_separation,
)

__all__ = ["build_parser", "main"]


_ROBOT_PROFILES = (
    "ur5",
    "ur10",
    "dual_ur5",
    "dual_ur10",
    "franka",
    "dual_franka",
)
_MODES: Final = ("image", "image-edit", "scene", "scene-edit")


def build_parser() -> argparse.ArgumentParser:
    """Build the Task Engine parser."""
    parser = argparse.ArgumentParser(
        prog="embodichain task-engine",
        description="Run one complete Scene and Action workflow.",
    )
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--task-id", "--task_id", required=True)
    instruction = parser.add_mutually_exclusive_group(required=True)
    instruction.add_argument("--instruction")
    instruction.add_argument("--task-file", "--task_file")
    parser.add_argument("--image")
    parser.add_argument("--scene")
    parser.add_argument("--scene-edit", "--scene_edit", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--dataset_saving",
        action="store_true",
        help="Opt in to the Gym project's dataset recorder during execution.",
    )
    parser.add_argument(
        "--robot-profile",
        choices=_ROBOT_PROFILES,
        default="franka",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one complete workflow and publish it under a new timestamped run."""
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        image, scene, edit = _mode_inputs(args)
    except ValueError as exc:
        parser.error(str(exc))
    if scene is not None:
        validate_scene_history_root(scene, args.output_root)
    instruction = _instruction(args)
    adapter = SceneAdapter(model=args.model, robot_profile=args.robot_profile)
    workflow = TaskEngineWorkflow(scene_adapter=adapter)
    with reserve_run_directory(args.output_root) as allocation:
        if scene is not None:
            validate_scene_output_separation(scene, allocation.path)
        result = workflow.run(
            {
                "schema_version": TASK_RUN_REQUEST_SCHEMA,
                "task_id": args.task_id,
                "task_instruction": instruction,
                "image_path": image,
                "gym_project": scene,
                "scene_edit_prompt": edit,
                "output_dir": allocation.path.as_posix(),
            },
            config_path=args.config,
            model=args.model,
            vlm_model=args.vlm_model,
            base_seed=args.base_seed,
            dataset_saving=args.dataset_saving,
            run_id=allocation.run_id,
            created_at=allocation.created_at,
        )
    _print_json(
        {
            "run_id": allocation.run_id,
            "status": result.status,
            "failure_class": result.failure_class,
            "output_dir": result.output_dir.as_posix(),
            "manifest": result.manifest_path.as_posix(),
            "final_bundle": (
                None if result.final_bundle is None else result.final_bundle.as_posix()
            ),
        }
    )
    return 0 if result.succeeded else 2


def _instruction(args: argparse.Namespace) -> str:
    instruction = (
        str(args.instruction).strip()
        if args.instruction is not None
        else Path(args.task_file).expanduser().read_text(encoding="utf-8").strip()
    )
    if not instruction:
        raise ValueError("Task instruction must not be empty.")
    return instruction


def _mode_inputs(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    image = None if args.image is None else str(args.image).strip()
    scene = None if args.scene is None else str(args.scene).strip()
    edit = None if args.scene_edit is None else str(args.scene_edit).strip()
    expected = {
        "image": (True, False, False),
        "image-edit": (True, False, True),
        "scene": (False, True, False),
        "scene-edit": (False, True, True),
    }[args.mode]
    actual = (bool(image), bool(scene), bool(edit))
    if actual != expected:
        raise ValueError(
            f"mode={args.mode!r} requires image/scene/edit={expected}, got {actual}."
        )
    return image, scene, edit


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
