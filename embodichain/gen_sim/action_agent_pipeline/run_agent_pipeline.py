#!/usr/bin/env python3
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

"""Run the Image2Tabletop -> config generation -> action-agent pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import shlex
import subprocess
import sys


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent
    return Path.cwd().resolve()


__all__ = ["main"]

_REPO_ROOT = _repo_root()
_DEFAULT_SERVER = "http://192.168.3.23:4523"
_DEFAULT_IMAGE = (
    _REPO_ROOT
    / "embodichain/gen_sim/action_agent_pipeline/gym_project_api/image/demo5.jpg"
)
_DEFAULT_IMAGE_DIR = _DEFAULT_IMAGE.parent
_DEFAULT_GYM_PROJECT_ROOT = _REPO_ROOT / "gym_project"
_DEFAULT_EXISTING_GYM_PROJECT = _DEFAULT_GYM_PROJECT_ROOT / "1780562837_gym_project"
_DEFAULT_CONFIG_OUTPUT_DIR = (
    _REPO_ROOT / "embodichain/gen_sim/action_agent_pipeline/configs/demo3_text"
)
_DEFAULT_TASK_NAME = "Depm3_Text"
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tabletop gym project from one image, generate action-agent "
            "configs from that project, then run the generated task."
        )
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image",
        default=None,
        help=f"Input image path. If omitted, defaults to {_DEFAULT_IMAGE.as_posix()}",
    )
    image_group.add_argument(
        "--image-name",
        "--image_name",
        dest="image_name",
        default=None,
        help=(
            "Image file name under the default image directory. The suffix is "
            'optional, e.g. "demo6" resolves to demo6.jpg.'
        ),
    )
    parser.add_argument(
        "--server",
        default=_DEFAULT_SERVER,
        help=f"Image2Tabletop API server. Defaults to {_DEFAULT_SERVER}",
    )
    parser.add_argument(
        "--gym-project-root",
        default=str(_DEFAULT_GYM_PROJECT_ROOT),
        help=(
            "Directory where Image2Tabletop generated gym projects are written. "
            f"Defaults to {_DEFAULT_GYM_PROJECT_ROOT.as_posix()}"
        ),
    )
    parser.add_argument(
        "--use-existing-gym-project",
        action="store_true",
        default=False,
        help=(
            "Skip Image2Tabletop API and start from --gym-project. Defaults to "
            "false."
        ),
    )
    parser.add_argument(
        "--gym-project",
        "--gym_project",
        dest="gym_project",
        default=str(_DEFAULT_EXISTING_GYM_PROJECT),
        help=(
            "Existing gym project used with --use-existing-gym-project. "
            f"Defaults to {_DEFAULT_EXISTING_GYM_PROJECT.as_posix()}"
        ),
    )
    parser.add_argument(
        "--config-output-dir",
        "--output_dir",
        dest="config_output_dir",
        default=str(_DEFAULT_CONFIG_OUTPUT_DIR),
        help=(
            "Destination directory for generated config files. Defaults to "
            f"{_DEFAULT_CONFIG_OUTPUT_DIR.as_posix()}"
        ),
    )
    parser.add_argument(
        "--task_name",
        "--task-name",
        dest="task_name",
        default=_DEFAULT_TASK_NAME,
        help=f"Task name passed to run_agent. Defaults to {_DEFAULT_TASK_NAME}",
    )
    parser.add_argument(
        "--task_description",
        "--task-description",
        dest="task_description",
        default="",
        help='Task description passed to config generation. Defaults to "".',
    )
    parser.add_argument(
        "--target_body_scale",
        "--target-body-scale",
        dest="target_body_scale",
        type=float,
        default=0.8,
        help="Uniform body_scale for generated non-table objects. Defaults to 0.8.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Image2Tabletop job polling interval in seconds. Defaults to 10.0.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        default=False,
        help="Skip GET /health before submitting the image.",
    )
    parser.add_argument(
        "--overwrite-gym-project",
        action="store_true",
        default=False,
        help="Replace an existing generated gym project with the same name.",
    )
    parser.add_argument(
        "--overwrite-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite generated config files. Defaults to true.",
    )
    parser.add_argument(
        "--regenerate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --regenerate to run_agent. Defaults to true.",
    )
    parser.add_argument(
        "--skip-run-agent",
        action="store_true",
        default=False,
        help="Stop after generating config files instead of launching run_agent.",
    )
    return parser


def _ensure_repo_on_pythonpath() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def _resolve_single_image(
    image_input: str,
    collect_image_paths: Callable[[Path], list[Path]],
) -> Path:
    image_paths = collect_image_paths(Path(image_input))
    if len(image_paths) != 1:
        paths = ", ".join(path.as_posix() for path in image_paths)
        raise ValueError(
            "This pipeline expects exactly one image, but got "
            f"{len(image_paths)}: {paths}"
        )
    return image_paths[0]


def _resolve_image_input(args: argparse.Namespace) -> Path:
    if args.image_name:
        return _resolve_image_name(args.image_name)
    if args.image:
        return Path(args.image)
    return _DEFAULT_IMAGE


def _resolve_image_name(image_name: str) -> Path:
    image_path = Path(image_name)
    if image_path.parent != Path("."):
        raise ValueError(
            "--image-name only accepts a file name under "
            f"{_DEFAULT_IMAGE_DIR.as_posix()}. Use --image for a full path."
        )
    if image_path.suffix:
        return _DEFAULT_IMAGE_DIR / image_path

    matches = [
        _DEFAULT_IMAGE_DIR / f"{image_name}{suffix}" for suffix in _IMAGE_SUFFIXES
    ]
    existing = [path for path in matches if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if not existing:
        candidates = ", ".join(path.name for path in matches)
        raise FileNotFoundError(
            f"Image name {image_name!r} was not found. Tried: {candidates}"
        )

    matched = ", ".join(path.name for path in existing)
    raise ValueError(
        f"Image name {image_name!r} is ambiguous. Use --image-name with a suffix: "
        f"{matched}"
    )


def _resolve_gym_project(args: argparse.Namespace) -> Path:
    if args.use_existing_gym_project:
        project_path = Path(args.gym_project).expanduser().resolve()
        if not project_path.exists():
            raise FileNotFoundError(f"gym project not found: {project_path}")
        print(f"Using existing gym project: {project_path}", flush=True)
        return project_path

    from embodichain.gen_sim.action_agent_pipeline.gym_project_api.image2tabletop_client import (
        check_health,
        collect_image_paths,
        process_image,
    )

    image_input = _resolve_image_input(args)
    image_path = _resolve_single_image(str(image_input), collect_image_paths)
    if not args.skip_health_check:
        check_health(args.server)

    return process_image(
        server=args.server,
        image_path=image_path,
        output_root=Path(args.gym_project_root),
        poll_interval=args.poll_interval,
        overwrite=args.overwrite_gym_project,
    )


def _run_agent_command(
    *,
    task_name: str,
    gym_config: Path,
    agent_config: Path,
    regenerate: bool,
) -> int:
    command = [
        sys.executable,
        "-m",
        "embodichain.gen_sim.action_agent_pipeline.cli.run_agent",
        "--task_name",
        task_name,
        "--gym_config",
        str(gym_config),
        "--agent_config",
        str(agent_config),
    ]
    if regenerate:
        command.append("--regenerate")

    print("Running task:")
    print(shlex.join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = _build_parser().parse_args()

    _ensure_repo_on_pythonpath()
    from embodichain.gen_sim.action_agent_pipeline.ur5_basket_config_generation import (
        generate_ur5_basket_config_from_project,
    )

    project_path = _resolve_gym_project(args)

    paths = generate_ur5_basket_config_from_project(
        gym_project=project_path,
        output_dir=args.config_output_dir,
        task_name=args.task_name,
        task_description=args.task_description,
        target_body_scale=args.target_body_scale,
        overwrite=args.overwrite_config,
    )

    print(f"Generated gym project: {project_path}", flush=True)
    print(f"Generated gym config: {paths.gym_config}", flush=True)
    print(f"Generated agent config: {paths.agent_config}", flush=True)
    if args.skip_run_agent:
        return 0

    return _run_agent_command(
        task_name=args.task_name,
        gym_config=paths.gym_config,
        agent_config=paths.agent_config,
        regenerate=args.regenerate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
