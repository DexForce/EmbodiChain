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
_DEFAULT_IMAGE2SCENE_ROOT = _REPO_ROOT / "gym_project/environment/image2tabletop"
_DEFAULT_IMAGE2SCENE_IMAGE = "scene_image/robotwin_example.png"
_DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR = "./downloads"
_DEFAULT_IMAGE2SCENE_OUTPUT_ROOT = "./generated"
_DEFAULT_IMAGE2SCENE_CONFIG = "./gen_config.json"
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
        help=(
            f"Input image path. If omitted, defaults to {_DEFAULT_IMAGE.as_posix()} "
            f"or {_DEFAULT_IMAGE2SCENE_IMAGE} with --use-image2scene."
        ),
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
        "--use-image2scene",
        action="store_true",
        default=False,
        help=(
            "Use gym_project/environment/image2tabletop/demo_api/client/"
            "image2scene_pipeline.py as the first stage and continue from its "
            "gym_config_merged.json output."
        ),
    )
    parser.add_argument(
        "--background",
        default=None,
        help=(
            "Background description passed to image2scene_pipeline.py. Required "
            "with --use-image2scene."
        ),
    )
    parser.add_argument(
        "--image2scene-root",
        default=str(_DEFAULT_IMAGE2SCENE_ROOT),
        help=(
            "Working directory for image2scene_pipeline.py. Defaults to "
            f"{_DEFAULT_IMAGE2SCENE_ROOT.as_posix()}"
        ),
    )
    parser.add_argument(
        "--image2scene-download-dir",
        default=_DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR,
        help=(
            "Download directory passed to image2scene_pipeline.py. Relative "
            "paths are interpreted under --image2scene-root. Defaults to "
            f"{_DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR}."
        ),
    )
    parser.add_argument(
        "--image2scene-output-root",
        default=_DEFAULT_IMAGE2SCENE_OUTPUT_ROOT,
        help=(
            "Generated EC project directory passed to image2scene_pipeline.py. "
            "Relative paths are interpreted under --image2scene-root. Defaults "
            f"to {_DEFAULT_IMAGE2SCENE_OUTPUT_ROOT}."
        ),
    )
    parser.add_argument(
        "--image2scene-gen-config",
        default=_DEFAULT_IMAGE2SCENE_CONFIG,
        help=(
            "Generation config passed to image2scene_pipeline.py. Relative "
            "paths are interpreted under --image2scene-root. Defaults to "
            f"{_DEFAULT_IMAGE2SCENE_CONFIG}."
        ),
    )
    parser.add_argument(
        "--image2scene-llm-config",
        default=_DEFAULT_IMAGE2SCENE_CONFIG,
        help=(
            "LLM config passed to image2scene_pipeline.py. Relative paths are "
            "interpreted under --image2scene-root. Defaults to "
            f"{_DEFAULT_IMAGE2SCENE_CONFIG}."
        ),
    )
    parser.add_argument(
        "--image2scene-extract-dir",
        default=None,
        help=(
            "Optional extract directory passed to image2scene_pipeline.py. "
            "Relative paths are interpreted under --image2scene-root."
        ),
    )
    parser.add_argument(
        "--image2scene-merged-output",
        default=None,
        help=(
            "Optional merged output path passed to image2scene_pipeline.py. "
            "Relative paths are interpreted under --image2scene-root."
        ),
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
        "--use-latest-image2scene-gym-project",
        "--use-latest-image2scene-gym_project",
        dest="use_latest_image2scene_gym_project",
        action="store_true",
        default=False,
        help=(
            "Skip image generation and start from the newest "
            "gym_config_merged.json under --image2scene-download-dir."
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
        help=(
            "Uniform body_scale for generated target objects. Basket-like "
            "containers keep their source body_scale. Defaults to 0.8."
        ),
    )
    parser.add_argument(
        "--target_replacement1",
        "--target-replacement1",
        nargs=2,
        metavar=("SOURCE_UID", "PROMPT"),
        default=None,
        help=(
            "Generate <gym_project>/mesh_assets/new1 from PROMPT and use it "
            "to replace SOURCE_UID in the generated config."
        ),
    )
    parser.add_argument(
        "--target_replacement2",
        "--target-replacement2",
        nargs=2,
        metavar=("SOURCE_UID", "PROMPT"),
        default=None,
        help=(
            "Generate <gym_project>/mesh_assets/new2 from PROMPT and use it "
            "to replace SOURCE_UID in the generated config."
        ),
    )
    parser.add_argument(
        "--sync_replacement_names",
        "--sync-replacement-names",
        action="store_true",
        default=False,
        help=(
            "Also update replacement target runtime UIDs and generated prompts "
            "from the replacement prompts."
        ),
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


def _resolve_under_root(root: Path, path_input: str | None) -> Path | None:
    if path_input is None:
        return None
    path = Path(path_input).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _collect_merged_gym_configs(download_dir: Path) -> list[Path]:
    if not download_dir.exists():
        return []
    return sorted(
        path.resolve() for path in download_dir.rglob("gym_config_merged.json")
    )


def _latest_path(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime)


def _resolve_image2scene_image(
    args: argparse.Namespace, image2scene_root: Path
) -> Path:
    if args.image_name:
        image_name = Path(args.image_name)
        if image_name.parent != Path("."):
            raise ValueError(
                "--image-name only accepts a file name under "
                f"{_DEFAULT_IMAGE_DIR.as_posix()} with "
                "--use-image2scene. Use --image for a full path."
            )
        if image_name.suffix:
            return (_DEFAULT_IMAGE_DIR / image_name).resolve()

        matches = [
            _DEFAULT_IMAGE_DIR / f"{args.image_name}{suffix}"
            for suffix in _IMAGE_SUFFIXES
        ]
        existing = [path.resolve() for path in matches if path.exists()]
        if len(existing) == 1:
            return existing[0]
        if not existing:
            candidates = ", ".join(path.name for path in matches)
            raise FileNotFoundError(
                f"Image name {args.image_name!r} was not found. Tried: {candidates}"
            )

        matched = ", ".join(path.name for path in existing)
        raise ValueError(
            f"Image name {args.image_name!r} is ambiguous. Use --image-name "
            f"with a suffix: {matched}"
        )

    image_input = args.image or _DEFAULT_IMAGE2SCENE_IMAGE
    image_path = Path(image_input).expanduser()
    if image_path.is_absolute():
        return image_path.resolve()
    return (image2scene_root / image_path).resolve()


def _run_image2scene_pipeline(args: argparse.Namespace) -> Path:
    if not args.background:
        raise ValueError("--background is required with --use-image2scene.")

    image2scene_root = Path(args.image2scene_root).expanduser().resolve()
    if not image2scene_root.is_dir():
        raise FileNotFoundError(f"image2scene root not found: {image2scene_root}")

    script_path = image2scene_root / "demo_api/client/image2scene_pipeline.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"image2scene pipeline not found: {script_path}")

    image_path = _resolve_image2scene_image(args, image2scene_root)
    download_dir = _resolve_under_root(image2scene_root, args.image2scene_download_dir)
    output_root = _resolve_under_root(image2scene_root, args.image2scene_output_root)
    gen_config = _resolve_under_root(image2scene_root, args.image2scene_gen_config)
    llm_config = _resolve_under_root(image2scene_root, args.image2scene_llm_config)
    extract_dir = _resolve_under_root(image2scene_root, args.image2scene_extract_dir)
    merged_output = _resolve_under_root(
        image2scene_root, args.image2scene_merged_output
    )

    if (
        download_dir is None
        or output_root is None
        or gen_config is None
        or llm_config is None
    ):
        raise ValueError("image2scene paths must not be empty.")

    before_configs = set(_collect_merged_gym_configs(download_dir))
    command = [
        sys.executable,
        str(script_path),
        "--server",
        args.server,
        "--image",
        str(image_path),
        "--download-dir",
        str(download_dir),
        "--background",
        args.background,
        "--output-root",
        str(output_root),
        "--gen-config",
        str(gen_config),
        "--llm-config",
        str(llm_config),
        "--poll-interval",
        str(args.poll_interval),
    ]
    if extract_dir is not None:
        command.extend(["--extract-dir", str(extract_dir)])
    if merged_output is not None:
        command.extend(["--merged-output", str(merged_output)])

    print("Running image2scene pipeline:")
    print(shlex.join(command), flush=True)
    completed = subprocess.run(command, cwd=image2scene_root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"image2scene pipeline failed with exit code {completed.returncode}"
        )

    if merged_output is not None:
        if not merged_output.is_file():
            raise FileNotFoundError(
                f"image2scene merged output not found: {merged_output}"
            )
        print(f"Using image2scene merged gym config: {merged_output}", flush=True)
        return merged_output

    after_configs = _collect_merged_gym_configs(download_dir)
    new_configs = [path for path in after_configs if path not in before_configs]
    if new_configs:
        merged_config = _latest_path(new_configs)
    elif after_configs:
        merged_config = _latest_path(after_configs)
    else:
        raise FileNotFoundError(
            f"gym_config_merged.json not found under: {download_dir}"
        )

    print(f"Using image2scene merged gym config: {merged_config}", flush=True)
    return merged_config


def _resolve_gym_project(args: argparse.Namespace) -> Path:
    selected_modes = [
        args.use_image2scene,
        args.use_existing_gym_project,
        args.use_latest_image2scene_gym_project,
    ]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise ValueError(
            "Use only one of --use-image2scene, --use-existing-gym-project, "
            "or --use-latest-image2scene-gym-project."
        )

    if args.use_existing_gym_project:
        project_path = Path(args.gym_project).expanduser().resolve()
        if not project_path.exists():
            raise FileNotFoundError(f"gym project not found: {project_path}")
        print(f"Using existing gym project: {project_path}", flush=True)
        return project_path

    if args.use_image2scene:
        return _run_image2scene_pipeline(args)

    if args.use_latest_image2scene_gym_project:
        return _resolve_latest_image2scene_gym_config(args)

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


def _resolve_latest_image2scene_gym_config(args: argparse.Namespace) -> Path:
    image2scene_root = Path(args.image2scene_root).expanduser().resolve()
    download_dir = _resolve_under_root(image2scene_root, args.image2scene_download_dir)
    if download_dir is None:
        raise ValueError("--image2scene-download-dir must not be empty.")

    merged_configs = _collect_merged_gym_configs(download_dir)
    if not merged_configs:
        raise FileNotFoundError(
            f"gym_config_merged.json not found under: {download_dir}"
        )

    merged_config = _latest_path(merged_configs)
    print(f"Using latest image2scene merged gym config: {merged_config}", flush=True)
    return merged_config


def _resolve_target_replacements(
    args: argparse.Namespace,
    target_replacement_spec_cls: Callable[..., object],
) -> list[object]:
    replacements = []
    if args.target_replacement1:
        source_uid, prompt = args.target_replacement1
        replacements.append(
            target_replacement_spec_cls(
                source_uid=source_uid,
                prompt=prompt,
                output_dir_name="new1",
            )
        )
    if args.target_replacement2:
        source_uid, prompt = args.target_replacement2
        replacements.append(
            target_replacement_spec_cls(
                source_uid=source_uid,
                prompt=prompt,
                output_dir_name="new2",
            )
        )
    return replacements


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
        TargetReplacementSpec,
        generate_ur5_basket_config_from_project,
    )

    project_path = _resolve_gym_project(args)
    target_replacements = _resolve_target_replacements(args, TargetReplacementSpec)

    paths = generate_ur5_basket_config_from_project(
        gym_project=project_path,
        output_dir=args.config_output_dir,
        task_name=args.task_name,
        task_description=args.task_description,
        target_body_scale=args.target_body_scale,
        target_replacements=target_replacements,
        sync_replacement_names=args.sync_replacement_names,
        overwrite=args.overwrite_config,
    )

    print(f"Using gym project/config: {project_path}", flush=True)
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
