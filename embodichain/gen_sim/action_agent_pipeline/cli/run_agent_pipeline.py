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
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent
    return Path.cwd().resolve()


__all__ = ["main"]

_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_records import (
    find_history_entry_by_index as _records_find_history_entry_by_index,
    history_entry_has_source as _records_history_entry_has_source,
    history_entry_index as _records_history_entry_index,
    path_from_history_entry as _records_path_from_history_entry,
    pipeline_history_path as _records_pipeline_history_path,
    read_pipeline_history as _records_read_pipeline_history,
    resolve_source_gym_config as _records_resolve_source_gym_config,
    write_pipeline_manifests as _records_write_pipeline_manifests,
)

_DEFAULT_SERVER = "http://192.168.3.23:4523"
_DEFAULT_GYM_PROJECT_ROOT = _REPO_ROOT / "gym_project"
_DEFAULT_ACTION_AGENT_WORKSPACE = _DEFAULT_GYM_PROJECT_ROOT / "action_agent_pipeline"
_DEFAULT_IMAGE = _DEFAULT_ACTION_AGENT_WORKSPACE / "images/demo1.jpg"
_DEFAULT_IMAGE_DIR = _DEFAULT_IMAGE.parent
_DEFAULT_EXISTING_GYM_PROJECT = _DEFAULT_GYM_PROJECT_ROOT / "1780562837_gym_project"
_DEFAULT_IMAGE2SCENE_ROOT = _REPO_ROOT / "gym_project/environment/image2tabletop"
_DEFAULT_IMAGE2SCENE_IMAGE = "scene_image/robotwin_example.png"
_DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR = "./downloads"
_DEFAULT_IMAGE2SCENE_OUTPUT_ROOT = "./generated"
_DEFAULT_IMAGE2SCENE_CONFIG = "./gen_config.json"
_DEFAULT_CONFIG_OUTPUT_DIR = _DEFAULT_ACTION_AGENT_WORKSPACE / "configs/demo3_text"
_DEFAULT_PIPELINE_HISTORY = (
    _DEFAULT_ACTION_AGENT_WORKSPACE / "configs/pipeline_history.json"
)
_DEFAULT_TASK_NAME = "Demo3_Text"
_DEFAULT_TASK_TEMPLATE_NAMES = frozenset({"Demo1_Text"})
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_GYM_CONFIG_PREFERENCE = ("gym_config_merged.json", "gym_config.json")
_PIPELINE_HISTORY_SCHEMA_VERSION = 1
_PIPELINE_MANIFEST_FILENAME = "pipeline_manifest.json"
_INDEXED_REPLACEMENT_ALIAS_RE = re.compile(
    r"^(?P<keyword>[A-Za-z][A-Za-z0-9 _-]*?)[ _-]?(?P<index>[0-9]+)$"
)


@dataclass(frozen=True)
class ProjectResolution:
    path: Path
    mode: str
    base_history: dict[str, Any] | None = None


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
        "--base-task-name",
        "--base_task_name",
        dest="base_task_name",
        default=None,
        help=(
            "Start from the latest pipeline history entry with this task name. "
            "Use this to chain demos, e.g. demo2 based on Demo1_Text."
        ),
    )
    parser.add_argument(
        "--base-history-index",
        "--base_history_index",
        dest="base_history_index",
        type=int,
        default=None,
        help=(
            "Start from a specific pipeline history index. When used with "
            "--base-task-name, the history entry must match that task name."
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
        "--pipeline-history-path",
        "--pipeline_history_path",
        dest="pipeline_history_path",
        default=str(_DEFAULT_PIPELINE_HISTORY),
        help=(
            "Global pipeline history JSON path. Defaults to "
            f"{_DEFAULT_PIPELINE_HISTORY.as_posix()}"
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
        help=(
            'Task description passed to config generation. Defaults to "". '
            "Ignored for default-template tasks such as Demo1_Text."
        ),
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
        nargs="+",
        metavar="SOURCE_OR_PROMPT",
        default=None,
        help=(
            "Generate <gym_project>/mesh_assets/new1 from PROMPT. Accepts either "
            "PROMPT, which auto-selects the lower-y duplicated rigid "
            "object, or SOURCE_UID PROMPT for explicit selection."
        ),
    )
    parser.add_argument(
        "--target_replacement2",
        "--target-replacement2",
        nargs="+",
        metavar="SOURCE_OR_PROMPT",
        default=None,
        help=(
            "Generate <gym_project>/mesh_assets/new2 from PROMPT. Accepts either "
            "PROMPT, which auto-selects the higher-y duplicated rigid "
            "object, or SOURCE_UID PROMPT for explicit selection."
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
        "--reuse-target-replacements",
        "--reuse_target_replacements",
        dest="reuse_target_replacements",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing prompt-generated replacement GLBs when the prompt "
            "and expected output name match. Defaults to true."
        ),
    )
    parser.add_argument(
        "--prewarm-coacd-cache",
        "--prewarm_coacd_cache",
        dest="prewarm_coacd_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Precompute environment CoACD cache files during config generation. "
            "Defaults to true."
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
    parser.add_argument(
        "--llm-usage-output",
        default=None,
        help=(
            "JSONL path for local LLM token usage records. Defaults to "
            "<config-output-dir>/llm_usage.jsonl."
        ),
    )
    parser.add_argument(
        "--llm-usage-summary-output",
        default=None,
        help=(
            "JSON path for the aggregated local LLM token usage summary. "
            "Defaults to <config-output-dir>/llm_usage_summary.json."
        ),
    )
    parser.add_argument(
        "--llm-usage-run-id",
        default=None,
        help="Optional run id written into local LLM token usage records.",
    )
    parser.add_argument(
        "--no-llm-usage",
        dest="llm_usage",
        action="store_false",
        default=True,
        help="Disable local LLM token usage recording for this pipeline run.",
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


def _image2scene_subprocess_env() -> dict[str, str]:
    from embodichain.gen_sim.action_agent_pipeline.utils.llm_config import (
        get_openai_compatible_llm_config,
    )
    from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
        scrub_usage_tracking_env,
    )

    env = scrub_usage_tracking_env()
    cfg = get_openai_compatible_llm_config(
        required=False,
        require_base_url=False,
    )
    env_overrides = {
        "OPENAI_API_KEY": cfg.get("api_key"),
        "OPENAI_MODEL": cfg.get("model"),
        "OPENAI_BASE_URL": cfg.get("base_url"),
        "EMBODICHAIN_LLM_PROXY": cfg.get("proxy_url"),
    }
    for name, value in env_overrides.items():
        if value:
            env[name] = str(value)

    if cfg.get("model") or cfg.get("base_url"):
        print(
            "Using shared LLM config for image2scene subprocess: "
            f"model={cfg.get('model')!r}, base_url={cfg.get('base_url')!r}",
            flush=True,
        )
    return env


def _resolve_task_description_for_generation(args: argparse.Namespace) -> str | None:
    task_description = str(args.task_description or "").strip()
    if args.task_name in _DEFAULT_TASK_TEMPLATE_NAMES:
        if task_description:
            print(
                f"Ignoring --task_description for {args.task_name}; "
                "using the default basket task template.",
                flush=True,
            )
        return None
    return task_description or None


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
    completed = subprocess.run(
        command,
        cwd=image2scene_root,
        check=False,
        env=_image2scene_subprocess_env(),
    )
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


def _resolve_gym_project(args: argparse.Namespace) -> ProjectResolution:
    use_history = args.base_task_name is not None or args.base_history_index is not None
    selected_modes = [
        args.use_image2scene,
        args.use_existing_gym_project,
        use_history,
    ]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise ValueError(
            "Use only one of --use-image2scene, --use-existing-gym-project, "
            "or --base-task-name/--base-history-index."
        )

    if args.use_existing_gym_project:
        project_path = Path(args.gym_project).expanduser().resolve()
        if not project_path.exists():
            raise FileNotFoundError(f"gym project not found: {project_path}")
        print(f"Using existing gym project: {project_path}", flush=True)
        return ProjectResolution(path=project_path, mode="existing_gym_project")

    if args.use_image2scene:
        return ProjectResolution(
            path=_run_image2scene_pipeline(args), mode="image2scene"
        )

    if use_history:
        history_entry = _resolve_base_history_entry(args)
        project_path = _path_from_history_entry(history_entry)
        print(
            "Using base history "
            f"#{history_entry.get('index')} ({history_entry.get('task_name')}): "
            f"{project_path}",
            flush=True,
        )
        return ProjectResolution(
            path=project_path,
            mode="history",
            base_history=history_entry,
        )

    from embodichain.gen_sim.action_agent_pipeline.gym_project_api.image2tabletop_client import (
        check_health,
        collect_image_paths,
        process_image,
    )

    image_input = _resolve_image_input(args)
    image_path = _resolve_single_image(str(image_input), collect_image_paths)
    if not args.skip_health_check:
        check_health(args.server)

    return ProjectResolution(
        path=process_image(
            server=args.server,
            image_path=image_path,
            output_root=Path(args.gym_project_root),
            poll_interval=args.poll_interval,
            overwrite=args.overwrite_gym_project,
        ),
        mode="image2tabletop",
    )


def _resolve_base_history_entry(args: argparse.Namespace) -> dict[str, Any]:
    if args.base_history_index is not None and args.base_history_index <= 0:
        raise ValueError("--base-history-index must be a positive integer.")

    history_path = _pipeline_history_path(args)
    history = _read_pipeline_history(history_path)
    runs = history["runs"]

    if args.base_history_index is not None:
        entry = _find_history_entry_by_index(runs, args.base_history_index)
        if entry is None:
            raise ValueError(
                f"Pipeline history index not found: {args.base_history_index}"
            )
        if args.base_task_name and entry.get("task_name") != args.base_task_name:
            raise ValueError(
                "Pipeline history entry "
                f"#{args.base_history_index} has task_name={entry.get('task_name')!r}, "
                f"expected {args.base_task_name!r}."
            )
        return dict(entry)

    if not args.base_task_name:
        raise ValueError("--base-task-name is required without --base-history-index.")

    candidates = [
        entry
        for entry in runs
        if entry.get("task_name") == args.base_task_name
        and _history_entry_has_source(entry)
    ]
    if not candidates:
        raise ValueError(
            "No pipeline history entry found for task_name="
            f"{args.base_task_name!r} in {history_path}"
        )
    return dict(max(candidates, key=_history_entry_index))


def _pipeline_history_path(args: argparse.Namespace) -> Path:
    return _records_pipeline_history_path(args)


def _read_pipeline_history(history_path: Path) -> dict[str, Any]:
    return _records_read_pipeline_history(
        history_path,
        schema_version=_PIPELINE_HISTORY_SCHEMA_VERSION,
    )


def _find_history_entry_by_index(
    runs: list[Any], history_index: int
) -> dict[str, Any] | None:
    return _records_find_history_entry_by_index(runs, history_index)


def _history_entry_index(entry: dict[str, Any]) -> int:
    return _records_history_entry_index(entry)


def _history_entry_has_source(entry: dict[str, Any]) -> bool:
    return _records_history_entry_has_source(entry)


def _path_from_history_entry(entry: dict[str, Any]) -> Path:
    return _records_path_from_history_entry(entry, repo_root=_REPO_ROOT)


def _resolve_target_replacements(
    args: argparse.Namespace,
    target_replacement_spec_cls: Callable[..., object],
    gym_project: Path,
) -> list[object]:
    replacements = []
    alias_config = None
    if args.target_replacement1:
        alias_config = alias_config or _load_replacement_alias_config(gym_project)
        source_uid, prompt = _resolve_target_replacement_arg(
            args.target_replacement1,
            alias_config,
            option_name="--target_replacement1",
            replacement_number=1,
        )
        replacements.append(
            target_replacement_spec_cls(
                source_uid=source_uid,
                prompt=prompt,
                output_dir_name="new1",
            )
        )
    if args.target_replacement2:
        alias_config = alias_config or _load_replacement_alias_config(gym_project)
        source_uid, prompt = _resolve_target_replacement_arg(
            args.target_replacement2,
            alias_config,
            option_name="--target_replacement2",
            replacement_number=2,
        )
        replacements.append(
            target_replacement_spec_cls(
                source_uid=source_uid,
                prompt=prompt,
                output_dir_name="new2",
            )
        )
    return replacements


def _resolve_target_replacement_arg(
    values: list[str],
    gym_config: dict[str, Any],
    *,
    option_name: str,
    replacement_number: int,
) -> tuple[str, str]:
    if len(values) == 1:
        prompt = str(values[0]).strip()
        if not prompt:
            raise ValueError(f"{option_name} prompt must be non-empty.")
        source_uid = _auto_replacement_source_uid(
            gym_config,
            replacement_number=replacement_number,
            option_name=option_name,
        )
        return source_uid, prompt

    if len(values) == 2:
        source_uid, prompt = values
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError(f"{option_name} prompt must be non-empty.")
        source_uid = _resolve_replacement_source_uid(
            source_uid,
            gym_config,
            option_name=option_name,
        )
        return source_uid, prompt

    raise ValueError(
        f"{option_name} expects either PROMPT or SOURCE_UID PROMPT, got "
        f"{len(values)} values: {values!r}. Quote multi-word prompts."
    )


def _load_replacement_alias_config(gym_project: Path) -> dict[str, Any]:
    config_path = _resolve_replacement_alias_gym_config(gym_project)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Gym config must be a JSON object: {config_path}")
    return data


def _resolve_replacement_alias_gym_config(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        sibling_gym_config = input_path.parent / "gym_config.json"
        if sibling_gym_config.is_file():
            return sibling_gym_config.resolve()
        return _resolve_source_gym_config(input_path)

    direct_gym_config = input_path / "gym_config.json"
    if direct_gym_config.is_file():
        return direct_gym_config.resolve()

    source_config = _resolve_source_gym_config(input_path)
    sibling_gym_config = source_config.parent / "gym_config.json"
    if sibling_gym_config.is_file():
        return sibling_gym_config.resolve()
    return source_config


def _auto_replacement_source_uid(
    gym_config: dict[str, Any],
    *,
    replacement_number: int,
    option_name: str,
) -> str:
    if replacement_number not in {1, 2}:
        raise ValueError(f"Unsupported replacement number: {replacement_number}")

    duplicate_groups = _duplicated_numbered_rigid_object_groups(gym_config)
    if len(duplicate_groups) != 1:
        candidates = _format_duplicate_group_candidates(duplicate_groups)
        raise ValueError(
            f"{option_name} was given without an explicit source uid, so the "
            "pipeline expected exactly one duplicated numbered rigid_object "
            f"group in gym_config.json. Found {len(duplicate_groups)} group(s): "
            f"{candidates}. Use SOURCE_UID PROMPT to disambiguate."
        )

    base_name, positioned_objects = duplicate_groups[0]
    if len(positioned_objects) != 2:
        candidates = _format_duplicate_group_candidates(duplicate_groups)
        raise ValueError(
            f"{option_name} auto-selection requires exactly two objects in the "
            f"duplicated group {base_name!r}, found {len(positioned_objects)}: "
            f"{candidates}. Use SOURCE_UID PROMPT to disambiguate."
        )

    if (
        abs(float(positioned_objects[0]["y"]) - float(positioned_objects[1]["y"]))
        < 1e-9
    ):
        candidates = _format_duplicate_group_candidates(duplicate_groups)
        raise ValueError(
            f"{option_name} auto-selection requires distinct y coordinates in "
            f"duplicated group {base_name!r}: {candidates}. Use SOURCE_UID PROMPT "
            "to disambiguate."
        )

    selected = positioned_objects[replacement_number - 1]
    source_uid = selected["object"]["uid"]
    print(
        f"Resolved {option_name} auto source -> {source_uid!r} "
        f"from duplicated rigid_object group {base_name!r} by y={selected['y']}",
        flush=True,
    )
    return source_uid


def _duplicated_numbered_rigid_object_groups(
    gym_config: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in _rigid_objects(gym_config):
        parsed = _parse_numbered_rigid_object_uid(obj["uid"])
        if parsed is None:
            continue
        base_name, number = parsed
        grouped.setdefault(base_name, []).append(
            {
                "number": number,
                "y": _rigid_object_y_coordinate(obj),
                "object": obj,
            }
        )

    duplicate_groups = []
    for base_name, entries in grouped.items():
        if len(entries) < 2:
            continue
        duplicate_groups.append(
            (
                base_name,
                sorted(
                    entries,
                    key=lambda entry: (
                        -float(entry["y"]),
                        str(entry["object"]["uid"]),
                    ),
                ),
            )
        )
    return sorted(duplicate_groups, key=lambda item: item[0])


def _parse_numbered_rigid_object_uid(uid: str) -> tuple[str, int] | None:
    match = re.match(r"^(?P<base>.+?)[_-]?(?P<number>[0-9]+)$", uid)
    if match is None:
        return None
    base_name = match.group("base").strip("_-")
    if not base_name:
        return None
    return base_name, int(match.group("number"))


def _rigid_object_y_coordinate(obj: dict[str, Any]) -> float:
    init_pos = obj.get("init_pos")
    if not isinstance(init_pos, (list, tuple)) or len(init_pos) < 2:
        raise ValueError(
            "Auto replacement source selection requires each duplicated "
            f"rigid_object to define init_pos with a y value, got {obj.get('uid')!r}."
        )
    try:
        return float(init_pos[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Auto replacement source selection requires numeric init_pos[1], "
            f"got {obj.get('uid')!r}: {init_pos[1]!r}"
        ) from exc


def _format_duplicate_group_candidates(
    groups: list[tuple[str, list[dict[str, Any]]]],
) -> str:
    if not groups:
        return "<none>"
    parts = []
    for base_name, entries in groups:
        values = ", ".join(
            f"{entry['object']['uid']}#number={entry['number']},y={entry['y']}"
            for entry in entries
        )
        parts.append(f"{base_name}: {values}")
    return "; ".join(parts)


def _resolve_replacement_source_uid(
    source_input: str,
    gym_config: dict[str, Any],
    *,
    option_name: str,
) -> str:
    source_input = str(source_input).strip()
    rigid_objects = _rigid_objects(gym_config)
    by_uid = {obj["uid"]: obj for obj in rigid_objects}
    if source_input in by_uid:
        return source_input

    alias = _parse_indexed_replacement_alias(source_input)
    if alias is None:
        candidates = _format_rigid_object_candidates(rigid_objects)
        raise ValueError(
            f"{option_name} source {source_input!r} is neither a rigid object uid "
            f"nor an indexed alias such as bread1. Rigid object candidates: "
            f"{candidates}"
        )

    keyword, alias_index = alias
    matches = [
        obj for obj in rigid_objects if _rigid_object_matches_keyword(obj, keyword)
    ]
    if alias_index > len(matches):
        candidates = _format_rigid_object_candidates(matches or rigid_objects)
        raise ValueError(
            f"{option_name} alias {source_input!r} requested match #{alias_index} "
            f"for keyword {keyword!r}, but only found {len(matches)} match(es). "
            f"Candidates: {candidates}"
        )

    resolved_uid = matches[alias_index - 1]["uid"]
    print(
        f"Resolved {option_name} source alias {source_input!r} -> {resolved_uid!r}",
        flush=True,
    )
    return resolved_uid


def _rigid_objects(gym_config: dict[str, Any]) -> list[dict[str, Any]]:
    value = gym_config.get("rigid_object", [])
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("gym config rigid_object must be a list or object.")

    rigid_objects = []
    for obj in value:
        if not isinstance(obj, dict):
            continue
        uid = str(obj.get("uid", "")).strip()
        if not uid:
            continue
        copied = dict(obj)
        copied["uid"] = uid
        rigid_objects.append(copied)
    if not rigid_objects:
        raise ValueError("No rigid_object entries found in gym config.")
    return rigid_objects


def _parse_indexed_replacement_alias(alias: str) -> tuple[str, int] | None:
    match = _INDEXED_REPLACEMENT_ALIAS_RE.match(alias.strip())
    if match is None:
        return None
    keyword = match.group("keyword").strip(" _-")
    index = int(match.group("index"))
    if not keyword or index < 1:
        return None
    return keyword, index


def _rigid_object_matches_keyword(obj: dict[str, Any], keyword: str) -> bool:
    keyword_tokens = _search_tokens(keyword)
    if not keyword_tokens:
        return False
    object_tokens = set(_search_tokens(_rigid_object_search_text(obj)))
    return all(token in object_tokens for token in keyword_tokens)


def _rigid_object_search_text(obj: dict[str, Any]) -> str:
    values = [
        obj.get("uid", ""),
        obj.get("source_uid", ""),
        obj.get("category", ""),
        obj.get("semantic_label", ""),
        obj.get("name", ""),
        obj.get("description", ""),
    ]
    shape = obj.get("shape", {})
    if isinstance(shape, dict):
        values.extend(
            [
                shape.get("fpath", ""),
                shape.get("file_path", ""),
                shape.get("category", ""),
            ]
        )
    return " ".join(str(value) for value in values if value)


def _search_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value).lower())


def _format_rigid_object_candidates(rigid_objects: list[dict[str, Any]]) -> str:
    if not rigid_objects:
        return "<none>"
    parts = []
    for obj in rigid_objects:
        shape = obj.get("shape", {})
        fpath = shape.get("fpath", "") if isinstance(shape, dict) else ""
        parts.append(f"{obj.get('uid')} ({fpath})")
    return ", ".join(parts)


def _write_pipeline_manifests(
    *,
    args: argparse.Namespace,
    resolution: ProjectResolution,
    generated_paths: object,
    target_replacements: list[object],
) -> dict[str, Any]:
    return _records_write_pipeline_manifests(
        args=args,
        resolution=resolution,
        generated_paths=generated_paths,
        target_replacements=target_replacements,
        repo_root=_REPO_ROOT,
        schema_version=_PIPELINE_HISTORY_SCHEMA_VERSION,
        manifest_filename=_PIPELINE_MANIFEST_FILENAME,
    )


def _resolve_source_gym_config(input_path: Path) -> Path:
    return _records_resolve_source_gym_config(
        input_path,
        gym_config_preference=_GYM_CONFIG_PREFERENCE,
    )


def _configure_llm_usage_tracking(
    args: argparse.Namespace,
) -> tuple[Path, Path] | None:
    if not args.llm_usage:
        from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
            disable_usage_tracking,
        )

        disable_usage_tracking()
        return None

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
        configure_usage_tracking,
    )

    output_dir = Path(args.config_output_dir).expanduser().resolve()
    usage_path = (
        Path(args.llm_usage_output).expanduser().resolve()
        if args.llm_usage_output
        else output_dir / "llm_usage.jsonl"
    )
    summary_path = (
        Path(args.llm_usage_summary_output).expanduser().resolve()
        if args.llm_usage_summary_output
        else output_dir / "llm_usage_summary.json"
    )
    run_id = args.llm_usage_run_id or (
        f"{args.task_name}_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    configure_usage_tracking(
        usage_path=usage_path,
        run_id=run_id,
        process_name="run_agent_pipeline",
        reset=True,
    )
    print(f"Recording local LLM token usage: {usage_path}", flush=True)
    print(f"Local LLM token usage summary: {summary_path}", flush=True)
    return usage_path, summary_path


def _write_llm_usage_summary(usage_paths: tuple[Path, Path] | None) -> None:
    if usage_paths is None:
        return

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
        write_usage_summary,
    )

    usage_path, summary_path = usage_paths
    summary = write_usage_summary(
        usage_path=usage_path,
        summary_path=summary_path,
    )
    total = summary["total"]
    print(
        "Local LLM token usage total: "
        f"calls={total['calls']}, "
        f"input={total['input_tokens']}, "
        f"output={total['output_tokens']}, "
        f"total={total['total_tokens']}",
        flush=True,
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

    env = os.environ.copy()
    if env.get("EMBODICHAIN_LLM_USAGE_PATH"):
        env["EMBODICHAIN_LLM_USAGE_PROCESS"] = "run_agent"

    print("Running task:")
    print(shlex.join(command), flush=True)
    return subprocess.run(command, check=False, env=env).returncode


def main() -> int:
    args = _build_parser().parse_args()

    _ensure_repo_on_pythonpath()
    from embodichain.gen_sim.action_agent_pipeline.generation.ur5_basket_config import (
        TargetReplacementSpec,
        generate_ur5_basket_config_from_project,
    )

    resolution = _resolve_gym_project(args)
    usage_paths = _configure_llm_usage_tracking(args)
    target_replacements = _resolve_target_replacements(
        args,
        TargetReplacementSpec,
        resolution.path,
    )
    task_description = _resolve_task_description_for_generation(args)
    args.task_description = task_description or ""

    paths = generate_ur5_basket_config_from_project(
        gym_project=resolution.path,
        output_dir=args.config_output_dir,
        task_name=args.task_name,
        task_description=task_description,
        target_body_scale=args.target_body_scale,
        target_replacements=target_replacements,
        sync_replacement_names=args.sync_replacement_names,
        reuse_target_replacements=args.reuse_target_replacements,
        prewarm_coacd_cache=args.prewarm_coacd_cache,
        overwrite=args.overwrite_config,
    )
    _write_pipeline_manifests(
        args=args,
        resolution=resolution,
        generated_paths=paths,
        target_replacements=target_replacements,
    )

    print(f"Using gym project/config: {resolution.path}", flush=True)
    print(f"Generated gym config: {paths.gym_config}", flush=True)
    print(f"Generated agent config: {paths.agent_config}", flush=True)
    if args.skip_run_agent:
        _write_llm_usage_summary(usage_paths)
        return 0

    return_code = _run_agent_command(
        task_name=args.task_name,
        gym_config=paths.gym_config,
        agent_config=paths.agent_config,
        regenerate=args.regenerate,
    )
    _write_llm_usage_summary(usage_paths)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
