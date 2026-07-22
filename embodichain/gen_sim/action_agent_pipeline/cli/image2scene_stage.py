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
import os
from pathlib import Path
import shlex
import subprocess
import sys

from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_defaults import (
    DEFAULT_IMAGE2SCENE_IMAGE,
    DEFAULT_IMAGE_DIR,
    IMAGE_SUFFIXES,
)
from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
    scrub_usage_tracking_env,
)

__all__ = [
    "collect_merged_gym_configs",
    "latest_path",
    "resolve_image2tabletop_server",
    "resolve_under_root",
    "run_image2scene_pipeline",
]


def resolve_under_root(root: Path, path_input: str | None) -> Path | None:
    if path_input is None:
        return None
    path = Path(path_input).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def collect_merged_gym_configs(download_dir: Path) -> list[Path]:
    if not download_dir.exists():
        return []
    return sorted(download_dir.rglob("gym_config_merged.json"))


def latest_path(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime)


def run_image2scene_pipeline(args: argparse.Namespace) -> Path:
    if not args.background:
        raise ValueError("--background is required with --use-image2scene.")

    image2scene_root = Path(args.image2scene_root).expanduser().resolve()
    if not image2scene_root.is_dir():
        raise FileNotFoundError(f"image2scene root not found: {image2scene_root}")

    script_path = image2scene_root / "demo_api/client/image2scene_pipeline.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"image2scene pipeline not found: {script_path}")

    image_path = _resolve_image2scene_image(args, image2scene_root)
    download_dir = resolve_under_root(image2scene_root, args.image2scene_download_dir)
    output_root = resolve_under_root(image2scene_root, args.image2scene_output_root)
    gen_config = resolve_under_root(image2scene_root, args.image2scene_gen_config)
    llm_config = resolve_under_root(image2scene_root, args.image2scene_llm_config)
    extract_dir = resolve_under_root(image2scene_root, args.image2scene_extract_dir)
    merged_output = resolve_under_root(image2scene_root, args.image2scene_merged_output)

    if (
        download_dir is None
        or output_root is None
        or gen_config is None
        or llm_config is None
    ):
        raise ValueError("image2scene paths must not be empty.")

    before_configs = set(collect_merged_gym_configs(download_dir))
    server = resolve_image2tabletop_server(args)
    client_url = resolve_image2scene_client_url(args, server)
    runtime_gen_config = _stage_b_gen_config_with_client_url(
        gen_config,
        client_url,
        output_root,
    )
    command = [
        sys.executable,
        str(script_path),
        "--server",
        server,
        "--image",
        str(image_path),
        "--download-dir",
        str(download_dir),
        "--background",
        args.background,
        "--output-root",
        str(output_root),
        "--gen-config",
        str(runtime_gen_config),
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

    after_configs = collect_merged_gym_configs(download_dir)
    new_configs = [path for path in after_configs if path not in before_configs]
    if new_configs:
        merged_config = latest_path(new_configs)
    elif after_configs:
        merged_config = latest_path(after_configs)
    else:
        raise FileNotFoundError(
            f"gym_config_merged.json not found under: {download_dir}"
        )

    print(f"Using image2scene merged gym config: {merged_config}", flush=True)
    return merged_config


def resolve_image2tabletop_server(args: argparse.Namespace) -> str:
    server = str(args.server or os.getenv("IMAGE2TABLETOP_SERVER") or "").strip()
    if not server:
        raise ValueError(
            "Image2Tabletop API server is required for this mode. Pass --server "
            "or set IMAGE2TABLETOP_SERVER."
        )
    return server.rstrip("/")


def resolve_image2scene_client_url(args: argparse.Namespace, server: str) -> str:
    client_url = str(getattr(args, "image2scene_client_url", "") or "").strip()
    if client_url:
        return client_url.rstrip("/")
    return server.rstrip("/")


def _stage_b_gen_config_with_client_url(
    gen_config: Path,
    client_url: str,
    output_root: Path,
) -> Path:
    normalized_client_url = str(client_url or "").strip().rstrip("/")
    if not normalized_client_url:
        return gen_config

    config = json.loads(gen_config.read_text(encoding="utf-8"))
    if config.get("DEFAULT_CLIENT_URL") == normalized_client_url:
        return gen_config

    runtime_dir = output_root / ".image2scene_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_dir / "gen_config.json"
    config["DEFAULT_CLIENT_URL"] = normalized_client_url
    runtime_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return runtime_config


def _image2scene_subprocess_env() -> dict[str, str]:
    env = scrub_usage_tracking_env(os.environ)
    try:
        from embodichain.gen_sim.action_agent_pipeline.utils.llm_config import (
            get_openai_compatible_llm_config,
        )
    except Exception:
        return env

    cfg = get_openai_compatible_llm_config(
        required=False,
        require_base_url=False,
    )
    env_updates = {
        "IMAGE2TABLETOP_LLM_API_KEY": cfg.get("api_key"),
        "IMAGE2TABLETOP_LLM_MODEL": cfg.get("model"),
        "IMAGE2TABLETOP_LLM_BASE_URL": cfg.get("base_url"),
    }
    for key, value in env_updates.items():
        if value and key not in env:
            env[key] = str(value)
    if cfg.get("api_key"):
        print(
            "Using shared LLM config for image2scene subprocess: "
            f"model={cfg.get('model') or '<default>'}",
            flush=True,
        )
    return env


def _resolve_image2scene_image(
    args: argparse.Namespace,
    image2scene_root: Path,
) -> Path:
    if args.image_name:
        image_path = Path(args.image_name).expanduser()
        if image_path.suffix:
            if not image_path.is_absolute() and image_path.parent == Path("."):
                return (DEFAULT_IMAGE_DIR / image_path).resolve()
            return image_path.resolve()
        candidates = [
            DEFAULT_IMAGE_DIR / f"{args.image_name}{suffix}"
            for suffix in IMAGE_SUFFIXES
        ]
        existing = [path for path in candidates if path.is_file()]
        if existing:
            return existing[0].resolve()
        searched = ", ".join(path.as_posix() for path in candidates)
        raise FileNotFoundError(
            f"Image name '{args.image_name}' was not found. Tried: {searched}"
        )

    image_input = args.image or DEFAULT_IMAGE2SCENE_IMAGE
    image_path = Path(image_input).expanduser()
    if image_path.is_absolute():
        return image_path.resolve()
    return (image2scene_root / image_path).resolve()
