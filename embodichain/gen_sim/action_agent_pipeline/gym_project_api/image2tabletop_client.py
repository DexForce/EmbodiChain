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

"""Client for the Image2Tabletop API."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from requests import exceptions as request_exceptions

__all__ = [
    "check_health",
    "collect_image_paths",
    "download_zip",
    "extract_gym_project",
    "main",
    "process_image",
    "submit_job",
    "wait_for_job",
]

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
_PROJECT_NAME_RE = re.compile(r"^[0-9]+_gym_project$")
_PROJECT_ID_RE = re.compile(r"Image2Tabletop-([0-9]+)-v[0-9]+")
_DEFAULT_JOB_TIMEOUT_S = 1800.0


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent
    return Path.cwd().resolve()


_REPO_ROOT = _repo_root()
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "gym_project"
_DEFAULT_IMAGE_INPUT = _DEFAULT_OUTPUT_ROOT / "action_agent_pipeline/images"


def _require_server(server: str | None) -> str:
    resolved = (
        str(server or os.getenv("IMAGE2TABLETOP_SERVER") or "").strip().rstrip("/")
    )
    if not resolved:
        raise ValueError(
            "Image2Tabletop API server is required. Pass --server or set "
            "IMAGE2TABLETOP_SERVER."
        )
    return resolved


def _server_url(base_url: str, path: str) -> str:
    return f"{_require_server(base_url)}{path}"


def check_health(server: str) -> None:
    try:
        response = requests.get(_server_url(server, "/health"), timeout=10)
    except request_exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"cannot connect to Image2Tabletop demo API: {server}. "
            "Start the server with: "
            "python demo_api/server/image2tabletop_api.py --host 0.0.0.0 --port 4523"
        ) from exc
    response.raise_for_status()


def submit_job(server: str, image_path: Path) -> str:
    try:
        with image_path.open("rb") as image_file:
            response = requests.post(
                _server_url(server, "/api/image2tabletop/start"),
                files={"image": (image_path.name, image_file)},
                timeout=60,
            )
    except request_exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"cannot connect to API server: {server}. "
            "Make sure the server is running and listening on this host/port."
        ) from exc
    response.raise_for_status()
    data = response.json()
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"API response does not contain job_id: {data}")
    return str(job_id)


def wait_for_job(
    server: str,
    job_id: str,
    poll_interval: float,
    timeout_s: float = _DEFAULT_JOB_TIMEOUT_S,
) -> dict:
    status_url = _server_url(server, f"/api/image2tabletop/status/{job_id}")
    deadline = time.monotonic() + timeout_s
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError(
                f"job {job_id} did not complete within {timeout_s}s: {status_url}"
            )
        response = requests.get(status_url, timeout=min(30, max(0.001, remaining_s)))
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        print(f"[{time.strftime('%H:%M:%S')}] job={job_id} status={status}", flush=True)
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"job failed: {data}")
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def download_zip(server: str, job_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{job_id}_formatted_tabletop_scene.zip"
    response = requests.get(
        _server_url(server, f"/api/image2tabletop/download/{job_id}"),
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    with zip_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
    return zip_path


def collect_image_paths(image_input: Path) -> list[Path]:
    image_input = image_input.expanduser().resolve()
    if image_input.is_file():
        if image_input.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {image_input}")
        return [image_input]
    if image_input.is_dir():
        image_paths = sorted(
            path
            for path in image_input.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if image_paths:
            return image_paths
        raise FileNotFoundError(f"no supported image files found under: {image_input}")
    raise FileNotFoundError(f"image input not found: {image_input}")


def extract_gym_project(
    zip_path: Path, output_root: Path, job_id: str, overwrite: bool
) -> Path:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=f"{job_id}_image2tabletop_") as temp_dir_name:
        extract_dir = Path(temp_dir_name).resolve()
        _safe_extract_zip(zip_path, extract_dir)
        gym_config_paths = sorted(extract_dir.rglob("gym_config.json"))
        if not gym_config_paths:
            raise FileNotFoundError(
                f"gym_config.json not found in downloaded archive: {zip_path}"
            )
        if len(gym_config_paths) > 1:
            matches = ", ".join(path.as_posix() for path in gym_config_paths)
            raise ValueError(
                f"multiple gym_config.json files found in archive: {matches}"
            )

        gym_config_path = gym_config_paths[0]
        project_name = _infer_project_name(gym_config_path, extract_dir, job_id)
        source_root = _infer_source_project_root(
            gym_config_path, extract_dir, project_name
        )
        destination = output_root / project_name
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"output project already exists: {destination}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(destination)
        shutil.copytree(source_root, destination)
        return destination


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target_path = (extract_dir / member.filename).resolve()
            if not target_path.is_relative_to(extract_dir):
                raise RuntimeError(f"unsafe archive member path: {member.filename}")
        archive.extractall(extract_dir)


def _infer_project_name(gym_config_path: Path, extract_dir: Path, job_id: str) -> str:
    for part in gym_config_path.relative_to(extract_dir).parts:
        if _PROJECT_NAME_RE.match(part):
            return part

    try:
        config = json.loads(gym_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        config = {}
    project_id = str(config.get("id", ""))
    match = _PROJECT_ID_RE.match(project_id)
    if match:
        return f"{match.group(1)}_gym_project"
    return f"{job_id}_gym_project"


def _infer_source_project_root(
    gym_config_path: Path, extract_dir: Path, project_name: str
) -> Path:
    current = extract_dir
    for part in gym_config_path.relative_to(extract_dir).parts:
        current = current / part
        if part == project_name:
            return current
    return gym_config_path.parent


def process_image(
    server: str,
    image_path: Path,
    output_root: Path,
    poll_interval: float,
    overwrite: bool,
    job_timeout_s: float = _DEFAULT_JOB_TIMEOUT_S,
) -> Path:
    job_id = submit_job(server, image_path)
    print(f"submitted job: {job_id} image={image_path}", flush=True)
    wait_for_job(server, job_id, poll_interval, timeout_s=job_timeout_s)
    with TemporaryDirectory(
        prefix=f"{job_id}_image2tabletop_download_"
    ) as temp_dir_name:
        zip_path = download_zip(server, job_id, Path(temp_dir_name))
        project_path = extract_gym_project(zip_path, output_root, job_id, overwrite)
    print(f"generated gym project: {project_path}", flush=True)
    return project_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit image files to Image2Tabletop API."
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Image2Tabletop demo API server. Defaults to IMAGE2TABLETOP_SERVER.",
    )
    parser.add_argument(
        "--image",
        default=str(_DEFAULT_IMAGE_INPUT),
        help=(
            "Input image file or directory. Defaults to "
            f"{_DEFAULT_IMAGE_INPUT.as_posix()}"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"Directory where generated gym projects are written. Defaults to {_DEFAULT_OUTPUT_ROOT.as_posix()}",
    )
    parser.add_argument(
        "--download-dir",
        dest="output_root",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--job-timeout-s", type=float, default=_DEFAULT_JOB_TIMEOUT_S)
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        default=False,
        help="Skip GET /health before submitting images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Replace an existing generated gym project with the same name.",
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(Path(args.image))
    server = _require_server(args.server)
    if not args.skip_health_check:
        check_health(server)

    project_paths = []
    for image_path in image_paths:
        project_paths.append(
            process_image(
                server=server,
                image_path=image_path,
                output_root=Path(args.output_root or _DEFAULT_OUTPUT_ROOT),
                poll_interval=args.poll_interval,
                overwrite=args.overwrite,
                job_timeout_s=args.job_timeout_s,
            )
        )

    print("gym_project paths:", flush=True)
    for project_path in project_paths:
        print(project_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
