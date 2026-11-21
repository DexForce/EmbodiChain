# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Download the pinned RoboCasa365 inputs for a selected replay episode."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import venv
import zlib

__all__: list[str] = []

_INDEX_BYTES = Path(__file__).with_suffix(".json").read_bytes()
_INDEX = json.loads(_INDEX_BYTES)
_TASKS = _INDEX["tasks"]
_BUNDLE_VERSION = f"v1-{hashlib.sha256(_INDEX_BYTES).hexdigest()[:12]}"
_CONVERTER_REQUIREMENTS = (
    "mujoco-usd-converter==0.10.0",
    "mujoco-core==4.0.1",
    "newton-usd-schemas==0.7.0",
    "usd-exchange==3.0.1",
)


def _download(
    url: str, *, size: int, offset: int | None = None, total: int | None = None
) -> bytes:
    headers = {"Accept-Encoding": "identity"}
    expected_range = None
    if offset is not None:
        end = offset + size - 1
        expected_range = f"bytes {offset}-{end}/{total}"
        headers["Range"] = f"bytes={offset}-{end}"
        # Use distinct cache keys for ranges on the upstream CDNs.
        url = f"{url}&range={offset}-{end}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=45) as response:
                if expected_range is not None and (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected_range
                ):
                    raise ValueError(
                        f"Unexpected response for {expected_range} at {url}"
                    )
                data = response.read(size + 1)
            if len(data) != size:
                raise OSError(
                    f"Expected {size} bytes from {url}, received {len(data)}."
                )
            return data
        except HTTPError as exc:
            if exc.code < 500 and exc.code not in (408, 429):
                raise
            if attempt == 2:
                raise
        except (OSError, IncompleteRead):
            if attempt == 2:
                raise
    raise AssertionError("Download retry loop did not return or raise.")


def _check_bytes(name: str, data: bytes, expected: str) -> bytes:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for RoboCasa365 source member {name}.")
    return data


def _range_groups(
    members: dict[str, dict[str, Any]],
) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    group: list[tuple[str, dict[str, Any]]] = []
    start = end = 0
    for name, member in sorted(members.items(), key=lambda item: item[1]["offset"]):
        offset = member["offset"]
        member_end = offset + member["size"]
        if group and (offset - end > 64 * 1024 or member_end - start > 4 * 1024**2):
            yield group
            group = []
        if not group:
            start = offset
        group.append((name, member))
        end = member_end
    if group:
        yield group


def _read_range(
    source: dict[str, Any], group: list[tuple[str, dict[str, Any]]]
) -> Iterator[tuple[str, bytes]]:
    start = group[0][1]["offset"]
    last = group[-1][1]
    data = _download(
        source["url"],
        offset=start,
        size=last["offset"] + last["size"] - start,
        total=source["size"],
    )
    for name, member in group:
        offset = member["offset"] - start
        payload = data[offset : offset + member["size"]]
        if member.get("deflated"):
            payload = zlib.decompress(payload, -zlib.MAX_WBITS)
        yield name, _check_bytes(name, payload, member["sha256"])


def _cache_assets(
    names: set[str], asset_root: Path, staging: Path, pool: ThreadPoolExecutor
) -> None:
    def publish(name: str, data: bytes) -> None:
        temporary = staging / name
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data)
        target = asset_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)

    def archive_group(
        source: dict[str, Any], group: list[tuple[str, dict[str, Any]]]
    ) -> None:
        for name, data in _read_range(source, group):
            publish(name, data)

    def robot_file(name: str, member: dict[str, Any]) -> None:
        relative = quote(name.removeprefix("robosuite/"))
        url = _INDEX["archives"]["robosuite"]["url"] + relative
        data = _download(url, size=member["size"])
        publish(name, _check_bytes(name, data, member["sha256"]))

    missing: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for name in sorted(names):
        member = _INDEX["assets"][name]
        target = asset_root / name
        if target.is_file():
            with target.open("rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    == member["sha256"]
                ):
                    continue
        missing[member["source"]][name] = member
    jobs = [
        pool.submit(robot_file, name, member)
        for name, member in missing.pop("robosuite", {}).items()
    ]
    for archive, members in missing.items():
        jobs.extend(
            pool.submit(archive_group, _INDEX["archives"][archive], group)
            for group in _range_groups(members)
        )
    for job in jobs:
        job.result()


def _converter(cache: Path) -> Path:
    path = cache / f"converter-py{sys.version_info.major}{sys.version_info.minor}"
    python = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = path / ".installed"
    if not (python.is_file() and marker.is_file()):
        print(
            "Installing the pinned USD converter in the replay cache.", file=sys.stderr
        )
        venv.EnvBuilder(with_pip=True, clear=True).create(path)
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", *_CONVERTER_REQUIREMENTS],
            check=True,
        )
        marker.touch()
    return python


def _cached_episode(task: str, root: Path) -> Path:
    from embodichain.data.dataset import _dataset_download_lock

    cache = root / _BUNDLE_VERSION
    output = cache / task
    spec = _TASKS[task]
    with _dataset_download_lock(str(root), _BUNDLE_VERSION):
        if output.is_dir():
            return output

        if __package__:
            from . import _prepare
        else:
            import _prepare

        print(
            f"Preparing {task} episode {spec['episode']} from public RoboCasa365 inputs.",
            file=sys.stderr,
        )
        converter_python = _converter(cache)
        members = {}
        with (
            TemporaryDirectory(dir=cache) as working,
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            groups = list(_range_groups(spec["members"]))
            # Materialize each result inside the worker to overlap network requests.
            for payloads in pool.map(
                lambda group: dict(_read_range(spec, group)), groups
            ):
                members.update(payloads)
            recording = _prepare._read_episode(
                members.__getitem__, spec["episode"], task
            )
            files = {
                "/".join(_prepare._source_asset(filename))
                for node in recording[0].iter()
                if (filename := node.get("file")) is not None
            }
            assets = cache / "assets"
            _cache_assets(files, assets, Path(working), pool)
        _prepare._prepare_bundle(
            recording,
            task=task,
            episode=spec["episode"],
            archive_sha256=spec["sha256"],
            robocasa_assets=assets / "robocasa",
            robosuite_assets=assets / "robosuite",
            converter_python=converter_python,
            output=output,
        )
        return output
