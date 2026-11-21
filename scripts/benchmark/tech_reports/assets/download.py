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

"""Resolve and verify benchmark inputs at a single Hugging Face commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import hf_hub_download

__all__ = ["download_report_assets"]

DATASET = "DexForceAI/embodichain_reports"
REPORT_ID = "simulation-report-2026-10-28"
DATASET_REVISION = "ba51ac54a4c28417691bbed5a23848b01cb21396"
_DEFAULT_CACHE = Path.home() / ".cache/embodichain/report-assets"
_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != value
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"Invalid relative input path {value!r}")
    return path


def _file_matches(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == digest


def _bundle_files(index: dict, bundle_id: str) -> tuple[PurePosixPath, list[dict]]:
    if index.get("schema_version") != 1 or not isinstance(index.get("bundles"), dict):
        raise ValueError("Unsupported report asset index")
    if bundle_id not in index["bundles"]:
        raise ValueError(f"Unknown report asset bundle {bundle_id!r}")
    entry = index["bundles"][bundle_id]
    report, benchmark = entry.get("report_id", ""), entry.get("benchmark_id", "")
    if not all(
        isinstance(value, str) and _ID.fullmatch(value) for value in (report, benchmark)
    ):
        raise ValueError(
            f"Invalid report or benchmark ID for asset bundle {bundle_id!r}"
        )
    root = PurePosixPath("reports", report, "inputs", benchmark)
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"No files declared for asset bundle {bundle_id!r}")
    seen = set()
    for record in files:
        path = _relative_path(record["path"])
        size, sha = record["size_bytes"], record["sha256"]
        if (
            not path.is_relative_to(root)
            or path == root
            or path in seen
            or type(size) is not int
            or size < 0
            or not isinstance(sha, str)
            or not _SHA256.fullmatch(sha)
        ):
            raise ValueError(f"Invalid file entry in {bundle_id!r}: {path}")
        seen.add(path)
    return root, files


def download_report_assets(
    bundle_id: str,
    *,
    revision: str | None = DATASET_REVISION,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Fetch a report input bundle and verify all files before returning its path.

    Args:
        bundle_id: Bundle key in the report dataset index.
        revision: Commit, branch or tag. Branches and tags resolve once via the index.
        cache_dir: Hugging Face Hub cache directory for these report inputs.
        local_files_only: Resolve and verify files using the local cache only.

    Returns:
        The input directory and the dataset, commit and file hashes used.
    """
    if not revision:
        raise ValueError("A published report asset revision is required")
    cache = cache_dir or _DEFAULT_CACHE
    options = dict(
        repo_id=DATASET,
        repo_type="dataset",
        cache_dir=cache,
        local_files_only=local_files_only,
        endpoint="https://huggingface.co",
        token=False,
    )
    index_path = Path(
        hf_hub_download(filename="index.json", revision=revision, **options)
    )
    snapshot = index_path.parent
    commit = snapshot.name
    if not _SHA1.fullmatch(commit) or (
        _SHA1.fullmatch(revision) and revision != commit
    ):
        raise ValueError(f"Unexpected Hugging Face snapshot for {revision!r}")
    payload = index_path.read_bytes()
    root, files = _bundle_files(json.loads(payload), bundle_id)
    for entry in files:
        name, size, digest = entry["path"], entry["size_bytes"], entry["sha256"]
        path = Path(hf_hub_download(filename=name, revision=commit, **options))
        if not _file_matches(path, size, digest):
            if not local_files_only:
                path = Path(
                    hf_hub_download(
                        filename=name, revision=commit, force_download=True, **options
                    )
                )
            if not _file_matches(path, size, digest):
                raise ValueError(f"Report input checksum mismatch at {commit}: {name}")
    return snapshot / root, {
        "repo_id": DATASET,
        "revision": commit,
        "bundle_id": bundle_id,
        "index_sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark", choices=("l-02", "l-03", "l-04", "p-01", "p-02", "p-03")
    )
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    path, provenance = download_report_assets(
        f"{REPORT_ID}-{args.benchmark}",
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    print(json.dumps({"input_directory": str(path), **provenance}, indent=2))


if __name__ == "__main__":
    main()
