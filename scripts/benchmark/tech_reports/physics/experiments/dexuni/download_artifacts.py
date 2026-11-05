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
"""Restore published DexUni runtime inputs outside measurement regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any
import zipfile

from filelock import FileLock
from huggingface_hub import hf_hub_download

from .paths import ASSETS, OUTPUTS

__all__ = ["restore_artifacts", "ensure_task_inputs"]
_REPO = "DexForceAI/embodichain_reports"
_INDEX = ASSETS / "publication.json"
_CACHE = Path(
    os.environ.get("DEXUNI_REPORT_CACHE_DIR", "~/.cache/embodichain/report-assets")
).expanduser()


def _relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Unsafe artifact path: {value!r}")
    parts = value.split("/")
    if any(p in ("", ".", "..") or ":" in p for p in parts):
        raise ValueError(f"Unsafe artifact path: {value!r}")
    if PurePosixPath(value).is_absolute():
        raise ValueError(f"Unsafe artifact path: {value!r}")
    return Path(*parts)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _verified(path: Path, artifact: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == artifact["bytes"]
        and _sha(path) == artifact["sha256"]
    )


def _fetch(artifact: dict[str, Any], revision: str, cache: Path, offline: bool) -> Path:
    for attempt in range(1 if offline else 2):
        path = Path(
            hf_hub_download(
                repo_id=_REPO,
                repo_type="dataset",
                filename=artifact["path"],
                revision=revision,
                endpoint="https://huggingface.co",
                token=False,
                cache_dir=cache,
                local_files_only=offline,
                force_download=bool(attempt),
            )
        )
        if _verified(path, artifact):
            return path
    raise ValueError(f"Artifact size/SHA256 mismatch: {artifact['path']} ({revision})")


def _destination(root: Path, relative: Path) -> Path:
    dest = root / relative
    # Reject symlinks even when they happen to resolve inside the output tree.
    for candidate in (dest, *dest.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"Refusing output symlink: {candidate}")
    if not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Output escapes destination: {relative}")
    return dest


def _materialize(path: Path, artifact: dict[str, Any], root: Path) -> None:
    # Fully stage and check one bundle before modifying its output files.
    with tempfile.TemporaryDirectory(prefix=".dexuni-restore-", dir=root) as name:
        staging = Path(name)
        files = []
        if artifact["path"].endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                seen = set()
                for info in archive.infolist():
                    relative = _relative(
                        info.filename.rstrip("/") if info.is_dir() else info.filename
                    )
                    if relative in seen:
                        raise ValueError(f"Duplicate archive member: {relative}")
                    seen.add(relative)
                    mode = info.external_attr >> 16
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ValueError(f"Unsupported archive member: {relative}")
                    if info.is_dir():
                        continue
                    if relative.parts[0] != artifact["task_folder"]:
                        raise ValueError(f"Archive member outside task: {relative}")
                    dest = _destination(root, relative)
                    staged = staging / relative
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as src, staged.open("wb") as target:
                        shutil.copyfileobj(src, target)
                    files.append((staged, dest))
        else:
            dest = _destination(root, _relative(artifact["output_path"]))
            staged = staging / "file"
            shutil.copyfile(path, staged)
            files.append((staged, dest))
        for staged, dest in files:
            if dest.exists() and (not dest.is_file() or _sha(dest) != _sha(staged)):
                raise FileExistsError(
                    f"Existing file differs; choose an empty --output-root: {dest}"
                )
        for staged, dest in files:
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, dest)


def _select(
    index: dict[str, Any], tasks: list[str], kinds: list[str]
) -> list[dict[str, Any]]:
    if index.get("schema_version") != 1 or index.get("repo_id") != _REPO:
        raise ValueError("Unsupported DexUni publication index")
    if not re.fullmatch(r"[0-9a-f]{40}", index.get("revision") or ""):
        raise ValueError("Publication requires a full verified dataset commit SHA")
    unknown = set(tasks) - set(index["tasks"])
    if unknown:
        raise ValueError(
            f"Unknown tasks: {sorted(unknown)}; available: {sorted(index['tasks'])}"
        )
    if not kinds or set(kinds) - {"inputs"}:
        raise ValueError(
            "Only input assets are published; results and videos are not downloadable bundles"
        )
    folders = (
        {index["tasks"][task] for task in tasks}
        if tasks
        else set(index["tasks"].values())
    )
    selected = []
    seen = set()
    for artifact in index["artifacts"]:
        _relative(artifact["path"])
        if not artifact["path"].startswith(f"reports/{index['report_id']}/"):
            raise ValueError("Artifact outside report")
        if artifact["path"] in seen:
            raise ValueError("Duplicate artifact path")
        seen.add(artifact["path"])
        if (
            not isinstance(artifact["bytes"], int)
            or artifact["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
        ):
            raise ValueError("Invalid artifact size/hash")
        if artifact["kind"] in kinds and artifact["task_folder"] in folders | {
            "__overview__"
        }:
            selected.append(artifact)
    return selected


def restore_artifacts(
    *,
    tasks: list[str] | None = None,
    kinds: list[str] | None = None,
    output_root: Path = OUTPUTS,
    cache_dir: Path = _CACHE,
    offline: bool = False,
    index_path: Path = _INDEX,
) -> list[Path]:
    """Download verified files at the published revision and restore local paths.

    Args:
        tasks: Public task codes; omitted means all published tasks.
        kinds: Artifact groups; defaults to inputs only.
        output_root: Writable local output root, outside source and Hub cache.
        cache_dir: Standard Hub cache root, isolated by repository/revision.
        offline: Consume only already cached files without network access.
        index_path: Reviewed source release index containing a full commit pin.

    Returns:
        Verified local Hub cache paths for the selected artifacts. Existing
        differing output files raise an error instead of being overwritten.
    """
    index = json.loads(index_path.read_text())
    selected = _select(index, tasks or [], kinds or ["inputs"])
    root = output_root.expanduser().absolute()
    cache = cache_dir.expanduser().resolve()
    if (
        root.is_symlink()
        or root.resolve().is_relative_to(cache)
        or cache.is_relative_to(root.resolve())
    ):
        raise ValueError(
            "Output and Hub cache must be separate directories without an output-root symlink"
        )
    tech_root = ASSETS.parents[2]
    if root.resolve().is_relative_to(tech_root):
        raise ValueError("Artifacts must be restored outside the source tree")
    root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    # HF handles cache locking; this lock serializes restoration into one tree.
    with FileLock(str(root / ".dexuni-restore.lock")):
        for artifact in selected:
            path = _fetch(artifact, index["revision"], cache, offline)
            _materialize(path, artifact, root)
            downloaded.append(path)
    return downloaded


def ensure_task_inputs(task: str, root: Path) -> None:
    """Restore missing runtime inputs before a task creates its simulator.

    Args:
        task: Published task code such as M01.
        root: Task directory under the caller's writable output root.

    Existing complete inputs use the task loader's normal integrity checks.
    Partial restores use the same pin, checksums and overwrite protection as
    the explicit downloader. Network failures propagate before simulation.
    """
    index = json.loads(_INDEX.read_text())
    folder = index["tasks"][task]
    required = json.loads((ASSETS / "runtime_checks.json").read_text())[folder]["files"]
    if all((root / name).is_file() for name in required):
        return
    if root.name != folder:
        raise ValueError(f"Restore {task} into a directory named {folder}: {root}")
    print(
        f"{task}: downloading missing input assets at {index['revision']}", flush=True
    )
    restore_artifacts(tasks=[task], output_root=root.parent, index_path=_INDEX)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Published {task} inputs are incomplete: {missing}")


def main() -> None:
    """Run explicit preparation before any simulation or measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task code; repeat to select several. Default: all",
    )
    parser.add_argument(
        "--kind",
        action="append",
        choices=["inputs"],
        help="Default: inputs only",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUTS)
    parser.add_argument("--cache-dir", type=Path, default=_CACHE)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show selected artifacts and bytes without downloading",
    )
    args = parser.parse_args()
    index = json.loads(_INDEX.read_text())
    selected = _select(index, args.task, args.kind or ["inputs"])
    if args.list:
        print(
            json.dumps(
                {
                    "revision": index["revision"],
                    "bytes": sum(a["bytes"] for a in selected),
                    "artifacts": [a["path"] for a in selected],
                },
                indent=2,
            )
        )
        return
    restored = restore_artifacts(
        tasks=args.task,
        kinds=args.kind,
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )
    print(
        f"Verified and restored {len(restored)} artifacts at {index['revision']} into {args.output_root}"
    )


if __name__ == "__main__":
    main()
