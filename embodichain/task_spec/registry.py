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

"""Content-addressed persistence for independently reusable TaskSpecs."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .spec import TaskSpec
from .validation import validate_task_spec

__all__ = [
    "TaskSpecCache",
    "TaskSpecCacheError",
    "TaskSpecRegistry",
]

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class TaskSpecCacheError(RuntimeError):
    """Raised when a content-addressed TaskSpec cache entry is corrupt."""


class TaskSpecCache:
    """Store and retrieve validated TaskSpecs by semantic hash.

    The cache contains one pretty-printed JSON document per semantic hash.  A
    temporary file plus ``os.replace`` makes writes atomic, so a process that
    is concurrently reading a reusable TaskSpec never observes a partial
    document.
    """

    def __init__(self, root: str | os.PathLike[str], *, create: bool = True) -> None:
        """Create a cache rooted at ``root``.

        Args:
            root: Directory containing ``<semantic_hash>.json`` entries.
            create: Create the directory when it does not exist.
        """
        self.root = Path(root).expanduser().resolve()
        if self.root.exists() and not self.root.is_dir():
            raise NotADirectoryError(
                f"TaskSpec cache root is not a directory: {self.root}"
            )
        if create:
            self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, semantic_hash: str) -> Path:
        """Return the safe on-disk path for one semantic hash."""
        _validate_hash(semantic_hash)
        return self.root / f"{semantic_hash}.json"

    def contains(self, semantic_hash: str) -> bool:
        """Return whether a cache entry exists."""
        return self.path_for(semantic_hash).is_file()

    def put(self, template: TaskSpec, *, overwrite: bool = False) -> Path:
        """Atomically cache one validated TaskSpec and return its path."""
        normalized = TaskSpec.from_dict(validate_task_spec(template))
        destination = self.path_for(normalized.semantic_hash)
        if destination.exists() and not overwrite:
            existing = self.get(normalized.semantic_hash)
            if existing is not None:
                # Metadata records the producing run and is intentionally
                # excluded from semantic identity.  Reusing an existing
                # content-addressed document must therefore tolerate metadata
                # differences between generation attempts.
                if existing.semantic_hash == normalized.semantic_hash:
                    return destination
                raise FileExistsError(
                    f"TaskSpec cache entry already exists with different content: {destination}"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                normalized.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f".{normalized.semantic_hash}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    def get(self, semantic_hash: str) -> TaskSpec | None:
        """Load one TaskSpec, returning ``None`` for a cache miss."""
        path = self.path_for(semantic_hash)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            template = TaskSpec.from_dict(value)
            if template.semantic_hash != semantic_hash:
                raise ValueError("document hash does not match requested cache key")
            return template
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise TaskSpecCacheError(
                f"Invalid TaskSpec cache entry {path}: {exc}"
            ) from exc

    def require(self, semantic_hash: str) -> TaskSpec:
        """Load a TaskSpec or raise ``KeyError`` on a cache miss."""
        result = self.get(semantic_hash)
        if result is None:
            raise KeyError(f"TaskSpec semantic hash is not cached: {semantic_hash}")
        return result

    def save(self, template: TaskSpec, *, overwrite: bool = False) -> Path:
        """Alias for :meth:`put` used by persistence adapters."""
        return self.put(template, overwrite=overwrite)

    def load(self, semantic_hash: str) -> TaskSpec | None:
        """Alias for :meth:`get` used by persistence adapters."""
        return self.get(semantic_hash)

    def delete(self, semantic_hash: str) -> bool:
        """Delete one cache entry and report whether it existed."""
        path = self.path_for(semantic_hash)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def hashes(self) -> tuple[str, ...]:
        """Return valid cached semantic hashes in deterministic order."""
        if not self.root.exists():
            return ()
        values = []
        for path in self.root.glob("*.json"):
            if _HASH_RE.fullmatch(path.stem):
                values.append(path.stem)
        return tuple(sorted(values))

    def templates(self) -> Iterator[TaskSpec]:
        """Yield validated TaskSpecs in semantic-hash order."""
        for semantic_hash in self.hashes():
            template = self.get(semantic_hash)
            if template is not None:
                yield template

    def find(
        self, task_id: str, *, instruction: str | None = None
    ) -> tuple[TaskSpec, ...]:
        """Return cached TaskSpecs matching a logical task ID.

        A task ID may intentionally have several semantic versions.  The
        result is therefore a tuple ordered by semantic hash rather than an
        implicit single mutable record.
        """
        normalized_task_id = str(task_id).strip()
        if not normalized_task_id:
            raise ValueError("task_id must be a non-empty string.")
        normalized_instruction = (
            None if instruction is None else str(instruction).strip()
        )
        return tuple(
            template
            for template in self.templates()
            if template.task_id == normalized_task_id
            and (
                normalized_instruction is None
                or template.instruction == normalized_instruction
            )
        )

    def __contains__(self, semantic_hash: object) -> bool:
        """Support ``semantic_hash in cache`` for string keys."""
        return isinstance(semantic_hash, str) and self.contains(semantic_hash)


class TaskSpecRegistry(TaskSpecCache):
    """Named registry alias for integrations that manage reusable specs."""


def _validate_hash(value: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError("TaskSpec semantic hash must be a lowercase SHA-256 digest.")
