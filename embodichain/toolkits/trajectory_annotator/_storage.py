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

"""Identity-bound JSON loading and revision-checked atomic persistence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

from ._models import ConflictError, validate_skill_intervals
from .frames import FrameSourceInfo

__all__ = ["AnnotationStore"]


class AnnotationStore:
    """Own one file; publish new revisions only after atomic replacement succeeds.

    Args:
        path: Destination JSON file, distinct from any backing source file.
        document_bytes: Maximum encoded file size.
    """

    def __init__(self, path: Path, document_bytes: int) -> None:
        self.path = path.resolve()
        self.limit = document_bytes
        self._lock = threading.RLock()
        self._document: dict[str, Any] | None = None
        self._digest: str | None = None

    def _read(self) -> bytes | None:
        try:
            with self.path.open("rb") as handle:
                content = handle.read(self.limit + 1)
        except FileNotFoundError:
            return None
        if len(content) > self.limit:
            raise ValueError("标注文件超过大小上限。")
        return content

    def initialize(self, info: FrameSourceInfo) -> None:
        """Load matching annotations; preserve legacy video JSON until next save."""
        if info.path is not None and (
            self.path == info.path.resolve()
            or (self.path.exists() and os.path.samefile(self.path, info.path))
        ):
            raise ValueError("标注文件不能与输入文件相同。")
        source: dict[str, Any] = dict(
            kind=info.kind,
            name=info.name,
            identity=info.identity,
            frame_count=info.frame_count,
            width=info.width,
            height=info.height,
            fps=info.fps,
            timing=info.timing,
            details=deepcopy(info.details),
        )
        if info.path is not None:
            source["path"] = os.path.relpath(info.path.resolve(), self.path.parent)
        # The versioned JSON key `segments` stores atomic skill intervals.
        document = dict(
            schema_version=2,
            frame_index_base=0,
            interval="closed",
            source=source,
            revision=0,
            segments=[],
        )
        with self._lock:
            content = self._read()
            if content is not None:
                try:
                    loaded = json.loads(content)
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f"无法解析标注 JSON：{error}") from error
                if not isinstance(loaded, dict):
                    raise ValueError("标注文件必须为 JSON 对象。")
                version = loaded.get("schema_version")
                source_key = "source_video" if version == 1 else "source"
                expected_keys = {
                    "schema_version",
                    "frame_index_base",
                    "interval",
                    source_key,
                    "revision",
                    "segments",
                }
                if loaded.keys() != expected_keys:
                    raise ValueError("标注文件字段不符合格式，不能覆盖原文件。")
                if (
                    type(version) is not int
                    or version not in {1, 2}
                    or type(loaded["frame_index_base"]) is not int
                    or loaded["frame_index_base"] != 0
                    or loaded["interval"] != "closed"
                ):
                    raise ValueError("只支持 schema_version=1/2、0 基帧号和闭区间。")
                original = loaded[source_key]
                if version == 1:
                    matching = (
                        info.kind == "video"
                        and isinstance(original, dict)
                        and all(
                            original.get(key) == info.details.get(key)
                            for key in (
                                "sha256",
                                "stream_index",
                                "frame_count",
                                "width",
                                "height",
                                "time_base",
                            )
                        )
                    )
                else:
                    matching = isinstance(original, dict) and all(
                        original.get(key) == source[key]
                        for key in (
                            "kind",
                            "identity",
                            "frame_count",
                            "width",
                            "height",
                            "fps",
                            "timing",
                        )
                    )
                if not matching:
                    raise ValueError(
                        "标注来源与图像序列不匹配（内容、相机、帧率或帧信息不同）。"
                    )
                revision = loaded["revision"]
                if type(revision) is not int or revision < 0:
                    raise ValueError("标注 revision 必须为非负整数。")
                document["revision"] = revision
                document["segments"] = validate_skill_intervals(
                    loaded["segments"], info.frame_count
                )
                self._digest = hashlib.sha256(content).hexdigest()
            self._document = document

    def snapshot(self) -> dict[str, Any]:
        """Return an owned document snapshot and whether it has been persisted."""
        with self._lock:
            if self._document is None:
                raise RuntimeError("帧源仍在初始化。")
            return {
                "document": deepcopy(self._document),
                "persisted": self._digest is not None,
            }

    def check_unchanged(self) -> None:
        """Reject fits based on an annotation file edited outside this session."""
        with self._lock:
            content = self._read()
            digest = (
                hashlib.sha256(content).hexdigest() if content is not None else None
            )
            if digest != self._digest:
                raise ConflictError(
                    "磁盘上的标注文件已被其他程序修改，请重新打开项目。"
                )

    def save(
        self, skill_intervals: object, expected_revision: object
    ) -> dict[str, Any]:
        """Validate and persist atomic skill intervals without overwriting stale edits."""
        with self._lock:
            if self._document is None:
                raise RuntimeError("帧源仍在初始化。")
            if (
                type(expected_revision) is not int
                or expected_revision != self._document["revision"]
            ):
                raise ConflictError("标注已有新修订，请先导出当前修改，再刷新页面。")
            content = self._read()
            digest = (
                hashlib.sha256(content).hexdigest() if content is not None else None
            )
            if digest != self._digest:
                raise ConflictError(
                    "磁盘上的标注文件已被其他程序修改，请导出当前修改后重新打开。"
                )
            candidate = deepcopy(self._document)
            candidate["segments"] = validate_skill_intervals(
                skill_intervals, candidate["source"]["frame_count"]
            )
            candidate["revision"] += 1
            encoded = (
                json.dumps(candidate, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            if len(encoded) > self.limit:
                raise ValueError("标注内容超过大小上限。")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            self._document = candidate
            self._digest = hashlib.sha256(encoded).hexdigest()
            return deepcopy(candidate)
