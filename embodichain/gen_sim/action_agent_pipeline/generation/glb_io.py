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

from pathlib import Path
from typing import Any
import json
import struct

__all__ = ["read_glb"]

_GLB_JSON_CHUNK_TYPE = 0x4E4F534A
_GLB_BINARY_CHUNK_TYPE = 0x004E4942


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read a GLB v2 file and return its JSON document and binary chunk."""
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError(f"GLB file is too small: {path}")

    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Only GLB version 2 files are supported: {path}")
    if declared_length > len(data):
        raise ValueError(f"GLB length header exceeds file size: {path}")

    offset = 12
    doc: dict[str, Any] | None = None
    binary_chunk = b""
    while offset + 8 <= declared_length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > declared_length:
            raise ValueError(f"GLB chunk exceeds file size: {path}")
        chunk = data[offset:chunk_end]
        offset = chunk_end
        if chunk_type == _GLB_JSON_CHUNK_TYPE:
            doc = json.loads(chunk.decode("utf-8").rstrip("\x00 "))
        elif chunk_type == _GLB_BINARY_CHUNK_TYPE:
            binary_chunk = chunk

    if doc is None:
        raise ValueError(f"GLB file does not contain a JSON chunk: {path}")
    return doc, binary_chunk
