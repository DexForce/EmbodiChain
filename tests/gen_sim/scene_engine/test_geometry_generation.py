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

import pytest

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
    _parse_objects_response,
)

_GLB_BYTES = b"glTF\x02\x00\x00\x00"


class _Response:
    def __init__(self, *, payload: object | None = None, content: bytes = b"") -> None:
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _Session:
    def __init__(self, *, payload: dict[str, Any], downloads: dict[str, bytes]) -> None:
        self._payload = payload
        self._downloads = downloads
        self.post_file_names: list[tuple[str, str]] = []
        self.closed = False

    def post(
        self, _url: str, *, files: list[tuple[str, tuple[Any, ...]]], **_: object
    ) -> _Response:
        self.post_file_names = [(field, str(value[0])) for field, value in files]
        return _Response(payload=self._payload)

    def get(self, url: str, **_: object) -> _Response:
        return _Response(content=self._downloads[url])

    def close(self) -> None:
        self.closed = True


def _object_response(object_id: str, mesh_path: str) -> dict[str, object]:
    return {
        "name": object_id,
        "mesh": mesh_path,
        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }


def test_generate_objects_preserves_requested_mask_order(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    table_mask_path = tmp_path / "table.png"
    cup_mask_path = tmp_path / "cup.png"
    for path in (image_path, table_mask_path, cup_mask_path):
        path.write_bytes(b"image")
    response_payload = {
        "ok": True,
        "result": {
            "objects": [
                _object_response("table", "/assets/table.glb"),
                _object_response("cup", "/assets/cup.glb"),
            ]
        },
    }
    session = _Session(
        payload=response_payload,
        downloads={
            "http://geometry.test/assets/table.glb": _GLB_BYTES,
            "http://geometry.test/assets/cup.glb": _GLB_BYTES,
        },
    )
    client = GeometryGenerationClient(
        base_url="http://geometry.test",
        timeout_s=1,
        max_attempts=1,
        health_path="/health",
        generate_objects_path="/generate_objects",
        session=session,
    )
    output_root = tmp_path / "generated" / "meshes"

    _, objects = client.generate_objects(
        image_path=image_path,
        object_masks=[("table", table_mask_path), ("cup", cup_mask_path)],
        output_root=output_root,
    )

    assert session.post_file_names == [
        ("image", "image.png"),
        ("masks", "table.png"),
        ("masks", "cup.png"),
    ]
    assert [object_data["mesh"] for object_data in objects] == [
        "/assets/table.glb",
        "/assets/cup.glb",
    ]
    assert (output_root / "table.glb").read_bytes() == _GLB_BYTES
    assert (output_root / "cup.glb").read_bytes() == _GLB_BYTES


def test_parse_objects_response_rejects_mismatched_object_name() -> None:
    payload = {
        "ok": True,
        "result": {"objects": [_object_response("wrong", "/assets/wrong.glb")]},
    }

    with pytest.raises(RuntimeError, match="does not match"):
        _parse_objects_response(payload, object_ids=["table"])
