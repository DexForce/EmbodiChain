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

from embodichain.gen_sim.scene_engine.clients import geometry_generation
from embodichain.gen_sim.scene_engine.clients import image_generation
from embodichain.gen_sim.scene_engine.clients import image_segmentation
from embodichain.gen_sim.scene_engine.llms import load_config


class _Response:
    """Minimal successful HTTP response used by client unit tests."""

    def __init__(
        self,
        payload: object,
        *,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _Session:
    """Capture HTTP calls without contacting an external service."""

    def __init__(
        self, *, get_payload: object, post_payload: object | None = None
    ) -> None:
        self.get_payload = get_payload
        self.post_payload = post_payload
        self.get_calls: list[tuple[str, int]] = []
        self.post_call: dict[str, object] | None = None

    def get(self, url: str, *, timeout: int) -> _Response:
        self.get_calls.append((url, timeout))
        return _Response(self.get_payload)

    def post(self, url: str, **kwargs: object) -> _Response:
        self.post_call = {"url": url, **kwargs}
        return _Response(self.post_payload)

    def close(self) -> None:
        return None


def test_clients_load_their_required_dotenv_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry_values = {
        "SCENE_ENGINE_GEOMETRY_GENERATION_BASE_URL": "http://geometry/",
        "SCENE_ENGINE_GEOMETRY_GENERATION_TIMEOUT_S": "60",
        "SCENE_ENGINE_GEOMETRY_GENERATION_MAX_ATTEMPTS": "2",
        "SCENE_ENGINE_GEOMETRY_GENERATION_HEALTH_PATH": "/health",
        "SCENE_ENGINE_GEOMETRY_GENERATION_OBJECTS_PATH": "/objects",
    }
    segmentation_values = {
        "SCENE_ENGINE_IMAGE_SEGMENTATION_BASE_URL": "http://segment/",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_TIMEOUT_S": "30",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_MAX_ATTEMPTS": "2",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_HEALTH_PATH": "/health",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_BY_PROMPT_PATH": "/segment_by_prompt",
    }
    image_generation_values = {
        "SCENE_ENGINE_IMAGE_GENERATION_BASE_URL": "http://image-generation/",
        "SCENE_ENGINE_IMAGE_GENERATION_TIMEOUT_S": "120",
        "SCENE_ENGINE_IMAGE_GENERATION_MAX_ATTEMPTS": "2",
        "SCENE_ENGINE_IMAGE_GENERATION_HEALTH_PATH": "/health",
        "SCENE_ENGINE_IMAGE_GENERATION_BY_PROMPT_PATH": "/generate_image_by_prompt",
    }
    llm_values = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "test-model",
        "OPENAI_BASE_URL": "http://llm/v1/",
        "SCENE_ENGINE_OPENAI_DEFAULT_QUERY": '{"api-version": "1"}',
        "OPENAI_MAX_ATTEMPTS": "2",
    }
    monkeypatch.setattr(
        geometry_generation, "read_scene_engine_env_values", lambda *_: geometry_values
    )
    monkeypatch.setattr(
        image_segmentation,
        "read_scene_engine_env_values",
        lambda *_: segmentation_values,
    )
    monkeypatch.setattr(
        image_generation,
        "read_scene_engine_env_values",
        lambda *_: image_generation_values,
    )
    monkeypatch.setattr(
        load_config, "read_scene_engine_env_values", lambda *_: llm_values
    )

    geometry_client = geometry_generation.GeometryGenerationClient.from_dotenv()
    segmentation_client = image_segmentation.ImageSegmentationClient.from_dotenv()
    image_generation_client = image_generation.ImageGenerationClient.from_dotenv()
    llm_client_config = load_config.load_llm_config()

    assert geometry_client._base_url == "http://geometry"
    assert geometry_client._generate_objects_path == "/objects"
    assert segmentation_client._base_url == "http://segment"
    assert segmentation_client._segment_by_prompt_path == "/segment_by_prompt"
    assert image_generation_client._base_url == "http://image-generation"
    assert (
        image_generation_client._generate_image_by_prompt_path
        == "/generate_image_by_prompt"
    )
    assert llm_client_config.default_query == {"api-version": "1"}
    assert llm_client_config.base_url == "http://llm/v1"


def test_geometry_dotenv_config_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "SCENE_ENGINE_GEOMETRY_GENERATION_BASE_URL": "http://geometry",
        "SCENE_ENGINE_GEOMETRY_GENERATION_TIMEOUT_S": "0",
        "SCENE_ENGINE_GEOMETRY_GENERATION_MAX_ATTEMPTS": "1",
        "SCENE_ENGINE_GEOMETRY_GENERATION_HEALTH_PATH": "/health",
        "SCENE_ENGINE_GEOMETRY_GENERATION_OBJECTS_PATH": "/objects",
    }
    monkeypatch.setattr(
        geometry_generation, "read_scene_engine_env_values", lambda *_: values
    )

    with pytest.raises(ValueError, match="TIMEOUT_S must be at least 1"):
        geometry_generation.GeometryGenerationClient.from_dotenv()


def test_service_health_checks_use_the_configured_health_path() -> None:
    geometry_session = _Session(get_payload={"ok": True})
    geometry_client = geometry_generation.GeometryGenerationClient(
        base_url="http://geometry",
        timeout_s=60,
        max_attempts=1,
        health_path="/health",
        generate_objects_path="/objects",
        session=geometry_session,
    )
    segmentation_session = _Session(get_payload={"ok": True})
    segmentation_client = image_segmentation.ImageSegmentationClient(
        base_url="http://segment",
        timeout_s=30,
        max_attempts=1,
        health_path="/health",
        segment_by_prompt_path="/segment_by_prompt",
        session=segmentation_session,
    )
    image_generation_session = _Session(get_payload={"ok": True})
    image_generation_client = image_generation.ImageGenerationClient(
        base_url="http://image-generation",
        timeout_s=120,
        max_attempts=1,
        health_path="/health",
        generate_image_by_prompt_path="/generate_image_by_prompt",
        session=image_generation_session,
    )

    geometry_client.check_health()
    segmentation_client.check_health()
    image_generation_client.check_health()

    assert geometry_session.get_calls == [("http://geometry/health", 10)]
    assert segmentation_session.get_calls == [("http://segment/health", 30)]
    assert image_generation_session.get_calls == [
        ("http://image-generation/health", 10)
    ]


def test_image_generation_client_posts_prompt_and_writes_png(
    tmp_path: Path,
) -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nimage"

    class ImageGenerationSession(_Session):
        def post(self, url: str, **kwargs: object) -> _Response:
            self.post_call = {"url": url, **kwargs}
            return _Response(
                {},
                content=png_bytes,
                headers={"content-type": "image/png"},
            )

    session = ImageGenerationSession(get_payload={"ok": True})
    client = image_generation.ImageGenerationClient(
        base_url="http://image-generation",
        timeout_s=120,
        max_attempts=1,
        health_path="/health",
        generate_image_by_prompt_path="/generate_image_by_prompt",
        session=session,
    )

    output_path = client.generate_image_by_prompt(
        prompt="a red mug on a wooden table",
        output_path=tmp_path / "generated.png",
    )

    assert output_path.read_bytes() == png_bytes
    assert session.post_call is not None
    assert session.post_call["url"] == (
        "http://image-generation/generate_image_by_prompt"
    )
    assert session.post_call["json"] == {"prompt": "a red mug on a wooden table"}


def test_image_generation_client_rejects_non_png_response(tmp_path: Path) -> None:
    class ImageGenerationSession(_Session):
        def post(self, url: str, **kwargs: object) -> _Response:
            self.post_call = {"url": url, **kwargs}
            return _Response(
                {"ok": False, "error": "failed"},
                content=b'{"ok": false}',
                headers={"content-type": "application/json"},
            )

    session = ImageGenerationSession(get_payload={"ok": True})
    client = image_generation.ImageGenerationClient(
        base_url="http://image-generation",
        timeout_s=120,
        max_attempts=1,
        health_path="/health",
        generate_image_by_prompt_path="/generate_image_by_prompt",
        session=session,
    )

    with pytest.raises(RuntimeError, match="request failed after 1 attempts") as exc:
        client.generate_image_by_prompt(
            prompt="a red mug on a wooden table",
            output_path=tmp_path / "generated.png",
        )
    assert "response is not a PNG image" in str(exc.value.__cause__)


def test_segmentation_client_posts_prompt_and_returns_rle_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png")
    rle_mask = {"counts": [0, 1], "size": [1, 1]}
    session = _Session(
        get_payload={"ok": True},
        post_payload={"result": {"masks": [rle_mask]}},
    )
    client = image_segmentation.ImageSegmentationClient(
        base_url="http://segment",
        timeout_s=30,
        max_attempts=1,
        health_path="/health",
        segment_by_prompt_path="/segment_by_prompt",
        session=session,
    )

    assert client.segment_single_object(image_path=image_path, prompt="table") == [
        rle_mask
    ]
    assert session.post_call is not None
    assert session.post_call["url"] == "http://segment/segment_by_prompt"
    assert session.post_call["data"] == {"prompt": "table"}


def test_segmentation_client_accepts_instance_mask_response() -> None:
    rle_mask = {"counts": [0, 1], "size": [1, 1]}

    masks = image_segmentation._extract_rle_masks(
        {"data": {"instances": [{"mask_rle": rle_mask}]}}
    )

    assert masks == [rle_mask]


def test_geometry_response_requires_matching_ordered_objects() -> None:
    response: dict[str, Any] = {
        "ok": True,
        "result": {
            "objects": [
                {
                    "name": "table_001",
                    "mesh": "/results/table.glb",
                    "rotation_quaternion_wxyz": [1, 0, 0, 0],
                    "translation": [0, 1, 2],
                    "scale": [1, 1, 1],
                }
            ]
        },
    }

    objects = geometry_generation._parse_objects_response(
        response,
        object_ids=["table_001"],
    )

    assert objects == [
        {
            "mesh": "/results/table.glb",
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 1.0, 2.0],
            "scale": [1.0, 1.0, 1.0],
        }
    ]


def test_geometry_client_posts_masks_and_downloads_glbs(tmp_path: Path) -> None:
    class GeometrySession(_Session):
        def post(self, url: str, **kwargs: object) -> _Response:
            self.post_call = {"url": url, **kwargs}
            return _Response(
                {
                    "ok": True,
                    "result": {
                        "objects": [
                            {
                                "name": "cup",
                                "mesh": "/results/cup.glb",
                                "rotation_quaternion_wxyz": [1, 0, 0, 0],
                                "translation": [0, 0, 0],
                                "scale": [1, 1, 1],
                            }
                        ]
                    },
                }
            )

        def get(self, url: str, *, timeout: int) -> _Response:
            self.get_calls.append((url, timeout))
            return _Response({}, content=b"glTF-mesh")

    image_path = tmp_path / "scene.png"
    mask_path = tmp_path / "cup.png"
    image_path.write_bytes(b"png")
    mask_path.write_bytes(b"png")
    session = GeometrySession(get_payload={"ok": True})
    client = geometry_generation.GeometryGenerationClient(
        base_url="http://geometry",
        timeout_s=30,
        max_attempts=1,
        health_path="/health",
        generate_objects_path="/objects",
        session=session,
    )

    _, objects = client.generate_objects(
        image_path=image_path,
        object_masks=[("cup", mask_path)],
        output_root=tmp_path / "output",
    )

    assert objects[0]["mesh"] == "/results/cup.glb"
    assert session.post_call is not None
    assert session.post_call["url"] == "http://geometry/objects"
    assert (tmp_path / "output/cup.glb").read_bytes() == b"glTF-mesh"
