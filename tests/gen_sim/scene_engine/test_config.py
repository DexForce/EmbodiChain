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

import json
from pathlib import Path
from typing import Any

import pytest

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
)
from embodichain.gen_sim.scene_engine.llms.load_config import load_llm_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPO_ROOT
    / "embodichain"
    / "gen_sim"
    / "scene_engine"
    / "configs"
    / "scene_engine_config.json"
)


@pytest.fixture(scope="module")
def scene_engine_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_scene_engine_config_declares_all_service_sections(
    scene_engine_config: dict[str, Any],
) -> None:
    assert set(scene_engine_config) == {
        "llm",
        "image_segmentation",
        "geometry_generation",
    }
    assert "openai_compatible" in scene_engine_config["llm"]


@pytest.mark.parametrize(
    ("section_name", "path_key"),
    [
        ("image_segmentation", "segment_single_object_path"),
        ("geometry_generation", "generate_objects_path"),
    ],
)
def test_service_template_has_valid_non_secret_defaults(
    scene_engine_config: dict[str, Any],
    section_name: str,
    path_key: str,
) -> None:
    service_config = scene_engine_config[section_name]

    assert isinstance(service_config["base_url"], str)
    assert isinstance(service_config["timeout_s"], int)
    assert service_config["timeout_s"] > 0
    assert isinstance(service_config["max_attempts"], int)
    assert service_config["max_attempts"] > 0
    assert service_config["health_path"].startswith("/")
    assert service_config[path_key].startswith("/")


def test_llm_environment_overrides_package_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-vision-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("OPENAI_MAX_ATTEMPTS", "5")

    config = load_llm_config()

    assert config.api_key == "test-api-key"
    assert config.model == "test-vision-model"
    assert config.base_url == "http://llm.test/v1"
    assert config.max_attempts == 5


def test_package_template_reports_missing_service_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for environment_name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_MAX_ATTEMPTS",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_BASE_URL",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_TIMEOUT_S",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_MAX_ATTEMPTS",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_HEALTH_PATH",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_PATH",
        "SCENE_ENGINE_GEOMETRY_GENERATION_BASE_URL",
        "SCENE_ENGINE_GEOMETRY_GENERATION_TIMEOUT_S",
        "SCENE_ENGINE_GEOMETRY_GENERATION_MAX_ATTEMPTS",
        "SCENE_ENGINE_GEOMETRY_GENERATION_HEALTH_PATH",
        "SCENE_ENGINE_GEOMETRY_GENERATION_PATH",
    ):
        monkeypatch.delenv(environment_name, raising=False)

    with pytest.raises(ValueError, match="Missing required LLM config keys"):
        load_llm_config()
    with pytest.raises(ValueError, match="base_url must be a non-empty string"):
        ImageSegmentationClient.from_config()
    with pytest.raises(ValueError, match="base_url must be a non-empty string"):
        GeometryGenerationClient.from_config()
