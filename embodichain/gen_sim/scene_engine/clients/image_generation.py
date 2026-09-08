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

import requests

from embodichain.gen_sim.scene_engine.configs.environment import (
    read_scene_engine_env_values,
)


class ImageGenerationClient:
    """Manage the Image Generation Server connection."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: int,
        max_attempts: int,
        health_path: str,
        generate_image_by_prompt_path: str,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._health_path = health_path
        self._generate_image_by_prompt_path = generate_image_by_prompt_path
        self._session = session or requests.Session()

    @classmethod
    def from_dotenv(cls) -> "ImageGenerationClient":
        """Create a client from its required ``gen_sim/.env`` settings."""
        return cls(**_load_dotenv_config())

    def check_health(self) -> None:
        last_error: requests.RequestException | RuntimeError | None = None
        for _ in range(self._max_attempts):
            try:
                response = self._session.get(
                    self._url(self._health_path),
                    timeout=10,  # Use a shorter timeout for avoiding long waits.
                )
                response.raise_for_status()
                response_data = response.json()
                if (
                    not isinstance(response_data, dict)
                    or response_data.get("ok") is not True
                ):
                    raise RuntimeError(
                        "Image Generation Server health response does not contain ok=true."
                    )
                return
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Image Generation Server health check failed after "
            f"{self._max_attempts} attempts."
        ) from last_error

    def close(self) -> None:
        self._session.close()

    def generate_image_by_prompt(
        self,
        *,
        prompt: str,
        output_path: str | Path,
    ) -> Path:
        """Generate one PNG image from ``prompt`` and save it to ``output_path``."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Image generation prompt must not be empty.")

        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                response = self._session.post(
                    self._url(self._generate_image_by_prompt_path),
                    json={"prompt": prompt},
                    timeout=self._timeout_s,
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";")[0]
                if content_type != "image/png":
                    raise RuntimeError(
                        "Image Generation Server response is not a PNG image."
                    )
                resolved_output_path.write_bytes(response.content)
                return resolved_output_path
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Image Generation Server request failed after "
            f"{self._max_attempts} attempts."
        ) from last_error

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"


def _load_dotenv_config() -> dict[str, Any]:
    values = read_scene_engine_env_values(
        "SCENE_ENGINE_IMAGE_GENERATION_BASE_URL",
        "SCENE_ENGINE_IMAGE_GENERATION_TIMEOUT_S",
        "SCENE_ENGINE_IMAGE_GENERATION_MAX_ATTEMPTS",
        "SCENE_ENGINE_IMAGE_GENERATION_HEALTH_PATH",
        "SCENE_ENGINE_IMAGE_GENERATION_BY_PROMPT_PATH",
    )
    try:
        timeout_s = int(values["SCENE_ENGINE_IMAGE_GENERATION_TIMEOUT_S"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SCENE_ENGINE_IMAGE_GENERATION_TIMEOUT_S must be an integer."
        ) from exc
    if timeout_s < 1:
        raise ValueError("SCENE_ENGINE_IMAGE_GENERATION_TIMEOUT_S must be at least 1.")

    try:
        max_attempts = int(values["SCENE_ENGINE_IMAGE_GENERATION_MAX_ATTEMPTS"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SCENE_ENGINE_IMAGE_GENERATION_MAX_ATTEMPTS must be an integer."
        ) from exc
    if max_attempts < 1:
        raise ValueError(
            "SCENE_ENGINE_IMAGE_GENERATION_MAX_ATTEMPTS must be at least 1."
        )

    string_keys = (
        "SCENE_ENGINE_IMAGE_GENERATION_BASE_URL",
        "SCENE_ENGINE_IMAGE_GENERATION_HEALTH_PATH",
        "SCENE_ENGINE_IMAGE_GENERATION_BY_PROMPT_PATH",
    )
    for key in string_keys:
        if not values[key].strip():
            raise ValueError(f"Scene Engine .env key {key} must be a non-empty string.")

    return {
        "base_url": values["SCENE_ENGINE_IMAGE_GENERATION_BASE_URL"].strip(),
        "timeout_s": timeout_s,
        "max_attempts": max_attempts,
        "health_path": values["SCENE_ENGINE_IMAGE_GENERATION_HEALTH_PATH"].strip(),
        "generate_image_by_prompt_path": values[
            "SCENE_ENGINE_IMAGE_GENERATION_BY_PROMPT_PATH"
        ].strip(),
    }
