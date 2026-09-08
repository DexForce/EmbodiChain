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


class ImageSegmentationClient:

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: int,
        max_attempts: int,
        health_path: str,
        segment_by_prompt_path: str,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._health_path = health_path
        self._segment_by_prompt_path = segment_by_prompt_path
        self._session = session or requests.Session()

    @classmethod
    def from_dotenv(cls) -> "ImageSegmentationClient":
        """Create a client from its required ``gen_sim/.env`` settings."""
        return cls(**_load_dotenv_config())

    def check_health(self) -> None:
        last_error: requests.RequestException | None = None
        for _ in range(self._max_attempts):
            try:
                response = self._session.get(
                    self._url(self._health_path),
                    timeout=self._timeout_s,
                )
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Image Segmentation Server health check failed after "
            f"{self._max_attempts} attempts."
        ) from last_error

    def close(self) -> None:
        self._session.close()

    def segment_single_object(
        self,
        *,
        image_path: str | Path,
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Segment one prompted concept and return its RLE masks.
        The returned list contains only RLE dictionaries, one per mask.
        """
        resolved_image_path = Path(image_path).expanduser().resolve()
        if not resolved_image_path.is_file():
            raise FileNotFoundError(
                f"Image segmentation input not found: {resolved_image_path}"
            )
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Image segmentation prompt must not be empty.")

        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                with resolved_image_path.open("rb") as image_file:
                    response = self._session.post(
                        self._url(self._segment_by_prompt_path),
                        data={"prompt": prompt},
                        files={"image": (resolved_image_path.name, image_file)},
                        timeout=self._timeout_s,
                    )
                response.raise_for_status()

                try:
                    response_data = response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        "Image Segmentation Server response is not valid JSON."
                    ) from exc
                if not isinstance(response_data, dict):
                    raise RuntimeError(
                        "Image Segmentation Server response must be a JSON object."
                    )
                if response_data.get("ok") is False:
                    raise RuntimeError(
                        "Image Segmentation Server request failed: "
                        f"{response_data.get('error', 'unknown error')}"
                    )
                return _extract_rle_masks(response_data)
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Image Segmentation Server request failed after "
            f"{self._max_attempts} attempts."
        ) from last_error

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"


def _load_dotenv_config() -> dict[str, Any]:
    values = read_scene_engine_env_values(
        "SCENE_ENGINE_IMAGE_SEGMENTATION_BASE_URL",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_TIMEOUT_S",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_MAX_ATTEMPTS",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_HEALTH_PATH",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_BY_PROMPT_PATH",
    )
    try:
        timeout_s = int(values["SCENE_ENGINE_IMAGE_SEGMENTATION_TIMEOUT_S"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SCENE_ENGINE_IMAGE_SEGMENTATION_TIMEOUT_S must be an integer."
        ) from exc
    if timeout_s < 1:
        raise ValueError(
            "SCENE_ENGINE_IMAGE_SEGMENTATION_TIMEOUT_S must be at least 1."
        )

    try:
        max_attempts = int(values["SCENE_ENGINE_IMAGE_SEGMENTATION_MAX_ATTEMPTS"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SCENE_ENGINE_IMAGE_SEGMENTATION_MAX_ATTEMPTS must be an integer."
        ) from exc
    if max_attempts < 1:
        raise ValueError(
            "SCENE_ENGINE_IMAGE_SEGMENTATION_MAX_ATTEMPTS must be at least 1."
        )

    string_keys = (
        "SCENE_ENGINE_IMAGE_SEGMENTATION_BASE_URL",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_HEALTH_PATH",
        "SCENE_ENGINE_IMAGE_SEGMENTATION_BY_PROMPT_PATH",
    )
    for key in string_keys:
        if not values[key].strip():
            raise ValueError(f"Scene Engine .env key {key} must be a non-empty string.")

    return {
        "base_url": values["SCENE_ENGINE_IMAGE_SEGMENTATION_BASE_URL"].strip(),
        "timeout_s": timeout_s,
        "max_attempts": max_attempts,
        "health_path": values["SCENE_ENGINE_IMAGE_SEGMENTATION_HEALTH_PATH"].strip(),
        "segment_by_prompt_path": values[
            "SCENE_ENGINE_IMAGE_SEGMENTATION_BY_PROMPT_PATH"
        ].strip(),
    }


def _extract_rle_masks(response_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract RLE masks from accepted Image Segmentation Server layouts."""
    result_data = response_data.get("result") or response_data.get("data")
    if not isinstance(result_data, dict):
        result_data = response_data

    masks = result_data.get("masks")
    if isinstance(masks, list):
        rle_masks = [mask for mask in masks if isinstance(mask, dict)]
        if rle_masks:
            return rle_masks

    instances = result_data.get("instances", [])
    if isinstance(instances, list):
        rle_masks: list[dict[str, Any]] = []
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            mask = (
                instance.get("mask_rle")
                or instance.get("mask")
                or instance.get("segmentation")
            )
            if isinstance(mask, dict):
                rle_masks.append(mask)
        return rle_masks

    return []
