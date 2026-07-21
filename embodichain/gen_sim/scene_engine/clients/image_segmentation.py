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

import requests

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "scene_engine_config.json"
)


class ImageSegmentationClient:

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: int,
        health_path: str,
        segment_single_object_path: str,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._health_path = health_path
        self._segment_single_object_path = segment_single_object_path
        self._session = session or requests.Session()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
    ) -> "ImageSegmentationClient":
        config = _load_config(config_path)
        return cls(**config)

    def check_health(self) -> None:
        response = self._session.get(
            self._url(self._health_path),
            timeout=self._timeout_s,
        )
        response.raise_for_status()

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

        with resolved_image_path.open("rb") as image_file:
            response = self._session.post(
                self._url(self._segment_single_object_path),
                data={"prompt": prompt},
                files={"image": (resolved_image_path.name, image_file)},
                timeout=self._timeout_s,
            )
        response.raise_for_status()

        try:
            response_data = response.json()
        except ValueError as exc:
            raise ValueError(
                "Image Segmentation Server response is not valid JSON."
            ) from exc
        if not isinstance(response_data, dict):
            raise ValueError(
                "Image Segmentation Server response must be a JSON object."
            )
        if response_data.get("ok") is False:
            raise RuntimeError(
                "Image Segmentation Server request failed: "
                f"{response_data.get('error', 'unknown error')}"
            )
        return _extract_rle_masks(response_data)

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    resolved_config_path = Path(config_path or _DEFAULT_CONFIG_PATH).expanduser()
    resolved_config_path = resolved_config_path.resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Config not found: {resolved_config_path}")

    try:
        config_data = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config is not valid JSON: {resolved_config_path}") from exc

    config = config_data.get("image_segmentation")
    if not isinstance(config, dict):
        raise ValueError("Config key image_segmentation must be an object.")

    required_keys = (
        "base_url",
        "timeout_s",
        "health_path",
        "segment_single_object_path",
    )
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing Image Segmentation Server config keys: {missing}")

    try:
        timeout_s = int(config["timeout_s"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Image Segmentation Server config timeout_s must be an integer."
        ) from exc
    if timeout_s < 1:
        raise ValueError(
            "Image Segmentation Server config timeout_s must be at least 1."
        )

    string_keys = ("base_url", "health_path", "segment_single_object_path")
    for key in string_keys:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(
                f"Image Segmentation Server config key {key} must be a non-empty string."
            )

    return {
        "base_url": config["base_url"].strip(),
        "timeout_s": timeout_s,
        "health_path": config["health_path"].strip(),
        "segment_single_object_path": config["segment_single_object_path"].strip(),
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
