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
import mimetypes
from pathlib import Path
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

from embodichain.gen_sim.scene_engine.configs.environment import (
    read_scene_engine_env_values,
)


class ArticulatedGenerationClient:
    """Request one articulated asset from articulation-server."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: int,
        max_attempts: int,
        health_path: str,
        generate_path: str,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._health_path = _validate_relative_path(health_path, "health_path")
        self._generate_path = _validate_relative_path(generate_path, "generate_path")
        self._session = session or requests.Session()

    @classmethod
    def from_dotenv(cls) -> "ArticulatedGenerationClient":
        """Create a client from its required ``gen_sim/.env`` settings."""
        return cls(**_load_dotenv_config())

    def check_health(self) -> None:
        """Raise when articulation-server does not report ``ok=true``."""
        response_data = self._request_json("get", self._health_path)
        if response_data.get("ok") is not True:
            raise RuntimeError(
                "Articulated Generation Server health response does not contain ok=true."
            )

    def close(self) -> None:
        """Close the HTTP session owned by this client."""
        self._session.close()

    def generate_articulated_object(
        self,
        *,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Request one articulated asset and return the server JSON response.

        Args:
            prompt: Text description supplied to articulation-server.
            image_path: Optional image observation sent with the prompt.

        Returns:
            JSON object returned by ``/generate_articulation``.

        Raises:
            FileNotFoundError: If ``image_path`` does not identify a file.
            RuntimeError: If the server request or response is invalid.
            ValueError: If the prompt is invalid.
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Articulated generation prompt must not be empty.")

        resolved_image_path = _resolve_optional_image_path(image_path)
        return self._request_generation(
            prompt=prompt,
            image_path=resolved_image_path,
        )

    def generate_articulated_usdc(
        self,
        *,
        prompt: str,
        image_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Generate one articulated USDC asset and write it to ``output_path``."""
        response_data = self.generate_articulated_object(
            prompt=prompt,
            image_path=image_path,
        )
        request_id = response_data.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(
                "Articulated Generation Server response must contain a request_id."
            )

        deadline = time.monotonic() + self._timeout_s
        while True:
            response_data = self._request_json("get", f"/tasks/{request_id}")
            status = response_data.get("status")
            if status == "succeeded":
                break
            if status in {"failed", "cancelled"}:
                raise RuntimeError(
                    "Articulated Generation Server generation failed: "
                    f"{response_data.get('error', 'unknown error')}"
                )
            if status not in {"queued", "running", "waiting"}:
                raise RuntimeError(
                    "Articulated Generation Server returned unknown generation status: "
                    f"{status!r}."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Articulated Generation Server did not finish within "
                    f"{self._timeout_s} seconds."
                )
            time.sleep(1.0)

        result = response_data.get("result")
        artifacts = result.get("artifacts") if isinstance(result, dict) else None
        usdc_path = artifacts.get("usdc") if isinstance(artifacts, dict) else None
        if not isinstance(usdc_path, str) or not usdc_path:
            raise RuntimeError(
                "Articulated Generation Server completed without a usdc."
            )
        _validate_relative_path(usdc_path, "usdc artifact path")

        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output_path = resolved_output_path.with_name(
            f".{resolved_output_path.name}.part"
        )
        try:
            temporary_output_path.write_bytes(self._request_content("get", usdc_path))
            temporary_output_path.replace(resolved_output_path)
        except BaseException:
            temporary_output_path.unlink(missing_ok=True)
            raise
        return resolved_output_path

    def _request_generation(
        self,
        *,
        prompt: str,
        image_path: Path | None,
    ) -> dict[str, Any]:
        if image_path is None:
            return self._request_json(
                "post",
                self._generate_path,
                json={"prompt": prompt},
            )

        with image_path.open("rb") as image_file:
            return self._request_json(
                "post",
                self._generate_path,
                data={"prompt": prompt},
                files={
                    "image": (
                        image_path.name,
                        image_file,
                        mimetypes.guess_type(image_path.name)[0]
                        or "application/octet-stream",
                    )
                },
            )

    def _request_json(
        self,
        method: str,
        path: str,
        **request_kwargs: object,
    ) -> dict[str, Any]:
        content = self._request_content(method, path, **request_kwargs)
        try:
            response_data = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(
                "Articulated Generation Server response is not valid JSON."
            ) from exc
        if not isinstance(response_data, dict):
            raise RuntimeError(
                "Articulated Generation Server response must be a JSON object."
            )
        return response_data

    def _request_content(
        self,
        method: str,
        path: str,
        **request_kwargs: object,
    ) -> bytes:
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                response = getattr(self._session, method)(
                    self._url(path),
                    timeout=self._timeout_s,
                    **request_kwargs,
                )
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Articulated Generation Server request failed after "
            f"{self._max_attempts} attempts."
        ) from last_error

    def _url(self, path: str) -> str:
        return urljoin(self._base_url, path.lstrip("/"))


def _load_dotenv_config() -> dict[str, Any]:
    values = read_scene_engine_env_values(
        "SCENE_ENGINE_ARTICULATED_GENERATION_BASE_URL",
        "SCENE_ENGINE_ARTICULATED_GENERATION_TIMEOUT_S",
        "SCENE_ENGINE_ARTICULATED_GENERATION_MAX_ATTEMPTS",
        "SCENE_ENGINE_ARTICULATED_GENERATION_HEALTH_PATH",
        "SCENE_ENGINE_ARTICULATED_GENERATION_GENERATE_PATH",
    )
    timeout_s = _positive_int(
        values["SCENE_ENGINE_ARTICULATED_GENERATION_TIMEOUT_S"],
        "SCENE_ENGINE_ARTICULATED_GENERATION_TIMEOUT_S",
    )
    max_attempts = _positive_int(
        values["SCENE_ENGINE_ARTICULATED_GENERATION_MAX_ATTEMPTS"],
        "SCENE_ENGINE_ARTICULATED_GENERATION_MAX_ATTEMPTS",
    )
    return {
        "base_url": values["SCENE_ENGINE_ARTICULATED_GENERATION_BASE_URL"].strip(),
        "timeout_s": timeout_s,
        "max_attempts": max_attempts,
        "health_path": values[
            "SCENE_ENGINE_ARTICULATED_GENERATION_HEALTH_PATH"
        ].strip(),
        "generate_path": values[
            "SCENE_ENGINE_ARTICULATED_GENERATION_GENERATE_PATH"
        ].strip(),
    }


def _positive_int(value: str, key: str) -> int:
    try:
        parsed_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc
    if parsed_value < 1:
        raise ValueError(f"{key} must be at least 1.")
    return parsed_value


def _validate_base_url(base_url: str) -> str:
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL.")
    if parsed_url.query or parsed_url.fragment:
        raise ValueError("base_url must not contain a query or fragment.")
    return base_url.rstrip("/") + "/"


def _validate_relative_path(path: str, field_name: str) -> str:
    if not path.strip():
        raise ValueError(f"{field_name} must be a non-empty relative URL path.")
    parsed_url = urlsplit(path)
    if (
        parsed_url.scheme
        or parsed_url.netloc
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError(f"{field_name} must be a relative URL path.")
    return path


def _resolve_optional_image_path(image_path: str | Path | None) -> Path | None:
    if image_path is None:
        return None
    resolved_image_path = Path(image_path).expanduser().resolve()
    if not resolved_image_path.is_file():
        raise FileNotFoundError(
            f"Articulated generation input not found: {resolved_image_path}"
        )
    return resolved_image_path
