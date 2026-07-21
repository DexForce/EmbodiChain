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

import base64
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from embodichain.gen_sim.scene_engine.llms.load_config import LLMConfig, load_llm_config

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleVLM:
    """Client for multimodal OpenAI-compatible chat-completions endpoints."""

    def __init__(self, config: LLMConfig):
        self._config = config

    @classmethod
    def from_config(
        cls, config_path: str | Path | None = None
    ) -> "OpenAICompatibleVLM":
        """Create a client from the scene-engine LLM configuration."""
        return cls(load_llm_config(config_path))

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path | None = None,
    ) -> str:
        """Send a text or text-and-image chat-completions request."""
        user_content: str | list[dict[str, object]] = user_prompt
        if image_path is not None:
            resolved_image_path = Path(image_path).expanduser().resolve()
            if not resolved_image_path.is_file():
                raise FileNotFoundError(f"Image input not found: {resolved_image_path}")
            if resolved_image_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
                raise ValueError("Image input must be a .jpg, .jpeg, or .png file.")
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(resolved_image_path)},
                },
            ]

        payload = dict(self._config.default_query)
        payload.update(
            {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
            }
        )
        return self._request_chat_completion(payload)

    def _request_chat_completion(self, payload: dict[str, Any]) -> str:
        """Execute a chat-completions HTTP request with transient retries."""
        endpoint = _chat_completions_endpoint(self._config.base_url)
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_attempts + 1):
            try:
                request = Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urlopen(request, timeout=120) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                return _extract_response_text(response_payload)
            except HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code not in _RETRYABLE_HTTP_STATUS_CODES:
                    raise RuntimeError(
                        f"VLM request failed with HTTP {exc.code}: {details}"
                    ) from exc
                last_error = RuntimeError(
                    f"VLM request failed with HTTP {exc.code}: {details}"
                )
            except URLError as exc:
                last_error = RuntimeError(f"VLM request failed: {exc.reason}")
            except (json.JSONDecodeError, ValueError):
                last_error = RuntimeError("VLM API returned a malformed response.")

        assert last_error is not None
        raise last_error


def _image_data_url(image_path: Path) -> str:
    mime_type = (
        "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    )
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def _chat_completions_endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _extract_response_text(response_payload: object) -> str:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "VLM response does not contain choices[0].message.content."
        ) from exc
    if not isinstance(content, str):
        raise ValueError("VLM response content must be a string.")
    return content
