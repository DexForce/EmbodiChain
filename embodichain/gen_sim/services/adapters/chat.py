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

"""Chat Completions adapter; protocol and provider extensions stay here."""

from __future__ import annotations
from .._telemetry import record_attempt
from http.client import IncompleteRead
import base64
import json
import mimetypes
from pathlib import Path
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from ..contracts import ChatRequest, ChatResponse, ImagePart, TextPart
from ..errors import (
    ServiceConfigurationError,
    ServiceResponseError,
    ServiceTimeout,
    ServiceTransportError,
)

__all__: list[str] = []


def image_url(source: Path | str) -> str:
    if isinstance(source, str) and source.startswith("data:"):
        return source
    path = Path(source).expanduser().resolve(strict=True)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def image_semantic_defaults(
    settings: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    """Preserve the upstream image planner's bounded MiMo response policy."""
    if (
        "mimo" in settings["model"].casefold()
        or "xiaomimimo.com" in settings["base_url"].casefold()
    ):
        return {
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_completion_tokens": 16000,
            **defaults,
        }
    return dict(defaults)


class ChatCompletionsAdapter:
    """Stateless, concurrent-call-safe OpenAI-compatible HTTP adapter.

    No SDK retry layer exists. Each call owns its transport. ``options`` are
    provider body options; the historical Scene default_query is a body overlay,
    whereas text/SimReady default_query remains an HTTP query mapping.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "",
        timeout_s: float = 120,
        max_attempts: int = 1,
        options: dict[str, Any] | None = None,
    ) -> None:
        if not model or not api_key:
            raise ServiceConfigurationError("Chat requires a model and API key")
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.options = dict(options or {})

    def complete(self, request: ChatRequest) -> ChatResponse:
        from urllib.parse import urlencode

        options = dict(self.options)
        query = options.pop("default_query", {})
        payload = dict(options.pop("body_defaults", {}))
        payload.update(options)
        messages = []
        for message in request.messages:
            parts = [
                (
                    {"type": "text", "text": part.text}
                    if isinstance(part, TextPart)
                    else {
                        "type": "image_url",
                        "image_url": {"url": image_url(part.source)},
                    }
                )
                for part in message.content
            ]
            content = (
                parts[0]["text"]
                if len(parts) == 1 and parts[0]["type"] == "text"
                else parts
            )
            messages.append({"role": message.role, "content": content})
        model = request.model or self.model
        payload.update(model=model, messages=messages)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.schema is not None:
            payload["response_format"] = {"type": "json_object"}
            schema_text = (
                "\nReturn one JSON object conforming exactly to this JSON Schema:\n"
                + json.dumps(dict(request.schema), ensure_ascii=False)
            )
            messages.append({"role": "user", "content": schema_text})
        if "mimo" in model.casefold() or "xiaomimimo.com" in self.base_url.casefold():
            payload.setdefault("thinking", {"type": "disabled"})
            payload.setdefault("max_completion_tokens", 4096)
        endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else self.base_url + "/chat/completions"
        )
        if query:
            endpoint += ("&" if "?" in endpoint else "?") + urlencode(query)
        for attempt in range(1, self.max_attempts + 1):
            try:
                http_request = Request(
                    endpoint,
                    data=json.dumps(payload).encode(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                record_attempt()
                with urlopen(http_request, timeout=self.timeout_s) as response:
                    raw = json.loads(response.read())
                text = raw["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise ValueError("Expected string model content")
                structured = None
                if request.schema is not None:
                    decoded = json.loads(text)
                    if not isinstance(decoded, dict):
                        raise ValueError("Expected JSON object")
                    structured = decoded
                return ChatResponse(
                    text=text, structured=structured, usage=raw.get("usage")
                )
            except HTTPError as error:
                # Never expose body/URL/key through public failures.
                if (
                    error.code not in {408, 429, 500, 502, 503, 504}
                    or attempt == self.max_attempts
                ):
                    raise ServiceTransportError(
                        f"Chat request failed with HTTP {error.code}", attempts=attempt
                    ) from None
            except (TimeoutError, socket.timeout):
                if attempt == self.max_attempts:
                    raise ServiceTimeout(
                        "Chat request timed out", attempts=attempt
                    ) from None
            except (URLError, OSError, IncompleteRead):
                if attempt == self.max_attempts:
                    raise ServiceTransportError(
                        "Chat transport failed", attempts=attempt
                    ) from None
            except (ValueError, KeyError, IndexError, TypeError):
                raise ServiceResponseError(
                    "Chat response is malformed", attempts=attempt
                ) from None
        raise AssertionError("positive attempt budget required")
