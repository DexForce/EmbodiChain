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
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import record_llm_usage

__all__ = ["OpenAICompatibleClient", "OpenAICompatibleClientError"]


class OpenAICompatibleClientError(RuntimeError):
    """Raised when an OpenAI-compatible chat request fails."""


class OpenAICompatibleClient:
    """Small OpenAI-compatible chat completions client."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 120.0,
        usage_stage: str | None = None,
    ):
        if not api_key.strip():
            raise ValueError("LLM api_key must be non-empty.")
        if not model.strip():
            raise ValueError("LLM model must be non-empty.")
        if not base_url.strip():
            raise ValueError("LLM base_url must be non-empty.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.usage_stage = usage_stage or "prompt2geometry.chat_json"

    def chat_json(self, *, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call chat completions and return the decoded JSON response content."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAICompatibleClientError(
                f"LLM request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise OpenAICompatibleClientError(
                f"LLM server is unreachable at {request.full_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OpenAICompatibleClientError(
                f"LLM request timed out after {self.timeout_s}s."
            ) from exc

        try:
            decoded = json.loads(body)
            choice = decoded["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise OpenAICompatibleClientError(
                f"LLM response has unsupported format: {body}"
            ) from exc
        record_llm_usage(
            stage=self.usage_stage,
            provider="openai_compatible_http",
            model=str(decoded.get("model") or self.model),
            usage=decoded.get("usage") if isinstance(decoded, dict) else None,
            request_id=str(decoded.get("id")) if decoded.get("id") else None,
            finish_reason=(
                str(choice.get("finish_reason"))
                if isinstance(choice, dict) and choice.get("finish_reason")
                else None
            ),
            raw_usage=(
                decoded.get("usage")
                if isinstance(decoded, dict) and isinstance(decoded.get("usage"), dict)
                else None
            ),
        )
        if not isinstance(content, str):
            raise OpenAICompatibleClientError("LLM message content must be a string.")
        return _parse_json_text(content)


def _parse_json_text(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output must be a JSON object.")
    return parsed
