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
import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "extract_json_object",
    "normalize_json_content",
]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(content: str | Mapping[str, Any]) -> dict[str, Any]:
    """Extract a JSON object from plain or fenced LLM content.

    Args:
        content: Raw LLM text, already parsed JSON-like mapping, or markdown fenced
            JSON content.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If no JSON object can be parsed.
    """
    if isinstance(content, Mapping):
        return dict(content)

    text = str(content).strip()
    candidates = [match.group(1).strip() for match in _JSON_FENCE_RE.finditer(text)]
    candidates.append(text)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            if start < 0:
                continue
            try:
                value, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue

        if isinstance(value, dict):
            return value

    raise ValueError("Expected a JSON object in the LLM response.")


def normalize_json_content(content: str | Mapping[str, Any]) -> str:
    """Normalize JSON-like LLM content into stable pretty-printed JSON text."""
    return json.dumps(extract_json_object(content), ensure_ascii=False, indent=2)
