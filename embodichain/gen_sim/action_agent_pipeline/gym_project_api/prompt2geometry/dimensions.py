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

import time
from typing import Any

try:
    from .llm_client import OpenAICompatibleClient
except ImportError:
    from llm_client import OpenAICompatibleClient

__all__ = ["DIMENSION_ESTIMATION_SYSTEM_PROMPT", "estimate_real_dimensions"]


DIMENSION_ESTIMATION_SYSTEM_PROMPT = """
<role>
You are a careful real-world object size estimation assistant.
</role>

<task>
Estimate the plausible real-world bounding-box dimensions of one physical object
from the user's object description.
</task>

<dimension_rules>
- Units are meters.
- length_m is the object's longest horizontal dimension.
- width_m is the object's shorter horizontal dimension.
- height_m is the vertical dimension when the object is in its common upright pose.
- Use common real-world size priors for everyday objects.
- If the object category is ambiguous, choose a conservative typical tabletop size.
- Do not include decorative background, shadows, or image canvas in the dimensions.
</dimension_rules>

<output_schema>
{
  "length_m": 0.08,
  "width_m": 0.08,
  "height_m": 0.08,
  "confidence": 0.7,
  "reason": "A typical apple is roughly 8 cm across."
}
</output_schema>

<notes>
- Output JSON only. Do not include markdown or text outside JSON.
- length_m, width_m, height_m, and confidence must be numbers.
- length_m, width_m, and height_m must be positive.
- confidence must be between 0 and 1.
- Keep reason short and specific.
</notes>
""".strip()


def estimate_real_dimensions(
    *,
    object_prompt: str,
    client: OpenAICompatibleClient,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Estimate real-world object dimensions with schema validation and retry."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    messages = [
        {"role": "system", "content": DIMENSION_ESTIMATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Object description:\n"
                f"{object_prompt.strip()}\n\n"
                "Return the dimensions JSON only."
            ),
        },
    ]
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = client.chat_json(messages=messages)
            return _validate_dimension_output(raw)
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            time.sleep(1.0)
    raise ValueError(
        "Failed to estimate object dimensions after " f"{max_attempts} attempts."
    ) from last_error


def _validate_dimension_output(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {"length_m", "width_m", "height_m", "confidence", "reason"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"Unexpected dimension keys: {sorted(extra)}")
    result: dict[str, Any] = {}
    for key in ("length_m", "width_m", "height_m"):
        value = raw.get(key)
        if not isinstance(value, int | float):
            raise ValueError(f"{key} must be a number.")
        value = float(value)
        if value <= 0:
            raise ValueError(f"{key} must be positive.")
        result[key] = value
    confidence = raw.get("confidence")
    if not isinstance(confidence, int | float):
        raise ValueError("confidence must be a number.")
    confidence = float(confidence)
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1.")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    result["confidence"] = confidence
    result["reason"] = reason.strip()
    return result
