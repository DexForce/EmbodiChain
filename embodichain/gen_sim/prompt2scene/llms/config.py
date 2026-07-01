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

from dataclasses import dataclass, field

__all__ = [
    "OpenAICompatibleLLMCfg",
]


@dataclass(frozen=True)
class OpenAICompatibleLLMCfg:
    """OpenAI-compatible LLM configuration."""

    api_key: str
    model: str
    base_url: str
    default_query: dict[str, str] = field(default_factory=dict)
    max_attempts: int = 3

    def to_manifest(self) -> dict[str, object]:
        """Convert the LLM config to a JSON-safe manifest.

        Returns:
            LLM config metadata with sensitive values removed.
        """
        return {
            "provider": "openai_compatible",
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "default_query": self.default_query,
            "max_attempts": self.max_attempts,
        }
