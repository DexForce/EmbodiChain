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

"""Backward-compatible VLM facade over the shared chat adapter."""

from __future__ import annotations
from pathlib import Path
from embodichain.gen_sim.services.adapters.chat import ChatCompletionsAdapter
from embodichain.gen_sim.services.contracts import (
    ChatRequest,
    ChatMessage,
    TextPart,
    ImagePart,
)
from .load_config import LLMConfig, load_llm_config

__all__: list[str] = []


class OpenAICompatibleVLM:
    """Preserve the single-image prompt API for existing callers."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._adapter = ChatCompletionsAdapter(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_attempts=config.max_attempts,
            options={"body_defaults": config.default_query},
        )

    @classmethod
    def from_dotenv(cls) -> OpenAICompatibleVLM:
        return cls(load_llm_config())

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path | None = None,
    ) -> str:
        if image_path is not None:
            path = Path(image_path).expanduser().resolve(strict=True)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Image input must be a .jpg, .jpeg, or .png file")
        parts = (TextPart(user_prompt),) + (
            (ImagePart(Path(image_path)),) if image_path is not None else ()
        )
        return self._adapter.complete(
            ChatRequest(
                (
                    ChatMessage("system", (TextPart(system_prompt),)),
                    ChatMessage("user", parts),
                )
            )
        ).text
