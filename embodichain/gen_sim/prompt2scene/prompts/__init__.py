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

from . import data
from .base import PromptRenderer

default_prompt_renderer = PromptRenderer(data)

__all__ = ["load_prompt", "load_prompt_data", "render_prompt", "default_prompt_renderer"]


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the bundled prompt data directory."""
    return default_prompt_renderer.load_prompt(prompt_name)


def load_prompt_data(prompt_name: str) -> dict[str, object]:
    """Load a YAML prompt data file from the bundled prompt data directory."""
    return default_prompt_renderer.load_prompt_data(prompt_name)


def render_prompt(
    prompt_name: str,
    values: dict[str, object] | None = None,
    *,
    prompt_key: str | None = None,
) -> str:
    """Load a prompt template and fill optional placeholders."""
    return default_prompt_renderer.render_prompt(
        prompt_name,
        values,
        prompt_key=prompt_key,
    )
