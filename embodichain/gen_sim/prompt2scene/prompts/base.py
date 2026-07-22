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

from functools import lru_cache
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any, Mapping

import yaml

__all__ = ["PromptRenderer"]


class PromptRenderer:
    """Load and render bundled prompt templates."""

    def __init__(self, package: Any) -> None:
        self._package = package

    @lru_cache(maxsize=None)
    def load_prompt(self, prompt_name: str) -> str:
        """Load a plain-text prompt template by file name."""
        prompt_path = self._get_prompt_path(prompt_name)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt data file not found: {prompt_name}")
        return prompt_path.read_text(encoding="utf-8").strip()

    @lru_cache(maxsize=None)
    def load_prompt_data(self, prompt_name: str) -> dict[str, Any]:
        """Load a YAML prompt data file by file name."""
        prompt_path = self._get_prompt_path(prompt_name)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt data file not found: {prompt_name}")

        prompt_data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        if not isinstance(prompt_data, dict):
            raise ValueError(f"Prompt YAML must contain a mapping: {prompt_name}")
        return prompt_data

    def render_prompt(
        self,
        prompt_name: str,
        values: Mapping[str, object] | None = None,
        *,
        prompt_key: str | None = None,
    ) -> str:
        """Render a prompt template and fill placeholders."""
        if prompt_key is None:
            template = self.load_prompt(prompt_name)
        else:
            prompt_data = self.load_prompt_data(prompt_name)
            template = prompt_data.get(prompt_key)
            if not isinstance(template, str):
                raise KeyError(f"Prompt key {prompt_key!r} not found in {prompt_name}")

        if values is None:
            return template
        return Template(template).safe_substitute(values)

    def _get_prompt_path(self, prompt_name: str) -> Path:
        if "/" in prompt_name or "\\" in prompt_name:
            raise ValueError(f"Prompt name must be a file name: {prompt_name}")
        return resources.files(self._package).joinpath(prompt_name)
