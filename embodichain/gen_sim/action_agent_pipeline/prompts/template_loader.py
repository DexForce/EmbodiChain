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

from pathlib import Path
from string import Template
from typing import Any

__all__ = ["render_prompt_template"]

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def render_prompt_template(name: str, **values: Any) -> str:
    """Render one packaged prompt template with strict variable checking.

    ``string.Template`` is used instead of ``str.format`` so JSON examples can
    keep ordinary braces. Strict substitution also turns a misspelled template
    variable into an immediate configuration error rather than a malformed LLM
    request that is difficult to diagnose later.

    Args:
        name: Template file name relative to the packaged template directory.
        **values: Values substituted for ``$variable`` placeholders.

    Returns:
        The rendered prompt without trailing blank lines.

    Raises:
        ValueError: If ``name`` attempts to escape the template directory.
        FileNotFoundError: If the requested template does not exist.
        KeyError: If a required placeholder value is missing.
    """
    template_path = (_TEMPLATE_DIR / name).resolve()
    if template_path.parent != _TEMPLATE_DIR.resolve():
        raise ValueError(f"Prompt template name must be a file name: {name!r}.")
    if not template_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    source = template_path.read_text(encoding="utf-8")
    rendered = Template(source).substitute(
        {key: str(value) for key, value in values.items()}
    )
    return rendered.rstrip()
