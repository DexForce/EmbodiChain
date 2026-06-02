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

import os
from pathlib import Path

from langchain_openai import ChatOpenAI

__all__ = ["create_llm", "task_llm", "recovery_llm", "compile_llm"]

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "gpt-4o"
ENV_FILE_NAMES = (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env")


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if it exists."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


for env_file in dict.fromkeys(ENV_FILE_NAMES):
    _load_env_file(env_file)

proxy_url = os.getenv("EMBODICHAIN_LLM_PROXY") or os.getenv("LLM_PROXY_URL")
if proxy_url:
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url


def _get_first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model=None):
    api_key = _get_first_env("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required. Set it in your shell environment or "
            "in a local .env file copied from .env.example."
        )

    kwargs = {
        "temperature": temperature,
        "model": model
        or _get_first_env("OPENAI_MODEL", "LLM_MODEL", default=DEFAULT_LLM_MODEL),
        "api_key": api_key,
    }
    base_url = _get_first_env("OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


# ------------------------------------------------------------------------------
# LLM instances
# ------------------------------------------------------------------------------


# Initialize LLM instances, but handle errors gracefully for documentation builds
def _create_llm_safe(*, temperature=0.0, model=None):
    try:
        return create_llm(temperature=temperature, model=model)
    except Exception:
        return None


task_llm = _create_llm_safe(temperature=0.0)
recovery_llm = _create_llm_safe(temperature=0.0)
compile_llm = _create_llm_safe(temperature=0.0)

if __name__ == "__main__":

    def call_llm(prompt, temperature=0.0, model=None):
        llm = create_llm(temperature=temperature, model=model)
        response = llm.invoke(prompt)
        return response.content

    response = call_llm(prompt="Which model you are?", temperature=0.0)
    print(response)
