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

from langchain_openai import ChatOpenAI

from embodichain.env import (
    DEFAULT_HTTP_PROXY,
    DEFAULT_HTTPS_PROXY,
    get_api_key,
    get_llm_model,
    get_llm_url,
)

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

for proxy_var in ("ALL_PROXY", "all_proxy"):
    os.environ[proxy_var] = ""

for proxy_var, proxy_value in (
    ("HTTP_PROXY", DEFAULT_HTTP_PROXY),
    ("HTTPS_PROXY", DEFAULT_HTTPS_PROXY),
    ("http_proxy", DEFAULT_HTTP_PROXY),
    ("https_proxy", DEFAULT_HTTPS_PROXY),
):
    os.environ[proxy_var] = proxy_value

DEFAULT_LLM_URL = get_llm_url()
DEFAULT_LLM_MODEL = get_llm_model()


def _get_api_key():
    return get_api_key()


# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model=None):
    model = model or DEFAULT_LLM_MODEL
    return ChatOpenAI(
        temperature=temperature,
        model=model,
        api_key=_get_api_key(),
        base_url=DEFAULT_LLM_URL,
    )


# ------------------------------------------------------------------------------
# LLM instances
# ------------------------------------------------------------------------------


# Initialize LLM instances, but handle errors gracefully for documentation builds.
def _create_llm_safe(*, temperature=0.0, model=None):
    try:
        return create_llm(temperature=temperature, model=model)
    except Exception:
        return None


task_llm = _create_llm_safe(temperature=0.0)
recovery_llm = _create_llm_safe(temperature=0.0)
compile_llm = _create_llm_safe(temperature=0.0)

# Backward-compatible aliases for newer branch naming.
failure_anticipation_llm = recovery_llm
code_llm = compile_llm


if __name__ == "__main__":

    def call_llm(prompt, temperature=0.0, model=None):
        llm = create_llm(temperature=temperature, model=model)
        response = llm.invoke(prompt)
        return response.content

    response = call_llm(prompt="Which model you are?", temperature=0.0)
    print(response)
