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

from embodichain.gen_sim.simready_pipeline.configs import (
    DEFAULT_LLM_MODEL,
    get_openai_compatible_llm_config,
)
from langchain_openai import ChatOpenAI

__all__ = ["create_llm", "task_llm", "recovery_llm", "compile_llm"]


# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model=None):
    cfg = get_openai_compatible_llm_config(required=True)
    api_key = cfg["api_key"]
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required. Set it in your shell environment, "
            ".env, embodichain/gen_sim/simready_pipeline/configs/.env, "
            "or embodichain/gen_sim/simready_pipeline/configs/gen_config.json."
        )

    proxy_url = cfg.get("proxy_url")
    if proxy_url:
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url

    kwargs = {
        "temperature": temperature,
        "model": model or cfg.get("model") or DEFAULT_LLM_MODEL,
        "api_key": api_key,
    }
    base_url = cfg.get("base_url")
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
