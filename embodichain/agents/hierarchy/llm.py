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

import os
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

DEBUG_LLM_MODEL = "mimo-v2.5"
DEBUG_LLM_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEBUG_OPENAI_API_KEY = "tp-cigd9h4eh33v79adk5wz77y9o9rngsvcrw527wxiic5jdeqq"

DEBUG_PROXY_URL = "http://127.0.0.1:7897"

for proxy_var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(proxy_var, None)

if DEBUG_PROXY_URL:
    os.environ["HTTP_PROXY"] = DEBUG_PROXY_URL
    os.environ["HTTPS_PROXY"] = DEBUG_PROXY_URL

# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model=None):
    return ChatOpenAI(
        temperature=temperature,
        model=model or DEBUG_LLM_MODEL,
        api_key=DEBUG_OPENAI_API_KEY,
        base_url=DEBUG_LLM_URL,
    )


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