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
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "mimo-v2.5"
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192
DEFAULT_PRECOMPILE_RECOVERY_ACTIONS = False
DEFAULT_PRECOMPILE_RECOVERY_MONITORS = False
DEFAULT_API_KEY = "tp-cigd9h4eh33v79adk5wz77y9o9rngsvcrw527wxiic5jdeqq"
DEFAULT_LLM_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_HTTPS_PROXY = "http://127.0.0.1:7897"
DEFAULT_HTTP_PROXY = "http://127.0.0.1:7897"
# ------------------------------------------------------------------------------
# LLM factory
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

def create_llm(*, temperature=0.0, model=None):
    return ChatOpenAI(
        temperature=temperature,
        model=DEFAULT_LLM_MODEL,
        api_key=DEFAULT_API_KEY,
        base_url=DEFAULT_LLM_URL,
    )


# ------------------------------------------------------------------------------
# LLM instances
# ------------------------------------------------------------------------------


# Initialize LLM instances, but handle errors gracefully for documentation builds
def _create_llm_safe(*, temperature=0.0, model="gpt-4o"):
    try:
        return create_llm(temperature=temperature, model=model)
    except Exception:
        return None


task_llm = _create_llm_safe(temperature=0.0, model="gpt-5")
recovery_llm = _create_llm_safe(temperature=0.0, model="gpt-5")
compile_llm = _create_llm_safe(temperature=0.0, model="gpt-5")

if __name__ == "__main__":
    def call_llm(prompt, temperature=0.0, model="gpt-4o"):
        llm = create_llm(temperature=temperature, model=model)
        response = llm.invoke(prompt)
        return response.content

    response = call_llm(prompt="Which model you are?", temperature=0.0, model="gpt-5")
    print(response)