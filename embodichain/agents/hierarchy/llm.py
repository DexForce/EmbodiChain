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

# os.environ['ALL_PROXY'] = ''
# os.environ['all_proxy'] = ''
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['LLM_URL'] = 'https://api.shubiaobiao.cn/v1/'
# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model="gpt-4o"):
    return ChatOpenAI(
        temperature=temperature,
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_URL"),
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


task_llm = _create_llm_safe(temperature=0.0, model="gpt-4o")
failure_anticipation_llm = _create_llm_safe(temperature=0.0, model="gpt-4o")
code_llm = _create_llm_safe(temperature=0.0, model="gpt-4o")

if __name__ == "__main__":
    def call_llm(prompt, temperature=0.0, model="gpt-4o"):
        llm = create_llm(temperature=temperature, model=model)
        response = llm.invoke(prompt)
        return response.content

    response = call_llm(prompt="Which model you are?", temperature=0.0, model="gpt-4o")
    print(response)