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
from langchain_openai import AzureChatOpenAI

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

# Clear proxy if not needed (optional, can be set via environment variables)

os.environ["ALL_PROXY"] = ""
os.environ["all_proxy"] = ""

# Proxy configuration (optional, uncomment if needed)
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# API version (optional, defaults to "2024-10-21" if not set)
# os.environ["OPENAI_API_VERSION"] = "2024-10-21"

# Note: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set via environment variables
# Example in bash:
#   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
#   export AZURE_OPENAI_API_KEY="your-api-key"

# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model="gpt-4o"):
    return AzureChatOpenAI(
        temperature=temperature,
        model=model,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-10-21"),
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
code_llm = _create_llm_safe(temperature=0.0, model="gpt-4o")
validation_llm = _create_llm_safe(temperature=0.0, model="gpt-4o")
view_selection_llm = _create_llm_safe(temperature=0.0, model="gpt-4o")
