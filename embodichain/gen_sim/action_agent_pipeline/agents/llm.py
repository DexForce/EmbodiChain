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

from embodichain.gen_sim.action_agent_pipeline.utils.mllm import create_chat_openai

__all__ = ["create_llm", "task_llm", "compile_llm"]


# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model=None, usage_stage=None):
    return create_chat_openai(
        temperature=temperature,
        model=model,
        usage_stage=usage_stage,
    )


# ------------------------------------------------------------------------------
# LLM instances
# ------------------------------------------------------------------------------


# Initialize LLM instances, but handle errors gracefully for documentation builds
def _create_llm_safe(*, temperature=0.0, model=None, usage_stage=None):
    try:
        return create_llm(
            temperature=temperature,
            model=model,
            usage_stage=usage_stage,
        )
    except Exception:
        return None


task_llm = _create_llm_safe(
    temperature=0.0,
    usage_stage="action_agent.task_graph",
)
compile_llm = _create_llm_safe(
    temperature=0.0,
    usage_stage="action_agent.compile_canonicalize",
)

if __name__ == "__main__":

    def call_llm(prompt, temperature=0.0, model=None):
        llm = create_llm(
            temperature=temperature,
            model=model,
            usage_stage="action_agent.debug",
        )
        response = llm.invoke(prompt)
        return response.content

    response = call_llm(prompt="Which model you are?", temperature=0.0)
    print(response)
