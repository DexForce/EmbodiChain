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

"""Shared JSON LLM boundary for config-generation semantics.

The primary config-generation path requests one combined task interpretation.
Legacy route-specific helpers use this same boundary so their compatibility
entry points retain identical rendering, invocation, and parsing behavior.
Only the template, usage-tracking stage, and system framing differ.

Centralizing the call keeps that boundary narrow and auditable: the model
selects *semantic* intent only, while slot geometry, support heights, arm
assignment, and every numeric target stay in deterministic code so generated
configs remain reproducible.
"""

from __future__ import annotations

import json
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.prompts.template_loader import (
    render_prompt_template,
)

__all__ = ["SPEC_SYSTEM_MESSAGE", "ROUTER_SYSTEM_MESSAGE", "request_json_spec"]

SPEC_SYSTEM_MESSAGE = (
    "You produce strict JSON specs for simulation config "
    "generation. Do not include markdown."
)
ROUTER_SYSTEM_MESSAGE = (
    "You are a strict JSON router for simulation config "
    "generation. Return only the requested JSON object."
)


def request_json_spec(
    *,
    template_name: str,
    usage_stage: str,
    project_name: str,
    task_description: str,
    scene_summary: list[dict[str, Any]],
    model: str | None,
    system_message: str = SPEC_SYSTEM_MESSAGE,
) -> dict[str, Any]:
    """Render ``template_name`` and return the model's JSON object.

    Args:
        template_name: Prompt template under ``prompts/templates``. The template
            owns all model-facing prose so wording can change without touching
            the invariants enforced by the caller's normalization code.
        usage_stage: Token-accounting label, such as
            ``config_generation.task_interpretation``.
        project_name: Scene project name interpolated into the template.
        task_description: Natural-language task goal.
        scene_summary: Per-object summary rows serialized into the template.
        model: Optional model override; ``None`` uses the configured default.
        system_message: System framing. Routes classifying a task pass
            :data:`ROUTER_SYSTEM_MESSAGE`; spec generators use the default.

    Returns:
        The parsed JSON object. Schema validation is the caller's job.
    """
    # Imported lazily so config generation stays importable (and testable)
    # without the LLM client stack installed.
    from langchain_core.messages import HumanMessage, SystemMessage

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
        extract_json_object,
    )
    from embodichain.gen_sim.action_agent_pipeline.utils.mllm import (
        create_chat_openai,
    )

    prompt = render_prompt_template(
        template_name,
        project_name=project_name,
        task_description=task_description,
        scene_summary=json.dumps(scene_summary, ensure_ascii=False, indent=2),
    )
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage=usage_stage,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, "content", response)
    return extract_json_object(content)
