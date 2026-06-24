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

from typing import Any
import json

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _BasketTaskRoles,
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _display_noun,
    _normalize_runtime_uid,
)

__all__ = [
    "_call_role_llm",
    "_refine_roles_with_llm",
]


def _refine_roles_with_llm(
    *,
    roles: _BasketTaskRoles,
    scene_objects: list[_SceneObject],
    project_name: str,
    model: str | None,
) -> _BasketTaskRoles:
    response = _call_role_llm(
        project_name=project_name,
        scene_summary=[
            {
                "source_uid": obj.source_uid,
                "role": obj.source_role,
                "mesh": obj.config.get("shape", {}).get("fpath"),
                "init_pos": obj.config.get("init_pos"),
            }
            for obj in scene_objects
        ],
        default_roles={
            "container_object": roles.container_source_uid,
            "left_target_object": roles.left_target_source_uid,
            "right_target_object": roles.right_target_source_uid,
            "target_noun": roles.target_noun,
            "container_runtime_uid": roles.container_runtime_uid,
        },
        model=model,
    )
    source_uids = {obj.source_uid for obj in scene_objects}
    left_target = str(response.get("left_target_object", roles.left_target_source_uid))
    right_target = str(
        response.get("right_target_object", roles.right_target_source_uid)
    )
    container = str(response.get("container_object", roles.container_source_uid))
    for uid in (left_target, right_target, container):
        if uid not in source_uids:
            raise ValueError(f"LLM returned unknown source uid: {uid!r}")
    if len({left_target, right_target, container}) != 3:
        raise ValueError("LLM role mapping must use three distinct source objects.")

    target_noun = _normalize_runtime_uid(
        str(response.get("target_noun", roles.target_noun))
    )
    container_runtime_uid = _normalize_runtime_uid(
        str(response.get("container_runtime_uid", roles.container_runtime_uid))
    )
    return _BasketTaskRoles(
        table_source_uid=roles.table_source_uid,
        container_source_uid=container,
        left_target_source_uid=left_target,
        right_target_source_uid=right_target,
        container_runtime_uid=container_runtime_uid,
        left_target_runtime_uid=f"left_{target_noun}",
        right_target_runtime_uid=f"right_{target_noun}",
        target_noun=target_noun,
        left_target_noun=target_noun,
        right_target_noun=target_noun,
        container_noun=_display_noun(container_runtime_uid),
    )


def _call_role_llm(
    *,
    project_name: str,
    scene_summary: list[dict[str, Any]],
    default_roles: dict[str, Any],
    model: str | None,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
        extract_json_object,
    )
    from embodichain.gen_sim.action_agent_pipeline.utils.mllm import (
        create_chat_openai,
    )

    prompt = (
        "Identify roles for a fixed Dual-UR5 basket-placement simulation task. "
        "Return only one JSON object with keys: container_object, "
        "left_target_object, right_target_object, target_noun, "
        "container_runtime_uid. Use only source_uid values from the scene. The "
        "rotated robot-view left target starts on the negative-y side, and the "
        "rotated robot-view right target starts on the positive-y side.\n\n"
        f"Project: {project_name}\n"
        f"Scene objects:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}\n"
        f"Default roles:\n{json.dumps(default_roles, ensure_ascii=False, indent=2)}"
    )
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage="config_generation.role_refinement",
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You produce strict JSON role mappings for simulation config "
                    "generation. Do not include markdown."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, "content", response)
    return extract_json_object(content)
