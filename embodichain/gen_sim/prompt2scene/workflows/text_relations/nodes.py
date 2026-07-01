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

from embodichain.gen_sim.prompt2scene.workflows.request import InputKind
from embodichain.gen_sim.prompt2scene.prompts.schemas import (
    TEXT_RELATIONS_JSON_SCHEMA,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.schema import (
    TextRelationSpec,
)
from embodichain.gen_sim.prompt2scene.utils import (
    log_api_request_start,
    log,
)
from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    TEXT_RELATIONS_STEP,
    WorkflowArtifactWriter,
)
from embodichain.gen_sim.prompt2scene.llms.llm_output import (
    StructuredModelCallError,
    call_structured_json_model_step,
)
from embodichain.gen_sim.prompt2scene.workflows.stage_errors import (
    format_attempt_error,
)
from embodichain.gen_sim.prompt2scene.prompts.builders import (
    build_text_relation_messages,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.state import (
    TextRelationsState,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.utils import (
    build_text_relation_spec,
)

__all__ = [
    "call_llm_text_relations_node",
    "normalize_text_relations_node",
    "prepare_text_relation_messages_node",
]


def prepare_text_relation_messages_node(
    state: TextRelationsState,
) -> dict[str, object]:
    """Prepare text-relation extraction messages."""
    request = state["request"]
    if request.input_kind != InputKind.TEXT:
        raise ValueError("Text relations requires a text input.")
    return {
        "messages": build_text_relation_messages(
            request=request,
            scene_intake=state["scene_intake"],
        )
    }


def call_llm_text_relations_node(
    state: TextRelationsState,
    *,
    llm: Any,
) -> dict[str, object]:
    """Call LLM to extract explicit text spatial constraints."""
    attempt_count = state["attempt_count"] + 1
    artifact_writer = WorkflowArtifactWriter(
        state["output_root"],
        TEXT_RELATIONS_STEP,
    )

    try:
        log_api_request_start(
            step=TEXT_RELATIONS_STEP,
            request="extract",
            attempt=attempt_count,
        )
        raw_model_output = call_structured_json_model_step(
            llm=llm,
            schema=TEXT_RELATIONS_JSON_SCHEMA,
            messages=state["messages"],
            context="Text relations",
            
            
            attempt_count=attempt_count,
            
            
        )
    except StructuredModelCallError as exc:
        error = format_attempt_error("Text relations", attempt_count, exc)
        log.log_warning(error)
        return {
            "attempt_count": attempt_count,
            "raw_model_output": None,
            "last_error": error,
            "errors": state["errors"] + [error],
        }

    return {
        "attempt_count": attempt_count,
        "raw_model_output": raw_model_output,
        "last_error": None,
    }


def normalize_text_relations_node(state: TextRelationsState) -> dict[str, object]:
    """Normalize raw LLM output into TextRelationSpec."""
    raw_model_output = state["raw_model_output"]
    if raw_model_output is None:
        return {}

    try:
        text_relations = build_text_relation_spec(
            scene_intake=state["scene_intake"],
            model_output=raw_model_output,
        )
    except ValueError as exc:
        error = format_attempt_error("Text relations", state["attempt_count"], exc)
        log.log_warning(error)
        return {
            "text_relations": None,
            "last_error": error,
            "errors": state["errors"] + [error],
        }

    artifact_writer = WorkflowArtifactWriter(
        state["output_root"],
        TEXT_RELATIONS_STEP,
    )
    artifact_writer.write_step_result(text_relations.to_manifest())
    return {"text_relations": text_relations, "last_error": None}
