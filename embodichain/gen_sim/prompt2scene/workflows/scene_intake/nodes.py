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

from embodichain.gen_sim.prompt2scene.prompts.schemas import (
    SCENE_INTAKE_JSON_SCHEMA,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeSpec,
)
from embodichain.gen_sim.prompt2scene.utils import (
    log_api_request_start,
    log,
)
from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    SCENE_INTAKE_STEP,
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
    build_scene_intake_messages,
    build_scene_intake_verifier_messages,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.state import (
    SceneIntakeState,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.utils import (
    build_scene_intake_spec,
)

__all__ = [
    "call_vlm_scene_intake_node",
    "call_vlm_verify_scene_intake_node",
    "normalize_scene_intake_node",
    "normalize_verified_scene_intake_node",
    "prepare_input_node",
]


def prepare_input_node(state: SceneIntakeState) -> dict[str, object]:
    """Prepare chat messages for the scene intake model call."""
    return {"messages": build_scene_intake_messages(state["request"])}


def call_vlm_scene_intake_node(
    state: SceneIntakeState,
    *,
    llm: Any,
) -> dict[str, object]:
    """Call the configured VLM for fixed scene intake extraction."""
    attempt_count = state["attempt_count"] + 1

    try:
        log_api_request_start(
            step=SCENE_INTAKE_STEP,
            request="extract",
            attempt=attempt_count,
        )
        artifact_writer = WorkflowArtifactWriter(
            state["request"].output_root,
            SCENE_INTAKE_STEP,
        )
        raw_model_output = call_structured_json_model_step(
            llm=llm,
            schema=SCENE_INTAKE_JSON_SCHEMA,
            messages=state["messages"],
            context="Scene intake",
            attempt_count=attempt_count,
        )
    except StructuredModelCallError as exc:
        error = format_attempt_error("Scene intake", attempt_count, exc)
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


def normalize_scene_intake_node(state: SceneIntakeState) -> dict[str, object]:
    """Normalize raw VLM JSON into a draft scene intake schema."""
    raw_model_output = state["raw_model_output"]
    if raw_model_output is None:
        return {}

    try:
        scene_intake = build_scene_intake_spec(
            request=state["request"],
            model_output=raw_model_output,
        )
    except ValueError as exc:
        error = format_attempt_error("Scene intake", state["attempt_count"], exc)
        return {
            "draft_scene_intake": None,
            "last_error": error,
            "errors": state["errors"] + [error],
        }

    return {"draft_scene_intake": scene_intake, "scene_intake": None}


def call_vlm_verify_scene_intake_node(
    state: SceneIntakeState,
    *,
    llm: Any,
) -> dict[str, object]:
    """Ask VLM to verify and correct scene-intake grouping and counts."""
    draft_scene_intake = state["draft_scene_intake"]
    if draft_scene_intake is None:
        return {}

    attempt_count = state["attempt_count"] + 1
    messages = build_scene_intake_verifier_messages(
        request=state["request"],
        scene_intake=draft_scene_intake,
    )

    try:
        log_api_request_start(
            step=SCENE_INTAKE_STEP,
            request="verify",
            attempt=attempt_count,
        )
        artifact_writer = WorkflowArtifactWriter(
            state["request"].output_root,
            SCENE_INTAKE_STEP,
        )
        raw_model_output = call_structured_json_model_step(
            llm=llm,
            schema=SCENE_INTAKE_JSON_SCHEMA,
            messages=messages,
            context="Scene intake verifier",
            attempt_count=attempt_count,
        )
    except StructuredModelCallError as exc:
        error = format_attempt_error("Scene intake verifier", attempt_count, exc)
        log.log_warning(error)
        return {
            "attempt_count": attempt_count,
            "raw_model_output": None,
            "scene_intake": None,
            "last_error": error,
            "errors": state["errors"] + [error],
        }

    return {
        "attempt_count": attempt_count,
        "raw_model_output": raw_model_output,
        "scene_intake": None,
        "last_error": None,
    }


def normalize_verified_scene_intake_node(
    state: SceneIntakeState,
) -> dict[str, object]:
    """Normalize verifier output into the final scene intake schema."""
    raw_model_output = state["raw_model_output"]
    if raw_model_output is None:
        return {}

    try:
        scene_intake = build_scene_intake_spec(
            request=state["request"],
            model_output=raw_model_output,
        )
    except ValueError as exc:
        error = format_attempt_error(
            "Scene intake verifier", state["attempt_count"], exc
        )
        log.log_warning(error)
        return {
            "scene_intake": None,
            "last_error": error,
            "errors": state["errors"] + [error],
        }

    return {"scene_intake": scene_intake, "last_error": None}
