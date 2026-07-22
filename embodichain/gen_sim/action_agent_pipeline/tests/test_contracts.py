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

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    AGENT_CONFIG_FILENAME,
    ATOM_ACTIONS_FILENAME,
    BASIC_BACKGROUND_FILENAME,
    FAST_GYM_CONFIG_FILENAME,
    MAX_COORDINATED_PAYLOADS,
    SUCCESS_TERM_TYPES,
    TASK_GRAPH_FILENAME,
    TASK_PROMPT_FILENAME,
    TASK_ROUTES,
    SuccessTerm,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_agent_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _object_in_container_success,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_router import (
    _route_task_with_llm,
)


def test_generated_agent_config_uses_shared_artifact_names() -> None:
    config = make_agent_config()

    assert config["TaskAgent"]["precomputed_task_graph"] == TASK_GRAPH_FILENAME
    prompt_kwargs = config["Agent"]["prompt_kwargs"]
    assert prompt_kwargs["task_prompt"]["name"] == TASK_PROMPT_FILENAME
    assert prompt_kwargs["basic_background"]["name"] == BASIC_BACKGROUND_FILENAME
    assert prompt_kwargs["atom_actions"]["name"] == ATOM_ACTIONS_FILENAME


def test_public_artifact_names_remain_backward_compatible() -> None:
    assert FAST_GYM_CONFIG_FILENAME == "fast_gym_config.json"
    assert AGENT_CONFIG_FILENAME == "agent_config.json"
    assert TASK_GRAPH_FILENAME == "task_graph.json"


def test_domain_contract_contains_supported_pipeline_values() -> None:
    assert "arrangement_line" in TASK_ROUTES
    assert MAX_COORDINATED_PAYLOADS == 4
    assert SuccessTerm.OBJECTS_ORDERED in SUCCESS_TERM_TYPES


def test_task4_arrangement_route_stays_compatible_with_shared_contract() -> None:
    scene_objects = [
        _SceneObject(
            source_uid=uid,
            source_role="rigid_object",
            config={"description": "罐头"},
        )
        for uid in ("can_1", "can_2")
    ]

    route = _route_task_with_llm(
        scene_objects=scene_objects,
        project_name="task4_2",
        task_description="将罐头摆成一排",
        model=None,
        task_router_llm_caller=lambda **_: {
            "route": "arrangement_line",
            "confidence": 1.0,
            "reason": "The objects should form one row.",
            "candidate_objects": ["can_1", "can_2"],
        },
    )

    assert route.route == "arrangement_line"
    assert route.candidate_objects == ("can_1", "can_2")


def test_generated_success_terms_use_central_defaults_and_type_names() -> None:
    term = _object_in_container_success("can_1", "basket")

    assert term == {
        "type": SuccessTerm.OBJECT_IN_CONTAINER,
        "object": "can_1",
        "container": "basket",
        "radius": 0.2,
        "min_z_offset": -0.05,
        "max_z_offset": 0.35,
    }
