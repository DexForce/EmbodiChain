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

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from embodichain.gen_sim.action_engine.domain import TASK_AGENT_SCHEMA
from embodichain.gen_sim.action_engine.planning import plan_task
from embodichain.gen_sim.action_engine.planning import planner as planner_module


def _scene() -> list[dict[str, Any]]:
    return [
        {
            "uid": "table",
            "runtime_uid": "table",
            "source_uid": "table",
            "role": "background",
            "description": "A table.",
        },
        *[
            {
                "uid": f"interact_soda_can_{index}_0",
                "runtime_uid": f"interact_soda_can_{index}",
                "source_uid": f"interact_soda_can_{index}_0",
                "role": "rigid_object",
                "description": "An aluminum soda can.",
            }
            for index in range(5)
        ],
    ]


def _dual_arm_scene() -> list[dict[str, Any]]:
    return [
        {
            "uid": uid,
            "runtime_uid": uid,
            "source_uid": uid,
            "role": "rigid_object",
            "description": description,
        }
        for uid, description in (
            ("cube", "A cube on the left side of the table."),
            ("cup", "A paper cup on the right side of the table."),
            ("basket", "A basket near the center of the table."),
        )
    ]


def _stack_scene() -> list[dict[str, Any]]:
    return [
        {
            "uid": uid,
            "runtime_uid": uid,
            "source_uid": f"{uid}_0",
            "role": role,
            "description": description,
        }
        for uid, role, description in (
            ("table", "background", "A table."),
            ("paper_cup", "rigid_object", "A paper cup."),
            ("popcorn_bucket", "rigid_object", "A popcorn bucket."),
            ("earbuds_case", "rigid_object", "A blue earbuds case."),
        )
    ]


def test_injected_planner_returns_only_semantics_and_resolves_aliases() -> None:
    observed: dict[str, Any] = {}

    def caller(*, prompt: str, model: str | None) -> dict[str, Any]:
        observed.update(prompt=prompt, model=model)
        return {
            "semantic_steps": [
                {
                    "id": "s01_place",
                    "operator": "place_relative",
                    "object": "interact_soda_can_0_0",
                    "goal": {"reference_object": "table", "relation": "on"},
                },
                {
                    "id": "s02_orient",
                    "operator": "orient_object",
                    "object": "interact_soda_can_1_0",
                    "goal": {
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                    },
                },
            ]
        }

    program = plan_task(
        task_name="injected",
        task_description="Place one object and then orient another.",
        scene_objects=_scene(),
        model="test-model",
        llm_caller=caller,
    )

    assert program["schema_version"] == TASK_AGENT_SCHEMA
    assert program["semantic_steps"][0]["object"] == "interact_soda_can_0"
    assert program["semantic_steps"][1]["depends_on"] == ["s01_place"]
    assert "Do not select a task route" in observed["prompt"]
    assert observed["model"] == "test-model"


def test_planner_repairs_a_non_visible_skill_once() -> None:
    calls = 0

    def caller(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "semantic_steps": [
                    {
                        "id": "s01_hold",
                        "operator": "hold_hover",
                        "object": "cube",
                        "goal": {},
                    }
                ]
            }
        return {
            "semantic_steps": [
                {
                    "id": "s01_place_cube",
                    "operator": "place_relative",
                    "object": "cube",
                    "goal": {
                        "reference_object": "basket",
                        "relation": "inside",
                        "orientation_goal": "preserve",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                },
                {
                    "id": "s02_place_cup",
                    "operator": "place_relative",
                    "object": "cup",
                    "goal": {
                        "reference_object": "basket",
                        "relation": "inside",
                    },
                    "depends_on": [],
                },
            ],
            "allocation_groups": [
                {
                    "id": "dual_arms_1",
                    "semantic_step_ids": ["s01_place_cube", "s02_place_cup"],
                    "arm_constraint": "distinct_arms",
                }
            ],
        }

    program = plan_task(
        task_name="dual_arm_basket",
        task_description="用双臂把两侧的方块和纸杯放到篮子里",
        scene_objects=_dual_arm_scene(),
        llm_caller=caller,
    )

    assert [
        (step["id"], step["operator"], step["object"], step["depends_on"])
        for step in program["semantic_steps"]
    ] == [
        ("s01_place_cube", "place_relative", "cube", []),
        ("s02_place_cup", "place_relative", "cup", []),
    ]
    assert calls == 2
    assert program["allocation_groups"][0]["arm_constraint"] == "distinct_arms"


def test_planner_repairs_build_stack_singular_object_contract() -> None:
    prompts: list[str] = []

    def caller(*, prompt: str, **_kwargs: Any) -> dict[str, Any]:
        prompts.append(prompt)
        if len(prompts) == 1:
            return {
                "semantic_steps": [
                    {
                        "id": "s01_build_stack",
                        "operator": "build_stack",
                        "object": "paper_cup",
                        "goal": {
                            "anchor": "popcorn_bucket",
                            "stack_mode": "on_top",
                        },
                        "depends_on": [],
                    }
                ],
                "allocation_groups": [],
            }
        return {
            "semantic_steps": [
                {
                    "id": "s01_build_stack",
                    "operator": "build_stack",
                    "objects": ["paper_cup", "earbuds_case"],
                    "goal": {
                        "anchor": "popcorn_bucket",
                        "stack_mode": "on_top",
                        "orientation_goal": "preserve",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                }
            ],
            "allocation_groups": [],
        }

    program = plan_task(
        task_name="task3_2",
        task_description="把纸杯叠放到爆米花桶上，然后把蓝色耳机盒叠放到纸杯上",
        scene_objects=_stack_scene(),
        llm_caller=caller,
    )

    assert len(prompts) == 2
    assert "build_stack requires an 'objects' list" in prompts[1]
    assert program["semantic_steps"][0]["objects"] == [
        "paper_cup",
        "earbuds_case",
    ]
    assert program["semantic_steps"][0]["goal"]["anchor"] == "popcorn_bucket"


def test_planner_rejects_a_non_visible_skill_after_one_repair() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {
            "semantic_steps": [
                {
                    "id": "s01_hold",
                    "operator": "hold_hover",
                    "object": "cube",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {},
                    "depends_on": [],
                },
                {
                    "id": "s02_place",
                    "operator": "place_relative",
                    "object": "cube",
                    "actor": {"mode": "required", "arm": "right_arm"},
                    "goal": {
                        "reference_object": "basket",
                        "relation": "inside",
                    },
                    "depends_on": ["s01_hold"],
                },
            ]
        }

    with pytest.raises(ValueError, match="after one repair"):
        plan_task(
            task_name="conflicting_arms",
            task_description="Hold the cube with the left arm, then place it "
            "with the right arm.",
            scene_objects=_dual_arm_scene(),
            llm_caller=caller,
        )


def test_spatial_two_sided_phrase_does_not_invent_arm_constraint() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {
            "semantic_steps": [
                {
                    "id": "s01_left",
                    "operator": "orient_object",
                    "object": "cube",
                    "goal": {
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                },
                {
                    "id": "s02_right",
                    "operator": "orient_object",
                    "object": "cup",
                    "goal": {
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                },
            ],
            "allocation_groups": [],
        }

    program = plan_task(
        task_name="two_sided_upright",
        task_description="把两边东西扶正",
        scene_objects=_dual_arm_scene(),
        llm_caller=caller,
    )

    assert program["allocation_groups"] == []


def test_planner_does_not_infer_arm_group_from_instruction_text() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {
            "semantic_steps": [
                {
                    "id": step_id,
                    "operator": "orient_object",
                    "object": object_uid,
                    "goal": {
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                }
                for step_id, object_uid in (("s01", "cube"), ("s02", "cup"))
            ],
            "allocation_groups": [],
        }

    program = plan_task(
        task_name="explicit_both_arms",
        task_description="用双臂把两个物体扶正",
        scene_objects=_dual_arm_scene(),
        llm_caller=caller,
    )

    assert program["allocation_groups"] == []


def test_planner_does_not_expose_internal_operator_contracts() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {
            "semantic_steps": [
                {
                    "id": "s01_hold",
                    "operator": "hold_hover",
                    "object": "cube",
                    "actor": {"mode": "auto"},
                    "goal": {},
                    "depends_on": [],
                },
                {
                    "id": "s02_place",
                    "operator": "place_relative",
                    "object": "cube",
                    "actor": {"mode": "auto"},
                    "goal": {
                        "reference_object": "basket",
                        "relation": "inside",
                    },
                    "depends_on": ["s01_hold"],
                },
            ]
        }

    with pytest.raises(ValueError, match="after one repair"):
        plan_task(
            task_name="nondefault_hover",
            task_description="Hold the cube in a special pose, then place it.",
            scene_objects=_dual_arm_scene(),
            llm_caller=caller,
        )


def test_plan_task_has_no_rule_fallback_parameter() -> None:
    assert "deterministic_fallback" not in inspect.signature(plan_task).parameters


def test_arrange_line_preserves_structured_orientation_output() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {
            "semantic_steps": [
                {
                    "id": "s01_line",
                    "operator": "arrange_line",
                    "objects": [
                        "interact_soda_can_0",
                        "interact_soda_can_1",
                    ],
                    "goal": {
                        "axis": "world_y",
                        "order_constraint": "free",
                        "orientation_goal": "upright",
                        "orientation_axis": "long_axis",
                    },
                    "depends_on": [],
                }
            ],
            "allocation_groups": [],
        }

    program = plan_task(
        task_name="neutral_line",
        task_description="将罐头摆成一排",
        scene_objects=_scene(),
        llm_caller=caller,
    )

    goal = program["semantic_steps"][0]["goal"]
    assert goal["orientation_goal"] == "upright"
    assert goal["orientation_axis"] == "long_axis"


def test_arrange_line_preserves_structured_axis_output() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {
            "semantic_steps": [
                {
                    "id": "s01_line",
                    "operator": "arrange_line",
                    "objects": [
                        "interact_soda_can_0",
                        "interact_soda_can_1",
                    ],
                    "goal": {
                        "anchor": "table_center",
                        "axis": "world_x",
                        "order_constraint": "free",
                    },
                    "depends_on": [],
                }
            ],
            "allocation_groups": [],
        }

    program = plan_task(
        task_name="ambiguous_line_axis",
        task_description="将罐头摆成一排",
        scene_objects=_scene(),
        llm_caller=caller,
    )

    assert program["semantic_steps"][0]["goal"]["axis"] == "world_x"


def test_instruction_text_does_not_override_structured_axis_output() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {
            "semantic_steps": [
                {
                    "id": "s01_line",
                    "operator": "arrange_line",
                    "objects": [
                        "interact_soda_can_0",
                        "interact_soda_can_1",
                    ],
                    "goal": {
                        "anchor": "table_center",
                        "axis": "world_y",
                        "order_constraint": "free",
                    },
                    "depends_on": [],
                }
            ],
            "allocation_groups": [],
        }

    program = plan_task(
        task_name="front_to_back_line_axis",
        task_description="将罐头沿前后方向摆成一列",
        scene_objects=_scene(),
        llm_caller=caller,
    )

    assert program["semantic_steps"][0]["goal"]["axis"] == "world_y"


def test_arrange_line_preserves_explicit_orientation_request() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {
            "semantic_steps": [
                {
                    "id": "s01_line",
                    "operator": "arrange_line",
                    "objects": [
                        "interact_soda_can_0",
                        "interact_soda_can_1",
                    ],
                    "goal": {
                        "axis": "world_y",
                        "order_constraint": "free",
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                }
            ],
            "allocation_groups": [],
        }

    program = plan_task(
        task_name="upright_line",
        task_description="先把罐头扶正，再摆成一排",
        scene_objects=_scene(),
        llm_caller=caller,
    )

    assert program["semantic_steps"][0]["goal"]["orientation_goal"] == "upright"


def test_planner_rejects_route_or_graph_output() -> None:
    def caller(**_kwargs: Any) -> dict[str, Any]:
        return {"route": "arrangement_line", "semantic_steps": []}

    with pytest.raises(ValueError, match="only 'semantic_steps'"):
        plan_task(
            task_name="bad",
            task_description="Arrange objects.",
            scene_objects=_scene(),
            llm_caller=caller,
        )


def test_llm_settings_read_gen_sim_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            (
                "# Local Action Engine credentials",
                'export OPENAI_API_KEY="dotenv-key"',
                "OPENAI_BASE_URL=https://dotenv.example/v1/",
                "OPENAI_MODEL=dotenv-model",
            )
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "gen_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llm": {
                    "openai_compatible": {
                        "api_key": "json-key",
                        "base_url": "https://json.example/v1",
                        "model": "json-model",
                        "default_query": {"api-version": "test"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(planner_module, "_GEN_SIM_ENV_PATH", env_path)
    monkeypatch.setattr(planner_module, "_GEN_CONFIG_PATH", config_path)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_MODEL",
        "LLM_MODEL",
        "LLM_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = planner_module._load_llm_settings(model=None)

    assert settings == {
        "api_key": "dotenv-key",
        "base_url": "https://dotenv.example/v1",
        "model": "dotenv-model",
        "default_query": {"api-version": "test"},
    }


def test_process_environment_and_model_argument_override_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=dotenv-key",
                "OPENAI_BASE_URL=https://dotenv.example/v1",
                "OPENAI_MODEL=dotenv-model",
            )
        ),
        encoding="utf-8",
    )
    missing_config = tmp_path / "missing.json"
    monkeypatch.setattr(planner_module, "_GEN_SIM_ENV_PATH", env_path)
    monkeypatch.setattr(planner_module, "_GEN_CONFIG_PATH", missing_config)
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://shell.example/v1/")
    monkeypatch.setenv("LLM_MODEL", "shell-model")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    settings = planner_module._load_llm_settings(model="argument-model")

    assert settings["api_key"] == "shell-key"
    assert settings["base_url"] == "https://shell.example/v1"
    assert settings["model"] == "argument-model"


def test_partial_process_transport_does_not_mix_with_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=dotenv-key",
                "OPENAI_BASE_URL=https://dotenv.example/v1",
                "OPENAI_MODEL=dotenv-model",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(planner_module, "_GEN_SIM_ENV_PATH", env_path)
    monkeypatch.setattr(
        planner_module,
        "_GEN_CONFIG_PATH",
        tmp_path / "missing.json",
    )
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_MODEL",
        "LLM_MODEL",
        "LLM_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-process-key")

    settings = planner_module._load_llm_settings(model=None)

    assert settings["api_key"] == "dotenv-key"
    assert settings["base_url"] == "https://dotenv.example/v1"
    assert settings["model"] == "dotenv-model"


def test_default_llm_caller_disables_custom_socket_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_openai

    captured: dict[str, Any] = {}

    class FakeRunnable:
        def invoke(self, _messages: Any) -> dict[str, list[Any]]:
            return {"semantic_steps": [], "allocation_groups": []}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def with_structured_output(
            self,
            _schema: dict[str, Any],
            **_kwargs: Any,
        ) -> FakeRunnable:
            return FakeRunnable()

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        planner_module,
        "_load_llm_settings",
        lambda *, model: {
            "api_key": "test-key",
            "model": model or "test-model",
            "base_url": "https://example.test/v1",
            "default_query": {},
        },
    )

    planner_module._default_llm_caller(prompt="plan", model="test-model")

    assert captured["http_socket_options"] == ()


def test_structured_output_transport_selects_json_mode_only_for_mimo() -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def with_structured_output(self, schema: dict[str, Any], **kwargs: Any) -> str:
            calls.append({"schema": schema, "kwargs": kwargs})
            return "structured"

    schema = {"type": "object"}
    client = FakeClient()
    mimo = planner_module._structured_output_runnable(
        client,
        schema,
        settings={
            "model": "mimo-v2.5",
            "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        },
    )
    generic = planner_module._structured_output_runnable(
        client,
        schema,
        settings={"model": "gpt-test", "base_url": "https://example.test/v1"},
    )

    assert mimo == generic == "structured"
    assert [call["kwargs"] for call in calls] == [
        {"method": "json_mode"},
        {"method": "json_schema"},
    ]
