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


def test_task4_2_fallback_selects_five_cans_and_excludes_table() -> None:
    program = plan_task(
        task_name="task4_2",
        task_description="将罐头摆成一排",
        scene_objects=_scene(),
        deterministic_fallback=True,
    )
    step = program["semantic_steps"][0]

    assert step["operator"] == "arrange_line"
    assert step["objects"] == [
        "interact_soda_can_0",
        "interact_soda_can_1",
        "interact_soda_can_2",
        "interact_soda_can_3",
        "interact_soda_can_4",
    ]
    assert "table" not in step["objects"]


def test_arrange_line_discards_unrequested_orientation_change() -> None:
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
    assert goal["orientation_goal"] == "preserve"
    assert goal["orientation_axis"] == "none"


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
