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

from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_engine.evaluation import evaluate_task_oracle
from embodichain.gen_sim.action_engine.tasks import TaskFactory


class _Object:
    def __init__(self, position: tuple[float, float, float]) -> None:
        self.pose = torch.eye(4).unsqueeze(0)
        self.pose[0, :3, 3] = torch.tensor(position)

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix is True
        return self.pose


class _Sim:
    def __init__(self, objects: dict[str, _Object]) -> None:
        self.objects = objects

    def get_rigid_object(self, uid: str) -> _Object | None:
        return self.objects.get(uid)


def _task(reasoning: str) -> tuple[dict, dict[str, str]]:
    factory = TaskFactory(73, executable_only=True)
    for index in range(100):
        task, requirements = factory.generate("L4", index)
        if task["reasoning_type"] == reasoning:
            return task, {
                item["role_id"]: f"uid_{item['role_id']}"
                for item in requirements["objects"]
            }
    raise AssertionError(f"No deterministic {reasoning!r} task found.")


def _env(bindings: dict[str, str]) -> SimpleNamespace:
    objects = {
        uid: _Object((2.0 + index, 0.0, 0.1))
        for index, uid in enumerate(bindings.values())
    }
    return SimpleNamespace(num_envs=1, device="cpu", sim=_Sim(objects))


@pytest.mark.parametrize(
    ("reasoning", "visual_relation"),
    [
        ("visual_semantics", "mouth_completed"),
        ("pattern", "pattern_completed"),
    ],
)
def test_visual_l4_oracles_use_post_execution_facts(
    reasoning: str, visual_relation: str
) -> None:
    task, bindings = _task(reasoning)
    env = _env(bindings)
    facts = {
        "entities": [],
        "relations": [],
        "task_predicates": [{"type": visual_relation, "confidence": 0.9}],
        "confidence": 0.9,
    }

    assert evaluate_task_oracle(task, env, bindings, visual_facts=facts).tolist() == [
        True
    ]


def test_memory_and_logic_oracles_check_only_final_state() -> None:
    memory, memory_bindings = _task("memory")
    memory_env = _env(memory_bindings)
    for index, role in enumerate(memory["oracle"]["order_bottom_to_top"]):
        memory_env.sim.objects[memory_bindings[role]] = _Object((0.0, 0.0, index * 0.1))
    assert evaluate_task_oracle(memory, memory_env, memory_bindings).all()

    logic, logic_bindings = _task("logic")
    logic_env = _env(logic_bindings)
    tray_uid = logic_bindings["selection_tray"]
    logic_env.sim.objects[tray_uid] = _Object((0.0, 0.0, 0.0))
    for role in ("cube_1", "cube_4"):
        logic_env.sim.objects[logic_bindings[role]] = _Object((0.0, 0.0, 0.1))
    assert evaluate_task_oracle(logic, logic_env, logic_bindings).all()


def test_common_sense_and_constraint_oracles_are_path_independent() -> None:
    common, common_bindings = _task("common_sense")
    common_env = _env(common_bindings)
    target_uid = common_bindings["dining_area"]
    common_env.sim.objects[target_uid] = _Object((0.0, 0.0, 0.0))
    for role in common["oracle"]["required_roles"]:
        common_env.sim.objects[common_bindings[role]] = _Object((0.0, 0.0, 0.1))
    assert evaluate_task_oracle(common, common_env, common_bindings).all()

    constrained, constraint_bindings = _task("constraint")
    constraint_env = _env(constraint_bindings)
    facts = {
        "entities": [
            {
                "uid": constraint_bindings["sign"],
                "visible": True,
                "confidence": 1.0,
            }
        ],
        "relations": [],
        "task_predicates": [],
        "confidence": 1.0,
    }
    assert evaluate_task_oracle(
        constrained,
        constraint_env,
        constraint_bindings,
        visual_facts=facts,
    ).all()

    blocker_uid = next(
        uid for role, uid in constraint_bindings.items() if role != "sign"
    )
    facts["relations"] = [
        {
            "type": "occludes",
            "uids": [blocker_uid, constraint_bindings["sign"]],
            "confidence": 1.0,
        }
    ]
    assert not evaluate_task_oracle(
        constrained,
        constraint_env,
        constraint_bindings,
        visual_facts=facts,
    ).any()

    facts["relations"][0]["uids"].reverse()
    assert evaluate_task_oracle(
        constrained,
        constraint_env,
        constraint_bindings,
        visual_facts=facts,
    ).all()
