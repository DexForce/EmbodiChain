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

from copy import deepcopy

import pytest

from embodichain.gen_sim.action_engine.tasks.grounding import (
    ground_scene_references,
)
from embodichain.gen_sim.action_engine.tasks.assembly import SceneInventory


def _selector(
    reference: str,
    *,
    quantifier: str = "one",
    count: int = 0,
) -> dict:
    return {
        "kind": "scene_ref",
        "step_id": "",
        "reference": reference,
        "quantifier": quantifier,
        "count": count,
    }


def _scene() -> list[dict]:
    return [
        {
            "runtime_uid": "table",
            "uid": "table",
            "role": "background",
            "category": "dining_table",
            "name": "work table",
            "description": "A rectangular work table.",
            "init_pos": [0.0, 0.0, 0.0],
        },
        {
            "runtime_uid": "cutting_board",
            "uid": "cutting_board",
            "role": "rigid_object",
            "category": "cutting_board",
            "name": "wood board",
            "description": "A large rectangular wooden cutting board.",
            "attributes": {
                "size": "large",
                "geometry": {"position": [0.0, 0.2, 0.7], "note": "flat"},
            },
            "initial_state": {"orientation": "fallen"},
            "init_pos": [0.0, 0.2, 0.7],
        },
        {
            "runtime_uid": "salt_shaker",
            "uid": "salt_shaker",
            "role": "rigid_object",
            "category": "salt_shaker",
            "description": "A small glass salt shaker.",
            "affordances": ["graspable"],
            "init_pos": [0.0, -0.2, 0.7],
        },
    ]


def _intent(
    *,
    object_selector: dict | None = None,
    target_selector: dict | None = None,
) -> dict:
    return {
        "steps": [
            {
                "id": "move",
                "task_type": "E1",
                "object": object_selector or _selector("木质长方体"),
                "target": target_selector or _selector("桌面"),
                "relation": "on",
            }
        ]
    }


def _binding(
    reference_id: str,
    uids: list[str],
    *,
    status: str = "resolved",
    confidence: float = 1.0,
    **extra: object,
) -> dict:
    return {
        "reference_id": reference_id,
        "status": status,
        "uids": uids,
        "confidence": confidence,
        **extra,
    }


def _run(intent: dict, caller) -> object:
    scene = _scene()
    return ground_scene_references(
        instruction="把木质长方体放到桌面上。",
        intent=intent,
        inventory=SceneInventory(scene, robot_profile="franka"),
        scene_objects=scene,
        model="test-model",
        caller=caller,
    )


def test_grounding_prompt_preserves_open_semantics_and_redacts_geometry() -> None:
    captured: dict[str, object] = {}

    def caller(**kwargs):
        captured.update(kwargs)
        return {
            "bindings": [
                _binding("move.object", ["cutting_board"]),
                _binding("move.target", ["table"]),
            ]
        }

    result = _run(_intent(), caller)

    prompt = str(captured["prompt"])
    assert result.bindings == {
        "move.object": ("cutting_board",),
        "move.target": ("table",),
    }
    assert '"category": "cutting_board"' in prompt
    assert '"category": "salt_shaker"' in prompt
    assert '"name": "wood board"' in prompt
    assert '"orientation": "fallen"' in prompt
    assert '"size": "large"' in prompt
    assert '"side": "left"' in prompt
    assert '"position"' not in prompt
    assert '"init_pos"' not in prompt


def test_grounding_repairs_one_invalid_uid_in_the_same_batch() -> None:
    responses = [
        {
            "bindings": [
                _binding("move.object", ["invented"]),
                _binding("move.target", ["table"]),
            ]
        },
        {
            "bindings": [
                _binding("move.object", ["cutting_board"]),
                _binding("move.target", ["table"]),
            ]
        },
    ]
    prompts: list[str] = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return deepcopy(responses[len(prompts) - 1])

    result = _run(_intent(), caller)

    assert result.attempts == 2
    assert "previous grounding JSON failed" in prompts[1]
    assert result.bindings["move.object"] == ("cutting_board",)


@pytest.mark.parametrize(
    "response,error",
    [
        (
            {
                "bindings": [
                    _binding(
                        "move.object",
                        ["cutting_board"],
                        status="ambiguous",
                    ),
                    _binding("move.target", ["table"]),
                ]
            },
            "was not resolved",
        ),
        (
            {
                "bindings": [
                    _binding(
                        "move.object",
                        [],
                        status="not_found",
                        confidence=0.0,
                    ),
                    _binding("move.target", ["table"]),
                ]
            },
            "was not resolved",
        ),
        (
            {
                "bindings": [
                    _binding("move.object", ["cutting_board"], confidence=0.49),
                    _binding("move.target", ["table"]),
                ]
            },
            "confidence is below",
        ),
        (
            {
                "bindings": [
                    _binding("move.object", ["cutting_board", "cutting_board"]),
                    _binding("move.target", ["table"]),
                ]
            },
            "duplicate UIDs",
        ),
        (
            {
                "bindings": [
                    _binding("move.object", ["cutting_board"]),
                    _binding("move.object", ["salt_shaker"]),
                    _binding("move.target", ["table"]),
                ]
            },
            "Duplicate grounding binding",
        ),
        (
            {
                "bindings": [
                    _binding("move.object", ["cutting_board", "salt_shaker"]),
                    _binding("move.target", ["table"]),
                ]
            },
            "quantifier=one requires exactly one UID",
        ),
        (
            {"bindings": [_binding("move.object", ["cutting_board"])]},
            "omitted requests",
        ),
        (
            {
                "bindings": [
                    _binding("move.object", ["table"]),
                    _binding("move.target", ["cutting_board"]),
                ]
            },
            "candidate range",
        ),
        (
            {
                "bindings": [
                    _binding("move.object", ["cutting_board"]),
                    _binding("move.target", ["cutting_board"]),
                ]
            },
            "same UID",
        ),
        (
            {
                "bindings": [
                    _binding(
                        "move.object",
                        ["cutting_board"],
                        affordances=["graspable"],
                    ),
                    _binding("move.target", ["table"]),
                ]
            },
            "unsupported",
        ),
    ],
)
def test_grounding_fails_closed_after_one_repair(response: dict, error: str) -> None:
    with pytest.raises(ValueError, match=f"after one repair.*{error}"):
        _run(_intent(), lambda **_kwargs: deepcopy(response))


def test_grounding_enforces_count_and_accepts_an_open_world_set() -> None:
    intent = _intent(
        object_selector=_selector("两个桌面物体", quantifier="count", count=2)
    )
    response = {
        "bindings": [
            _binding("move.object", ["cutting_board", "salt_shaker"]),
            _binding("move.target", ["table"]),
        ]
    }

    result = _run(intent, lambda **_kwargs: response)
    assert result.bindings["move.object"] == ("cutting_board", "salt_shaker")

    invalid = deepcopy(response)
    invalid["bindings"][0]["uids"] = ["cutting_board"]
    with pytest.raises(ValueError, match="requires exactly 2 UIDs"):
        _run(intent, lambda **_kwargs: invalid)


def test_grounding_accepts_a_nonempty_all_binding() -> None:
    intent = _intent(object_selector=_selector("所有桌面物体", quantifier="all"))
    response = {
        "bindings": [
            _binding("move.object", ["cutting_board", "salt_shaker"]),
            _binding("move.target", ["table"]),
        ]
    }

    result = _run(intent, lambda **_kwargs: response)

    assert result.bindings["move.object"] == ("cutting_board", "salt_shaker")
