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
import json

import pytest

from embodichain.gen_sim.task_engine.orchestration.grounding import (
    ground_articulation_parts,
    ground_scene_references,
)
from embodichain.gen_sim.task_engine.orchestration.scene_inventory import SceneInventory


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
                "object": object_selector or _selector("object-alpha"),
                "target": target_selector or _selector("target-alpha"),
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
        instruction="test-instruction",
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
    prompt_inventory = json.loads(prompt.split("Redacted scene inventory:\n", 1)[1])
    side_by_uid = {item["uid"]: item["side"] for item in prompt_inventory}
    assert side_by_uid["cutting_board"] == "right"
    assert side_by_uid["salt_shaker"] == "left"
    assert '"position"' not in prompt
    assert '"init_pos"' not in prompt


@pytest.mark.parametrize("robot_profile", ["ur5", "ur10", "franka"])
def test_scene_inventory_uses_the_shared_final_world_lateral_axis(
    robot_profile: str,
) -> None:
    inventory = SceneInventory(_scene(), robot_profile=robot_profile)

    assert inventory.left_score(inventory.by_uid["salt_shaker"]) > 0.0
    assert inventory.left_score(inventory.by_uid["cutting_board"]) < 0.0


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
    expected = error if error == "was not resolved" else f"after one repair.*{error}"
    with pytest.raises(ValueError, match=expected):
        _run(_intent(), lambda **_kwargs: deepcopy(response))


@pytest.mark.parametrize("status", ["ambiguous", "not_found"])
def test_unresolved_evidence_is_not_retried_into_a_guessed_binding(status: str) -> None:
    calls = []

    def caller(**kwargs):
        calls.append(kwargs)
        return {
            "bindings": [
                _binding("move.object", [], status=status, confidence=0.0),
                _binding("move.target", ["table"]),
            ]
        }

    with pytest.raises(ValueError, match="was not resolved"):
        _run(_intent(), caller)
    assert len(calls) == 1


def test_exact_scene_identity_cannot_be_replaced_by_another_known_object() -> None:
    calls = []

    def caller(**kwargs):
        calls.append(kwargs)
        return {
            "bindings": [
                _binding("move.object", ["salt_shaker"]),
                _binding("move.target", ["table"]),
            ]
        }

    with pytest.raises(ValueError, match="names exact UID"):
        _run(_intent(object_selector=_selector("cutting_board")), caller)
    assert len(calls) == 1


def test_exact_scene_identity_is_accepted_when_binding_matches() -> None:
    result = _run(
        _intent(object_selector=_selector("cutting_board")),
        lambda **kwargs: {
            "bindings": [
                _binding("move.object", ["cutting_board"]),
                _binding("move.target", ["table"]),
            ]
        },
    )
    assert result.bindings["move.object"] == ("cutting_board",)


def test_grounding_enforces_count_and_accepts_an_open_world_set() -> None:
    intent = _intent(
        object_selector=_selector("object-set", quantifier="count", count=2)
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
    intent = _intent(object_selector=_selector("object-set", quantifier="all"))
    response = {
        "bindings": [
            _binding("move.object", ["cutting_board", "salt_shaker"]),
            _binding("move.target", ["table"]),
        ]
    }

    result = _run(intent, lambda **_kwargs: response)

    assert result.bindings["move.object"] == ("cutting_board", "salt_shaker")


def test_part_grounding_selects_only_catalog_ids() -> None:
    intent = {
        "steps": [
            {
                "id": "slide",
                "task_type": "E6",
                "object": _selector("right drawer"),
            }
        ]
    }
    catalogs = {
        "drawer_unit": [
            {"part_id": "part_left", "joint": "left_slide", "link": "left_drawer"},
            {"part_id": "part_right", "joint": "right_slide", "link": "right_drawer"},
        ]
    }

    def caller(**kwargs):
        assert "part_right" in kwargs["prompt"]
        return {
            "bindings": [
                {
                    "reference_id": "slide.object",
                    "status": "resolved",
                    "part_id": "part_right",
                    "confidence": 0.95,
                }
            ]
        }

    result = ground_articulation_parts(
        "open the right drawer",
        intent,
        {"slide.object": ["drawer_unit"]},
        catalogs,
        "test-model",
        caller,
    )
    assert result == {"slide.object": "part_right"}


def test_inside_target_uses_the_same_part_catalog_as_slide() -> None:
    intent = {
        "steps": [
            {
                "id": "place",
                "task_type": "E1",
                "relation": "inside",
                "target": _selector("top drawer"),
            }
        ]
    }
    result = ground_articulation_parts(
        "put cube in top drawer",
        intent,
        {"place.target": ["cabinet"]},
        {
            "cabinet": [
                {"part_id": "part_top", "vertical_rank": "top"},
                {"part_id": "part_bottom", "vertical_rank": "bottom"},
            ]
        },
        "test-model",
        lambda **kw: pytest.fail("Explicit vertical selection must not call a model"),
    )
    assert result == {"place.target": "part_top"}


def test_part_grounding_uses_vertical_geometry_over_preselection() -> None:
    intent = {
        "steps": [
            {
                "id": "top",
                "task_type": "E6",
                "object": _selector("上面的抽屉"),
            },
            {
                "id": "middle",
                "task_type": "E6",
                "object": _selector("中间的抽屉"),
            },
            {
                "id": "bottom",
                "task_type": "E6",
                "object": _selector("最下面的抽屉"),
            },
        ]
    }
    catalogs = {
        "drawer_unit": [
            {
                "part_id": "part_1",
                "joint": "drawer_1_slide",
                "vertical_rank": "bottom",
            },
            {
                "part_id": "part_2",
                "joint": "drawer_2_slide",
                "vertical_rank": "middle",
            },
            {
                "part_id": "part_3",
                "joint": "drawer_3_slide",
                "vertical_rank": "top",
            },
        ]
    }

    def caller(**_kwargs):
        raise AssertionError("deterministic vertical grounding must not call the model")

    result = ground_articulation_parts(
        "依次拉开上面、中间和最下面的抽屉",
        intent,
        {
            "top.object": ["drawer_unit"],
            "middle.object": ["drawer_unit"],
            "bottom.object": ["drawer_unit"],
        },
        catalogs,
        "test-model",
        caller,
        preselected={
            "top.object": "part_1",
            "middle.object": "part_2",
            "bottom.object": "part_3",
        },
    )

    assert result == {
        "top.object": "part_3",
        "middle.object": "part_2",
        "bottom.object": "part_1",
    }


def test_part_grounding_rejects_unavailable_vertical_rank() -> None:
    intent = {
        "steps": [
            {
                "id": "middle",
                "task_type": "E6",
                "object": _selector("middle drawer"),
            }
        ]
    }
    catalogs = {
        "drawer_unit": [
            {"part_id": "part_lower", "vertical_rank": "bottom"},
            {"part_id": "part_upper", "vertical_rank": "top"},
        ]
    }

    with pytest.raises(ValueError, match="middle.*not uniquely available"):
        ground_articulation_parts(
            "open the middle drawer",
            intent,
            {"middle.object": ["drawer_unit"]},
            catalogs,
            "test-model",
            lambda **_kwargs: {},
        )
