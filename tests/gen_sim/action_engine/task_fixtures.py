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

"""Language-neutral structured fixtures for Action Engine tests."""

from __future__ import annotations

from typing import Any

from embodichain.gen_sim.action_engine.domain import (
    task_success_type,
    validate_scene_requirements,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.protocol import (
    SCENE_REQUIREMENTS_SCHEMA,
    TASK_SPEC_SCHEMA,
)

__all__ = ["make_task_level", "make_task_spec"]

_OBJECT_FIXTURES = {
    "E1": ("can", ["graspable", "placeable"], {}),
    "E2": ("can", ["graspable", "orientable"], {"orientation": "fallen"}),
    "E3": ("container", ["graspable", "pourable"], {"held_by": "left_arm"}),
    "E4": ("cup", ["graspable", "handover"], {}),
    "E5": ("tray", ["dual_graspable", "rigid"], {}),
    "E6": ("drawer", ["articulated", "pullable"], {"joint_state": "closed"}),
    "E7": ("drawer", ["articulated", "pushable"], {"joint_state": "open"}),
    "E8": ("knob", ["turnable"], {}),
    "E9": ("button", ["pressable"], {"activation": "inactive"}),
}


def make_task_spec(
    task_type: str = "E1",
    *,
    task_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one validated L1 TaskSpec and matching scene requirements."""
    if task_type not in _OBJECT_FIXTURES:
        raise ValueError(f"Unsupported fixture task type {task_type!r}.")
    category, affordances, initial_state = _OBJECT_FIXTURES[task_type]
    object_role = "object_01"
    params: dict[str, Any] = {"object_role": object_role}
    objects = [
        {
            "role_id": object_role,
            "category": category,
            "count": 1,
            "affordances": affordances,
            "initial_state": initial_state,
            "attributes": {},
        }
    ]
    if task_type in {"E1", "E3"}:
        target_role = "target_01"
        params.update({"target_role": target_role, "relation": "inside"})
        if task_type == "E3":
            params["source_role"] = params.pop("object_role")
        objects.append(
            {
                "role_id": target_role,
                "category": "container",
                "count": 1,
                "affordances": ["container", "support_surface"],
                "initial_state": {},
                "attributes": {},
            }
        )
    elif task_type == "E2":
        params.update(
            {
                "orientation_goal": "upright",
                "support_role": "table",
                "upright_local_axis": "long_axis",
            }
        )
    elif task_type == "E4":
        params.update(
            {
                "transfer_arm": "left_arm",
                "receive_arm": "right_arm",
                "orientation_goal": "none",
            }
        )
    elif task_type == "E5":
        params.update({"direction": "up", "terminal_behavior": "hold"})
    elif task_type == "E6":
        params["target_state"] = "open"
    elif task_type == "E7":
        params["target_state"] = "closed"
    elif task_type == "E8":
        params["target_setting"] = 2
    elif task_type == "E9":
        params["target_state"] = "activated"

    effective_id = task_id or f"fixture-{task_type.lower()}"
    task = validate_task_spec(
        {
            "schema_version": TASK_SPEC_SCHEMA,
            "task_id": effective_id,
            "level": "L1",
            "instruction": "test-instruction",
            "reasoning_type": "none",
            "task_instances": [
                {
                    "id": "task_01",
                    "task_type": task_type,
                    "params": params,
                    "depends_on": [],
                    "role": "primary",
                }
            ],
            "success": {
                "type": task_success_type(task_type, params),
                "task_instance_id": "task_01",
            },
            "oracle": {},
            "metadata": {"fixture": True},
        }
    )
    requirements = validate_scene_requirements(
        {
            "schema_version": SCENE_REQUIREMENTS_SCHEMA,
            "task_id": effective_id,
            "objects": objects,
            "cameras": [],
            "spatial_constraints": [
                {"type": "reachable", "roles": "all_interaction_objects"}
            ],
            "distractor_count": 0,
            "metadata": {"fixture": True},
        }
    )
    return task, requirements


def make_task_level(
    level: str,
    *,
    reasoning: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a validated fixture for one public TaskSpec level."""
    if level == "L1":
        return make_task_spec("E1")
    first, requirements = make_task_spec("E1", task_id=f"fixture-{level.lower()}")
    if level == "L2":
        second = {
            "id": "task_02",
            "task_type": "E1",
            "params": {
                "object_role": "object_02",
                "target_role": "target_02",
                "relation": "inside",
            },
            "depends_on": ["task_01"],
            "role": "primary",
        }
        first["level"] = "L2"
        first["task_instances"].append(second)
        first["success"] = {
            "op": "all",
            "terms": [
                {"type": "semantic_goal", "task_instance_id": "task_01"},
                {"type": "semantic_goal", "task_instance_id": "task_02"},
            ],
        }
        requirements["objects"].extend(
            [
                {
                    "role_id": "object_02",
                    "category": "can",
                    "count": 1,
                    "affordances": ["graspable", "placeable"],
                    "initial_state": {},
                    "attributes": {},
                },
                {
                    "role_id": "target_02",
                    "category": "container",
                    "count": 1,
                    "affordances": ["container", "support_surface"],
                    "initial_state": {},
                    "attributes": {},
                },
            ]
        )
        return validate_task_spec(first), validate_scene_requirements(requirements)
    if level == "L4":
        first["level"] = "L4"
        first["reasoning_type"] = reasoning or "visual_semantics"
        first["success"] = {
            "visual_semantics": {
                "type": "visual_relation",
                "relation": "mouth_completed",
            },
            "pattern": {
                "type": "visual_relation",
                "relation": "pattern_completed",
            },
            "logic": {"type": "sum_equals", "value": 5},
            "memory": {"type": "original_order_restored"},
            "common_sense": {"type": "functional_place_setting"},
            "constraint": {"type": "stable_unobstructed"},
        }[first["reasoning_type"]]
        first["oracle"] = {"fixture": True}
        requirements["cameras"] = [
            {
                "role": "reasoning_view",
                "modalities": ["rgb", "depth"],
                "coverage": "all_interaction_objects",
            }
        ]
        return validate_task_spec(first), validate_scene_requirements(requirements)
    raise ValueError(f"Unsupported fixture task level {level!r}.")
