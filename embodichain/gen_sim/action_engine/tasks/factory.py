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

"""Deterministic E1-E9 and L1-L4 task generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import (
    TASK_CONTRACTS,
    task_success_type,
    validate_scene_requirements,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.protocol import (
    SCENE_REQUIREMENTS_FILENAME,
    SCENE_REQUIREMENTS_SCHEMA,
    TASK_SPEC_FILENAME,
    TASK_SPEC_SCHEMA,
)

__all__ = ["BatchGenerationResult", "TaskFactory", "task_capability_catalog"]


@dataclass(frozen=True)
class BatchGenerationResult:
    """One reproducible batch, optionally persisted task by task."""

    tasks: tuple[dict[str, Any], ...]
    scene_requirements: tuple[dict[str, Any], ...]
    skipped_existing: tuple[str, ...] = ()


def task_capability_catalog() -> dict[str, dict[str, Any]]:
    """Return the thin E1-E9 semantics supplied to high-level planners."""
    registry = build_atomic_capability_registry()
    executable = set(registry.executable_names())
    return {
        task_type: {
            "semantics": contract.semantics,
            "core_actions": list(contract.core_actions),
            "runtime_available": set(contract.core_actions) <= executable,
        }
        for task_type, contract in TASK_CONTRACTS.items()
    }


_L4_TEMPLATES = (
    "memory",
    "visual_semantics",
    "pattern",
    "logic",
    "common_sense",
    "constraint",
)

_REPEATABLE_TASK_TYPES = frozenset({"E1", "E2", "E6", "E7", "E8", "E9"})
_OBJECT_COLORS = ("red", "orange", "yellow", "green", "blue", "white", "black")
_OBJECT_SIZES = ("small", "medium", "large")
_OBJECT_MATERIALS = ("metal", "plastic", "ceramic", "wood")


class TaskFactory:
    """Generate reproducible task-first specifications without scene UIDs."""

    def __init__(self, seed: int = 0, *, executable_only: bool = False) -> None:
        self.seed = int(seed)
        self.executable_only = bool(executable_only)
        registry = build_atomic_capability_registry()
        executable = set(registry.executable_names())
        self.available_task_types = tuple(
            task_type
            for task_type, contract in TASK_CONTRACTS.items()
            if not executable_only or set(contract.core_actions).issubset(executable)
        )
        if not self.available_task_types:
            raise ValueError("No task types satisfy executable_only.")

    def generate(
        self, level: str, index: int = 0
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate one deterministic TaskSpec and SceneRequirements pair."""
        rng = random.Random(f"action-engine-v2:{self.seed}:{level}:{int(index)}")
        draft, roles = self._draft(level, rng)
        identity = _digest({"seed": self.seed, "index": int(index), **draft})[:12]
        task_id = f"{level.lower()}-{identity}"
        task = validate_task_spec(
            {
                "schema_version": TASK_SPEC_SCHEMA,
                "task_id": task_id,
                **draft,
                "metadata": {
                    "generator": "TaskFactory-v2",
                    "seed": self.seed,
                    "index": int(index),
                    "executable_only": self.executable_only,
                },
            }
        )
        requirements = validate_scene_requirements(
            {
                "schema_version": SCENE_REQUIREMENTS_SCHEMA,
                "task_id": task_id,
                "objects": list(roles.values()),
                "cameras": (
                    [
                        {
                            "role": "reasoning_view",
                            "modalities": ["rgb", "depth"],
                            "coverage": "all_interaction_objects",
                        }
                    ]
                    if level == "L4"
                    else []
                ),
                "spatial_constraints": self._spatial_constraints(task),
                "distractor_count": rng.randint(0, 3),
                "metadata": {"task_first": True},
            }
        )
        return task, requirements

    def generate_batch(
        self,
        count: int,
        *,
        level_quotas: Mapping[str, int] | None = None,
    ) -> BatchGenerationResult:
        """Generate a stable, duplicate-free task batch."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer.")
        levels = self._level_schedule(count, level_quotas)
        tasks = []
        requirements = []
        seen_ids: set[str] = set()
        seen_tasks: set[str] = set()
        candidate_index = 0
        for level in levels:
            for _ in range(10000):
                task, scene = self.generate(level, candidate_index)
                candidate_index += 1
                semantic_key = _task_semantic_key(task)
                if semantic_key not in seen_tasks:
                    break
            else:
                raise RuntimeError(
                    f"Unable to generate another unique {level} task after 10000 attempts."
                )
            if task["task_id"] in seen_ids:
                raise RuntimeError(f"Duplicate generated task ID {task['task_id']!r}.")
            seen_ids.add(task["task_id"])
            seen_tasks.add(semantic_key)
            tasks.append(task)
            requirements.append(scene)
        return BatchGenerationResult(tuple(tasks), tuple(requirements))

    def write_batch(
        self,
        output_dir: str | Path,
        count: int,
        *,
        level_quotas: Mapping[str, int] | None = None,
        resume: bool = True,
    ) -> BatchGenerationResult:
        """Persist a batch using one resumable directory per stable task ID."""
        batch = self.generate_batch(count, level_quotas=level_quotas)
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        skipped = []
        for task, requirements in zip(batch.tasks, batch.scene_requirements):
            task_dir = root / task["task_id"]
            task_path = task_dir / TASK_SPEC_FILENAME
            requirements_path = task_dir / SCENE_REQUIREMENTS_FILENAME
            if task_path.exists() and requirements_path.exists() and resume:
                persisted_task = json.loads(task_path.read_text(encoding="utf-8"))
                persisted_requirements = json.loads(
                    requirements_path.read_text(encoding="utf-8")
                )
                if persisted_task != task or persisted_requirements != requirements:
                    raise ValueError(
                        f"Existing task artifacts in {task_dir} do not match the "
                        "deterministic batch."
                    )
                skipped.append(task["task_id"])
                continue
            if (task_path.exists() or requirements_path.exists()) and not resume:
                raise FileExistsError(f"Task artifacts already exist in {task_dir}.")
            task_dir.mkdir(parents=True, exist_ok=True)
            task_path.write_text(_json(task), encoding="utf-8")
            requirements_path.write_text(_json(requirements), encoding="utf-8")
        return BatchGenerationResult(
            batch.tasks,
            batch.scene_requirements,
            tuple(skipped),
        )

    def _draft(
        self,
        level: str,
        rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if level == "L1":
            task_type = rng.choice(self.available_task_types)
            return self._flat_draft(level, [task_type], rng)
        if level == "L2":
            repeatable = tuple(
                task_type
                for task_type in self.available_task_types
                if task_type in _REPEATABLE_TASK_TYPES
            )
            if not repeatable:
                raise ValueError("No repeatable task type satisfies executable_only.")
            task_type = rng.choice(repeatable)
            return self._flat_draft(level, [task_type] * rng.randint(2, 5), rng)
        if level == "L3":
            return self._l3_draft(rng)
        if level == "L4":
            return self._l4_draft(rng)
        raise ValueError("level must be one of L1, L2, L3, or L4.")

    def _flat_draft(
        self,
        level: str,
        task_types: Sequence[str],
        rng: random.Random,
        *,
        share_object: bool = False,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        roles: dict[str, dict[str, Any]] = {}
        instances = []
        clauses = []
        previous: str | None = None
        shared_role: str | None = None
        for index, task_type in enumerate(task_types, start=1):
            instance_id = f"task_{index:02d}"
            params, instance_roles, clause = self._instance(
                task_type,
                index,
                rng,
                shared_role=shared_role,
            )
            if share_object and shared_role is None:
                shared_role = (
                    str(params.get("object_role", params.get("source_role", "")))
                    or None
                )
            _merge_roles(roles, instance_roles)
            instances.append(
                {
                    "id": instance_id,
                    "task_type": task_type,
                    "params": params,
                    "depends_on": [] if previous is None else [previous],
                    "role": "primary",
                }
            )
            clauses.append(clause)
            previous = instance_id
        if level == "L2":
            instruction = _l2_instruction(task_types[0], len(task_types))
        else:
            instruction = "，然后".join(clauses) + "。"
        success_terms = [
            {
                "type": task_success_type(item["task_type"], item["params"]),
                "task_instance_id": item["id"],
            }
            for item in instances
        ]
        return (
            {
                "level": level,
                "instruction": instruction,
                "reasoning_type": "none",
                "task_instances": instances,
                "success": {"op": "all", "terms": success_terms},
                "oracle": {"task_order": [item["id"] for item in instances]},
            },
            roles,
        )

    def _instance(
        self,
        task_type: str,
        index: int,
        rng: random.Random,
        *,
        shared_role: str | None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
        contract = TASK_CONTRACTS[task_type]
        object_role = shared_role or f"object_{index:02d}"
        selector = (
            {}
            if shared_role is not None
            else {
                "color": rng.choice(_OBJECT_COLORS),
                "size": rng.choice(_OBJECT_SIZES),
                "material": rng.choice(_OBJECT_MATERIALS),
            }
        )
        roles = {
            object_role: _role(
                object_role,
                contract.example_category,
                contract.scene_affordances,
                initial_state=_initial_state(task_type),
                attributes=selector,
            )
        }
        params: dict[str, Any] = {"object_role": object_role}
        if selector:
            params["selector"] = selector
        names = {"object": object_role, "source": object_role}
        if task_type in {"E1", "E3"}:
            target_role = f"target_{index:02d}"
            target_category = "cup" if task_type == "E3" else "tray"
            target_selector = {
                "color": rng.choice(_OBJECT_COLORS),
                "size": rng.choice(_OBJECT_SIZES),
                "material": rng.choice(_OBJECT_MATERIALS),
            }
            roles[target_role] = _role(
                target_role,
                target_category,
                ("container", "support_surface"),
                attributes=target_selector,
            )
            params.update(
                {
                    "target_role": target_role,
                    "target_selector": target_selector,
                    "relation": "inside",
                }
            )
            if task_type == "E3":
                params["source_role"] = params.pop("object_role")
            names["target"] = target_role
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
        elif task_type in {"E6", "E7"}:
            params.update({"target_state": "open" if task_type == "E6" else "closed"})
        elif task_type == "E8":
            params.update({"target_setting": rng.randint(1, 4)})
        elif task_type == "E9":
            params.update({"terminal_state": "activated"})
        clause = contract.instruction_template.format(**names).rstrip("。")
        return params, roles, clause

    def _l4_draft(
        self,
        rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        reasoning = rng.choice(_L4_TEMPLATES)
        builders = {
            "memory": self._l4_memory,
            "visual_semantics": self._l4_visual,
            "pattern": self._l4_pattern,
            "logic": self._l4_logic,
            "common_sense": self._l4_common_sense,
            "constraint": self._l4_constraint,
        }
        draft, roles = builders[reasoning]()
        scene_seed = rng.randrange(2**31)
        draft.setdefault("oracle", {})["scene_seed"] = scene_seed
        for requirement in roles.values():
            requirement.setdefault("attributes", {})[
                "reasoning_scene_seed"
            ] = scene_seed
        draft["level"] = "L4"
        draft["reasoning_type"] = reasoning
        return draft, roles

    def _l3_draft(
        self,
        rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        valid: list[tuple[str, ...] | str] = []
        if all(task_type in self.available_task_types for task_type in ("E2", "E1")):
            valid.append(("E2", "E1"))
        if all(
            task_type in self.available_task_types for task_type in ("E6", "E1", "E7")
        ):
            valid.append("drawer_cycle")
        if not valid:
            raise ValueError("No compatible L3 chain satisfies executable_only.")
        selected = rng.choice(valid)
        if selected != "drawer_cycle":
            return self._flat_draft("L3", list(selected), rng, share_object=True)

        roles = {
            "drawer": _role(
                "drawer",
                "drawer",
                ("articulated", "pullable", "pushable"),
                initial_state={"joint_state": "closed"},
            ),
            "apple": _role("apple", "apple", ("graspable", "placeable")),
        }
        instances = [
            {
                "id": "task_01",
                "task_type": "E6",
                "params": {"object_role": "drawer", "target_state": "open"},
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_02",
                "task_type": "E1",
                "params": {
                    "object_role": "apple",
                    "target_role": "table",
                    "relation": "on",
                },
                "depends_on": ["task_01"],
                "role": "primary",
            },
            {
                "id": "task_03",
                "task_type": "E7",
                "params": {"object_role": "drawer", "target_state": "closed"},
                "depends_on": ["task_02"],
                "role": "primary",
            },
        ]
        return (
            {
                "level": "L3",
                "instruction": "打开抽屉，取出苹果放到桌上，再关闭抽屉。",
                "reasoning_type": "none",
                "task_instances": instances,
                "success": {
                    "op": "all",
                    "terms": [
                        {
                            "type": task_success_type(
                                item["task_type"], item["params"]
                            ),
                            "task_instance_id": item["id"],
                        }
                        for item in instances
                    ],
                },
                "oracle": {"task_order": [item["id"] for item in instances]},
            },
            roles,
        )

    def _l4_memory(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        roles = {
            f"block_{index}": _role(
                f"block_{index}",
                "cube",
                ("graspable", "stackable"),
                initial_state={"stack_layer": index, "color": color},
            )
            for index, color in enumerate(("red", "yellow", "blue"), start=1)
        }
        instances = [
            {
                "id": "task_01",
                "task_type": "E1",
                "params": {
                    "object_role": "block_3",
                    "target_role": "table",
                    "relation": "on",
                    "slot": "right",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_02",
                "task_type": "E1",
                "params": {
                    "object_role": "block_2",
                    "target_role": "table",
                    "relation": "on",
                    "slot": "left",
                },
                "depends_on": ["task_01"],
                "role": "primary",
            },
            {
                "id": "task_03",
                "task_type": "E1",
                "params": {
                    "object_role": "block_2",
                    "target_role": "block_1",
                    "relation": "on_top",
                },
                "depends_on": ["task_02"],
                "role": "primary",
            },
            {
                "id": "task_04",
                "task_type": "E1",
                "params": {
                    "object_role": "block_3",
                    "target_role": "block_2",
                    "relation": "on_top",
                },
                "depends_on": ["task_03"],
                "role": "primary",
            },
        ]
        return (
            {
                "instruction": "拆开堆叠，然后按原来的顺序重新组装。",
                "task_instances": instances,
                "success": {"type": "original_order_restored"},
                "oracle": {"order_bottom_to_top": list(roles)},
            },
            roles,
        )

    def _l4_visual(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        roles = {
            "mouth_piece": _role("mouth_piece", "face_part", ("graspable",)),
            "face_board": _role("face_board", "face_board", ("visual_target",)),
        }
        return _one_l4(
            "给这张脸补上缺失的嘴巴。",
            "E1",
            {
                "object_role": "mouth_piece",
                "target_role": "face_board",
                "relation": "visual_slot",
            },
            {"missing_part": "mouth", "target_role": "face_board"},
            roles,
            success={"type": "visual_relation", "relation": "mouth_completed"},
        )

    def _l4_pattern(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        roles = {
            "pattern_piece": _role("pattern_piece", "tile", ("graspable",)),
            "pattern_board": _role(
                "pattern_board", "pattern_board", ("visual_target",)
            ),
        }
        return _one_l4(
            "补全这个对称图案。",
            "E1",
            {
                "object_role": "pattern_piece",
                "target_role": "pattern_board",
                "relation": "symmetric_slot",
            },
            {"rule": "bilateral_symmetry"},
            roles,
            success={"type": "visual_relation", "relation": "pattern_completed"},
        )

    def _l4_logic(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        roles = {
            f"cube_{value}": _role(
                f"cube_{value}",
                "number_cube",
                ("graspable",),
                attributes={"value": value},
            )
            for value in (1, 2, 3, 4)
        }
        roles["selection_tray"] = _role(
            "selection_tray", "tray", ("container", "support_surface")
        )
        instances = _instances("E1", ["cube_1", "cube_4"])
        for item in instances:
            item["params"]["target_role"] = "selection_tray"
            item["params"]["relation"] = "inside"
        return (
            {
                "instruction": "选择合适的方块，使它们的数字之和为5。",
                "task_instances": instances,
                "success": {"type": "sum_equals", "value": 5},
                "oracle": {
                    "valid_selections": [["cube_1", "cube_4"], ["cube_2", "cube_3"]]
                },
            },
            roles,
        )

    def _l4_common_sense(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        roles = {
            role: _role(role, category, ("graspable", "placeable"))
            for role, category in (
                ("plate", "plate"),
                ("fork", "cutlery"),
                ("cup", "cup"),
                ("dining_area", "table_region"),
            )
        }
        instances = _instances("E1", ["plate", "fork", "cup"])
        for item in instances:
            item["params"].update(
                {"target_role": "dining_area", "relation": "functional_layout"}
            )
        return (
            {
                "instruction": "为一位客人摆好餐位。",
                "task_instances": instances,
                "success": {"type": "functional_place_setting"},
                "oracle": {"required_roles": ["plate", "fork", "cup"]},
            },
            roles,
        )

    def _l4_constraint(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        roles = {
            "object_a": _role("object_a", "can", ("graspable", "placeable")),
            "object_b": _role("object_b", "cup", ("graspable", "placeable")),
            "sign": _role("sign", "sign", ("visual_target",)),
        }
        instances = _instances("E1", ["object_a", "object_b"])
        for item in instances:
            item["params"].update(
                {"target_role": "table", "relation": "stable_visible"}
            )
        return (
            {
                "instruction": "让所有物体都放得稳且不挡住标志。",
                "task_instances": instances,
                "success": {"type": "stable_unobstructed", "reference_role": "sign"},
                "oracle": {"constraints": ["stable", "sign_visible"]},
            },
            roles,
        )

    def _spatial_constraints(self, task: Mapping[str, Any]) -> list[dict[str, Any]]:
        constraints = [{"type": "reachable", "roles": "all_interaction_objects"}]
        if task["level"] == "L4":
            constraints.append({"type": "camera_visible", "roles": "all"})
        return constraints

    @staticmethod
    def _level_schedule(
        count: int,
        quotas: Mapping[str, int] | None,
    ) -> list[str]:
        if quotas is None:
            return [f"L{index % 4 + 1}" for index in range(count)]
        allowed = {"L1", "L2", "L3", "L4"}
        if set(quotas) - allowed:
            raise ValueError("level_quotas contains an unknown task level.")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in quotas.values()
        ):
            raise ValueError("level_quotas values must be non-negative integers.")
        if sum(quotas.values()) != count:
            raise ValueError("level_quotas must sum exactly to count.")
        return [level for level in sorted(allowed) for _ in range(quotas.get(level, 0))]


def _role(
    role_id: str,
    category: str,
    affordances: Sequence[str],
    *,
    initial_state: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "role_id": role_id,
        "category": category,
        "count": 1,
        "affordances": sorted(affordances),
        "initial_state": dict(initial_state or {}),
        "attributes": dict(attributes or {}),
    }


def _initial_state(task_type: str) -> dict[str, Any]:
    if task_type == "E2":
        return {"orientation": "fallen"}
    if task_type == "E6":
        return {"joint_state": "closed"}
    if task_type == "E7":
        return {"joint_state": "open"}
    if task_type == "E9":
        return {"activation": "inactive"}
    if task_type == "E3":
        return {"held_by": "left_arm"}
    return {}


def _merge_roles(
    destination: dict[str, dict[str, Any]],
    incoming: Mapping[str, Mapping[str, Any]],
) -> None:
    for role_id, value in incoming.items():
        candidate = dict(value)
        if role_id not in destination:
            destination[role_id] = candidate
            continue
        current = destination[role_id]
        if current["category"] != candidate["category"]:
            raise ValueError(
                f"Shared role {role_id!r} has incompatible categories "
                f"{current['category']!r} and {candidate['category']!r}."
            )
        current["affordances"] = sorted(
            set(current["affordances"]) | set(candidate["affordances"])
        )
        for key in ("initial_state", "attributes"):
            conflicts = {
                item_key
                for item_key, item_value in candidate[key].items()
                if item_key in current[key] and current[key][item_key] != item_value
            }
            if conflicts:
                raise ValueError(
                    f"Shared role {role_id!r} has conflicting {key}: "
                    f"{sorted(conflicts)}."
                )
            current[key].update(candidate[key])


def _instances(task_type: str, roles: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    previous = None
    for index, role in enumerate(roles, start=1):
        item = {
            "id": f"task_{index:02d}",
            "task_type": task_type,
            "params": {"object_role": role},
            "depends_on": [] if previous is None else [previous],
            "role": "primary",
        }
        result.append(item)
        previous = item["id"]
    return result


def _one_l4(
    instruction: str,
    task_type: str,
    params: Mapping[str, Any],
    oracle: Mapping[str, Any],
    roles: dict[str, dict[str, Any]],
    *,
    success: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    return (
        {
            "instruction": instruction,
            "task_instances": [
                {
                    "id": "task_01",
                    "task_type": task_type,
                    "params": dict(params),
                    "depends_on": [],
                    "role": "primary",
                }
            ],
            "success": dict(success),
            "oracle": dict(oracle),
        },
        roles,
    )


def _l2_instruction(task_type: str, count: int) -> str:
    templates = {
        "E1": f"把{count}个物体放入托盘。",
        "E2": f"扶正{count}个倒下的物体。",
        "E3": f"依次完成{count}次倾倒。",
        "E4": f"依次交接{count}个物体。",
        "E5": f"依次双臂拿起{count}个物体。",
        "E6": "拉开所有指定的抽屉。",
        "E7": "关闭所有打开的抽屉。",
        "E8": "依次调整所有指定的旋钮。",
        "E9": "按下所有指定的按钮。",
    }
    return templates[task_type]


def _task_semantic_key(task: Mapping[str, Any]) -> str:
    semantic = {
        key: value for key, value in task.items() if key not in {"task_id", "metadata"}
    }
    return _digest(semantic)


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
