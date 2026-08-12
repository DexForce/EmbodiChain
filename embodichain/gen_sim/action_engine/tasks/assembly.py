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

"""Language-neutral scene inventory and grounded TaskSpec assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from embodichain.gen_sim.action_engine.domain import (
    TASK_CONTRACTS,
    task_contract,
    task_success_type,
    validate_scene_requirements,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.generation.config_builder import (
    canonical_robot_profile,
)
from embodichain.gen_sim.action_engine.protocol import (
    SCENE_REQUIREMENTS_SCHEMA,
    TASK_SPEC_SCHEMA,
)

__all__ = [
    "GroundedTaskBuilder",
    "GroundedTaskSpec",
    "SceneEntity",
    "SceneInventory",
    "validate_source_compatibility",
    "validate_target_compatibility",
]


@dataclass(frozen=True)
class GroundedTaskSpec:
    """One explicit TaskSpec plus verified scene role bindings."""

    task_spec: dict[str, Any]
    scene_requirements: dict[str, Any]
    role_bindings: dict[str, str]


@dataclass(frozen=True)
class SceneEntity:
    """One scene entity with source semantics preserved verbatim."""

    uid: str
    role: str
    name: str
    description: str
    category: str
    color: str | None
    position: tuple[float, float, float]
    affordances: frozenset[str] = frozenset()
    initial_state: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source_uid: str = ""

class SceneInventory:
    """Structural scene index without natural-language matching rules."""

    _PASSIVE_ROLES = frozenset(
        {
            "background",
            "camera",
            "light",
            "robot",
            "sensor",
            "support_surface",
            "table",
        }
    )

    def __init__(
        self,
        scene_objects: Sequence[Mapping[str, Any]],
        *,
        robot_profile: str,
    ) -> None:
        self.profile = canonical_robot_profile(robot_profile)
        self.entities = tuple(_scene_entity(item) for item in scene_objects)
        self.by_uid = {entity.uid: entity for entity in self.entities}
        if len(self.by_uid) != len(self.entities):
            raise ValueError("Scene inventory contains duplicate runtime UIDs.")
        self.support = tuple(
            entity
            for entity in self.entities
            if entity.uid == "table" or entity.role in {"table", "support_surface"}
        )
        self.passive = tuple(
            entity
            for entity in self.entities
            if entity in self.support or entity.role in self._PASSIVE_ROLES
        )
        self.interactive = tuple(
            entity for entity in self.entities if entity not in self.passive
        )
        if not self.interactive:
            raise ValueError("Task planning requires at least one interaction object.")

    @property
    def movable(self) -> tuple[SceneEntity, ...]:
        """Compatibility alias for callers that mean source candidates."""
        return self.interactive

    def left_score(self, entity: SceneEntity) -> float:
        """Return robot-relative lateral score; positive values are left."""
        sign = 1.0 if self.profile == "dual_franka" else -1.0
        return sign * entity.position[1]


class GroundedTaskBuilder:
    """Assemble grounded E1-E9 instances without parsing instruction text."""

    def __init__(
        self,
        task_id: str,
        instruction: str,
        inventory: SceneInventory,
        *,
        planner: str = "structured_llm_v2",
    ) -> None:
        self.task_id = task_id
        self.instruction = instruction
        self.inventory = inventory
        # ``index`` is retained as a short-lived compatibility alias for the
        # isolated deterministic adapter. It exposes structural data only.
        self.index = inventory
        self.planner = planner
        self.instances: list[dict[str, Any]] = []
        self.role_by_uid: dict[str, str] = {}
        self.requirements: dict[str, dict[str, Any]] = {}
        self.previous_object_uid: str | None = None
        self.previous_arm: str | None = None
        self.last_task_by_object_uid: dict[str, tuple[str, str]] = {}

    def add(
        self,
        task_type: str,
        object_entity: SceneEntity,
        *,
        target: SceneEntity | None = None,
        params: Mapping[str, Any] | None = None,
        depends_on: Sequence[str] | None = None,
    ) -> str:
        values = deepcopy(dict(params or {}))
        relation = str(values.get("relation", "none"))
        validate_source_compatibility(task_type, (object_entity,))
        validate_target_compatibility(task_type, target, relation=relation)

        instance_id = f"task_{len(self.instances) + 1:02d}"
        object_role = self._role(
            object_entity,
            required_affordances=task_contract(task_type).required_affordances,
            initial_state={"orientation": "fallen"} if task_type == "E2" else {},
        )
        values = {"object_role": object_role, **values}
        if task_type == "E3":
            values["source_role"] = values.pop("object_role")
        if target is not None:
            values["target_role"] = self._role(
                target,
                required_affordances=_target_affordances(task_type, relation),
            )
        if depends_on is None:
            dependencies = [self.instances[-1]["id"]] if self.instances else []
        else:
            dependencies = list(depends_on)
        previous_for_object = self.last_task_by_object_uid.get(object_entity.uid)
        if (
            task_type == "E4"
            and previous_for_object is not None
            and previous_for_object[1] == "E2"
            and previous_for_object[0] not in dependencies
        ):
            dependencies.append(previous_for_object[0])
        self.instances.append(
            {
                "id": instance_id,
                "task_type": task_type,
                "params": values,
                "depends_on": dependencies,
                "role": "primary",
            }
        )
        self.last_task_by_object_uid[object_entity.uid] = (instance_id, task_type)
        self.previous_object_uid = object_entity.uid
        if task_type == "E4":
            receive_arm = str(values.get("receive_arm", ""))
            self.previous_arm = (
                receive_arm if receive_arm in {"left_arm", "right_arm"} else None
            )
        elif str(values.get("required_arm", "")) in {"left_arm", "right_arm"}:
            self.previous_arm = str(values["required_arm"])
        return instance_id

    def build(self) -> GroundedTaskSpec:
        types = {item["task_type"] for item in self.instances}
        if len(self.instances) == 1:
            level = "L1"
        elif len(types) == 1:
            level = "L2"
        else:
            level = "L3"
        success_terms = [
            {
                "type": task_success_type(item["task_type"], item.get("params")),
                "task_instance_id": item["id"],
            }
            for item in self.instances
        ]
        task = validate_task_spec(
            {
                "schema_version": TASK_SPEC_SCHEMA,
                "task_id": self.task_id,
                "level": level,
                "instruction": self.instruction,
                "reasoning_type": "none",
                "task_instances": self.instances,
                "success": {"op": "all", "terms": success_terms},
                "oracle": {
                    "task_order": [item["id"] for item in self.instances],
                    "role_bindings": dict(sorted(self.role_bindings().items())),
                },
                "metadata": {"planner": self.planner},
            }
        )
        requirements = validate_scene_requirements(
            {
                "schema_version": SCENE_REQUIREMENTS_SCHEMA,
                "task_id": self.task_id,
                "objects": list(self.requirements.values()),
                "cameras": [],
                "spatial_constraints": [{"type": "preserve_source_scene"}],
                "distractor_count": max(
                    0,
                    len(self.inventory.interactive) - len(self.role_by_uid),
                ),
                "metadata": {"source": "existing_gym_project"},
            }
        )
        return GroundedTaskSpec(task, requirements, self.role_bindings())

    def role_bindings(self) -> dict[str, str]:
        return {role: uid for uid, role in self.role_by_uid.items()}

    def _role(
        self,
        entity: SceneEntity,
        task_type: str | None = None,
        *,
        required_affordances: Sequence[str] = (),
        initial_state: Mapping[str, Any] | None = None,
    ) -> str:
        if task_type in TASK_CONTRACTS:
            required_affordances = tuple(
                set(required_affordances)
                | set(task_contract(str(task_type)).required_affordances)
            )
            if task_type == "E2":
                initial_state = {"orientation": "fallen", **dict(initial_state or {})}
        elif task_type == "target":
            required_affordances = tuple(
                set(required_affordances) | {"support_surface"}
            )
        existing = self.role_by_uid.get(entity.uid)
        if existing is not None:
            requirement = self.requirements[existing]
            requirement["affordances"] = sorted(
                set(requirement["affordances"]) | set(required_affordances)
            )
            requirement["initial_state"].update(dict(initial_state or {}))
            return existing
        role = f"object_{len(self.role_by_uid) + 1:02d}"
        self.role_by_uid[entity.uid] = role
        attributes = deepcopy(dict(entity.attributes))
        if entity.color is not None:
            attributes.setdefault("color", entity.color)
        self.requirements[role] = {
            "role_id": role,
            "category": entity.category or entity.role,
            "count": 1,
            "affordances": sorted(set(required_affordances)),
            "initial_state": dict(initial_state or {}),
            "attributes": attributes,
        }
        return role


def validate_source_compatibility(
    task_type: str,
    objects: Sequence[SceneEntity],
) -> None:
    """Apply structural/explicit-affordance checks without a category taxonomy."""
    contract = task_contract(task_type)
    if contract.source_structure == "articulation":
        invalid = [entity.uid for entity in objects if entity.role != "articulation"]
    else:
        invalid = [
            entity.uid
            for entity in objects
            if entity.role not in {"object", "rigid_object"}
        ]
    if invalid:
        structure_label = (
            "articulation"
            if contract.source_structure == "articulation"
            else "movable rigid-object"
        )
        raise ValueError(
            f"{task_type} requires {structure_label} structure; "
            f"incompatible scene objects are {invalid}."
        )
    required = set(contract.required_affordances)
    for entity in objects:
        if entity.affordances:
            missing = required - set(entity.affordances)
            if missing:
                raise ValueError(
                    f"{task_type} is incompatible with scene object {entity.uid!r}; "
                    f"missing affordances {sorted(missing)}."
                )


def validate_target_compatibility(
    task_type: str,
    target: SceneEntity | None,
    *,
    relation: str,
) -> None:
    """Reject only structural or explicitly declared target contradictions."""
    if task_type == "E1" and relation == "on" and target is not None:
        if target.affordances and "support_surface" not in target.affordances:
            raise ValueError(
                f"E1 target {target.uid!r} has explicit affordances but does "
                "not support placement."
            )
        return
    requires_container = task_type == "E3" or (
        task_type == "E1" and relation == "inside"
    )
    if requires_container and target is None:
        raise ValueError(
            f"{task_type} {relation} relation requires a target container."
        )
    if not requires_container or target is None:
        return
    if target.role in SceneInventory._PASSIVE_ROLES:
        raise ValueError(
            f"{task_type} target {target.uid!r} is structurally incompatible "
            "with containment."
        )
    if target.affordances:
        compatible = {"container", "fillable", "liquid_container", "receptacle"}
        if set(target.affordances).isdisjoint(compatible):
            raise ValueError(
                f"{task_type} target {target.uid!r} has explicit affordances but "
                f"none support containment; expected one of {sorted(compatible)}."
            )


def _target_affordances(task_type: str, relation: str) -> tuple[str, ...]:
    if task_type == "E1" and relation == "on":
        return ("support_surface",)
    if task_type == "E3" or (task_type == "E1" and relation == "inside"):
        return ("container",)
    return ()


def _scene_entity(raw: Mapping[str, Any]) -> SceneEntity:
    uid = str(raw.get("runtime_uid", raw.get("uid", ""))).strip()
    if not uid:
        raise ValueError("Every scene object requires a runtime UID.")
    role = str(raw.get("role", raw.get("source_role", "object"))).strip().lower()
    raw_category = raw.get("category", raw.get("object_category", ""))
    category = "" if raw_category is None else str(raw_category).strip()
    raw_color = raw.get("color")
    attributes = raw.get("attributes", {})
    if not isinstance(attributes, Mapping):
        raise ValueError(f"Scene object {uid!r} attributes must be a mapping.")
    if raw_color is None:
        raw_color = attributes.get("color")
    color = str(raw_color).strip() if raw_color not in (None, "") else None
    position = raw.get("init_pos", raw.get("position", (0.0, 0.0, 0.0)))
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 3
    ):
        raise ValueError(f"Scene object {uid!r} requires a three-value init_pos.")
    raw_affordances = raw.get("affordances", raw.get("capabilities", ()))
    affordances = (
        frozenset(
            str(item).strip().lower() for item in raw_affordances if str(item).strip()
        )
        if isinstance(raw_affordances, Sequence)
        and not isinstance(raw_affordances, (str, bytes))
        else frozenset()
    )
    initial_state = raw.get("initial_state", raw.get("state", {}))
    if not isinstance(initial_state, Mapping):
        raise ValueError(f"Scene object {uid!r} initial_state must be a mapping.")
    return SceneEntity(
        uid=uid,
        role=role,
        name=str(raw.get("name", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        category=category,
        color=color,
        position=tuple(float(value) for value in position),
        affordances=affordances,
        initial_state=dict(initial_state),
        attributes=dict(attributes),
        source_uid=str(raw.get("source_uid", "")).strip(),
    )
