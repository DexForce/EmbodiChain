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

"""Structural scene inventory used by Task Engine semantic binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodichain.gen_sim.task_engine.ontology import task_contract

__all__ = [
    "SceneEntity",
    "SceneInventory",
    "validate_source_compatibility",
    "validate_target_compatibility",
]


@dataclass(frozen=True)
class SceneEntity:
    """One scene entity with source semantics preserved verbatim.

    Attributes:
        uid: Stable runtime identity used by semantic calls.
        role: Structural scene role such as ``rigid_object`` or ``background``.
        name: Source-provided display name.
        description: Source-provided semantic description.
        category: Open-world source category label.
        color: Optional source-provided color label.
        position: Initial world position used only for relative-side scoring.
        affordances: Explicit source affordances; an empty set remains unknown.
        initial_state: Source-provided initial semantic state.
        attributes: Additional source semantic attributes.
        source_uid: Original identity before runtime normalization.
    """

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
    """Index structural scene facts without natural-language matching rules.

    Args:
        scene_objects: Source scene objects with canonical runtime identities.
        robot_profile: Non-empty profile identifier retained for audit context.

    Raises:
        ValueError: If identities are missing or duplicated, the profile is
            empty, or the scene has no interactive object.
    """

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
        profile = str(robot_profile).strip().lower().replace("-", "_")
        if not profile:
            raise ValueError("robot_profile must be non-empty.")
        self.profile = profile
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
        """Return interaction entities accepted as source candidates."""
        return self.interactive

    def left_score(self, entity: SceneEntity) -> float:
        """Return robot-relative lateral score; positive values are left."""
        return -entity.position[1]


def validate_source_compatibility(
    task_type: str,
    objects: Sequence[SceneEntity],
) -> None:
    """Validate source structure and explicitly declared affordances.

    Args:
        task_type: Canonical E-task identifier.
        objects: Bound source entities to validate.

    Raises:
        ValueError: If ``task_type`` is unknown, or a source has incompatible
            structure or contradicts the task's required affordances.
    """
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
    """Reject structural or explicitly declared target contradictions.

    Args:
        task_type: Canonical E-task identifier.
        target: Bound target entity, or ``None`` when the call has no target.
        relation: Canonical spatial relation requested by the task.

    Raises:
        ValueError: If containment lacks a target or the target explicitly
            contradicts the required container structure or affordance.
    """
    if relation == "on" and target is not None:
        return
    requires_container = task_type == "E3" or relation == "inside"
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


def _scene_entity(raw: Mapping[str, Any]) -> SceneEntity:
    uid = str(raw.get("runtime_uid", raw.get("uid", ""))).strip()
    if not uid:
        raise ValueError("Every scene object requires a runtime UID.")
    role = str(raw.get("role", raw.get("source_role", "object"))).strip().lower()
    raw_category = raw.get("category", raw.get("object_category", ""))
    category = "" if raw_category is None else str(raw_category).strip()
    attributes = raw.get("attributes", {})
    if not isinstance(attributes, Mapping):
        raise ValueError(f"Scene object {uid!r} attributes must be a mapping.")
    raw_color = raw.get("color", attributes.get("color"))
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
