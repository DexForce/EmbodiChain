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

"""Validate a Scene Engine result against task-first requirements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from embodichain.gen_sim.action_engine.domain import validate_scene_requirements

__all__ = ["SceneHandoff", "validate_scene_handoff"]


@dataclass(frozen=True)
class SceneHandoff:
    """Validated role-to-UID resolution returned by an external Scene Engine."""

    task_id: str
    role_bindings: dict[str, Any]
    object_uids: tuple[str, ...]
    camera_uids: tuple[str, ...]


def validate_scene_handoff(
    requirements: Mapping[str, Any],
    scene: Mapping[str, Any],
    role_bindings: Mapping[str, str],
) -> SceneHandoff:
    """Reject scenes that do not satisfy roles, affordances, state, or cameras."""
    required = validate_scene_requirements(requirements)
    objects = scene.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("Scene hand-off requires an objects list.")
    object_by_uid: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise ValueError(f"Scene objects[{index}] must be a mapping.")
        uid = item.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"Scene objects[{index}] requires a UID.")
        if uid in object_by_uid:
            raise ValueError(f"Scene contains duplicate object UID {uid!r}.")
        object_by_uid[uid] = item

    bindings = dict(role_bindings)
    required_roles = {item["role_id"] for item in required["objects"]}
    if set(bindings) != required_roles:
        missing = sorted(required_roles - set(bindings))
        extra = sorted(set(bindings) - required_roles)
        raise ValueError(
            f"Scene role bindings mismatch; missing={missing}, extra={extra}."
        )
    normalized_bindings: dict[str, str | tuple[str, ...]] = {}
    assigned_uids: list[str] = []
    for requirement in required["objects"]:
        role = requirement["role_id"]
        count = int(requirement["count"])
        binding = bindings[role]
        if isinstance(binding, str):
            uids = [binding]
        elif isinstance(binding, Sequence) and not isinstance(binding, (str, bytes)):
            uids = [str(uid) for uid in binding]
        else:
            raise ValueError(f"Scene role {role!r} has an invalid UID binding.")
        if len(uids) != count or any(not uid for uid in uids):
            raise ValueError(
                f"Scene role {role!r} requires exactly {count} UID binding(s)."
            )
        normalized_bindings[role] = uids[0] if count == 1 else tuple(uids)
        assigned_uids.extend(uids)
        for uid in uids:
            _validate_bound_object(object_by_uid, uid, role, requirement)
    if len(assigned_uids) != len(set(assigned_uids)):
        raise ValueError("Each scene requirement role must resolve to unique UIDs.")

    cameras = scene.get("cameras", [])
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
        raise ValueError("Scene cameras must be a list.")
    camera_uids = []
    normalized_cameras = []
    for camera in cameras:
        if not isinstance(camera, Mapping) or not isinstance(camera.get("uid"), str):
            raise ValueError("Every scene camera requires a UID.")
        camera_uids.append(str(camera["uid"]))
        normalized_cameras.append(camera)
    for camera_requirement in required["cameras"]:
        modalities = set(camera_requirement.get("modalities", ()))
        coverage = camera_requirement.get("coverage")
        if not any(
            modalities <= set(camera.get("modalities", ()))
            and (coverage is None or camera.get("coverage") == coverage)
            for camera in normalized_cameras
        ):
            raise ValueError(
                "Scene cameras do not satisfy requirement "
                f"{dict(camera_requirement)!r}."
            )
    reported_constraints = scene.get("satisfied_spatial_constraints", [])
    if not isinstance(reported_constraints, Sequence) or isinstance(
        reported_constraints, (str, bytes)
    ):
        raise ValueError("Scene satisfied_spatial_constraints must be a list.")
    reported = {_canonical(item) for item in reported_constraints}
    missing_constraints = [
        constraint
        for constraint in required["spatial_constraints"]
        if _canonical(constraint) not in reported
    ]
    if missing_constraints:
        raise ValueError(
            "Scene does not satisfy spatial constraints: " f"{missing_constraints}."
        )
    return SceneHandoff(
        task_id=required["task_id"],
        role_bindings=normalized_bindings,
        object_uids=tuple(sorted(object_by_uid)),
        camera_uids=tuple(sorted(camera_uids)),
    )


def _validate_bound_object(
    object_by_uid: Mapping[str, Mapping[str, Any]],
    uid: str,
    role: str,
    requirement: Mapping[str, Any],
) -> None:
    if uid not in object_by_uid:
        raise ValueError(f"Scene role {role!r} references unknown UID {uid!r}.")
    actual = object_by_uid[uid]
    if actual.get("category") != requirement["category"]:
        raise ValueError(
            f"Scene object {uid!r} category does not satisfy role {role!r}."
        )
    missing_affordances = set(requirement["affordances"]) - set(
        actual.get("affordances", ())
    )
    if missing_affordances:
        raise ValueError(
            f"Scene object {uid!r} lacks affordances {sorted(missing_affordances)}."
        )
    for field in ("initial_state", "attributes"):
        actual_values = actual.get(field, {})
        if not isinstance(actual_values, Mapping):
            raise ValueError(f"Scene object {uid!r} {field} must be a mapping.")
        mismatched = {
            key: expected
            for key, expected in requirement[field].items()
            if actual_values.get(key) != expected
        }
        if mismatched:
            raise ValueError(
                f"Scene object {uid!r} does not satisfy {field} {mismatched}."
            )


def _canonical(value: Any) -> str:
    import json

    if not isinstance(value, Mapping):
        raise ValueError("Every satisfied spatial constraint must be a mapping.")
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
