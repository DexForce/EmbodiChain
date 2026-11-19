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

"""Owned arena-space anchors that restrict an existing prepared layout domain."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from .._validation import _identifier, _rigid_pose

if TYPE_CHECKING:
    from .layout import PreparedSceneRandomization

__all__ = ["SceneLayoutCondition"]


class _LayoutConditionDomainError(ValueError):
    """A well-formed anchor conflicts with its prepared layout domain."""


@dataclass(frozen=True)
class SceneLayoutCondition:
    """Fix declared entities to arena poses without enlarging their motion domain.

    Args:
        poses: Native UID to rigid arena-space pose. Inputs are independently
            owned immutable tuples. Group roots retain their authored rotation
            and height; selected members retain their declared support-frame
            XY/yaw constraints. A moving ancestor of a member anchor must also
            be anchored, so an absolute anchor cannot drift with its parent.

    Conditions do not authorize new motion, topology changes or physics steps.
    Geometric infeasibility yields an empty/budget-exhausted generation result;
    malformed or unsupported conditions reject before candidate generation.
    """

    poses: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.poses, Mapping):
            raise TypeError("Condition poses must be a UID-to-pose mapping")
        owned = {}
        for uid, pose in self.poses.items():
            _identifier(uid, "condition UID")
            owned[uid] = _rigid_pose(pose)
        object.__setattr__(self, "poses", MappingProxyType(dict(sorted(owned.items()))))

    def to_dict(self) -> dict[str, list[list[float]]]:
        """Return a detached JSON-compatible projection of the anchors."""
        return {uid: [list(row) for row in pose] for uid, pose in self.poses.items()}


def _condition_prepared(
    prepared: PreparedSceneRandomization, condition: SceneLayoutCondition
) -> PreparedSceneRandomization:
    """Compile anchors to zero-width bounds, keeping existing geometry checks."""
    if not isinstance(condition, SceneLayoutCondition):
        raise TypeError("conditions must be SceneLayoutCondition")
    if not condition.poses:
        return prepared
    source, policy = prepared.source, prepared.policy
    unknown = set(condition.poses) - set(source.poses)
    if unknown:
        raise ValueError(f"Unknown condition entities: {sorted(unknown)}")
    nodes = prepared._geometry["nodes"]
    index = {n["uid"]: i for i, n in enumerate(nodes)}
    roots = {nodes[i]["uid"] for i in prepared._movable_roots}
    targets = policy["intra_group"]["targets"]
    variable = roots | set(targets)
    world = {source.support_uid: np.asarray(source.poses[source.support_uid])}
    for n in nodes:
        uid = n["uid"]
        original = np.asarray(source.poses[uid])
        parent_uid = source.parent_by_uid[uid]
        anchored = uid in condition.poses
        if anchored:
            parent = parent_uid
            while parent != source.support_uid:
                if parent in variable and parent not in condition.poses:
                    raise _LayoutConditionDomainError(
                        f"Anchor {uid!r} requires its moving ancestor {parent!r} to be anchored"
                    )
                parent = source.parent_by_uid[parent]
        parent_world = world[parent_uid]
        if n["parent"] < 0:
            if not anchored:
                world[uid] = original
                continue
            wanted = np.asarray(condition.poses[uid])
            if (
                not np.allclose(wanted[:3, :3], original[:3, :3], rtol=0, atol=1e-9)
                or abs(wanted[2, 3] - original[2, 3]) > 1e-9
            ):
                raise _LayoutConditionDomainError(
                    "Root anchors support declared XY translation only; rotation/height must remain authored"
                )
            delta = wanted[:2, 3] - original[:2, 3]
            if uid not in roots:
                if not np.allclose(delta, 0, rtol=0, atol=1e-9):
                    raise _LayoutConditionDomainError(
                        f"Condition cannot move fixed group {uid!r}"
                    )
            else:
                bounds = policy["group_xy_offset_bounds_m"].get(uid)
                if bounds is not None and (
                    np.any(delta < np.asarray(bounds)[0] - 1e-9)
                    or np.any(delta > np.asarray(bounds)[1] + 1e-9)
                ):
                    raise _LayoutConditionDomainError(
                        "Anchor lies outside declared root XY bounds"
                    )
                policy["group_xy_offset_bounds_m"][uid] = [
                    delta.tolist(),
                    delta.tolist(),
                ]
            world[uid] = wanted
            continue
        relative = np.asarray(n["relative_pose"])
        if not anchored:
            world[uid] = parent_world @ relative
            continue
        wanted = np.asarray(condition.poses[uid])
        if uid not in targets:
            if not np.allclose(wanted, parent_world @ relative, rtol=0, atol=1e-9):
                raise _LayoutConditionDomainError(
                    f"Condition cannot move undeclared member {uid!r}"
                )
            world[uid] = wanted
            continue
        spec = targets[uid]
        support = prepared._geometry["supports"][parent_uid]
        inv_frame = np.linalg.inv(np.asarray(support["parent_from_support"]))
        baseline = inv_frame @ relative
        local = inv_frame @ np.linalg.inv(parent_world) @ wanted
        rotation = local[:3, :3] @ np.linalg.inv(baseline[:3, :3])
        angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        low, high = spec["yaw_delta_range_deg"]
        angle += 360 * math.floor(((low + high) / 2 - angle) / 360 + 0.5)
        height = (
            policy["intra_group"]["support_gap_m"]
            + support.get("height_ceiling_m", 0)
            - support["root_min_z"][uid]
        )
        if (
            not np.allclose(rotation, yaw_rotation, rtol=0, atol=1e-9)
            or abs(local[2, 3] - height) > 1e-9
        ):
            raise _LayoutConditionDomainError(
                "Member anchor violates its authored tilt or support height"
            )
        if angle < low - 1e-8 or angle > high + 1e-8:
            raise _LayoutConditionDomainError(
                "Member anchor lies outside declared yaw bounds"
            )
        offset = local[:2, 3] - baseline[:2, 3]
        bounds = spec["xy_offset_bounds_m"]
        if bounds is not None and (
            np.any(offset < np.asarray(bounds)[0] - 1e-9)
            or np.any(offset > np.asarray(bounds)[1] + 1e-9)
        ):
            raise _LayoutConditionDomainError(
                "Member anchor lies outside declared XY bounds"
            )
        spec["xy_offset_bounds_m"] = [offset.tolist(), offset.tolist()]
        spec["yaw_delta_range_deg"] = [angle, angle]
        world[uid] = wanted
    if source.support_uid in condition.poses and not np.allclose(
        condition.poses[source.support_uid],
        source.poses[source.support_uid],
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("The main support is fixed")
    return replace(prepared, _policy=deepcopy(policy))
