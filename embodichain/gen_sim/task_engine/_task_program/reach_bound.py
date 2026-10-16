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

"""Necessary-only reach bounds for a fixed-base rotational URDF chain."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["RevoluteReachBound", "build_revolute_reach_bound"]


def _se3(value: Any, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real SE(3) transform.")
    try:
        pose = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite 4x4 SE(3) transform.") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite 4x4 SE(3) transform.")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-6, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must contain a proper SE(3) transform.")
    return pose


def _tolerance(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class RevoluteReachBound:
    """Outer sphere for the last rotational pivot, not an IK certificate.

    ``tail`` maps the last rotational joint's child frame to the configured TCP.
    The bound ignores joint limits, axis restrictions, collisions and dynamics.
    Arrays are detached and read-only; a caller cannot alter the selected model.
    """

    pivot_world: np.ndarray
    radius: float
    tail: np.ndarray

    def __post_init__(self) -> None:
        pivot = np.array(self.pivot_world, dtype=float, copy=True)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError("pivot_world must be one finite three-vector.")
        radius = _tolerance(self.radius, "radius")
        tail = _se3(self.tail, "tail")
        pivot.setflags(write=False)
        tail.setflags(write=False)
        object.__setattr__(self, "pivot_world", pivot)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "tail", tail)

    def check(
        self,
        target_tcp: np.ndarray,
        pos_tolerance: float = 0.0,
        rot_tolerance: float = 0.0,
    ) -> dict[str, bool | float | str]:
        """Reject only targets outside the tolerance-expanded necessary sphere.

        Position tolerance is in metres; rotation tolerance is an angle in
        radians. Rotation uncertainty moves the recovered pivot by at most
        ``2 * ||tail.translation|| * sin(min(angle, pi) / 2)``. Passing this
        check proves neither IK feasibility nor a safe or executable trajectory.
        """
        target = _se3(target_tcp, "target_tcp")
        position = _tolerance(pos_tolerance, "pos_tolerance")
        angle = _tolerance(rot_tolerance, "rot_tolerance")
        pivot_target = target @ np.linalg.inv(self.tail)
        distance = float(np.linalg.norm(pivot_target[:3, 3] - self.pivot_world))
        allowance = position + 2 * float(np.linalg.norm(self.tail[:3, 3])) * np.sin(
            min(angle, np.pi) / 2
        )
        return {
            "within_bound": bool(
                distance <= np.nextafter(self.radius + allowance, np.inf)
            ),
            "required_distance_m": distance,
            "radius_m": self.radius,
            "tolerance_allowance_m": float(allowance),
            "scope": "necessary_only",
            "ik_qualified": False,
        }


def _vector(element: ET.Element | None, key: str, default: str) -> np.ndarray:
    raw = default if element is None else element.get(key, default)
    try:
        value = np.array([float(item) for item in raw.split()], dtype=float)
    except ValueError as error:
        raise ValueError(f"URDF {key} must contain three finite numbers.") from error
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"URDF {key} must contain three finite numbers.")
    return value


def _origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    pose = np.eye(4)
    pose[:3, 3] = _vector(origin, "xyz", "0 0 0")
    pose[:3, :3] = Rotation.from_euler(
        "xyz", _vector(origin, "rpy", "0 0 0")
    ).as_matrix()
    return pose


def build_revolute_reach_bound(
    urdf_path: str | Path,
    root_link: str,
    end_link: str,
    root_world: np.ndarray,
    tcp: np.ndarray,
) -> RevoluteReachBound:
    """Build a necessary bound from one explicit URDF root-to-end chain.

    Fixed joints before the first rotational joint locate its world pivot.
    The triangle inequality bounds the distance to the last rotational pivot
    by the sum of intervening origin-translation norms. Fixed joints after it
    and TCP form the removable tail. Revolute and continuous joints are
    supported; prismatic, floating and planar chains are explicitly rejected.

    Raises:
        ValueError: For ambiguous/disconnected/cyclic chains or invalid geometry.
        NotImplementedError: For a selected joint type without this bound.
    """
    world, tool = _se3(root_world, "root_world"), _se3(tcp, "tcp")
    robot = ET.parse(Path(urdf_path)).getroot()
    if robot.tag != "robot":
        raise ValueError("The URDF root element must be robot.")
    names = [element.get("name", "") for element in robot.findall("link")]
    if any(not name or name != name.strip() for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("URDF links must have unique nonempty names.")
    if root_link not in names or end_link not in names:
        raise ValueError("The selected root/end links must exist in the URDF.")
    by_child: dict[str, ET.Element] = {}
    joint_names: set[str] = set()
    for joint in robot.findall("joint"):
        name = joint.get("name", "")
        parent, child = joint.find("parent"), joint.find("child")
        parent_name = None if parent is None else parent.get("link")
        child_name = None if child is None else child.get("link")
        if (
            not name
            or name in joint_names
            or parent_name not in names
            or child_name not in names
            or child_name in by_child
        ):
            raise ValueError("URDF joints must have unique names and child links.")
        joint_names.add(name)
        by_child[child_name] = joint
    chain, visited, current = [], set(), end_link
    while current != root_link:
        if current in visited:
            raise ValueError("The selected URDF chain contains a cycle.")
        visited.add(current)
        joint = by_child.get(current)
        if joint is None:
            raise ValueError("The end link is not descended from the selected root.")
        chain.append(joint)
        current = joint.find("parent").get("link")
    chain.reverse()
    moving = []
    for index, joint in enumerate(chain):
        kind = joint.get("type")
        if kind not in {"fixed", "revolute", "continuous"}:
            raise NotImplementedError(f"Unsupported reach-bound joint type: {kind!r}.")
        if kind != "fixed":
            if np.linalg.norm(_vector(joint.find("axis"), "xyz", "1 0 0")) == 0:
                raise ValueError("A rotational joint axis must be nonzero.")
            moving.append(index)
    if not moving:
        raise ValueError("The selected chain requires a rotational joint.")
    origins = [_origin(joint) for joint in chain]
    first, last = moving[0], moving[-1]
    pivot = world.copy()
    for origin in origins[: first + 1]:
        pivot = pivot @ origin
    radius = sum(float(np.linalg.norm(p[:3, 3])) for p in origins[first + 1 : last + 1])
    tail = np.eye(4)
    for origin in origins[last + 1 :]:
        tail = tail @ origin
    return RevoluteReachBound(pivot[:3, 3], radius, tail @ tool)
