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

"""Mechanical self-collision declarations for the existing cuRobo model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import xml.etree.ElementTree as ET

__all__: list[str] = []


def _self_collision_config(
    urdf_path: str | None,
    collision_links: Sequence[str],
    *,
    ignore: Mapping[str, Sequence[str]] | None = None,
    buffer: float | Mapping[str, float] = 0.0,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Normalize authored exemptions or derive neighboring rigid components.

    Fixed joints join one rigid component. Only components connected by one
    moving joint are mechanically adjacent. No robot/task names, initial-pose
    collisions, or entire end-effector subtrees select exemptions.
    """
    links = tuple(collision_links)
    if len(set(links)) != len(links) or any(
        type(name) is not str or not name or name != name.strip() for name in links
    ):
        raise ValueError("Self-collision links require unique exact identifiers.")
    known = set(links)
    if ignore is not None:
        if not isinstance(ignore, Mapping) or set(ignore) - known:
            raise ValueError("Self-collision exemptions name unknown collision links.")
        ignored = {}
        for name, values in ignore.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError("Self-collision exemptions require link sequences.")
            values = tuple(values)
            if len(set(values)) != len(values) or any(
                type(other) is not str or other not in known or other == name
                for other in values
            ):
                raise ValueError(
                    "Self-collision exemptions require distinct known links."
                )
            ignored[name] = list(values)
    else:
        ignored = {}
        if urdf_path is not None:
            tree = ET.parse(urdf_path).getroot()
            parents = {
                node.attrib["name"]: node.attrib["name"]
                for node in tree.findall("link")
            }
            if known - parents.keys():
                raise ValueError(
                    "Collision geometry names links outside the assembled URDF."
                )

            def root(name: str) -> str:
                while parents[name] != name:
                    parents[name] = parents[parents[name]]
                    name = parents[name]
                return name

            joints = []
            for joint in tree.findall("joint"):
                a, b = joint.find("parent"), joint.find("child")
                if a is None or b is None:
                    raise ValueError(
                        "Self-collision topology requires complete URDF joints."
                    )
                a, b = a.attrib["link"], b.attrib["link"]
                if a not in parents or b not in parents:
                    raise ValueError("URDF joint references an unknown link.")
                joints.append((a, b))
                if joint.attrib["type"] == "fixed":
                    parents[root(b)] = root(a)
            adjacent = {frozenset((root(a), root(b))) for a, b in joints}
            for index, name in enumerate(links):
                values = [
                    other
                    for other in links[index + 1 :]
                    if root(name) == root(other)
                    or frozenset((root(name), root(other))) in adjacent
                ]
                if values:
                    ignored[name] = values
    if isinstance(buffer, Mapping):
        if set(buffer) - known:
            raise ValueError("Self-collision buffers name unknown collision links.")
        buffers = {name: buffer.get(name, 0.0) for name in links}
    else:
        buffers = {name: buffer for name in links}
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in buffers.values()
    ):
        raise ValueError("Self-collision buffers must be finite nonnegative distances.")
    return ignored, {name: float(value) for name, value in buffers.items()}


def _complete_self_collision_config(kinematics: dict, *, config_path: str) -> None:
    """Preserve authored maps and fill only absent cuRobo model declarations."""
    links = kinematics.get("collision_link_names")
    if links is None:
        links = tuple(kinematics.get("collision_spheres") or ())
    urdf_path = kinematics.get("urdf_path")
    if urdf_path is not None:
        path = Path(urdf_path)
        if not path.is_absolute():
            path = (
                Path(kinematics.get("asset_root_path") or Path(config_path).parent)
                / path
            )
        urdf_path = str(path)
    ignored, buffers = _self_collision_config(
        urdf_path,
        links,
        ignore=kinematics.get("self_collision_ignore"),
        buffer=kinematics.get("self_collision_buffer") or 0.0,
    )
    if kinematics.get("self_collision_ignore") is None:
        kinematics["self_collision_ignore"] = ignored
    if kinematics.get("self_collision_buffer") is None:
        kinematics["self_collision_buffer"] = buffers
