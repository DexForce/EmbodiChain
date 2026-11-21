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

"""Portable collision/kinematic identities and pinned legacy reference migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

__all__: list[str] = []

_SCHEMA = "embodichain-real2sim-robot-geometry/v2"
_NUMERIC_ATTRIBUTES = frozenset(
    {
        "xyz",
        "rpy",
        "scale",
        "size",
        "radius",
        "length",
        "lower",
        "upper",
        "effort",
        "velocity",
        "multiplier",
        "offset",
        "damping",
        "friction",
        "reference_position",
        "rising",
        "falling",
        "soft_lower_limit",
        "soft_upper_limit",
        "k_position",
        "k_velocity",
    }
)

# Real2SimCartonV1's immutable published source binds this byte digest. The v2
# value is independently computed from the URDF that reproduces that v1 digest,
# including its collision meshes (FrankaV2 + Robotiq_V2, 18 mm tool mount).
# Never infer aliases from the live robot or rewrite the published sources.
_LEGACY_ROBOT_GEOMETRY = {
    "6b08aff1b8de77edfd1deab3da3f139d2ecfb20838bd21cf2d6207058c0d4d82": "ca151f538efa2025baecfa7d8e66ca5ef5cd47f57be6435aa3e65ef425f9a1ff",
}


def _canonical_robot_geometry_sha256(value: str | None) -> str | None:
    """Resolve only independently pinned legacy identities; unknowns stay strict."""
    return _LEGACY_ROBOT_GEOMETRY.get(value, value) if value is not None else None


def _robot_geometry_sha256(path: Path, mesh_paths: tuple[Path, ...]) -> str:
    """Hash named joints and collision geometry, independent of assembly metadata.

    Mesh references are replaced by content digests. Numerical attributes use
    exact binary64 values, normalizing signed zero without rounding geometry.
    Visuals, inertials, XML comments and the assembly's robot label are excluded.
    Joint names, limits, mimic relationships and all joint attributes remain.
    """
    root = ET.parse(path).getroot()
    mesh_digests = {
        mesh: hashlib.sha256(mesh.read_bytes()).hexdigest() for mesh in mesh_paths
    }

    def mesh_digest(filename: str) -> str:
        candidate = (path.parent / filename).resolve()
        if candidate in mesh_digests:
            return mesh_digests[candidate]
        # Pinocchio resolves package URIs before handing us mesh_paths. Match
        # their package-relative suffix rather than retaining an absolute path.
        if filename.startswith("package://"):
            relative = filename.removeprefix("package://").split("/", 1)[-1]
            matches = {
                digest
                for mesh, digest in mesh_digests.items()
                if mesh.as_posix().endswith("/" + relative)
            }
            if len(matches) == 1:
                return matches.pop()
        raise ValueError(
            f"collision mesh is not resolved by the loaded model: {filename}"
        )

    def element(node: ET.Element) -> tuple:
        attributes = {}
        for key, value in node.attrib.items():
            if node.tag == "mesh" and key == "filename":
                attributes[key] = mesh_digest(value)
            elif key in _NUMERIC_ATTRIBUTES:
                attributes[key] = tuple(
                    (number if number else 0.0).hex()
                    for number in map(float, value.split())
                )
            else:
                attributes[key] = value
        children = sorted((element(child) for child in node), key=repr)
        return node.tag, sorted(attributes.items()), children

    links = []
    for link in root.findall("link"):
        links.append(
            (
                link.attrib["name"],
                sorted(
                    (element(collision) for collision in link.findall("collision")),
                    key=repr,
                ),
            )
        )
    payload = {
        "schema": _SCHEMA,
        "links": sorted(links),
        "joints": sorted((element(joint) for joint in root.findall("joint")), key=repr),
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
