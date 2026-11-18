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
"""Simulator-independent assembly manifests with explicit body ownership."""

from __future__ import annotations

from dataclasses import asdict, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from embodichain.utils import configclass

__all__ = ["AssemblyGeometry", "AssemblyJoint", "AssemblyBody", "AssemblyAsset"]


@configclass
class AssemblyGeometry:
    """One mesh or primitive in its owning body's local frame.

    ``size`` follows MuJoCo: box half extents, cylinder radius/half height,
    sphere radius. ``inertia`` is never computed from visual geometry.
    """

    name: str = ""
    kind: str = "mesh"
    position: list[float] = [0.0, 0.0, 0.0]
    quaternion: list[float] = [0.0, 0.0, 0.0, 1.0]
    vertices: list[list[float]] = field(default_factory=list)
    triangles: list[list[int]] = field(default_factory=list)
    size: list[float] = field(default_factory=list)
    rgba: list[float] = [0.65, 0.65, 0.7, 1.0]
    collision: bool = True
    visible: bool = True
    friction: list[float] = [1.0, 0.01, 0.001]
    condim: int = 3
    priority: int = 0
    solmix: float = 1.0
    solref: list[float] = [0.002, 1.0]
    solimp: list[float] = [0.99, 0.999, 0.001, 0.5, 2.0]


@configclass
class AssemblyJoint:
    """A supported world attachment; axis and anchor are body-local.

    Free bodies have no constraint. A hinge is retained as a real joint by
    both USD export and the direct route's single-link URDF articulation.
    """

    name: str = ""
    kind: str = "fixed"
    position: list[float] = [0.0, 0.0, 0.0]
    axis: list[float] = [0.0, 0.0, 1.0]
    damping: float = 0.0
    armature: float = 0.0
    frictionloss: float = 0.0
    motor_gear: float = 0.0
    control_range: list[float] = [-1.0, 1.0]


@configclass
class AssemblyBody:
    """A single physical body, independent collision pieces and one inertia."""

    name: str = ""
    position: list[float] = [0.0, 0.0, 0.0]
    quaternion: list[float] = [0.0, 0.0, 0.0, 1.0]
    mass: float = 0.0
    com: list[float] = [0.0, 0.0, 0.0]
    inertia: list[list[float]] = field(default_factory=list)
    geometries: list[AssemblyGeometry] = field(default_factory=list)
    joint: AssemblyJoint = AssemblyJoint()

    @property
    def body_type(self) -> str:
        """Return the rigid-body type implied by its attachment."""
        return "static" if self.joint.kind == "fixed" else "dynamic"


@configclass
class AssemblyAsset:
    """Portable, content-addressed assembly and its source provenance.

    The manifest is self-contained. It never requires a generator dependency
    at load time, and ``source_mjcf`` preserves the independent reference.
    """

    schema_version: int = 1
    converter_version: str = "1"
    generator_revision: str = ""
    generator_version: str = ""
    source_mjcf: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    bodies: list[AssemblyBody] = field(default_factory=list)
    source_options: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 identity of all geometry, physics and source metadata."""
        return hashlib.sha256(self._canonical_bytes()).hexdigest()

    def _canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    def save(self, path: str | Path) -> Path:
        """Atomically write a self-contained manifest and content checksum."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {"sha256": self.content_hash, "asset": asdict(self)}
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    @classmethod
    def load(cls, path: str | Path) -> AssemblyAsset:
        """Read and verify a manifest without importing a simulator."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = data["asset"]
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported assembly manifest schema.")
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if data.get("sha256") != hashlib.sha256(encoded).hexdigest():
            raise ValueError("Assembly manifest checksum mismatch.")
        bodies = []
        for raw_body in payload.pop("bodies"):
            geometries = [AssemblyGeometry(**g) for g in raw_body.pop("geometries")]
            joint = AssemblyJoint(**raw_body.pop("joint"))
            bodies.append(AssemblyBody(**raw_body, geometries=geometries, joint=joint))
        result = cls(**payload, bodies=bodies)
        return result

    def write_usd(self, path: str | Path) -> Path:
        """Export the same collision pieces, masses and joint attachments."""
        from .usd import write_assembly_usd

        return write_assembly_usd(self, path)
