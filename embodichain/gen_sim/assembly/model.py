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
from typing import Any, TYPE_CHECKING

import numpy as np

from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg
    from embodichain.lab.sim.shapes import ShapePartCfg

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

    def to_shape_part(self) -> ShapePartCfg:
        """Construct a native subshape without re-decomposing the body."""
        from embodichain.lab.sim.shapes import (
            CubeCfg,
            CylinderCfg,
            MeshCfg,
            MeshCollisionCfg,
            ShapePartCfg,
            SphereCfg,
        )
        from embodichain.lab.sim.material import VisualMaterialCfg

        if self.kind == "mesh":
            shape = MeshCfg(
                vertices=self.vertices,
                triangles=self.triangles,
                collision=MeshCollisionCfg(approximation="convex_hull"),
            )
        elif self.kind == "cylinder":
            shape = CylinderCfg(radius=self.size[0], half_height=self.size[1])
        elif self.kind == "box":
            shape = CubeCfg(size=[2 * x for x in self.size[:3]])
        elif self.kind == "sphere":
            shape = SphereCfg(radius=self.size[0])
        else:
            raise ValueError(f"Unsupported assembly geometry {self.kind!r}.")
        shape.visual_material = VisualMaterialCfg(base_color=list(self.rgba))
        return ShapePartCfg(
            shape=shape,
            position=tuple(self.position),
            quaternion=tuple(self.quaternion),
        )


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

    def to_rigid_config(
        self,
        *,
        uid: str | None = None,
        source_contact: bool = True,
        backend: str = "newton",
    ) -> RigidObjectCfg:
        """Create one compound body; hinges require articulation construction."""
        from embodichain.lab.sim.cfg import (
            CollisionPropertiesCfg,
            MassPropertiesCfg,
            NewtonCollisionPropertiesCfg,
            NewtonRigidBodyMaterialCfg,
            RigidBodyMaterialCfg,
            RigidBodyPhysicsCfg,
            RigidObjectCfg,
        )
        from embodichain.lab.sim.shapes import CompoundShapeCfg
        from scipy.spatial.transform import Rotation

        if self.joint.kind == "hinge":
            raise ValueError(
                "Hinged assembly bodies require an articulation, not a free rigid body."
            )
        collisions = [g for g in self.geometries if g.collision]
        if not collisions:
            raise ValueError(f"Assembly body {self.name!r} has no collision geometry.")
        reference = collisions[0]
        for geom in collisions[1:]:
            if any(
                getattr(geom, key) != getattr(reference, key)
                for key in (
                    "friction",
                    "condim",
                    "solref",
                    "solimp",
                    "priority",
                    "solmix",
                )
            ):
                raise ValueError(
                    "Per-piece contact properties cannot be represented by a body-wide physics overlay."
                )
        if backend not in {"newton", "default"}:
            raise ValueError("Assembly body backend must be newton or default.")
        if backend == "default" and source_contact:
            raise NotImplementedError(
                "Default cannot preserve MuJoCo solref/solimp or torsional/rolling "
                "friction. Set source_contact=False to explicitly measure its backend contact model."
            )
        collision_props = (
            NewtonCollisionPropertiesCfg(
                contact_offset=0.0,
                rest_offset=0.0,
                condim=reference.condim,
                priority=reference.priority,
                solref=tuple(reference.solref),
                solimp=tuple(reference.solimp),
                solmix=reference.solmix,
            )
            if backend == "newton" and source_contact
            else CollisionPropertiesCfg(contact_offset=0.0, rest_offset=0.0)
        )
        shape = CompoundShapeCfg(
            parts=[g.to_shape_part() for g in collisions],
            visual_parts=[g.to_shape_part() for g in self.geometries if g.visible],
        )
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(self.quaternion).as_matrix()
        transform[:3, 3] = self.position
        return RigidObjectCfg(
            uid=uid or self.name,
            shape=shape,
            init_local_pose=transform,
            body_type=self.body_type,
            attrs=RigidBodyPhysicsCfg(
                mass_props=MassPropertiesCfg(
                    mass=self.mass if self.body_type == "dynamic" else None,
                    inertia=self.inertia if self.body_type == "dynamic" else None,
                    com_position=self.com if self.body_type == "dynamic" else None,
                ),
                collision_props=collision_props,
                material_props=(
                    NewtonRigidBodyMaterialCfg(
                        dynamic_friction=reference.friction[0],
                        torsional_friction=reference.friction[1],
                        rolling_friction=reference.friction[2],
                    )
                    if backend == "newton"
                    else RigidBodyMaterialCfg(
                        static_friction=reference.friction[0],
                        dynamic_friction=reference.friction[0],
                    )
                ),
            ),
        )


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

    def to_rigid_configs(
        self, *, prefix: str = "", source_contact: bool = True, backend: str = "newton"
    ) -> dict[str, RigidObjectCfg]:
        """Create native configs for free/fixed bodies; reject lost constraints."""
        return {
            body.name: body.to_rigid_config(
                uid=prefix + body.name, source_contact=source_contact, backend=backend
            )
            for body in self.bodies
        }

    def write_usd(self, path: str | Path) -> Path:
        """Export the same collision pieces, masses and joint attachments."""
        from .usd import write_assembly_usd

        return write_assembly_usd(self, path)

    def to_articulation_configs(
        self,
        directory: str | Path,
        *,
        source_contact: bool = True,
        backend: str = "newton",
    ) -> dict[str, ArticulationCfg]:
        """Write single-link hinge URDFs for native articulation construction."""
        from .urdf import articulation_configs

        return articulation_configs(
            self, directory, source_contact=source_contact, backend=backend
        )
