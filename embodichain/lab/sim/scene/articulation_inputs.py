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

"""Read-only, fingerprinted bindings for generated single-prismatic USD assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
from numbers import Real
from pathlib import Path
import re
from typing import Any

import numpy as np

from embodichain.lab.sim.spawn.source_geometry import UsdSourceSnapshot, read_usd_source

__all__ = [
    "PrismaticBinding",
    "ArticulationPartCandidate",
    "discover_prismatic_parts",
    "inspect_prismatic",
    "inspect_prismatic_part",
    "handle_mesh",
]


def _fixed_base(config: dict[str, Any]) -> bool:
    if "fix_base" in config:
        return config["fix_base"] is True
    root_props = config.get("root_props")
    return isinstance(root_props, dict) and root_props.get("fixed_base") is True


@dataclass(frozen=True, slots=True)
class PrismaticBinding:
    """Source-qualified geometry with optional open/closed endpoint semantics.

    Geometry inspection preserves an unknown ``closed_position``. Resolving
    an open/closed target requires an explicit source or task-owned declaration.
    """

    object_id: str
    joint: str
    link: str
    parent: str
    handle_path: str
    source_sha256: str
    scale: float
    axis: tuple[float, float, float]
    limits: tuple[float, float]
    part_id: str = ""
    closed_position: float | None = None

    def __post_init__(self) -> None:
        for name in ("object_id", "joint", "link", "parent", "handle_path"):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"Prismatic {name} must be an exact identifier.")
        if self.link == self.parent or not self.handle_path.startswith("/"):
            raise ValueError(
                "Prismatic binding requires distinct bodies and an absolute handle path."
            )
        if self.part_id and (
            not self.part_id.startswith("part_") or "/" in self.part_id
        ):
            raise ValueError("Prismatic part_id must be a stable local identifier.")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("Prismatic binding requires an asset SHA-256.")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, Real)
            or not np.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("Prismatic scale must be finite and positive.")
        for value, size in ((self.axis, 3), (self.limits, 2)):
            if (
                not isinstance(value, (tuple, list))
                or len(value) != size
                or any(isinstance(x, bool) or not isinstance(x, Real) for x in value)
            ):
                raise ValueError("Prismatic axes and limits must contain real numbers.")
        if (
            len(self.axis) != 3
            or not np.isfinite(self.axis).all()
            or not np.isclose(np.linalg.norm(self.axis), 1.0, atol=1e-6)
        ):
            raise ValueError("Prismatic axis must be a finite unit vector.")
        if (
            len(self.limits) != 2
            or not np.isfinite(self.limits).all()
            or self.limits[0] >= self.limits[1]
        ):
            raise ValueError("Prismatic limits must be finite and strictly increasing.")
        if self.limits[1] - self.limits[0] <= 0.004:
            raise ValueError(
                "Prismatic endpoints must be distinguishable at millimetre resolution."
            )
        object.__setattr__(self, "axis", tuple(map(float, self.axis)))
        object.__setattr__(self, "limits", tuple(map(float, self.limits)))
        if self.closed_position is not None:
            if (
                isinstance(self.closed_position, bool)
                or not isinstance(self.closed_position, Real)
                or not np.isfinite(self.closed_position)
                or not any(
                    np.isclose(self.closed_position, endpoint, atol=1e-6, rtol=1e-6)
                    for endpoint in self.limits
                )
            ):
                raise ValueError(
                    "closed_position must name a finite joint-limit endpoint."
                )
            object.__setattr__(self, "closed_position", float(self.closed_position))

    @property
    def axis_sign(self) -> int:
        """Return the inward direction relative to increasing joint position."""
        return -1 if self.target("open") > self.target("closed") else 1

    def target(self, state: str) -> float:
        """Resolve a declared endpoint without evaluating runtime success.

        Args:
            state: Either ``open`` or ``closed``.

        Returns:
            The matching authored joint coordinate in runtime metres.

        Raises:
            ValueError: The state is invalid or closed-position semantics are unknown.
        """
        if state not in {"open", "closed"}:
            raise ValueError("Prismatic target state must be open or closed.")
        if self.closed_position is None:
            raise ValueError(
                "Prismatic open/closed targets require an explicit closed_position "
                "in the binding or gen_sim:closedPosition in the source USD."
            )
        closed = self.closed_position
        return (
            closed
            if state == "closed"
            else max(self.limits, key=lambda p: abs(p - closed))
        )

    def payload(self) -> dict[str, Any]:
        """Return a detached declaration, omitting absent optional fields."""
        value = asdict(self)
        if not self.part_id:
            value.pop("part_id")
        if self.closed_position is None:
            value.pop("closed_position")
        return value

    @classmethod
    def decode(cls, value: object) -> PrismaticBinding:
        """Validate exactly the declared physical binding fields.

        Args:
            value: Plain decoded binding mapping.

        Returns:
            An independently normalized binding.
        """
        allowed = {f.name for f in fields(cls)}
        required = allowed - {"part_id", "closed_position"}
        if (
            type(value) is not dict
            or not required.issubset(value)
            or not set(value).issubset(allowed)
        ):
            raise ValueError("Prismatic binding requires exactly its declared fields.")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ArticulationPartCandidate:
    """Deterministic, execution-free description of one prismatic part."""

    part_id: str
    joint_path: str
    joint: str
    parent_path: str
    parent: str
    link_path: str
    link: str
    handle_paths: tuple[str, ...]
    axis: tuple[float, float, float]
    limits: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.part_id or not self.joint_path or not self.link_path:
            raise ValueError("Articulation part identity must be non-empty.")
        if not self.handle_paths:
            raise ValueError(
                "Articulation part requires at least one handle candidate."
            )
        if len(set(self.handle_paths)) != len(self.handle_paths):
            raise ValueError("Articulation part handle paths must be unique.")
        if len(self.axis) != 3 or not np.isfinite(self.axis).all():
            raise ValueError("Articulation part axis must be finite and 3D.")
        if not np.isclose(np.linalg.norm(self.axis), 1.0, atol=1e-6):
            raise ValueError("Articulation part axis must be unit length.")
        if len(self.limits) != 2 or not np.isfinite(self.limits).all():
            raise ValueError("Articulation part limits must be finite and 2D.")
        if self.limits[0] >= self.limits[1]:
            raise ValueError("Articulation part limits must be increasing.")
        for name in ("handle_paths", "axis", "limits"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def payload(self) -> dict[str, Any]:
        """Return a detached serializable candidate description."""
        return {
            "part_id": self.part_id,
            "joint_path": self.joint_path,
            "joint": self.joint,
            "parent_path": self.parent_path,
            "parent": self.parent,
            "link_path": self.link_path,
            "link": self.link,
            "handle_paths": list(self.handle_paths),
            "axis": [float(value) for value in self.axis],
            "limits": [float(value) for value in self.limits],
        }


def discover_prismatic_parts(
    config: dict[str, Any],
    *,
    source: UsdSourceSnapshot | None = None,
) -> tuple[ArticulationPartCandidate, ...]:
    """Discover pullable prismatic parts without selecting one for execution.

    The result is intentionally a catalog candidate.  Handle selection and
    runtime binding remain separate so that natural-language grounding cannot
    bypass deterministic USD validation. Named handleless push controls belong
    to Press, not this catalog; other handleless parts still reject.

    Args:
        config: Fixed-base asset declaration with positive uniform scale.
        source: Optional shared static read matching the declared asset.

    Returns:
        Candidates in deterministic part-identity order.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    if not _fixed_base(config):
        raise ValueError("GenSim E6 requires a declared fixed base.")
    scale = np.asarray(config.get("body_scale", (1.0, 1.0, 1.0)), dtype=float)
    if (
        scale.shape != (3,)
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.allclose(scale, scale[0], atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("GenSim E6 requires positive uniform body_scale.")
    source = read_usd_source(config["fpath"] if source is None else source)
    source.assert_current(config["fpath"])
    source.require_geometry()
    stage = source.stage
    source_path = source.path
    source_hash = source.source_sha256
    if source_path.with_suffix(".mass.json").is_file():
        from .control_mass_source import qualify_mass_source_copy

        # Physics-only copies retain geometric IDs; bindings still seal runtime bytes.
        source_hash = qualify_mass_source_copy(source_path).source_sha256
    fixed_children: dict[str, set[str]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        fixed = UsdPhysics.Joint(prim)
        if fixed.GetJointEnabledAttr().Get() is False:
            continue
        parents = fixed.GetBody0Rel().GetTargets()
        children = fixed.GetBody1Rel().GetTargets()
        if len(parents) == 1 and len(children) == 1:
            fixed_children.setdefault(str(parents[0]), set()).add(str(children[0]))

    candidates: list[ArticulationPartCandidate] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        joint = UsdPhysics.PrismaticJoint(prim)
        if UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is False:
            continue
        parents, children = (
            joint.GetBody0Rel().GetTargets(),
            joint.GetBody1Rel().GetTargets(),
        )
        if len(parents) != 1 or len(children) != 1:
            raise ValueError(
                f"Prismatic joint {prim.GetPath()} requires one parent and child."
            )
        parent = stage.GetPrimAtPath(parents[0])
        body = stage.GetPrimAtPath(children[0])
        if not parent.HasAPI(UsdPhysics.RigidBodyAPI) or not body.HasAPI(
            UsdPhysics.RigidBodyAPI
        ):
            raise ValueError(
                f"Prismatic joint {prim.GetPath()} bodies must be rigid bodies."
            )
        # USD namespace ancestry need not match physical fixed-joint ancestry.
        attached_bodies = {str(children[0])}
        pending = list(attached_bodies)
        while pending:
            for child_path in fixed_children.get(pending.pop(), ()):
                if child_path not in attached_bodies:
                    attached_bodies.add(child_path)
                    pending.append(child_path)

        def belongs_to_part(mesh_prim: Any) -> bool:
            owner = mesh_prim
            while owner and not owner.IsPseudoRoot():
                if owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    return str(owner.GetPath()) in attached_bodies
                owner = owner.GetParent()
            return False

        handles = tuple(
            str(p.GetPath())
            for p in stage.Traverse()
            if p.IsA(UsdGeom.Mesh)
            and belongs_to_part(p)
            and p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
            and any(
                token in p.GetName().lower() for token in ("handle", "grip", "pull")
            )
            and not any(
                token in p.GetName().lower()
                for token in ("mount", "support", "post", "leg", "plate")
            )
        )
        # Legacy generators often decompose one U-handle into collision meshes
        # for feet, drops and one explicitly named grip. Prefer only declared
        # grip bars when present; multiple grips remain deliberately ambiguous.
        primary_grips = tuple(
            path
            for path in handles
            if "grip" in path.rsplit("/", 1)[-1].lower()
            or "pull_bar" in path.rsplit("/", 1)[-1].lower()
        )
        if primary_grips:
            handles = primary_grips
        if not handles and re.search(
            r"(?:^|[_\s])(?:button|press|plunger|switch)(?:[_\s]|$)",
            f"{prim.GetName()} {body.GetName()}",
            re.I,
        ):
            continue
        axis = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[str(joint.GetAxisAttr().Get())]]
        q = joint.GetLocalRot1Attr().Get()
        axis = (
            np.asarray(Gf.Matrix3d(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())))).T
            @ axis
        )
        joint_path = str(prim.GetPath())
        link_path = str(children[0])
        part_id = (
            "part_"
            + hashlib.sha256(
                f"{source_hash}:{joint_path}:{link_path}".encode("utf-8")
            ).hexdigest()[:16]
        )
        candidates.append(
            ArticulationPartCandidate(
                part_id=part_id,
                joint_path=joint_path,
                joint=prim.GetName(),
                parent_path=str(parents[0]),
                parent=parent.GetName(),
                link_path=link_path,
                link=body.GetName(),
                handle_paths=handles,
                axis=tuple(axis),
                limits=tuple(
                    float(v) * scale[0]
                    for v in (
                        joint.GetLowerLimitAttr().Get(),
                        joint.GetUpperLimitAttr().Get(),
                    )
                ),
            )
        )
    if not candidates:
        raise ValueError("GenSim E6 requires at least one enabled prismatic joint.")
    source.assert_current()
    return tuple(sorted(candidates, key=lambda item: item.part_id))


def _binding_from_candidate(
    config: dict[str, Any],
    candidate: ArticulationPartCandidate,
    *,
    include_part_id: bool = True,
    source: UsdSourceSnapshot | None = None,
) -> PrismaticBinding:
    if len(candidate.handle_paths) != 1:
        raise ValueError(
            f"GenSim E6 part {candidate.part_id!r} requires one unambiguous handle."
        )
    scale = float(np.asarray(config.get("body_scale", (1.0, 1.0, 1.0)))[0])
    from pxr import UsdGeom

    source = read_usd_source(config["fpath"] if source is None else source)
    source.assert_current(config["fpath"])
    source.require_geometry()
    stage = source.stage
    handle_body = stage.GetPrimAtPath(source.rigid_owner(candidate.handle_paths[0]))
    handle_axis = np.asarray(candidate.axis)
    if str(handle_body.GetPath()) != candidate.link_path:
        cache = UsdGeom.XformCache()
        moving_pose = np.asarray(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(candidate.link_path))
        ).T
        handle_pose = np.asarray(cache.GetLocalToWorldTransform(handle_body)).T
        handle_axis = np.linalg.solve(
            handle_pose[:3, :3], moving_pose[:3, :3] @ candidate.axis
        )
        handle_axis /= np.linalg.norm(handle_axis)
    closed_attr = stage.GetPrimAtPath(candidate.joint_path).GetAttribute(
        "gen_sim:closedPosition"
    )
    closed = closed_attr.Get() if closed_attr else None
    if closed is not None and (
        isinstance(closed, bool) or not isinstance(closed, Real)
    ):
        raise ValueError("gen_sim:closedPosition must be a real coordinate.")
    binding = PrismaticBinding(
        object_id=str(config["uid"]),
        joint=candidate.joint,
        link=handle_body.GetName(),
        parent=candidate.parent,
        handle_path=candidate.handle_paths[0],
        source_sha256=source.source_sha256,
        scale=scale,
        axis=tuple(handle_axis),
        limits=candidate.limits,
        part_id=candidate.part_id if include_part_id else "",
        closed_position=None if closed is None else float(closed) * scale,
    )
    handle_mesh(binding, source)
    return binding


def inspect_prismatic(
    config: dict[str, Any], *, source: UsdSourceSnapshot | None = None
) -> PrismaticBinding:
    """Require one explicit mechanism; never select an arbitrary joint or mesh.

    Args:
        config: Asset declaration with exactly one supported mechanism.
        source: Optional shared static read matching the declared asset.

    Returns:
        A qualified binding to its explicit collision handle.
    """
    source = read_usd_source(config["fpath"] if source is None else source)
    parts = discover_prismatic_parts(config, source=source)
    if len(parts) != 1:
        raise ValueError("GenSim E6 requires exactly one enabled prismatic joint.")
    return _binding_from_candidate(
        config, parts[0], include_part_id=False, source=source
    )


def inspect_prismatic_part(
    config: dict[str, Any], part_id: str, *, source: UsdSourceSnapshot | None = None
) -> PrismaticBinding:
    """Resolve one catalog part into a source-qualified physical binding.

    Args:
        config: Existing physical asset declaration.
        part_id: Stable catalog part identity.
        source: Optional shared static read matching the declared asset.

    Returns:
        The selected binding; open/closed targets require declared endpoint semantics.
    """
    source = read_usd_source(config["fpath"] if source is None else source)
    matches = [
        part
        for part in discover_prismatic_parts(config, source=source)
        if part.part_id == part_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown GenSim E6 part {part_id!r}.")
    return _binding_from_candidate(config, matches[0], source=source)


def handle_mesh(
    binding: PrismaticBinding, path: str | Path | UsdSourceSnapshot
) -> tuple[np.ndarray, np.ndarray]:
    """Bake one collision mesh into its owning link frame, without scene edits.

    Args:
        binding: Previously inspected physical handle binding.
        path: Its selected USD asset source.

    Returns:
        Scaled link-local vertices and fan-triangulated face indices.
    """
    source = read_usd_source(path)
    if source.source_sha256 != binding.source_sha256:
        raise ValueError("Prismatic source differs from its inspected binding.")
    body = source.stage.GetPrimAtPath(source.rigid_owner(binding.handle_path))
    if body.GetName() != binding.link:
        raise ValueError("Prismatic handle owner differs from its inspected binding.")
    vertices, faces = source.link_mesh_arrays(
        binding.handle_path,
        str(body.GetPath()),
        scale=binding.scale,
        rigid_tolerance=1e-5,
    )
    source.assert_current()
    return vertices, faces
