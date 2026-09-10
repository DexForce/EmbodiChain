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
from typing import Any

import numpy as np

__all__: list[str] = []

SLIDE_CALL = "gen_sim.articulation_slide"
WITHDRAW_CALL = "gen_sim.articulation_withdraw"


@dataclass(frozen=True, slots=True)
class PrismaticBinding:
    object_id: str
    joint: str
    link: str
    parent: str
    handle_path: str
    source_sha256: str
    scale: float
    axis: tuple[float, float, float]
    limits: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("object_id", "joint", "link", "parent", "handle_path"):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"Prismatic {name} must be an exact identifier.")
        if self.link == self.parent or not self.handle_path.startswith("/"):
            raise ValueError(
                "Prismatic binding requires distinct bodies and an absolute handle path."
            )
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
            or min(map(abs, self.limits)) > 1e-6
        ):
            raise ValueError(
                "Prismatic limits require a zero closed endpoint and a finite open endpoint."
            )
        if self.limits[1] - self.limits[0] <= 0.004:
            raise ValueError(
                "Prismatic endpoints must be distinguishable at millimetre resolution."
            )
        object.__setattr__(self, "axis", tuple(map(float, self.axis)))
        object.__setattr__(self, "limits", tuple(map(float, self.limits)))

    @property
    def link_id(self) -> str:
        return f"{self.object_id}__{self.link}"

    @property
    def axis_sign(self) -> int:
        return -1 if self.target("open") > self.target("closed") else 1

    def target(self, state: str) -> float:
        if state not in {"open", "closed"}:
            raise ValueError("Prismatic target state must be open or closed.")
        return (max if state == "open" else min)(self.limits, key=abs)

    def tolerance(self, state: str) -> float:
        self.target(state)
        return min(
            0.03 if state == "open" else 0.005, (self.limits[1] - self.limits[0]) / 4
        )

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def decode(cls, value: object) -> PrismaticBinding:
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Prismatic binding requires exactly its declared fields.")
        return cls(**value)


def _stage(path: str | Path) -> Any:
    from pxr import Usd, UsdGeom

    if Path(path).suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(
            "GenSim E6 currently requires a self-contained USD articulation."
        )
    stage = Usd.Stage.Open(str(path))
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("GenSim E6 requires metre-authored USD geometry.")
    if any(
        not layer.anonymous and layer != stage.GetRootLayer()
        for layer in stage.GetUsedLayers()
    ):
        raise ValueError("GenSim E6 requires a self-contained USD layer.")
    return stage


def inspect_prismatic(config: dict[str, Any]) -> PrismaticBinding:
    """Require one explicit mechanism; never select an arbitrary joint or mesh."""
    from pxr import Gf, UsdGeom, UsdPhysics

    if config.get("fix_base") is not True:
        raise ValueError("GenSim E6 requires a declared fixed base.")
    scale = np.asarray(config.get("body_scale", (1.0, 1.0, 1.0)), dtype=float)
    if (
        scale.shape != (3,)
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.allclose(scale, scale[0], atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("GenSim E6 requires positive uniform body_scale.")
    stage = _stage(config["fpath"])
    joints = [
        p
        for p in stage.Traverse()
        if p.IsA(UsdPhysics.Joint)
        and not p.IsA(UsdPhysics.FixedJoint)
        and UsdPhysics.Joint(p).GetJointEnabledAttr().Get() is not False
    ]
    if len(joints) != 1 or not joints[0].IsA(UsdPhysics.PrismaticJoint):
        raise ValueError("GenSim E6 requires exactly one enabled prismatic joint.")
    joint = UsdPhysics.PrismaticJoint(joints[0])
    parents, children = (
        joint.GetBody0Rel().GetTargets(),
        joint.GetBody1Rel().GetTargets(),
    )
    if len(parents) != 1 or len(children) != 1:
        raise ValueError("GenSim E6 requires an explicit parent and moving body.")
    parent, body = stage.GetPrimAtPath(parents[0]), stage.GetPrimAtPath(children[0])
    if not parent.HasAPI(UsdPhysics.RigidBodyAPI) or not body.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        raise ValueError("GenSim E6 joint bodies must be rigid bodies.")
    for selected in (parent, body, joints[0]):
        if sum(p.GetName() == selected.GetName() for p in stage.Traverse()) != 1:
            raise ValueError(
                "GenSim E6 native body and joint names must be unambiguous."
            )
    handles = [
        p
        for p in stage.Traverse()
        if p.GetPath().HasPrefix(children[0])
        and p.IsA(UsdGeom.Mesh)
        and p.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
        and any(token in p.GetName().lower() for token in ("handle", "grip", "pull"))
        and not any(
            token in p.GetName().lower()
            for token in ("mount", "support", "post", "leg")
        )
    ]
    if len(handles) != 1:
        raise ValueError("GenSim E6 requires one unambiguous handle collision mesh.")
    axis = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[str(joint.GetAxisAttr().Get())]]
    q = joint.GetLocalRot1Attr().Get()
    axis = (
        np.asarray(Gf.Matrix3d(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())))).T
        @ axis
    )
    binding = PrismaticBinding(
        object_id=str(config["uid"]),
        joint=joints[0].GetName(),
        link=body.GetName(),
        parent=parent.GetName(),
        handle_path=str(handles[0].GetPath()),
        source_sha256=hashlib.sha256(Path(config["fpath"]).read_bytes()).hexdigest(),
        scale=float(scale[0]),
        axis=tuple(axis),
        limits=tuple(
            float(v) * scale[0]
            for v in (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
        ),
    )
    handle_mesh(binding, config["fpath"])
    return binding


def handle_mesh(
    binding: PrismaticBinding, path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Bake one collision mesh into its owning link frame, without scene edits."""
    from pxr import UsdGeom

    stage = _stage(path)
    prim = stage.GetPrimAtPath(binding.handle_path)
    body = next(p for p in stage.Traverse() if p.GetName() == binding.link)
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or not len(indices)
        or (counts < 3).any()
        or counts.sum() != len(indices)
        or indices.min() < 0
        or indices.max() >= len(vertices)
    ):
        raise ValueError("GenSim E6 handle has invalid mesh topology.")
    cache = UsdGeom.XformCache()
    body_pose = np.asarray(cache.GetLocalToWorldTransform(body)).T
    if not np.allclose(body_pose[:3, :3].T @ body_pose[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("GenSim E6 requires unscaled USD rigid-body frames.")
    transform = (
        np.linalg.inv(body_pose) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
    )
    vertices = (vertices @ transform[:3, :3].T + transform[:3, 3]) * binding.scale
    faces, offset = [], 0
    for count in counts:
        polygon = indices[offset : offset + count]
        faces.extend(
            (polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1)
        )
        offset += count
    return vertices, np.asarray(faces)


def recipe(object_id: str, state: str, arm: str) -> list[dict[str, Any]]:
    if state not in {"open", "closed"} or arm not in {"left", "right"}:
        raise ValueError("GenSim E6 requires an open/closed target and one arm.")
    return [
        {
            "kind": "registered",
            "call_id": call_id,
            "arguments": {"object": object_id, "state": state},
            "resources": {"primary": arm},
        }
        for call_id in (SLIDE_CALL, WITHDRAW_CALL)
    ] + [
        {
            "kind": "registered",
            "call_id": "simulation.park",
            "arguments": {},
            "resources": {"primary": arm},
        }
    ]


def graph_bindings(graph: dict, scene: Any) -> dict[str, PrismaticBinding]:
    """Validate complete E6 recipes before copying any executable bundle assets."""
    if not any(group["task_type"] == "E6" for group in graph["task_groups"]):
        return {}
    nodes = {node["id"]: node for node in graph["nodes"]}
    configs = {} if scene is None else {cfg["uid"]: cfg for cfg in scene.articulations}
    result = {}
    for group in graph["task_groups"]:
        if group["task_type"] != "E6":
            continue
        sequence = [nodes[key] for key in group["node_ids"]]
        if len(sequence) != 3:
            raise ValueError("E6 requires the complete Slide/withdraw/Park recipe.")
        call = sequence[0]["call"]
        args = call.get("arguments", {})
        if set(args) != {"object", "state"}:
            raise ValueError("E6 requires an exact object and state declaration.")
        arm = call.get("resources", {}).get("primary")
        if [node["call"] for node in sequence] != recipe(
            args["object"], args["state"], arm
        ):
            raise ValueError("E6 requires the complete Slide/withdraw/Park recipe.")
        if [node["role"] for node in sequence] != ["primary", "cleanup", "cleanup"]:
            raise ValueError("E6 interaction and cleanup roles must be explicit.")
        if any(
            current["depends_on"] != [previous["id"]]
            for previous, current in zip(sequence, sequence[1:])
        ):
            raise ValueError(
                "E6 withdrawal and Park must follow their preceding action."
            )
        uid = args["object"]
        if uid not in configs:
            raise ValueError(
                f"E6 articulation {uid!r} is absent from the prepared scene."
            )
        if uid not in result:
            result[uid] = inspect_prismatic(configs[uid])
            validate_placement(result[uid], configs[uid], scene.table_top_z)
    return result


def validate_placement(
    binding: PrismaticBinding, config: dict, table_top_z: float | None
) -> None:
    """Reject an unmeasured or table-embedded E6 before publishing executable assets."""
    from ..scene.articulation_geometry import read_articulation_geometry
    from scipy.spatial.transform import Rotation

    if table_top_z is None or not np.isfinite(table_top_z):
        raise ValueError("E6 placement requires a measured tabletop height.")
    geometry = read_articulation_geometry(config["fpath"])
    if geometry.root_link != binding.parent:
        raise ValueError("E6 geometry root differs from the declared fixed parent.")
    vertices = geometry.world_vertices(config)
    if vertices[:, 2].min() < table_top_z - 0.002:
        raise ValueError(
            f"E6 articulation {binding.object_id!r} intersects the tabletop: "
            f"bottom={vertices[:, 2].min():.6f}, table={table_top_z:.6f}. "
            "Repair the source scene scale/placement before execution."
        )
    handle, _ = handle_mesh(binding, config["fpath"])
    link_pose = geometry.link_poses[binding.link]
    handle = handle @ link_pose[:3, :3].T + link_pose[:3, 3] * binding.scale
    rotation = Rotation.from_euler(
        "XYZ", config.get("init_rot", [0, 0, 0]), degrees=True
    )
    handle = rotation.apply(handle) + np.asarray(config.get("init_pos", [0, 0, 0]))
    if handle[:, 2].min() <= table_top_z:
        raise ValueError("E6 handle is not clear of the tabletop.")
