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

"""Audit L-03 G1 collision import from saved runs and an in-memory USD layer.

Run: python -m scripts.benchmark.tech_reports.reporting.l03_collision_review
    --root COLLIDER_AUDIT_ROOT --manifest SHARED_MANIFEST --output NEW_DIRECTORY
No simulator or GPU is initialized; the source USD files are never saved.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Sdf, Usd, UsdPhysics

__all__ = ["main"]


def main() -> None:
    """Compare actual counts and isolate the OpenUSD shape-index difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "analysis-source.py")
    cfg = json.loads(args.manifest.read_text())["robots"]["g1"]
    hashes = {}
    for entry in cfg["files"]:
        digest = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"]
        hashes[entry["path"]] = digest
    native = {}
    for platform in ("isaaclab", "embodichain"):
        run = args.root / "g1" / f"g1_0001_{platform}"
        meta = json.loads((run / "run.json").read_text())
        assert meta["status"] == "completed"
        native[platform] = {
            "run": str(run.resolve()),
            "packages": meta["packages"],
            "joint_count": len(meta["joint_names"]),
            "body_count": len(meta["link_names"]),
        }
        with np.load(run / "model.npz") as arrays:
            if platform == "embodichain":
                mask = arrays["native_shape_world"] == 0
                native[platform].update(
                    robot_collision_shapes=int(mask.sum()),
                    non_robot_collision_shapes=int((~mask).sum()),
                    robot_shape_types={
                        str(k): int(v)
                        for k, v in Counter(
                            arrays["native_shape_type"][mask].tolist()
                        ).items()
                    },
                    robot_shape_labels=[
                        name
                        for name, selected in zip(meta["native_shape_labels"], mask)
                        if selected
                    ],
                )
            else:
                native[platform]["robot_collision_shapes"] = meta[
                    "native_shapes_per_articulation"
                ]
        for name in (
            "run.json",
            "source.py",
            "input.json",
            "model.npz",
            "states.npz",
            "source-colliders.json",
            "initialization-trace.npz",
        ):
            path = run / name
            hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        log = run.parent / (run.name + ".log")
        hashes[str(log.resolve())] = hashlib.sha256(log.read_bytes()).hexdigest()
    stage = Usd.Stage.Open(cfg["usd_path"])
    stage.SetEditTarget(stage.GetSessionLayer())
    records = {}
    shape_types = {
        getattr(UsdPhysics.ObjectType, name)
        for name in (
            "SphereShape",
            "CubeShape",
            "CapsuleShape",
            "Capsule1Shape",
            "CylinderShape",
            "Cylinder1Shape",
            "ConeShape",
            "MeshShape",
            "PlaneShape",
            "CustomShape",
            "SpherePointsShape",
        )
    }
    for phase in ("original", "deinstanced_only", "explicit_mesh_collision_api"):
        parsed = UsdPhysics.UsdPhysicsLoadStageFromPrimRange(
            stage, [Sdf.Path.absoluteRootPath]
        )
        shapes = [
            {
                "type": str(kind),
                "path": str(path),
                "body": str(desc.rigidBody),
                "enabled": desc.collisionEnabled,
            }
            for kind, (paths, descs) in parsed.items()
            if kind in shape_types
            for path, desc in zip(paths, descs)
        ]
        records[phase] = {"shape_count": len(shapes), "shapes": shapes}
        if phase == "original":
            roots = []
            while instances := [
                p.GetPath() for p in stage.Traverse() if p.IsInstance()
            ]:
                for path in instances:
                    stage.GetPrimAtPath(path).SetInstanceable(False)
                    roots.append(str(path))
            records["deinstanced_paths"] = roots
        elif phase == "deinstanced_only":
            edits = []
            for prim in stage.Traverse():
                if prim.GetTypeName() != "Xform" or not prim.HasAPI(
                    UsdPhysics.CollisionAPI
                ):
                    continue
                meshes = [p for p in Usd.PrimRange(prim) if p.GetTypeName() == "Mesh"]
                assert len(meshes) == 1, str(prim.GetPath())
                mesh = meshes[0]
                UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                )
                approximation = (
                    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                )
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(
                    approximation
                )
                edits.append(
                    {
                        "parent": str(prim.GetPath()),
                        "child_mesh": str(mesh.GetPath()),
                        "approximation": approximation,
                    }
                )
            records["session_mesh_edits"] = edits
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "native": native,
        "openusd_index": records,
        "usd_core_version": importlib.metadata.version("usd-core"),
        "scope": "Source unchanged. Session-layer edits diagnose indexing only; no native simulation or performance claim for the edited stage.",
        "sha256": hashes,
    }
    (args.output / "collision-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "native_counts": {
                    k: v["robot_collision_shapes"] for k, v in native.items()
                },
                "index_counts": {
                    k: v["shape_count"]
                    for k, v in records.items()
                    if isinstance(v, dict)
                },
                "usd_core_version": result["usd_core_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
