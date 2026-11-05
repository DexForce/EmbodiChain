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

"""Audit the L-04 drawer blocker without initializing either simulator.

Run: python -m scripts.benchmark.tech_reports.reporting.l04_asset_review
    --output outputs/benchmarks/L-04/analysis/NEW_DIRECTORY
Session-layer edits are diagnostic only; source USD files are never saved.
"""

from __future__ import annotations

import os

import argparse
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
    """Preserve model counts, indexing evidence and every raw input/output hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[4] / "outputs/benchmarks/L-04"
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "analysis-source.py")
    manifest = json.loads((base / "prepared/drawer-probe-manifest.json").read_text())
    cfg = manifest["robots"]["drawer"]
    for item in cfg["files"]:
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    raw = sorted(
        p
        for directory in (base / "prepared", base / "drawer-preflight-v1")
        for p in directory.rglob("*")
        if p.is_file()
    )
    hashes = {
        str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in raw
    }
    native = []
    for n in (1, 4):
        for platform in ("embodichain", "isaaclab"):
            run = base / "drawer-preflight-v1" / f"drawer_{n:04d}_{platform}"
            meta = json.loads((run / "run.json").read_text())
            assert meta["status"] == "completed"
            with (
                np.load(run / "model.npz") as model,
                np.load(run / "states.npz") as states,
            ):
                counts = (
                    [int((model["native_shape_world"] == i).sum()) for i in range(n)]
                    if platform == "embodichain"
                    else [meta["native_shapes_per_articulation"]] * n
                )
                assert states["first_qpos"].shape == (n, 1)
                assert all(np.isfinite(states[k]).all() for k in states.files)
                native.append(
                    dict(
                        platform=platform,
                        num_envs=n,
                        run=str(run.relative_to(base)),
                        collisions_per_env=counts,
                        joint_names=meta["joint_names"],
                        link_names=meta["link_names"],
                        mass_per_env_kg=meta["mass_per_env_kg"],
                        packages=meta["packages"],
                        performance_eligible=False,
                    )
                )
    stage = Usd.Stage.Open(cfg["usd_path"])
    stage.SetEditTarget(stage.GetSessionLayer())
    index = {}
    edits = []
    deinstanced_paths = []
    for phase in (
        "original",
        "deinstanced_only",
        "explicit_child_mesh_api_session_only",
    ):
        parsed = UsdPhysics.UsdPhysicsLoadStageFromPrimRange(
            stage, [Sdf.Path.absoluteRootPath]
        )
        index[phase] = [
            dict(type=str(kind), path=str(path), rigid_body=str(desc.rigidBody))
            for kind, (paths, descs) in parsed.items()
            if "Shape" in str(kind)
            for path, desc in zip(paths, descs)
        ]
        if phase == "original":
            while instances := [
                p.GetPath() for p in stage.Traverse() if p.IsInstance()
            ]:
                for path in instances:
                    stage.GetPrimAtPath(path).SetInstanceable(False)
                    deinstanced_paths.append(str(path))
        elif phase == "deinstanced_only":
            for prim in stage.Traverse():
                if prim.GetTypeName() != "Xform" or not prim.HasAPI(
                    UsdPhysics.CollisionAPI
                ):
                    continue
                meshes = [p for p in Usd.PrimRange(prim) if p.GetTypeName() == "Mesh"]
                assert len(meshes) == 1
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
                    dict(
                        parent=str(prim.GetPath()),
                        mesh=str(mesh.GetPath()),
                        approximation=approximation,
                    )
                )
    for relative, digest in hashes.items():
        assert hashlib.sha256((base / relative).read_bytes()).hexdigest() == digest
    result = dict(
        experiment="L-04",
        created_at=datetime.now().astimezone().isoformat(),
        status="asset_import_blocked",
        native=native,
        openusd_index=index,
        session_edits=edits,
        deinstanced_paths=deinstanced_paths,
        usd_core_version=importlib.metadata.version("usd-core"),
        scope="Diagnostic only. No solver/source USD edits, no performance samples. Reuses the frozen L-03 articulation probe; its raw experiment field retains L-03.",
        sha256=hashes,
        raw_files_unchanged=True,
    )
    (args.output / "collision-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "native": native,
                "index_counts": {k: len(v) for k, v in index.items()},
                "raw_files_verified": len(hashes),
            },
            indent=2,
        )
    )
    assert len(index["original"]) == len(index["deinstanced_only"]) == 0
    assert len(index["explicit_child_mesh_api_session_only"]) == 2


if __name__ == "__main__":
    main()
