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

"""Freeze L-03 common USD inputs without initializing a simulator.

Run with --asset-audit the local asset inventory, --g1-conversion conversion.json,
and --output a fresh directory. Use a Python environment providing pxr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

from pxr import Usd, UsdPhysics

__all__ = ["main"]


def main() -> None:
    """Copy audited dependencies and declare the common initial conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-audit", type=Path, required=True)
    parser.add_argument("--g1-conversion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = {a["name"]: a for a in json.loads(args.asset_audit.read_text())}
    converted = json.loads(args.g1_conversion.read_text())
    assert converted["status"] == "converted_static_audit_complete"
    sources = {
        "cartpole": inventory["cartpole"],
        "franka": inventory["franka_p03"],
        "go2": inventory["go2"],
        "g1": dict(path=converted["output_usd"], files=converted["files"]),
    }
    result = {"phase": "shared_input_preparation", "robots": {}}
    for name, entry in sources.items():
        original = Path(entry["path"])
        destination = args.output.resolve() / name
        destination.mkdir()
        copied = []
        external = []
        for item in entry["files"]:
            source = Path(item["path"])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]
            if not source.is_relative_to(original.parent):
                external.append(item)
                continue
            target = destination / source.relative_to(original.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append((target, item))
        usd_path = destination / original.name
        stage = Usd.Stage.Open(str(usd_path))
        edits = []
        if name == "g1":
            # The 5.1 importer leaves the wrapper without a defaultPrim.
            # Add reference-entry metadata to the shared copy, retaining raw output.
            before = stage.GetRootLayer().ExportToString()
            stage.SetDefaultPrim(stage.GetPrimAtPath("/g1_23dof_mode_10"))
            after = stage.GetRootLayer().ExportToString()
            assert after.replace('    defaultPrim = "g1_23dof_mode_10"\n', "") == before
            stage.GetRootLayer().Save()
            edits.append({"defaultPrim": "g1_23dof_mode_10", "physics_changed": False})
        assert stage.GetDefaultPrim()
        joints = {}
        bodies = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(prim.GetName())
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.Joint(prim)
                if joint.GetJointEnabledAttr().Get() is False:
                    continue
                scale = math.pi / 180 if prim.IsA(UsdPhysics.RevoluteJoint) else 1
                lo = float(prim.GetAttribute("physics:lowerLimit").Get()) * scale
                hi = float(prim.GetAttribute("physics:upperLimit").Get()) * scale
                joints[prim.GetName()] = max(lo, min(hi, 0.0))
        if name == "franka":
            joints.update(
                dict(
                    zip(
                        [f"panda_joint{i}" for i in range(1, 8)]
                        + ["panda_finger_joint1", "panda_finger_joint2"],
                        [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04],
                    )
                )
            )
        result["robots"][name] = {
            "usd_path": str(usd_path),
            "source": str(original),
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "expected_joint_names": sorted(joints),
            "source_body_names": bodies,
            "initial_qpos": joints,
            "fixed_base": name in ("cartpole", "franka"),
            "initial_position": [0.0, 0.0, 2.0],
            "env_spacing": 3.0,
            "preparation_edits": edits,
            "external_shader_dependencies": external,
            "variant_selections": {
                str(p.GetPath()): {
                    v: p.GetVariantSet(v).GetVariantSelection()
                    for v in p.GetVariantSets().GetNames()
                }
                for p in stage.Traverse()
                if p.GetVariantSets().GetNames()
            },
            "files": [
                {
                    "path": str(p),
                    "source": item["path"],
                    "source_sha256": item["sha256"],
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p, item in copied
            ],
        }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                n: {
                    "joints": len(r["expected_joint_names"]),
                    "bodies": len(r["source_body_names"]),
                    "usd": r["usd_path"],
                }
                for n, r in result["robots"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
