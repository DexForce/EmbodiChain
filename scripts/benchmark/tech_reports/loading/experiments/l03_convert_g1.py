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

"""Prepare the shared L-03 G1 USD with the standalone Isaac Sim 5.1 importer.

Run with Isaac Sim's python.sh, --input URDF and --output a new directory.
This offline conversion does not execute the loading benchmark or physics steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

__all__ = ["main"]


def main() -> None:
    """Convert once and retain parameters, source identity and USD audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.input.resolve()
    xml = ET.parse(source).getroot()
    output = args.output.resolve() / "g1_23dof_mode_10.usd"
    record = {
        "phase": "offline_asset_conversion",
        "status": "running",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": sys.executable,
        "source_joint_names": sorted(
            j.attrib["name"]
            for j in xml.findall("joint")
            if j.attrib["type"] != "fixed"
        ),
        "source_total_mass_kg": sum(
            float(m.attrib["value"]) for m in xml.findall("link/inertial/mass")
        ),
        "output_usd": str(output),
        "physics_steps": 0,
    }
    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "active_gpu": 0, "multi_gpu": False})
        import omni.kit.commands
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

        enable_extension("isaacsim.asset.importer.urdf")
        from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

        ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
        assert ok
        parameters = {
            "merge_fixed_joints": True,
            "fix_base": False,
            "import_inertia_tensor": True,
            "distance_scale": 1.0,
            "density": 0.0,
            "self_collision": False,
            "convex_decomp": False,
            "collision_from_visuals": False,
            "replace_cylinders_with_capsules": False,
            "create_physics_scene": False,
            "default_drive_strength": 0.0,
            "default_position_drive_damping": 0.0,
            "default_drive_type": UrdfJointTargetType.JOINT_DRIVE_NONE,
        }
        for key, value in parameters.items():
            setattr(config, key, value)
        record["import_config"] = {key: getattr(config, key) for key in parameters}
        (args.output / "conversion.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n"
        )
        ok, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(source),
            import_config=config,
            dest_path=str(output),
            get_articulation_root=True,
        )
        assert ok and output.is_file(), (ok, prim_path)
        stage = Usd.Stage.Open(str(output))
        joints = []
        bodies = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mass = UsdPhysics.MassAPI(prim)
                bodies.append(
                    {
                        "path": str(prim.GetPath()),
                        "mass": mass.GetMassAttr().Get(),
                        "diagonal_inertia": mass.GetDiagonalInertiaAttr().Get(),
                        "principal_axes": mass.GetPrincipalAxesAttr().Get(),
                    }
                )
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
                joints.append(prim.GetName())
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(output))
        record.update(
            articulation_root=prim_path,
            default_prim=str(stage.GetDefaultPrim().GetPath()),
            joint_names=sorted(joints),
            bodies=bodies,
            usd_total_mass_kg=sum(b["mass"] for b in bodies),
            meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
            up_axis=str(UsdGeom.GetStageUpAxis(stage)),
            unresolved=unresolved,
            files=[
                {
                    "path": str(p),
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in sorted(
                    {Path(layer.realPath).resolve() for layer in layers}
                    | {Path(asset).resolve() for asset in assets}
                )
            ],
        )
        assert record["joint_names"] == record["source_joint_names"]
        assert len(joints) == 23, joints
        assert set(unresolved) <= {"OmniPBR.mdl"}, unresolved
        record["status"] = "converted_static_audit_complete"
        print(json.dumps(record, indent=2, default=str), flush=True)
    except BaseException:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "conversion.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n"
        )
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
