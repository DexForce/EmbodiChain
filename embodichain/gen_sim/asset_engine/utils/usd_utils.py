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

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

DEFAULT_PHYSICS_PARAMS = {
    "mass": 1.0,
    "density": 1000.0,
    "static_friction": 0.5,
    "dynamic_friction": 0.5,
    "restitution": 0.0,
    "linear_damping": 0.7,
    "angular_damping": 0.7,
    "enable_collision": True,
    "enable_ccd": False,
    "contact_offset": 0.001,
    "rest_offset": 0.0,
    "max_linear_velocity": 100.0,
    "max_angular_velocity": 50.0,
    "max_depenetration_velocity": 100.0,
    "solver_min_position_iters": 4,
    "solver_min_velocity_iters": 1,
    "sleep_threshold": 0.001,
}


def parse_glb_with_trimesh(path: Path, texture_dir: Path) -> Dict[str, Any]:
    scene = trimesh.load(str(path))
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene

    tex_filename = "diffuse.png"
    tex_path = texture_dir / tex_filename

    material = mesh.visual.material
    if hasattr(material, "image") and material.image is not None:
        material.image.save(str(tex_path))
    elif (
        hasattr(material, "baseColorTexture") and material.baseColorTexture is not None
    ):
        material.baseColorTexture.save(str(tex_path))

    return {
        "vertices": np.asarray(mesh.vertices),
        "faces": np.asarray(mesh.faces),
        "uv": (
            np.asarray(mesh.visual.uv)
            if getattr(mesh.visual, "uv", None) is not None
            else None
        ),
        "tex_path": f"./textures/{tex_filename}",
    }


def build_clean_usd(
    data: Dict[str, Any], output_path: Path, physics_params: Dict[str, float]
) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")

    root_prim = UsdGeom.Xform.Define(stage, "/RootNode")
    stage.SetDefaultPrim(root_prim.GetPrim())

    stage.DefinePrim("/RootNode/Looks", "Scope")
    UsdGeom.Xform.Define(stage, "/RootNode/geometry_inst")

    new_mat_path = "/RootNode/Looks/Material_0"
    new_geo_path = "/RootNode/geometry_inst/geometry_0"

    # --- A. Mesh Definition ---
    mesh = UsdGeom.Mesh.Define(stage, new_geo_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in data["vertices"]]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(data["faces"].flatten().tolist()))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(data["faces"])))

    if data.get("uv") is not None:
        tex_coords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
        )
        tex_coords.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in data["uv"]]))

    mesh.CreateDoubleSidedAttr(True)

    # --- B. Material Definition ---
    material = UsdShade.Material.Define(stage, new_mat_path)
    pbr_shader = UsdShade.Shader.Define(stage, f"{new_mat_path}/PBRShader")
    pbr_shader.CreateIdAttr("UsdPreviewSurface")

    st_reader = UsdShade.Shader.Define(stage, f"{new_mat_path}/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex_sampler = UsdShade.Shader.Define(stage, f"{new_mat_path}/DiffuseSampler")
    tex_sampler.CreateIdAttr("UsdUVTexture")
    tex_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(data["tex_path"])
    tex_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )

    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex_sampler.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(
        pbr_shader.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    # --- C. Physics Material Injection ---
    binding_api = UsdShade.MaterialBindingAPI(mesh.GetPrim())
    bound_material, _ = binding_api.ComputeBoundMaterial()

    if bound_material:
        bound_prim = bound_material.GetPrim()
        UsdPhysics.MaterialAPI.Apply(bound_prim)
        material_api = UsdPhysics.MaterialAPI(bound_prim)
        material_api.CreateDensityAttr().Set(physics_params["density"])
        material_api.CreateRestitutionAttr().Set(physics_params["restitution"])
        material_api.CreateStaticFrictionAttr().Set(physics_params["static_friction"])
        material_api.CreateDynamicFrictionAttr().Set(physics_params["dynamic_friction"])

    # --- D. Core Rigid Body ---
    prim = mesh.GetPrim()

    prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(
            ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
        ),
    )

    prim.SetMetadata("kind", "component")

    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(physics_params["enable_collision"])

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.CreateApproximationAttr().Set(
        UsdPhysics.Tokens.convexDecomposition
    )

    def set_attr(name, type_name, value):
        attr = prim.CreateAttribute(name, type_name)
        attr.Set(value)

    set_attr("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, True)
    set_attr("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, False)
    set_attr("physics:startsAsleep", Sdf.ValueTypeNames.Bool, False)

    set_attr("physics:velocity", Sdf.ValueTypeNames.Vector3f, Gf.Vec3f(0, 0, 0))
    set_attr("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f, Gf.Vec3f(0, 0, 0))
    set_attr("physics:centerOfMass", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0, 0, 0))
    set_attr("physics:mass", Sdf.ValueTypeNames.Float, physics_params["mass"])

    def set_physx(name, type_name, value):
        attr = prim.CreateAttribute(f"physxRigidBody:{name}", type_name)
        attr.Set(value)

    set_physx(
        "linearDamping", Sdf.ValueTypeNames.Float, physics_params["linear_damping"]
    )
    set_physx(
        "angularDamping", Sdf.ValueTypeNames.Float, physics_params["angular_damping"]
    )

    set_physx(
        "maxLinearVelocity",
        Sdf.ValueTypeNames.Float,
        physics_params["max_linear_velocity"],
    )
    set_physx(
        "maxAngularVelocity",
        Sdf.ValueTypeNames.Float,
        physics_params["max_angular_velocity"],
    )
    set_physx(
        "maxDepenetrationVelocity",
        Sdf.ValueTypeNames.Float,
        physics_params["max_depenetration_velocity"],
    )

    set_physx("enableCCD", Sdf.ValueTypeNames.Bool, physics_params["enable_ccd"])
    set_physx("enableSpeculativeCCD", Sdf.ValueTypeNames.Bool, False)

    set_physx(
        "sleepThreshold", Sdf.ValueTypeNames.Float, physics_params["sleep_threshold"]
    )
    set_physx("stabilizationThreshold", Sdf.ValueTypeNames.Float, 0.001)

    set_physx(
        "solverPositionIterationCount",
        Sdf.ValueTypeNames.Int,
        physics_params["solver_min_position_iters"],
    )
    set_physx(
        "solverVelocityIterationCount",
        Sdf.ValueTypeNames.Int,
        physics_params["solver_min_velocity_iters"],
    )

    set_physx("lockedPosAxis", Sdf.ValueTypeNames.Int, 0)
    set_physx("lockedRotAxis", Sdf.ValueTypeNames.Int, 0)

    # --- E. Collision ---
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(physics_params["enable_collision"])

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.CreateApproximationAttr().Set(
        UsdPhysics.Tokens.convexDecomposition
    )

    # --- F. Extended ---
    prim.CreateAttribute("sim:linearDamping", Sdf.ValueTypeNames.Float).Set(
        float(physics_params["linear_damping"])
    )
    prim.CreateAttribute("sim:angularDamping", Sdf.ValueTypeNames.Float).Set(
        float(physics_params["angular_damping"])
    )
    prim.CreateAttribute("sim:contactOffset", Sdf.ValueTypeNames.Float).Set(
        float(physics_params["contact_offset"])
    )
    prim.CreateAttribute("sim:restOffset", Sdf.ValueTypeNames.Float).Set(
        float(physics_params["rest_offset"])
    )

    prim.CreateAttribute("physx:enableCCD", Sdf.ValueTypeNames.Bool).Set(
        physics_params["enable_ccd"]
    )
    prim.CreateAttribute("physx:maxLinearVelocity", Sdf.ValueTypeNames.Float).Set(
        physics_params["max_linear_velocity"]
    )
    prim.CreateAttribute("physx:maxAngularVelocity", Sdf.ValueTypeNames.Float).Set(
        physics_params["max_angular_velocity"]
    )
    prim.CreateAttribute(
        "physx:solverPositionIterationCount", Sdf.ValueTypeNames.Int
    ).Set(physics_params["solver_min_position_iters"])
    prim.CreateAttribute(
        "physx:solverVelocityIterationCount", Sdf.ValueTypeNames.Int
    ).Set(physics_params["solver_min_velocity_iters"])
    prim.CreateAttribute(
        "physx:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float
    ).Set(physics_params["max_depenetration_velocity"])
    prim.CreateAttribute("physx:sleepThreshold", Sdf.ValueTypeNames.Float).Set(
        physics_params["sleep_threshold"]
    )

    stage.GetRootLayer().Save()
    print(f"--- Exported base USD: {output_path} ---")


def convert_model_to_usd(
    input_path: Union[str, Path],
    out_dir: Union[str, Path] = "./output_usd",
    physics_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Path]:
    """
    Importable conversion entry point.

    Args:
        input_path: source .glb / mesh path
        out_dir: output directory
        physics_params: optional override of DEFAULT_PHYSICS_PARAMS

    Returns:
        dict with output paths
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(out_dir).resolve()
    base_name = input_path.stem

    final_params = DEFAULT_PHYSICS_PARAMS.copy()
    if physics_params:
        final_params.update(physics_params)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with tempfile.TemporaryDirectory() as temp_str:
        temp_dir = Path(temp_str)
        print(f"\n>>> Processing: {base_name}")

        temp_tex_dir = temp_dir / "textures"
        temp_tex_dir.mkdir(parents=True, exist_ok=True)

        temp_base_usd = temp_dir / f"{base_name}_inst_base.usda"
        temp_inst_usdc = temp_dir / f"{base_name}_inst.usdc"
        temp_usdz = temp_dir / f"{base_name}_inst.usdz"

        mesh_data = parse_glb_with_trimesh(input_path, temp_tex_dir)
        build_clean_usd(mesh_data, temp_base_usd, final_params)

        inst_stage = Usd.Stage.CreateNew(str(temp_inst_usdc))
        UsdGeom.SetStageUpAxis(inst_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(inst_stage, 1.0)

        inst_root = UsdGeom.Xform.Define(inst_stage, "/RootNode")
        inst_stage.SetDefaultPrim(inst_root.GetPrim())
        inst_root.GetPrim().GetReferences().AddReference(f"./{temp_base_usd.name}")
        inst_stage.GetRootLayer().Save()

        UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(str(temp_inst_usdc)), str(temp_usdz)
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(temp_base_usd, output_dir / temp_base_usd.name)
        shutil.copy2(temp_inst_usdc, output_dir / temp_inst_usdc.name)

        if temp_usdz.exists():
            shutil.copy2(temp_usdz, output_dir / temp_usdz.name)
        if temp_tex_dir.exists():
            shutil.copytree(temp_tex_dir, output_dir / "textures", dirs_exist_ok=True)

        print(f"\n>>> Pipeline completed successfully: {output_dir}")

        return {
            "output_dir": output_dir,
            "base_usd": output_dir / temp_base_usd.name,
            "inst_usdc": output_dir / temp_inst_usdc.name,
            "usdz": output_dir / temp_usdz.name,
            "textures_dir": output_dir / "textures",
        }


def load_physics_from_json(json_path: Optional[Path]) -> Optional[Dict[str, Any]]:

    if not json_path:
        return None

    if not json_path.exists():
        print(
            f"[Warning] JSON file not found: {json_path}, using default physics params."
        )
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        physics_data = json_data.get("physics", {}).get("properties", {}).get("data")

        if physics_data and isinstance(physics_data, dict):
            print(f"[Info] Successfully loaded physics params from JSON.")
            return physics_data
        else:
            print(
                f"[Warning] Invalid JSON structure: missing physics.properties.data, using default params."
            )
            return None

    except Exception as e:
        print(
            f"[Warning] Failed to parse JSON file: {str(e)}, using default physics params."
        )
        return None


def main():
    parser = argparse.ArgumentParser(
        description="3D Assets to USD/USDZ conversion pipeline with full physics support."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to the source .glb mesh file."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to the metadata JSON file (optional, for physics params).",
    )
    parser.add_argument(
        "--out_dir",
        default=Path("./output_usd"),
        type=Path,
        help="Target directory for final USD/USDZ assets.",
    )
    args = parser.parse_args()

    user_physics_params = load_physics_from_json(args.json)

    convert_model_to_usd(
        input_path=args.input, out_dir=args.out_dir, physics_params=user_physics_params
    )


if __name__ == "__main__":
    main()
