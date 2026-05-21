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

from __future__ import annotations

import uuid
import trimesh
import json
from pathlib import Path
from typing import Union, Dict, Any
from embodichain.toolkits.simready_pipeline.utils.texture_utils import classify_visual
import hashlib
import os
from embodichain.toolkits.simready_pipeline.core.asset import Asset


def new_uuid() -> str:
    return uuid.uuid4().hex


def compute_folder_sha256(folder_path: Union[str, Path]) -> str:

    folder_path = Path(folder_path).resolve()

    if not folder_path.is_dir():
        raise ValueError(f"Path {folder_path} is not a valid directory.")

    sha256_hash = hashlib.sha256()

    all_files = []
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        files.sort()
        for file_name in files:
            file_path = Path(root) / file_name
            relative_path = file_path.relative_to(folder_path)
            all_files.append(relative_path)

    for rel_path in sorted(all_files):
        full_path = folder_path / rel_path
        sha256_hash.update(str(rel_path).encode("utf-8"))
        with open(full_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def inject_semantic_from_config(asset_source: Path, asset: Asset) -> None:

    config_path = asset_source / "config.json"

    if not config_path.exists():
        print(f"[INFO] No config.json found at {config_path}")
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read config.json: {e}")
        return

    semantic = config.get("semantic")
    if not semantic:
        print("[INFO] No semantic field in config.json")
        return

    asset.semantics.setdefault("tags", [])
    asset.semantics.setdefault("description", None)

    if "tags" in semantic and isinstance(semantic["tags"], list):
        existing_tags = set(asset.semantics.get("tags", []))
        new_tags = set(semantic["tags"])
        asset.semantics["tags"] = list(existing_tags | new_tags)

    if "description" in semantic and semantic["description"]:
        if not asset.semantics.get("description"):
            asset.semantics["description"] = semantic["description"]

    print(f"[INFO] Injected semantic from {config_path}")


def inject_user_extra_info(asset_source: Path, asset: Asset) -> None:

    config_path = asset_source / "config.json"
    asset.ingest_info.setdefault("extra_info", {})
    if not config_path.exists():
        print(f"[INFO] No config.json found at {config_path}")
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read config.json: {e}")
        return

    extra_info = config.get("extra_info")
    if not extra_info:
        print("[INFO] No extra_info field in config.json")
        return

    asset.ingest_info["extra_info"].update(extra_info)

    print(f"[INFO] Injected extra_info from {config_path}")


def load_one_trimesh(
    path: str,
    scene_mesh_strategy: str = "first",
) -> Union[
    trimesh.Trimesh, None
]:  # The input may be a scene; process only the first geometry unless configured to concatenate.
    try:
        mesh_or_scene = trimesh.load_mesh(path)
        if isinstance(mesh_or_scene, trimesh.Scene):
            if len(mesh_or_scene.geometry) == 0:
                print(f"No geometry found in Scene: {path}")
                return None
            if scene_mesh_strategy == "concatenate":
                meshes = list(mesh_or_scene.geometry.values())
                return trimesh.util.concatenate(meshes)
            first_mesh = list(mesh_or_scene.geometry.values())[0]
            return first_mesh
        if isinstance(mesh_or_scene, trimesh.Trimesh):
            return mesh_or_scene
        print(f"Unexpected type: {type(mesh_or_scene)}")
        return None

    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def trimesh_parse_ingest(
    source_file: Path,
    asset_source: Path,
    obj_name: str = "asset.obj",
    mtl_name: str = "asset.mtl",
    write_files: bool = True,
    config: Dict[str, Any] | None = None,
):
    config = config or {}
    visual_config = config.get("visual", {})
    export_config = config.get("export", {})
    scene_mesh_strategy = config.get("scene_mesh_strategy", "first")
    mtl_name = config.get("mtl_name", mtl_name)

    mesh = load_one_trimesh(source_file, scene_mesh_strategy=scene_mesh_strategy)
    if mesh is None:
        return None

    texture_info = classify_visual(mesh)
    visual_category = texture_info.get("visual_category")
    material_kind = texture_info.get("material_kind")
    textures = texture_info.get("material", {}).get("textures", {})
    uv_present = texture_info.get("uv_present")

    visual = {
        "visual_category": visual_category,
        "uv_present": uv_present,
        "texture_count_total": texture_info.get("texture_count_total"),
        "material_kind": material_kind,
        "textures": textures,
    }
    visual_ingest = None
    asset_source = Path(asset_source)
    asset_source.mkdir(parents=True, exist_ok=True)
    obj_path = asset_source / obj_name

    # ========= CASE 1: no visual =========
    if visual_category == "None":
        print("[INFO] No visual → assign default gray")

        mesh.visual = trimesh.visual.ColorVisuals(
            mesh,
            face_colors=visual_config.get(
                "default_face_color", [128, 128, 128, 255]
            ),
        )
        visual_ingest = "no visual"

    # ========= CASE 2: color =========
    elif visual_category in ["color_face", "color_vertex"]:
        print("[INFO] Vertex/Face color → export directly")
        visual_ingest = "Color Visual"

    # ========= CASE 3: texture =========
    elif visual_category == "texture":

        vis = mesh.visual

        if not uv_present:
            visual_ingest = "no UV! But detected as Visual.Texture"
            print("[WARN] texture but no UV → export raw")

        else:
            # ---------- PBR ----------
            if material_kind == "pbr" and visual_config.get(
                "pbr_base_color_only", True
            ):
                print("[WARN] PBR → only baseColorTexture will be used")

                base_tex = textures.get("baseColorTexture", {})

                if base_tex.get("present"):
                    base_img = vis.material.baseColorTexture

                    simple_mat = trimesh.visual.material.SimpleMaterial(image=base_img)

                    mesh.visual = trimesh.visual.texture.TextureVisuals(
                        uv=vis.uv, image=base_img, material=simple_mat
                    )
                    visual_ingest = "Basecolor Texture from PBR as Visual"
                else:
                    print("[WARN] No baseColorTexture → fallback raw")

            # ---------- Simple ----------
            else:
                visual_ingest = "Simple Texture"
                print("[INFO] Simple texture → use directly")

    else:
        print("[WARN] Unknown visual type → export raw")

    if write_files:
        obj_str, tex_dict = trimesh.exchange.obj.export_obj(
            mesh,
            include_normals=export_config.get("include_normals", True),
            include_color=export_config.get("include_color", True),
            include_texture=export_config.get("include_texture", True),
            return_texture=True,
            write_texture=export_config.get("write_texture", False),
            mtl_name=mtl_name,
        )

        # ===== Write OBJ =====
        with open(obj_path, "w") as f:
            f.write(obj_str)

        # ===== Write texture / MTL =====
        for name, data in tex_dict.items():
            file_path = asset_source / name

            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(data)

    return {"visual_ingest": visual_ingest, "visual_source": visual}


import bpy


def modify_mtl_file(mtl_path: Path, diffuse_name: str, normal_name: str) -> None:
    """Modify an exported OBJ .mtl to reference baked textures."""
    mtl_path = Path(mtl_path)
    if not mtl_path.exists():
        return

    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    new_lines = []
    for line in lines:
        if line.startswith("Ns "):
            new_lines.append("Ns 500.000000\n")
        elif line.startswith("Ka "):
            new_lines.append("Ka 1.000000 1.000000 1.000000\n")
        elif line.startswith("Ks "):
            new_lines.append("Ks 0.500000 0.500000 0.500000\n")
        else:
            new_lines.append(line)

    new_lines.append(f"map_Kd {diffuse_name}\n")
    new_lines.append(f"map_Bump {normal_name}\n")
    new_lines.append(f"bump {normal_name} -bm 1.0\n")

    mtl_path.write_text("".join(new_lines), encoding="utf-8")


def blender_remesh_bake(
    source_file: Path,
    asset_source: Path,
    texture_size: int | None = None,
    png_name: str | None = None,
    voxel_size: float | None = None,
    decimate_ratio: float | None = None,
    obj_name: str = "asset.obj",
    config: Dict[str, Any] | None = None,
):
    """Remesh a high-poly mesh into a low-poly one and bake textures via Blender."""
    config = config or {}
    remesh_config = config.get("remesh", {})
    decimate_config = config.get("decimate", {})
    uv_config = config.get("uv", {})
    bake_config = config.get("bake", {})
    material_config = config.get("material", {})

    texture_size = int(
        texture_size
        or bake_config.get("texture_size", config.get("texture_size", 2048))
    )
    diffuse_texture_name = png_name or bake_config.get(
        "diffuse_texture_name", config.get("texture_name", "diffuse.png")
    )
    normal_texture_name = bake_config.get(
        "normal_texture_name", config.get("normal_texture_name", "normal.png")
    )
    voxel_size = float(
        voxel_size
        if voxel_size is not None
        else remesh_config.get("voxel_size", config.get("voxel_size", 0.01))
    )
    min_voxel_size_ratio = float(remesh_config.get("min_voxel_size_ratio", 0.005))
    use_smooth_shade = bool(remesh_config.get("use_smooth_shade", True))
    decimate_ratio = float(
        decimate_ratio
        if decimate_ratio is not None
        else decimate_config.get("ratio", config.get("decimate_ratio", 0.5))
    )
    angle_limit = float(uv_config.get("angle_limit", 66.0))
    island_margin = float(uv_config.get("island_margin", 0.02))
    cage_extrusion_ratio = float(bake_config.get("cage_extrusion_ratio", 0.05))
    material_name = material_config.get("name", "BakeMat")

    asset_source = Path(asset_source)
    asset_source.mkdir(parents=True, exist_ok=True)
    source_file = Path(source_file)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = source_file.suffix.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source_file))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source_file))
    elif ext in [".gltf", ".glb"]:
        bpy.ops.import_scene.gltf(filepath=str(source_file))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(source_file))
    else:
        raise RuntimeError(f"Unsupported extension: {ext}")

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    imported_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not imported_meshes:
        raise RuntimeError("No mesh object after import")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_meshes[0]

    if len(imported_meshes) > 1:
        bpy.ops.object.join()
    high_poly = bpy.context.view_layer.objects.active
    if not high_poly or high_poly.type != "MESH":
        raise RuntimeError("No active mesh object after import")
    high_poly.name = "High_Poly"

    auto_extrusion = max(high_poly.dimensions) * cage_extrusion_ratio

    bpy.ops.object.select_all(action="DESELECT")
    high_poly.select_set(True)
    bpy.context.view_layer.objects.active = high_poly
    bpy.ops.object.duplicate()
    low_poly = bpy.context.active_object
    if not low_poly:
        raise RuntimeError("Failed to duplicate object")
    low_poly.name = "Low_Poly_Target"
    try:
        low_poly.data.materials.clear()
    except Exception:
        pass

    rem = low_poly.modifiers.new(name="Remesh", type="REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = max(
        float(voxel_size), max(high_poly.dimensions) * min_voxel_size_ratio
    )
    rem.use_smooth_shade = use_smooth_shade
    bpy.ops.object.modifier_apply(modifier="Remesh")

    dec = low_poly.modifiers.new(name="Decimate", type="DECIMATE")
    dec.ratio = float(decimate_ratio)
    bpy.ops.object.modifier_apply(modifier="Decimate")

    bpy.context.view_layer.objects.active = low_poly
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
    bpy.ops.object.mode_set(mode="OBJECT")

    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    low_poly.data.materials.append(mat)
    nodes = mat.node_tree.nodes
    nodes.clear()

    def setup_node(name: str, is_color: bool):
        img = bpy.data.images.new(
            name, width=int(texture_size), height=int(texture_size)
        )
        node = nodes.new("ShaderNodeTexImage")
        node.image = img
        if not is_color:
            img.colorspace_settings.name = "Non-Color"
        return node, img

    diff_node, diff_img = setup_node(diffuse_texture_name, True)
    norm_node, norm_img = setup_node(normal_texture_name, False)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = auto_extrusion

    bpy.ops.object.select_all(action="DESELECT")
    high_poly.select_set(True)
    low_poly.select_set(True)
    bpy.context.view_layer.objects.active = low_poly

    nodes.active = diff_node
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    diff_img.filepath_raw = str(asset_source / diffuse_texture_name)
    diff_img.save()

    nodes.active = norm_node
    bpy.ops.object.bake(type="NORMAL")
    norm_img.filepath_raw = str(asset_source / normal_texture_name)
    norm_img.save()

    export_path = asset_source / obj_name
    bpy.ops.object.select_all(action="DESELECT")
    low_poly.select_set(True)
    bpy.ops.wm.obj_export(filepath=str(export_path), export_selected_objects=True)

    mtl_path = asset_source / Path(obj_name).with_suffix(".mtl").name
    modify_mtl_file(mtl_path, diffuse_texture_name, normal_texture_name)

    return {
        "png": str(asset_source / diffuse_texture_name),
        "obj": str(export_path),
        "mtl": str(mtl_path.name),
    }


def blender_parse_ingest(
    source_file: Path,
    asset_source: Path,
    trimesh_config: Dict[str, Any] | None = None,
    **kwargs,
):
    res = blender_remesh_bake(
        source_file=source_file,
        asset_source=asset_source,
        **kwargs,
    )
    try:
        asset_obj = Path(res["obj"])
        vis = trimesh_parse_ingest(
            asset_obj,
            asset_source,
            write_files=False,
            config=trimesh_config,
        )
        if isinstance(vis, dict):
            res.update(vis)
    except Exception:
        pass
    return res


def blender_parser_ingest(source_file: Path, asset_source: Path, **kwargs):
    return blender_parse_ingest(source_file, asset_source, **kwargs)
