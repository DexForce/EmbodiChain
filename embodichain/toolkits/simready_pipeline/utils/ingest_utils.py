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
) -> Union[
    trimesh.Trimesh, None
]:  # 可能是个scene，但是我们只处理scene中的第一个geometry，如果有多个mesh，复合起来需要下一个版本
    try:
        mesh_or_scene = trimesh.load_mesh(path)
        if isinstance(mesh_or_scene, trimesh.Scene):
            if len(mesh_or_scene.geometry) == 0:
                print(f"No geometry found in Scene: {path}")
                return None
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
):
    mesh = load_one_trimesh(source_file)
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
            mesh, face_colors=[128, 128, 128, 255]
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
            if material_kind == "pbr":
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

    obj_str, tex_dict = trimesh.exchange.obj.export_obj(
        mesh,
        include_normals=True,
        include_color=True,
        include_texture=True,
        return_texture=True,
        write_texture=False,
        mtl_name=mtl_name,
    )

    # ===== 写 OBJ =====
    with open(obj_path, "w") as f:
        f.write(obj_str)

    # ===== 写 texture / mtl =====
    for name, data in tex_dict.items():
        file_path = asset_source / name

        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(data)

    return {"visual_ingest": visual_ingest, "visual_source": visual}


import bpy


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)
    for block in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.collections,
    ):
        for item in list(block):
            try:
                block.remove(item)
            except:
                pass


def import_model(path: Path):
    ext = path.suffix.lower()

    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext in [".fbx"]:
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext in [".gltf", ".glb"]:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext in [".ply"]:
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise RuntimeError(f"Unsupported extension: {ext}")

    imported = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    return imported


def setup_studio_lighting():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    cycles.samples = 128
    cycles.use_adaptive_sampling = True

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    out = nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])


def duplicate_and_join(objs, name="BAKE_MESH"):
    if not objs:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.duplicate()
    dupes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    bpy.context.view_layer.objects.active = dupes[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name
    return joined


def ensure_uv(obj):
    me = obj.data
    if len(me.uv_layers) == 0:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")


def get_vertex_color_layer(obj):
    me = obj.data
    if hasattr(me, "color_attributes") and len(me.color_attributes) > 0:
        return me.color_attributes.active_color.name
    return None


def inject_vertex_color_to_material(mat, vcol_name):
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    pnode = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if not pnode:
        pnode = nodes.new(type="ShaderNodeBsdfPrincipled")

    attr = nodes.new(type="ShaderNodeAttribute")
    attr.attribute_name = vcol_name
    links.new(attr.outputs["Color"], pnode.inputs["Base Color"])


def add_bake_image_node(mat, image):
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes

    img_node = nodes.new(type="ShaderNodeTexImage")
    img_node.image = image
    img_node.name = "BAKE_TARGET"

    nodes.active = img_node
    img_node.select = True
    return img_node


def create_baked_material_assign(obj, image, mat_name="BAKED_SURFACE_MAT"):
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    img_node = nodes.new(type="ShaderNodeTexImage")
    img_node.image = image
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    out = nodes.new(type="ShaderNodeOutputMaterial")

    mat.node_tree.links.new(img_node.outputs["Color"], bsdf.inputs["Base Color"])
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return mat


# -------------------------
# Main bake routine
# -------------------------
def blender_parser_ingest(
    source_file: Path,
    asset_source: Path,
    texture_size=2048,
    png_name="surface_texture.png",
    obj_name="asset.obj",
):
    asset_source.mkdir(parents=True, exist_ok=True)
    png_path = asset_source / png_name

    clear_scene()
    imported = import_model(source_file)
    if not imported:
        raise RuntimeError("No mesh objects found after import.")

    setup_studio_lighting()
    joined = duplicate_and_join(imported, name="BAKE_MESH")

    bpy.context.view_layer.objects.active = joined
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    ensure_uv(joined)

    vcol_name = get_vertex_color_layer(joined)

    img_name = Path(png_name).stem
    bake_image = bpy.data.images.new(
        img_name, width=int(texture_size), height=int(texture_size)
    )

    if not joined.data.materials:
        tmp_mat = bpy.data.materials.new(name="Bake_Temp_Material")
        joined.data.materials.append(tmp_mat)

    for slot in joined.material_slots:
        if slot.material:
            if vcol_name:
                inject_vertex_color_to_material(slot.material, vcol_name)
            add_bake_image_node(slot.material, bake_image)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.ops.object.select_all(action="DESELECT")
    joined.select_set(True)
    bpy.context.view_layer.objects.active = joined

    print("Baking...")
    bpy.ops.object.bake(
        type="DIFFUSE",
        pass_filter={"COLOR"},
        use_clear=True,
        use_selected_to_active=False,
        margin=16,
    )

    bake_image.filepath_raw = str(png_path)
    bake_image.save()

    create_baked_material_assign(joined, bake_image)

    out_obj = asset_source / obj_name
    bpy.ops.wm.obj_export(filepath=str(out_obj), export_selected_objects=True)

    return {"png": str(png_path), "obj": str(out_obj), "mtl": "asset.mtl"}
