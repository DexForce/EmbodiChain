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

"""Process Asset Engine meshes inside the isolated Blender worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bpy

__all__ = []


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


def _blender_remesh_bake(
    source_file: Path,
    asset_source: Path,
    texture_size: int | None = None,
    png_name: str | None = None,
    voxel_size: float | None = None,
    decimate_ratio: float | None = None,
    obj_name: str = "asset.obj",
    config: dict[str, Any] | None = None,
):
    """Remesh a high-poly mesh into a low-poly one and bake textures via Blender."""
    import bpy

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


def load_obj(filepath):
    bpy.ops.wm.obj_import(filepath=str(filepath))
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    return objs


def join_meshes(objs):
    if not objs:
        raise RuntimeError("No mesh objects to join.")

    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)

    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def decimate_optimized(
    obj,
    ratio: float = 0.5,
    weld_distance: float = 0.0001,
    collapse_triangulate: bool = True,
):

    bpy.context.view_layer.objects.active = obj

    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # 1) Weld
    # weld_mod = obj.modifiers.new(name="Weld", type="WELD")
    # weld_mod.merge_threshold = weld_distance
    # bpy.ops.object.modifier_apply(modifier=weld_mod.name)
    # bpy.ops.object.mode_set(mode="EDIT")
    # bpy.ops.mesh.select_all(action="SELECT")

    # bpy.ops.mesh.normals_make_consistent(inside=False)
    # bpy.ops.mesh.customdata_custom_splitnormals_clear()

    # bpy.ops.object.mode_set(mode="OBJECT")

    # 2) remove loose
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_loose()
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    # 3) decimate
    print(f"Simplifying mesh (Ratio: {ratio})...")
    decimate_mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
    decimate_mod.ratio = ratio
    decimate_mod.use_collapse_triangulate = collapse_triangulate
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)

    # 4) post clean
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=weld_distance)
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode="OBJECT")

    print(
        f"[Info] Optimized state: Vertices {len(obj.data.vertices)}, Faces {len(obj.data.polygons)}"
    )

    return obj


def clean_mesh(obj, merge_dist=1e-5, remove_non_manifold=True, triangulate=False):
    bpy.context.view_layer.objects.active = obj

    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.mesh.remove_doubles(threshold=merge_dist)

    bpy.ops.mesh.delete_loose()

    bpy.ops.mesh.dissolve_degenerate()

    bpy.ops.mesh.normals_make_consistent(inside=False)

    if remove_non_manifold:
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.delete(type="VERT")

    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=merge_dist)
    bpy.ops.mesh.delete_loose()

    if triangulate:
        bpy.ops.mesh.quads_convert_to_tris()

    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def fill_holes(obj, max_sides=8):
    bpy.context.view_layer.objects.active = obj

    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.mesh.fill_holes(sides=max_sides)

    bpy.ops.mesh.beautify_fill()
    bpy.ops.mesh.dissolve_degenerate()
    bpy.ops.mesh.normals_make_consistent(inside=False)

    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def export_obj(obj, out_path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.wm.obj_export(filepath=str(out_path), export_selected_objects=True)


def process_obj(
    input_path,
    output_path,
    ratio=0.5,
    weld_distance=0.0001,
    merge_dist=1e-5,
    remove_non_manifold=True,
    triangulate=False,
    collapse_triangulate=True,
):
    clear_scene()
    objs = load_obj(input_path)
    if not objs:
        raise RuntimeError("No mesh objects imported.")

    obj = join_meshes(objs)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    clean_mesh(
        obj,
        merge_dist=merge_dist,
        remove_non_manifold=remove_non_manifold,
        triangulate=triangulate,
    )
    decimate_optimized(
        obj,
        ratio=ratio,
        weld_distance=weld_distance,
        collapse_triangulate=collapse_triangulate,
    )

    export_obj(obj, output_path)
    print("Clean mesh saved to:", output_path)
