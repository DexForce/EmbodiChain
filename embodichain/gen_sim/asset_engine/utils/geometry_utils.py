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

import bpy
from pathlib import Path


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
