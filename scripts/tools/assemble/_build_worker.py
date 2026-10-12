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

"""Execute a saved Blender program in a separate process and export local meshes."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import TYPE_CHECKING

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.tools.assemble._json_io import write_json

if TYPE_CHECKING:
    import bmesh


def _clean(mesh: bmesh.types.BMesh) -> None:
    import bmesh

    # Collapse Boolean seam edges before triangulation, preserving closed topology.
    bmesh.ops.remove_doubles(mesh, verts=list(mesh.verts), dist=1e-7)
    bmesh.ops.triangulate(mesh, faces=list(mesh.faces))
    bmesh.ops.dissolve_degenerate(mesh, edges=list(mesh.edges), dist=1e-8)
    bmesh.ops.triangulate(mesh, faces=list(mesh.faces))
    bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))


def _fit_face_budget(mesh: bmesh.types.BMesh, role: str) -> dict:
    import bpy

    original_count = len(mesh.faces)
    # Leave margin for cleanup/retriangulation. Work on evaluated, triangulated
    # geometry so quad-heavy voxel remeshes cannot hide their exported face count.
    for _ in range(3):
        count = len(mesh.faces)
        if count <= _MAX_FACES:
            break
        temporary_mesh = bpy.data.meshes.new("assembly_decimation")
        temporary = bpy.data.objects.new("assembly_decimation", temporary_mesh)
        bpy.context.scene.collection.objects.link(temporary)
        try:
            mesh.to_mesh(temporary_mesh)
            modifier = temporary.modifiers.new("Face budget", "DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.ratio = 0.95 * _MAX_FACES / count
            modifier.use_collapse_triangulate = True
            bpy.context.view_layer.update()
            evaluated = temporary.evaluated_get(bpy.context.evaluated_depsgraph_get())
            reduced = evaluated.to_mesh()
            try:
                mesh.clear()
                mesh.from_mesh(reduced)
            finally:
                evaluated.to_mesh_clear()
            _clean(mesh)
        finally:
            bpy.data.objects.remove(temporary, do_unlink=True)
            bpy.data.meshes.remove(temporary_mesh)
        if len(mesh.faces) >= count:
            break
    if len(mesh.faces) > _MAX_FACES:
        raise ValueError(
            f"{role} still has {len(mesh.faces)} triangles after simplification "
            f"(originally {original_count}), exceeding max_faces={_MAX_FACES}; "
            "reduce mesh resolution or complexity"
        )
    return {
        "input_triangles": original_count,
        "exported_triangles": len(mesh.faces),
        "decimated": original_count > _MAX_FACES,
    }


def _export(obj: object, path: Path, color: tuple[float, float, float]) -> dict:
    import bmesh
    import bpy
    from mathutils import Matrix
    import numpy as np
    from scripts.tools.assemble._geometry import load_mesh

    if not isinstance(obj, bpy.types.Object) or obj.type != "MESH":
        raise ValueError(f"{path.stem} must be a Blender mesh object")
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    cleaned = bmesh.new()
    try:
        # Bake the Blender transform explicitly: OBJ XYZ stays Z-up with no export-axis remap.
        try:
            cleaned.from_mesh(mesh)
            cleaned.transform(evaluated.matrix_world)
        finally:
            evaluated.to_mesh_clear()
        _clean(cleaned)
        processing = _fit_face_budget(cleaned, path.stem)
        if processing["decimated"]:
            # Keep assets.blend consistent with the actual simplified OBJ geometry.
            baked = bpy.data.meshes.new(f"{obj.name}_export")
            cleaned.to_mesh(baked)
            obj.modifiers.clear()
            obj.data = baked
            obj.matrix_world = Matrix.Identity(4)
        cleaned.verts.index_update()
        vertices = np.array([v.co[:] for v in cleaned.verts])
        faces = np.array([[v.index for v in face.verts] for face in cleaned.faces])
    finally:
        cleaned.free()
    with path.open("w") as stream:
        stream.write(f"mtllib {path.stem}.mtl\no {path.stem}\nusemtl {path.stem}\n")
        for vertex in vertices:
            stream.write(
                "v " + " ".join(format(float(x), ".12g") for x in vertex) + "\n"
            )
        for face in faces:
            stream.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")
    path.with_suffix(".mtl").write_text(
        f"newmtl {path.stem}\nKd {' '.join(map(str, color))}\nKa 0.1 0.1 0.1\nKs 0.1 0.1 0.1\nNs 24\n"
    )
    solid = load_mesh(path)
    if solid.extents.min() < 0.0001 or solid.extents.max() > 3:
        raise ValueError(
            "Generated assets must be between 0.1 mm and 3 m along every axis"
        )
    return processing


def main() -> None:
    """Build both assets and persist the source scene and machine observations."""
    global _MAX_FACES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, required=True)
    args = parser.parse_args()
    _MAX_FACES = args.max_faces
    import bpy
    from scripts.tools.assemble._geometry import load_mesh

    bpy.ops.wm.read_factory_settings(use_empty=True)
    namespace = runpy.run_path(str(args.source), run_name="assembly_generated")
    objects = namespace["build"]()
    if (
        not isinstance(objects, dict)
        or not {"base", "assemble", "metadata"} <= objects.keys()
    ):
        raise ValueError("build() must return base, assemble, and metadata")
    if objects["base"] == objects["assemble"]:
        raise ValueError("base and assemble must be different objects")
    output = args.source.parent
    observations = {"metadata": objects["metadata"], "assets": {}}
    for role, color in (("base", (0.35, 0.22, 0.12)), ("assemble", (0.12, 0.58, 0.42))):
        path = output / f"{role}.obj"
        processing = _export(objects[role], path, color)
        mesh = load_mesh(path)
        observations["assets"][role] = {
            "path": str(path),
            "bounds": mesh.bounds.tolist(),
            "volume_m3": float(mesh.volume),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "mesh_processing": processing,
        }
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "assets.blend"))
    write_json(output / "geometry.json", observations)


_MAX_FACES = 100000

if __name__ == "__main__":
    main()
