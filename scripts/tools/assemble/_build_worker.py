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

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.tools.mug_rack_pose._json_io import write_json


def _export(obj: object, path: Path, color: tuple[float, float, float]) -> None:
    import bmesh
    import bpy
    import numpy as np
    from scripts.tools.mug_rack_pose._geometry import load_mesh

    if not isinstance(obj, bpy.types.Object) or obj.type != "MESH":
        raise ValueError(f"{path.stem} must be a Blender mesh object")
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    cleaned = bmesh.new()
    try:
        # Bake the Blender transform explicitly: OBJ XYZ stays Z-up with no export-axis remap.
        cleaned.from_mesh(mesh)
        cleaned.transform(evaluated.matrix_world)
        # Boolean seams can contain sub-micron edges. Collapse their topology before
        # triangulation instead of dropping triangles and leaving tiny open cracks.
        bmesh.ops.remove_doubles(cleaned, verts=list(cleaned.verts), dist=1e-7)
        bmesh.ops.triangulate(cleaned, faces=list(cleaned.faces))
        bmesh.ops.dissolve_degenerate(cleaned, edges=list(cleaned.edges), dist=1e-8)
        bmesh.ops.triangulate(cleaned, faces=list(cleaned.faces))
        bmesh.ops.recalc_face_normals(cleaned, faces=list(cleaned.faces))
        cleaned.verts.index_update()
        vertices = np.array([v.co[:] for v in cleaned.verts])
        faces = np.array([[v.index for v in face.verts] for face in cleaned.faces])
    finally:
        cleaned.free()
        evaluated.to_mesh_clear()
    if len(faces) > _MAX_FACES:
        raise ValueError(
            f"{path.stem} has {len(faces)} triangles, exceeding max_faces={_MAX_FACES}; reduce mesh resolution or complexity"
        )
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


def main() -> None:
    """Build both assets and persist the source scene and machine observations."""
    global _MAX_FACES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, required=True)
    args = parser.parse_args()
    _MAX_FACES = args.max_faces
    import bpy
    from scripts.tools.mug_rack_pose._geometry import load_mesh

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
        _export(objects[role], path, color)
        mesh = load_mesh(path)
        observations["assets"][role] = {
            "path": str(path),
            "bounds": mesh.bounds.tolist(),
            "volume_m3": float(mesh.volume),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
        }
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "assets.blend"))
    write_json(output / "geometry.json", observations)


_MAX_FACES = 100000

if __name__ == "__main__":
    main()
