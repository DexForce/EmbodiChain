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

from embodichain.gen_sim.asset_engine.utils._simready_pipeline import (
    _main,
    _process_mesh,
    ask_llm_dimension,
    ask_llm_full_side_profile,
    ask_llm_semantics_info,
    ask_llm_upright_2a1,
    ask_llm_upright_rotation,
    ask_mllm_primary_surface,
    build_image_inputs,
    cardinal_views,
    delete_rendered_pngs,
    diagonal_views,
    dims_dict_to_xyz,
    down_views,
    encode_image,
    extract_json,
    front_views,
    get_chat_completion_content,
    rot_x,
    rot_y,
    rot_z,
    rotate_image_deg,
    side_profile,
    up_down_views,
    up_views,
)

from embodichain.gen_sim.services.runtime import GenSimServices, service_entry
from embodichain.gen_sim.services.compat import chat_from_legacy_messages

__all__: list[str] = []

import sys
from pathlib import Path
import numpy as np
import bpy
from scipy.spatial.transform import Rotation
from typing import Any

# ----------------------------------------------------------------------------
# Blender (bpy) backend
#
# trimesh / pyrender are replaced by Blender for mesh loading, rotation
# handling, rendering and export so that materials survive load -> export.
# Only the I/O / geometry backend is swapped here; every downstream decision
# uses the exact same math as before. A "mesh" handed around the pipeline is a
# ``bpy.types.Object`` whose transform is accumulated in ``matrix_world``.
# ----------------------------------------------------------------------------


def _get_matrix_world(obj):
    """Return the object's world matrix as a 4x4 numpy array."""
    return np.array(obj.matrix_world, dtype=np.float64)


def _set_matrix_world(obj, M):
    """Assign a 4x4 numpy array (or list) to the object's world matrix.

    Blender fills a nested-sequence assignment column-major, so the matrix is
    transposed before assignment to preserve the conventional row-major layout.
    """
    obj.matrix_world = np.asarray(M, dtype=np.float64).T.tolist()


def _rotation_matrix_4(angle_rad, axis):
    """4x4 rotation matrix about ``axis`` through the origin (numpy/scipy)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(angle_rad * axis).as_matrix()
    return M


def _bpy_reset_scene():
    """Delete every object in the current scene and purge orphan data."""
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    except Exception:
        pass


def _bpy_import_file(filepath):
    """Import a mesh file using the importer matching its extension."""
    filepath = str(filepath)
    ext = Path(filepath).suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=filepath)
        else:
            bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=filepath)
        else:
            bpy.ops.import_mesh.ply(filepath=filepath)
    elif ext == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=filepath)
        else:
            bpy.ops.import_mesh.stl(filepath=filepath)
    else:
        bpy.ops.import_scene.gltf(filepath=filepath)


def _bpy_load_object(mesh_input):
    """Load a mesh file into a single Blender mesh object.

    Imported mesh parts are baked to world space and joined into one object so
    that all material slots are preserved.
    """
    mesh_path = Path(mesh_input).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    _bpy_reset_scene()
    _bpy_import_file(mesh_path)

    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        raise ValueError(f"No mesh geometry found in file: {mesh_path}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]

    try:
        bpy.ops.object.make_single_user(
            object=True, obdata=True, material=False, animation=False
        )
    except Exception:
        pass

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    if len(mesh_objs) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    _set_matrix_world(obj, np.eye(4))
    return obj


def _bpy_world_bounds(obj):
    """Return ``(min_xyz, max_xyz)`` of the object's world-space AABB."""
    mw = np.array(obj.matrix_world, dtype=np.float64)
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    world = co @ mw[:3, :3].T + mw[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def _bpy_apply_matrix(obj, M4):
    """Left-multiply a 4x4 (numpy) world transform onto ``obj``."""
    _set_matrix_world(obj, np.asarray(M4, dtype=np.float64) @ _get_matrix_world(obj))


def _bpy_bake_transform(obj):
    """Bake the object's world matrix into its mesh data."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _bpy_remove_objects(objs):
    """Remove objects and their now-orphaned data blocks."""
    for o in objs:
        if o is None:
            continue
        data = o.data
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass
        if data is not None and getattr(data, "users", 1) == 0:
            for coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
                try:
                    coll.remove(data)
                    break
                except Exception:
                    pass


def _bpy_setup_world(scene, rgb):
    """Set a flat background color on the scene world."""
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bg.inputs[1].default_value = 1.0


def _bpy_setup_render(scene, res):
    """Configure a deterministic CPU render at ``res`` x ``res``."""
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "CPU"
        scene.cycles.samples = 16
    except Exception:
        pass
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass


def _new_sun(scene, name, strength):
    """Create and link a directional (sun) light."""
    ld = bpy.data.lights.new(name, type="SUN")
    ld.energy = strength
    lo = bpy.data.objects.new(name, ld)
    scene.collection.objects.link(lo)
    return lo


def normalize_to_unit_cube(mesh):
    minb, maxb = _bpy_world_bounds(mesh)
    size = maxb - minb
    size = np.maximum(size, 1e-8)
    scale = 1.0 / np.max(size)
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = scale
    _bpy_apply_matrix(mesh, S)
    minb_scaled, maxb_scaled = _bpy_world_bounds(mesh)
    center_scaled = (minb_scaled + maxb_scaled) / 2
    translation = np.array([0.5, 0.5, 0.5]) - center_scaled
    T = np.eye(4)
    T[:3, 3] = translation
    _bpy_apply_matrix(mesh, T)


def init_pose_simple(mesh_input):
    mesh = _bpy_load_object(mesh_input)
    rot_x = _rotation_matrix_4(np.radians(90), [1, 0, 0])
    rot_z = _rotation_matrix_4(np.radians(90), [0, 0, 1])
    combined_transform = rot_z @ rot_x
    _bpy_apply_matrix(mesh, combined_transform)
    normalize_to_unit_cube(mesh)
    return mesh


def render_views(mesh, views, out_dir, res=512):
    out_dir = Path(out_dir)
    scene = bpy.context.scene

    # Render-only copy: center to origin and scale the max extent to 1.
    minb, maxb = _bpy_world_bounds(mesh)
    center = (minb + maxb) / 2.0
    extents = maxb - minb
    scale = 1.0 / float(np.max(extents))

    dup = mesh.copy()
    dup.data = mesh.data.copy()
    scene.collection.objects.link(dup)
    T_center = np.eye(4)
    T_center[:3, 3] = -center
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = scale
    _set_matrix_world(dup, (S @ T_center) @ _get_matrix_world(dup))

    _bpy_setup_render(scene, res)
    _bpy_setup_world(scene, (230.0 / 255.0, 235.0 / 255.0, 245.0 / 255.0))

    cam_data = bpy.data.cameras.new("simready_cam")
    cam_data.angle = float(np.pi / 3.0)
    cam_obj = bpy.data.objects.new("simready_cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    key = _new_sun(scene, "simready_key", 4.0)
    fill = _new_sun(scene, "simready_fill", 1.5)
    back = _new_sun(scene, "simready_back", 1.2)

    # Hide every other renderable object so only the copy is rendered.
    hidden = []
    for o in scene.objects:
        if o.type == "MESH" and o is not dup:
            hidden.append((o, o.hide_render))
            o.hide_render = True

    results = []
    for name, eye in views:

        if name in ["view_from_111", "view_from_000"]:
            up = np.array([-1.0, -1.0, 0.0]) / np.sqrt(2.0)
        elif name == "view_from_up_to_bottom":
            up = np.array([-1.0, 0, 0.0])
        elif name == "view_from_bottom_to_up":
            up = np.array([1.0, 0, 0.0])
        else:
            up = np.array([0.0, 0.0, 1.0])

        target = np.array([0.0, 0.0, 0.0])
        f = target - eye
        f_hat = f / np.linalg.norm(f)

        r = np.cross(f_hat, up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f_hat)

        R = np.column_stack((r, u, -f_hat))

        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = eye

        _set_matrix_world(cam_obj, M)
        _set_matrix_world(key, M)

        fill_pose = np.eye(4)
        fill_pose[:3, 3] = eye + np.array([1.0, 1.0, 1.0])
        _set_matrix_world(fill, fill_pose)

        back_pose = np.eye(4)
        back_pose[:3, 3] = eye + np.array([-1.0, -1.0, -1.0])
        _set_matrix_world(back, back_pose)

        path = out_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)

        results.append({"path": str(path), "name": name, "camera_pose": M.tolist()})

    for o, flag in hidden:
        o.hide_render = flag
    _bpy_remove_objects([dup, cam_obj, key, fill, back])
    return results


def ask_mllm_detect_and_classify(
    views_data, extra_text="", *, services: GenSimServices | None = None
):

    instruction_text = """
            You are a single-purpose multimodal classifier. You will be given several images (multiple views) of a single object plus an OPTIONAL short text note ("Additional context"). Your job is twofold and must be completed in one step:
            1) Identify the object in plain short form (e.g. "coffee mug", "soccer ball", "laptop", "rock") and put it into the JSON field "detected_object" (string) or null if you truly cannot identify it.
            2) Classify the object's placement/orientation constraint into exactly one of three categories (0,1,2) using the provided canonical definitions and examples, and provide the additional fields described below.

            Important behavior constraints:
            - Return ONLY a single valid JSON string (no extra text, no explanation, no comments, no reasoning).
            - JSON must be syntactically valid, parseable, and use JSON literals (true/false/null where applicable).
            - Field order MUST be EXACTLY: detected_object, category, main_surface, orientation_requirement.
            - If a field is not applicable, use the JSON literal null.
            - Use common/public default usage (not niche). Follow the TIE-BREAKER rule below if ambiguous.
            - Use all provided views. If any view contradicts others, prioritize views that reveal human-interaction surfaces (front/diagonals) but still obey tie-breaker.
            - If an OPTIONAL "Additional context" text is provided, use it as auxiliary information to help identification/classification. If the text conflicts with clear visual evidence, prioritize visual evidence. If the images are ambiguous, allow the text to resolve the ambiguity. Do NOT output the additional context—only use it internally for judgment.

            CATEGORY MAPPING (exact):
            0 = Omnidirectional, no constraint
            1 = Rotation-insensitive, upright required
            2 = Has forward-facing primary use surface

            DECISION DEFINITIONS (the ONLY basis for judgment — use common/public default usage):

            Omnidirectional, no constraint (0)
            - Object is approx spherical or isotropic; function & appearance essentially identical under arbitrary orientation.
            - No placement posture (upright/sideways/flipped/rotated) is expected in public use.

            Rotation-insensitive, upright required (1)
            - Object has a stable upright support and a defined upright posture (flat bottom or center-of-gravity alignment).
            - Rotating around vertical axis does NOT change its function; but it must be upright (not upside-down or on its side) for normal function.

            Has forward-facing primary use surface (2)
            - Object has a single unique surface that carries its core function or primary human interaction (viewing, operating, aiming, serving, etc.).
            - In normal public use the object is expected to be oriented so that this surface faces the user/target/line-of-sight. Multiple equivalent faces mean it does NOT qualify.

            TIE-BREAKER / AMBIGUITY RULE (mandatory):
            - If more than one category could apply, choose the category with the stricter orientation constraint (precedence: 2 → 1 → 0).
            - Prefer common/public default usage, not niche setups.

            EXTENSIVE CANONICAL EXAMPLES (STRONG PRIOR — MUST FOLLOW)
            CATEGORY 0 examples: ball, basketball, soccer ball, tennis ball, marble, pebble, orange, balloon(round)
            CATEGORY 1 examples: cup, coffee cup,moka pot, drinking glass, bottle, vase, bowl, suitcase(standing), candle
            CATEGORY 2 examples: monitor, laptop, smartphone, table lamp (head facing), flashlight, camera, car, bicycle, oven(front), speaker(front grille), keyboard, painting, wall clock

            OUTPUT JSON FORMAT (strict — EXACT four fields in this order; use JSON literals):
            {
            "detected_object": string or null,
            "category": integer,          // 0 | 1 | 2
            "main_surface": string or null,
            "orientation_requirement": string or null
            }

            FIELD RULES:
            - "detected_object": short, common object name (lowercase preferred) representing the model's best identification, or null if unidentifiable.
            - "category": integer 0|1|2.
            - "main_surface": Only provide a short, specific name of the forward-facing surface when category == 2 (e.g. "screen", "lamp_head", "door_face", "keyboard_surface"). Otherwise null.
            - "orientation_requirement": Only provide a concise canonical resting-orientation instruction when category == 2. You MUST choose exactly one of the following three semantic directions for the object's normal real-world static pose:
              * "face_up"      -> the main surface is intended to face upward toward +Z / gravity opposite, e.g. smartphone lying flat with screen up, keyboard on table, tray-like objects. Laptop need keyboard facing up to count as face_up, even if screen is more front-facing.
              * "face_forward" -> the main surface is intended to face the user/target in a vertical stance, e.g. monitor screen, oven front, speaker grille, camera front.
              * "face_down"    -> the main surface is intended to face downward in the usual stable static pose, e.g. brush bristles or contact surface downward when naturally placed/used.
              If the object is category 1 or 0, set null.
            - Do NOT add any other fields.

            VALIDATION RULES (model must satisfy):
            - JSON must be syntactically valid and parseable.
            - Field order must be exactly as above.
            - No extraneous text.

            INSTRUCTIONS FOR IMAGE USE:
            - You will be provided a list of labeled views (each labeled with a short tag like "Front", "Back", "Right", "Left", "Diagonal_1", "Diagonal_2"). Use all images to resolve shape, symmetry, handles, screens, bases, cutouts, wheels, or any directional cues.
            - Remember the mesh was normalized to the unit cube [0,0,0]→[1,1,1] for rendering—do NOT infer real-world size from pixel dimensions; rely on shape & functional features.
            - If the object is clearly symmetric with no single primary face and no stable base, prefer category 0. If there is a clear base but no single forward-facing use surface, prefer category 1. If there is a screen, grill, face, nozzle, spout, or other unique human-facing surface, prefer category 2.
            - For category 2 objects, infer the NORMAL STATIC RESTING ORIENTATION in the real world, not merely the visible camera view. Decide whether the primary surface is usually face_up, face_forward, or face_down in its standard placed state.

            NOW: classify the provided object and identify it using the images and the OPTIONAL Additional context text.
"""

    content = [{"type": "text", "text": instruction_text}]

    if extra_text and extra_text.strip():
        content.append(
            {"type": "text", "text": f"Additional context: {extra_text.strip()}"}
        )

    content.extend(build_image_inputs(views_data))
    resp = chat_from_legacy_messages(
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
        services=services,
        role="asset_orientation",
    )
    raw = get_chat_completion_content(resp)
    return extract_json(raw)


def apply_rotations(mesh, rotations):
    R = np.eye(4)
    R[:3, :3] = rotations
    _bpy_apply_matrix(mesh, R)


def get_aabb_dims(mesh):

    minb, maxb = _bpy_world_bounds(mesh)
    extents = np.asarray(maxb, dtype=float) - np.asarray(minb, dtype=float)
    return {
        "height": float(extents[2]),
        "width": float(extents[1]),
        "depth": float(extents[0]),
    }


def scale_mesh_uniform_to_dimensions(
    mesh,
    target_dims: dict,
    current_dims: dict | None = None,
    eps: float = 1e-8,
):

    if current_dims is None:
        current_dims = get_aabb_dims(mesh)

    cur = dims_dict_to_xyz(current_dims)
    tgt = dims_dict_to_xyz(target_dims)

    valid = np.isfinite(cur) & np.isfinite(tgt) & (cur > eps) & (tgt > eps)
    if not np.any(valid):
        raise ValueError(f"Invalid dims. current={current_dims}, target={target_dims}")

    ratios = tgt[valid] / cur[valid]

    scale = float(np.median(ratios))

    minb, maxb = _bpy_world_bounds(mesh)
    center = (minb + maxb) / 2.0
    T_neg = np.eye(4)
    T_neg[:3, 3] = -center
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = scale
    T_pos = np.eye(4)
    T_pos[:3, 3] = center
    _bpy_apply_matrix(mesh, T_neg)
    _bpy_apply_matrix(mesh, S)
    _bpy_apply_matrix(mesh, T_pos)

    return mesh, scale


def export_final_mesh(mesh, name, out_dir: Path):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)
    minb, maxb = _bpy_world_bounds(mesh)
    bottom_center = np.array(
        [(minb[0] + maxb[0]) / 2.0, (minb[1] + maxb[1]) / 2.0, minb[2]], dtype=float
    )
    T_trans = np.eye(4)
    T_trans[:3, 3] = -bottom_center
    _bpy_apply_matrix(mesh, T_trans)
    _bpy_bake_transform(mesh)

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

    out_path = out_dir / f"{name}_simready.glb"
    out_path = out_path.resolve()

    print(f"Exporting final mesh to: {out_path} (bottom-face center moved to origin)")
    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLB",
        use_selection=True,
    )

    out_path = out_dir / f"asset_simready.obj"
    out_path = out_path.resolve()
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(out_path),
            export_selected_objects=True,
            export_materials=True,
        )
    else:
        bpy.ops.export_scene.obj(filepath=str(out_path), use_selection=True)
    return str(out_path)


@service_entry
def process_mesh(
    file: Any,
    name: str | None = None,
    extra_text: str = "",
    out_dir: str | Path = "renders",
    res: int = 1024,
    *,
    services: GenSimServices | None = None,
) -> dict[str, Any]:
    """Orient, scale and export an asset through this module's geometry backend.

    Args:
        file: Input mesh or source path.
        name: Output asset name for an in-memory mesh.
        extra_text: Additional semantic and dimension context.
        out_dir: Directory for rendered views and exported geometry.
        res: Rendered image resolution.
        services: Caller-owned model services, or the entry point's scoped run.

    Returns:
        Exported path, scaling measurements and semantic classification.
    """
    return _process_mesh(
        file,
        name,
        extra_text,
        out_dir,
        res,
        backend=sys.modules[__name__],
        services=services,
    )


@service_entry
def main(*, services: GenSimServices | None = None) -> None:
    """Run the asset-processing CLI with this module's geometry backend."""
    _main(backend=sys.modules[__name__], services=services)


if __name__ == "__main__":
    main()
