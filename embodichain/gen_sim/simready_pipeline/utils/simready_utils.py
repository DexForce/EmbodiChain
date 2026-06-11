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
import base64
import json
import os
import re
from pathlib import Path
import numpy as np
import trimesh
import pyrender
from PIL import Image
from openai import OpenAI
import itertools
from scipy.spatial import ConvexHull
from typing import Dict, Any, List


def _load_gen_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "gen_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"gen_config.json not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    cfg = raw_cfg.get("llm", {}).get("openai_compatible", {})
    cfg["api_key"] = os.getenv("OPENAI_API_KEY") or cfg.get("api_key", "")
    cfg["model"] = os.getenv("OPENAI_MODEL") or cfg.get("model", "")
    cfg["base_url"] = os.getenv("OPENAI_BASE_URL") or cfg.get("base_url", "")
    cfg["default_query"] = cfg.get("default_query", {})
    if cfg["base_url"]:
        cfg["base_url"] = cfg["base_url"].rstrip("/")

    required = ["api_key", "model", "base_url"]
    missing = [k for k in required if k not in cfg or not cfg[k]]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return cfg


_GEN_CONFIG = _load_gen_config()

DEPLOYMENT = _GEN_CONFIG["model"]

client = OpenAI(
    api_key=_GEN_CONFIG["api_key"],
    base_url=_GEN_CONFIG["base_url"],
    default_query=_GEN_CONFIG.get("default_query") or None,
)

STRATEGY = None


def get_chat_completion_content(resp: Any) -> str:
    """Return chat completion message content with a clearer URL hint on bad responses."""

    try:
        choices = resp.choices
    except AttributeError as exc:
        raise AttributeError(
            "Unexpected OpenAI-compatible chat completion response: missing "
            "`choices`. Please check OPENAI_BASE_URL. OpenAI-style chat "
            "completion endpoints usually require the version path, for example "
            '"https://api.openai.com/v1", and many compatible APIs also require '
            'a "/v1" suffix.'
        ) from exc
    return choices[0].message.content


diagonal_views = [
    ("view_from_111", np.array([1.3, 1.3, 1.3], dtype=float)),
    ("view_from_000", np.array([-0.8, -0.8, -0.8], dtype=float)),
]
cardinal_views = [
    ("view_from_front", np.array([1.8, 0.5, 0.5], dtype=float)),
    ("view_from_left", np.array([0.5, -1.8, 0.5], dtype=float)),
    ("view_from_right", np.array([0.5, 1.8, 0.5], dtype=float)),
    ("view_from_back", np.array([-1.8, 0.5, 0.5], dtype=float)),
]
up_down_views = [
    ("view_from_up_to_bottom", np.array([0.5, 0.5, 2.2], dtype=float)),
    ("view_from_bottom_to_up", np.array([0.5, 0.5, -1.2], dtype=float)),
]

up_views = [
    ("view_from_up_to_bottom", np.array([0.5, 0.5, 2.2], dtype=float)),
]

down_views = [
    ("view_from_bottom_to_up", np.array([0.5, 0.5, -1.2], dtype=float)),
]

front_views = [
    ("view_from_front", np.array([1.8, 0.5, 0.5], dtype=float)),
]

side_profile = [
    ("view_from_up_to_bottom", np.array([0.5, 0.5, 2.2], dtype=float)),
    ("view_from_front", np.array([1.8, 0.5, 0.5], dtype=float)),
]


def normalize_to_unit_cube(mesh):
    minb, maxb = mesh.bounds
    size = maxb - minb
    size = np.maximum(size, 1e-8)
    scale = 1.0 / np.max(size)
    mesh.apply_scale(scale)
    minb_scaled, maxb_scaled = mesh.bounds
    center_scaled = (minb_scaled + maxb_scaled) / 2
    translation = np.array([0.5, 0.5, 0.5]) - center_scaled
    mesh.apply_translation(translation)


def compute_support_area(mesh, eps=1e-2):
    z_min = mesh.bounds[0][2]
    verts = np.asarray(mesh.vertices)
    mask = np.abs(verts[:, 2] - z_min) < eps
    pts = verts[mask][:, :2]
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


import numpy as np
import trimesh
from pathlib import Path


def init_pose(mesh_input):

    fallback_mesh = None
    mesh: trimesh.Trimesh = None

    if isinstance(mesh_input, trimesh.Trimesh):
        mesh = mesh_input.copy()
        fallback_mesh = mesh_input.copy()
    else:
        mesh_path = Path(mesh_input).resolve()
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        mesh = trimesh.load(mesh_path, force="mesh")
        fallback_mesh = mesh.copy()

    def compute_pca_axes(mesh):
        verts = np.asarray(mesh.vertices)
        centroid = verts.mean(axis=0)
        centered = verts - centroid
        cov = np.cov(centered.T)
        U, _, _ = np.linalg.svd(cov)
        R = U
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        return R

    def closest_axis(v):
        idx = np.argmax(np.abs(v))
        sign = np.sign(v[idx])
        axis = np.zeros(3)
        axis[idx] = sign
        return axis

    def generate_discrete_flips():
        rotations = []
        Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        I = np.eye(3)
        rotations.append(I)
        Rx180 = Rx90 @ Rx90
        rotations.append(Rx180)
        rotations.append(Rx90)
        Rx_neg90 = Rx90.T
        rotations.append(Rx_neg90)
        rotations.append(Ry90)
        Ry_neg90 = Ry90.T
        rotations.append(Ry_neg90)
        return rotations

    def compute_support_area(mesh):
        hull = trimesh.convex.convex_hull(mesh)
        support_poly = hull.project(plane=[0, 0, 1], origin=[0, 0, 0])
        return support_poly.area

    def stability_score(mesh):
        area = compute_support_area(mesh)
        com_z = mesh.center_mass[2]
        return -(area / (com_z + 1e-6))

    def normalize_to_unit_cube(mesh):
        extents = mesh.extents
        scale = 1.0 / np.max(extents)
        mesh.apply_scale(scale)
        mesh.vertices -= mesh.vertices.mean(axis=0)
        z_min = mesh.bounds[0][2]
        mesh.apply_translation([0, 0, -z_min])

    def process_alignment(initial_mesh, align_type):
        m = initial_mesh.copy()
        if align_type == "pca":
            R_pca = compute_pca_axes(m)
            T = np.eye(4)
            T[:3, :3] = R_pca.T
            m.apply_transform(T)
            U = compute_pca_axes(m)
            x, y, z = U[:, 0], U[:, 1], U[:, 2]
            nx = closest_axis(x)
            ny = closest_axis(y)
            nz = closest_axis(z)
            nz /= np.linalg.norm(nz)
            nx = nx - nz * np.dot(nx, nz)
            nx /= np.linalg.norm(nx)
            ny = np.cross(nz, nx)
            R_snap = np.column_stack([nx, ny, nz])
            m.apply_transform(np.eye(4)[:3, :3] @ R_snap)

        elif align_type == "obb":
            to_origin, _ = trimesh.bounds.oriented_bounds(m)
            m.apply_transform(to_origin)
            R = to_origin[:3, :3]
            if np.linalg.det(R) < 0:
                m.apply_transform(np.diag([1, 1, -1, 1]))
        else:
            raise ValueError(f"Unknown type {align_type}")

        best_score = float("inf")
        best = None
        for Rf in generate_discrete_flips():
            mc = m.copy()
            Tf = np.eye(4)
            Tf[:3, :3] = Rf
            mc.apply_transform(Tf)
            zmin = mc.bounds[0][2]
            mc.apply_translation([0, 0, -zmin])
            s = stability_score(mc)
            if s < best_score:
                best_score = s
                best = mc.copy()
        return best, best_score

    try:
        mesh_pca, score_pca = process_alignment(mesh, "pca")
        mesh_obb, score_obb = process_alignment(mesh, "obb")

        area_pca = compute_support_area(mesh_pca)
        area_obb = compute_support_area(mesh_obb)

        result_mesh = mesh_obb
        STRATEGY = "OBB"

        if area_pca > area_obb * 1.3:
            result_mesh = mesh_pca
            STRATEGY = "PCA"

        normalize_to_unit_cube(result_mesh)
        return result_mesh

    except Exception as e:
        return fallback_mesh


def extract_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response:\n" + text)
    return json.loads(match.group())


def encode_image(p):
    img_path = Path(p).resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_image_inputs(views_data):
    content = []
    for v in views_data:
        name = v["name"]
        img_b64 = encode_image(v["path"])
        content.append({"type": "text", "text": f'View "{name}"'})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
        )
    return content


def render_views(mesh, views, out_dir, res=512):
    import numpy as np
    import pyrender
    from PIL import Image
    import trimesh

    mesh = mesh.copy()

    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    scale = 1.0 / np.max(mesh.extents)
    mesh.apply_scale(scale)
    mesh_pyr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    renderer = pyrender.OffscreenRenderer(res, res)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
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

        scene = pyrender.Scene(bg_color=[230, 235, 245, 255])

        scene.add(mesh_pyr)
        scene.add(cam, pose=M)

        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=M)

        fill_pose = np.eye(4)
        fill_pose[:3, 3] = eye + np.array([1.0, 1.0, 1.0])
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=fill_pose
        )

        back_pose = np.eye(4)
        back_pose[:3, 3] = eye + np.array([-1.0, -1.0, -1.0])
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=back_pose
        )

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        img = Image.fromarray(color)
        img = img.convert("RGB")

        path = out_dir / f"{name}.png"
        img.save(path, quality=95)

        results.append({"path": str(path), "name": name, "camera_pose": M.tolist()})

    renderer.delete()
    return results


def ask_mllm_detect_and_classify(views_data, extra_text=""):

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
              * "face_up"      -> the main surface is intended to face upward toward +Z / gravity opposite, e.g. smartphone lying flat with screen up, keyboard on table, tray-like objects.
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
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
    )
    raw = get_chat_completion_content(resp)
    return extract_json(raw)


def ask_mllm_primary_surface(
    views_data,
    object_name="None",
    main_surface="None",
    orientation_requirement="None",
    extra_text="",
):

    instruction_text = f"""
    You are a single-purpose multimodal classifier. You will be given 6 images of a single object, rendered from different views. Your task is to identify **the image that best shows the object's forward-facing primary use surface**, defined as the surface that:

    - Carries the object's core function (viewing, operating, aiming, serving, pressing, interacting, etc.)
    - Faces the human user or line-of-sight in normal use
    - Is unique and human-accessible (not a symmetrical or bottom/support surface)
    - Should prioritize the **front-facing view**, even if other angles also partially show it (e.g., top-down view of a laptop shows screen but front view is preferred)

    Additional guidance based on prior classification:
    - Detected object: {object_name}
    - Possible main surface: {main_surface}
    - Orientation requirement: {orientation_requirement}

    If {main_surface} or {orientation_requirement} are provided (not "None"), use them to help identify which image shows the main functional surface. If they conflict with visual evidence, prioritize visual evidence. 

    Return a single valid JSON string with exactly one field:

    {{
    "primary_surface_view": string // the name of the image that best shows the forward-facing primary use surface
    }}

    Rules:
    - Use only the image IDs (names) provided in input.
    - If the object has no clear forward-facing primary surface (fully isotropic or omnidirectional), return null.
    - Do NOT add any extra text, explanation, or comments.
    - Ensure the JSON is syntactically valid and parseable.

    Use the six views to judge shape, handles, screens, bases, spouts, lenses, doors, or other directional human-facing cues. Prioritize the image that a person would naturally face to use or interact with the object. You can also use any Additional context text provided: {extra_text if extra_text else "None"}.
    """

    content = [{"type": "text", "text": instruction_text}]

    if extra_text and extra_text.strip():
        content.append(
            {"type": "text", "text": f"Additional context: {extra_text.strip()}"}
        )

    content.extend(build_image_inputs(views_data))
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
    )
    raw = get_chat_completion_content(resp)
    return extract_json(raw)


def ask_llm_upright_2a1(object_name, upright_img_path, flipped_img_path):
    for p in [upright_img_path, flipped_img_path]:
        img_path = Path(p).resolve()
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image required by LLM for upright judgment not found: {img_path}"
            )

    imgs_payload = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(upright_img_path)}"
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(flipped_img_path)}"
            },
        },
    ]

    prompt = f"""
You are a physical-world perception model.

An object of category: "{object_name}" is shown in TWO images.

IMPORTANT:
- The two images show the SAME object.
- One image is physically correct (upright).
- The other image is rotated 180 degrees (upside-down).
- Exactly ONE image shows the object in its natural real-world upright orientation.

Your task: choose which image is upright based on common human-world object orientation knowledge.

Image A = first image  
Image B = second image  

Rules:
- Think about gravity, support base, typical usage posture.
- Objects are not used upside-down in normal life.
- Do NOT say "both", "uncertain", or explanations.
- You MUST choose one.

OUTPUT JSON ONLY:

{{
  "upright_image": "A" or "B",
  "confidence": 0.0-1.0
}}
"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
    )
    return extract_json(get_chat_completion_content(resp))


def ask_llm_full_side_profile(object_name, views_data):
    img_paths = []
    for v in views_data:
        name = v["name"]
        path = v["path"]
        img_paths.append(path)

    for p in img_paths:
        img_path = Path(p).resolve()
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image required by LLM for upright judgment not found: {img_path}"
            )
    imgs_payload = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(p)}"},
        }
        for p in img_paths
    ]

    prompt = f"""
            You are a visual reasoning model. 

            An object of category: "{object_name}" is shown in TWO images. Both images show the same object in upright posture, but from different angles.

            Your task: determine **which image shows the object's full height and side profile**—that is, the complete body shape and natural standing posture.

            Rules:
            - Choose exactly ONE image that best shows the object's full side profile.
            - Think about how this object would stand in real life.
            - Do NOT output explanations.
            - Only return the index of the image.

            OUTPUT JSON ONLY:

            {{
            "full_side_profile_image": "A" or "B",
            "confidence": 0.0-1.0
            }}
            """

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
    )
    return extract_json(get_chat_completion_content(resp))


def ask_llm_upright_rotation(object_name, rotated_imgs_paths):
    """
    rotated_imgs_paths: list of image paths in order [0°, 90°, 180°, 270°]
    object_name: string, name of the object
    """

    for p in rotated_imgs_paths:
        img_path = Path(p).resolve()
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image required by LLM for upright judgment not found: {img_path}"
            )
    imgs_payload = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(p)}"},
        }
        for p in rotated_imgs_paths
    ]

    prompt = f"""
ou are a physical-world orientation judgment model.

An object of category: "{object_name}" is shown in FOUR images.
All images show the SAME object from the SAME camera viewpoint.

Your task is to choose the image that best matches the object's natural upright pose in everyday life.

Think about:
- how the object would normally rest on a table, floor, or other surface
- gravity and stable support
- the object's base, feet, bottom, opening, handle, screen, or functional side
- the orientation people would normally place, hold, or use it in real life

Important:
- Choose the image that looks most naturally upright and stable in the real world.
- Do NOT rely on any hidden rotation pattern.
- Do NOT assume the object is already upright in the original image.
- Do NOT explain your reasoning.
- Only return the index of the best upright image.
The correct answer must be the image that a person would most likely consider the object's normal real-world standing orientation.

Image indices:
- 0 = first image
- 1 = second image
- 2 = third image
- 3 = fourth image

OUTPUT JSON ONLY:

{{
  "upright_index": 0|1|2|3,
  "confidence": 0.0-1.0
}}
"""
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
    )
    return extract_json(get_chat_completion_content(resp))


def ask_llm_dimension(object_name, img_paths, user_text_hint, current_bbox_dims):

    if isinstance(img_paths, (str, Path)):
        img_paths = [{"path": str(img_paths)}]

    imgs_payload = []
    for item in img_paths:
        img_path = Path(item["path"]).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image required by LLM not found: {img_path}")
        imgs_payload.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(img_path)}"},
            }
        )

    current_bbox_json = json.dumps(current_bbox_dims, ensure_ascii=False)

    prompt = f"""
You are a robotics perception and scene analysis expert.
Your task is to estimate the REAL-WORLD physical size of the object in meters.

CONTEXT:
- The mesh has already been normalized for rendering.
- You are given the object's CURRENT NORMALIZED AABB SIZE (ordinary axis-aligned bounding box, NOT PCA, NOT minimum-volume OBB).
- Use that normalized bbox size as a STRONG SHAPE PRIOR.
- Your output MUST be a plausible real-world size in meters for the exact state shown in the images.
- You must preserve the object's proportions as much as possible; do NOT invent an anisotropic resize. The downstream system will apply ONLY a uniform scale.

CURRENT NORMALIZED AABB SIZE (unitless, from ordinary bbox):
{current_bbox_json}

DEFINITIONS:
- height = vertical size when a human faces the object (top -> bottom), Z axis
- width  = left-to-right size when facing the object, Y axis
- depth  = front-back thickness, X axis

USER PROVIDED HINT:
- object_name: {object_name}
- extra_hint: {user_text_hint}

INSTRUCTIONS:
1. Analyze ALL provided images together.
2. Determine the exact visible state first (open/closed/folded/etc.).
3. Estimate the object's real-world physical dimensions in meters for that exact state.
4. Use the normalized bbox as a shape prior so the returned dimensions are consistent with the object's proportions.
5. If uncertain, give the most plausible central estimate. Do not return null unless completely unrecognizable.

Return JSON ONLY with:
{{
  "object_name": string,
  "object_description": string,
  "dimensions_m": {{
    "height": float,
    "width": float,
    "depth": float
  }},
  "confidence": float
}}

CRITICAL:
- JSON only.
- Units must be meters.
- Output real physical dimensions, not normalized values.
- Do not explain anything.
"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
    )
    return extract_json(get_chat_completion_content(resp))


def rotate_image_deg(input_path, deg, output_path):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file for image rotation not found: {input_path}"
        )

    img = Image.open(input_path)
    img_rot = img.rotate(deg, expand=True)
    img_rot.save(output_path)
    return str(output_path)


def rot_x(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def apply_rotations(mesh, rotations):
    R = np.eye(3)
    T = np.eye(4)
    T[:3, :3] = rotations
    mesh.apply_transform(T)


def get_aabb_dims(mesh: trimesh.Trimesh):

    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    return {
        "height": float(extents[2]),
        "width": float(extents[1]),
        "depth": float(extents[0]),
    }


def dims_dict_to_xyz(dims: dict):

    return np.array(
        [
            float(dims.get("depth", np.nan)),
            float(dims.get("width", np.nan)),
            float(dims.get("height", np.nan)),
        ],
        dtype=float,
    )


def scale_mesh_uniform_to_dimensions(
    mesh: trimesh.Trimesh,
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

    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    mesh.apply_translation(center)

    return mesh, scale


def ask_llm_semantics_info(object_name, img_paths, user_text_hint=""):

    imgs_payload = []
    for item in img_paths:
        img_path = Path(item["path"]).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image required by LLM not found: {img_path}")
        imgs_payload.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(img_path)}"},
            }
        )

    prompt = f"""
You are a robotics asset semantics expert.

Your task is to infer semantic information from multiple rendered views of a 3D object.
The object will later be used for robotics simulation, physical property estimation, and manipulation planning.

INPUTS:
- object_name: {object_name}
- user_hint: {user_text_hint}

INSTRUCTIONS:
1. Use the front view and diagonal views jointly.
2. Infer the most likely semantic category of the object.
3. Identify the most likely main material(s) visible from the object appearance.
4. Write a concise but information-rich description that includes:
   - object type / category
   - likely main material(s)
   - surface finish / texture
   - rigid or flexible nature
   - notable functional or structural parts
5. Be conservative and grounded in visual evidence.
6. If material is uncertain, provide the most likely hypothesis rather than leaving it empty.
7. The output will be used later to derive physical properties such as density, mass, friction, etc., so the description should be useful for that purpose.

SEMANTIC TAG RULES:
- Use lowercase snake_case.
- Prefer specific tags when possible, e.g.:
  - ceramic_mug
  - plastic_storage_box
  - wooden_chair
  - metal_tool
  - glass_bottle
  - fabric_soft_toy
  - electronic_device
- If uncertain, use a broader but still useful tag such as:
  - container
  - kitchenware
  - hand_tool
  - furniture
  - toy
  - household_item

OUTPUT JSON SCHEMA:
{{
  "object_name": string,
  "semantic_tag": string,
  "description": string,
  "primary_materials": [string, ...],
  "material_confidence": float,
  "confidence": float
}}

FIELD GUIDANCE:
- object_name: canonical short name for the object
- semantic_tag: concise semantic class tag
- description: 1-3 sentences; mention likely material and structural/functional semantics
- primary_materials: list of likely materials in descending plausibility
- material_confidence: confidence in material estimate, from 0.0 to 1.0
- confidence: confidence in the semantic classification overall, from 0.0 to 1.0

CRITICAL RULES:
- OUTPUT JSON ONLY.
- No markdown.
- No extra text.
- Do not return null unless the object is completely unrecognizable.
"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
    )
    return extract_json(get_chat_completion_content(resp))


def export_final_mesh(mesh, name, out_dir: Path):
    out_dir = out_dir.resolve()
    out_dir.mkdir(exist_ok=True, parents=True)
    bounds = mesh.bounds
    minb = bounds[0]
    maxb = bounds[1]
    bottom_center = np.array(
        [(minb[0] + maxb[0]) / 2.0, (minb[1] + maxb[1]) / 2.0, minb[2]], dtype=float
    )
    T_trans = np.eye(4)
    T_trans[:3, 3] = -bottom_center
    mesh.apply_transform(T_trans)
    out_path = out_dir / f"{name}_simready.obj"
    out_path = out_path.resolve()

    print(f"Exporting final mesh to: {out_path} (bottom-face center moved to origin)")
    mesh.export(out_path)

    return str(out_path)


def delete_rendered_pngs(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return

    patterns = [
        "view_*.png",
        "*_flipped.png",
    ]

    for pattern in patterns:
        for p in output_dir.glob(pattern):
            p.unlink()


def process_mesh(file, name=None, extra_text="", out_dir="renders", res=1024):
    if isinstance(file, (str, Path)):
        file = Path(file).resolve()
        name = file.stem
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)
    mesh = init_pose(file)

    images_first = render_views(
        mesh, diagonal_views + cardinal_views + up_down_views, out_dir, res
    )
    category_res = ask_mllm_detect_and_classify(images_first, extra_text=extra_text)
    print(category_res)
    category = int(category_res.get("category", 0))
    object_name = str(category_res.get("detected_object", "None"))
    main_surface = str(category_res.get("main_surface", "None"))
    orientation_requirement = str(category_res.get("orientation_requirement", "None"))

    if category == 0:
        pass

    elif category == 1:
        images_for_1_1 = render_views(mesh, side_profile, out_dir, res)
        side_profile_result = ask_llm_full_side_profile(object_name, images_for_1_1)
        print(side_profile_result)
        side_profile_result = side_profile_result.get("full_side_profile_image", "B")
        if side_profile_result == "B":
            upright_img = render_views(mesh, front_views, out_dir, res)
            upright_img = upright_img[0]["path"]
            flipped_path = str(
                Path(upright_img).with_name(
                    Path(upright_img).stem + f"_180_flipped.png"
                )
            )
            rotate_image_deg(upright_img, 180, flipped_path)
            upright_result = ask_llm_upright_2a1(object_name, upright_img, flipped_path)
            print(upright_result)
            try:
                upright_choice = upright_result.get("upright_image", "A")
            except Exception:
                upright_choice = "A"
            if upright_choice == "B":
                x_flip = rot_x(180)
                apply_rotations(mesh, x_flip)

        elif side_profile_result == "A":
            upright_img = render_views(
                mesh,
                [("view_from_up_to_bottom", np.array([0.5, 0.5, 2.2], dtype=float))],
                out_dir,
                res,
            )
            upright_img = upright_img[0]["path"]
            rotated_imgs = []
            rotated_imgs.append(upright_img)
            rotate_deg = [90, 180, 270]
            for deg in rotate_deg:
                flipped_path = str(
                    Path(upright_img).with_name(
                        Path(upright_img).stem + f"_{deg}_flipped.png"
                    )
                )
                rotated_imgs.append(rotate_image_deg(upright_img, deg, flipped_path))
            side_rotation_result = ask_llm_upright_rotation(object_name, rotated_imgs)
            side_rotation_result = side_rotation_result.get("upright_index", 0)
            print("side rotation is", side_rotation_result)
            if side_rotation_result == 0:
                pass
            elif side_rotation_result == 1:
                side_r = rot_z(90)
                apply_rotations(mesh, side_r)
            elif side_rotation_result == 2:
                side_r = rot_z(180)
                apply_rotations(mesh, side_r)
            elif side_rotation_result == 3:
                side_r = rot_z(270)
                apply_rotations(mesh, side_r)
            else:
                raise ValueError("no upright index choosen")
            side_r = rot_y(90)
            apply_rotations(mesh, side_r)
        else:
            raise ValueError("no side profil choosen")

    elif category == 2:
        images_for_2_1 = render_views(
            mesh, cardinal_views + up_down_views, out_dir, res
        )
        result_main_surface = ask_mllm_primary_surface(
            images_for_2_1, object_name, main_surface, orientation_requirement
        )
        print(result_main_surface)
        primary_view = result_main_surface.get("primary_surface_view", "None")

        if orientation_requirement == "face_forward":

            if primary_view in [i[0] for i in cardinal_views]:
                if primary_view == "view_from_front":
                    print("no need to rotate round z")
                elif primary_view == "view_from_left":  # left
                    R = rot_z(90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_right":  # right
                    R = rot_z(-90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_back":  # back
                    R = rot_z(180)
                    apply_rotations(mesh, R)

                else:
                    raise ValueError("unknow views")

            elif primary_view in [i[0] for i in up_down_views]:
                if primary_view == "view_from_up_to_bottom":
                    R = rot_y(90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_bottom_to_up":
                    R = rot_y(-90)
                    apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            else:
                raise ValueError("unknow views")
            normalize_to_unit_cube(mesh)
            upright_img = render_views(mesh, front_views, out_dir, res)
            upright_img = upright_img[0]["path"]
            rotated_imgs = []
            rotated_imgs.append(upright_img)
            rotate_deg = [90, 180, 270]
            for deg in rotate_deg:
                flipped_path = str(
                    Path(upright_img).with_name(
                        Path(upright_img).stem + f"_{deg}_flipped.png"
                    )
                )
                rotated_imgs.append(rotate_image_deg(upright_img, deg, flipped_path))
            result = ask_llm_upright_rotation(object_name, rotated_imgs)
            print(result)
            upright_result = result.get("upright_index", 0)
            if upright_result == 0:
                pass
            elif upright_result == 1:
                upright_deg = rot_x(90)
                apply_rotations(mesh, upright_deg)
            elif upright_result == 2:
                upright_deg = rot_x(180)
                apply_rotations(mesh, upright_deg)
            elif upright_result == 3:
                upright_deg = rot_x(-90)
                apply_rotations(mesh, upright_deg)
            else:
                raise ValueError("upright index unknow")

        elif orientation_requirement == "face_up":

            if primary_view in [i[0] for i in cardinal_views]:
                if primary_view == "view_from_front":
                    R = rot_y(-90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_left":
                    R = rot_x(-90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_right":
                    R = rot_x(90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_back":
                    R = rot_y(90)
                    apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            elif primary_view in [i[0] for i in up_down_views]:
                if primary_view == "view_from_up_to_bottom":
                    print("no need to rotate")
                elif primary_view == "view_from_bottom_to_up":
                    R = rot_x(180)
                    apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            else:
                raise ValueError("unknow views")
            normalize_to_unit_cube(mesh)
            upright_img = render_views(mesh, up_views, out_dir, res)
            upright_img = upright_img[0]["path"]
            rotated_imgs = []
            rotated_imgs.append(upright_img)
            rotate_deg = [90, 180, 270]
            for deg in rotate_deg:
                flipped_path = str(
                    Path(upright_img).with_name(
                        Path(upright_img).stem + f"_{deg}_flipped.png"
                    )
                )
                rotated_imgs.append(rotate_image_deg(upright_img, deg, flipped_path))
            result = ask_llm_upright_rotation(object_name, rotated_imgs)
            print(result)
            upright_result = result.get("upright_index", 0)
            if upright_result == 0:
                pass
            elif upright_result == 1:
                upright_deg = rot_z(90)
                apply_rotations(mesh, upright_deg)
            elif upright_result == 2:
                upright_deg = rot_z(180)
                apply_rotations(mesh, upright_deg)
            elif upright_result == 3:
                upright_deg = rot_z(-90)
                apply_rotations(mesh, upright_deg)
            else:
                raise ValueError("upright index unknow")

        elif orientation_requirement == "face_down":
            if primary_view in [i[0] for i in cardinal_views]:
                if primary_view == "view_from_front":
                    R = rot_y(90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_left":
                    R = rot_x(90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_right":
                    R = rot_x(-90)
                    apply_rotations(mesh, R)
                elif primary_view == "view_from_back":
                    R = rot_y(-90)
                    apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            elif primary_view in [i[0] for i in up_down_views]:
                if primary_view == "view_from_up_to_bottom":
                    print("no need to rotate")
                elif primary_view == "view_from_bottom_to_up":
                    R = rot_x(180)
                    apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            else:
                raise ValueError("unknow views")
            normalize_to_unit_cube(mesh)
            upright_img = render_views(mesh, down_views, out_dir, res)
            upright_img = upright_img[0]["path"]
            rotated_imgs = []
            rotated_imgs.append(upright_img)
            rotate_deg = [90, 180, 270]
            for deg in rotate_deg:
                flipped_path = str(
                    Path(upright_img).with_name(
                        Path(upright_img).stem + f"_{deg}_flipped.png"
                    )
                )
                rotated_imgs.append(rotate_image_deg(upright_img, deg, flipped_path))
            result = ask_llm_upright_rotation(object_name, rotated_imgs)
            print(result)
            upright_result = result.get("upright_index", 0)
            if upright_result == 0:
                apply_rotations(mesh, upright_deg)
            elif upright_result == 1:
                upright_deg = rot_z(90)
                apply_rotations(mesh, upright_deg)
            elif upright_result == 2:
                upright_deg = rot_z(180)
                pass
            elif upright_result == 3:
                upright_deg = rot_z(-90)
                apply_rotations(mesh, upright_deg)
            else:
                raise ValueError("upright index unknow")

        else:
            raise ValueError("unknow orientationrequirement")

    else:
        raise ValueError()

    # TODO: Add alignment analysis to avoid tilted outputs.

    normalize_to_unit_cube(mesh)

    current_bbox_dims = get_aabb_dims(mesh)

    dimension_views = render_views(
        mesh, diagonal_views + cardinal_views + up_down_views, out_dir, res
    )

    dimension_result = ask_llm_dimension(
        object_name=object_name,
        img_paths=dimension_views,
        user_text_hint=extra_text,
        current_bbox_dims=current_bbox_dims,
    )
    print(dimension_result)

    target_dims = dimension_result.get("dimensions_m", None)
    if target_dims is None:
        raise ValueError("LLM failed to return dimensions_m")

    mesh, uniform_scale = scale_mesh_uniform_to_dimensions(
        mesh=mesh,
        target_dims=target_dims,
        current_dims=current_bbox_dims,
    )

    print(
        {
            "uniform_scale": uniform_scale,
            "current_bbox_dims": current_bbox_dims,
            "target_dims_m": target_dims,
        }
    )

    out_path = export_final_mesh(mesh, name, out_dir)

    semantics_result = ask_llm_semantics_info(
        object_name=object_name,
        img_paths=dimension_views,
        user_text_hint=extra_text,
    )
    return {
        "Path": out_path,
        "uniform_scale": uniform_scale,
        "target_dims_m": target_dims,
        "semantics_result": semantics_result,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        required=True,
        help="Path to input 3D mesh file (absolute path supported)",
    )
    ap.add_argument(
        "--extra_text",
        default="",
        help="Text description for your object, mainly describe the dimension and category",
    )
    ap.add_argument(
        "--out_dir",
        default="renders",
        help="Output directory (absolute path supported)",
    )
    ap.add_argument(
        "--name",
        default="test",
        help="Output directory (absolute path supported)",
    )
    ap.add_argument("--res", type=int, default=1024, help="Rendered image resolution")
    args = ap.parse_args()
    args.file = Path(args.file).resolve()
    args.out_dir = Path(args.out_dir).resolve()
    if not args.file.exists():
        print(f"Error: Input file does not exist - {args.file}")
        exit(1)

    process_mesh(args.file, args.name, args.extra_text, args.out_dir, args.res)


if __name__ == "__main__":
    main()
