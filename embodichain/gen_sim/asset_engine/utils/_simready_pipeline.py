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


"""SimReady decisions shared by the Blender and trimesh geometry adapters."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from embodichain.gen_sim.services.runtime import GenSimServices
from embodichain.gen_sim.services.compat import chat_from_legacy_messages

__all__: list[str] = []


def get_chat_completion_content(resp: Any) -> str:
    """Return chat completion message content with a clearer URL hint on bad responses."""

    if isinstance(resp, str):
        return resp
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


def ask_mllm_primary_surface(
    views_data,
    object_name="None",
    main_surface="None",
    orientation_requirement="None",
    extra_text="",
    *,
    services: GenSimServices | None = None,
):

    instruction_text = f"""
    You are a single-purpose multimodal classifier. You will be given 6 images of a single object, rendered from different views. Your task is to identify **the image that best shows the object's forward-facing primary use surface**, defined as the surface that:

    - Carries the object's core function (viewing, operating, aiming, serving, pressing, interacting, etc.)
    - Faces the human user or line-of-sight in normal use
    - Is unique and human-accessible (not a symmetrical or bottom/support surface)
    - Should prioritize the **front-facing view**, even if other angles also partially show it.

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
    resp = chat_from_legacy_messages(
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
        services=services,
        role="asset_orientation",
    )
    raw = get_chat_completion_content(resp)
    return extract_json(raw)


def ask_llm_upright_2a1(
    object_name,
    upright_img_path,
    flipped_img_path,
    *,
    services: GenSimServices | None = None,
):
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

    resp = chat_from_legacy_messages(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
        services=services,
        role="asset_orientation",
    )
    return extract_json(get_chat_completion_content(resp))


def ask_llm_full_side_profile(
    object_name, views_data, *, services: GenSimServices | None = None
):
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

    resp = chat_from_legacy_messages(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
        services=services,
        role="asset_orientation",
    )
    return extract_json(get_chat_completion_content(resp))


def ask_llm_upright_rotation(
    object_name, rotated_imgs_paths, *, services: GenSimServices | None = None
):
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
    resp = chat_from_legacy_messages(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
        services=services,
        role="asset_orientation",
    )
    return extract_json(get_chat_completion_content(resp))


def ask_llm_dimension(
    object_name,
    img_paths,
    user_text_hint,
    current_bbox_dims,
    *,
    services: GenSimServices | None = None,
):

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

    resp = chat_from_legacy_messages(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
        services=services,
        role="asset_dimensions",
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


def dims_dict_to_xyz(dims: dict):

    return np.array(
        [
            float(dims.get("depth", np.nan)),
            float(dims.get("width", np.nan)),
            float(dims.get("height", np.nan)),
        ],
        dtype=float,
    )


def ask_llm_semantics_info(
    object_name, img_paths, user_text_hint="", *, services: GenSimServices | None = None
):

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

    resp = chat_from_legacy_messages(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *imgs_payload],
            }
        ],
        temperature=0.0,
        services=services,
        role="asset_semantics",
    )
    return extract_json(get_chat_completion_content(resp))


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


def _process_mesh(
    file,
    name=None,
    extra_text="",
    out_dir="renders",
    res=1024,
    *,
    backend: ModuleType,
    services: GenSimServices | None = None,
):
    if isinstance(file, (str, Path)):
        file = Path(file).resolve()
        name = file.stem
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)
    mesh = backend.init_pose_simple(file)
    images_first = backend.render_views(
        mesh,
        backend.diagonal_views + backend.cardinal_views + backend.up_down_views,
        out_dir,
        res,
    )
    category_res = backend.ask_mllm_detect_and_classify(
        images_first, extra_text=extra_text, services=services
    )
    print(category_res)
    category = int(category_res.get("category", 0))
    object_name = str(category_res.get("detected_object", "None"))
    main_surface = str(category_res.get("main_surface", "None"))
    orientation_requirement = str(category_res.get("orientation_requirement", "None"))

    if category == 0:
        pass

    elif category == 1:
        images_for_1_1 = backend.render_views(mesh, backend.side_profile, out_dir, res)
        side_profile_result = backend.ask_llm_full_side_profile(
            object_name, images_for_1_1, services=services
        )
        print(side_profile_result)
        side_profile_result = side_profile_result.get("full_side_profile_image", "B")
        if side_profile_result == "B":
            upright_img = backend.render_views(mesh, backend.front_views, out_dir, res)
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
                rotated_imgs.append(
                    backend.rotate_image_deg(upright_img, deg, flipped_path)
                )
            side_rotation_result = backend.ask_llm_upright_rotation(
                object_name, rotated_imgs, services=services
            )
            side_rotation_result = side_rotation_result.get("upright_index", 0)
            print("side rotation is", side_rotation_result)
            if side_rotation_result == 0:
                pass
            elif side_rotation_result == 1:
                side_r = backend.rot_x(90)
                backend.apply_rotations(mesh, side_r)
            elif side_rotation_result == 2:
                side_r = backend.rot_x(180)
                backend.apply_rotations(mesh, side_r)
            elif side_rotation_result == 3:
                side_r = backend.rot_x(270)
                backend.apply_rotations(mesh, side_r)
            else:
                raise ValueError("no upright index choosen")

        elif side_profile_result == "A":
            upright_img = backend.render_views(
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
                rotated_imgs.append(
                    backend.rotate_image_deg(upright_img, deg, flipped_path)
                )
            side_rotation_result = backend.ask_llm_upright_rotation(
                object_name, rotated_imgs, services=services
            )
            side_rotation_result = side_rotation_result.get("upright_index", 0)
            print("side rotation is", side_rotation_result)
            if side_rotation_result == 0:
                pass
            elif side_rotation_result == 1:
                side_r = backend.rot_z(90)
                backend.apply_rotations(mesh, side_r)
            elif side_rotation_result == 2:
                side_r = backend.rot_z(180)
                backend.apply_rotations(mesh, side_r)
            elif side_rotation_result == 3:
                side_r = backend.rot_z(270)
                backend.apply_rotations(mesh, side_r)
            else:
                raise ValueError("no upright index choosen")
            side_r = backend.rot_y(90)
            backend.apply_rotations(mesh, side_r)
        else:
            raise ValueError("no side profil choosen")

    elif category == 2:
        images_for_2_1 = backend.render_views(
            mesh, backend.cardinal_views + backend.up_down_views, out_dir, res
        )
        result_main_surface = backend.ask_mllm_primary_surface(
            images_for_2_1,
            object_name,
            main_surface,
            orientation_requirement,
            services=services,
        )
        print(result_main_surface)
        primary_view = result_main_surface.get("primary_surface_view", "None")

        if orientation_requirement == "face_forward":

            if primary_view in [i[0] for i in backend.cardinal_views]:
                if primary_view == "view_from_front":
                    print("no need to rotate round z")
                elif primary_view == "view_from_left":  # left
                    R = backend.rot_z(90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_right":  # right
                    R = backend.rot_z(-90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_back":  # back
                    R = backend.rot_z(180)
                    backend.apply_rotations(mesh, R)

                else:
                    raise ValueError("unknow views")

            elif primary_view in [i[0] for i in backend.up_down_views]:
                if primary_view == "view_from_up_to_bottom":
                    R = backend.rot_y(90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_bottom_to_up":
                    R = backend.rot_y(-90)
                    backend.apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            else:
                raise ValueError("unknow views")
            backend.normalize_to_unit_cube(mesh)
            upright_img = backend.render_views(mesh, backend.front_views, out_dir, res)
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
                rotated_imgs.append(
                    backend.rotate_image_deg(upright_img, deg, flipped_path)
                )
            result = backend.ask_llm_upright_rotation(
                object_name, rotated_imgs, services=services
            )
            print(result)
            upright_result = result.get("upright_index", 0)
            if upright_result == 0:
                pass
            elif upright_result == 1:
                upright_deg = backend.rot_x(90)
                backend.apply_rotations(mesh, upright_deg)
            elif upright_result == 2:
                upright_deg = backend.rot_x(180)
                backend.apply_rotations(mesh, upright_deg)
            elif upright_result == 3:
                upright_deg = backend.rot_x(-90)
                backend.apply_rotations(mesh, upright_deg)
            else:
                raise ValueError("upright index unknow")

        elif orientation_requirement == "face_up":

            if primary_view in [i[0] for i in backend.cardinal_views]:
                if primary_view == "view_from_front":
                    R = backend.rot_y(-90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_left":
                    R = backend.rot_x(-90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_right":
                    R = backend.rot_x(90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_back":
                    R = backend.rot_y(90)
                    backend.apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            elif primary_view in [i[0] for i in backend.up_down_views]:
                if primary_view == "view_from_up_to_bottom":
                    print("no need to rotate")
                elif primary_view == "view_from_bottom_to_up":
                    R = backend.rot_x(180)
                    backend.apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            else:
                raise ValueError("unknow views")
            backend.normalize_to_unit_cube(mesh)
            upright_img = backend.render_views(mesh, backend.up_views, out_dir, res)
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
                rotated_imgs.append(
                    backend.rotate_image_deg(upright_img, deg, flipped_path)
                )
            result = backend.ask_llm_upright_rotation(
                object_name, rotated_imgs, services=services
            )
            print(result)
            upright_result = result.get("upright_index", 0)
            if upright_result == 0:
                pass
            elif upright_result == 1:
                upright_deg = backend.rot_z(90)
                backend.apply_rotations(mesh, upright_deg)
            elif upright_result == 2:
                upright_deg = backend.rot_z(180)
                backend.apply_rotations(mesh, upright_deg)
            elif upright_result == 3:
                upright_deg = backend.rot_z(-90)
                backend.apply_rotations(mesh, upright_deg)
            else:
                raise ValueError("upright index unknow")

        elif orientation_requirement == "face_down":
            if primary_view in [i[0] for i in backend.cardinal_views]:
                if primary_view == "view_from_front":
                    R = backend.rot_y(90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_left":
                    R = backend.rot_x(90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_right":
                    R = backend.rot_x(-90)
                    backend.apply_rotations(mesh, R)
                elif primary_view == "view_from_back":
                    R = backend.rot_y(-90)
                    backend.apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            elif primary_view in [i[0] for i in backend.up_down_views]:
                if primary_view == "view_from_up_to_bottom":
                    print("no need to rotate")
                elif primary_view == "view_from_bottom_to_up":
                    R = backend.rot_x(180)
                    backend.apply_rotations(mesh, R)
                else:
                    raise ValueError("unknow views")

            else:
                raise ValueError("unknow views")
            backend.normalize_to_unit_cube(mesh)
            upright_img = backend.render_views(mesh, backend.down_views, out_dir, res)
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
                rotated_imgs.append(
                    backend.rotate_image_deg(upright_img, deg, flipped_path)
                )
            result = backend.ask_llm_upright_rotation(
                object_name, rotated_imgs, services=services
            )
            print(result)
            upright_result = result.get("upright_index", 0)
            if upright_result == 0:
                backend.apply_rotations(mesh, upright_deg)
            elif upright_result == 1:
                upright_deg = backend.rot_z(90)
                backend.apply_rotations(mesh, upright_deg)
            elif upright_result == 2:
                upright_deg = backend.rot_z(180)
                pass
            elif upright_result == 3:
                upright_deg = backend.rot_z(-90)
                backend.apply_rotations(mesh, upright_deg)
            else:
                raise ValueError("upright index unknow")

        else:
            raise ValueError("unknow orientationrequirement")

    else:
        raise ValueError()

    # TODO: Add alignment analysis to avoid tilted outputs.

    backend.normalize_to_unit_cube(mesh)

    current_bbox_dims = backend.get_aabb_dims(mesh)

    dimension_views = backend.render_views(
        mesh,
        backend.diagonal_views + backend.cardinal_views + backend.up_down_views,
        out_dir,
        res,
    )

    dimension_result = backend.ask_llm_dimension(
        object_name=object_name,
        img_paths=dimension_views,
        user_text_hint=extra_text,
        current_bbox_dims=current_bbox_dims,
        services=services,
    )
    print(dimension_result)

    target_dims = dimension_result.get("dimensions_m", None)
    if target_dims is None:
        raise ValueError("LLM failed to return dimensions_m")

    mesh, uniform_scale = backend.scale_mesh_uniform_to_dimensions(
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

    out_path = backend.export_final_mesh(mesh, name, out_dir)

    semantics_result = backend.ask_llm_semantics_info(
        object_name=object_name,
        img_paths=dimension_views,
        user_text_hint=extra_text,
        services=services,
    )
    return {
        "Path": out_path,
        "uniform_scale": uniform_scale,
        "target_dims_m": target_dims,
        "semantics_result": semantics_result,
    }


def _main(*, backend: ModuleType, services: GenSimServices | None = None):
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

    backend.process_mesh(
        args.file, args.name, args.extra_text, args.out_dir, args.res, services=services
    )
