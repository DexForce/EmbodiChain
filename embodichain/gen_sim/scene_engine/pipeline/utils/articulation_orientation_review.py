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

"""Select a native articulation orientation from the source observation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from embodichain.gen_sim._blender_runtime import _run_blender_job

from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    _load_label_font_at_size,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
    _strip_json_code_fence,
)
from embodichain.gen_sim.services.compat import vision_for_role
from embodichain.gen_sim.services.errors import ServiceError

if TYPE_CHECKING:
    from embodichain.gen_sim.services.compat import VisionClient
    from embodichain.lab.sim.scene.articulation_geometry import ArticulationGeometry
    from embodichain.lab.sim.spawn.source_geometry import UsdSourceSnapshot

__all__ = ["review_articulation_orientation"]

_YAWS = (0, 90, 180, 270)
_VIEW_ELEVATION = 25
_TILE_SIZE = 336
_BACKGROUND = (245, 247, 250)
_MAX_ATTEMPTS = 2
_SYSTEM_PROMPT = """Align the native articulated asset to the real object's approximate facing direction in the original image crop at left. Each candidate shows the same native asset viewed from a fixed front-elevated camera with the labeled yaw offset already applied to its scene placement. Compare the direction its functional front faces, which sides are visible, and where recognizable features such as an opening, door, handle, hinge, or control panel sit in the image. Allow for different camera perspective, occlusion, joint opening, color, and shape. Choose the closest of the four candidates even when the alignment is approximate. Return null only if the original crop is unusable for observing the object. Give a short visual reason for the chosen direction. Return one JSON object with exactly the keys "yaw_degrees" and "reason". The yaw is the selected candidate label, one of 0, 90, 180, 270, or null."""


def review_articulation_orientation(
    *,
    reference_image: Path,
    source: UsdSourceSnapshot,
    geometry: ArticulationGeometry,
    native_to_upright: np.ndarray,
    proxy_yaw_degrees: float,
    category: str,
    description: str,
    vision_client: VisionClient,
    output_root: Path,
) -> tuple[int | None, str]:
    """Return a Z-up native yaw offset from the scene proxy and its reason.

    Each preview uses ``proxy_yaw_degrees + offset`` after
    ``native_to_upright``. The original source crop retains its view and aspect
    ratio. A null result lets the caller choose a fallback. Persist the preview,
    exact input prompts, and every response and validation result.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    preview_path = output_root / "orientation_candidates.png"
    _render_candidates(
        reference_image, source, geometry, native_to_upright, proxy_yaw_degrees
    ).save(preview_path)
    audit_path = output_root / "orientation_review.json"
    user_prompt = json.dumps(
        {"category": category, "description": description}, ensure_ascii=False
    )
    attempts: list[dict[str, str]] = []
    audit: dict[str, object] = {
        "system_prompt": _SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "image_path": str(preview_path),
        "reference_image": str(reference_image),
        "proxy_yaw_degrees": proxy_yaw_degrees,
        "candidate_world_yaw_degrees": {
            str(yaw): (proxy_yaw_degrees + yaw) % 360 for yaw in _YAWS
        },
        "renderer": "blender_cycles",
        "camera": {"azimuth_degrees": -90, "elevation_degrees": _VIEW_ELEVATION},
        "attempts": attempts,
    }
    client = vision_for_role(vision_client, "scene_asset_orientation")
    yaw = None
    reason = "The vision model did not return a valid orientation decision."
    status = "invalid_response"
    for _ in range(_MAX_ATTEMPTS):
        try:
            raw = client.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_path=preview_path,
            )
        except ServiceError as exc:
            audit["service_error"] = exc.code
            audit_path.write_text(
                json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise
        attempt = {"raw_response": raw}
        attempts.append(attempt)
        try:
            yaw, reason = _parse_review(raw)
        except ValueError as exc:
            attempt["validation_error"] = str(exc)
        else:
            attempt["validation"] = "valid"
            status = "unusable_reference" if yaw is None else "selected"
            break
    audit["decision"] = {"status": status, "yaw_degrees": yaw, "reason": reason}
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return yaw, reason


def _parse_review(raw: str) -> tuple[int | None, str]:
    result = json.loads(_strip_json_code_fence(raw))
    if not isinstance(result, dict) or set(result) != {"yaw_degrees", "reason"}:
        raise ValueError("Expected an object with yaw_degrees and reason.")
    yaw = result["yaw_degrees"]
    if yaw is not None and (type(yaw) is not int or yaw not in _YAWS):
        raise ValueError("Expected a candidate yaw in degrees or null.")
    reason = result["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Expected a nonempty review reason.")
    return yaw, reason.strip()


def _render_candidates(
    reference_image: Path,
    source: UsdSourceSnapshot,
    geometry: ArticulationGeometry,
    native_to_upright: np.ndarray,
    proxy_yaw_degrees: float,
) -> Image.Image:
    with TemporaryDirectory(prefix="embodichain-articulation-views-") as directory:
        root = Path(directory)
        preview_usd = root / "preview.usda"
        _write_preview_stage(source, geometry, preview_usd)
        tiles = [root / f"{yaw}.png" for yaw in _YAWS]
        _run_blender_job(
            "articulation_orientation",
            {
                "usdc_path": str(preview_usd),
                "native_to_upright": native_to_upright.tolist(),
                "proxy_yaw_degrees": proxy_yaw_degrees,
                "output_paths": [str(tile) for tile in tiles],
                "resolution": _TILE_SIZE,
            },
            timeout=180.0,
        )
        return _compose_candidates(reference_image, tiles)


def _write_preview_stage(
    source: UsdSourceSnapshot, geometry: ArticulationGeometry, path: Path
) -> None:
    """Reference the original look while posing links at the measured reset state."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim(source.stage.GetDefaultPrim().GetPath())
    root.GetReferences().AddReference(source.path.as_posix())
    stage.SetDefaultPrim(root)
    prims = list(Usd.PrimRange(root))
    meshes = [UsdGeom.Mesh(prim) for prim in prims if prim.IsA(UsdGeom.Mesh)]
    colliders = [
        mesh for mesh in meshes if mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    ]
    has_visual_mesh = any(
        mesh.ComputeVisibility() != UsdGeom.Tokens.invisible
        and mesh.ComputePurpose() not in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy)
        for mesh in meshes
    )
    for prim in prims:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            xform = UsdGeom.Xformable(prim)
            pose = geometry.link_poses[prim.GetName()]
            xform.MakeMatrixXform().Set(Gf.Matrix4d(pose.T.tolist()))
            xform.SetResetXformStack(True)
    if not has_visual_mesh:
        for collider in colliders:
            collider.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            collider.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
    source.assert_current()
    stage.GetRootLayer().Save()


def _compose_candidates(reference_image: Path, tiles: list[Path]) -> Image.Image:
    labels = ("Original image crop", *(f"USDC offset {yaw}°" for yaw in _YAWS))
    header_height = 42
    sheet = Image.new(
        "RGB",
        (_TILE_SIZE * len(labels), header_height + _TILE_SIZE),
        _BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    font = _load_label_font_at_size(23)
    with Image.open(reference_image) as source:
        crop = ImageOps.contain(
            ImageOps.exif_transpose(source).convert("RGBA"),
            (_TILE_SIZE, _TILE_SIZE),
            method=Image.Resampling.LANCZOS,
        )
    sheet.paste(
        crop,
        (
            (_TILE_SIZE - crop.width) // 2,
            header_height + (_TILE_SIZE - crop.height) // 2,
        ),
        crop.getchannel("A"),
    )
    for column, label in enumerate(labels):
        draw.text(
            ((column + 0.5) * _TILE_SIZE, header_height / 2),
            label,
            fill="#192330",
            font=font,
            anchor="mm",
        )
    for column, tile in enumerate(tiles, start=1):
        with Image.open(tile) as view:
            sheet.paste(view.convert("RGB"), (column * _TILE_SIZE, header_height))
    return sheet
