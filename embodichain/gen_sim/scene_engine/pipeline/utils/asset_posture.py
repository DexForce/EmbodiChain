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

from collections.abc import Callable
import json
from pathlib import Path
from typing import TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.gen_sim._blender_runtime import _run_blender_job
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
    _strip_json_code_fence,
)
from embodichain.gen_sim.services.compat import VisionClient, vision_for_role

__all__ = ["review_asset_posture"]

_CRITIC_PROMPT = """Review the resting posture of the isolated 3D object against the requested posture.
The render uses Y-up and places the lowest point on a neutral display surface.
Use the object's whole body, contact face, and functional up/down features as evidence.
An upward-facing feature alone does not establish the correct posture. Accept equivalent
resting faces and ignore yaw, size, the scene position, and the preview grounding.
For ordinary stable posture without a specific instruction, keep an already plausible pose.
Return JSON only with exactly {"accepted": true or false, "reason": "short visual reason"}."""

_ACTOR_PROMPT = """Propose one rotation that gives the isolated 3D object the requested resting posture.
The image shows the original asset. Use the latest review feedback and avoid every previously
tried rotation. This is a right-handed Y-up frame: X goes right, +Y goes up, and +Z faces the
front. Use the fixed preview axes. Return an absolute rotation of the original geometry
about horizontal X or Z by -90, 90, or 180 degrees. Prioritize the resting face and functional
up/down direction and leave scene yaw to the caller.
Return JSON only with exactly {"axis": "x" or "z", "angle_degrees": -90, 90, or 180}."""

_MAX_PROPOSALS = 3
_MAX_JSON_ATTEMPTS = 2
_Reply = TypeVar("_Reply")


def review_asset_posture(
    *,
    glb_path: Path,
    description: str,
    pose_description: str | None,
    vlm_client: VisionClient,
    debug_output_root: Path,
) -> np.ndarray | None:
    """Return an approved local-to-rest rotation without changing source geometry.

    The matrix is a proper 3×3 rotation in the asset's Y-up frame. The critic
    may accept the identity posture. Exhausting the proposal budget without
    acceptance returns ``None`` so the caller can retain its original pose.
    """
    debug_output_root.mkdir(parents=True, exist_ok=True)
    client = vision_for_role(vlm_client, "scene_asset_orientation")
    task = {
        "object_description": description,
        "requested_posture": pose_description
        or "Use the object's usual stable resting posture.",
    }
    initial = debug_output_root / "original.png"
    rotation = np.eye(3)
    _render(glb_path, rotation, initial)
    accepted, reason = _query(
        client,
        _CRITIC_PROMPT,
        task,
        initial,
        debug_output_root / "original_review.json",
        _parse_review,
    )
    if accepted:
        return rotation

    tried: list[tuple[str, int]] = []
    for index in range(1, _MAX_PROPOSALS + 1):
        prefix = f"candidate_{index}"
        proposal = _query(
            client,
            _ACTOR_PROMPT,
            {**task, "latest_review": reason, "tried_rotations": tried},
            initial,
            debug_output_root / f"{prefix}_proposal.json",
            _parse_proposal,
        )
        if proposal in tried:
            reason = f"The rotation {proposal} already failed review. Propose a different rotation."
            continue
        tried.append(proposal)
        axis, angle = proposal
        rotation = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
        candidate = debug_output_root / f"{prefix}.png"
        _render(glb_path, rotation, candidate)
        accepted, reason = _query(
            client,
            _CRITIC_PROMPT,
            task,
            candidate,
            debug_output_root / f"{prefix}_review.json",
            _parse_review,
        )
        if accepted:
            return rotation
    return None


def _render(glb_path: Path, rotation: np.ndarray, output_path: Path) -> None:
    _run_blender_job(
        "asset_posture",
        {
            "glb_path": str(glb_path.expanduser().resolve()),
            "rotation_y_up": rotation.tolist(),
            "output_path": str(output_path.expanduser().resolve()),
            "resolution": 384,
        },
        timeout=60.0,
    )
    if not output_path.is_file():
        raise FileNotFoundError(
            f"Asset posture preview was not rendered: {output_path}"
        )


def _query(
    client: VisionClient,
    system_prompt: str,
    task: dict[str, object],
    preview_path: Path,
    audit_path: Path,
    parse: Callable[[str], _Reply],
) -> _Reply:
    user_prompt = json.dumps(task, ensure_ascii=False)
    audit: dict[str, object] = {
        "system_prompt": system_prompt,
        "user_prompt": task,
        "image_path": str(preview_path),
    }
    responses: list[str] = []
    for _ in range(_MAX_JSON_ATTEMPTS):
        raw_response = client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=preview_path,
        )
        responses.append(raw_response)
        try:
            reply = parse(raw_response)
        except ValueError as exc:
            audit["last_validation_error"] = str(exc)
        else:
            _write_audit(audit_path, {**audit, "responses": responses, "reply": reply})
            return reply

    _write_audit(audit_path, {**audit, "responses": responses})
    raise ValueError(
        f"Invalid asset posture response: {audit['last_validation_error']}"
    )


def _write_audit(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _parse_review(text: str) -> tuple[bool, str]:
    payload = json.loads(_strip_json_code_fence(text))
    if not isinstance(payload, dict) or set(payload) != {"accepted", "reason"}:
        raise ValueError("Asset posture review requires exactly accepted and reason.")
    accepted = payload["accepted"]
    reason = payload["reason"]
    if type(accepted) is not bool or not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            "Asset posture review requires a boolean and a nonblank reason."
        )
    return accepted, reason.strip()


def _parse_proposal(text: str) -> tuple[str, int]:
    payload = json.loads(_strip_json_code_fence(text))
    if not isinstance(payload, dict) or set(payload) != {"axis", "angle_degrees"}:
        raise ValueError(
            "Asset posture proposal requires exactly axis and angle_degrees."
        )
    axis = payload["axis"]
    angle = payload["angle_degrees"]
    if axis not in ("x", "z") or type(angle) is not int or angle not in (-90, 90, 180):
        raise ValueError(
            "Asset posture proposal requires X or Z and -90, 90, or 180 degrees."
        )
    return axis, angle
