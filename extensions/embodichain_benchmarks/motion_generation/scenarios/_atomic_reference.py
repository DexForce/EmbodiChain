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

"""Recover independent reference witnesses inside declared arrival tolerances."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from embodichain.compute.kinematics import pose_error
from .atomic_case_utils import canonical_case_qpos

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from ..config import ProtocolCfg

__all__: list[str] = []


def _recover_reference_ik(
    robot: Robot,
    target: torch.Tensor,
    seed: torch.Tensor,
    success: torch.Tensor,
    solved: torch.Tensor,
    *,
    control_part: str,
    protocol: ProtocolCfg,
    env_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Certify failed rows against original requests, preserving accepted rows.

    Exact endpoint IK may reject a pose whose arrival region intersects the
    reachable workspace. Probe at most 90 additional poses inside that region;
    never replace the requested pose. Every accepted joint state must pass FK
    against the original request and the unchanged joint-bound tolerance.
    This constructs endpoint evidence only, not a path or physical certificate.
    """
    found = (
        torch.as_tensor(success, device=seed.device, dtype=torch.bool)
        .reshape(-1)
        .clone()
    )
    result = solved.clone()
    eye = torch.eye(3, dtype=target.dtype, device=target.device)
    rotations = [eye]
    for fraction in (0.4, -0.4, 0.9, -0.9):
        angle = fraction * protocol.rotation_threshold_rad
        for axis in range(3):
            vector = eye[axis]
            x, y, z = vector.unbind()
            zero = torch.zeros_like(x)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
            rotations.append(
                eye + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
            )
    offsets = [torch.zeros(3, dtype=target.dtype, device=target.device)]
    for sign in (1.0, -1.0):
        offsets.extend(sign * 0.9 * protocol.position_threshold_m * eye)
    for offset_index, offset in enumerate(offsets):
        for rotation_index, rotation in enumerate(rotations):
            if offset_index == rotation_index == 0:
                continue  # The caller already tried the original exact pose.
            ids = torch.nonzero(~found, as_tuple=False).flatten()
            if ids.numel() == 0:
                return found, result
            local_ids = ids.cpu().tolist()
            selected_env_ids = (
                local_ids if env_ids is None else [env_ids[i] for i in local_ids]
            )
            original = target.index_select(0, ids)
            probe = original.clone()
            probe[:, :3, :3] = rotation @ original[:, :3, :3]
            probe[:, :3, 3] += offset
            flags, candidate = robot.compute_ik(
                pose=probe,
                joint_seed=seed.index_select(0, ids),
                name=control_part,
                env_ids=selected_env_ids,
            )
            candidate = canonical_case_qpos(candidate)
            lower, upper = (
                robot.get_qpos_limits(name=control_part, env_ids=selected_env_ids)
                .to(candidate)
                .unbind(-1)
            )
            violation = (
                torch.maximum(lower - candidate, candidate - upper)
                .clamp_min(0)
                .amax(-1)
            )
            valid = torch.as_tensor(
                flags, device=seed.device, dtype=torch.bool
            ).reshape(-1)
            valid = (
                valid
                & torch.isfinite(candidate).all(-1)
                & (violation <= protocol.joint_limit_tolerance_rad)
            )
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            rows = ids.index_select(0, indices)
            qpos = candidate.index_select(0, indices)
            actual = robot.compute_fk(
                qpos=qpos,
                name=control_part,
                env_ids=(
                    rows.cpu().tolist()
                    if env_ids is None
                    else [env_ids[i] for i in rows.cpu().tolist()]
                ),
                to_matrix=True,
            )
            position, rotation_error = pose_error(actual, target.index_select(0, rows))
            qualified = (position.norm(dim=-1) <= protocol.position_threshold_m) & (
                rotation_error.norm(dim=-1) <= protocol.rotation_threshold_rad
            )
            accepted = rows[qualified]
            result[accepted] = qpos[qualified]
            found[accepted] = True
    return found, result
