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

"""Frozen affordance sampling, observed-pose rebasing and tutorial batch audits."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json

import numpy as np
import torch

from embodichain.toolkits.graspkit import GraspCandidateBatch
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
    from embodichain.lab.trajectory_generation.integrations.atomic_affordance import (
        AtomicAffordanceBatch,
    )
    from embodichain.toolkits.graspkit import GraspPoseGenerator

__all__ = [
    "sample_affordance_grasps",
    "rebase_affordance_grasps",
    "selected_grasps_from_result",
    "save_affordance_result",
]


def _validate_object_poses(poses: torch.Tensor, name: str) -> None:
    if (
        not isinstance(poses, torch.Tensor)
        or not poses.is_floating_point()
        or poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or poses.shape[0] < 1
        or not torch.isfinite(poses).all()
    ):
        raise ValueError(f"{name} must contain finite (E,4,4) object poses")
    checked = poses.double()
    rotation = checked[:, :3, :3]
    if (
        not torch.allclose(
            checked[:, 3],
            checked.new_tensor([0, 0, 0, 1]).expand(poses.shape[0], -1),
            atol=1e-6,
            rtol=0,
        )
        or not torch.allclose(
            rotation.transpose(-2, -1) @ rotation,
            torch.eye(3, device=poses.device, dtype=checked.dtype).expand(
                poses.shape[0], -1, -1
            ),
            atol=1e-6,
            rtol=0,
        )
        or not torch.allclose(
            torch.linalg.det(rotation),
            checked.new_ones(poses.shape[0]),
            atol=1e-6,
            rtol=0,
        )
    ):
        raise ValueError(f"{name} must contain proper SE(3) transforms")


def sample_affordance_grasps(
    generator: GraspPoseGenerator,
    affordance: AntipodalAffordance,
    *,
    object_poses: torch.Tensor,
    approach_direction: torch.Tensor,
    seed: int = 13,
) -> GraspCandidateBatch:
    """Sample one frozen raw grasp set and reproject it to every physical row.

    Args:
        generator: Standalone rich grasp service, not a simulator callback.
        affordance: Target-local triangle mesh used in every environment.
        object_poses: Observed local-arena object poses, ``(E,4,4)``.
        approach_direction: One common ``(3,)`` or per-row ``(E,3)`` local-arena
            direction. Directions must agree in object-local coordinates.
        seed: Isolated sampling seed; global Torch randomness is unchanged.

    Returns:
        ``(E,G,4,4)`` raw candidates with shared stable grasp IDs. This certifies
        neither scene equivalence nor IK/path/physical feasibility.
    """
    _validate_object_poses(object_poses, "object_poses")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    direction = torch.as_tensor(
        approach_direction, device=object_poses.device, dtype=object_poses.dtype
    )
    if direction.shape == (3,):
        direction = direction.unsqueeze(0).expand(object_poses.shape[0], -1)
    if (
        direction.shape != (object_poses.shape[0], 3)
        or not torch.isfinite(direction).all()
    ):
        raise ValueError("approach_direction must contain finite (3,) or (E,3) values")
    lengths = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if torch.any(lengths <= 1e-8):
        raise ValueError("approach_direction cannot contain zero vectors")
    direction = direction / lengths
    local_directions = (
        object_poses[:, :3, :3].transpose(-2, -1) @ direction.unsqueeze(-1)
    ).squeeze(-1)
    if not torch.allclose(
        local_directions,
        local_directions[:1].expand_as(local_directions),
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError(
            "frozen grasp sampling requires matching object-local approach directions"
        )
    vertices, triangles = affordance.mesh_vertices, affordance.mesh_triangles
    if not isinstance(vertices, torch.Tensor) or not isinstance(
        triangles, torch.Tensor
    ):
        raise ValueError("affordance must supply target-local mesh tensors")
    sampled = generator.get_grasp_candidates(
        mesh_vertices=vertices.to(object_poses.device),
        mesh_triangles=triangles.to(object_poses.device),
        obj_poses=object_poses[:1],
        approach_direction=direction[:1],
        generator=torch.Generator(device=object_poses.device).manual_seed(seed),
        frame="local_arena",
    )
    if (
        not isinstance(sampled, GraspCandidateBatch)
        or sampled.poses.shape[0] != 1
        or sampled.frame != "local_arena"
    ):
        raise ValueError("grasp service must return one local-arena candidate row")
    poses = object_poses.to(sampled.poses)
    local_grasps = torch.linalg.inv(poses[0]) @ sampled.poses[0]
    count = object_poses.shape[0]
    return GraspCandidateBatch(
        poses=poses[:, None] @ local_grasps[None],
        costs=sampled.costs.expand(count, -1),
        valid_mask=sampled.valid_mask.expand(count, -1),
        opening_widths=(
            None
            if sampled.opening_widths is None
            else sampled.opening_widths.expand(count, -1)
        ),
        grasp_ids=sampled.grasp_ids * count,
        frame="local_arena",
        rejection_reasons=sampled.rejection_reasons * count,
    )


def rebase_affordance_grasps(
    grasps: GraspCandidateBatch,
    old_object_poses: torch.Tensor,
    new_object_poses: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> GraspCandidateBatch:
    """Keep each raw object-relative grasp while using each row's observed pose.

    Args:
        grasps: Old local-arena candidates with stable IDs.
        old_object_poses: Object poses at the original sampling/planning instant.
        new_object_poses: Fresh observed object poses, not predicted travel.
        active_mask: Optional eligible physical rows, used to retain failures.

    Returns:
        Reprojected candidates with unchanged IDs, widths and raw local geometry.
        Failed/ineligible rows remain masked and require no new sampling.
    """
    _validate_object_poses(old_object_poses, "old_object_poses")
    _validate_object_poses(new_object_poses, "new_object_poses")
    count = grasps.poses.shape[0]
    if (
        grasps.frame != "local_arena"
        or old_object_poses.shape != (count, 4, 4)
        or new_object_poses.shape != old_object_poses.shape
    ):
        raise ValueError("grasp and observed pose rows must align in local_arena")
    if active_mask is None:
        active_mask = torch.ones(count, dtype=torch.bool, device=grasps.poses.device)
    if (
        not isinstance(active_mask, torch.Tensor)
        or active_mask.dtype != torch.bool
        or active_mask.shape != (count,)
    ):
        raise ValueError("active_mask must be a boolean physical-row vector")
    active_mask = active_mask.to(grasps.poses.device)
    relative = (
        torch.linalg.inv(old_object_poses.to(grasps.poses))[:, None] @ grasps.poses
    )
    reasons = tuple(
        tuple(
            reason if bool(active_mask[row]) else "INACTIVE_ROW"
            for reason in old_reasons
        )
        for row, old_reasons in enumerate(grasps.rejection_reasons)
    )
    return GraspCandidateBatch(
        poses=new_object_poses.to(grasps.poses)[:, None] @ relative,
        costs=grasps.costs,
        valid_mask=grasps.valid_mask & active_mask[:, None],
        opening_widths=grasps.opening_widths,
        grasp_ids=grasps.grasp_ids,
        frame="local_arena",
        rejection_reasons=reasons,
    )


def selected_grasps_from_result(
    result: AtomicAffordanceBatch,
    old_object_poses: torch.Tensor,
    new_object_poses: torch.Tensor,
) -> GraspCandidateBatch:
    """Rebase each successful selected raw grasp for a follow-up Slide stage.

    Args:
        result: Previous fixed-row planning result. ``grasp_poses`` must be the
            selected input raw poses, before any EEF calibration/variant.
        old_object_poses: Per-row handle poses observed before the previous plan.
        new_object_poses: Per-row handle poses observed after actual replay.

    Returns:
        One stable raw candidate per physical row. Previously failed rows stay
        invalid; surviving rows are not compacted or reassigned to another slot.
    """
    mask = result.success_mask
    if (
        mask.ndim != 1
        or mask.dtype != torch.bool
        or result.grasp_poses.shape != (mask.numel(), 4, 4)
    ):
        raise ValueError("result must expose aligned selected poses and success mask")
    if len(result.grasp_ids) != mask.numel():
        raise ValueError("result grasp_ids must align with physical rows")
    ids = []
    for row, grasp_id in enumerate(result.grasp_ids):
        if bool(mask[row]) and (not isinstance(grasp_id, str) or not grasp_id):
            raise ValueError("successful rows must retain their original grasp ID")
        ids.append((grasp_id or f"inactive:{row}",))
    selected = GraspCandidateBatch(
        poses=result.grasp_poses.unsqueeze(1),
        costs=result.grasp_poses.new_zeros((mask.numel(), 1)),
        valid_mask=mask[:, None],
        grasp_ids=tuple(ids),
        frame="local_arena",
    )
    return rebase_affordance_grasps(
        selected, old_object_poses, new_object_poses, active_mask=mask
    )


def _json_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_affordance_result(
    result: AtomicAffordanceBatch,
    *,
    output_dir: Path | None,
    name: str,
    physical_envs: int,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Log a bounded plan audit and optionally save only successful trajectories.

    Args:
        result: Full physical-row result; failures are idle holds for replay.
        output_dir: Optional output folder. ``None`` only logs the summary.
        name: Safe filename stem, such as ``pickup`` or ``slide_pull``.
        physical_envs: Requested real physical environment count.
        metadata: Additional caller-owned audit data; it does not certify success.

    Returns:
        JSON-compatible report with compact-row to physical-env mapping. Failed
        idle rows are excluded from saved trajectories. Physical validation is
        explicitly false, even if the caller subsequently replays the plans.
    """
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or Path(name).name != name
        or "\\" in name
    ):
        raise ValueError("name must be a nonempty safe filename stem")
    mask = result.success_mask
    if (
        type(physical_envs) is not int
        or physical_envs < 1
        or mask.shape != (physical_envs,)
        or mask.dtype != torch.bool
    ):
        raise ValueError(
            "success mask must cover the requested physical environment count"
        )
    trajectory = result.trajectory
    if trajectory.positions.shape[0] != physical_envs or not torch.equal(
        trajectory.env_ids,
        torch.arange(physical_envs, device=trajectory.env_ids.device),
    ):
        raise ValueError("result trajectory must retain complete ordered physical rows")
    lengths = result.valid_length[mask]
    if (
        lengths.dtype != torch.long
        or torch.any(lengths < 1)
        or torch.any(lengths > trajectory.positions.shape[1])
    ):
        raise ValueError(
            "successful valid lengths must address real trajectory samples"
        )
    slots = torch.nonzero(mask, as_tuple=False).flatten()
    horizon = int(lengths.max()) if lengths.numel() else 0
    qpos = trajectory.positions[mask, :horizon].clone()
    dt = trajectory.dt[mask, :horizon].clone()
    valid = torch.arange(horizon, device=mask.device)[None] < lengths[:, None]
    for row, length in enumerate(lengths.tolist()):
        qpos[row, length:] = qpos[row, length - 1]
        dt[row, length:] = 0
    compact_ids = [result.grasp_ids[int(slot)] for slot in slots]
    report = _json_value(
        {
            "name": name,
            "requested": physical_envs,
            "output_count": int(mask.sum()),
            "output_shape": list(qpos.shape),
            "success_mask": mask,
            "compact_env_ids": slots,
            "grasp_ids": result.grasp_ids,
            "candidate_indices": result.candidate_indices,
            "summary": result.summary,
            "rejections": result.rejections,
            "physical_validation": False,
            "expert_episodes_committed": 0,
            "metadata": {} if metadata is None else metadata,
        }
    )
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if output_dir is not None:
        output_dir = Path(output_dir)
        paths = (output_dir / f"{name}.npz", output_dir / f"{name}.json")
        if any(path.exists() for path in paths):
            raise FileExistsError(
                "affordance output files already exist; choose a new folder or name"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths[0],
            qpos=qpos.detach().cpu().numpy(),
            dt=dt.detach().cpu().numpy(),
            valid_length=lengths.detach().cpu().numpy(),
            valid_mask=valid.detach().cpu().numpy(),
            env_ids=slots.detach().cpu().numpy(),
            success_mask=mask.detach().cpu().numpy(),
            grasp_ids=np.asarray(compact_ids, dtype=np.str_),
            grasp_poses=result.grasp_poses[mask].detach().cpu().numpy(),
            candidate_indices=result.candidate_indices[mask].detach().cpu().numpy(),
        )
        paths[1].write_text(encoded, encoding="utf-8")
    logger.log_info(
        json.dumps(
            {
                "affordance": name,
                "requested": physical_envs,
                "output_count": int(mask.sum()),
                "summary": _json_value(result.summary),
                "physical_validation": False,
            },
            allow_nan=False,
        )
    )
    return report
