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

"""Observed query state and backend-local payload attachment values."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from .curobo_planner import CuroboPlannerCfg

__all__ = ["CuroboPayload", "CuroboQuerySnapshot", "curobo_motion_generator"]

# Preserve the qualified source model's reserved link identity and ordering.
_PAYLOAD_LINK = "gen_sim_payload"


def _owned_pose(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (1, 4, 4)
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} requires one finite floating pose (1, 4, 4).")
    owned = value.detach().clone()
    # B=1 construction validates a tiny metadata matrix on CPU, avoiding a
    # cuSOLVER initialization merely to check a 3x3 determinant on the sim GPU.
    matrix = owned.to(device="cpu", dtype=torch.float64)
    rotation = matrix[0, :3, :3]
    if (
        not torch.isfinite(matrix).all()
        or not torch.allclose(
            matrix[0, 3], matrix.new_tensor([0, 0, 0, 1]), atol=1e-6, rtol=0
        )
        or not torch.allclose(
            rotation.T @ rotation,
            torch.eye(3, device=matrix.device, dtype=matrix.dtype),
            atol=1e-5,
            rtol=0,
        )
        or not torch.isclose(
            torch.linalg.det(rotation), matrix.new_tensor(1.0), atol=1e-5, rtol=0
        )
    ):
        raise ValueError(f"{name} must be a rigid transform.")
    return owned


@dataclass(frozen=True, slots=True, kw_only=True)
class CuroboPayload:
    """One carried body's owned local sphere envelope and observed world pose.

    Args:
        object_id: Exact carried-object identity for diagnostics.
        control_part: Robot control part whose tool carries the object.
        spheres: Already-scaled object-local centers/radii, shape ``(N, 4)``,
            with positive radii.
        object_pose: Same-observation world pose, shape ``(1, 4, 4)``.

    Geometry is caller-provided input, not a native-cooked or CCD certificate.
    This value owns copies and carries no robot or simulation handle.
    """

    object_id: str
    control_part: str
    spheres: torch.Tensor
    object_pose: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("object_id", "control_part"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty canonical string.")
        if (
            not isinstance(self.spheres, torch.Tensor)
            or self.spheres.ndim != 2
            or self.spheres.shape[1] != 4
            or not self.spheres.shape[0]
            or not self.spheres.is_floating_point()
            or not torch.isfinite(self.spheres).all()
            or (self.spheres[:, 3] <= 0).any()
        ):
            raise ValueError(
                "Payload spheres must be finite (N, 4) with positive radii."
            )
        object.__setattr__(self, "spheres", self.spheres.detach().clone())
        object.__setattr__(
            self, "object_pose", _owned_pose(self.object_pose, "object_pose")
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CuroboQuerySnapshot:
    """Owned B=1 state for one fresh, single-control-part planning lifetime.

    Args:
        full_qpos: Same-observation full robot positions in ``robot.joint_names``
            order, shape ``(1, robot.dof)``. Passive locks use these values.
        world_to_base: Same-observation world-to-cuRobo-base transform,
            shape ``(1, 4, 4)``. World snapshots, goals and payload share it.
        payload: Optional carried body from the same observation.

    The caller owns observation coherence and keeps the world fixed during the
    query. These runtime values are not serialized planner configuration. A
    snapshot is bound to the first requested control part; payloads name that
    part explicitly. Switching parts requires a new planner/snapshot.
    """

    full_qpos: torch.Tensor
    world_to_base: torch.Tensor
    payload: CuroboPayload | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.full_qpos, torch.Tensor)
            or self.full_qpos.ndim != 2
            or self.full_qpos.shape[0] != 1
            or not self.full_qpos.shape[1]
            or not self.full_qpos.is_floating_point()
            or not torch.isfinite(self.full_qpos).all()
        ):
            raise ValueError("Query snapshots require finite full_qpos (1, robot.dof).")
        if self.payload is not None and type(self.payload) is not CuroboPayload:
            raise TypeError("payload must be CuroboPayload or None.")
        object.__setattr__(self, "full_qpos", self.full_qpos.detach().clone())
        object.__setattr__(
            self, "world_to_base", _owned_pose(self.world_to_base, "world_to_base")
        )
        if self.payload is not None:
            object.__setattr__(self, "payload", replace(self.payload))


def _query_robot_config(
    config: dict[str, Any], snapshot: CuroboQuerySnapshot, joint_names: Sequence[str]
) -> dict[str, Any]:
    result = deepcopy(config)
    if snapshot.full_qpos.shape[1] != len(joint_names):
        raise ValueError("Query full_qpos must match the full robot joint order.")
    observed = dict(zip(joint_names, snapshot.full_qpos[0].cpu().tolist(), strict=True))
    kinematics = result["robot_cfg"]["kinematics"]
    kinematics["lock_joints"] = {
        name: observed[name] for name in (kinematics.get("lock_joints") or {})
    }
    payload = snapshot.payload
    if payload is None:
        return result
    if (
        _PAYLOAD_LINK in (kinematics.get("extra_links") or {})
        or _PAYLOAD_LINK in kinematics["collision_link_names"]
    ):
        raise ValueError("The reserved payload link already exists in the robot model.")
    parent = kinematics["tool_frames"][0]
    kinematics["extra_links"] = dict(kinematics.get("extra_links") or {})
    kinematics["extra_links"][_PAYLOAD_LINK] = {
        "link_name": _PAYLOAD_LINK,
        "joint_name": f"{_PAYLOAD_LINK}_joint",
        "joint_type": "FIXED",
        "parent_link_name": parent,
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
    }
    kinematics["extra_collision_spheres"] = dict(
        kinematics.get("extra_collision_spheres") or {}
    )
    kinematics["extra_collision_spheres"][_PAYLOAD_LINK] = len(payload.spheres)
    kinematics["collision_link_names"] = [
        *kinematics["collision_link_names"],
        _PAYLOAD_LINK,
    ]
    return result


def _attachment_models(planner: Any) -> tuple[Any, ...]:
    """Preserve source order and alias-deduplication across backend-owned buffers."""
    models = [planner.kinematics]
    rollouts = []
    for solver in (planner.ik_solver, planner.trajopt_solver):
        rollouts.extend(solver.get_all_rollout_instances())
    rollouts.extend(planner.trajopt_solver.additional_metrics_rollouts.values())
    if planner.graph_planner is not None:
        rollouts.extend(planner.graph_planner.get_all_rollout_instances())
    for rollout in rollouts:
        models.extend(
            (
                rollout.transition_model.robot_model,
                rollout.metrics_transition_model.robot_model,
            )
        )
    distinct = {}
    for model in models:
        spheres = model.config.kinematics_config.link_spheres
        distinct[(str(spheres.device), spheres.data_ptr())] = model
    return tuple(distinct.values())


@contextmanager
def curobo_motion_generator(
    cfg: CuroboPlannerCfg,
    *,
    query_snapshot: CuroboQuerySnapshot | None = None,
) -> Iterator[MotionGenerator]:
    """Own one standard motion facade and close its cuRobo backend on exit.

    Args:
        cfg: Standard cuRobo planner configuration, including an explicitly
            projected world. Query snapshots require B=1, disabled CUDA graphs,
            retained planner samples and an immutable world during this lifetime.
        query_snapshot: Optional same-observation state and carried-body envelope.

    Yields:
        A MotionGenerator using the existing cuRobo backend. Callers select all
        motion options explicitly; this factory makes no skill-phase decisions.
        Returned trajectory tensors remain owned after closing the planner.
    """
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenCfg,
        MotionGenerator,
    )
    from .curobo_planner import CuroboPlanner

    class _QueryMotionGenerator(MotionGenerator):
        def _create_planner(self, planner_cfg: CuroboPlannerCfg) -> CuroboPlanner:
            return CuroboPlanner(planner_cfg, query_snapshot=query_snapshot)

    generator = _QueryMotionGenerator(MotionGenCfg(planner_cfg=cfg))
    try:
        yield generator
    finally:
        generator.planner.close()
