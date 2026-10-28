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

"""Transport-neutral nominal motion supplied to an atomic skill."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from embodichain.data_pipeline.real2sim import TrajectoryTimeMap

from .invocation import ResolvedActionRequest
from .plans import TimedTrajectory
from .state import PlanningContext

__all__ = ["NominalMotionSource", "InstantiatedSkillMotion"]


@dataclass(frozen=True)
class InstantiatedSkillMotion:
    """An external full-joint motion with explicit skill-stage boundaries.

    The skill owns semantic effects and command assembly. A source cannot
    commit held state or step the environment. ``grasp_pose`` is an optional
    world-frame candidate used by skills with an acquisition boundary.
    """

    trajectory: TimedTrajectory
    success: torch.Tensor
    segment_lengths: Mapping[str, int]
    initial_qpos: torch.Tensor
    source_id: str
    source_revision: str
    unit_id: str
    origin: str
    authority: str = "qpos"
    collision_evidence: str = "unverified"
    pose_origin: str = "unknown"
    qpos_semantics: str = "unknown"
    source_command_semantics: str | None = None
    fk_position_error_m: float | None = None
    fk_rotation_error_rad: float | None = None
    gripper_mode: str = "recorded"
    initial_state_origin: str = "unknown"
    calibration_revision: str = "unverified"
    annotation_revision: str = "unverified"
    prototype_sha256: str | None = None
    source_time_map: TrajectoryTimeMap | None = None
    source_sample_offset: int = 0
    ik_smoothing_iterations: int = 0
    phase_permissions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    grasp_pose: torch.Tensor | None = None
    joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, TimedTrajectory):
            raise TypeError("trajectory must be a TimedTrajectory")
        for name in ("source_id", "source_revision", "unit_id", "origin"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("calibration_revision", "annotation_revision"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty string")
        if self.prototype_sha256 is not None and (
            not isinstance(self.prototype_sha256, str)
            or len(self.prototype_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.prototype_sha256)
        ):
            raise ValueError("prototype_sha256 must be a lowercase SHA-256 digest")
        if self.authority not in {
            "qpos",
            "eef_pose",
            "hybrid",
        } or self.collision_evidence not in {"unverified", "sampled_passed"}:
            raise ValueError("unknown motion authority or collision evidence")
        if self.pose_origin not in {
            "unknown",
            "measured",
            "fk_derived",
            "transformed",
            "fk_derived_at_execution",
        }:
            raise ValueError("unknown nominal pose origin")
        if self.qpos_semantics not in {"unknown", "measured_state", "design_reference"}:
            raise ValueError("unknown source qpos semantics")
        if self.source_command_semantics not in {
            None,
            "actual_submission",
            "design_reference",
        }:
            raise ValueError("unknown source command semantics")
        for name in ("fk_position_error_m", "fk_rotation_error_rad"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative or None")
        if self.gripper_mode not in {"recorded", "semantic"}:
            raise ValueError("unknown nominal gripper mode")
        if self.initial_state_origin not in {
            "unknown",
            "observed",
            "first_measured_sample",
            "reference_assumption",
            "current_observation",
        }:
            raise ValueError("unknown nominal initial-state origin")
        if (
            type(self.source_sample_offset) is not int
            or not 0 <= self.source_sample_offset < self.trajectory.waypoint_count
        ):
            raise ValueError("source_sample_offset must address a trajectory sample")
        if (
            type(self.ik_smoothing_iterations) is not int
            or self.ik_smoothing_iterations < 0
        ):
            raise ValueError("ik_smoothing_iterations must be nonnegative")
        if self.source_time_map is not None:
            if (
                not isinstance(self.source_time_map, TrajectoryTimeMap)
                or len(self.source_time_map.left) + self.source_sample_offset
                != self.trajectory.waypoint_count
            ):
                raise ValueError(
                    "source time map must align with the non-connection trajectory"
                )
            mapping = self.source_time_map
            object.__setattr__(
                self,
                "source_time_map",
                TrajectoryTimeMap(
                    mapping.left,
                    mapping.right,
                    mapping.weight,
                    mapping.source_event_indices,
                    mapping.execution_event_indices,
                    mapping.max_event_error_s,
                ),
            )
        success = self.success
        if (
            not isinstance(success, torch.Tensor)
            or success.dtype != torch.bool
            or success.shape != (self.trajectory.batch_size,)
            or success.device != self.trajectory.positions.device
        ):
            raise ValueError(
                "success must be a per-row boolean tensor on the trajectory device"
            )
        segments = dict(self.segment_lengths)
        if not segments or any(
            not isinstance(name, str)
            or not name.strip()
            or type(length) is not int
            or length <= 0
            for name, length in segments.items()
        ):
            raise ValueError(
                "segment lengths must be a nonempty ordered mapping of positive lengths"
            )
        if sum(segments.values()) != self.trajectory.waypoint_count:
            raise ValueError("segment lengths must cover every trajectory sample")
        if any(
            isinstance(values, (str, bytes))
            for values in self.phase_permissions.values()
        ):
            raise ValueError("phase permissions must be sequences of operator names")
        permissions = {
            name: tuple(values) for name, values in self.phase_permissions.items()
        }
        if permissions and (
            set(permissions) != set(segments)
            or any(
                any(
                    not isinstance(operator, str) or not operator for operator in values
                )
                or len(set(values)) != len(values)
                for values in permissions.values()
            )
        ):
            raise ValueError(
                "phase permissions must cover every segment with unique operators"
            )
        initial = self.initial_qpos
        if (
            not isinstance(initial, torch.Tensor)
            or initial.shape != (self.trajectory.batch_size, self.trajectory.robot_dof)
            or initial.device != success.device
            or not bool(torch.isfinite(initial).all())
        ):
            raise ValueError(
                "initial_qpos must be a finite full-robot state on the trajectory device"
            )
        names = tuple(self.joint_names)
        if (
            not names
            or len(names) != self.trajectory.robot_dof
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name.strip() for name in names)
        ):
            raise ValueError(
                "joint_names must describe the complete unique joint layout"
            )
        if self.grasp_pose is not None:
            grasp = self.grasp_pose
            if (
                not isinstance(grasp, torch.Tensor)
                or grasp.shape != (self.trajectory.batch_size, 4, 4)
                or grasp.device != success.device
                or not bool(torch.isfinite(grasp).all())
            ):
                raise ValueError("grasp_pose must be a finite batched 4x4 tensor")
            rotation = grasp[:, :3, :3]
            eye = torch.eye(3, dtype=grasp.dtype, device=grasp.device).expand_as(
                rotation
            )
            bottom = grasp.new_tensor([0, 0, 0, 1]).expand(grasp.shape[0], -1)
            if (
                not torch.allclose(
                    rotation.transpose(1, 2) @ rotation, eye, atol=1e-5, rtol=0
                )
                or not torch.allclose(
                    torch.linalg.det(rotation),
                    grasp.new_ones(grasp.shape[0]),
                    atol=1e-5,
                    rtol=0,
                )
                or not torch.allclose(grasp[:, 3], bottom, atol=1e-5, rtol=0)
            ):
                raise ValueError("grasp_pose must contain proper rigid transforms")
            object.__setattr__(self, "grasp_pose", grasp.detach().clone())
        object.__setattr__(self, "trajectory", self.trajectory.snapshot())
        object.__setattr__(self, "success", success.detach().clone())
        object.__setattr__(self, "initial_qpos", initial.detach().clone())
        object.__setattr__(self, "segment_lengths", MappingProxyType(segments))
        object.__setattr__(self, "phase_permissions", MappingProxyType(permissions))
        object.__setattr__(self, "joint_names", names)

    def provenance(self) -> dict[str, object]:
        """Return stable source evidence for the assembled plan."""
        return {
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "unit_id": self.unit_id,
            "origin": self.origin,
            "authority": self.authority,
            "collision_evidence": self.collision_evidence,
            "collision_scope": (
                "planner_configured_world"
                if self.collision_evidence == "sampled_passed"
                else "not_checked"
            ),
            "geometry_compatibility": "unverified",
            "pose_origin": self.pose_origin,
            "qpos_semantics": self.qpos_semantics,
            "source_command_semantics": self.source_command_semantics,
            "fk_position_error_m": self.fk_position_error_m,
            "fk_rotation_error_rad": self.fk_rotation_error_rad,
            "gripper_mode": self.gripper_mode,
            "initial_state_origin": self.initial_state_origin,
            "calibration_revision": self.calibration_revision,
            "annotation_revision": self.annotation_revision,
            "prototype_sha256": self.prototype_sha256,
            "source_sample_offset": self.source_sample_offset,
            "ik_smoothing_iterations": self.ik_smoothing_iterations,
            "source_samples_mapped": (
                None if self.source_time_map is None else len(self.source_time_map.left)
            ),
            "prototype_phase_permissions": dict(self.phase_permissions),
        }


@runtime_checkable
class NominalMotionSource(Protocol):
    """Provide motion for one grounded invocation without owning its effects."""

    def instantiate(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
    ) -> InstantiatedSkillMotion:
        """Create a side-effect-free motion candidate for the current scene."""
