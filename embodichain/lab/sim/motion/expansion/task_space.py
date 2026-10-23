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

"""Endpoint-preserving task-space deformation and trajectory IK projection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import math

import torch

from embodichain.compute.kinematics import (
    DifferentiableIKCfg,
    pose_error,
    solve_dls_ik,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix

from .contracts import (
    TrajectoryPhase,
    TrajectoryTemplate,
    _names,
    _phases,
    _tensor,
    _text,
)

__all__ = [
    "TASK_SPACE_RESIDUAL_OPERATOR",
    "TaskSpaceDeformationConstraint",
    "TaskSpaceTrajectory",
    "TaskSpaceProjectionResult",
    "task_space_from_joint_template",
    "deform_task_space_trajectory",
    "project_task_space_trajectory",
]

TASK_SPACE_RESIDUAL_OPERATOR = "task_space_residual"
"""Phase permission used by endpoint-preserving Cartesian deformation."""

KinematicsFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _axes(values: Sequence[bool], name: str) -> tuple[bool, bool, bool]:
    result = tuple(values)
    if len(result) != 3 or any(type(value) is not bool for value in result):
        raise ValueError(f"{name} must contain exactly three booleans")
    return result  # type: ignore[return-value]


def _nonnegative(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _rigid_transforms(value: torch.Tensor, name: str) -> torch.Tensor:
    result = _tensor(value, name, 3)
    if result.shape[0] < 1 or result.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (T >= 1, 4, 4)")
    check = result.to(dtype=torch.float64)
    rotation = check[:, :3, :3]
    identity = torch.eye(3, dtype=check.dtype, device=check.device).expand_as(rotation)
    bottom = check.new_tensor([0, 0, 0, 1]).expand(result.shape[0], -1)
    valid = (
        torch.isclose(check[:, 3], bottom, atol=1e-5, rtol=0).all(dim=-1)
        & torch.isclose(
            rotation.transpose(-1, -2) @ rotation,
            identity,
            atol=1e-5,
            rtol=0,
        ).all(dim=(-2, -1))
        & torch.isclose(
            torch.linalg.det(rotation),
            check.new_ones(result.shape[0]),
            atol=1e-5,
            rtol=0,
        )
    )
    if not bool(valid.all()):
        raise ValueError(f"{name} must contain proper SE(3) transforms")
    return result


@dataclass(frozen=True)
class TaskSpaceDeformationConstraint:
    """Per-phase bounds for one root-frame Cartesian deformation.

    Args:
        phase_id: Phase authorized to use this constraint.
        translation_axes: Root-frame translation axes that may change.
        rotation_axes: Root-frame rotation-vector axes that may change.
        max_translation_m: Maximum translation-vector norm in meters.
        max_rotation_rad: Maximum rotation-vector norm in radians.
    """

    phase_id: str
    translation_axes: tuple[bool, bool, bool] = (True, True, True)
    rotation_axes: tuple[bool, bool, bool] = (True, True, True)
    max_translation_m: float = 0.0
    max_rotation_rad: float = 0.0

    def __post_init__(self) -> None:
        _text(self.phase_id, "phase_id")
        translation_axes = _axes(self.translation_axes, "translation_axes")
        rotation_axes = _axes(self.rotation_axes, "rotation_axes")
        translation = _nonnegative(self.max_translation_m, "max_translation_m")
        rotation = _nonnegative(self.max_rotation_rad, "max_rotation_rad")
        if not any(translation_axes):
            translation = 0.0
        if not any(rotation_axes):
            rotation = 0.0
        if translation == 0 and rotation == 0:
            raise ValueError("a deformation constraint must permit a non-zero change")
        object.__setattr__(self, "translation_axes", translation_axes)
        object.__setattr__(self, "rotation_axes", rotation_axes)
        object.__setattr__(self, "max_translation_m", translation)
        object.__setattr__(self, "max_rotation_rad", rotation)


@dataclass(frozen=True)
class TaskSpaceTrajectory:
    """One single-frame Cartesian trajectory aligned with a joint template.

    Poses map the controlled frame into the kinematics root frame. This is an
    intermediate expansion value: a caller must project and physically validate
    it before execution.

    Args:
        source_id: Stable source identity.
        source_revision: Source revision used to create the trajectory.
        template_id: Source-unit identity aligned with the joint template.
        frame_id: Controlled tool or end-effector frame.
        poses: Root-frame transforms with shape ``(T, 4, 4)``.
        dt: Arrival intervals with shape ``(T,)``; the first value is zero.
        phases: Ordered semantic phase annotations.
        allowed_operators: Operators authorized anywhere in the trajectory.
        validator_id: Physical validation contract required downstream.
    """

    source_id: str
    source_revision: str
    template_id: str
    frame_id: str
    poses: torch.Tensor
    dt: torch.Tensor
    phases: tuple[TrajectoryPhase, ...] = ()
    allowed_operators: tuple[str, ...] = ()
    validator_id: str = "default"

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "source_revision",
            "template_id",
            "frame_id",
            "validator_id",
        ):
            _text(getattr(self, name), name)
        poses = _rigid_transforms(self.poses, "poses")
        dt = _tensor(self.dt, "dt", 1)
        if dt.shape != poses.shape[:1] or dt.device != poses.device:
            raise ValueError("dt must match poses length and device")
        if dt[0] != 0 or bool((dt[1:] <= 0).any()):
            raise ValueError("dt[0] must be zero and later arrival intervals positive")
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "phases", _phases(self.phases, poses.shape[0]))
        object.__setattr__(
            self,
            "allowed_operators",
            _names(self.allowed_operators, "allowed_operators"),
        )


@dataclass(frozen=True)
class TaskSpaceProjectionResult:
    """Projected qpos trajectory and per-sample Cartesian residual evidence.

    Args:
        template: Final DLS iterate in the original full joint layout.
        success: Per-sample IK and locked-reference verification mask.
        position_error: Final root-frame translation residuals, shape ``(T, 3)``.
        rotation_error: Final root-frame rotation residuals, shape ``(T, 3)``.
    """

    template: TrajectoryTemplate
    success: torch.Tensor
    position_error: torch.Tensor
    rotation_error: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.template, TrajectoryTemplate):
            raise TypeError("template must be a TrajectoryTemplate")
        count = self.template.positions.shape[0]
        if (
            not isinstance(self.success, torch.Tensor)
            or self.success.shape != (count,)
            or self.success.dtype != torch.bool
        ):
            raise ValueError("success must be a boolean tensor with shape (T,)")
        position = _tensor(self.position_error, "position_error", 2)
        rotation = _tensor(self.rotation_error, "rotation_error", 2)
        if position.shape != (count, 3) or rotation.shape != position.shape:
            raise ValueError("projection errors must have shape (T, 3)")
        if (
            position.device != self.success.device
            or rotation.device != self.success.device
            or self.success.device != self.template.positions.device
        ):
            raise ValueError("projection evidence must share one device")
        object.__setattr__(self, "success", self.success.detach().clone())
        object.__setattr__(self, "position_error", position)
        object.__setattr__(self, "rotation_error", rotation)

    @property
    def accepted(self) -> bool:
        """Return whether every trajectory sample satisfies the IK contract.

        Returns:
            ``True`` only when every sample passes pose and joint-limit checks.
        """
        return bool(self.success.all())


def task_space_from_joint_template(
    template: TrajectoryTemplate,
    *,
    frame_id: str,
    kinematics: KinematicsFn,
    editable_phase_ids: Sequence[str] = (),
) -> TaskSpaceTrajectory:
    """Evaluate one joint template and authorize selected free phases.

    Args:
        template: Full-joint qpos reference trajectory.
        frame_id: Controlled kinematic frame identity.
        kinematics: Batched FK/Jacobian callable over controlled joints.
        editable_phase_ids: Free phases allowed to receive Cartesian residuals.

    Returns:
        An owned task-space trajectory aligned sample-for-sample with ``template``.
    """
    if not isinstance(template, TrajectoryTemplate):
        raise TypeError("template must be a TrajectoryTemplate")
    if not callable(kinematics):
        raise TypeError("kinematics must be callable")
    _text(frame_id, "frame_id")
    if isinstance(editable_phase_ids, str):
        raise ValueError("editable_phase_ids must be a sequence of names")
    editable = _names(tuple(editable_phase_ids), "editable_phase_ids")
    known = {phase.phase_id: phase for phase in template.phases}
    if set(editable) - set(known):
        raise ValueError("editable_phase_ids contains an unknown phase")
    if any(known[name].kind != "free" for name in editable):
        raise ValueError("only free phases may receive Cartesian deformation")
    indices = list(template.controlled_joint_indices)
    if not indices:
        raise ValueError("task-space conversion requires controlled joint indices")
    poses, jacobians = kinematics(template.positions[:, indices])
    if poses.shape != (template.positions.shape[0], 4, 4):
        raise ValueError("kinematics poses must have shape (T, 4, 4)")
    if jacobians.shape != (template.positions.shape[0], 6, len(indices)):
        raise ValueError("kinematics Jacobians must have shape (T, 6, C)")
    phases = tuple(
        replace(
            phase,
            allowed_operators=(
                tuple(
                    dict.fromkeys(
                        (*phase.allowed_operators, TASK_SPACE_RESIDUAL_OPERATOR)
                    )
                )
                if phase.phase_id in editable
                else phase.allowed_operators
            ),
        )
        for phase in template.phases
    )
    allowed = tuple(
        dict.fromkeys(
            (
                *template.allowed_operators,
                *(operator for phase in phases for operator in phase.allowed_operators),
            )
        )
    )
    return TaskSpaceTrajectory(
        source_id=template.source_id,
        source_revision=template.source_revision,
        template_id=template.template_id,
        frame_id=frame_id,
        poses=poses,
        dt=template.dt,
        phases=phases,
        allowed_operators=allowed,
        validator_id=template.validator_id,
    )


def _sample_bounded_vector(
    *,
    axes: tuple[bool, bool, bool],
    maximum: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if maximum == 0:
        return torch.zeros(3, device=device, dtype=dtype)
    values = torch.randn(
        3,
        generator=generator,
        device=generator.device,
        dtype=torch.float64,
    )
    mask = torch.tensor(axes, device=generator.device, dtype=torch.bool)
    values = torch.where(mask, values, torch.zeros_like(values))
    norm = torch.linalg.vector_norm(values).clamp_min(torch.finfo(values.dtype).eps)
    magnitude = torch.rand(
        (), generator=generator, device=generator.device, dtype=torch.float64
    )
    return (values / norm * magnitude * maximum).to(device=device, dtype=dtype)


def deform_task_space_trajectory(
    trajectory: TaskSpaceTrajectory,
    *,
    constraints: Mapping[str, TaskSpaceDeformationConstraint],
    generator: torch.Generator,
) -> TaskSpaceTrajectory:
    """Apply smooth root-frame twists while preserving every phase endpoint.

    Args:
        trajectory: Cartesian reference with explicit phase permissions.
        constraints: Exactly one bounded deformation contract per editable phase.
        generator: Explicit local generator; global RNG is never consumed.

    Returns:
        A new Cartesian trajectory requiring IK, path, and physical validation.
    """
    if not isinstance(trajectory, TaskSpaceTrajectory):
        raise TypeError("trajectory must be a TaskSpaceTrajectory")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    editable = tuple(
        phase
        for phase in trajectory.phases
        if TASK_SPACE_RESIDUAL_OPERATOR in phase.allowed_operators
    )
    if not editable:
        raise ValueError(
            "trajectory has no phase authorized for task-space deformation"
        )
    values = dict(constraints)
    if set(values) != {phase.phase_id for phase in editable}:
        raise ValueError("constraints must exactly cover the editable phases")
    if any(
        not isinstance(value, TaskSpaceDeformationConstraint) or key != value.phase_id
        for key, value in values.items()
    ):
        raise TypeError("constraints must map phase IDs to matching value contracts")
    poses = trajectory.poses.clone()
    for phase in editable:
        length = phase.stop_index - phase.start_index
        if length < 3:
            raise ValueError("a task-space deformation phase needs an interior sample")
        constraint = values[phase.phase_id]
        translation = _sample_bounded_vector(
            axes=constraint.translation_axes,
            maximum=constraint.max_translation_m,
            generator=generator,
            device=poses.device,
            dtype=poses.dtype,
        )
        rotation = _sample_bounded_vector(
            axes=constraint.rotation_axes,
            maximum=constraint.max_rotation_rad,
            generator=generator,
            device=poses.device,
            dtype=poses.dtype,
        )
        progress = torch.linspace(0, 1, length, device=poses.device, dtype=poses.dtype)
        envelope = 16 * progress.square() * (1 - progress).square()
        window = poses[phase.start_index : phase.stop_index]
        window[:, :3, 3] += envelope[:, None] * translation
        delta = axis_angle_to_rotation_matrix(envelope[:, None] * rotation)
        window[:, :3, :3] = delta @ window[:, :3, :3]
    return replace(trajectory, poses=poses)


def _aligned_phases(
    task_space: TaskSpaceTrajectory, reference: TrajectoryTemplate
) -> None:
    task_signature = tuple(
        (phase.phase_id, phase.start_index, phase.stop_index, phase.kind)
        for phase in task_space.phases
    )
    reference_signature = tuple(
        (phase.phase_id, phase.start_index, phase.stop_index, phase.kind)
        for phase in reference.phases
    )
    if task_signature != reference_signature:
        raise ValueError(
            "task-space and joint trajectories must share phase boundaries"
        )


def project_task_space_trajectory(
    trajectory: TaskSpaceTrajectory,
    reference: TrajectoryTemplate,
    *,
    kinematics: KinematicsFn,
    joint_limits: torch.Tensor,
    ik_cfg: DifferentiableIKCfg | None = None,
) -> TaskSpaceProjectionResult:
    """Project editable Cartesian samples back into the reference joint layout.

    Locked samples and every phase endpoint retain the reference qpos exactly.
    Editable samples are solved together with fixed-iteration damped least
    squares, keeping storage linear in ``T * (DOF + task_dim)``.

    Args:
        trajectory: Task-space target aligned with ``reference``.
        reference: Full-joint seed and locked-sample source.
        kinematics: Batched FK/Jacobian callable over controlled joints.
        joint_limits: Full ordered lower/upper limits with shape ``(D, 2)``.
        ik_cfg: Optional fixed-step DLS configuration.

    Returns:
        Final qpos iterate and per-sample residual evidence. Failed samples are
        retained for diagnostics and must not be executed as expert data.
    """
    if not isinstance(trajectory, TaskSpaceTrajectory):
        raise TypeError("trajectory must be a TaskSpaceTrajectory")
    if not isinstance(reference, TrajectoryTemplate):
        raise TypeError("reference must be a TrajectoryTemplate")
    if not callable(kinematics):
        raise TypeError("kinematics must be callable")
    if (
        trajectory.source_id != reference.source_id
        or trajectory.source_revision != reference.source_revision
        or trajectory.template_id != reference.template_id
    ):
        raise ValueError("task-space and joint trajectories must share source identity")
    if trajectory.poses.shape[0] != reference.positions.shape[0]:
        raise ValueError("task-space and joint trajectories must share sample count")
    if (
        trajectory.dt.dtype != reference.dt.dtype
        or trajectory.dt.device != reference.dt.device
    ):
        raise ValueError("task-space and joint timing must share dtype and device")
    if not torch.equal(trajectory.dt, reference.dt):
        raise ValueError("task-space and joint trajectories must share exact timing")
    _aligned_phases(trajectory, reference)
    limits = _tensor(joint_limits, "joint_limits", 2).to(
        device=reference.positions.device, dtype=reference.positions.dtype
    )
    if limits.shape != (reference.positions.shape[1], 2) or not bool(
        (limits[:, 1] > limits[:, 0]).all()
    ):
        raise ValueError("joint_limits must be finite increasing intervals (D, 2)")
    if (
        trajectory.poses.dtype != reference.positions.dtype
        or trajectory.poses.device != reference.positions.device
    ):
        raise ValueError(
            "task-space targets and joint reference must share dtype/device"
        )
    indices = list(reference.controlled_joint_indices)
    if not indices:
        raise ValueError("trajectory IK requires controlled joint indices")
    editable = torch.zeros(
        reference.positions.shape[0],
        dtype=torch.bool,
        device=reference.positions.device,
    )
    for phase in trajectory.phases:
        if TASK_SPACE_RESIDUAL_OPERATOR in phase.allowed_operators:
            editable[phase.start_index + 1 : phase.stop_index - 1] = True
    q = reference.positions.clone()
    controlled = q[:, indices]
    selected = torch.nonzero(editable, as_tuple=False).flatten()
    cfg = DifferentiableIKCfg() if ik_cfg is None else ik_cfg
    if not isinstance(cfg, DifferentiableIKCfg):
        raise TypeError("ik_cfg must be a DifferentiableIKCfg")
    if selected.numel():
        projected = solve_dls_ik(
            trajectory.poses.index_select(0, selected),
            controlled.index_select(0, selected),
            kinematics,
            limits[indices, 0],
            limits[indices, 1],
            cfg,
        )
        controlled = controlled.clone()
        controlled[selected] = projected.qpos
        q[:, indices] = controlled
    achieved, jacobians = kinematics(q[:, indices])
    if achieved.shape != trajectory.poses.shape:
        raise ValueError("kinematics poses must match task-space targets")
    if jacobians.shape != (q.shape[0], 6, len(indices)):
        raise ValueError("kinematics Jacobians must have shape (T, 6, C)")
    position, rotation = pose_error(achieved, trajectory.poses)
    finite = (
        torch.isfinite(q).all(dim=-1)
        & torch.isfinite(position).all(dim=-1)
        & torch.isfinite(rotation).all(dim=-1)
    )
    success = (
        finite
        & (position.norm(dim=-1) < cfg.pos_eps)
        & (rotation.norm(dim=-1) < cfg.rot_eps)
        & ((q >= limits[:, 0]) & (q <= limits[:, 1])).all(dim=-1)
    )
    return TaskSpaceProjectionResult(
        template=replace(reference, positions=q),
        success=success,
        position_error=position,
        rotation_error=rotation,
    )
