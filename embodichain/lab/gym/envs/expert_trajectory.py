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

"""Expert action encoding at the Gym and dataset boundary.

Trajectory timing remains owned by :class:`PlanResult` and the shared compute
trajectory helpers.  The legacy ``ExpertJointTrajectory`` wrapper is accepted
by the preparation adapter for compatibility, but new callers should pass a
``PlanResult`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

import torch
from tensordict import TensorDictBase

from embodichain.compute.trajectory import retime_to_control_grid
from embodichain.lab.sim.motion.planners import PlanResult
from embodichain.utils import configclass

JointCommandMode = Literal["position", "position_velocity"]
EXPERT_TRAJECTORY_SCHEMA_VERSION = 1

__all__ = [
    "EXPERT_TRAJECTORY_SCHEMA_VERSION",
    "ExpertActionSpec",
    "ExpertTrajectoryCfg",
    "JointCommandMode",
    "build_expert_action_spec",
    "encode_expert_action",
    "prepare_expert_joint_trajectory",
]


def _validate_joint_command_mode(value: object) -> JointCommandMode:
    if value not in ("position", "position_velocity"):
        raise ValueError(
            "joint_command_mode must be 'position' or 'position_velocity'."
        )
    return cast(JointCommandMode, value)


@configclass
class ExpertTrajectoryCfg:
    """Environment-owned expert trajectory execution and recording settings.

    Args:
        joint_command_mode: Position-only or position-velocity expert targets.
    """

    joint_command_mode: JointCommandMode = "position"
    """Joint targets emitted and stored by expert trajectory generation."""

    def __post_init__(self) -> None:
        self.joint_command_mode = _validate_joint_command_mode(self.joint_command_mode)


@dataclass(frozen=True, slots=True)
class ExpertActionSpec:
    """Canonical flat action layout used by expert trajectory persistence.

    Args:
        joint_command_mode: Position-only or position-velocity storage mode.
        joint_names: Active joint names in stored column order.
    """

    joint_command_mode: JointCommandMode
    joint_names: tuple[str, ...]

    @property
    def width(self) -> int:
        """Number of scalar values in one stored expert action."""
        multiplier = 2 if self.joint_command_mode == "position_velocity" else 1
        return len(self.joint_names) * multiplier

    @property
    def qpos_slice(self) -> tuple[int, int]:
        """Half-open qpos slice within the stored action vector."""
        return (0, len(self.joint_names))

    @property
    def qvel_slice(self) -> tuple[int, int] | None:
        """Half-open qvel slice, or ``None`` for position-only actions."""
        if self.joint_command_mode == "position":
            return None
        return (len(self.joint_names), 2 * len(self.joint_names))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered per-dimension names for dataset feature metadata."""
        if self.joint_command_mode == "position":
            return self.joint_names
        return tuple(f"{name}.position" for name in self.joint_names) + tuple(
            f"{name}.velocity" for name in self.joint_names
        )

    def metadata(self, *, step_dt: float) -> dict[str, object]:
        """Return serializable metadata describing this expert action schema.

        Args:
            step_dt: Environment command period associated with stored actions.

        Returns:
            JSON-compatible schema metadata.
        """
        return {
            "expert_trajectory_schema_version": EXPERT_TRAJECTORY_SCHEMA_VERSION,
            "joint_command_mode": self.joint_command_mode,
            "step_dt": float(step_dt),
            "joint_names": list(self.joint_names),
            "qpos_slice": list(self.qpos_slice),
            "qvel_slice": (None if self.qvel_slice is None else list(self.qvel_slice)),
        }


def build_expert_action_spec(
    *,
    joint_names: Sequence[str],
    joint_command_mode: JointCommandMode,
) -> ExpertActionSpec:
    """Build the immutable stored-action layout for one environment.

    Args:
        joint_names: Active robot joint names in stored column order.
        joint_command_mode: Position-only or position-velocity storage mode.

    Returns:
        Validated expert action layout.

    Raises:
        ValueError: If the mode is unsupported or a joint name is empty.
    """
    mode = _validate_joint_command_mode(joint_command_mode)
    names = tuple(joint_names)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("joint_names must contain non-empty strings.")
    return ExpertActionSpec(joint_command_mode=mode, joint_names=names)


def prepare_expert_joint_trajectory(
    trajectory: PlanResult,
    *,
    control_dt: float,
    joint_command_mode: JointCommandMode,
) -> PlanResult:
    """Prepare a planner result for a destination's fixed command clock.

    Timed trajectories are retimed to ``control_dt`` and their velocities are
    recomputed. Untimed position-velocity trajectories must provide velocity
    samples explicitly.

    Args:
        trajectory: Planner result with positions and explicit timing.
        control_dt: Positive destination command period in seconds.
        joint_command_mode: Position-only or position-velocity output mode.

    Returns:
        A detached trajectory aligned with the destination command clock.

    Raises:
        TypeError: If ``trajectory`` has the wrong type.
        ValueError: If timing, mode, or required velocity data is invalid.
    """
    if not isinstance(trajectory, PlanResult):
        raise TypeError("trajectory must be a PlanResult.")
    mode = _validate_joint_command_mode(joint_command_mode)
    if (
        isinstance(control_dt, bool)
        or not isinstance(control_dt, (int, float))
        or not torch.isfinite(torch.tensor(float(control_dt)))
        or control_dt <= 0
    ):
        raise ValueError("control_dt must be a positive finite number.")

    if trajectory.positions is None or trajectory.dt is None:
        raise ValueError("PlanResult must contain positions and explicit dt.")
    positions, velocities, dt, _ = retime_to_control_grid(
        trajectory.positions, trajectory.dt, control_dt
    )
    return PlanResult(
        success=trajectory.success,
        xpos_list=None,
        positions=positions,
        velocities=velocities if mode == "position_velocity" else None,
        accelerations=None,
        dt=dt,
    )


def encode_expert_action(
    action: torch.Tensor | TensorDictBase,
    *,
    spec: ExpertActionSpec,
    active_joint_ids: Sequence[int],
) -> torch.Tensor:
    """Flatten effective qpos/qvel targets into the expert dataset layout.

    Args:
        action: Effective controller action after validation and row masking.
        spec: Destination expert action layout.
        active_joint_ids: Full-robot columns represented by ``spec``.

    Returns:
        Active-joint qpos, or concatenated active-joint ``[qpos, qvel]``.

    Raises:
        TypeError: If the action or its joint targets have invalid types.
        ValueError: If required targets are missing or cannot be mapped to the
            configured active joints.
    """
    if not isinstance(spec, ExpertActionSpec):
        raise TypeError("spec must be an ExpertActionSpec.")
    joint_ids = tuple(active_joint_ids)
    if len(joint_ids) != len(spec.joint_names):
        raise ValueError("active_joint_ids must match the expert joint count.")
    if any(type(joint_id) is not int or joint_id < 0 for joint_id in joint_ids):
        raise ValueError("active_joint_ids must contain non-negative integers.")

    if isinstance(action, torch.Tensor):
        if spec.joint_command_mode == "position_velocity":
            raise ValueError("position_velocity expert actions must include qvel.")
        qpos = action
        qvel = None
    elif isinstance(action, TensorDictBase):
        if "qpos" not in action:
            raise ValueError("Expert controller action must include qpos.")
        qpos = action["qpos"]
        qvel = action.get("qvel")
        if spec.joint_command_mode == "position_velocity" and qvel is None:
            raise ValueError("position_velocity expert actions must include qvel.")
    else:
        raise TypeError("Expert action must be a torch.Tensor or TensorDict.")

    def select_active(value: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"Expert {name} target must be a floating tensor.")
        if value.dim() < 1:
            raise ValueError(f"Expert {name} target must have a feature dimension.")
        if not torch.isfinite(value).all():
            raise ValueError(f"Expert {name} target must contain only finite values.")
        if value.shape[-1] == len(joint_ids):
            return value
        if not joint_ids or max(joint_ids) >= value.shape[-1]:
            raise ValueError(
                f"Expert {name} target cannot select active_joint_ids from width "
                f"{value.shape[-1]}."
            )
        columns = torch.tensor(joint_ids, dtype=torch.long, device=value.device)
        return value.index_select(-1, columns)

    qpos = select_active(qpos, "qpos")
    if spec.joint_command_mode == "position":
        return qpos
    assert qvel is not None
    qvel = select_active(qvel, "qvel")
    if qpos.shape != qvel.shape or qpos.device != qvel.device:
        raise ValueError("Expert qpos and qvel targets must share shape and device.")
    return torch.cat((qpos, qvel), dim=-1)
