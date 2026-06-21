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

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Union, TYPE_CHECKING

from embodichain.lab.sim.planners import PlanResult, PlanState, MoveType
from embodichain.utils import configclass

from embodichain.lab.sim.common import BatchEntity
from embodichain.utils import logger

from .affordance import Affordance

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator, MotionGenOptions
    from embodichain.lab.sim.objects import Robot


# =============================================================================
# ObjectSemantics
# =============================================================================


@dataclass
class ObjectSemantics:
    """Semantic information about interaction target.

    This class encapsulates all semantic and geometric information about
    an object needed for intelligent interaction planning.
    """

    affordance: Affordance
    """Affordance data (GraspPose, InteractionPoints, etc.)."""

    geometry: Dict[str, Any]
    """Non-affordance geometric metadata (e.g., bounding_box). Mesh tensors now live on AntipodalAffordance, not here."""

    properties: Dict[str, Any] = field(default_factory=dict)
    """Physical properties: mass, friction, etc."""

    label: str = "none"
    """Object category label (e.g., 'apple', 'bottle')."""

    entity: BatchEntity | None = None
    """Optional reference to the underlying simulation entity representing this object."""

    def __post_init__(self) -> None:
        """Bind affordance metadata to this semantic object."""
        self.affordance.object_label = self.label


@dataclass
class HeldObjectState:
    """State shared by actions while an object is held by the robot."""

    semantics: ObjectSemantics
    """Semantic object currently held by the gripper."""

    object_to_eef: torch.Tensor
    """Batched transform from object frame to end-effector frame, shape [B, 4, 4]."""

    grasp_xpos: torch.Tensor
    """Batched end-effector grasp pose selected during pickup, shape [B, 4, 4]."""


@dataclass
class MoveObjectTarget:
    """Object-centric target for moving a held object without releasing it."""

    object_target_pose: torch.Tensor
    """Target object pose, shape [4, 4] or [B, 4, 4]."""


# =============================================================================
# ActionCfg and AtomicAction
# =============================================================================


@configclass
class ActionCfg:
    """Configuration for atomic actions."""

    name: str = "default"
    """Name of the action, used for identification and logging."""

    control_part: str = "arm"
    """Control part name for the action."""

    interpolation_type: str = "linear"
    """Interpolation type: 'linear', 'cubic'."""

    velocity_limit: Optional[float] = None
    """Optional velocity limit for the motion."""

    acceleration_limit: Optional[float] = None
    """Optional acceleration limit for the motion."""


class AtomicAction(ABC):
    """Abstract base class for atomic actions.

    All atomic actions use PlanResult from embodichain.lab.sim.planners
    as the return type for execute() method, ensuring consistency with
    the existing motion planning infrastructure.
    """

    updates_held_object_state: ClassVar[bool] = False
    """Whether the engine should read held-object state after this action."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: ActionCfg = ActionCfg(),
    ):
        """
        Initialize the atomic action.
        Args:
            motion_generator: The motion generator instance to use for planning.
            cfg: Configuration for the action.
        """
        self.motion_generator = motion_generator
        self.cfg = cfg
        self.robot = motion_generator.robot
        self.control_part = cfg.control_part
        self.device = self.robot.device

    def get_held_object_state(self) -> HeldObjectState | None:
        """Return held-object state after execution if this action updates it."""
        return None

    @abstractmethod
    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list[float]]:
        """execute pick up action

        Args:
            target (ObjectSemantics): object semantics containing grasp affordance and entity information
            start_qpos (Optional[torch.Tensor], optional): Planning start qpos. Defaults to None.

        Returns:
            tuple[bool, torch.Tensor, list[float]]:
            is_success,
            trajectory of shape (n_envs, n_waypoints, dof),
            joint_ids corresponding to trajectory
        """

    @abstractmethod
    def validate(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> bool:
        """Validate if the action is feasible without executing.

        This method performs a quick feasibility check (e.g., IK solvability)
        without generating a full trajectory.

        Returns:
            True if action appears feasible, False otherwise
        """
        pass

    def _ik_solve(
        self, target_pose: torch.Tensor, qpos_seed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Solve IK for target pose.

        Args:
            target_pose: Target pose [4, 4]
            qpos_seed: Seed configuration [DOF]

        Returns:
            Joint configuration [DOF]

        Raises:
            RuntimeError: If IK fails to find a solution
        """
        if qpos_seed is None:
            qpos_seed = self.robot.get_qpos()

        success, qpos = self.robot.compute_ik(
            pose=target_pose.unsqueeze(0),
            qpos_seed=qpos_seed.unsqueeze(0),
            name=self.control_part,
        )

        if not success.all():
            raise RuntimeError(f"IK failed for target pose: {target_pose}")

        return qpos.squeeze(0)

    def _fk_compute(self, qpos: torch.Tensor) -> torch.Tensor:
        """Compute forward kinematics.

        Args:
            qpos: Joint configuration [DOF] or [B, DOF]

        Returns:
            End-effector pose [4, 4] or [B, 4, 4]
        """
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)

        xpos = self.robot.compute_fk(
            qpos=qpos,
            name=self.control_part,
            to_matrix=True,
        )

        return xpos.squeeze(0) if xpos.shape[0] == 1 else xpos

    def _apply_offset(self, pose: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Apply offset to pose in local frame.

        Args:
            pose: Base pose [N, 4, 4]
            offset: Offset in local frame [N, 3] or [3]

        Returns:
            Pose with offset applied [N, 4, 4]
        """
        if not len(pose.shape) == 3 or pose.shape[1:] != (4, 4):
            logger.log_error("pose must have shape [N, 4, 4]")
        if len(offset.shape) == 1:
            offset = offset.unsqueeze(0)
        if not len(offset.shape) == 2 or offset.shape[1] != 3:
            logger.log_error("offset must have shape [N, 3] or [3]")
        result = pose.clone()
        result[:, :3, 3] += offset
        return result

    def plan_trajectory(
        self,
        target_states: List[PlanState],
        options: Optional["MotionGenOptions"] = None,
    ) -> "PlanResult":
        """Plan trajectory using motion generator."""
        from embodichain.lab.sim.planners import MotionGenOptions

        if options is None:
            options = MotionGenOptions(control_part=self.control_part)
        return self.motion_generator.generate(target_states, options)
