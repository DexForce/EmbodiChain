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

import torch

import functools
from abc import ABC, abstractmethod
from dataclasses import MISSING

from embodichain.utils import logger
from embodichain.utils import configclass
from embodichain.lab.sim.sim_manager import SimulationManager
from .utils import PlanState, PlanResult

__all__ = ["BasePlannerCfg", "PlanOptions", "BasePlanner", "validate_plan_options"]


@configclass
class BasePlannerCfg:

    robot_uid: str = MISSING
    """UID of the robot to control. Must correspond to a robot added to the simulation with this UID."""

    planner_type: str = "base"


@configclass
class PlanOptions:
    pass


def _infer_batch_size(target_states: list[PlanState]) -> int | None:
    for s in target_states:
        for t in (s.qpos, s.xpos, s.qvel, s.qacc):
            if isinstance(t, torch.Tensor) and t.dim() >= 1:
                return int(t.shape[0])
    return None


def _check_batch_consistency(
    target_states: list[PlanState],
    expected_b: int | None,
    robot_num_instances: int,
) -> int:
    """Validate that all PlanState tensors share the same leading B and match the robot."""
    if len(target_states) < 2:
        logger.log_error(
            "target_states must contain at least 2 waypoints", ValueError
        )
    bs = set()
    for i, s in enumerate(target_states):
        b = _infer_batch_size([s])
        if b is not None:
            bs.add(b)
    if len(bs) > 1:
        logger.log_error(
            f"All PlanState entries must share the same batch dim B, got {sorted(bs)}",
            ValueError,
        )
    b = bs.pop() if bs else 1
    if expected_b is not None and b != expected_b:
        logger.log_error(
            f"Batch dim B={b} does not match robot.num_instances={expected_b}",
            ValueError,
        )
    if robot_num_instances is not None and b not in (1, robot_num_instances):
        logger.log_error(
            f"Batch dim B={b} must be 1 or robot.num_instances={robot_num_instances}",
            ValueError,
        )
    return b


def validate_plan_options(_func=None, *, options_cls: type = PlanOptions):
    """Decorator (factory) that validates the ``options`` argument is a ``PlanOptions`` instance.

    Supports three usage styles:

    .. code-block:: python

        # 1. Bare decorator — validates against PlanOptions (default)
        @validate_plan_options
        def plan(self, target_states, options=PlanOptions()): ...

        # 2. Called with no arguments — same as above
        @validate_plan_options()
        def plan(self, target_states, options=PlanOptions()): ...

        # 3. Custom options class — useful in BasePlanner subclasses
        @validate_plan_options(options_cls=MyPlanOptions)
        def plan(self, target_states, options=MyPlanOptions()): ...

    Args:
        _func: Populated automatically when used as a bare decorator (no parentheses).
        options_cls: The expected type for the ``options`` argument. Subclasses of
            this type are also accepted. Defaults to :class:`PlanOptions`.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            options = kwargs.get("options", args[1] if len(args) > 1 else None)
            if options is not None and not isinstance(options, options_cls):
                logger.log_error(
                    f"Expected 'options' to be of type {options_cls.__name__} "
                    f"(or a subclass), but got {type(options).__name__}.",
                    TypeError,
                )
            target_states = kwargs.get("target_states", args[0] if args else None)
            if target_states is not None and hasattr(self, "robot"):
                robot_num = getattr(self.robot, "num_instances", None)
                _check_batch_consistency(target_states, expected_b=robot_num, robot_num_instances=robot_num)
            return func(self, *args, **kwargs)

        return wrapper

    if _func is not None:
        # Used as @validate_plan_options (no parentheses) — decorate immediately.
        return decorator(_func)
    # Used as @validate_plan_options() or @validate_plan_options(options_cls=...).
    return decorator


class BasePlanner(ABC):
    r"""Base class for trajectory planners.

    This class provides common functionality that can be shared across different
    planner implementations.

    Args:
        cfg: Configuration object for the planner.
    """

    def __init__(self, cfg: BasePlannerCfg):
        self.cfg: BasePlannerCfg = cfg

        if cfg.robot_uid is MISSING:
            logger.log_error("robot_uid is required in planner config", ValueError)

        self.robot = SimulationManager.get_instance().get_robot(cfg.robot_uid)
        if self.robot is None:
            logger.log_error(f"Robot {cfg.robot_uid} not found", ValueError)

        self.device = self.robot.device

    @validate_plan_options
    @abstractmethod
    def plan(
        self,
        target_states: list[PlanState],
        options: PlanOptions = PlanOptions(),
    ) -> PlanResult:
        r"""Execute trajectory planning.

        This method must be implemented by subclasses to provide the specific
        planning algorithm.

        Args:
            target_states: List of dictionaries containing target states

        Returns:
            PlanResult: An object containing:
                - success: bool, whether planning succeeded
                - positions: torch.Tensor (N, DOF), joint positions along trajectory
                - velocities: torch.Tensor (N, DOF), joint velocities along trajectory
                - accelerations: torch.Tensor (N, DOF), joint accelerations along trajectory
                - times: torch.Tensor (N,), time stamps for each point
                - duration: float, total trajectory duration
                - error_msg: Optional error message if planning failed
        """
        logger.log_error("Subclasses must implement plan() method", NotImplementedError)

    def is_satisfied_constraint(
        self, vels: torch.Tensor, accs: torch.Tensor, constraints: dict
    ) -> bool:
        r"""Check if the trajectory satisfies velocity and acceleration constraints.

        This method checks whether the given velocities and accelerations satisfy
        the constraints defined in constraints. It allows for some tolerance
        to account for numerical errors in dense waypoint scenarios.

        Args:
            vels: Velocity tensor (..., DOF) where the last dimension is DOF
            accs: Acceleration tensor (..., DOF) where the last dimension is DOF
            constraints: Dictionary containing 'velocity' and 'acceleration' limits

        Returns:
            bool: True if all constraints are satisfied, False otherwise

        Note:
            - Allows 10% tolerance for velocity constraints
            - Allows 25% tolerance for acceleration constraints
            - Prints exceed information if constraints are violated
            - Assumes symmetric constraints (velocities and accelerations can be positive or negative)
            - Supports batch dimension computation, e.g. (B, N, DOF) or (N, DOF)
        """
        device = vels.device

        max_vel = torch.tensor(constraints["velocity"], dtype=vels.dtype, device=device)
        max_acc = torch.tensor(
            constraints["acceleration"], dtype=accs.dtype, device=device
        )

        # To support batching, we compute along all dimensions except the last one (DOF)
        reduce_dims = tuple(range(vels.ndim - 1))

        # Check bounds
        vel_check = torch.all(torch.abs(vels) <= max_vel).item()
        acc_check = torch.all(torch.abs(accs) <= max_acc).item()

        if not vel_check:
            # max absolute value over all trajectory points and batches
            max_abs_vel = torch.amax(torch.abs(vels), dim=reduce_dims)
            exceed_percentage = torch.clamp((max_abs_vel - max_vel) / max_vel, min=0.0)
            vel_exceed_info = (exceed_percentage * 100).tolist()
            logger.log_info(f"Velocity exceed info: {vel_exceed_info} percentage")

        if not acc_check:
            max_abs_acc = torch.amax(torch.abs(accs), dim=reduce_dims)
            exceed_percentage = torch.clamp((max_abs_acc - max_acc) / max_acc, min=0.0)
            acc_exceed_info = (exceed_percentage * 100).tolist()
            logger.log_info(f"Acceleration exceed info: {acc_exceed_info} percentage")

        return vel_check and acc_check
