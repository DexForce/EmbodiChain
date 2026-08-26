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

"""Task-space inverse kinematics powered by Pink and Pinocchio."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from embodichain.lab.sim.solvers import BaseSolver, SolverCfg
from embodichain.lab.sim.utility.import_utils import (
    lazy_import_pinocchio,
    lazy_import_pink,
)
from embodichain.lab.sim.utility.solver_utils import (
    build_reduced_pinocchio_robot,
)
from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    import pink

__all__ = ["PinkSolver", "PinkSolverCfg"]


@configclass
class PinkSolverCfg(SolverCfg):
    """Configure the Pink task-space IK solver."""

    class_type: str = "PinkSolver"

    pos_eps: float = 5e-4
    """Position convergence tolerance in metres."""

    rot_eps: float = 5e-4
    """Orientation convergence tolerance in radians."""

    max_iterations: int = 1000
    """Maximum number of differential-IK iterations."""

    dt: float = 0.1
    """Integration timestep in seconds."""

    damp: float = 1e-6
    """Initial isotropic QP damping."""

    is_only_position_constraint: bool = False
    """Stop once position converges without requiring orientation convergence."""

    mesh_path: str | None = None
    """Optional directory containing URDF mesh assets."""

    variable_input_tasks: list["pink.tasks.Task"] | None = None
    """Tasks whose first frame target is updated by :meth:`PinkSolver.get_ik`."""

    fixed_input_tasks: list["pink.tasks.Task"] | None = None
    """Tasks initialized once and kept fixed during IK calls."""

    show_ik_warnings: bool = True
    """Log solver exceptions and non-convergence warnings."""

    fail_on_joint_limit_violation: bool = True
    """Enable Pink's joint-limit safety break."""

    solver_type: str = "osqp"
    """QP backend passed to :func:`pink.solve_ik`."""

    stagnation_tolerance: float = 1e-10
    """Minimum accepted objective improvement before an iteration stagnates."""

    stagnation_iterations: int = 8
    """Consecutive stagnant iterations before terminating."""

    max_backtracks: int = 4
    """Maximum damping/backtracking retries for a non-improving step."""

    damping_growth: float = 10.0
    """Multiplier applied after a rejected step."""

    damping_decay: float = 0.5
    """Multiplier applied after an accepted step."""

    max_damping: float = 1e6
    """Upper bound for adaptive damping."""

    def init_solver(self, device: torch.device, **kwargs: Any) -> PinkSolver:
        """Create a Pink solver and apply the configured TCP.

        Args:
            device: Torch device used by the solver.
            **kwargs: Arguments forwarded to :class:`PinkSolver`.

        Returns:
            Initialized Pink solver.
        """
        solver = PinkSolver(cfg=self, device=device, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


class PinkSolver(BaseSolver):
    """Iterative task-space IK with adaptive damping and convergence checks."""

    def __init__(self, cfg: PinkSolverCfg, **kwargs: Any) -> None:
        """Initialize Pinocchio, Pink, task state, and joint ordering.

        Args:
            cfg: Pink solver configuration.
            **kwargs: Arguments forwarded to :class:`BaseSolver`.
        """
        self.cfg = cfg
        self._validate_cfg()
        self._configured_lower_limits: torch.Tensor | None = None
        self._configured_upper_limits: torch.Tensor | None = None
        self._runtime_robot_lower_limits: torch.Tensor | None = None
        self._runtime_robot_upper_limits: torch.Tensor | None = None
        super().__init__(cfg=cfg, **kwargs)
        self.pin = lazy_import_pinocchio()
        self.pink = lazy_import_pink()

        from embodichain.lab.sim.solvers.null_space_posture_task import (
            NullSpacePostureTask,
        )

        mesh_path = cfg.mesh_path or os.path.dirname(cfg.urdf_path)
        self.entire_robot = self.pin.RobotWrapper.BuildFromURDF(
            cfg.urdf_path, mesh_path, root_joint=None
        )
        self.robot = build_reduced_pinocchio_robot(self.entire_robot, self.joint_names)
        self.pink_cfg = self.pink.configuration.Configuration(
            self.robot.model, self.robot.data, self.robot.q0
        )
        self.init_qpos = np.asarray(self.robot.q0, dtype=float).copy()
        self.pin.framesForwardKinematics(
            self.robot.model, self.robot.data, self.init_qpos
        )
        if self.root_link_name is None:
            self._world_from_root = np.eye(4)
        else:
            root_frame_id = self.robot.model.getFrameId(self.root_link_name)
            if root_frame_id >= self.robot.model.nframes:
                raise ValueError(
                    f"Root link name '{self.root_link_name}' is not in the Pink model"
                )
            # Pinocchio's oMf is ^world M_frame: it maps coordinates expressed
            # in the root frame into the world frame.
            self._world_from_root = np.asarray(
                self.robot.data.oMf[root_frame_id].homogeneous, dtype=float
            ).copy()
        self._root_from_world = np.linalg.inv(self._world_from_root)
        self._end_frame_id = self.robot.model.getFrameId(self.end_link_name)
        if self._end_frame_id >= self.robot.model.nframes:
            raise ValueError(
                f"End link name '{self.end_link_name}' is not in the Pink model"
            )

        if cfg.variable_input_tasks is None:
            orientation_cost = 0.0 if cfg.is_only_position_constraint else 1.0
            self.variable_input_tasks: list[Any] = [
                self.pink.tasks.FrameTask(
                    frame=cfg.end_link_name,
                    position_cost=1.0,
                    orientation_cost=orientation_cost,
                )
            ]
        else:
            self.variable_input_tasks = list(cfg.variable_input_tasks)
        self.fixed_input_tasks: list[Any] = list(cfg.fixed_input_tasks or [])
        self.tasks = self.variable_input_tasks + self.fixed_input_tasks
        self._frame_tasks = [
            task
            for task in self.variable_input_tasks
            if isinstance(task, self.pink.tasks.FrameTask)
        ]
        if not self._frame_tasks:
            raise ValueError("variable_input_tasks must contain at least one FrameTask")
        if len(self._frame_tasks) != 1:
            raise ValueError(
                "PinkSolver expects exactly one FrameTask in variable_input_tasks; "
                "additional FrameTask instances should be passed via fixed_input_tasks"
            )
        self._target_task = self._frame_tasks[0]
        self._frame_task_ids = {id(self._target_task)}

        pink_joint_names = self.robot.model.names.tolist()[1:]
        if self.joint_names:
            missing = set(self.joint_names).difference(pink_joint_names)
            if missing:
                raise ValueError(f"Pink model is missing joints: {sorted(missing)}")
            self.dexsim_to_pink_ordering = np.asarray(
                [self.joint_names.index(name) for name in pink_joint_names], dtype=int
            )
            self.pink_to_dexsim_ordering = np.asarray(
                [pink_joint_names.index(name) for name in self.joint_names], dtype=int
            )
        else:
            self.dexsim_to_pink_ordering = None
            self.pink_to_dexsim_ordering = None

        if self.robot.model.nq != self.dof or self.robot.model.nv != self.dof:
            raise ValueError(
                "PinkSolver currently requires one configuration and velocity "
                "coordinate per controlled joint"
            )
        self._urdf_model_lower = np.asarray(
            self.robot.model.lowerPositionLimit, dtype=float
        ).copy()
        self._urdf_model_upper = np.asarray(
            self.robot.model.upperPositionLimit, dtype=float
        ).copy()
        self._sync_effective_limits()
        self.init_qpos = self._project_model_limits(self.init_qpos)
        self.pink_cfg.update(self.init_qpos)

        for task in self.variable_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(self.init_qpos)
            else:
                task.set_target_from_configuration(self.pink_cfg)
        for task in self.fixed_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(self.init_qpos)
            else:
                task.set_target_from_configuration(self.pink_cfg)

        self._tcp_inverse = np.linalg.inv(np.asarray(self.tcp_xpos, dtype=float))

    def _validate_cfg(self) -> None:
        """Validate numerical controls before constructing optional dependencies."""
        if not self.cfg.joint_names:
            raise ValueError("joint_names must contain at least one controlled joint")
        if not self.cfg.end_link_name:
            raise ValueError("end_link_name must be configured")
        positive = {
            "pos_eps": self.cfg.pos_eps,
            "rot_eps": self.cfg.rot_eps,
            "max_iterations": self.cfg.max_iterations,
            "dt": self.cfg.dt,
            "stagnation_iterations": self.cfg.stagnation_iterations,
            "damping_growth": self.cfg.damping_growth,
            "max_damping": self.cfg.max_damping,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.cfg.damp) or self.cfg.damp < 0:
            raise ValueError("damp must be finite and non-negative")
        if (
            not np.isfinite(self.cfg.stagnation_tolerance)
            or self.cfg.stagnation_tolerance < 0
        ):
            raise ValueError("stagnation_tolerance must be finite and non-negative")
        for name in ("max_iterations", "stagnation_iterations", "max_backtracks"):
            value = getattr(self.cfg, name)
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.cfg.max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative")
        if self.cfg.damping_growth <= 1:
            raise ValueError("damping_growth must be greater than 1")
        if not 0 < self.cfg.damping_decay <= 1:
            raise ValueError("damping_decay must be in the range (0, 1]")
        if self.cfg.max_damping < self.cfg.damp:
            raise ValueError("max_damping must be greater than or equal to damp")

    def set_tcp(self, xpos: np.ndarray) -> None:
        """Set the TCP and refresh its inverse used for IK targets.

        Args:
            xpos: Homogeneous end-frame-to-TCP transform.
        """
        tcp = np.asarray(xpos, dtype=float)
        if tcp.shape != (4, 4) or not np.isfinite(tcp).all():
            raise ValueError("TCP must be a finite 4x4 homogeneous matrix")
        tcp_inverse = np.linalg.inv(tcp)
        super().set_tcp(tcp)
        self._tcp_inverse = tcp_inverse

    def update_with_robot_limit(self, robot_qpos_limits: torch.Tensor) -> None:
        """Intersect robot limits and synchronize them with Pink.

        Args:
            robot_qpos_limits: Joint limits in simulator order with shape
                ``(dof, 2)``.
        """
        limits = torch.as_tensor(
            robot_qpos_limits, dtype=torch.float32, device=self.device
        )
        if limits.shape != (self.dof, 2):
            raise ValueError(
                f"robot_qpos_limits must have shape ({self.dof}, 2), "
                f"got {tuple(limits.shape)}"
            )
        if not torch.isfinite(limits).all() or torch.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("robot_qpos_limits must be finite and ordered")
        self._calculate_effective_limits(
            self._configured_lower_limits,
            self._configured_upper_limits,
            limits[:, 0],
            limits[:, 1],
        )
        self._runtime_robot_lower_limits = limits[:, 0].clone()
        self._runtime_robot_upper_limits = limits[:, 1].clone()
        self._sync_effective_limits()

    def set_qpos_limits(
        self,
        lower_qpos_limits: list[float] | np.ndarray | torch.Tensor,
        upper_qpos_limits: list[float] | np.ndarray | torch.Tensor,
    ) -> bool:
        """Set simulator-ordered limits and synchronize an initialized Pink model.

        Args:
            lower_qpos_limits: Lower limit for every controlled joint.
            upper_qpos_limits: Upper limit for every controlled joint.

        Returns:
            Whether the limits were accepted.
        """
        lower = torch.as_tensor(
            lower_qpos_limits, dtype=torch.float32, device=self.device
        )
        upper = torch.as_tensor(
            upper_qpos_limits, dtype=torch.float32, device=self.device
        )
        if lower.shape != (self.dof,) or upper.shape != (self.dof,):
            raise ValueError(
                f"qpos limits must both have shape ({self.dof},), got "
                f"{tuple(lower.shape)} and {tuple(upper.shape)}"
            )
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("qpos limits must contain only finite values")
        if torch.any(lower > upper):
            raise ValueError("lower qpos limits must not exceed upper limits")
        if hasattr(self, "_urdf_model_lower"):
            self._calculate_effective_limits(
                lower,
                upper,
                self._runtime_robot_lower_limits,
                self._runtime_robot_upper_limits,
            )
        self._configured_lower_limits = lower.clone()
        self._configured_upper_limits = upper.clone()
        updated = super().set_qpos_limits(lower, upper)
        if updated and hasattr(self, "_urdf_model_lower"):
            self._sync_effective_limits()
        return updated

    def _calculate_effective_limits(
        self,
        configured_lower: torch.Tensor | None,
        configured_upper: torch.Tensor | None,
        runtime_lower: torch.Tensor | None,
        runtime_upper: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate an effective limit intersection without mutating state."""
        lower = self._urdf_model_lower.copy()
        upper = self._urdf_model_upper.copy()
        if configured_lower is not None:
            configured_lower = self._to_pink_order(
                configured_lower.detach().cpu().numpy()
            )
            lower = np.maximum(lower, configured_lower)
        if configured_upper is not None:
            configured_upper = self._to_pink_order(
                configured_upper.detach().cpu().numpy()
            )
            upper = np.minimum(upper, configured_upper)
        if runtime_lower is not None:
            runtime_lower = self._to_pink_order(runtime_lower.detach().cpu().numpy())
            lower = np.maximum(lower, runtime_lower)
        if runtime_upper is not None:
            runtime_upper = self._to_pink_order(runtime_upper.detach().cpu().numpy())
            upper = np.minimum(upper, runtime_upper)
        if np.any(lower > upper):
            raise ValueError("Effective Pink joint limits have an empty intersection")
        return lower, upper

    def _sync_effective_limits(self) -> None:
        """Apply configured and robot-synchronized limits to the Pink model."""
        lower, upper = self._calculate_effective_limits(
            self._configured_lower_limits,
            self._configured_upper_limits,
            self._runtime_robot_lower_limits,
            self._runtime_robot_upper_limits,
        )
        self._model_lower = lower
        self._model_upper = upper
        self.robot.model.lowerPositionLimit[:] = lower
        self.robot.model.upperPositionLimit[:] = upper
        self.lower_qpos_limits = torch.as_tensor(
            self._to_output_order(lower), dtype=torch.float32, device=self.device
        )
        self.upper_qpos_limits = torch.as_tensor(
            self._to_output_order(upper), dtype=torch.float32, device=self.device
        )

    @staticmethod
    def reorder_array(
        input_array: Sequence[float], reordering_array: Sequence[int]
    ) -> np.ndarray:
        """Reorder an array with an index mapping.

        Args:
            input_array: Values to reorder.
            reordering_array: Source indices in output order.

        Returns:
            Reordered NumPy array.
        """
        return np.asarray(input_array)[np.asarray(reordering_array, dtype=int)]

    def update_null_space_joint_targets(
        self, current_qpos: torch.Tensor | np.ndarray
    ) -> None:
        """Update all null-space posture targets.

        Args:
            current_qpos: Joint target in simulator ordering.
        """
        from embodichain.lab.sim.solvers.null_space_posture_task import (
            NullSpacePostureTask,
        )

        if isinstance(current_qpos, torch.Tensor):
            current_qpos = current_qpos.detach().cpu().numpy()
        target = self._to_pink_order(np.asarray(current_qpos, dtype=float))
        if target.shape != (self.dof,) or not np.isfinite(target).all():
            raise ValueError(
                f"current_qpos must be a finite vector with shape ({self.dof},)"
            )
        for task in self.tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(target)

    def _normalize_inputs(
        self,
        target_xpos: torch.Tensor | np.ndarray,
        qpos_seed: torch.Tensor | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize targets and seeds into matched CPU batches."""
        targets = np.asarray(
            (
                target_xpos.detach().cpu().numpy()
                if isinstance(target_xpos, torch.Tensor)
                else target_xpos
            ),
            dtype=float,
        )
        if targets.shape == (4, 4):
            targets = targets[None]
        if targets.ndim != 3 or targets.shape[1:] != (4, 4):
            raise ValueError(
                f"target_xpos must have shape (4, 4) or (N, 4, 4), got {targets.shape}"
            )
        if not np.isfinite(targets).all():
            raise ValueError("target_xpos must contain only finite values")

        if qpos_seed is None:
            initial_seed = self._to_output_order(self.init_qpos)
            seeds = np.broadcast_to(initial_seed, (targets.shape[0], self.dof)).copy()
        else:
            seeds = np.asarray(
                (
                    qpos_seed.detach().cpu().numpy()
                    if isinstance(qpos_seed, torch.Tensor)
                    else qpos_seed
                ),
                dtype=float,
            )
            if seeds.shape == (self.dof,):
                seeds = np.broadcast_to(seeds, (targets.shape[0], self.dof)).copy()
            if seeds.ndim == 3 and seeds.shape[1] == 1:
                seeds = seeds[:, 0]
            if seeds.shape == (1, self.dof) and targets.shape[0] != 1:
                seeds = np.broadcast_to(seeds, (targets.shape[0], self.dof)).copy()
            if seeds.shape != (targets.shape[0], self.dof):
                raise ValueError(
                    f"qpos_seed must have shape ({self.dof},) or "
                    f"({targets.shape[0]}, {self.dof}), got {seeds.shape}"
                )
            if not np.isfinite(seeds).all():
                raise ValueError("qpos_seed must contain only finite values")
        return targets, seeds

    def _to_pink_order(self, qpos: np.ndarray) -> np.ndarray:
        """Convert simulator joint order to the reduced Pink model order."""
        if self.dexsim_to_pink_ordering is None:
            return qpos.copy()
        return qpos[self.dexsim_to_pink_ordering]

    def _to_output_order(self, qpos: np.ndarray) -> np.ndarray:
        """Convert Pink model joint order to simulator order."""
        if self.pink_to_dexsim_ordering is None:
            return qpos.copy()
        return qpos[self.pink_to_dexsim_ordering]

    def _project_model_limits(self, qpos: np.ndarray) -> np.ndarray:
        """Project Euclidean configurations into finite Pinocchio limits."""
        if self.robot.model.nq != self.robot.model.nv:
            return qpos
        lower = np.where(np.isfinite(self._model_lower), self._model_lower, -np.inf)
        upper = np.where(np.isfinite(self._model_upper), self._model_upper, np.inf)
        return np.clip(qpos, lower, upper)

    def _set_target(self, target_xpos: np.ndarray) -> None:
        """Set the controlled frame target, removing TCP when appropriate."""
        frame_target = self._world_from_root @ target_xpos
        if self._target_task.frame == self.end_link_name:
            frame_target = frame_target @ self._tcp_inverse
        self._target_task.set_target(self.pin.SE3(frame_target))

    def _task_metrics(self) -> tuple[float, float, float, float]:
        """Return lexicographic task merits and frame convergence errors."""
        from embodichain.lab.sim.solvers.null_space_posture_task import (
            NullSpacePostureTask,
        )

        primary_objective = 0.0
        secondary_objective = 0.0
        position_error = 0.0
        orientation_error = 0.0
        for task in self.tasks:
            error = np.asarray(task.compute_error(self.pink_cfg), dtype=float)
            cost = 1.0 if task.cost is None else np.asarray(task.cost, dtype=float)
            if isinstance(task, self.pink.tasks.FrameTask):
                weighted = cost * float(task.gain) * error
                primary_objective += 0.5 * float(weighted @ weighted)
            elif isinstance(task, NullSpacePostureTask):
                jacobian = np.asarray(task.compute_jacobian(self.pink_cfg), dtype=float)
                weighted_error = cost * float(task.gain) * error
                controllable_gradient = jacobian.T @ weighted_error
                secondary_objective += 0.5 * float(
                    controllable_gradient @ controllable_gradient
                )

            if id(task) in self._frame_task_ids:
                cost_vector = np.broadcast_to(cost, error.shape)
                active_position = cost_vector[:3] > 0.0
                active_orientation = cost_vector[3:] > 0.0
                position_error = max(
                    position_error,
                    float(np.linalg.norm(error[:3][active_position])),
                )
                orientation_error = max(
                    orientation_error,
                    float(np.linalg.norm(error[3:][active_orientation])),
                )
        return (
            primary_objective,
            secondary_objective,
            position_error,
            orientation_error,
        )

    def _converged(self, position_error: float, orientation_error: float) -> bool:
        """Return whether configured task tolerances are satisfied."""
        if position_error > self.cfg.pos_eps:
            return False
        return (
            self.cfg.is_only_position_constraint
            or orientation_error <= self.cfg.rot_eps
        )

    def _solve_one(
        self, target_xpos: np.ndarray, qpos_seed: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        """Solve one target with adaptive damping and backtracking."""
        self._set_target(target_xpos)
        pink_seed = self._to_pink_order(qpos_seed)
        self.pink_cfg.update(self._project_model_limits(pink_seed))
        damping_floor = min(
            self.cfg.max_damping,
            max(self.cfg.damp, float(np.sqrt(np.finfo(float).eps))),
        )
        damping = damping_floor
        stagnant = 0

        for _ in range(self.cfg.max_iterations):
            primary, secondary, position_error, orientation_error = self._task_metrics()
            if self._converged(position_error, orientation_error):
                return True, self._to_output_order(np.asarray(self.pink_cfg.q))

            base_q = np.asarray(self.pink_cfg.q).copy()
            accepted = False
            for backtrack in range(self.cfg.max_backtracks + 1):
                try:
                    velocity = self.pink.solve_ik(
                        configuration=self.pink_cfg,
                        tasks=self.tasks,
                        damping=damping,
                        dt=self.cfg.dt,
                        solver=self.cfg.solver_type,
                        safety_break=self.cfg.fail_on_joint_limit_violation,
                    )
                except Exception:
                    self.pink_cfg.update(base_q)
                    raise
                scale = 0.5**backtrack
                candidate = self.pin.integrate(
                    self.robot.model, base_q, velocity * self.cfg.dt * scale
                )
                self.pink_cfg.update(self._project_model_limits(candidate))
                candidate_primary, candidate_secondary, _, _ = self._task_metrics()
                primary_improvement = primary - candidate_primary
                primary_tied = abs(primary_improvement) <= np.finfo(float).eps * max(
                    1.0, abs(primary)
                )
                if primary_improvement > 0.0 or (
                    primary_tied and candidate_secondary < secondary
                ):
                    improvement = max(0.0, primary_improvement)
                    accepted = True
                    damping = max(damping_floor, damping * self.cfg.damping_decay)
                    stagnant = (
                        stagnant + 1
                        if improvement <= self.cfg.stagnation_tolerance
                        else 0
                    )
                    break
                self.pink_cfg.update(base_q)
                damping = min(
                    max(damping * self.cfg.damping_growth, damping_floor),
                    self.cfg.max_damping,
                )

            if not accepted:
                stagnant += 1
            if stagnant >= self.cfg.stagnation_iterations:
                break

        _, _, position_error, orientation_error = self._task_metrics()
        success = self._converged(position_error, orientation_error)
        return success, self._to_output_order(np.asarray(self.pink_cfg.q))

    def get_ik(
        self,
        target_xpos: torch.Tensor | np.ndarray,
        qpos_seed: torch.Tensor | np.ndarray | None = None,
        return_all_solutions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve one or more target poses sequentially.

        Args:
            target_xpos: Target TCP pose with shape ``(4, 4)`` or
                ``(N, 4, 4)``.
            qpos_seed: Joint seed with shape ``(dof,)``, ``(1, dof)``,
                ``(N, dof)``, or ``(N, 1, dof)``. A single seed is broadcast
                over the batch.
            return_all_solutions: Accepted for solver-interface compatibility;
                Pink returns one locally optimal solution per target.
            **kwargs: Reserved for future solver options.

        Returns:
            A success tensor with shape ``(N,)`` and joint solutions with shape
            ``(N, 1, dof)``. Failed targets return their corresponding seeds.
        """
        del kwargs
        targets, seeds = self._normalize_inputs(target_xpos, qpos_seed)
        success = np.zeros(targets.shape[0], dtype=bool)
        solutions = seeds.copy()
        for index, (target, seed) in enumerate(zip(targets, seeds)):
            try:
                solved, candidate = self._solve_one(target, seed)
                success[index] = solved
                if solved:
                    solutions[index] = candidate
                elif self.cfg.show_ik_warnings:
                    logger.log_warning(
                        f"Pink IK did not converge for target index {index}; returning its seed."
                    )
            except Exception as exc:
                if self.cfg.show_ik_warnings:
                    logger.log_warning(
                        f"Pink IK failed for target index {index}; returning its seed. Error: {exc}"
                    )
        if return_all_solutions:
            logger.log_warning(
                "return_all_solutions=True is not supported by PinkSolver; "
                "returning one local solution per target."
            )
        success_tensor = torch.as_tensor(success, dtype=torch.bool, device=self.device)
        solution_tensor = torch.as_tensor(
            solutions, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        return success_tensor, solution_tensor

    def _get_fk(
        self,
        qpos: torch.Tensor | np.ndarray,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Pinocchio FK for one configuration.

        Args:
            qpos: Joint configuration in simulator ordering.
            **kwargs: Reserved for solver-interface compatibility.

        Returns:
            Homogeneous TCP pose.
        """
        del kwargs
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu().numpy()
        configuration = np.asarray(qpos, dtype=float).squeeze()
        if configuration.shape != (self.dof,) or not np.isfinite(configuration).all():
            raise ValueError(f"qpos must be a finite vector with shape ({self.dof},)")
        configuration = self._to_pink_order(configuration)
        self.pin.framesForwardKinematics(
            self.robot.model, self.robot.data, configuration
        )
        world_from_end = np.asarray(
            self.robot.data.oMf[self._end_frame_id].homogeneous, dtype=float
        )
        result = self._root_from_world @ world_from_end @ self.tcp_xpos
        return torch.as_tensor(result, dtype=torch.float32, device=self.device)
