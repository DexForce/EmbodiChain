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

"""Install solver compatibility corrections scoped to Action Engine."""

from __future__ import annotations

from collections.abc import Mapping
import functools
import threading
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.solvers import PytorchSolver, URSolver, URSolverCfg

__all__ = [
    "install_action_engine_solver_compat",
    "install_pytorch_solver_tcp_compat",
    "install_ur5_solver_frame_compat",
    "repair_action_engine_ur5_solver_cfg",
]

_PYTORCH_INSTALL_MARKER = "_action_engine_tcp_inverse_compat_installed"
_UR5_INSTALL_MARKER = "_action_engine_ur5_frame_compat_installed"
_UR5_ANALYTIC_TO_URDF_EE = np.eye(4, dtype=np.float32)
_UR5_ANALYTIC_TO_URDF_EE[0, 3] = -0.01
_UR_DH_FIELDS = ("d1", "a2", "a3", "d4", "d5", "d6")


def repair_action_engine_ur5_solver_cfg(robot_cfg: Any) -> int:
    """Repair stale UR10 DH defaults before Action Engine creates a UR5 robot.

    ``SolverCfg.from_dict`` constructs a UR10 config before assigning a
    non-default ``ur_type``. Generated UR5 Action Engine configs therefore
    reach the environment with UR10 DH values. Repair only that exact stale
    signature so explicitly calibrated parameters remain untouched.

    Args:
        robot_cfg: Robot configuration whose solver configs will be inspected.

    Returns:
        Number of unique solver configs repaired by this call.
    """
    configured = getattr(robot_cfg, "solver_cfg", None)
    candidates = (
        configured.values() if isinstance(configured, Mapping) else (configured,)
    )
    stale_defaults = URSolverCfg()
    stale_dh = tuple(float(getattr(stale_defaults, name)) for name in _UR_DH_FIELDS)

    repaired = 0
    visited: set[int] = set()
    for solver_cfg in candidates:
        cfg_id = id(solver_cfg)
        if cfg_id in visited:
            continue
        visited.add(cfg_id)
        if not isinstance(solver_cfg, URSolverCfg):
            continue
        ur_type = str(getattr(solver_cfg, "ur_type", ""))
        if ur_type != "ur5":
            continue
        current_dh = tuple(float(getattr(solver_cfg, name)) for name in _UR_DH_FIELDS)
        if not np.allclose(current_dh, stale_dh, rtol=0.0, atol=1.0e-12):
            continue
        canonical = URSolverCfg(ur_type=ur_type)
        for name in _UR_DH_FIELDS:
            setattr(solver_cfg, name, getattr(canonical, name))
        repaired += 1
    return repaired


def install_action_engine_solver_compat(robot: Any) -> int:
    """Install all solver corrections required by the Action Engine runtime."""
    return install_pytorch_solver_tcp_compat(robot) + install_ur5_solver_frame_compat(
        robot
    )


def install_pytorch_solver_tcp_compat(robot: Any) -> int:
    """Correct TCP inversion on every PytorchSolver owned by ``robot``.

    The shared solver currently transposes a rotation into an overlapping
    tensor view. This wrapper transforms the requested TCP pose with a proper
    matrix inverse, temporarily presents an identity TCP to the original
    implementation, and otherwise preserves its sampling and ranking behavior.

    Args:
        robot: Initialized robot containing its private solver registry.

    Returns:
        Number of solver instances wrapped by this call.
    """
    solvers = getattr(robot, "_solvers", None)
    if not isinstance(solvers, Mapping):
        return 0

    installed = 0
    visited: set[int] = set()
    for solver in solvers.values():
        solver_id = id(solver)
        if solver_id in visited:
            continue
        visited.add(solver_id)
        if not isinstance(solver, PytorchSolver) or bool(
            getattr(solver, _PYTORCH_INSTALL_MARKER, False)
        ):
            continue
        _wrap_solver(solver)
        installed += 1
    return installed


def _wrap_solver(solver: PytorchSolver) -> None:
    original_get_ik = solver.get_ik
    call_lock = threading.RLock()

    @functools.wraps(original_get_ik)
    def corrected_get_ik(
        target_xpos: torch.Tensor | np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        target = torch.as_tensor(
            target_xpos,
            dtype=torch.float32,
            device=solver.device,
        )
        tcp = torch.as_tensor(
            solver.tcp_xpos,
            dtype=torch.float32,
            device=solver.device,
        )
        link_target = target @ torch.linalg.inv(tcp)

        # The solver instance is shared by vectorized environments. Protect the
        # temporary TCP substitution in case a caller plans from another thread.
        with call_lock:
            active_tcp = solver.tcp_xpos
            solver.tcp_xpos = np.eye(4, dtype=np.float32)
            try:
                return original_get_ik(
                    target_xpos=link_target,
                    *args,
                    **kwargs,
                )
            finally:
                solver.tcp_xpos = active_tcp

    solver.get_ik = corrected_get_ik
    setattr(solver, _PYTORCH_INSTALL_MARKER, True)


def install_ur5_solver_frame_compat(robot: Any) -> int:
    """Align UR5 analytic IK targets with the URDF ``ee_link`` frame.

    The UR5 asset carries a fixed ``-0.01 m`` local-x offset on ``ee_link``
    that is absent from the analytic DH model. The correction is installed
    only for UR5 solvers owned by an Action Engine environment.

    Args:
        robot: Initialized robot containing its private solver registry.

    Returns:
        Number of solver instances wrapped by this call.
    """
    solvers = getattr(robot, "_solvers", None)
    if not isinstance(solvers, Mapping):
        return 0

    installed = 0
    visited: set[int] = set()
    for solver in solvers.values():
        solver_id = id(solver)
        if solver_id in visited:
            continue
        visited.add(solver_id)
        if (
            not isinstance(solver, URSolver)
            or str(getattr(getattr(solver, "cfg", None), "ur_type", "")) != "ur5"
            or bool(getattr(solver, _UR5_INSTALL_MARKER, False))
        ):
            continue
        _wrap_ur5_solver(solver)
        installed += 1
    return installed


def _wrap_ur5_solver(solver: URSolver) -> None:
    original_get_ik = solver.get_ik

    @functools.wraps(original_get_ik)
    def corrected_get_ik(
        target_xpos: torch.Tensor | np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        target = torch.as_tensor(
            target_xpos,
            dtype=torch.float32,
            device=solver.device,
        )
        tcp = torch.as_tensor(
            solver.tcp_xpos,
            dtype=torch.float32,
            device=solver.device,
        )
        analytic_to_urdf = torch.as_tensor(
            _UR5_ANALYTIC_TO_URDF_EE,
            dtype=torch.float32,
            device=solver.device,
        )
        corrected_target = (
            target @ torch.linalg.inv(tcp) @ torch.linalg.inv(analytic_to_urdf) @ tcp
        )
        return original_get_ik(corrected_target, *args, **kwargs)

    solver.get_ik = corrected_get_ik
    setattr(solver, _UR5_INSTALL_MARKER, True)
