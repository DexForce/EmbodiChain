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

"""Install the narrow PytorchSolver TCP inverse compatibility correction."""

from __future__ import annotations

from collections.abc import Mapping
import functools
import threading
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.solvers import PytorchSolver

__all__ = ["install_pytorch_solver_tcp_compat"]

_INSTALL_MARKER = "_action_engine_tcp_inverse_compat_installed"


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
            getattr(solver, _INSTALL_MARKER, False)
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
    setattr(solver, _INSTALL_MARKER, True)
