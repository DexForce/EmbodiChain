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

import functools
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.solvers import PytorchSolver

__all__ = ["install_action_agent_pytorch_solver_compat"]

_INSTALL_MARKER = "_action_agent_tcp_inverse_compat_installed"


def install_action_agent_pytorch_solver_compat(robot: Any) -> int:
    """Install the Action Agent TCP inverse fix on robot Pytorch solvers.

    The Main implementation applies an overlapping in-place transpose while
    building the TCP inverse. Action Agent performs the correct TCP transform
    first, then invokes the original IK implementation with an identity TCP so
    the rest of its sampling and solution-selection behavior stays unchanged.

    Args:
        robot: Robot instance whose private solver map has already been created.

    Returns:
        Number of newly wrapped Pytorch solver instances.
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
        _wrap_pytorch_solver_get_ik(solver)
        installed += 1
    return installed


def _wrap_pytorch_solver_get_ik(solver: PytorchSolver) -> None:
    original_get_ik = solver.get_ik
    call_lock = threading.RLock()

    @functools.wraps(original_get_ik)
    def corrected_get_ik(
        target_xpos: torch.Tensor | np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        target_xpos_t = torch.as_tensor(
            target_xpos,
            dtype=torch.float32,
            device=solver.device,
        )
        tcp_xpos_t = torch.as_tensor(
            solver.tcp_xpos,
            dtype=torch.float32,
            device=solver.device,
        )
        link_target_xpos = target_xpos_t @ torch.linalg.inv(tcp_xpos_t)

        with call_lock:
            active_tcp_xpos = solver.tcp_xpos
            solver.tcp_xpos = np.eye(4, dtype=np.float32)
            try:
                return original_get_ik(
                    target_xpos=link_target_xpos,
                    *args,
                    **kwargs,
                )
            finally:
                solver.tcp_xpos = active_tcp_xpos

    solver.get_ik = corrected_get_ik
    setattr(solver, _INSTALL_MARKER, True)
