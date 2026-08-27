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

"""Generation-time IK solver selection for GenSim robot bundles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

__all__ = [
    "IK_SOLVER_MODES",
    "expected_ik_solver_class",
    "resolve_ik_solver_mode",
    "validate_robot_ik_solver_contract",
]

IK_SOLVER_MODES: Final = ("auto", "ur", "pytorch")
_UR_PROFILES = frozenset({"dual_ur3", "dual_ur5", "dual_ur10"})
_SUPPORTED_PROFILES = _UR_PROFILES | {"dual_franka"}
_CLASS_BY_MODE = {"ur": "URSolver", "pytorch": "PytorchSolver"}


def resolve_ik_solver_mode(mode: str, robot_profile: str) -> str:
    """Resolve one requested mode to a concrete solver for a robot profile."""
    if not isinstance(mode, str):
        raise TypeError(
            "IK solver mode must be a string; expected one of: auto, ur, pytorch."
        )
    if mode not in IK_SOLVER_MODES:
        raise ValueError(
            f"Unsupported IK solver mode {mode!r}; expected one of: "
            "auto, ur, pytorch."
        )
    profile = str(robot_profile)
    if profile not in _SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported IK solver robot profile {profile!r}.")
    resolved = "pytorch" if mode == "auto" and profile == "dual_franka" else mode
    if resolved == "auto":
        resolved = "ur"
    if resolved == "ur" and profile == "dual_franka":
        raise ValueError("Franka does not support the analytical URSolver.")
    return resolved


def expected_ik_solver_class(mode: str) -> str:
    """Return the serialized/runtime class name for one concrete mode."""
    if mode not in _CLASS_BY_MODE:
        raise ValueError("Concrete IK solver mode must be 'ur' or 'pytorch'.")
    return _CLASS_BY_MODE[mode]


def validate_robot_ik_solver_contract(
    robot: Mapping[str, Any],
    mode: str,
) -> None:
    """Validate that both generated arms use the declared concrete solver."""
    expected = expected_ik_solver_class(mode)
    solvers = robot.get("solver_cfg")
    if not isinstance(solvers, Mapping):
        raise ValueError("Generated robot requires a solver_cfg mapping.")
    for arm in ("left_arm", "right_arm"):
        solver = solvers.get(arm)
        if not isinstance(solver, Mapping):
            raise ValueError(f"Generated robot requires solver_cfg.{arm}.")
        actual = solver.get("class_type")
        if actual != expected:
            raise ValueError(
                f"Generated robot {arm} must use {expected} for ik_solver={mode!r}, "
                f"got {actual!r}."
            )
