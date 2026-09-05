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

import pytest

from embodichain.gen_sim.action_engine.solver_profiles import (
    IK_SOLVER_MODES,
    resolve_ik_solver_mode,
)


def test_auto_solver_preserves_current_robot_family_defaults() -> None:
    assert IK_SOLVER_MODES == ("auto", "ur", "pytorch")
    assert resolve_ik_solver_mode("auto", "dual_ur10") == "ur"
    assert resolve_ik_solver_mode("auto", "dual_ur5") == "ur"
    assert resolve_ik_solver_mode("auto", "dual_franka") == "pytorch"


def test_explicit_solver_mode_is_strict_and_profile_compatible() -> None:
    assert resolve_ik_solver_mode("pytorch", "dual_ur10") == "pytorch"
    assert resolve_ik_solver_mode("ur", "dual_ur10") == "ur"

    with pytest.raises(ValueError, match="Franka.*URSolver"):
        resolve_ik_solver_mode("ur", "dual_franka")
    for invalid in ("", "UR", "torch", None):
        with pytest.raises((TypeError, ValueError), match="auto.*ur.*pytorch"):
            resolve_ik_solver_mode(invalid, "dual_ur10")  # type: ignore[arg-type]
