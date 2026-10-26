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

"""Planning inset and physical drawer containment have different meanings."""

from __future__ import annotations

import numpy as np
import pytest

from embodichain.gen_sim.task_engine._task_program.drawer_geometry import (
    drawer_containment_report,
)


def _report(center: list[float], **kwargs) -> dict:
    pose = np.eye(4)
    pose[:3, 3] = center
    return drawer_containment_report(
        np.array([[-0.03, -0.03, -0.03], [0.03, 0.03, 0.03]]),
        pose,
        np.eye(4),
        np.array([-0.1, -0.1, 0.0]),
        np.array([0.1, 0.1, 0.1]),
        **kwargs,
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sign", [-1, 1])
def test_close_allows_real_interior_without_planning_inset(axis, sign):
    center = [0.0, 0.0, 0.03]
    center[axis] = sign * (0.1 - 0.03 - 0.0028)
    assert not _report(center)["accepted"]
    physical = _report(center, lateral_margin=0.0)
    assert physical["accepted"]
    assert physical["margin"] == 0.005
    assert physical["lateral_margin"] == 0.0
    assert physical["safe_lower"][2] == -0.005
    assert physical["safe_upper"][2] == 0.095


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sign", [-1, 1])
def test_real_boundary_overflow_still_rejects(axis, sign):
    center = [0.0, 0.0, 0.03]
    center[axis] = sign * (0.1 - 0.03 + 0.001)
    assert not _report(center, lateral_margin=0.0)["accepted"]


@pytest.mark.parametrize("z", [0.024, 0.076])
def test_floor_penetration_and_top_overflow_still_reject(z):
    assert not _report([0.0, 0.0, z], lateral_margin=0.0)["accepted"]


@pytest.mark.parametrize("margin", [-0.001, float("nan"), float("inf")])
def test_invalid_lateral_margin_rejects(margin):
    with pytest.raises(ValueError, match="lateral margin"):
        _report([0.0, 0.0, 0.03], lateral_margin=margin)
