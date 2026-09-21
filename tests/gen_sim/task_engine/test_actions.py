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

from embodichain.gen_sim.task_engine._task_program.actions import (
    GenSimMoveHeldObject,
    GenSimPour,
    jaw_tilt_axis,
)
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    MoveHeldObject,
)
from embodichain.lab.sim.atomic_actions.primitives.pour import Pour


def test_gensim_wrappers_keep_builtin_descriptors() -> None:
    assert GenSimMoveHeldObject.descriptor() == MoveHeldObject.descriptor()
    assert GenSimPour.descriptor() == Pour.descriptor()


def test_visible_tilt_direction_is_perpendicular_to_actual_jaw_line() -> None:
    jaw = torch.tensor([[0.0, 1.0, 0.0]])
    up = torch.tensor([[0.0, 0.0, 1.0]])
    axis, direction, valid = jaw_tilt_axis(jaw, up)
    assert valid.all()
    assert torch.abs((direction * jaw).sum(-1)).item() < 1e-6
    assert torch.abs((axis * jaw).sum(-1)).item() > 0.999
    assert torch.abs((direction * up).sum(-1)).item() < 1e-6


def test_degenerate_vertical_jaw_is_rejected_by_axis_mask() -> None:
    axis, direction, valid = jaw_tilt_axis(
        torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[0.0, 0.0, 1.0]])
    )
    assert not valid.all()
    assert torch.isfinite(axis).all() and torch.isfinite(direction).all()
