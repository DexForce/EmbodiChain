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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dexsim.types import DriveType

from embodichain.lab.sim.objects.articulation import Articulation

pytestmark = pytest.mark.no_sim


def test_newton_target_modes_map_to_portable_drive_types() -> None:
    target_modes = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    entity = SimpleNamespace(
        get_newton_drive=lambda: (None, None, None, None, None, None, target_modes)
    )
    articulation = object.__new__(Articulation)
    articulation._data = SimpleNamespace(
        is_newton_backend=True,
        dof=len(target_modes),
    )
    articulation._all_indices = np.asarray([0], dtype=np.int32)
    articulation._entities = [entity]

    assert articulation.get_joint_drive_type() == [
        [
            DriveType.NONE,
            DriveType.FORCE,
            DriveType.FORCE,
            DriveType.FORCE,
            DriveType.NONE,
        ]
    ]


def test_newton_drive_type_query_honors_joint_selection() -> None:
    target_modes = np.asarray([0, 3, 0], dtype=np.int32)
    entity = SimpleNamespace(
        get_newton_drive=lambda: (None, None, None, None, None, None, target_modes)
    )
    articulation = object.__new__(Articulation)
    articulation._data = SimpleNamespace(is_newton_backend=True, dof=3)
    articulation._all_indices = np.asarray([0], dtype=np.int32)
    articulation._entities = [entity]

    assert articulation.get_joint_drive_type(joint_ids=[2, 1]) == [
        [DriveType.NONE, DriveType.FORCE]
    ]


def test_runtime_effort_mode_disables_pd_gains_on_newton() -> None:
    calls: list[dict[str, object]] = []
    entity = SimpleNamespace(set_newton_drive=lambda **kwargs: calls.append(kwargs))
    articulation = object.__new__(Articulation)
    articulation._spawn_result = object()
    articulation._entities = [entity]
    articulation._all_indices = np.asarray([0], dtype=np.int32)
    articulation._data = SimpleNamespace(is_newton_backend=True, dof=1)
    articulation.device = torch.device("cpu")

    articulation.set_joint_drive(
        stiffness=torch.tensor([[12.0]]),
        damping=torch.tensor([[4.0]]),
        drive_type="force",
        target_mode="effort",
    )

    assert len(calls) == 1
    assert calls[0]["target_mode"] == 4
    assert calls[0]["target_ke"] == 0.0
    assert calls[0]["target_kd"] == 0.0
