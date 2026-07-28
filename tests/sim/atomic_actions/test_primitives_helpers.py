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

"""Tests for atomic action primitive helpers."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    resolve_object_target,
)


def test_resolve_object_target_uses_custom_name_in_shape_error() -> None:
    with pytest.raises(ValueError, match="placing_object_target_pose"):
        resolve_object_target(
            torch.zeros(2, 4, 4),
            n_envs=3,
            device=torch.device("cpu"),
            name="placing_object_target_pose",
        )
