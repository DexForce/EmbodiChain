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

"""Tests for shared Gym environment types."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.gym.envs.types import ControllerAction


def test_controller_action_owns_value_and_metadata() -> None:
    value = torch.tensor([[1.0, 2.0]])
    metadata = {"semantic_id": "pick", "segments": ["approach"]}

    action = ControllerAction(value=value, metadata=metadata)
    value.zero_()
    metadata["segments"].append("close")
    snapshot = action.snapshot()

    assert action.value.tolist() == [[1.0, 2.0]]
    assert dict(action.metadata) == {
        "semantic_id": "pick",
        "segments": ["approach"],
    }
    assert snapshot is not action
    assert snapshot.value is not action.value


def test_controller_action_rejects_non_json_metadata() -> None:
    with pytest.raises(TypeError, match="non-JSON value Tensor"):
        ControllerAction(
            value=torch.ones(1, 2),
            metadata={"mask": torch.tensor([True])},
        )
