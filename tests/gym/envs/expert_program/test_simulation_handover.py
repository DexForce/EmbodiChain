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

"""Tests for configured semantic hand-over pose integration."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.gym.envs.expert_program import ConfiguredHandOverPoseProvider


def _provider() -> ConfiguredHandOverPoseProvider:
    """Return one deterministic dual-arm transfer declaration."""
    return ConfiguredHandOverPoseProvider(
        middle_position=(0.0, 0.0, 0.7),
        middle_quaternion_wxyz=(1.0, 1.0, 0.0, 0.0),
        final_position=(0.0, -0.2, 0.7),
        final_quaternion_wxyz=(1.0, 1.0, 0.0, 0.0),
    )


def test_configured_handover_provider_normalizes_and_owns_targets() -> None:
    """Configured poses are normalized and returned as independent values."""
    provider = _provider()

    first = provider.resolve(object(), context=object(), bound=object())
    second = provider.resolve(object(), context=object(), bound=object())

    expected_rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert first.middle.pose is not second.middle.pose
    assert first.final.pose is not second.final.pose
    assert torch.allclose(
        first.middle.pose.to_matrix()[:3, :3], expected_rotation, atol=1e-6
    )
    assert torch.allclose(
        first.middle.pose.to_matrix()[:3, 3], torch.tensor([0.0, 0.0, 0.7])
    )
    assert torch.allclose(
        first.final.pose.to_matrix()[:3, 3], torch.tensor([0.0, -0.2, 0.7])
    )


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"middle_position": (0.0, 0.0)}, TypeError),
        ({"middle_quaternion_wxyz": (0.0, 0.0, 0.0, 0.0)}, ValueError),
    ],
)
def test_configured_handover_provider_rejects_invalid_declarations(
    overrides: dict[str, object],
    error_type: type[Exception],
) -> None:
    """Malformed provider declarations fail before simulation construction."""
    values: dict[str, object] = {
        "middle_position": (0.0, 0.0, 0.7),
        "middle_quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
        "final_position": (0.0, -0.2, 0.7),
        "final_quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
    }
    values.update(overrides)

    with pytest.raises(error_type):
        ConfiguredHandOverPoseProvider(**values)  # type: ignore[arg-type]
