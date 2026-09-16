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
from unittest.mock import Mock

import pytest
import torch

from scripts.tutorials.sim import create_articulation as tutorial


@pytest.mark.parametrize("top_joint_id", [0, 1, 2])
def test_usd_control_only_moves_named_top_drawer(top_joint_id: int) -> None:
    """Other joints cannot receive forces or prevent the top drawer reversing."""
    names = ["middle_slide", "bottom_slide"]
    names.insert(top_joint_id, "top_slide")
    # Other joints deliberately remain unsettled in both environments.
    qpos = torch.full((2, 3), 0.1)
    qvel = torch.ones_like(qpos)
    qpos[:, top_joint_id] = 0.0
    qvel[:, top_joint_id] = 0.0
    limits = torch.tensor([[[-0.26, 0.0]]]).expand(2, 1, 2)
    get_limits = Mock(return_value=limits)
    forces: list[tuple[torch.Tensor, list[int]]] = []

    def set_qf(values: torch.Tensor, joint_ids: list[int]) -> None:
        forces.append((values.clone(), list(joint_ids)))

    articulation = SimpleNamespace(
        joint_names=names,
        get_qpos=lambda: qpos,
        get_qvel=lambda: qvel,
        get_qpos_limits=get_limits,
        set_qf=set_qf,
    )

    def update(*, step: int) -> None:
        assert step == 1
        qpos[:, top_joint_id] = -0.26

    tutorial.run_simulation(
        SimpleNamespace(update=update),
        articulation,
        max_steps=2,
        joint_name="top_slide",
        open_at_lower_limit=True,
    )

    get_limits.assert_called_once_with(joint_ids=[top_joint_id])
    # First pull toward the negative limit, then close, then clear effort.
    assert len(forces) == 3
    for (force, joint_ids), direction in zip(forces, [-1.0, 1.0, 0.0]):
        assert joint_ids == [top_joint_id]
        torch.testing.assert_close(
            force,
            torch.full((2, 1), direction * tutorial.DRAWER_JOINT_FORCE_LIMIT),
        )


def test_urdf_remains_default_and_usd_can_be_selected() -> None:
    parser = tutorial.build_parser()
    assert parser.parse_args([]).asset == "urdf"
    assert parser.parse_args(["--asset", "usd"]).asset == "usd"
