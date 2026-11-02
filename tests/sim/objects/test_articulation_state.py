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

from embodichain.lab.sim.objects.articulation import Articulation
from embodichain.lab.sim.objects.robot import Robot

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("clamp_targets", [True, False])
def test_combined_state_position_limits(clamp_targets: bool) -> None:
    art = object.__new__(Articulation)
    art.device = torch.device("cpu")
    art._entities = [object()] * 3
    art.__dict__["dof"] = 2
    art._data = SimpleNamespace(
        is_newton_backend=False,
        qpos_limits=torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]] * 3),
        articulation_view=SimpleNamespace(apply_state=Mock()),
    )
    art._world = SimpleNamespace(update=Mock())
    art._stabilize_newton_mimic_target_write = Mock()
    art.set_state(
        env_ids=[2, 0],
        joint_ids=[1, 0],
        qpos=torch.tensor([[5.0, -3.0], [-5.0, 3.0]]),
        target_qpos=torch.tensor([[-4.0, 4.0], [4.0, -4.0]]),
        clear_dynamics=True,
        clamp_target_qpos=clamp_targets,
    )
    call = art._data.articulation_view.apply_state
    call.assert_called_once()
    art._world.update.assert_called_once_with(0.0)
    rows, columns = call.call_args.args
    assert torch.as_tensor(rows).tolist() == [2, 0]
    assert torch.as_tensor(columns).tolist() == [1, 0]
    torch.testing.assert_close(
        call.call_args.kwargs["qpos"], torch.tensor([[2.0, -1.0], [-2.0, 1.0]])
    )
    torch.testing.assert_close(
        call.call_args.kwargs["target_qpos"],
        (
            torch.tensor([[-2.0, 1.0], [2.0, -1.0]])
            if clamp_targets
            else torch.tensor([[-4.0, 4.0], [4.0, -4.0]])
        ),
    )
    call.reset_mock()
    art._world.update.reset_mock()
    with pytest.raises(ValueError, match="root_velocity"):
        art.set_state(
            qpos=torch.zeros((3, 2)),
            root_velocity=torch.zeros((3, 5)),
            clear_dynamics=True,
        )
    call.assert_not_called()
    art._world.update.assert_not_called()


@pytest.mark.parametrize("newton", [False, True])
@pytest.mark.parametrize(
    "field", ["qpos", "target_qpos", "qvel", "target_qvel", "root_pose"]
)
def test_cpu_transform_publication_only_for_direct_pose_writes(
    newton: bool, field: str
) -> None:
    art = object.__new__(Articulation)
    art.device = torch.device("cpu")
    art._entities = [object()]
    art.__dict__["dof"] = 2
    art._data = SimpleNamespace(
        is_newton_backend=newton,
        qpos_limits=torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]]),
        articulation_view=SimpleNamespace(apply_state=Mock()),
    )
    art._world = SimpleNamespace(update=Mock())
    art._stabilize_newton_mimic_target_write = Mock()
    value = torch.tensor([[0.1, 0.2]])
    if field == "root_pose":
        value = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
    art.set_state(**{field: value})
    if not newton and field in {"qpos", "root_pose"}:
        art._world.update.assert_called_once_with(0.0)
    else:
        art._world.update.assert_not_called()


@pytest.mark.parametrize("target", [True, False])
@pytest.mark.parametrize("route", ["articulation", "robot", "robot_part"])
def test_set_qpos_limits_with_unclamped_targets(target: bool, route: str) -> None:
    art = object.__new__(Articulation if route == "articulation" else Robot)
    art.device = torch.device("cpu")
    art._entities = [object()]
    art.__dict__["dof"] = 2
    art._data = SimpleNamespace(
        is_newton_backend=True,
        qpos_limits=torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]]),
        articulation_view=SimpleNamespace(apply_state=Mock()),
    )
    art._stabilize_newton_mimic_target_write = Mock()
    selection = {"joint_ids": [1]}
    if route == "robot_part":
        art.cfg = SimpleNamespace(control_parts={"arm": ["elbow"]})
        art._joint_ids = {"arm": [1]}
        selection = {"name": "arm"}
    art.set_qpos(torch.tensor([[3.0]]), **selection)
    torch.testing.assert_close(
        art._data.articulation_view.apply_state.call_args.kwargs["target_qpos"],
        torch.tensor([[2.0]]),
    )
    art.set_qpos(
        torch.tensor([[3.0]]), **selection, target=target, clamp_target_qpos=False
    )
    torch.testing.assert_close(
        art._data.articulation_view.apply_state.call_args.kwargs[
            "target_qpos" if target else "qpos"
        ],
        torch.tensor([[3.0 if target else 2.0]]),
    )


@pytest.mark.parametrize("clamp_targets", [True, False])
def test_mimic_follower_preserves_target_limit_selection(clamp_targets: bool) -> None:
    art = object.__new__(Articulation)
    art.device = torch.device("cpu")
    art._entities = [object()]
    art.__dict__["dof"] = 2
    art._data = SimpleNamespace(
        is_newton_backend=True,
        qpos_limits=torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]]),
        qvel_limits=torch.tensor([[10.0, 10.0]]),
        articulation_view=SimpleNamespace(apply_state=Mock(), apply_qpos=Mock()),
    )
    art._newton_mimic_compliance_configured = True
    art._mimic_info = SimpleNamespace(
        mimic_id=torch.tensor([1]),
        mimic_parent=torch.tensor([0]),
        mimic_multiplier=torch.tensor([3.0]),
        mimic_offset=torch.tensor([0.0]),
    )
    art.set_qpos(torch.tensor([[1.5]]), joint_ids=[0], clamp_target_qpos=clamp_targets)
    torch.testing.assert_close(
        art._data.articulation_view.apply_qpos.call_args.args[0],
        torch.tensor([[2.0 if clamp_targets else 4.5]]),
    )
