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
import torch
import pytest

from embodichain.lab.scripts.preview_joint_control import (
    ArticulationPreviewController,
)
from embodichain.lab.visualization import JointControlCommand


class _Articulation:
    dof = 3
    joint_names = ["hinge", "mimic", "slide"]
    active_joint_ids = [0, 2]
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(uid="door")
        self.requested_joint_names: list[str] = []
        self._joint_types = {
            "hinge": "revolute",
            "mimic": "revolute",
            "slide": "prismatic",
        }
        self.qpos = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        self.targets = self.qpos.clone()
        self.limits = torch.tensor(
            [[[-1.0, 1.0], [-0.5, 0.5], [0.0, float("inf")]]],
            dtype=torch.float32,
        )
        self.qpos_writes: list[tuple[torch.Tensor, list[int], list[int], bool]] = []
        self.qvel_writes: list[tuple[torch.Tensor, list[int], list[int], bool]] = []
        self.qf_writes: list[tuple[torch.Tensor, list[int], list[int]]] = []

    def get_qpos(self, target: bool = False) -> torch.Tensor:
        return self.targets if target else self.qpos

    def get_joint_type(self, joint_name: str) -> str:
        self.requested_joint_names.append(joint_name)
        return self._joint_types[joint_name]

    def get_qpos_limits(self, env_ids: list[int]) -> torch.Tensor:
        assert env_ids == [0]
        return self.limits

    def set_qpos(
        self,
        value: torch.Tensor,
        *,
        joint_ids: list[int],
        env_ids: list[int],
        target: bool,
    ) -> None:
        self.qpos_writes.append((value.clone(), joint_ids, env_ids, target))
        if not target:
            self.qpos[0, joint_ids] = value[0]
        else:
            self.targets[0, joint_ids] = value[0]

    def set_qvel(
        self,
        value: torch.Tensor,
        *,
        joint_ids: list[int],
        env_ids: list[int],
        target: bool,
    ) -> None:
        self.qvel_writes.append((value.clone(), joint_ids, env_ids, target))

    def set_qf(
        self,
        value: torch.Tensor,
        *,
        joint_ids: list[int],
        env_ids: list[int],
    ) -> None:
        self.qf_writes.append((value.clone(), joint_ids, env_ids))


class _Runtime:
    def __init__(self) -> None:
        self.exporter = SimpleNamespace(run_id="run", scene_revision=2)
        self.commands: list[JointControlCommand] = []

    def drain_joint_control_commands(self) -> tuple[JointControlCommand, ...]:
        commands = tuple(self.commands)
        self.commands.clear()
        return commands


def _command(
    control_id: str,
    value: float,
    *,
    sequence: int,
    scene_revision: int = 2,
) -> JointControlCommand:
    return JointControlCommand(
        run_id="run",
        scene_revision=scene_revision,
        sequence=sequence,
        client_id="client-a",
        control_id=control_id,
        value=value,
    )


def test_preview_controller_exposes_only_independent_scalar_joints() -> None:
    articulation = _Articulation()
    runtime = _Runtime()

    controller = ArticulationPreviewController([articulation], runtime)
    specs = controller.joint_control_specs()

    assert controller.has_controls
    assert [spec.joint_name for spec in specs] == ["hinge", "slide"]
    assert articulation.requested_joint_names == ["hinge", "slide"]
    assert specs[0].lower == -1.0
    assert specs[0].upper == 1.0
    assert specs[1].lower == 0.0
    assert specs[1].upper is None
    np.testing.assert_allclose(specs[0].step, np.deg2rad(1.0))
    assert specs[1].step == 0.001


def test_continuous_joint_ignores_backend_limit_sentinels() -> None:
    limits = ArticulationPreviewController._finite_limits(
        0.0,
        0.0,
        uid="wheel",
        joint_name="axle",
        joint_type="continuous",
    )

    assert limits == (None, None)


def test_preview_controller_applies_valid_commands_and_holds_pose() -> None:
    articulation = _Articulation()
    runtime = _Runtime()
    controller = ArticulationPreviewController([articulation], runtime)
    hinge_id, slide_id = (spec.control_id for spec in controller.joint_control_specs())
    runtime.commands.extend(
        (
            _command(hinge_id, 5.0, sequence=7),
            _command(slide_id, 0.8, sequence=8, scene_revision=1),
            _command("unknown", 0.5, sequence=9),
        )
    )

    accepted = controller.update()

    assert accepted == 1
    assert len(articulation.qpos_writes) == 2
    for values, joint_ids, env_ids, _ in articulation.qpos_writes:
        assert joint_ids == [0, 2]
        assert env_ids == [0]
        torch.testing.assert_close(values, torch.tensor([[1.0, 0.3]]))
    assert [write[-1] for write in articulation.qpos_writes] == [False, True]
    assert len(articulation.qvel_writes) == 2
    assert [write[-1] for write in articulation.qvel_writes] == [False, True]
    torch.testing.assert_close(
        articulation.qvel_writes[0][0],
        torch.zeros((1, 2)),
    )
    assert len(articulation.qf_writes) == 1
    torch.testing.assert_close(
        articulation.qf_writes[0][0],
        torch.zeros((1, 2)),
    )

    states = controller.joint_control_states()
    assert states[0].control_id == hinge_id
    assert states[0].value == 1.0
    assert states[0].applied_sequence == 7
    assert states[1].control_id == slide_id
    np.testing.assert_allclose(states[1].value, 0.3)
    assert states[1].applied_sequence == 0


def test_gizmo_targets_persist_and_new_slider_commands_take_effect() -> None:
    articulation = _Articulation()
    runtime = _Runtime()
    controller = ArticulationPreviewController(
        [articulation], runtime, follow_external_targets=True
    )
    hinge = controller.joint_control_specs()[0]

    # Gizmo writes a new drive target after the preceding preview step.
    articulation.targets[0, 0] = 0.7
    controller.update()
    torch.testing.assert_close(articulation.qpos, torch.tensor([[0.7, 0.2, 0.3]]))
    assert controller.update() == 0
    assert controller.joint_control_states()[0].value == pytest.approx(0.7)

    runtime.commands.append(_command(hinge.control_id, -0.4, sequence=1))
    assert controller.update() == 1
    assert articulation.targets[0, 0] == pytest.approx(-0.4)

    # Reset uses the existing panel's initial-value command.
    runtime.commands.append(_command(hinge.control_id, hinge.initial_value, sequence=2))
    controller.update()
    controller.update()
    assert articulation.qpos[0, 0] == pytest.approx(hinge.initial_value)
    assert articulation.qpos[0, 1] == pytest.approx(0.2)  # Mimic is never commanded.


def test_same_step_gizmo_target_wins_and_survives_next_step() -> None:
    from embodichain.lab.scripts._preview import _step_preview_simulation

    articulation = _Articulation()
    runtime = _Runtime()
    controller = ArticulationPreviewController(
        [articulation], runtime, follow_external_targets=True
    )
    hinge = controller.joint_control_specs()[0]
    runtime.commands.append(_command(hinge.control_id, 0.4, sequence=1))

    def apply_gizmo(*, step: int) -> None:
        assert step == 1
        assert articulation.targets[0, 0] == pytest.approx(0.4)
        articulation.targets[0, 0] = 0.8

    sim = SimpleNamespace(update=apply_gizmo)
    _step_preview_simulation(sim, controller)
    controller.update()

    assert articulation.qpos[0, 0] == pytest.approx(0.8)
    assert articulation.targets[0, 0] == pytest.approx(0.8)


def test_asset_preview_keeps_its_original_pose_holding_default() -> None:
    articulation = _Articulation()
    controller = ArticulationPreviewController([articulation], _Runtime())
    articulation.targets[0, 0] = 0.7

    controller.update()

    assert articulation.qpos[0, 0] == pytest.approx(0.1)
    assert articulation.targets[0, 0] == pytest.approx(0.1)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_external_nonfinite_target_is_rejected_before_writing(value: float) -> None:
    articulation = _Articulation()
    controller = ArticulationPreviewController(
        [articulation], _Runtime(), follow_external_targets=True
    )
    articulation.targets[0, 0] = value

    with pytest.raises(ValueError, match="Invalid drive target"):
        controller.update()

    assert not articulation.qpos_writes


def test_overlapping_robot_control_parts_do_not_duplicate_joint_controls() -> None:
    articulation = _Articulation()
    articulation.cfg.control_parts = {"arm": ["hinge"], "full_body": ["hinge", "slide"]}
    controller = ArticulationPreviewController(
        [articulation], _Runtime(), follow_external_targets=True
    )

    assert [spec.joint_id for spec in controller.joint_control_specs()] == [0, 2]
