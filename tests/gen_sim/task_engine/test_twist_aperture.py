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

import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import twist_runtime
from embodichain.gen_sim.task_engine._task_program.twist_binding import (
    KnobBinding,
    TwistRoute,
)
from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
    _grasp_tip_offset,
    _TwistLowerer,
)
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    EndpointBinding,
    GRASP_COMMAND,
    JointPositionCommand,
    JointPositionTarget,
    OPEN_COMMAND,
    TwistOptions,
)

OPEN_GAP = 0.100
CLOSED_GAP = 0.020
PAD_HALF_WIDTH = 0.001
PAD_HEIGHT = 0.002
TIP_BASE_OFFSET = 0.020
TIP_APERTURE_OFFSET = 0.010
SEARCH_RESOLUTION = 1.0 / 4096.0
REST_OFFSET = 0.001


class _Robot:
    """CPU jaw model with a moving tip and a known aperture interval."""

    link_names = ("left_inner_finger_pad", "left_right_inner_finger_pad")
    joint_names = ("arm_joint", "left_finger_joint", "right_finger_joint")

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            solver_cfg={
                "left_arm": SimpleNamespace(
                    end_link_name="wrist", tcp=torch.eye(4).tolist()
                )
            }
        )

    def collision_points(self, name: str) -> torch.Tensor:
        return torch.tensor(
            [(-PAD_HALF_WIDTH, 0.0, 0.0), (PAD_HALF_WIDTH, 0.0, PAD_HEIGHT)],
            dtype=torch.float64,
        )

    def compute_fk(
        self, *, qpos: torch.Tensor, link_names: list[str], qpos_joint_names: Any
    ) -> torch.Tensor:
        fraction = float(qpos[0, 1])
        gap = OPEN_GAP - fraction * (OPEN_GAP - CLOSED_GAP)
        poses = torch.eye(4, dtype=qpos.dtype).repeat(1, len(link_names), 1, 1)
        poses[:, :, :3, 3] = poses.new_tensor((2.0, -1.0, 0.5))
        for index, side in ((1, -1.0), (2, 1.0)):
            poses[0, index, 0, 3] += side * (gap / 2.0 + PAD_HALF_WIDTH)
            poses[0, index, 2, 3] += TIP_BASE_OFFSET + fraction * TIP_APERTURE_OFFSET
        return poses

    def get_link_physical_attr(self, names: list[str]) -> list[SimpleNamespace]:
        return [SimpleNamespace(rest_offset=REST_OFFSET) for _ in names]


@pytest.fixture(autouse=True)
def collision_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Aperture tests isolate the jaw/FK calculation from source-file loading.
    monkeypatch.setattr(
        twist_runtime,
        "_robot_collision_points",
        lambda robot, names: {
            name: robot.collision_points(name).tolist() for name in names
        },
    )


def _fixture() -> tuple[_Robot, SimpleNamespace, SimpleNamespace]:
    binding = ActionBinding(
        owner_id="aperture-test",
        endpoints=(
            EndpointBinding(
                slot_id="primary",
                endpoint_id="motion",
                resource_id="left",
                adapter_id="control_part",
                target=JointPositionTarget("left_arm", (0,)),
            ),
            EndpointBinding(
                slot_id="primary",
                endpoint_id="grasp",
                resource_id="left",
                adapter_id="control_part",
                target=JointPositionTarget("left_eef", (1, 2)),
                commands={
                    OPEN_COMMAND: JointPositionCommand(
                        torch.zeros(2, dtype=torch.float64)
                    ),
                    GRASP_COMMAND: JointPositionCommand(
                        torch.tensor((1.0, -1.0), dtype=torch.float64)
                    ),
                },
            ),
        ),
    )
    context = SimpleNamespace(
        batch_size=1,
        robot=SimpleNamespace(qpos=torch.zeros((1, 3), dtype=torch.float64)),
        task=SimpleNamespace(
            get_held_object=lambda key: None, coordinated_held_objects=()
        ),
    )
    return (
        _Robot(),
        context,
        SimpleNamespace(binding=SimpleNamespace(action_binding=binding)),
    )


def test_aperture_command_and_tip_share_the_measured_width() -> None:
    robot, context, bound = _fixture()
    width = (OPEN_GAP + CLOSED_GAP) / 2.0

    offset, command = _grasp_tip_offset(robot, context, bound, width)
    fraction = float(command.positions[0, 0])

    assert isinstance(command, JointPositionCommand)
    assert fraction == pytest.approx(0.5, abs=SEARCH_RESOLUTION)
    assert OPEN_GAP - fraction * (OPEN_GAP - CLOSED_GAP) <= width
    assert offset == pytest.approx(
        TIP_BASE_OFFSET + fraction * TIP_APERTURE_OFFSET + PAD_HEIGHT
    )
    assert command.positions[0, 1] == pytest.approx(-fraction)
    assert torch.equal(context.robot.qpos, torch.zeros_like(context.robot.qpos))


@pytest.mark.parametrize(
    "width, expected_fraction", [(OPEN_GAP, 0.0), (CLOSED_GAP, 1.0)]
)
def test_aperture_accepts_both_opening_boundaries(
    width: float, expected_fraction: float
) -> None:
    robot, context, bound = _fixture()

    _, command = _grasp_tip_offset(robot, context, bound, width)

    assert command.positions[0, 0] == pytest.approx(
        expected_fraction, abs=SEARCH_RESOLUTION
    )


@pytest.mark.parametrize("width", [OPEN_GAP + 0.001, CLOSED_GAP - 0.001])
def test_aperture_rejects_width_outside_measured_opening(width: float) -> None:
    robot, context, bound = _fixture()

    with pytest.raises(ValueError, match="outside the measured gripper opening"):
        _grasp_tip_offset(robot, context, bound, width)


@pytest.mark.parametrize("width", [0.0, -0.1, float("nan"), float("inf")])
def test_aperture_rejects_invalid_width(width: float) -> None:
    robot, context, bound = _fixture()

    with pytest.raises(ValueError, match="finite and positive"):
        _grasp_tip_offset(robot, context, bound, width)


def test_lowerer_overrides_only_invocation_grasp_without_mutating_defaults() -> None:
    robot, context, bound = _fixture()
    width = 0.050
    route = TwistRoute(
        binding=KnobBinding(
            object_id="control",
            joint="knob_axis",
            link="knob",
            parent="panel",
            source_sha256="a" * 64,
            scale=0.5,
            axis=(0.0, 0.0, -1.0),
            axis_sign=-1.0,
            origin=(0.0, 0.0, 0.0),
            outer_point=(0.0, 0.0, 0.0085),
            grip_depth=0.0075,
            grip_width=width,
            limits=(-math.pi / 2.0, 0.0),
            target_setting=90,
            target_qpos=-math.pi / 2.0,
        ),
        arm="left",
    )
    lowerer = object.__new__(_TwistLowerer)
    lowerer.route, lowerer.robot, lowerer.joint_index = route, robot, 0
    grip_vertices = np.frombuffer(
        np.asarray(
            [[-width / 2, 0, 0], [width / 2, 0, 0.0075]], dtype=np.float64
        ).tobytes(),
        dtype=np.float64,
    ).reshape(-1, 3)
    parent_vertices = np.frombuffer(
        np.asarray([[-0.1, -0.1, -0.01], [0.1, 0.1, 0]], dtype=np.float64).tobytes(),
        dtype=np.float64,
    ).reshape(-1, 3)
    lowerer.geometry = SimpleNamespace(
        grip=SimpleNamespace(vertices=grip_vertices),
        parent_collisions=(SimpleNamespace(vertices=parent_vertices),),
    )
    lowerer.art = SimpleNamespace(
        get_qpos=lambda: torch.zeros((1, 1)),
        get_link_physical_attr=lambda name: [SimpleNamespace(rest_offset=REST_OFFSET)],
        get_link_pose=lambda name, to_matrix: torch.eye(4).unsqueeze(0),
    )
    options = TwistOptions()
    original_options = asdict(options)

    result = lowerer.lower(
        SimpleNamespace(arguments={"object": "control", "setting": 90}),
        context=context,
        bound=bound,
        option_template=options,
    )
    source_binding = bound.binding.action_binding
    overridden = source_binding.with_command_overrides(
        result.control_overrides.as_flat_mapping()
    )
    source = source_binding.endpoint("primary", "grasp")
    resolved = overridden.endpoint("primary", "grasp")
    expected_fraction = (OPEN_GAP - width - 4 * REST_OFFSET) / (OPEN_GAP - CLOSED_GAP)

    assert set(result.control_overrides.as_flat_mapping()) == {("primary", "grasp")}
    assert set(result.control_overrides.endpoints["primary"]["grasp"]) == {
        GRASP_COMMAND
    }
    assert resolved.command(GRASP_COMMAND).positions[0, 0] == pytest.approx(
        expected_fraction, abs=SEARCH_RESOLUTION
    )
    assert source.command(GRASP_COMMAND).positions.tolist() == [1.0, -1.0]
    assert resolved.command(OPEN_COMMAND).equivalent_to(source.command(OPEN_COMMAND))
    assert result.skill_options is None
    assert asdict(options) == original_options
