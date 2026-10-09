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

import pytest
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


@pytest.mark.parametrize("angle", [-1.0471975512, math.pi / 3])
def test_pour_signed_arc_tilts_toward_the_bound_receiver(angle: float) -> None:
    from types import SimpleNamespace
    from embodichain.lab.sim.atomic_actions import (
        AxisAlignAffordance,
        ObjectSemantics,
        JointPositionTarget,
    )
    from embodichain.lab.sim.atomic_actions.primitives.pour import PourOptions
    from embodichain.lab.sim.motion.planners.utils import PlanResult

    identity = torch.eye(4)[None]
    qpos = torch.zeros(1, 8)
    pads = [identity.clone(), identity.clone()]
    pads[0][0, 1, 3], pads[1][0, 1, 3] = -0.05, 0.05
    robot = SimpleNamespace(
        compute_fk=lambda **kwargs: identity.clone(),
        get_qpos=lambda: qpos.clone(),
        link_names=["right_left_inner_finger_pad", "right_inner_finger_pad"],
        get_link_pose=lambda name, **kwargs: (
            pads[0] if name == "right_left_inner_finger_pad" else pads[1]
        ),
    )
    semantics = ObjectSemantics(
        AxisAlignAffordance(internal_axis=torch.tensor([0.0, 0.0, 1.0])),
        {},
        entity_id="vessel",
    )
    receiver = identity.clone()
    receiver[0, 0, 3] = 1.0
    held = SimpleNamespace(object_to_eef=identity.clone(), semantics=semantics)
    context = SimpleNamespace(
        get_held_object=lambda key: held,
        robot=SimpleNamespace(qpos=qpos),
        last_qpos=qpos.clone(),
        batch_size=1,
        env_ids=torch.tensor([0]),
        require_control_dt=lambda: 0.04,
        task=SimpleNamespace(
            exclusive_held_object_mask=lambda key: torch.tensor([True])
        ),
        scene=SimpleNamespace(
            entities={"receiver": SimpleNamespace(pose=receiver, confidence=1.0)}
        ),
    )
    motion = SimpleNamespace(
        task_state_key="right",
        require_target=lambda cls: JointPositionTarget("right_arm", tuple(range(7))),
    )
    hand = SimpleNamespace(
        task_state_key="right",
        require_target=lambda cls: JointPositionTarget("right_eef", (7,)),
        joint_positions=lambda *args, **kwargs: torch.zeros(1, 1),
    )
    targets = []

    def generate(states, options=None):
        targets.extend(state.xpos.clone() for state in states)
        dt = torch.full((1, options.sample_count), 0.04)
        dt[:, 0] = 0
        return PlanResult(
            success=torch.tensor([True]),
            positions=torch.zeros(1, options.sample_count, 7),
            dt=dt,
        )

    action = GenSimPour({"pour": "receiver"})
    action._planning_services = SimpleNamespace(
        robot=robot,
        device=torch.device("cpu"),
        motion_generator=SimpleNamespace(generate=generate),
        planner_name="mock",
    )
    action.build_plan = lambda *args, **kwargs: kwargs
    request = SimpleNamespace(
        binding=SimpleNamespace(
            endpoint=lambda slot, name: motion if name == "motion" else hand
        ),
        invocation_id="pour",
        skill_options=PourOptions(rotate_angle=angle),
        motion_policy=SimpleNamespace(
            sample_count=12,
            to_motion_gen_options=lambda **kwargs: SimpleNamespace(
                **kwargs, strategy="ik_interp", sample_count=12
            ),
        ),
    )
    result = action._plan(request, context)
    peak = max(
        targets, key=lambda pose: float(torch.linalg.vector_norm(pose[0, :2, 2]))
    )
    assert result["success"].all()
    assert peak[0, 0, 2] > 0.8  # Receiver lies along +X for either angle sign.
