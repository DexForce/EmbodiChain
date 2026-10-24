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
"""Explicit DC effort preserves reference braking and joint ownership."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
import torch
import warp as wp

from embodichain.lab.sim._joint_motor import _BatchedJointDCMotorControl, _dc_motor

__all__: list[str] = []
pytestmark = pytest.mark.no_sim


def test_motor_binds_public_names_and_holds_effort_between_solver_substeps() -> None:
    """Arena order and scalar q/qd offsets remain distinct from policy order."""
    wp.init()
    parameters = dict(
        joint_names=["b", "a"],
        stiffness=[2.0, 3.0],
        damping=[1.0, 1.0],
        effort_limit=[20.0, 20.0],
        velocity_limit=[30.0, 30.0],
        default_joint_position=[1.0, -1.0],
    )
    paths = ("arena_1/robot", "arena_0/robot")
    articulations = {}
    for arena in range(2):
        joints = [
            SimpleNamespace(
                name=name,
                q_start=20 * arena + q,
                qd_start=20 * arena + qd,
                q_size=1,
                qd_size=1,
                joint_id=10 * arena + index,
            )
            for index, (name, q, qd) in enumerate([("a", 7, 6), ("b", 9, 8)])
        ]
        articulations[f"arena_{arena}/robot"] = SimpleNamespace(
            _physics_binding=SimpleNamespace(public_joints=joints)
        )
    context = SimpleNamespace(
        result=SimpleNamespace(get_articulation=articulations.__getitem__),
        device="cpu",
        model=SimpleNamespace(joint_f=wp.zeros(40, dtype=float, device="cpu")),
        control=SimpleNamespace(joint_f=wp.zeros(40, dtype=float, device="cpu")),
        current_state=SimpleNamespace(
            joint_q=wp.zeros(40, dtype=float, device="cpu"),
            joint_qd=wp.zeros(40, dtype=float, device="cpu"),
        ),
    )
    control = _BatchedJointDCMotorControl(parameters, paths)
    control.initialize(context)
    assert control.q_indices.numpy().tolist() == [[29, 27], [9, 7]]
    assert control.qd_indices.numpy().tolist() == [[28, 26], [8, 6]]
    assert control.exclusive_resource_claims() == (
        ("newton_joint", 11),
        ("newton_joint", 10),
        ("newton_joint", 1),
        ("newton_joint", 0),
    )
    control(context, 0, 2, 0.0025)
    torch.testing.assert_close(
        wp.to_torch(control.effort), torch.tensor([[2.0, -3.0], [2.0, -3.0]])
    )
    control.targets.zero_()
    control(context, 1, 2, 0.0025)
    torch.testing.assert_close(
        wp.to_torch(control.effort), torch.tensor([[2.0, -3.0], [2.0, -3.0]])
    )
    control(context, 0, 2, 0.0025)
    torch.testing.assert_close(wp.to_torch(control.effort), torch.zeros(2, 2))
    control.close()
    assert control.targets is None and control.exclusive_resource_claims() == ()


def test_dc_motor_matches_reference_for_scalar_dof_routing() -> None:
    """Cover reordered free-root DOFs, overspeed braking and nonfinite inputs."""
    wp.init()
    device = "cpu"
    torch.manual_seed(42)
    rows, columns = 128, 12
    order = torch.randperm(columns, device=device)
    q_indices = torch.arange(rows, device=device)[:, None] * (columns + 7) + 7 + order
    qd_indices = torch.arange(rows, device=device)[:, None] * (columns + 6) + 6 + order
    q = torch.randn(rows * (columns + 7) + 13, device=device)
    qd = torch.randn(rows * (columns + 6) + 13, device=device) * 100
    targets = torch.randn(rows, columns, device=device)
    kp = torch.rand(columns, device=device) * 100
    kd = torch.rand(columns, device=device) * 10
    limits = torch.rand(columns, device=device) * 200 + 1
    speeds = torch.rand(columns, device=device) * 80 + 1
    qd[qd_indices[0]] = torch.stack(
        (
            speeds[0] * 2,
            -speeds[1] * 2,
            speeds[2] * 2.1,
            -speeds[3] * 2.1,
            speeds[4],
            -speeds[5],
            speeds[6] * 0,
            speeds[7] * 0.5,
            speeds[8] * -0.5,
            speeds[9] * float("inf"),
            speeds[10] * -float("inf"),
            speeds[11] * float("nan"),
        )
    )
    targets[1, 0] = float("nan")
    # Independent torque-speed envelope in Torch; no task-package dependency.
    speed = qd[qd_indices].clamp(-2 * speeds, 2 * speeds)
    upper = torch.minimum(limits * (1 - speed / speeds), limits)
    lower = torch.maximum(limits * (-1 - speed / speeds), -limits)
    requested = kp * (targets - q[q_indices]) - kd * qd[qd_indices]
    expected = torch.clamp(requested, min=lower, max=upper)
    model_force = torch.full_like(qd, 2.3)
    control_force = torch.full_like(qd, 2.3)
    effort = torch.zeros_like(targets)
    velocity = torch.zeros_like(targets)
    indices = [
        wp.from_torch(value.to(torch.int32)) for value in (q_indices, qd_indices)
    ]
    wp.launch(
        _dc_motor,
        dim=(rows, columns),
        inputs=[
            wp.from_torch(q),
            wp.from_torch(qd),
            *indices,
            wp.from_torch(targets),
            wp.from_torch(kp),
            wp.from_torch(kd),
            wp.from_torch(limits),
            wp.from_torch(speeds),
            wp.from_torch(model_force),
            wp.from_torch(control_force),
            wp.from_torch(effort),
            wp.from_torch(velocity),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    torch.testing.assert_close(effort, expected, atol=2e-5, rtol=1e-6, equal_nan=True)
    torch.testing.assert_close(velocity, qd[qd_indices], atol=0, rtol=0, equal_nan=True)
    mask = torch.ones_like(qd, dtype=torch.bool)
    mask[qd_indices] = False
    for owner in (model_force, control_force):
        torch.testing.assert_close(
            owner[qd_indices], expected, atol=2e-5, rtol=1e-6, equal_nan=True
        )
        torch.testing.assert_close(
            owner[mask], torch.full_like(owner[mask], 2.3), atol=0, rtol=0
        )
