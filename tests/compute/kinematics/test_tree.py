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

"""Tree kinematics contracts on an analytically solvable branched model."""

from __future__ import annotations

import math
import pytest
import torch

from embodichain.compute.kinematics.tree import TreeKinematics

URDF = """<robot name="branched">
<link name="base"/><link name="arm"/><link name="tip"/><link name="slider"/>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
<origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/></joint>
<joint name="fixed" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/></joint>
<joint name="slide" type="prismatic"><parent link="base"/><child link="slider"/>
<origin xyz="0 1 0"/><axis xyz="1 0 0"/><limit lower="-2" upper="2"/></joint></robot>"""


def test_branched_pose_and_link_origin_velocity():
    kin = TreeKinematics(URDF, ["slide", "hinge"], dtype=torch.float64)
    q = torch.tensor([[0.7, math.pi / 2]], dtype=torch.float64)
    root = torch.zeros(1, 3, dtype=q.dtype)
    linear = torch.tensor([[0.2, -0.1, 0.3]], dtype=q.dtype)
    angular = torch.tensor([[0.0, 0.0, 2.0]], dtype=q.dtype)
    state = kin.evaluate(
        root,
        torch.eye(3, dtype=q.dtype)[None],
        q,
        linear,
        angular,
        torch.tensor([[0.4, 3.0]], dtype=q.dtype),
    )
    tip, slide = kin.link_indices(["tip", "slider"])
    torch.testing.assert_close(
        state.position[:, tip], torch.tensor([[1.0, 1.0, 0.0]], dtype=q.dtype)
    )
    torch.testing.assert_close(
        state.linear_velocity[:, tip], torch.tensor([[-4.8, 1.9, 0.3]], dtype=q.dtype)
    )
    torch.testing.assert_close(
        state.angular_velocity[:, tip], torch.tensor([[0.0, 0.0, 5.0]], dtype=q.dtype)
    )
    torch.testing.assert_close(
        state.position[:, slide], torch.tensor([[0.7, 1.0, 0.0]], dtype=q.dtype)
    )
    torch.testing.assert_close(
        state.linear_velocity[:, slide], torch.tensor([[-1.4, 1.3, 0.3]], dtype=q.dtype)
    )


def test_tree_gradients_and_arbitrary_batch_axes():
    kin = TreeKinematics(URDF, ["slide", "hinge"], dtype=torch.float64)
    q = torch.tensor(
        [[[0.3, 0.4], [0.8, -0.2]]], dtype=torch.float64, requires_grad=True
    )
    dq = torch.tensor(
        [[[0.1, 0.2], [0.2, 0.1]]], dtype=torch.float64, requires_grad=True
    )
    root = torch.zeros(1, 2, 3, dtype=q.dtype)
    rot = torch.eye(3, dtype=q.dtype).expand(1, 2, 3, 3)

    def evaluate(x, dx):
        k = kin.evaluate(root, rot, x, root, root, dx)
        return k.position, k.rotation, k.linear_velocity, k.angular_velocity

    assert torch.autograd.gradcheck(evaluate, (q, dq))
    first = evaluate(q, dq)[0]
    original = first.detach().clone()
    evaluate(q + 1, dq)
    torch.testing.assert_close(first, original)


def test_tree_rejects_incomplete_joint_order():
    with pytest.raises(ValueError, match="every movable"):
        TreeKinematics(URDF, ["hinge"])
