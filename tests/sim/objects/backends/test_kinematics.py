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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dexsim.spawn.descs import JointDesc
from dexsim.engine import JointType

from embodichain.lab.sim.objects import Articulation
from embodichain.lab.sim.objects.backends._kinematics import _build_pk_chain
from embodichain.lab.sim.utility.solver_utils import create_pk_chain


def _articulation(links: list[str], joints: list[JointDesc]) -> Articulation:
    entity = SimpleNamespace(
        get_link_names=lambda: links,
        get_root_link_name=lambda: "root",
        get_joint_names=lambda: [j.name for j in joints],
        get_joint_desc=lambda name: next(j for j in joints if j.name == name),
    )
    articulation = object.__new__(Articulation)
    articulation.device = torch.device("cpu")
    articulation.pk_chain = _build_pk_chain(entity, articulation.device)
    articulation.joint_names = [
        joint.name for joint in joints if joint.joint_type != JointType.FIXED
    ]
    # This path must never be opened by FK or Jacobian computation.
    articulation.cfg = SimpleNamespace(fpath="missing.usdc")
    return articulation


def _hinge() -> JointDesc:
    # Both joint frames have rotation and translation; omitting the child frame
    # gives the wrong link pose and the wrong revolute lever arm.
    return JointDesc(
        "hinge",
        "root",
        "door",
        JointType.REVOLUTE,
        origin_pose=np.array(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        target_pose=np.array(
            [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 0.0, -1.0, 0.1],
                [0.0, 1.0, 0.0, 0.4],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        axis=np.array([0.0, 0.0, 1.0]),
        lower_limit=-2.0,
        upper_limit=2.0,
    )


def test_branched_fk_preserves_both_joint_frames_and_named_state() -> None:
    hinge = _hinge()
    slider = JointDesc(
        "slide", "root", "drawer", JointType.PRISMATIC, axis=np.array([0.0, 1.0, 0.0])
    )
    slider.origin_pose[2, 3] = slider.target_pose[2, 3] = -0.2
    handle = JointDesc("weld", "door", "handle", JointType.FIXED)
    handle.origin_pose[0, 3] = 0.5
    handle.target_pose[1, 3] = 0.2
    obj = _articulation(["root", "door", "drawer", "handle"], [slider, handle, hinge])
    # Input joint order deliberately differs from the PK parameter order.
    qpos = torch.tensor([[0.4, 0.13], [-0.2, -0.07]])
    result = obj.compute_fk(
        qpos,
        link_names=["door", "drawer", "handle", "root"],
        qpos_joint_names=["hinge", "slide"],
    )
    expected = torch.eye(4).repeat(2, 4, 1, 1)
    for i, (angle, distance) in enumerate(qpos):
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.tensor(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        expected[i, 0] = (
            torch.from_numpy(hinge.origin_pose)
            @ rotation
            @ torch.linalg.inv(torch.from_numpy(hinge.target_pose))
        )
        expected[i, 1, 1, 3] = distance
        expected[i, 2] = (
            expected[i, 0]
            @ torch.from_numpy(handle.origin_pose)
            @ torch.linalg.inv(torch.from_numpy(handle.target_pose))
        )
    torch.testing.assert_close(result, expected)


def test_jacobian_reuses_resolved_chain_and_child_frame_lever_arm() -> None:
    obj = _articulation(["root", "door"], [_hinge()])
    qpos = torch.tensor([[0.4]])
    # Defaults must include the fixed root and physical end link, not the
    # internal moving frame, and must not try to read cfg.fpath as URDF.
    jacobian = obj.compute_jacobian(qpos)
    epsilon = 0.001
    poses = obj.compute_fk(
        torch.cat([qpos + epsilon, qpos - epsilon]),
        link_names=["door"],
        qpos_joint_names=["hinge"],
    )
    finite_difference = (poses[0, 0, :3, 3] - poses[1, 0, :3, 3]) / (2 * epsilon)
    torch.testing.assert_close(
        jacobian[0, :3, 0], finite_difference, atol=1e-4, rtol=1e-3
    )
    torch.testing.assert_close(jacobian[0, 3:, 0], torch.tensor([0.0, 0.0, 1.0]))
    serial_pose = obj.compute_fk(qpos, root_link_name="root", end_link_name="door")
    named_pose = obj.compute_fk(qpos, link_names=["door"], qpos_joint_names=["hinge"])
    torch.testing.assert_close(serial_pose, named_pose[:, 0])


def test_fixed_world_attachment_does_not_move_fk_root() -> None:
    anchor = JointDesc("anchor", "", "root", JointType.FIXED)
    anchor.origin_pose[0, 3] = 5.0
    obj = _articulation(["root"], [anchor])
    actual = obj.compute_fk(torch.empty(1, 0), link_names=["root"], qpos_joint_names=[])
    torch.testing.assert_close(actual, torch.eye(4).reshape(1, 1, 4, 4))


def test_urdf_jacobian_keeps_fixed_tip_offset(tmp_path: Path) -> None:
    """Reusing a source URDF chain preserves the physical tip's lever arm."""
    path = tmp_path / "arm.urdf"
    path.write_text(
        '<robot name="arm"><link name="root"/><link name="arm"/>'
        '<link name="tip"/><joint name="hinge" type="revolute">'
        '<parent link="root"/><child link="arm"/><axis xyz="0 0 1"/>'
        '<limit lower="-2" upper="2" effort="10" velocity="2"/></joint>'
        '<joint name="weld" type="fixed"><parent link="arm"/>'
        '<child link="tip"/><origin xyz="1 0 0"/></joint></robot>',
        encoding="utf-8",
    )
    obj = object.__new__(Articulation)
    obj.device = torch.device("cpu")
    obj.pk_chain = create_pk_chain(str(path), obj.device)
    obj.joint_names = ["hinge"]
    angle = torch.tensor(0.3)
    actual = obj.compute_jacobian(angle.reshape(1, 1))
    expected = torch.tensor([-angle.sin(), angle.cos(), 0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(actual[0, :, 0], expected)


@pytest.mark.parametrize("branched", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_jacobian_maps_public_state_and_columns_by_name(
    branched: bool, batched: bool
) -> None:
    """Nonzero rotational FK exposes permutations even at equal state widths."""
    hinge = _hinge()
    extension = JointDesc(
        "extension",
        "door",
        "tip",
        JointType.PRISMATIC,
        axis=np.array([1.0, 0.0, 0.0]),
    )
    links, joints = ["root", "door", "tip"], [hinge, extension]
    if branched:
        links.append("sibling")
        joints.append(
            JointDesc("sibling_slide", "root", "sibling", JointType.PRISMATIC)
        )
    obj = _articulation(links, joints)
    # Equal-width input must also be reordered; width alone cannot identify
    # serial-order input. The unrelated sibling must never reach PK Jacobian.
    obj.joint_names = ["extension", "hinge"]
    qpos = torch.tensor([[0.17, 0.4], [-0.09, -0.3]])
    if branched:
        obj.joint_names.insert(1, "sibling_slide")
        qpos = torch.stack([qpos[:, 0], torch.tensor([0.8, -0.6]), qpos[:, 1]], dim=-1)
    if not batched:
        qpos = qpos[0]
    actual = obj.compute_jacobian(qpos, root_link_name="root", end_link_name="tip")
    assert actual.shape == (2 if batched else 1, 6, 2)
    epsilon = 0.001
    for column, name in enumerate(["hinge", "extension"]):
        offset = torch.zeros_like(qpos)
        offset[..., obj.joint_names.index(name)] = epsilon
        plus = obj.compute_fk(
            qpos + offset, link_names=["tip"], qpos_joint_names=obj.joint_names
        )
        minus = obj.compute_fk(
            qpos - offset, link_names=["tip"], qpos_joint_names=obj.joint_names
        )
        derivative = (plus[:, 0, :3, 3] - minus[:, 0, :3, 3]) / (2 * epsilon)
        torch.testing.assert_close(
            actual[:, :3, column], derivative, atol=2e-4, rtol=1e-3
        )


def test_jacobian_accepts_legacy_serial_input_and_zero_state() -> None:
    sibling = JointDesc("slide", "root", "drawer", JointType.PRISMATIC)
    obj = _articulation(["root", "door", "drawer"], [sibling, _hinge()])
    full = obj.compute_jacobian(torch.tensor([[0.8, 0.4]]), end_link_name="door")
    serial = obj.compute_jacobian(np.array([0.4]), end_link_name="door")
    torch.testing.assert_close(serial, full)
    zero = obj.compute_jacobian(None, end_link_name="door")
    torch.testing.assert_close(
        zero, obj.compute_jacobian(torch.zeros(2), end_link_name="door")
    )
    for kind, rows in [("trans", slice(0, 3)), ("rot", slice(3, 6))]:
        actual = obj.compute_jacobian(
            torch.tensor([0.8, 0.4]), end_link_name="door", jac_type=kind
        )
        torch.testing.assert_close(actual, full[:, rows, :])


@pytest.mark.parametrize("shape", [(), (1, 1, 1), (3,)])
def test_jacobian_rejects_invalid_state_shape(shape: tuple[int, ...]) -> None:
    obj = _articulation(["root", "door"], [_hinge()])
    with pytest.raises(ValueError, match="qpos"):
        obj.compute_jacobian(torch.zeros(shape), end_link_name="door")


def test_jacobian_rejects_missing_public_joint_name() -> None:
    obj = _articulation(["root", "door"], [_hinge()])
    obj.joint_names = ["unrelated"]
    with pytest.raises(ValueError, match="hinge"):
        obj.compute_jacobian(torch.zeros(1), end_link_name="door")


@pytest.mark.parametrize(
    "problem", ["unknown", "multiple_parents", "cycle", "disconnected"]
)
def test_rejects_invalid_tree(problem: str) -> None:
    joint = _hinge()
    links, joints = ["root", "door"], [joint]
    if problem == "unknown":
        joint.parent_link_name = "missing"
    elif problem == "multiple_parents":
        joints.append(JointDesc("other", "root", "door", JointType.FIXED))
    elif problem == "cycle":
        joint.parent_link_name = "door"
    else:
        links.append("orphan")
    with pytest.raises(ValueError, match="unknown|parents|cycle|disconnected"):
        _articulation(links, joints)


def test_rejects_unsupported_joint_type() -> None:
    joint = _hinge()
    joint.joint_type = SimpleNamespace(name="SPHERICAL")
    with pytest.raises(NotImplementedError, match="build_pk_chain=False"):
        _articulation(["root", "door"], [joint])


def test_rejects_zero_movable_joint_axis() -> None:
    joint = _hinge()
    joint.axis[:] = 0.0
    with pytest.raises(ValueError, match="nonzero axis"):
        _articulation(["root", "door"], [joint])
