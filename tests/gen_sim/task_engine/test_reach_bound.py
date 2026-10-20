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

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.task_engine._task_program.reach_bound import (
    build_revolute_reach_bound,
)


def _pose(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def _urdf(tmp_path: Path, kind: str = "revolute") -> Path:
    path = tmp_path / "chain.urdf"
    path.write_text(f"""<robot name="test">
        <link name="root"/><link name="mount"/><link name="upper"/>
        <link name="offset"/><link name="lower"/><link name="tip"/>
        <joint name="prefix" type="fixed"><parent link="root"/><child link="mount"/>
          <origin xyz="0.5 0 0" rpy="0 0 1.5707963267948966"/></joint>
        <joint name="first" type="revolute"><parent link="mount"/><child link="upper"/>
          <origin xyz="0.1 0 0"/><axis xyz="0 0 1"/></joint>
        <joint name="offset" type="fixed"><parent link="upper"/><child link="offset"/>
          <origin xyz="0.3 0 0" rpy="0.2 -0.3 0.4"/></joint>
        <joint name="last" type="{kind}"><parent link="offset"/><child link="lower"/>
          <origin xyz="0 0.4 0"/><axis xyz="0 1 0"/></joint>
        <joint name="tail" type="fixed"><parent link="lower"/><child link="tip"/>
          <origin xyz="0.1 0 0" rpy="0.2 0.3 -0.4"/></joint>
        </robot>""")
    return path


def test_transformed_root_and_tcp_strip_only_fixed_tail(tmp_path: Path) -> None:
    path = _urdf(tmp_path)
    world = _pose((2.0, -1.0, 0.7), (0.2, -0.4, 0.6))
    tcp = _pose((0.0, 0.0, 0.2), (-0.3, 0.2, -0.1))
    before_world, before_tcp, before_urdf = world.copy(), tcp.copy(), path.read_bytes()
    bound = build_revolute_reach_bound(path, "root", "tip", world, tcp)
    expected_pivot = (
        world @ _pose((0.5, 0, 0), (0, 0, np.pi / 2)) @ _pose((0.1, 0, 0))
    )[:3, 3]
    expected_tail = _pose((0.1, 0, 0), (0.2, 0.3, -0.4)) @ tcp
    np.testing.assert_allclose(bound.pivot_world, expected_pivot, atol=1e-12)
    np.testing.assert_allclose(bound.tail, expected_tail, atol=1e-12)
    assert bound.radius == pytest.approx(0.7)
    last = _pose(expected_pivot + np.array([0.65, 0.0, 0.0]), (0.1, 0.2, 0.3))
    target = last @ expected_tail
    before_target = target.copy()
    receipt = bound.check(target)
    assert receipt["within_bound"] is True
    assert receipt["required_distance_m"] == pytest.approx(0.65)
    assert receipt["scope"] == "necessary_only"
    assert receipt["ik_qualified"] is False
    np.testing.assert_array_equal(world, before_world)
    np.testing.assert_array_equal(tcp, before_tcp)
    np.testing.assert_array_equal(target, before_target)
    assert path.read_bytes() == before_urdf
    with pytest.raises(ValueError):
        bound.pivot_world[0] = 0
    with pytest.raises(ValueError):
        bound.tail[0, 3] = 0


def test_outside_and_tolerance_expansion(tmp_path: Path) -> None:
    bound = build_revolute_reach_bound(
        _urdf(tmp_path), "root", "tip", np.eye(4), _pose((0, 0, 0.2))
    )
    target = _pose(bound.pivot_world + np.array([0.74, 0.0, 0.0])) @ bound.tail
    assert bound.check(target)["within_bound"] is False
    receipt = bound.check(target, pos_tolerance=0.01, rot_tolerance=0.2)
    allowance = 0.01 + 2 * np.linalg.norm(bound.tail[:3, 3]) * np.sin(0.1)
    assert receipt["tolerance_allowance_m"] == pytest.approx(allowance)
    assert receipt["within_bound"] is True


def test_single_continuous_joint_has_zero_pivot_radius(tmp_path: Path) -> None:
    path = tmp_path / "one.urdf"
    path.write_text(
        '<robot name="r"><link name="a"/><link name="b"/><joint name="j" type="continuous"><parent link="a"/><child link="b"/><origin xyz="0 0 2"/></joint></robot>'
    )
    bound = build_revolute_reach_bound(path, "a", "b", np.eye(4), _pose((0, 0, 3)))
    assert bound.radius == 0.0
    assert bound.check(_pose((0, 0, 5)))["within_bound"] is True
    assert bound.check(_pose((0, 0, 5.01)))["within_bound"] is False


@pytest.mark.parametrize("kind", ["prismatic", "floating", "planar", "unknown"])
def test_unsupported_selected_joint_rejected(tmp_path: Path, kind: str) -> None:
    with pytest.raises(NotImplementedError, match="joint type"):
        build_revolute_reach_bound(
            _urdf(tmp_path, kind), "root", "tip", np.eye(4), np.eye(4)
        )


@pytest.mark.parametrize("field", ["root", "tcp", "target"])
@pytest.mark.parametrize("damage", ["nan", "scale", "reflection", "bottom", "shape"])
def test_invalid_pose_rejected(tmp_path: Path, field: str, damage: str) -> None:
    value = np.eye(4)
    if damage == "nan":
        value[0, 3] = np.nan
    elif damage == "scale":
        value[0, 0] = 2
    elif damage == "reflection":
        value[0, 0] = -1
    elif damage == "bottom":
        value[3, 0] = 1
    else:
        value = np.eye(3)
    root, tcp = (value if field == "root" else np.eye(4)), (
        value if field == "tcp" else np.eye(4)
    )
    if field == "target":
        bound = build_revolute_reach_bound(_urdf(tmp_path), "root", "tip", root, tcp)
        with pytest.raises(ValueError, match="SE\\(3\\)"):
            bound.check(value)
    else:
        with pytest.raises(ValueError, match="SE\\(3\\)"):
            build_revolute_reach_bound(_urdf(tmp_path), "root", "tip", root, tcp)


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, True])
@pytest.mark.parametrize("key", ["pos_tolerance", "rot_tolerance"])
def test_invalid_tolerance_rejected(tmp_path: Path, value: float, key: str) -> None:
    bound = build_revolute_reach_bound(
        _urdf(tmp_path), "root", "tip", np.eye(4), np.eye(4)
    )
    with pytest.raises(ValueError, match="finite non-negative"):
        bound.check(np.eye(4), **{key: value})


@pytest.mark.parametrize(
    "damage",
    ["missing", "disconnected", "cycle", "duplicate", "origin", "axis", "fixed"],
)
def test_invalid_chain_rejected(tmp_path: Path, damage: str) -> None:
    path = _urdf(tmp_path)
    text = path.read_text()
    root = "root"
    if damage == "missing":
        root = "absent"
    elif damage == "disconnected":
        root = "mount"
        text = text.replace('<parent link="root"/>', '<parent link="tip"/>')
        text = text.replace('<parent link="mount"/>', '<parent link="root"/>')
    elif damage == "cycle":
        text = text.replace('<parent link="mount"/>', '<parent link="tip"/>')
    elif damage == "duplicate":
        text = text.replace('<child link="lower"/>', '<child link="upper"/>')
    elif damage == "origin":
        text = text.replace('xyz="0.3 0 0"', 'xyz="nan 0 0"')
    elif damage == "axis":
        text = text.replace('xyz="0 1 0"', 'xyz="0 0 0"')
    else:
        text = text.replace('type="revolute"', 'type="fixed"')
    path.write_text(text)
    with pytest.raises(ValueError):
        build_revolute_reach_bound(path, root, "tip", np.eye(4), np.eye(4))
