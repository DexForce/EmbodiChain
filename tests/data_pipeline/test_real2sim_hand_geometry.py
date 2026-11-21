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
"""Full-hand collision selection and pre-grasp clearance regressions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from embodichain.data_pipeline.real2sim.adaptation._hand_geometry import (
    hand_collision_surfaces,
)
from embodichain.data_pipeline.real2sim.adaptation._approach import clear_approach


def _robot(tmp_path, articulated=False):
    def link(name, x):
        return f'<link name="{name}"><collision><origin xyz="{x} 0 0"/><geometry><box size="0.02 0.02 0.02"/></geometry></collision></link>'

    xml = '<robot name="test">' + link("arm", 0) + link("palm", 0)
    xml += '<joint name="wrist" type="fixed"><parent link="arm"/><child link="palm"/></joint>'
    names = []
    for side, x in [("left", -0.1), ("right", 0.1)]:
        child = side + ("_knuckle" if articulated else "_pad")
        names.append(side + "_jaw")
        xml += link(child, x)
        xml += f'<joint name="{names[-1]}" type="{"revolute" if articulated else "prismatic"}"><parent link="palm"/><child link="{child}"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>'
        if articulated:
            xml += link(side + "_pad", x)
            xml += f'<joint name="{side}_fixed" type="fixed"><parent link="{child}"/><child link="{side}_pad"/></joint>'
    path = tmp_path / "hand.urdf"
    path.write_text(xml + "</robot>")
    robot = SimpleNamespace(path=path, names=names, jaw_columns=[[0, 1]])
    robot.finger_surfaces = lambda side: (
        ["left_pad", "right_pad"],
        np.array([[-1, 0, 0], [1, 0, 0]]),
    )
    return robot


@pytest.mark.parametrize("articulated", [False, True])
def test_collision_subtree_includes_palm_and_linkage_excludes_arm(
    tmp_path, articulated
):
    robot = _robot(tmp_path, articulated)
    original = robot.path.read_bytes()
    clouds = hand_collision_surfaces(robot, 0)
    assert set(clouds) == {"palm", "left_pad", "right_pad"} | (
        {"left_knuckle", "right_knuckle"} if articulated else set()
    )
    assert np.allclose(clouds["left_pad"].mean(0), [-0.1, 0, 0], atol=1e-7)
    assert robot.path.read_bytes() == original


def test_approach_lifts_palm_collision_even_when_pads_clear(tmp_path):
    robot = _robot(tmp_path)

    def transforms(q):
        result = np.tile(np.eye(4), (len(q), 1, 1))
        result[:, 2, 3] = q[:, 0]
        return result

    robot.links = lambda q: {
        link: transforms(q) for link in ["palm", "left_pad", "right_pad"]
    }
    robot.tcp = lambda q, side: transforms(q)
    obstacle = tmp_path / "obstacle.glb"
    trimesh.creation.box([0.025, 0.025, 0.025]).export(obstacle)
    scene = {
        "objects": [
            {"id": "target", "path": str(obstacle), "transform": np.eye(4).tolist()}
        ]
    }

    def solve(robot, q, side, target):
        return np.array([target[2, 3]]), {"tcp_error_m": 0.0}

    qpos, report = clear_approach(
        robot,
        np.zeros((3, 1)),
        0,
        scene,
        {},
        (),
        0,
        1,
        solve,
        grasp=2,
        clearance_m=0.003,
    )
    assert qpos[:2, 0].min() > 0.025
    assert set(report["collision_links"]) == {"palm", "left_pad", "right_pad"}
    assert report["minimum_checked_gap_m"] >= 0.003


def test_rejects_hand_root_that_contains_an_arm_joint(tmp_path):
    robot = _robot(tmp_path)
    robot.names = ["left_jaw"]
    robot.jaw_columns = [[0]]
    with pytest.raises(ValueError, match="non-hand moving joint"):
        hand_collision_surfaces(robot, 0)


def test_idle_window_checks_transition_until_next_closure():
    from embodichain.data_pipeline.real2sim.adaptation._approach import (
        _idle_windows,
    )

    phases = [
        dict(side=0, close_start=60, grasp=90, open_end=539),
        dict(side=0, close_start=711, grasp=741, open_end=1154),
    ]
    windows = _idle_windows(
        phases,
        {(0, 90): {"object_id": None}, (0, 741): {"object_id": "block"}},
        1398,
        180,
    )
    assert windows[0]["checked_end"] == 711
    assert windows[0]["fade_end"] == 741
    assert windows[0]["start"] <= 539 < 544 <= windows[0]["checked_end"]


def test_grasp_returns_lowest_cost_candidate_with_palm_penetration_diagnostics(
    tmp_path, monkeypatch
):
    from embodichain.data_pipeline.real2sim.adaptation import _contact_fit

    robot = _robot(tmp_path)
    robot.arm_columns = [np.arange(3)]
    robot.arm_limits = [(np.full(3, -1.0), np.full(3, 1.0))]
    robot.config = {"tcp_links": ["palm"]}
    robot.prefixes = ["arm"]
    robot.offset = np.eye(4)
    robot.opening_limits = [(0.001, 0.06)]

    def tcp(q, side):
        matrices = np.tile(np.eye(4), (len(q), 1, 1))
        matrices[:, :3, 3] = q[:, :3]
        return matrices

    robot.tcp = tcp
    robot.ik = lambda source, side, target: (
        np.r_[target[:3, 3], source[3:]],
        {},
    )
    robot.links = lambda q: {
        link: tcp(q, 0) for link in ["palm", "left_pad", "right_pad"]
    }
    robot.half_open = lambda q, side: q[..., 3]

    def set_open(q, side, opening):
        q = q.copy()
        q[..., 3] = opening
        return q

    robot.set_half_open = set_open
    monkeypatch.setattr(
        _contact_fit,
        "least_squares",
        lambda fun, x0, **kw: SimpleNamespace(
            x=np.asarray(x0),
            cost=float((x0[-1] - 0.05) ** 2 + np.sum(np.asarray(x0[:3]) ** 2)),
            success=True,
            nfev=0,
        ),
    )
    mesh = tmp_path / "box.glb"
    trimesh.creation.box([0.04, 0.04, 0.04]).export(mesh)
    target = dict(id="target", path=str(mesh), transform=np.eye(4))
    table_pose = np.eye(4)
    table_pose[2, 3] = -0.2
    scene = dict(
        objects=[target, dict(id="table", path=str(mesh), transform=table_pose)]
    )
    visuals = []
    for link, x in [("left_pad", -0.1), ("right_pad", 0.1)]:
        origin = np.eye(4)
        origin[0, 3] = x
        visuals.append(dict(link=link, path=str(mesh), origin=origin))
    fitted, opening, report = _contact_fit.fit_grasp(
        scene,
        tuple(visuals),
        robot,
        np.array([0.0, 0.0, 0.0, 0.05]),
        0,
        target,
        np.eye(4),
        closing_half_open_m=0.06,
        seed_qpos=(
            np.array([0.3, 0.0, 0.0, 0.05]),
            np.array([0.0, 0.0, 0.0, 0.05]),
        ),
    )
    assert np.isfinite(fitted).all()
    assert opening == pytest.approx(0.05)
    assert report["hand_clearance_m"]["palm"]["target"] < -0.009
    assert report["hand_clearance_violation_m"] > 0.009
    assert len(report["fingers"]) == 2
    assert "open_target_reverse_penetration_m" in report
    assert "before" in report
    assert "lowest soft objective" in report["search"]["selection"]
    assert report["search"]["candidate_source"] == "caller"
    assert [c["seed"] for c in report["search"]["candidates"]] == [
        "supplied_0",
        "supplied_1",
    ]
    supplied = report["search"]["candidates"][0]
    assert supplied["seed"] == "supplied_0"
    assert supplied["arm_and_half_open"][0] == pytest.approx(0.3)
    for candidate in report["search"]["candidates"][1:]:
        assert candidate["hand_clearance_m"]["palm"]["target"] < -0.009
        assert candidate["hand_clearance_violation_m"] > 0.009


def test_open_patch_penetration_uses_reverse_closed_hand_distance(tmp_path):
    from embodichain.data_pipeline.real2sim.adaptation._hand_geometry import (
        closed_hand_collision_parts,
        reverse_hand_penetration,
    )
    from embodichain.data_pipeline.real2sim.adaptation._contact_fit import (
        _body,
        _obstacle_distances,
    )

    robot = _robot(tmp_path)
    original = robot.path.read_bytes()
    parts = closed_hand_collision_parts(robot, ["palm"])["palm"]
    # Open planar target cuts through the palm's centre. A forward unsigned
    # query sees proximity only, not how deeply the target enters the hand.
    patch = trimesh.Trimesh(
        vertices=[[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    body = _body("patch", patch, np.eye(4)[None], 16)
    assert not body.closed
    forward = _obstacle_distances(body, np.array([[-0.01, 0, 0], [0.01, 0, 0]]), 0.004)
    assert np.all(forward >= 0)
    assert np.max(forward) < 1e-7
    samples = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    assert reverse_hand_penetration(parts, samples, np.eye(4)[None]) == pytest.approx(
        0.01, abs=1e-7
    )
    moved = np.eye(4)
    moved[2, 3] = 0.02
    assert reverse_hand_penetration(parts, samples, moved[None]) == 0
    assert robot.path.read_bytes() == original


def test_overlapping_closed_collision_parts_do_not_cancel_inside_parity(tmp_path):
    from embodichain.data_pipeline.real2sim.adaptation._hand_geometry import (
        closed_hand_collision_parts,
        reverse_hand_penetration,
    )

    path = tmp_path / "overlap.urdf"
    path.write_text(
        '<robot name="overlap"><link name="palm">'
        + "".join(
            f'<collision><origin xyz="{x} 0 0"/><geometry><box size="0.04 0.04 0.04"/></geometry></collision>'
            for x in [-0.005, 0.005]
        )
        + "</link></robot>"
    )
    parts = closed_hand_collision_parts(SimpleNamespace(path=path), ["palm"])["palm"]
    assert len(parts) == 2
    # This point is inside both solids. One combined parity query could label
    # it outside, whereas their union must retain penetration.
    assert reverse_hand_penetration(
        parts, np.zeros((1, 3)), np.eye(4)[None]
    ) == pytest.approx(0.015, abs=1e-7)


def test_reverse_collision_query_applies_authored_origin_and_world_fk_once(tmp_path):
    from embodichain.data_pipeline.real2sim.adaptation._hand_geometry import (
        closed_hand_collision_parts,
        reverse_hand_penetration,
    )

    robot = _robot(tmp_path)
    parts = closed_hand_collision_parts(robot, ["left_pad"])["left_pad"]
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [0.4, 0.5, 0.6]
    point = np.array([[-0.1, 0, 0]]) @ transform[:3, :3].T + transform[:3, 3]
    assert reverse_hand_penetration(parts, point, transform[None]) == pytest.approx(
        0.01, abs=1e-7
    )
