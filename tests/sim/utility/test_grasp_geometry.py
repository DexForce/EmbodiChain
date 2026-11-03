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

"""Read-only hand capture and per-runtime deferred paired-provider contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest
import torch

from embodichain.lab.sim.utility import grasp_geometry as capture
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    FixedGripperCollisionGeometry,
    ParallelJawGraspCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.coordinated import (
    CoordinatedGraspCfg,
    CoordinatedGraspPoseGenerator,
)


def _fixture():
    return json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("hand_geometry_f1.json")
        .read_text()
    )


class _Robot:
    def __init__(self, path, *, renamed_pads=False, moving_tcp=False):
        self.joint_names = ["rail", "finger_left", "finger_right"]
        self.qpos = torch.tensor([[0.4, 0.0, 0.0]])
        self.reads, self.calls = 0, []
        self.pads = (
            ("contact_A", "contact_B")
            if renamed_pads
            else ("negative_finger_pad", "positive_finger_pad")
        )
        tcp = torch.eye(4)
        tcp[2, 3] = 0.1
        self.cfg = NS(
            fpath=str(path),
            control_parts={"custom_grip": ["finger_right", "finger_left"]},
            solver_cfg={
                "custom_arm": NS(
                    end_link_name=self.pads[0] if moving_tcp else "mount",
                    tcp=tcp.tolist(),
                )
            },
        )

    def get_qpos(self):
        self.reads += 1
        return self.qpos

    def compute_fk(self, *, qpos, link_names, qpos_joint_names):
        self.calls.append(
            {
                "link_names": link_names,
                "joint_names": list(qpos_joint_names),
                "qpos": qpos.tolist(),
            }
        )
        result = []
        for name in link_names:
            pose = torch.eye(4).repeat(len(qpos), 1, 1)
            pose[:, 0, 3] = qpos[:, 0]
            if name == self.pads[0]:
                pose[:, 0, 3] += qpos[:, 1]
                pose[:, 2, 3] = 0.1
            elif name == self.pads[1]:
                pose[:, 0, 3] += qpos[:, 2]
                pose[:, 2, 3] = 0.1
            result.append(pose)
        return torch.stack(result, 1)

    def set_qpos(self, *args, **kwargs):
        raise AssertionError("Geometry capture may not write robot state.")

    def update(self, *args, **kwargs):
        raise AssertionError("Geometry capture may not advance physics.")


def _case(tmp_path, *, renamed_pads=False, moving_tcp=False):
    fixture = _fixture()
    source = fixture["urdf"]
    if renamed_pads:
        source = source.replace("negative_finger_pad", "contact_A").replace(
            "positive_finger_pad", "contact_B"
        )
        document = ET.fromstring(source)
        positive = document.find("./link[@name='contact_B']")
        document.remove(positive)
        document.insert(3, positive)
        source = ET.tostring(document, encoding="unicode")
    path = tmp_path / "hand.urdf"
    path.write_text(source)
    robot = _Robot(path, renamed_pads=renamed_pads, moving_tcp=moving_tcp)
    cfg = capture.RobotHandGeometryCfg(
        motion_part="custom_arm",
        hand_part="custom_grip",
        hand_root_link="hand_base",
        pad_link_names=robot.pads,
        open_qpos=(0.06, -0.06),
        closed_qpos=(0.01, -0.01),
        sample_count=65,
        body_max_edge=0.01,
        pad_max_edge=0.004,
        max_subdivisions=10,
    )
    return robot, cfg


@pytest.mark.parametrize(
    "renamed_pads,virtual_tcp", [(False, False), (True, False), (True, True)]
)
def test_named_fk_sampling_matches_source_without_control_or_pad_name_guessing(
    tmp_path, renamed_pads, virtual_tcp
):
    robot, cfg = _case(tmp_path, renamed_pads=renamed_pads)
    if virtual_tcp:
        document = ET.parse(robot.cfg.fpath).getroot()
        ET.SubElement(document, "link", name="virtual_tcp")
        joint = ET.SubElement(document, "joint", name="fixed_virtual_tcp", type="fixed")
        ET.SubElement(joint, "parent", link="hand_base")
        ET.SubElement(joint, "child", link="virtual_tcp")
        ET.ElementTree(document).write(robot.cfg.fpath)
        robot.cfg.solver_cfg[cfg.motion_part].end_link_name = "virtual_tcp"
    before = robot.qpos.clone()
    result = capture.build_parallel_jaw_hand_geometry(robot, cfg=cfg)
    fixture = _fixture()
    assert robot.reads == fixture["reads"] == 1
    assert len(robot.calls) == 1
    expected_call = fixture["fk_calls"][0]
    assert robot.calls[0]["joint_names"] == expected_call["joint_names"]
    assert robot.calls[0]["qpos"] == expected_call["qpos"]
    pad_order = tuple(reversed(robot.pads)) if renamed_pads else robot.pads
    assert robot.calls[0]["link_names"] == [
        "hand_base",
        *pad_order,
        *(["virtual_tcp"] if virtual_tcp else []),
        "virtual_tcp" if virtual_tcp else "mount",
    ]
    for name, expected in fixture["expected"].items():
        value = getattr(result, name)
        assert list(value.shape) == expected["shape"]
        assert hashlib.sha256(value.numpy().tobytes()).hexdigest() == expected["sha256"]
        assert not value.requires_grad
    torch.testing.assert_close(
        result.gaps[[0, 32, 64]], torch.tensor(fixture["gap_samples"]), rtol=0, atol=0
    )
    torch.testing.assert_close(robot.qpos, before)


def test_missing_mimic_command_is_rejected_before_fk(tmp_path):
    robot, cfg = _case(tmp_path)
    robot.cfg.control_parts["custom_grip"] = ["finger_right"]
    cfg.open_qpos, cfg.closed_qpos = (0.06,), (0.01,)
    with pytest.raises(ValueError, match="including mimic"):
        capture.build_parallel_jaw_hand_geometry(robot, cfg=cfg)
    assert robot.reads == 0 and robot.calls == []


def test_opposite_arm_tcp_is_rejected_even_when_it_is_stationary(tmp_path):
    robot, cfg = _case(tmp_path)
    document = ET.parse(robot.cfg.fpath).getroot()
    ET.SubElement(document, "link", name="opposite_tcp")
    joint = ET.SubElement(document, "joint", name="opposite_arm_joint", type="revolute")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="opposite_tcp")
    ET.ElementTree(document).write(robot.cfg.fpath)
    robot.joint_names.append("opposite_arm_joint")
    robot.qpos = torch.cat((robot.qpos, torch.tensor([[0.2]])), dim=1)
    robot.cfg.solver_cfg["other_motion"] = NS(
        end_link_name="opposite_tcp", tcp=torch.eye(4).tolist()
    )
    cfg.motion_part = "other_motion"

    with pytest.raises(ValueError, match="fixed URDF joints"):
        capture.build_parallel_jaw_hand_geometry(robot, cfg=cfg)
    assert robot.reads == 0 and robot.calls == []


def test_moving_finger_tcp_is_rejected_before_surface_capture(tmp_path, monkeypatch):
    robot, cfg = _case(tmp_path, moving_tcp=True)
    surfaces = Mock(
        side_effect=AssertionError("Do not sample geometry in a moving TCP frame.")
    )
    monkeypatch.setattr(capture, "sample_urdf_collision_surfaces", surfaces)
    with pytest.raises(ValueError, match="fixed URDF joints"):
        capture.build_parallel_jaw_hand_geometry(robot, cfg=cfg)
    surfaces.assert_not_called()
    assert robot.calls == [] and robot.reads == 0


def _delegate():
    return AntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(),
        collision_geometry=FixedGripperCollisionGeometry(
            torch.zeros(1, 3), {"grasp_from_tcp": torch.eye(4).tolist()}
        ),
        collision_cfg=ParallelJawGraspCollisionCfg(filter_ground_collision=False),
    )


def _paired_cfg():
    return CoordinatedGraspCfg(
        contact_tolerance=0.0005,
        minimum_normal_cosine=0.5,
        normal_probe_distance=1e-5,
        max_candidates=32,
        candidate_chunk_size=32,
        cargo_clearance_reserve=0.008,
    )


def _query(bounds):
    return dict(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        obj_poses=torch.eye(4)[None],
        left_to_right_arm_direction=torch.tensor([1.0, 0, 0]),
        approach_direction=torch.tensor([0.0, 0, -1.0]),
        cargo_bounds=bounds,
    )


def test_single_arm_does_not_capture_and_geometry_reuses_the_same_environment_owner(
    tmp_path, monkeypatch
):
    constructed, queries = [], []

    class Sampler:
        def __init__(self, left, right, left_geometry, right_geometry, *, cfg):
            constructed.append((left_geometry, right_geometry))

        def get_dual_arm_valid_grasp_poses(self, **kwargs):
            queries.append(kwargs)
            return [None]

    monkeypatch.setattr(capture, "CoordinatedGraspSampler", Sampler)
    robot, cfg = _case(tmp_path)
    left, right = _delegate(), _delegate()
    left_capture = capture.RobotHandGeometryProvider(robot, cfg=cfg)
    right_capture = capture.RobotHandGeometryProvider(robot, cfg=cfg)
    provider = capture.RobotCoordinatedGraspProvider(
        left, right, left_capture, right_capture, cfg=_paired_cfg()
    )
    facade = CoordinatedGraspPoseGenerator(left, provider)
    single = [(torch.eye(4)[None], torch.zeros(1))]
    left.get_valid_grasp_poses = Mock(return_value=single)
    assert facade.get_valid_grasp_poses(marker="unchanged") is single
    left.get_valid_grasp_poses.assert_called_once_with(marker="unchanged")
    assert robot.reads == 0 and robot.calls == [] and constructed == []

    bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    assert facade.get_dual_arm_valid_grasp_poses(**_query(bounds)) == [None]
    assert robot.reads == 2 and len(robot.calls) == 2 and len(constructed) == 1
    robot.qpos[0, 0] = 0.7
    # A new bridge/engine may obtain another facade while the adapter retains
    # its owner-bound provider across the episode's reset posture.
    second_facade = CoordinatedGraspPoseGenerator(right, provider)
    assert second_facade.get_dual_arm_valid_grasp_poses(**_query(None)) == [None]
    assert robot.reads == 2 and len(robot.calls) == 2 and len(constructed) == 1
    assert queries[0]["cargo_bounds"] is bounds and queries[1]["cargo_bounds"] is None

    other = _Robot(Path(robot.cfg.fpath))
    other_left = capture.RobotHandGeometryProvider(other, cfg=cfg)
    other_right = capture.RobotHandGeometryProvider(other, cfg=cfg)
    with pytest.raises(ValueError, match="same runtime"):
        capture.RobotCoordinatedGraspProvider(
            left, right, left_capture, other_right, cfg=_paired_cfg()
        )
    separate = capture.RobotCoordinatedGraspProvider(
        left, right, other_left, other_right, cfg=_paired_cfg()
    )
    assert other.reads == 0
    separate.get_dual_arm_valid_grasp_poses(**_query(None))
    assert other.reads == 2 and len(constructed) == 2
    assert constructed[0][0] is not constructed[1][0]
