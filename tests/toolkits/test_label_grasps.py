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

"""Check the coordinate, observation and row-alignment contracts of grasp labels."""

from __future__ import annotations

import math

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from embodichain.lab.scripts import label_grasps as labels


def test_object_frame_nms_keeps_width_and_verification_placement() -> None:
    placements = torch.eye(4).repeat(2, 1, 1)
    placements[1, :3, 3] = torch.tensor([0.4, -0.2, 0.1])
    placements[1, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    local = torch.eye(4).repeat(3, 1, 1)
    local[:, 0, 3] = torch.tensor([0.0, 0.01, 0.02])
    directions = torch.tensor([[0.0, 0.0, -1.0], [0.6, 0.0, -0.8]])
    lower = dict(
        poses=local[:1],
        widths=torch.tensor([0.02]),
        costs=torch.tensor([2.0]),
        tested=torch.tensor([True]),
        success=torch.tensor([True]),
    )
    generator = SimpleNamespace(
        get_grasp_candidates=lambda **_: [
            (
                placements[1] @ local,
                torch.tensor([0.03, 0.04, 0.05]),
                torch.tensor([0.0, 1.0, torch.inf]),
            )
        ]
    )
    higher = labels._candidates(
        generator, (torch.zeros(2, 3), None), placements[1], directions[1]
    )
    higher["tested"][:] = True
    higher["success"][0] = True
    result = labels._labels(
        [(0, 0, lower), (1, 1, higher)], placements, directions, labels._SamplingCfg()
    )
    torch.testing.assert_close(result["grasp_pose"], local[:1])
    torch.testing.assert_close(result["open_length"], torch.tensor([0.03]))
    assert result["object_pose_index"].tolist() == [1]
    torch.testing.assert_close(result["approach_direction_world"], directions[1:])
    lower["success"][:] = False
    empty = labels._labels(
        [(0, 0, lower)], placements, directions, labels._SamplingCfg()
    )
    assert empty["grasp_pose"].shape == (0, 4, 4) and empty["open_length"].shape == (0,)


def test_partial_batch_retries_only_missing_observations(tmp_path: Path) -> None:
    calls = []

    def validate(batch):
        calls.append(batch[:, 0, 3].tolist())
        if len(calls) == 1:
            return [dict(status="ok", success=False), dict(status="missing")]
        return [dict(status="ok", success=True)] * 2

    v = SimpleNamespace(
        sim=SimpleNamespace(num_envs=2),
        restore_placement=lambda p: None,
        validate=validate,
    )
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, 0, 3] = torch.arange(3)
    group = dict(
        poses=poses,
        tested=torch.zeros(3, dtype=torch.bool),
        success=torch.zeros(3, dtype=torch.bool),
    )
    groups = [
        ({key: value[:1] for key, value in group.items()}, tmp_path / "first.pt"),
        ({key: value[1:] for key, value in group.items()}, tmp_path / "second.pt"),
    ]
    assert labels._validate_placement(v, torch.eye(4), groups) == 1
    loaded = [(torch.load(path, weights_only=True), path) for _, path in groups]
    assert labels._validate_placement(v, torch.eye(4), loaded) == 0
    assert calls == [[0, 1], [2, 2], [1, 1]]
    assert loaded[0][0]["success"].tolist() == [False]
    assert loaded[1][0]["success"].tolist() == [True, True]


def test_stable_orientations_ignore_translation_and_nearby_rotations() -> None:
    cfg = labels._SamplingCfg(stable_poses=2, max_drops=4)
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[1, 0, 3] = 0.5
    angle = np.deg2rad(10)
    poses[2, :2, :2] = torch.tensor(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    poses[3, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    v = SimpleNamespace(
        sim=SimpleNamespace(num_envs=4), drop_objects=lambda rots, height: poses
    )
    actual = labels._placements(v, cfg, np.random.default_rng(3))
    torch.testing.assert_close(actual, poses[[0, 3]])
    directions = labels._directions(np.random.default_rng(8), 24)
    assert (directions[:, 2] <= 0).all()
    torch.testing.assert_close(directions.norm(dim=1), torch.ones(24))


def test_validated_pickup_cache_identity(tmp_path: Path) -> None:
    from embodichain.lab.sim.utility._validated_grasps import _cache_key
    import hashlib

    hand = tmp_path / "hand.urdf"
    hand.write_bytes(b"hand")
    vertices = torch.eye(3)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    policy = {"seed": 0, "directions": 64}
    expected = hashlib.md5(
        vertices.numpy().tobytes()
        + triangles.numpy().tobytes()
        + b'{"directions":64,"seed":0}'
    ).hexdigest()
    key = _cache_key(hand, vertices, triangles, policy)
    assert key == f"{hashlib.md5(b'hand').hexdigest()}-{expected}"
    assert key != _cache_key(hand, vertices + 1, triangles, policy)
    assert key != _cache_key(hand, vertices, triangles.flip(1), policy)
    assert key != _cache_key(hand, vertices, triangles, policy | {"seed": 1})
    hand.write_bytes(b"different hand")
    assert key != _cache_key(hand, vertices, triangles, policy)


def test_live_cost_precedes_tcp_alignment_and_uses_the_current_support() -> None:
    from embodichain.lab.sim.utility._validated_grasps import _rank_labels
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGeneratorCfg,
        ParallelJawGraspCollisionCfg,
    )

    # Candidate 0 points exactly along the first approach, but is off center.
    # 1 and 2 have equal cost and opposite TCP axes. Rotating the object swaps them.
    tcp = torch.eye(4).repeat(3, 1, 1)
    tcp[0, 0, 3] = 0.02
    tcp[1, 1, 1] = tcp[1, 2, 2] = -1
    obj = torch.eye(4).repeat(2, 1, 1)
    obj[1, 1, 1] = obj[1, 2, 2] = -1
    geom = SimpleNamespace(
        provenance={"grasp_from_tcp": torch.eye(4).tolist()},
        points=torch.tensor([[0.0, 0.0, 0.1]]),
    )
    args = (
        tcp,
        torch.tensor([[-0.03, -0.03, -0.03], [0.03, 0.03, 0.03]]),
        obj,
        torch.tensor([0.0, 0.0, 1.0]),
        geom,
        AntipodalGraspPoseGeneratorCfg(),
    )
    order = _rank_labels(
        *args, ParallelJawGraspCollisionCfg(filter_ground_collision=False), "center"
    )
    assert order.tolist() == [[2, 1, 0], [1, 2, 0]]
    within_cone = _rank_labels(
        *args,
        ParallelJawGraspCollisionCfg(filter_ground_collision=False),
        tcp_tolerance_deg=15,
    )
    assert within_cone.tolist() == [[2, 0], [1, -1]]
    on_table = _rank_labels(*args, ParallelJawGraspCollisionCfg(), "center")
    assert on_table.tolist() == [[2, 0], [1, -1]]


def test_validated_pickup_scores_the_contact_axis_in_a_calibrated_hand() -> None:
    from embodichain.lab.sim.utility._validated_grasps import _rank_labels
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGeneratorCfg,
        ParallelJawGraspCollisionCfg,
    )

    tcp = torch.eye(4).repeat(2, 1, 1)
    # Rx(90) leaves TCP X horizontal. Ry(90) leaves TCP Y horizontal.
    tcp[0, :3, :3] = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    tcp[1, :3, :3] = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    grasp_from_tcp = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    geom = SimpleNamespace(
        provenance={"grasp_from_tcp": grasp_from_tcp}, points=torch.zeros(1, 3)
    )
    order = _rank_labels(
        tcp,
        torch.zeros(1, 3),
        torch.eye(4)[None],
        torch.tensor([0.0, 0.0, 1.0]),
        geom,
        AntipodalGraspPoseGeneratorCfg(
            angle_weight=1, center_weight=0, approach_deviation_angle=math.radians(20)
        ),
        ParallelJawGraspCollisionCfg(filter_ground_collision=False),
        "center",
    )
    assert order.tolist() == [[1]]  # Candidate 0 closes parallel to the live approach.


def test_validated_grasps_partition_contact_centers_per_arm_in_the_live_frame() -> None:
    from embodichain.lab.sim.utility._validated_grasps import _rank_labels
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGeneratorCfg,
        ParallelJawGraspCollisionCfg,
    )

    tcp = torch.eye(4).repeat(5, 1, 1)
    tcp[:, 0, 3] = torch.tensor([-0.9, -0.1, 0.0, 0.2, 0.8])
    objects = torch.eye(4).repeat(2, 1, 1)
    objects[1, :2, :2] *= -1
    args = (
        tcp,
        torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        objects,
        torch.tensor([0.0, 0.0, -1.0]),
        SimpleNamespace(
            provenance={"grasp_from_tcp": torch.eye(4).tolist()},
            points=torch.zeros(1, 3),
        ),
        AntipodalGraspPoseGeneratorCfg(),
        ParallelJawGraspCollisionCfg(),
    )
    params = dict(axis=torch.tensor([1.0, 0.0, 0.0]), on_support=False)
    assert _rank_labels(*args, **params, positive=True, middle_gap=0.5).tolist() == [
        [4],
        [0],
    ]
    assert _rank_labels(
        *args, **params, positive=torch.tensor([False, True]), middle_gap=0.5
    ).tolist() == [[0], [0]]
    assert _rank_labels(*args, **params, middle_only=True).tolist() == [
        [2, 1, 3],
        [2, 1, 3],
    ]


def test_physical_label_adjustments_apply_in_local_tcp_axes_before_validation() -> None:
    from embodichain.lab.sim.utility._validated_grasps import _adjust_candidates

    pose = torch.eye(4)[None]
    adjusted = _adjust_candidates(
        pose,
        dict(
            mirrored=True,
            rotate_upright=math.pi / 2,
            upright_axis=[0, 1, 0],
            retract=0.01,
        ),
    )
    torch.testing.assert_close(
        adjusted[0, :3, 0], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        adjusted[0, :3, 2], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        adjusted[0, :3, 3], torch.tensor([0.0, 0.01, 0.0]), atol=1e-6, rtol=0.0
    )
    calibration = torch.eye(4)
    calibration[0, 3] = 0.05
    calibrated = _adjust_candidates(
        pose,
        dict(
            mirrored=True,
            rotate_upright=math.pi / 2,
            upright_axis=[0, 1, 0],
            retract=0.01,
            grasp_frame_to_eef=calibration.tolist(),
        ),
    )
    torch.testing.assert_close(calibrated, adjusted @ calibration)
    assert torch.equal(pose[0], torch.eye(4))
