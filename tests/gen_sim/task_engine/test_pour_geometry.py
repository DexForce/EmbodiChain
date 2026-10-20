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

"""CPU-only native-frame numeric contracts, not poured or physical success."""

from __future__ import annotations

from itertools import product
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.task_engine._task_program.pour_geometry import (
    pour_frames,
    staging_pose,
)

GRAVITY = np.array([0.0, 0.0, -9.81])


def _box(size):
    return np.array(list(product(*[(-extent / 2, extent / 2) for extent in size])))


def _pose(xyz=(0.1, -0.2, 0.8), angles=(0, 0, 0)):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("XYZ", angles).as_matrix()
    pose[:3, 3] = xyz
    return pose


def _case(angles=(0, 0, 0)):
    source, receiver = _box((0.10, 0.06, 0.20)), _box((0.25, 0.20, 0.05))
    entered, received = _pose(angles=angles), _pose((0.35, 0.2, 0.75), (0, 0, 0.2))
    stage = staging_pose(
        source, receiver, entered, received, clearance_m=0.015, gravity_world=GRAVITY
    )
    up = stage[:3, :3].T @ [0, 0, 1]
    up /= np.linalg.norm(up)
    jaw = np.cross(up, np.eye(3)[np.argmin(np.abs(up))])
    return source, receiver, entered, received, stage, jaw


def _frames(case, *, jaw=None, angle=-math.pi / 3, count=140):
    source, receiver, entered, received, stage, default_jaw = case
    return pour_frames(
        source,
        receiver,
        stage,
        received,
        default_jaw if jaw is None else jaw,
        angle,
        entered,
        sample_count=count,
        gravity_world=GRAVITY,
    )


@pytest.mark.parametrize(
    "angles", [(0, 0, 0), (0.3, -0.7, 1.1), (1.2, 0.6, -2.3), (0.7, 1.3, 0.8)]
)
def test_preserves_observed_R_with_explicit_staging_restore_and_return(angles):
    case = _case(angles)
    result = _frames(case)
    assert np.array_equal(case[4][:3, :3], case[2][:3, :3])
    assert result["target_roles"] == (
        "staging",
        "peak",
        "restore",
        "return_transport",
        "return_place",
    )
    for index in (0, 2):
        assert np.array_equal(result["object_targets"][index], case[4])
    for index in (3, 4):
        assert np.array_equal(result["object_targets"][index], case[2])
    assert result["physical_evidence"] is False


def test_recorded_native_float32_R_passes_without_SVD_or_rotation_rewriting():
    # NativeSupportObservation.object_pose in r4_final_e1_1181_bottle_s0.
    native = np.array(
        [
            [
                0.005061089992523193,
                -0.9999504685401917,
                -0.008564522489905357,
                0.18017150461673737,
            ],
            [
                0.9999846816062927,
                0.005080044269561768,
                -0.0021989948581904173,
                -0.0802346020936966,
            ],
            [
                0.0022423940245062113,
                -0.008553261868655682,
                0.9999608993530273,
                0.6757163405418396,
            ],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    rotation = native[:3, :3].astype(np.float64)
    assert not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
    assert abs(np.linalg.det(rotation) - 1) > 1e-8
    source, receiver, _, received, _, _ = _case()
    before = native.copy()
    stage = staging_pose(
        source, receiver, native, received, clearance_m=0.015, gravity_world=GRAVITY
    )
    assert np.array_equal(stage[:3, :3], native[:3, :3])
    up = stage[:3, :3].T @ [0, 0, 1]
    up /= np.linalg.norm(up)
    result = pour_frames(
        source,
        receiver,
        stage,
        received,
        np.cross(up, [1, 0, 0]),
        math.pi / 3,
        native,
        sample_count=140,
        gravity_world=GRAVITY,
    )
    assert np.array_equal(result["object_targets"][2, :3, :3], native[:3, :3])
    assert np.array_equal(result["object_targets"][3], native)
    assert np.array_equal(native, before)
    with pytest.raises(ValueError, match="degenerate"):
        pour_frames(
            source,
            receiver,
            stage,
            received,
            up,
            math.pi / 3,
            native,
            sample_count=140,
            gravity_world=GRAVITY,
        )


@pytest.mark.parametrize(
    "damage", ["shear", "scale", "reflection", "nan", "complex", "batch"]
)
def test_nonrigid_or_invalid_pose_still_rejects(damage):
    source, receiver, entry, received, _, _ = _case()
    if damage == "shear":
        entry[0, 1] = 0.001
    elif damage == "scale":
        entry[0, 0] += 0.001
    elif damage == "reflection":
        entry[:3, 0] *= -1
    elif damage == "nan":
        entry[0, 3] = math.nan
    elif damage == "complex":
        entry = entry.astype(complex)
        entry[0, 3] += 0.1j
    else:
        entry = entry[None]
    with pytest.raises(ValueError, match="SE3"):
        staging_pose(
            source, receiver, entry, received, clearance_m=0.015, gravity_world=GRAVITY
        )


@pytest.mark.parametrize("count", [3, 12, 139, 140])
def test_fixed_pivot_inverse_restore_and_conservative_source_only_sphere(count):
    case = _case((0.3, -0.7, 1.1))
    result = _frames(case, count=count)
    arc = result["object_arc"]
    assert len(arc) == count
    pivot = arc[:, :3, :3] @ result["pivot_object"] + arc[:, :3, 3]
    np.testing.assert_allclose(
        pivot, np.broadcast_to(result["pivot_world"], pivot.shape), atol=1e-12
    )
    inverse = Rotation.from_rotvec(
        -result["effective_angle_rad"] * result["tilt_axis_object"]
    ).as_matrix()
    np.testing.assert_allclose(
        arc[result["peak_index"], :3, :3] @ inverse, case[4][:3, :3], atol=1e-12
    )
    assert np.array_equal(arc[0], case[4]) and np.array_equal(arc[-1], case[4])
    world = (arc[:, None, :3, :3] @ case[0][None, :, :, None])[..., 0] + arc[
        :, None, :3, 3
    ]
    receiver_world = case[1] @ case[3][:3, :3].T + case[3][:3, 3]
    assert world[..., 2].min() >= receiver_world[:, 2].max() + 0.015 - 1e-12


def test_same_candidate_and_actual_kernel_inputs_are_bitwise_identical():
    case = _case((0.3, -0.7, 1.1))
    before = [value.copy() for value in case]
    rng = np.random.get_state()
    candidate, actual = _frames(case), _frames(case)
    for key in candidate:
        if isinstance(candidate[key], np.ndarray):
            assert np.array_equal(candidate[key], actual[key])
        else:
            assert candidate[key] == actual[key]
    assert all(np.array_equal(a, b) for a, b in zip(before, case, strict=True))
    after = np.random.get_state()
    assert (
        rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]
    )


def test_candidate_G_and_measured_world_pad_conversions_use_one_kernel():
    source, receiver, entered, received, stage, jaw = case = _case((0.3, -0.7, 1.1))
    g = _pose((0.02, -0.01, 0.04), (0.2, -0.3, 0.7))
    nominal_tcp_line = g[:3, :3].T @ jaw
    actual_world_line = stage[:3, :3] @ jaw
    candidate = _frames(case, jaw=g[:3, :3] @ nominal_tcp_line)
    actual = _frames(case, jaw=stage[:3, :3].T @ actual_world_line)
    np.testing.assert_allclose(
        candidate["object_arc"], actual["object_arc"], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        candidate["object_targets"] @ g,
        actual["object_targets"] @ g,
        atol=1e-12,
        rtol=0,
    )


def test_zero_receiver_bearing_is_a_legal_stable_ordered_jaw_plus_angle_tie():
    case = _case()
    positive, negative = _frames(case, angle=math.pi / 3), _frames(
        case, angle=-math.pi / 3
    )
    assert positive["receiver_bearing_tie"] and negative["receiver_bearing_tie"]
    assert positive["tilt_direction_policy"] == "ordered_jaw_axis_plus_angle"
    assert np.array_equal(positive["object_arc"], negative["object_arc"])
    for delta in (-1e-6, 1e-6):
        shifted = list(case)
        shifted[3] = shifted[3].copy()
        shifted[3][1, 3] += delta
        result = _frames(shifted)
        assert result["receiver_bearing_tie"]
        assert np.array_equal(result["object_arc"], positive["object_arc"])


@pytest.mark.parametrize(
    "offset", [-0.01, -0.003, -11e-6, -1e-6, 0.0, 1e-6, 11e-6, 0.003, 0.01]
)
@pytest.mark.parametrize("jaw_sign", [-1.0, 1.0])
def test_direction_free_arc_does_not_flip_with_receiver_tracking_error(
    offset, jaw_sign
):
    case = list(_case())
    jaw = np.array([jaw_sign, 0.0, 0.0])
    expected = _frames(case, jaw=jaw)
    case[3] = case[3].copy()
    case[3][1, 3] += offset
    result = _frames(case, jaw=jaw)
    assert result["receiver_bearing_tie"] == (abs(offset) <= 1e-5)
    assert result["receiver_bearing_m"] == pytest.approx(-jaw_sign * offset)
    np.testing.assert_array_equal(result["object_arc"], expected["object_arc"])
    np.testing.assert_array_equal(result["object_targets"], expected["object_targets"])
    assert result["tilt_direction_policy"] == "ordered_jaw_axis_plus_angle"


@pytest.mark.parametrize(
    "jaw",
    [
        np.zeros(3),
        np.array([0.0, 0, 1]),
        np.array([math.nan, 0, 0]),
        np.ones((2, 3)),
        np.array([1j, 0, 0]),
    ],
)
def test_wrong_or_gravity_parallel_jaw_rejects_without_axis_substitution(jaw):
    with pytest.raises(ValueError):
        _frames(_case(), jaw=jaw)


def test_two_mirrored_rolls_can_be_qualified_independently():
    case = _case()
    jaw_tcp = np.array([1.0, 0, 1])
    first = _pose((0, 0, 0), (0, -math.pi / 4, 0))
    mirrored = first @ _pose((0, 0, 0), (0, 0, math.pi))
    outcomes = []
    for g in (first, mirrored):
        try:
            result = _frames(case, jaw=g[:3, :3] @ jaw_tcp)
            outcomes.append(result["object_arc"].shape == (140, 4, 4))
        except ValueError as error:
            assert "degenerate" in str(error)
            outcomes.append(False)
    assert outcomes == [False, True]


def test_world_yaw_translation_and_source_receiver_origin_rebasing():
    source, receiver, entered, received, stage, jaw = case = _case((0.3, -0.7, 1.1))
    original = _frames(case)
    world = _pose((0.7, -0.4, 0.2), (0, 0, 0.8))
    moved = staging_pose(
        source,
        receiver,
        world @ entered,
        world @ received,
        clearance_m=0.015,
        gravity_world=GRAVITY,
    )
    np.testing.assert_allclose(moved, world @ stage, atol=1e-12)
    transformed = pour_frames(
        source,
        receiver,
        moved,
        world @ received,
        jaw,
        math.pi / 3,
        world @ entered,
        sample_count=140,
        gravity_world=GRAVITY,
    )
    np.testing.assert_allclose(
        transformed["object_arc"], world @ original["object_arc"], atol=1e-12
    )
    shift = _pose((0.12, -0.07, 0.03))
    rebased = staging_pose(
        source - shift[:3, 3],
        receiver,
        entered @ shift,
        received,
        clearance_m=0.015,
        gravity_world=GRAVITY,
    )
    np.testing.assert_allclose(rebased, stage @ shift, atol=1e-12)
    frames = pour_frames(
        source - shift[:3, 3],
        receiver,
        rebased,
        received,
        jaw,
        math.pi / 3,
        entered @ shift,
        sample_count=140,
        gravity_world=GRAVITY,
    )
    g = _pose((0.02, 0.01, -0.03), (0.1, 0.3, -0.2))
    np.testing.assert_allclose(
        frames["object_arc"] @ (np.linalg.inv(shift) @ g),
        original["object_arc"] @ g,
        atol=1e-12,
    )
    moved_receiver = staging_pose(
        source,
        receiver - shift[:3, 3],
        entered,
        received @ shift,
        clearance_m=0.015,
        gravity_world=GRAVITY,
    )
    np.testing.assert_allclose(moved_receiver, stage, atol=1e-12)


def test_longest_extent_near_tie_does_not_choose_an_upright_axis():
    entry, receive, receiver = (
        _pose(angles=(0.4, 0.2, -0.6)),
        _pose((0.4, 0.1, 0.7)),
        _box((0.3,) * 3),
    )
    for extents in ((0.200001, 0.2, 0.2), (0.2, 0.2, 0.200001)):
        result = staging_pose(
            _box(extents),
            receiver,
            entry,
            receive,
            clearance_m=0.015,
            gravity_world=GRAVITY,
        )
        assert np.array_equal(result[:3, :3], entry[:3, :3])


@pytest.mark.parametrize(
    "gravity",
    [
        (1, 0, -9.81),
        (0, 1, -9.81),
        (1e-12, 0, -9.81),
        (0, 0, 0),
        (0, 0, 9.81),
        (0, 0, math.inf),
        (0, 0, -math.inf),
        (0, 0, math.nan),
    ],
)
def test_both_functions_require_explicit_finite_nonzero_negative_z_gravity(gravity):
    source, receiver, entered, received, stage, jaw = _case()
    with pytest.raises(ValueError, match="negative-Z World gravity"):
        staging_pose(
            source,
            receiver,
            entered,
            received,
            clearance_m=0.015,
            gravity_world=np.array(gravity),
        )
    with pytest.raises(ValueError, match="negative-Z World gravity"):
        pour_frames(
            source,
            receiver,
            stage,
            received,
            jaw,
            math.pi / 3,
            entered,
            sample_count=140,
            gravity_world=np.array(gravity),
        )


@pytest.mark.parametrize("magnitude", [1e-200, 0.01, 1, 9.81, 1000])
def test_gravity_domain_adds_no_magnitude_threshold(magnitude):
    source, receiver, entered, received, _, _ = _case()
    staging_pose(
        source,
        receiver,
        entered,
        received,
        clearance_m=0.015,
        gravity_world=np.array([0, 0, -magnitude]),
    )


@pytest.mark.parametrize(
    "angle,count",
    [(0, 140), (math.nan, 140), (True, 140), (math.pi / 3, 2), (math.pi / 3, True)],
)
def test_invalid_arc_parameters_are_rejected(angle, count):
    with pytest.raises(ValueError):
        _frames(_case(), angle=angle, count=count)
