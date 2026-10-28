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

"""Independent straight-line and rotation fixtures for bounded pose repair."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.compute.trajectory import refine_pose_trajectory


def _line(count: int = 81) -> np.ndarray:
    result = np.repeat(np.eye(4)[None], count, axis=0)
    result[:, 0, 3] = np.linspace(0, 0.1, count)
    return result


def test_linear_reference_and_exact_anchors_are_preserved_without_aliasing() -> None:
    source = _line()
    saved = source.copy()
    result = refine_pose_trajectory(source, anchors={0: source[0], 80: source[-1]})
    np.testing.assert_allclose(result.poses, saved, atol=1e-9)
    assert result.max_translation_m < 1e-9
    assert not result.poses.flags.writeable
    source[:] = 0
    np.testing.assert_allclose(result.poses, saved, atol=1e-9)


def test_confidence_weight_reduces_an_independent_sensor_outlier() -> None:
    oracle = _line()
    source = oracle.copy()
    source[40, 1, 3] += 0.008
    weights = np.ones(81)
    weights[40] = 0.01
    anchors = {0: oracle[0], 80: oracle[-1]}
    uniform = refine_pose_trajectory(source, anchors=anchors, smoothing_weight=5)
    weighted = refine_pose_trajectory(
        source, anchors=anchors, weights=weights, smoothing_weight=5
    )
    uniform_error = np.linalg.norm(uniform.poses[:, :3, 3] - oracle[:, :3, 3])
    weighted_error = np.linalg.norm(weighted.poses[:, :3, 3] - oracle[:, :3, 3])
    assert weighted_error < uniform_error / 10
    assert np.abs(weighted.poses[:, 1, 3]).max() < 0.0002
    np.testing.assert_array_equal(source[40, :3, 3], [0.05, 0.008, 0])


def test_rotation_uses_geodesic_correction_and_preserves_hard_event() -> None:
    oracle = _line()
    source = oracle.copy()
    source[:, :3, :3] = Rotation.from_euler("z", 4, degrees=True).as_matrix()
    result = refine_pose_trajectory(
        source, reference=oracle, anchors={0: oracle[0], 80: oracle[-1]}
    )
    np.testing.assert_allclose(result.poses, oracle, atol=1e-9)
    assert result.max_rotation_rad == pytest.approx(np.deg2rad(4))


def test_total_budget_includes_initial_warp_and_vector_norm() -> None:
    source = _line()
    reference = source.copy()
    reference[
        :, :3, 3
    ] += 0.015  # All component bounds fit; the Euclidean norm does not.
    with pytest.raises(ValueError, match="exact correction budget"):
        refine_pose_trajectory(
            source,
            reference=reference,
            anchors={0: reference[0], 80: reference[-1]},
            max_translation_m=0.02,
        )
    reference[:, 0, 3] += 0.02
    with pytest.raises(ValueError, match="anchor exceeds"):
        refine_pose_trajectory(source, reference=reference, anchors={0: reference[0]})


@pytest.mark.parametrize("bad", [np.nan, -1.0, True])
def test_invalid_budget_is_rejected(bad: float) -> None:
    source = _line()
    with pytest.raises(ValueError):
        refine_pose_trajectory(source, anchors={0: source[0]}, max_translation_m=bad)


def test_invalid_uncertainty_is_rejected() -> None:
    source = _line()
    weights = np.ones(81)
    weights[40] = np.nan
    with pytest.raises(ValueError, match="weights"):
        refine_pose_trajectory(source, anchors={0: source[0]}, weights=weights)


def test_iteration_budget_cannot_return_an_unconverged_path() -> None:
    source = _line()
    source[30:50, 1, 3] = 0.005
    with pytest.raises(ValueError, match="did not converge"):
        refine_pose_trajectory(
            source, anchors={0: source[0], 80: source[-1]}, max_iterations=1
        )
