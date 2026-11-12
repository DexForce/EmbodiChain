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

"""Numerical volume sampling for file-free particle initialization."""

from __future__ import annotations

import numpy as np
import pytest
from embodichain.compute.geometry.particles import sample_volume_particles

pytestmark = pytest.mark.no_sim


def test_constant_revolved_profile_matches_cylinder():
    actual = sample_volume_particles(
        "revolved_volume",
        0.004,
        profile=[[-0.02, 0.015], [0.02, 0.015]],
        clearance=0.001,
    )
    expected = sample_volume_particles(
        "cylinder", 0.004, radius=0.015, height=0.04, clearance=0.001
    )
    np.testing.assert_array_equal(actual, expected)


def test_tapered_fill_volume_and_mass_converge_with_resolution():
    # A linear wall allows an independent integral of its inward offset.
    profile = [[-0.04, 0.02], [0.04, 0.03]]
    clearance = 0.002
    fill_level = 0.8
    slope = (0.03 - 0.02) / 0.08
    lower_z = -0.04 + clearance
    surface_z = -0.04 + fill_level * 0.08
    lower_radius = 0.02 + slope * clearance - clearance * np.sqrt(1 + slope**2)
    depth = surface_z - lower_z
    expected_volume = np.pi * (
        lower_radius**2 * depth
        + lower_radius * slope * depth**2
        + slope**2 * depth**3 / 3
    )
    counts = []
    for spacing in (0.002, 0.001):
        points = sample_volume_particles(
            "revolved_volume",
            spacing,
            profile=profile,
            clearance=clearance,
            fill_level=fill_level,
        )
        radius = np.linalg.norm(points[:, :2], axis=1)
        wall_distance = (0.02 + slope * (points[:, 2] + 0.04) - radius) / np.sqrt(
            1 + slope**2
        )
        assert np.all(wall_distance >= clearance - 1e-8)
        assert np.all(points[:, 2] > lower_z)
        assert np.all(points[:, 2] < surface_z)
        assert len(points) * spacing**3 == pytest.approx(expected_volume, rel=0.025)
        counts.append(len(points))
    assert counts[1] / counts[0] == pytest.approx(8, rel=0.025)


def test_lower_fill_level_keeps_lattice_origin_and_has_no_solid_free_surface_margin():
    kwargs = dict(size=(0.08, 0.08, 0.1), clearance=0.02)
    full = sample_volume_particles("box", 0.01, **kwargs)
    half = sample_volume_particles("box", 0.01, fill_level=0.5, **kwargs)
    np.testing.assert_array_equal(half, full[full[:, 2] < 0])
    assert half[:, 2].max() == pytest.approx(-0.005)


def test_multi_segment_profile_is_deterministic_and_can_narrow_again():
    profile = [[-0.03, 0.02], [0.0, 0.01], [0.03, 0.025]]
    kwargs = dict(profile=profile, clearance=0.001)
    points = sample_volume_particles("revolved_volume", 0.004, **kwargs)
    np.testing.assert_array_equal(
        points, sample_volume_particles("revolved_volume", 0.004, **kwargs)
    )
    assert np.all(
        np.linalg.norm(points[:, :2], axis=1)
        < np.interp(points[:, 2], np.array(profile)[:, 0], np.array(profile)[:, 1])
    )
    assert points.flags.c_contiguous


@pytest.mark.parametrize(
    "profile",
    [
        None,
        [],
        [[0, 1]],
        [[0, 1], [0, 2]],
        [[1, 1], [0, 2]],
        [[0, -1], [1, 1]],
        [[0, 0], [1, 0]],
        [[0, 1], [float("nan"), 1]],
        [[0, 1, 2], [1, 1, 2]],
    ],
)
def test_invalid_profiles_fail_before_grid_allocation(profile):
    with pytest.raises(ValueError, match="profile"):
        sample_volume_particles("revolved_volume", 0.001, profile=profile)


@pytest.mark.parametrize("fill_level", [0, -0.1, 1.01, float("inf"), float("nan")])
def test_invalid_fill_levels_are_rejected(fill_level):
    with pytest.raises(ValueError, match="fill_level"):
        sample_volume_particles("box", 0.01, fill_level=fill_level)


def test_profile_source_obeys_candidate_limit_before_allocation():
    with pytest.raises(ValueError, match="max_particles"):
        sample_volume_particles(
            "revolved_volume", 1e-9, profile=[[0, 1], [1, 1]], max_particles=100
        )


def test_profile_on_another_volume_is_not_ignored():
    with pytest.raises(ValueError, match="profile"):
        sample_volume_particles("box", 0.01, profile=[[0, 1], [1, 1]])
