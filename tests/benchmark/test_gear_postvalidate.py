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
"""Independent regressions for physical-property diagnostic acceptance."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from scripts.benchmark.assembly.gear_postvalidate import (
    _compare_models,
    _global_options,
)


def _model(reverse: bool = False):
    parts = [
        '<geom type="mesh" mesh="tetra" pos="-.02 0 0" friction=".6 .01 .001"/>',
        '<geom type="mesh" mesh="tetra" pos=".02 0 0" friction=".8 .01 .001"/>',
    ]
    if reverse:
        parts.reverse()
    bodies = "".join(
        f'<body name="{name}" pos="{position} 0 0"><joint type="hinge"/>'
        '<inertial mass="2" pos="0 0 0" diaginertia=".01 .02 .025"/>'
        + "".join(parts)
        + "</body>"
        for name, position in (("gear_a", -0.05), ("gear_b", 0.05))
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="tetra" vertex="0 0 0 .01 0 0 0 .01 0 0 0 .01" '
        'face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset><worldbody>'
        + bodies
        + "</worldbody></mujoco>"
    )


@pytest.mark.parametrize("change", ["principal_frame", "com", "condim"])
def test_postvalidation_detects_physics_changes_with_unchanged_eigenvalues(change):
    source, actual = _model(), _model()
    if change == "principal_frame":
        # Same three principal moments; only the physical inertia orientation changes.
        actual.body_iquat[actual.body("gear_a").id] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    elif change == "com":
        actual.body_ipos[actual.body("gear_a").id, 0] = 100e-6
    else:
        actual.geom_condim[0] = 4
    np.testing.assert_array_equal(source.body_inertia, actual.body_inertia)
    assert not _compare_models(source, actual)["passed"]


def test_geometry_matching_checks_contact_ownership_after_importer_reordering():
    source, actual = _model(), _model(reverse=True)
    measured = _compare_models(source, actual)
    assert measured["passed"]
    assert measured["geometry"]["pairs"][0]["actual_geom"] == 1
    # The overall friction multiset remains unchanged; attaching each value to
    # the wrong collider must fail rather than passing a sorted comparison.
    actual.geom_friction[:2] = actual.geom_friction[:2][::-1].copy()
    assert not _compare_models(source, actual)["contacts"]["geom_friction"]["passed"]


def test_arena_mask_bit_renumbering_preserves_collision_eligibility():
    source, actual = _model(), _model()
    actual.geom_contype[:] = 8
    actual.geom_conaffinity[:] = 8
    assert _compare_models(source, actual)["passed"]
    actual.geom_conaffinity[:] = 0
    assert not _compare_models(source, actual)["collision_filtering"]["passed"]


def test_distinct_global_contact_profile_is_disclosed_separately():
    source, actual = _model(), _model()
    actual.opt.disableflags = (
        mujoco.mjtDisableBit.mjDSBL_MULTICCD | mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    )
    options = _global_options(source, actual, {"tolerance": [1e-6]})
    assert not options["source_global_options_equivalent"]
    assert options["source_multiccd_enabled"]
    assert not options["actual_host_multiccd_enabled"]
    assert set(options["changed_disable_bits"]) == {
        "mjDSBL_MULTICCD",
        "mjDSBL_MIDPHASE",
    }
    assert "tolerance" in options["captured_live_differences"]
    # Representation checks must not relabel this distinct profile as an
    # identical-settings source replication, or discard their own evidence.
    assert _compare_models(source, actual)["passed"]
