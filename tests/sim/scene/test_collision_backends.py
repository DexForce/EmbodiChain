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

"""Optional collision backends fail explicitly when their binding is absent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodichain.lab.sim.scene import _collision_backends as backends


@pytest.mark.parametrize(
    "failure", [ModuleNotFoundError("newton"), OSError("missing native library")]
)
def test_missing_newton_has_actionable_dependency_error_and_no_fallback(
    monkeypatch, failure
):
    requested = []

    def missing(name):
        requested.append(name)
        raise failure

    monkeypatch.setattr(backends, "import_module", missing)
    with pytest.raises(ImportError, match="Newton") as caught:
        backends._require_newton()
    assert requested == ["newton"]
    assert caught.value.__cause__ is failure


def test_incomplete_newton_binding_is_unavailable(monkeypatch):
    monkeypatch.setattr(backends, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ImportError, match="collision pipeline"):
        backends._require_newton()


def test_installed_newton_preflight():
    pytest.importorskip(
        "newton", reason="Newton runtime is not installed; coverage is unqualified"
    )
    backends._require_newton()


def test_newton_distance_band_keeps_threshold_uncertainty_and_closes_owned_state():
    import numpy as np
    import trimesh
    from embodichain.lab.sim.scene._newton_queries import _ConvexCollection
    from embodichain.lab.sim.scene.feasibility.collision import (
        UnsupportedFeasibilityError,
    )

    left, right = _ConvexCollection(), _ConvexCollection()
    mesh = trimesh.creation.box((0.2, 0.2, 0.2))
    left.add_object("payload", mesh)
    pose = np.eye(4)
    pose[0, 3] = 0.202
    right.add_object("obstacle", mesh, transform=pose)
    interval = left.distance_interval(right, 0.002001)
    assert interval.lower < 0.002001 < interval.upper
    with pytest.raises(UnsupportedFeasibilityError, match="precision interval"):
        interval.below(0.002001)
    pose[0, 3] = 1.0
    right.set_transform("obstacle", pose)
    far = left.distance_interval(right, 0.002001)
    assert far.lower < 0.03 and far.upper == float("inf")
    assert far.below(0.002001) is False
    left.close()
    right.close()
    with pytest.raises(RuntimeError, match="closed"):
        left.distance_interval(right, 0.002001)
