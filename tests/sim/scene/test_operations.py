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
import numpy as np
import pytest
from embodichain.lab.sim.scene.query import SceneView
from embodichain.lab.sim.scene.operations import SceneChangeSet, apply_changes
from ._runtime_ports import Sim


def test_changes_clear_dynamics_publish_and_request_host_cache_refresh():
    sim = Sim()
    pose = np.eye(4)
    pose[0, 3] = 0.7
    changes = SceneChangeSet({"cube": pose}, ("cube",))
    pose[0, 3] = 9
    result = apply_changes(SceneView(sim), changes)
    assert sim.rigid["cube"].value[0, 0].item() == pytest.approx(0.7)
    assert not sim.rigid["cube"].value[:, 7:].any()
    assert sim.steps == 0 and sim.syncs == 1
    assert "planning" in result.refresh_required


def test_unknown_uid_preflight_does_not_modify_valid_earlier_uid():
    sim = Sim()
    changes = SceneChangeSet(
        {"cube": np.eye(4), "missing": np.eye(4)}, ("cube", "missing")
    )
    with pytest.raises(ValueError, match="existing rigid object"):
        apply_changes(SceneView(sim), changes)
    assert sim.rigid["cube"].writes == 0


def test_permission_and_nonrigid_transforms_fail_before_application():
    with pytest.raises(ValueError, match="permission"):
        SceneChangeSet({"cube": np.eye(4)}, ())
    scaled = np.eye(4)
    scaled[0, 0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        SceneChangeSet({"cube": scaled}, ("cube",))


def test_native_failure_reports_partial_application():
    sim = Sim()

    def fail(*args, **kwargs):
        raise OSError("native")

    sim.rigid["other"].set_local_pose = fail
    with pytest.raises(RuntimeError, match="earlier writes"):
        apply_changes(
            SceneView(sim),
            SceneChangeSet({"cube": np.eye(4), "other": np.eye(4)}, ("cube", "other")),
        )
    assert sim.rigid["cube"].value[0, 0] == 0 and sim.steps == 0


def test_backend_overflow_preflight_preserves_every_entity():
    sim = Sim()
    oversized = np.eye(4)
    oversized[0, 3] = 1e100  # Finite Python float, outside the backend float32 range.
    changes = SceneChangeSet({"cube": np.eye(4), "other": oversized}, ("cube", "other"))
    with pytest.raises(ValueError, match="backend"):
        apply_changes(SceneView(sim), changes)
    assert all(body.writes == 0 for body in sim.rigid.values())
    assert sim.syncs == 0
