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
import torch
from embodichain.lab.sim.scene.query import SceneView
from embodichain.lab.sim.scene.operations import (
    SceneChangeSet,
    apply_changes,
    apply_changes_batch,
    apply_group_transform,
)
from ._runtime_ports import Sim, Visual


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


def test_distinct_batch_changes_preflight_every_row_and_preserve_neighbors():
    from embodichain.lab.sim.scene.operations import apply_changes_batch

    sim = Sim(4)
    view = SceneView(sim, env_ids=(3, 1))
    a, b = torch.eye(4), torch.eye(4)
    a[0, 3], b[0, 3] = 0.7, 0.9
    changes = {
        3: SceneChangeSet({"cube": a.tolist()}, ("cube",)),
        1: SceneChangeSet({"missing": b.tolist()}, ("missing",)),
    }
    with pytest.raises(ValueError, match="existing rigid"):
        apply_changes_batch(view, changes)
    assert sim.rigid["cube"].writes == 0
    before = sim.rigid["cube"].value[[0, 2]].clone()
    changes[1] = SceneChangeSet({"cube": b.tolist()}, ("cube",))
    result = apply_changes_batch(view, changes)
    assert result.changed_uids == ("cube",)
    torch.testing.assert_close(
        sim.rigid["cube"].value[[3, 1], 0], torch.tensor([0.7, 0.9])
    )
    torch.testing.assert_close(sim.rigid["cube"].value[[0, 2]], before, atol=0, rtol=0)
    assert sim.steps == 0 and sim.syncs == 1


def _static_group():
    sim = Sim(4)
    body = sim.rigid["cube"]
    body.body_type = "static"
    body.value[:, 7:] = 0
    body.value[:, 0] = torch.arange(4) + 1
    visual = sim.visuals["scan"] = Visual(4)
    visual.value[:, :3, 3] = body.value[:, :3] + torch.tensor([0, 2, 0.2])
    visual.value[:, :3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return sim, SceneView(sim, env_ids=(3, 1), entity_uids=("scan", "cube"))


def test_group_rotation_translation_preserves_relative_pose_and_neighbors():
    sim, view = _static_group()
    before = {uid: entity.get_local_pose().clone() for _, uid, entity in view.members()}
    delta = torch.tensor([[0.0, -1, 0, 10], [1, 0, 0, -2], [0, 0, 1, 1], [0, 0, 0, 1]])
    result = apply_group_transform(view, delta, movable_uids=("scan", "cube"))
    torch.testing.assert_close(
        view.pose("cube")[:, :3, 3], torch.tensor([[10.0, 2, 1.1], [10, 0, 1.1]])
    )
    torch.testing.assert_close(
        view.pose("scan")[:, :3, 3], torch.tensor([[8.0, 2, 1.3], [8, 0, 1.3]])
    )
    torch.testing.assert_close(
        view.pose("scan")[:, :3, :3],
        torch.diag(torch.tensor([-1.0, -1, 1])).expand(2, -1, -1),
    )
    relative = torch.linalg.inv(before["cube"][[3, 1]]) @ before["scan"][[3, 1]]
    torch.testing.assert_close(
        torch.linalg.inv(view.pose("cube")) @ view.pose("scan"), relative
    )
    for _, uid, entity in view.members():
        torch.testing.assert_close(
            entity.get_local_pose()[[0, 2]], before[uid][[0, 2]], atol=0, rtol=0
        )
    assert result.changed_uids == ("scan", "cube")
    assert sim.rigid["cube"].body_type == "static"
    assert sim.rigid["other"].writes == 0 and sim.steps == 0 and sim.syncs == 1


@pytest.mark.parametrize(
    "defect",
    ["dynamic", "kinematic", "articulation", "permission", "reflection", "last_row"],
)
def test_group_rejection_precedes_every_member_write(defect):
    sim, view = _static_group()
    delta = np.eye(4)
    permissions = ("scan", "cube")
    if defect in ("dynamic", "kinematic"):
        sim.rigid["cube"].body_type = defect
        view = SceneView(sim, env_ids=(3, 1), entity_uids=("scan", "cube"))
    elif defect == "articulation":
        view = SceneView(sim, entity_uids=("scan", "drawer"))
        permissions = ("scan", "drawer")
    elif defect == "permission":
        permissions = ("cube",)
    elif defect == "reflection":
        delta[0, 0] = -1
    else:
        sim.visuals["scan"].value[1, 0, 1] *= 2
    with pytest.raises(ValueError):
        apply_group_transform(view, delta, movable_uids=permissions)
    assert all(body.writes == 0 for body in sim.rigid.values())
    assert sim.visuals["scan"].writes == 0 and sim.syncs == 0 and sim.steps == 0


def test_batch_changes_keep_mixed_group_scope_and_preflight_later_rows():
    sim, view = _static_group()
    changes = {
        3: SceneChangeSet({"scan": np.eye(4)}, ("scan",)),
        1: SceneChangeSet({"other": np.eye(4)}, ("other",)),
    }
    with pytest.raises(ValueError, match="existing rigid"):
        apply_changes_batch(view, changes)
    assert sim.visuals["scan"].writes == sim.rigid["other"].writes == sim.syncs == 0
    changes[1] = SceneChangeSet({"scan": np.eye(4)}, ("scan",))
    before = sim.visuals["scan"].value[[0, 2]].clone()
    apply_changes_batch(view, changes)
    torch.testing.assert_close(
        sim.visuals["scan"].value[[3, 1]], torch.eye(4).expand(2, -1, -1)
    )
    torch.testing.assert_close(
        sim.visuals["scan"].value[[0, 2]], before, atol=0, rtol=0
    )
    assert sim.syncs == 1 and sim.steps == 0
