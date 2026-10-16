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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    actions,
    motion,
    place_branch,
    place_recovery,
)
from embodichain.lab.sim.atomic_actions.primitives.place import Place


@pytest.mark.parametrize("original_qualified", [True, False])
@pytest.mark.parametrize("branch_success", [True, False])
def test_owned_place_has_one_dense_attempt_and_no_legacy_recovery(
    monkeypatch, original_qualified, branch_success
):
    action = actions.GenSimPlace()
    action._planning_services = SimpleNamespace(motion_generator=SimpleNamespace())
    request, context = object(), object()
    original = SimpleNamespace(name="original")
    candidate = SimpleNamespace(
        name="bounded", plan_success=torch.tensor([branch_success])
    )
    monkeypatch.setattr(
        actions.GenSimStateHandoff,
        "from_request",
        lambda *a, **kw: SimpleNamespace(resource="left"),
    )
    release = Mock(side_effect=lambda plan, resource: plan)
    monkeypatch.setattr(actions, "require_release_effect", release)
    monkeypatch.setattr(action, "_with_contact_monitor_boundary", lambda plan: plan)
    monkeypatch.setattr(place_branch, "owns", lambda *a: True)
    inspect = Mock(return_value={"valid": original_qualified})
    monkeypatch.setattr(place_branch, "inspect_emitted", inspect)
    bounded = Mock(return_value=candidate)
    monkeypatch.setattr(place_branch, "plan", bounded)
    legacy = Mock(side_effect=AssertionError("Owned branch entered legacy recovery"))
    monkeypatch.setattr(place_recovery, "recover", legacy)
    monkeypatch.setattr(action, "_retry_place_routes", legacy)
    builds = []

    def build(_action, selected, observed):
        assert _action is action and selected is request and observed is context
        assert motion._PLACE_IK_RECOVERY.get().defer_retries is True
        builds.append(1)
        return original

    monkeypatch.setattr(Place, "_plan", build)
    result = action._plan(request, context)
    assert builds == [1]
    inspect.assert_called_once_with(action, request, context, original)
    legacy.assert_not_called()
    assert motion._PLACE_IK_RECOVERY.get() is None
    expected = original if original_qualified else candidate
    assert result is expected
    release.assert_called_once_with(expected, "left")
    if original_qualified:
        bounded.assert_not_called()
    else:
        bounded.assert_called_once()
        assert bounded.call_args.args[:3] == (action, request, context)


def test_unowned_place_retains_existing_success_path(monkeypatch):
    action = actions.GenSimPlace()
    action._planning_services = SimpleNamespace(motion_generator=SimpleNamespace())
    context = SimpleNamespace(batch_size=1, env_ids=torch.tensor([0]))
    request = SimpleNamespace(
        motion_policy=SimpleNamespace(
            strategy="ik_interp", dynamic_collision_mode="off"
        ),
        skill_options=SimpleNamespace(preserve_current_object_orientation=False),
    )
    original = SimpleNamespace(plan_success=torch.tensor([True]))
    monkeypatch.setattr(
        actions.GenSimStateHandoff,
        "from_request",
        lambda *a, **kw: SimpleNamespace(resource="left"),
    )
    monkeypatch.setattr(actions, "require_release_effect", lambda plan, _: plan)
    monkeypatch.setattr(action, "_with_contact_monitor_boundary", lambda plan: plan)
    monkeypatch.setattr(place_branch, "owns", lambda *a: False)
    unused = Mock(side_effect=AssertionError("Unowned Place entered bounded route"))
    monkeypatch.setattr(place_branch, "inspect_emitted", unused)
    monkeypatch.setattr(place_branch, "plan", unused)
    calls = []

    def build(_action, selected, observed):
        assert selected is request and observed is context
        assert motion._PLACE_IK_RECOVERY.get().defer_retries is False
        calls.append(1)
        return original

    monkeypatch.setattr(Place, "_plan", build)
    assert action._plan(request, context) is original
    assert calls == [1]
    unused.assert_not_called()
    assert motion._PLACE_IK_RECOVERY.get() is None
