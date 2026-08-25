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

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.atomic_compat import (
    ExactTargetMoveHeldObject,
    ExactTargetMoveHeldObjectOptions,
)
from embodichain.lab.sim.atomic_actions import MoveHeldObject, MoveHeldObjectOptions


def test_grounded_target_transport_never_applies_an_implicit_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied = []
    result = object()

    def fake_apply(self, move_eef_xpos, end_arm_xpos) -> None:
        del self, move_eef_xpos, end_arm_xpos
        applied.append(True)

    def fake_plan(self, request, context):
        del request, context
        self._apply_automatic_transport_rotation(torch.eye(4), torch.eye(4))
        return result

    monkeypatch.setattr(
        MoveHeldObject,
        "_apply_automatic_transport_rotation",
        fake_apply,
    )
    monkeypatch.setattr(MoveHeldObject, "_plan", fake_plan)
    action = ExactTargetMoveHeldObject()
    assert type(action).__dict__["binding_contract"] is MoveHeldObject.binding_contract
    request = SimpleNamespace(skill_options=ExactTargetMoveHeldObjectOptions())

    assert action._plan(request, object()) is result
    assert not applied


def test_semantic_transport_config_has_no_task_facing_rotation_switch() -> None:
    adapter = AtomicActionAdapter.__new__(AtomicActionAdapter)
    action = SimpleNamespace(cfg={})
    capability = SimpleNamespace(
        config_type=MoveHeldObjectOptions,
        target_materializer="semantic_held_object",
    )

    options = adapter._build_single_arm_config(action, capability)

    assert isinstance(options, ExactTargetMoveHeldObjectOptions)
    assert not hasattr(options, "allow_automatic_transport_rotation")
