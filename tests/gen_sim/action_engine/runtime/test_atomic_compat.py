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


def test_exact_target_transport_only_disables_rotation_when_requested(
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
    disabled_request = SimpleNamespace(
        skill_options=ExactTargetMoveHeldObjectOptions(
            allow_automatic_transport_rotation=False,
        )
    )
    enabled_request = SimpleNamespace(
        skill_options=ExactTargetMoveHeldObjectOptions(),
    )

    assert action._plan(disabled_request, object()) is result
    assert not applied
    assert action._plan(enabled_request, object()) is result
    assert applied == [True]


@pytest.mark.parametrize(
    ("yaw_samples", "expected"),
    [(1, True), (8, False)],
)
def test_semantic_transport_config_scopes_rotation_override(
    yaw_samples: int,
    expected: bool,
) -> None:
    adapter = AtomicActionAdapter.__new__(AtomicActionAdapter)
    action = SimpleNamespace(cfg={"upright_yaw_samples": yaw_samples})
    capability = SimpleNamespace(
        config_type=MoveHeldObjectOptions,
        target_materializer="semantic_held_object",
    )

    options = adapter._build_single_arm_config(action, capability)

    assert isinstance(options, ExactTargetMoveHeldObjectOptions)
    assert options.allow_automatic_transport_rotation is expected
