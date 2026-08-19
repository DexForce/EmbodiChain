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

pytest.importorskip("dexsim.kit.motion_policy.evaluator")

from embodichain.learning.rl.policy_evaluation.bridge import (
    evaluate_motion_profile,
)
from embodichain.learning.rl.policy_evaluation.profile import MotionProfile


def test_bridge_forwards_physics_backend_and_exact_control_steps(
    monkeypatch,
):
    profile = MotionProfile(
        profile_id="example",
        policy_spec={"schema_version": 1},
    )
    options = []
    monkeypatch.setattr(
        "embodichain.learning.rl.policy_evaluation.bridge.parse_policy_spec",
        lambda value: "parsed",
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.policy_evaluation.bridge.resolve_policy_spec",
        lambda spec, resolver: "resolved",
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.policy_evaluation.bridge.policy_spec_to_dict",
        lambda value: {"policy_id": "example"},
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.policy_evaluation.bridge.scene_config_to_dict",
        lambda value: {"style": "standard"},
    )

    def run_policy(resolved, run_options):
        options.append(run_options)
        return SimpleNamespace(
            reason="control steps reached",
            simulation_time=0.2,
            simulation_steps=40,
            control_steps=10,
            physics_backend="default",
            requested_duration=None,
            effective_duration=0.2,
            metrics={"tracking/error": 0.25},
        )

    monkeypatch.setattr(
        "embodichain.learning.rl.policy_evaluation.bridge.run_motion_policy",
        run_policy,
    )

    result = evaluate_motion_profile(
        profile,
        control_steps=10,
        physics_backend="default",
    )

    assert options[0].control_steps == 10
    assert options[0].physics_backend == "default"
    assert result.episodes[0]["control_steps"] == 10
    assert result.episodes[0]["effective_duration"] == 0.2
    assert result.summary["metrics"]["tracking/error"] == 0.25
