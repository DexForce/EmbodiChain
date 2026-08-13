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

from embodichain.learning.rl.motion_policy_evaluation.bridge import (
    create_motion_profile_evaluator,
    evaluate_motion_profile,
)
from embodichain.learning.rl.motion_policy_evaluation.profile import MotionProfile


def test_bridge_forwards_physics_backend_and_exact_control_steps(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    profile = MotionProfile(
        profile_id="example",
        checkpoint=checkpoint,
        policy_spec={"schema_version": 1},
    )
    options = []
    forwarded = []
    monkeypatch.setattr(
        "embodichain.learning.rl.motion_policy_evaluation.bridge.parse_policy_spec",
        lambda value: "parsed",
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.motion_policy_evaluation.bridge.resolve_policy_spec",
        lambda spec, resolver: "resolved",
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.motion_policy_evaluation.bridge.policy_spec_to_dict",
        lambda value: {"policy_id": "example"},
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.motion_policy_evaluation.bridge.scene_config_to_dict",
        lambda value: {"style": "standard"},
    )

    def run_policy(resolved, run_options, **kwargs):
        options.append(run_options)
        forwarded.append(kwargs)
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
        "embodichain.learning.rl.motion_policy_evaluation.bridge.run_motion_policy",
        run_policy,
    )

    input_provider = object()
    environment = object()
    result = evaluate_motion_profile(
        profile,
        control_steps=10,
        physics_backend="default",
        input_provider=input_provider,
        environment=environment,
    )

    assert options[0].control_steps == 10
    assert options[0].physics_backend == "default"
    assert forwarded == [
        {
            "environment": environment,
            "input_provider": input_provider,
        }
    ]
    assert result.episodes[0]["control_steps"] == 10
    assert result.episodes[0]["effective_duration"] == 0.2
    assert result.summary["metrics"]["tracking/error"] == 0.25


def test_create_profile_evaluator_forwards_the_environment(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    profile = MotionProfile(
        profile_id="example",
        checkpoint=checkpoint,
        policy_spec={"schema_version": 1},
    )
    evaluator = object()
    adapter = object()
    environment = object()
    calls = []
    monkeypatch.setattr(
        "embodichain.learning.rl.motion_policy_evaluation.bridge.parse_policy_spec",
        lambda value: "parsed",
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.motion_policy_evaluation.bridge.resolve_policy_spec",
        lambda spec, resolver: "resolved",
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.motion_policy_evaluation.bridge.create_motion_policy_evaluator",
        lambda resolved, options, **kwargs: calls.append((resolved, options, kwargs))
        or evaluator,
    )

    result = create_motion_profile_evaluator(
        profile,
        adapter=adapter,
        environment=environment,
    )

    assert result is evaluator
    assert calls == [
        (
            "resolved",
            None,
            {"adapter": adapter, "environment": environment},
        ),
    ]


def test_prebuilt_environment_requires_one_episode(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    profile = MotionProfile(
        profile_id="example",
        checkpoint=checkpoint,
        policy_spec={"schema_version": 1},
    )

    try:
        evaluate_motion_profile(profile, episodes=2, environment=object())
    except ValueError as error:
        assert str(error) == "A prebuilt environment supports one episode"
    else:
        raise AssertionError("Expected multi-episode validation")
