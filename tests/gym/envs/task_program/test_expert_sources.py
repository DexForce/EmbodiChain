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

"""Registered expert admission, capabilities and exact call dispatch."""

from __future__ import annotations
from dataclasses import dataclass
import gc
import weakref
import torch
import hashlib
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock
import numpy as np
import pytest

from embodichain.data_pipeline.real2sim import (
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
)
from embodichain.data_pipeline.real2sim.normalization import Real2SimNormalizationCfg
from embodichain.data_pipeline.real2sim.bundle import write_real2sim_bundle
from embodichain.lab.task_program.integrations.expert_sources import (
    ExpertSourceRegistry,
    ExpertSourceSelection,
)
from embodichain.lab.task_program.integrations.real2sim import (
    Real2SimExpertSourceProviderFactory,
)
from embodichain.lab.task_program.integrations.configured import (
    _decode_runtime_services,
)
from embodichain.lab.sim.atomic_actions import MotionPolicy, RecoveryPolicy


def _declaration(tmp_path, positions=None):
    (tmp_path / "robot.urdf").write_text("<robot name='fixture'/>")
    q = np.zeros((12, 2)) if positions is None else positions
    source = Real2SimTrajectory(
        "fixture",
        "raw1",
        "pick0",
        "robot",
        "r1",
        ("arm", "finger"),
        ("rad", "rad"),
        q,
        np.arange(12) * 0.01,
        "s",
        "cal1",
        "phase1",
        "verified",
        "verified",
        (
            SourcePhase("approach", 0, 6, "free", ("joint_residual",)),
            SourcePhase("close", 6, 9, "contact"),
            SourcePhase("lift", 9, 12, "hold"),
        ),
    )
    path = write_real2sim_bundle(
        source,
        RobotJointLayout(
            "robot", "r1", source.joint_names, source.joint_units, q[0], ("arm",)
        ),
        Real2SimNormalizationCfg(control_dt=0.01),
        tmp_path / "data",
    )
    return dict(
        call_id="pick",
        call_index=0,
        provider_id="real2sim_qpos",
        revision="1",
        target_profile_id="fixture",
        target_asset_sha256=hashlib.sha256(
            (tmp_path / "robot.urdf").read_bytes()
        ).hexdigest(),
        bundle_path=str(path),
        bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_configured_source_uses_stable_ids_and_required_capabilities(tmp_path):
    value = _declaration(tmp_path)
    decoded = _decode_runtime_services({"expert_source_providers": [value]})
    assert decoded.expert_sources.selections[0].call_id == "pick"
    with pytest.raises(ValueError, match="capabilities"):
        _decode_runtime_services(
            {
                "expert_source_providers": [
                    {**value, "required_capabilities": ["replanning"]}
                ]
            }
        )
    with pytest.raises(ValueError, match="unregistered"):
        _decode_runtime_services(
            {
                "expert_source_providers": [
                    {**value, "provider_id": "arbitrary.module:Factory"}
                ]
            }
        )
    with pytest.raises(ValueError, match="unsupported"):
        _decode_runtime_services(
            {"expert_source_providers": [{**value, "callable": "arbitrary.module.fn"}]}
        )


def test_fixed_replay_policy_rejected_before_any_provider_creation(tmp_path):
    decoded = _decode_runtime_services(
        {"expert_source_providers": [_declaration(tmp_path)]}
    )
    with pytest.raises(ValueError, match="max_replans"):
        decoded.expert_sources.validate_policy(
            SimpleNamespace(
                recovery_policy=RecoveryPolicy(), motion_policy=MotionPolicy()
            )
        )
    decoded.expert_sources.validate_policy(
        SimpleNamespace(
            recovery_policy=RecoveryPolicy(max_replans=0, max_action_retries=0),
            motion_policy=MotionPolicy(),
        )
    )


def test_dispatch_is_exact_and_call_local_with_clock_preflight(tmp_path):
    decoded = _decode_runtime_services(
        {"expert_source_providers": [_declaration(tmp_path)]}
    )
    selector = decoded.expert_sources.create_runtime(
        scene_provider=Mock(),
        robot_profile=SimpleNamespace(profile_id="fixture"),
        control_dt=0.01,
    )
    engine = Mock()
    engine.robot.cfg.fpath = str(tmp_path / "robot.urdf")
    request = SimpleNamespace(
        analysis_call_index=0, call=SimpleNamespace(semantic_id="pick")
    )
    first = selector.create_nominal_plan_provider(request, engine=engine)
    second = selector.create_nominal_plan_provider(request, engine=engine)
    assert first.__self__ is not second.__self__
    assert (
        selector.create_nominal_plan_provider(
            SimpleNamespace(analysis_call_index=1), engine=engine
        )
        is None
    )
    with pytest.raises(ValueError, match="kind"):
        selector.create_nominal_plan_provider(
            SimpleNamespace(
                analysis_call_index=0, call=SimpleNamespace(semantic_id="place")
            ),
            engine=Mock(),
        )
    bad = decoded.expert_sources.create_runtime(
        scene_provider=Mock(),
        robot_profile=SimpleNamespace(profile_id="fixture"),
        control_dt=0.02,
    )
    with pytest.raises(ValueError, match="clock"):
        bad.create_nominal_plan_provider(request, engine=engine)


def test_duplicate_provider_or_call_selection_is_not_silently_replaced(tmp_path):
    value = _declaration(tmp_path)
    factory = Real2SimExpertSourceProviderFactory(
        value["bundle_path"],
        value["bundle_sha256"],
        "fixture",
        value["target_asset_sha256"],
    )
    selection = ExpertSourceSelection("pick", "real2sim_qpos", "1")
    with pytest.raises(ValueError, match="Duplicate expert"):
        ExpertSourceRegistry((factory, factory), (selection,))
    with pytest.raises(ValueError, match="occurrence"):
        ExpertSourceRegistry((factory,), (selection, selection))


def test_live_target_asset_mismatch_is_rejected(tmp_path):
    value = _declaration(tmp_path)
    factory = Real2SimExpertSourceProviderFactory(
        value["bundle_path"],
        value["bundle_sha256"],
        "fixture",
        value["target_asset_sha256"],
    )
    (tmp_path / "robot.urdf").write_text("<robot name='different_model'/>")
    engine = Mock()
    engine.robot.cfg.fpath = str(tmp_path / "robot.urdf")
    with pytest.raises(ValueError, match="asset digest"):
        factory.create(
            call_id="pick",
            scene_provider=Mock(),
            robot_profile=SimpleNamespace(profile_id="fixture"),
            control_dt=0.01,
            engine=engine,
        )


def _selector_and_engine(tmp_path, positions=None):
    decoded = _decode_runtime_services(
        {"expert_source_providers": [_declaration(tmp_path, positions)]}
    )
    selector = decoded.expert_sources.create_runtime(
        scene_provider=Mock(),
        robot_profile=SimpleNamespace(profile_id="fixture"),
        control_dt=0.01,
    )
    engine = Mock()
    engine.robot.cfg.fpath = str(tmp_path / "robot.urdf")
    engine.robot.joint_names = ["arm", "finger"]
    engine.robot.get_qpos_limits.return_value = torch.tensor(
        [[[-2.0, 2.0], [-2.0, 2.0]]]
    )
    engine.robot.get_qvel_limits.return_value = torch.full((1, 2), 2.0)
    request = SimpleNamespace(
        analysis_call_index=0, call=SimpleNamespace(semantic_id="pick")
    )
    return selector, engine, request


@pytest.mark.parametrize(
    "amplitude,rejected", [(0.019, False), (0.02, False), (0.03, True)]
)
def test_recorded_source_checks_each_interval_at_reversals(
    tmp_path, amplitude, rejected
):
    positions = np.zeros((12, 2))
    positions[2, 0] = amplitude
    selector, engine, selected = _selector_and_engine(tmp_path, positions)
    callback = selector.create_nominal_plan_provider(selected, engine=engine)
    provider = callback.__self__
    # Averaged velocities alone pass even though the .03 rad reversal needs
    # 3 rad/s over individual 10 ms intervals, above the 2 rad/s live limit.
    assert np.abs(provider.unit.velocities).max() <= 2.0
    request = SimpleNamespace(
        skill_id="pick_up",
        revision=0,
        recovery_policy=RecoveryPolicy(max_replans=0, max_action_retries=0),
        binding=SimpleNamespace(
            endpoints=(SimpleNamespace(target=SimpleNamespace(joint_ids=(0, 1))),)
        ),
    )
    context = SimpleNamespace(
        batch_size=1,
        robot=SimpleNamespace(qpos=torch.zeros((1, 2))),
        env_ids=torch.tensor([0]),
        require_control_dt=lambda: 0.01,
    )
    if rejected:
        with pytest.raises(ValueError, match="interval speeds.*velocity limits"):
            callback(request, context)
        engine.build_plan_from_recorded_trajectory.assert_not_called()
    else:
        assert (
            callback(request, context)
            is engine.build_plan_from_recorded_trajectory.return_value
        )
        engine.build_plan_from_recorded_trajectory.assert_called_once()


def test_completed_source_is_collectable_after_callback_is_released(tmp_path):
    selector, engine, request = _selector_and_engine(tmp_path)
    callback = selector.create_nominal_plan_provider(request, engine=engine)
    provider_ref = weakref.ref(callback.__self__)
    assert (
        callback.__self__.__hash__ is None
    )  # dataclass providers need not be hashable
    del callback
    gc.collect()
    assert provider_ref() is None
    # The selector remains live and can serve another call without old arrays.
    again = selector.create_nominal_plan_provider(request, engine=engine)
    assert again is not None


def test_cached_provider_reuse_is_rejected_even_after_its_callback_is_released(
    tmp_path, monkeypatch
):
    selector, engine, request = _selector_and_engine(tmp_path)
    callback = selector.create_nominal_plan_provider(request, engine=engine)
    cached = callback.__self__
    monkeypatch.setattr(
        type(selector.registry.factories[0]), "create", lambda self, **kwargs: cached
    )
    with pytest.raises(ValueError, match="reused"):
        selector.create_nominal_plan_provider(request, engine=engine)
    del callback
    gc.collect()
    with pytest.raises(ValueError, match="reused"):
        selector.create_nominal_plan_provider(request, engine=engine)


def test_non_weak_referenceable_provider_fails_with_explicit_lifecycle_error(
    tmp_path, monkeypatch
):
    selector, engine, request = _selector_and_engine(tmp_path)

    class SlottedProvider:
        __slots__ = ()

        def plan(self, request, context):
            raise AssertionError("Unsupported lifecycle must fail before planning")

    monkeypatch.setattr(
        type(selector.registry.factories[0]),
        "create",
        lambda self, **kwargs: SlottedProvider(),
    )
    with pytest.raises(TypeError, match="weak references"):
        selector.create_nominal_plan_provider(request, engine=engine)
