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


def _declaration(tmp_path):
    (tmp_path / "robot.urdf").write_text("<robot name='fixture'/>")
    q = np.zeros((12, 2))
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
