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

"""Explicit hand calibration from closed declarations to isolated runtime data."""

from __future__ import annotations

import builtins
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.lab.sim.atomic_actions import AtomicActionEngine
from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryProvider
from embodichain.lab.task_program.integrations.configured import (
    _decode_configured_task_program_integration,
)
from embodichain.lab.task_program.integrations._configured_composition import (
    _merge_runtime_services,
)
from embodichain.lab.task_program.integrations.simulation import environment as live


def _hand():
    return {
        "kind": "sampled_parallel_jaw",
        "motion_part": "lift_group",
        "hand_part": "custom_claw",
        "command_preset": "close_commands",
        "hand_root_link": "tool_base",
        "pad_link_names": ["pad_B", "pad_A"],
        "sample_count": 65,
        "body_max_edge": 0.01,
        "pad_max_edge": 0.004,
        "max_subdivisions": 10,
    }


def _payload():
    return {
        "scene": {"registry_id": "scene"},
        "robot_profile": {
            "profile_id": "robot",
            "resources": [
                {
                    "resource_id": "manipulator",
                    "endpoints": [
                        {
                            "endpoint_id": "motion",
                            "control_part": "lift_group",
                            "capabilities": ["motion.joint_position"],
                        },
                        {
                            "endpoint_id": "grasp",
                            "control_part": "custom_claw",
                            "capabilities": ["interaction.grasp"],
                            "command_preset": "close_commands",
                        },
                    ],
                }
            ],
            "command_presets": [
                {
                    "preset_id": "close_commands",
                    "control_part": "custom_claw",
                    "commands": {"open": [0.06, -0.06], "grasp": [0.01, -0.01]},
                }
            ],
            "defaults": {},
            "presets": [],
            "default_preset": None,
        },
        "runtime_services": {"hand_geometries": {"custom_claw": _hand()}},
    }


def test_declared_calibration_reaches_owned_engine_services_without_capture(
    monkeypatch,
):
    capture = Mock(
        side_effect=AssertionError("Configuration must not capture geometry.")
    )
    monkeypatch.setattr(RobotHandGeometryProvider, "capture", capture)
    received = []

    def adapter(environment, **kwargs):
        cfgs = kwargs["hand_geometry_cfgs"]
        engine = AtomicActionEngine(
            SimpleNamespace(robot=environment, device="cpu"),
            hand_geometry_cfgs=cfgs,
            load_builtins=False,
        )
        received.append((cfgs, engine))
        return SimpleNamespace(_set_configured_nominal_sources=lambda sources: None)

    monkeypatch.setattr(live, "create_simulation_task_program_adapter", adapter)
    payload = _payload()
    decoded = _decode_configured_task_program_integration(payload)
    payload["runtime_services"]["hand_geometries"]["custom_claw"][
        "pad_link_names"
    ].reverse()
    payload["robot_profile"]["command_presets"][0]["commands"]["grasp"][0] = 99
    decoded.adapter_factory.create_adapter(object())
    decoded.adapter_factory.create_adapter(object())
    first, second = (entry[1] for entry in received)
    cfg = first.planning_services.hand_geometry_cfg("custom_claw")
    assert cfg.pad_link_names == ("pad_B", "pad_A")
    assert cfg.hand_root_link == "tool_base" and cfg.motion_part == "lift_group"
    assert cfg.open_qpos == (0.06, -0.06) and cfg.closed_qpos == (0.01, -0.01)
    assert (
        cfg.sample_count,
        cfg.body_max_edge,
        cfg.pad_max_edge,
        cfg.max_subdivisions,
    ) == (65, 0.01, 0.004, 10)
    # Invocation-specific commands may replace this owned copy only.
    cfg.closed_qpos = (0.02, -0.02)
    received[0][0]["custom_claw"].pad_link_names = ("changed", "other")
    first.hand_geometry_cfgs["custom_claw"].sample_count = 2
    for engine in (first, second):
        preserved = engine.planning_services.hand_geometry_cfg("custom_claw")
        assert preserved.closed_qpos == (0.01, -0.01)
        assert preserved.pad_link_names == ("pad_B", "pad_A")
        assert preserved.sample_count == 65
        with pytest.raises(KeyError, match="No hand geometry"):
            engine.planning_services.hand_geometry_cfg("undeclared")
    capture.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "unknown_field",
        "missing_root",
        "unknown_kind",
        "repeated_pad",
        "wrong_preset",
        "missing_grasp",
        "wrong_target",
        "invalid_samples",
    ],
)
def test_closed_calibration_rejects_incomplete_or_ambiguous_declarations(change):
    payload = _payload()
    value = payload["runtime_services"]["hand_geometries"]["custom_claw"]
    if change == "unknown_field":
        value["model_id"] = "infer_geometry"
    elif change == "missing_root":
        value.pop("hand_root_link")
    elif change == "unknown_kind":
        value["kind"] = "import.arbitrary.Factory"
    elif change == "repeated_pad":
        value["pad_link_names"] = ["pad_A", "pad_A"]
    elif change == "wrong_preset":
        value["command_preset"] = "absent"
    elif change == "missing_grasp":
        payload["robot_profile"]["command_presets"][0]["commands"].pop("grasp")
    elif change == "wrong_target":
        payload["runtime_services"]["hand_geometries"] = {"other": value}
    elif change == "invalid_samples":
        value["sample_count"] = True
    with pytest.raises((ValueError, TypeError)):
        _decode_configured_task_program_integration(payload)


def test_calibration_has_one_component_owner_and_no_implicit_overrides():
    profile = {"hand_geometries": {"custom_claw": _hand()}}
    combined = _merge_runtime_services(profile, {})
    profile["hand_geometries"]["custom_claw"]["pad_link_names"].reverse()
    assert combined["hand_geometries"]["custom_claw"]["pad_link_names"] == [
        "pad_B",
        "pad_A",
    ]
    with pytest.raises(ValueError, match="exactly one owner"):
        _merge_runtime_services(profile, combined)


def test_calibration_changes_fingerprint_and_omission_preserves_legacy_payload():
    payload = _payload()
    first = _decode_configured_task_program_integration(payload)
    altered = deepcopy(payload)
    altered["runtime_services"]["hand_geometries"]["custom_claw"][
        "pad_link_names"
    ].reverse()
    assert (
        _decode_configured_task_program_integration(altered).integration_fingerprint
        != first.integration_fingerprint
    )
    payload.pop("runtime_services")
    omitted = _decode_configured_task_program_integration(payload)
    payload["runtime_services"] = {"hand_geometries": {}}
    empty = _decode_configured_task_program_integration(payload)
    assert omitted.integration_fingerprint == empty.integration_fingerprint
    # Captured from the unmodified 1e1cd2d6d configured decoder for this payload.
    assert (
        omitted.integration_fingerprint
        == "7134455eabc8ab42677da60fae4436399cd7be02415ef6ba6f11ffeee4b19b3a"
    )


def test_omitted_engine_calibration_does_not_load_geometry_provider(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "embodichain.lab.sim.utility.grasp_geometry":
            raise AssertionError(
                "Omitted hand configuration must not load its geometry module."
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    engine = AtomicActionEngine(
        SimpleNamespace(robot=object(), device="cpu"), load_builtins=False
    )
    assert dict(engine.hand_geometry_cfgs) == {}
