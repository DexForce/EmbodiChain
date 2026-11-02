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

from types import SimpleNamespace as NS

import pytest

from embodichain.gen_sim.task_engine.task_program_bundle import _integration_payload
from embodichain.lab.task_program.compiler.lowering import SemanticCallCompiler
from embodichain.lab.task_program.semantics.effects import (
    CompositeEffectMonitorFactory,
    EffectMonitorRegistry,
)
from embodichain.lab.task_program.integrations._configured_composition import (
    _compose_integration_payload,
)
from embodichain.lab.task_program.integrations.configured import _decode_robot_profile
from embodichain.lab.task_program.semantics import EffectAssurance, Pick, SceneObjectRef
from embodichain.utils.utility import load_config
from .test_semantic_graph import _prepared_axis_scene


@pytest.mark.parametrize("with_place", [False, True])
def test_alias_acquisition_place_recovery_has_real_verified_pick_monitor(
    tmp_path, with_place
):
    scene = _prepared_axis_scene(tmp_path)
    graph = {
        "targets": {
            "target": {
                "kind": "pose",
                "values": [
                    {
                        "position": [0.1, -0.2, 0.85],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        },
        "nodes": [
            {
                "id": "acquire",
                "task_instance_id": "group",
                "task_type": "E2",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.pick",
                    "arguments": {"object": "bottle", "target": "target"},
                    "resources": {"primary": "left"},
                },
            }
        ],
    }
    if with_place:
        graph["nodes"].append(
            {
                "id": "release",
                "task_instance_id": "group",
                "task_type": "E2",
                "call": {
                    "kind": "place",
                    "object": "bottle",
                    "at": {"kind": "target", "target": "target"},
                    "resources": {"primary": "left"},
                },
            }
        )
    embodiment = load_config(
        "extensions/embodichain_tasks/configs/components/embodiments/dual_franka_robotiq_arg2f_140.yaml"
    )
    task = _integration_payload(
        graph,
        scene,
        program_id="program",
        scene_contract="scene",
        embodiment=embodiment,
    )
    monitors = task["profile"]["effect_monitors"]
    assert ("pick" in monitors) is with_place
    assert "simulation.pick" in monitors
    if with_place:
        policy = load_config(
            "extensions/embodichain_tasks/configs/components/execution_policies/dual_arm_trajectory_verified.yaml"
        )
        composed = _compose_integration_payload(
            task=task,
            policy=policy,
            skill_profile=embodiment["skill_profile"],
            scene=task["scene_binding"],
        )
        profile = _decode_robot_profile(composed["robot_profile"])
        preset = profile.presets[0]
        assert preset.effect_assurance is EffectAssurance.VERIFIED
        bound = NS(linked=NS(call=Pick(object=SceneObjectRef("bottle"))), preset=preset)
        compiler = object.__new__(SemanticCallCompiler)
        compiler._effect_monitor_registry = EffectMonitorRegistry(
            (CompositeEffectMonitorFactory(),)
        )
        monitor = compiler._effect_monitor_ref(bound, path=())
        assert monitor == preset.effect_monitors["pick"]
        assert monitor.monitor_id == "builtin.composite_effect"
