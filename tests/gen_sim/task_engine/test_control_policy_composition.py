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

"""Unrelated E recipes must not select a different execution authority."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from .test_press import button
from .test_press_integration import graph as press_graph
from .test_press_integration import scene as press_scene
from .test_support_bundle import _load_bundle, _placement_graph, support_scene
from .test_twist import graph as twist_graph
from .test_twist import knob_scene
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.lab.task_program.language import load_task_program
from embodichain.lab.task_program.semantics import EffectAssurance
from embodichain.utils.utility import load_config


@pytest.mark.parametrize("family", ["E8", "E9"])
@pytest.mark.parametrize("prepend", [False, True])
@pytest.mark.parametrize("same_resource", [False, True])
def test_control_authority_is_independent_of_unrelated_e1(
    support_scene,
    knob_scene,
    press_scene,
    tmp_path: Path,
    family: str,
    prepend: bool,
    same_resource: bool,
) -> None:
    control_scene = knob_scene if family == "E8" else press_scene
    articulations = deepcopy(control_scene.articulations)
    for articulation in articulations:
        articulation["init_pos"] = [-0.25, 0.25, 0.8]
        articulation.pop("init_local_pose", None)
    scene = replace(
        support_scene,
        articulations=articulations,
        planner_objects=(
            *support_scene.planner_objects,
            *(
                {**item, "runtime_uid": item["uid"], "role": "articulation"}
                for item in articulations
            ),
        ),
    )
    standalone = twist_graph(scene) if family == "E8" else press_graph(scene)
    composed = deepcopy(standalone)
    neighbor = _placement_graph(reference="tray")
    if same_resource:
        for node in neighbor["nodes"]:
            node["call"]["resources"] = {
                key: "right" for key in node["call"]["resources"]
            }
    for field in ("nodes", "task_groups"):
        composed[field] = (
            (neighbor[field] + composed[field])
            if prepend
            else (composed[field] + neighbor[field])
        )
    _, first = generate_task_program_bundle(
        standalone, scene, tmp_path / "standalone", robot_profile="dual_franka"
    )
    _, second = generate_task_program_bundle(
        composed, scene, tmp_path / "composed", robot_profile="dual_franka"
    )
    assert load_config(first.execution_policy) == load_config(second.execution_policy)
    assert load_config(first.embodiment) == load_config(second.embodiment)
    original, mixed = _load_bundle(first), _load_bundle(second)
    first_preset = original.integration.registration.robot_profile_binding.presets[0]
    second_preset = mixed.integration.registration.robot_profile_binding.presets[0]
    for field in (
        "preset_id",
        "required_planner",
        "effect_assurance",
        "motion_policy",
        "tracking_policy",
        "recovery_policy",
        "workflow_recovery_policy",
        "runner_cfg",
    ):
        assert getattr(first_preset, field) == getattr(second_preset, field)
    for node in standalone["nodes"]:
        semantic_id = node["call"]["call_id"]
        assert first_preset.effect_monitors.get(semantic_id) is None
        assert second_preset.effect_monitors.get(semantic_id) is None
        first_options = first_preset.action_option_template(semantic_id)
        second_options = second_preset.action_option_template(semantic_id)
        assert type(first_options) is type(second_options)
        assert asdict(first_options) == asdict(second_options)
    assert first_preset.effect_assurance is EffectAssurance.VERIFIED
    before = load_config(first.program)["program"]["items"]
    after = load_config(second.program)["program"]["items"]
    control_names = {node["id"] for node in standalone["nodes"]}
    assert [item for item in before if item["name"] in control_names] == [
        item for item in after if item["name"] in control_names
    ]
    assert any(item.get("post") for item in before)
    for deployment, paths in ((original, first), (mixed, second)):
        program = load_task_program(
            paths.program,
            integration=deployment.selection,
            validation_context=deployment.integration.registration.catalog,
        )
        deployment.integration.registration.catalog.preflight(program)
