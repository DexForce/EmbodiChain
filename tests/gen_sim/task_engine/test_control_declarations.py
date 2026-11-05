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
from dataclasses import replace
import math
import pytest
from embodichain.gen_sim.task_engine import control_declarations as owner
from embodichain.lab.sim.scene.press_inputs import PressBinding
from embodichain.lab.sim.scene.revolute_inputs import RevoluteControlBinding
from embodichain_robots.robotiq.task_program import Robotiq140TaskHandCalibration


def _hand():
    return Robotiq140TaskHandCalibration(
        "custom_resource",
        "right",
        "right_eef",
        "right_arm",
        "right_eef",
        "command",
        "right_robotiq_arg2f_base_link",
        ("pad_a", "pad_b"),
        ("finger_a", "finger_b"),
        {},
    )


def _graph(family, count=1):
    args = {"object": "control"}
    if family == "twist":
        args["setting"] = 90
    calls = [
        dict(
            kind="registered",
            call_id=identity,
            arguments=dict(args),
            resources={"primary": "custom_resource"},
        )
        for identity in [
            f"gen_sim.{family}_prepare.operation",
            *([f"gen_sim.{family}.operation"] * count),
        ]
    ] + [
        dict(
            kind="registered",
            call_id="gen_sim.articulation_park",
            arguments={},
            resources={"primary": "custom_resource"},
        )
    ]
    nodes = [
        dict(
            id=f"node{index}",
            role="cleanup" if index == len(calls) - 1 else "primary",
            depends_on=[] if not index else [f"node{index-1}"],
            call=call,
        )
        for index, call in enumerate(calls)
    ]
    return dict(
        nodes=nodes,
        task_groups=[
            dict(
                id="operation",
                task_type="E9" if family == "press" else "E8",
                node_ids=[node["id"] for node in nodes],
            )
        ],
    )


def test_press_declarations_bind_source_mass_and_distinct_release_terminal_windows(
    monkeypatch,
):
    source = PressBinding(
        "control",
        "joint",
        "cap",
        "base",
        "source.usda",
        "a" * 64,
        "/Button/cap/cap_contact",
        1.0,
        (0.0, 0.0, -1.0),
        (-0.005, 0.0),
        (0.0, 0.0, 0.02),
        0.0,
        -0.005,
    )
    monkeypatch.setattr(owner, "discover_press_source", lambda config: (source, 0.02))
    graph = _graph("press")
    result = owner.resolve_control_program_declarations(
        graph,
        SimpleNamespace(articulations=({"uid": "control"},)),
        robot_uid="native_robot",
        hands=(_hand(),),
        table_uid="native_table",
    )
    operation = result.runtime_declarations()[0]
    assert (
        operation["moving_mass"] == 0.02
        and operation["binding"]["source_sha256"] == "a" * 64
    )
    assert "source_path" not in operation["binding"]
    assert (
        operation["entity_id"] == "control__cap"
        and operation["grasp_target_id"] == "right_eef"
    )
    assert result.action_options["gen_sim.press.operation"]["press_distance"] == 0.015
    assert [cfg.phase for cfg in result.conditions.values()] == [
        "ready",
        "released",
        "released",
    ]
    assert [cfg.terminal for cfg in result.conditions.values()] == [False, False, True]
    bindings = result.operation_groups["operation"]
    assert (
        bindings[-2].post[0]["preset"]
        == bindings[-1].post[0]["preset"]
        == "gen_sim.press.operation.pressed"
    )
    assert result.post_policy_presets[bindings[-2].node_id] == (
        "gen_sim.press.operation.pressed",
    )
    assert result.post_policy_presets[bindings[-1].node_id] == (
        "gen_sim.press.operation.terminal",
    )

    assert result.sensors[0]["filter_need_both_actor"] is True
    assert result.sensors[0]["articulation_cfg_list"] == [
        dict(articulation_uid="native_robot", link_name_list=[]),
        dict(articulation_uid="control", link_name_list=[]),
    ]
    assert all(
        binding.post[0]["entity"] == "control"
        for binding in result.operation_groups["operation"]
    )


def test_twist_declarations_preserve_final_release_without_adding_a_chunk_wait(
    monkeypatch,
):
    source = RevoluteControlBinding(
        "control",
        "joint",
        "knob",
        "base",
        "a" * 64,
        1.0,
        (0.0, 0.0, -1.0),
        -1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.03),
        0.02,
        0.07,
        (-math.pi / 2, 0.0),
        90,
        -math.pi / 2,
        "b" * 64,
        "",
    )
    monkeypatch.setattr(owner, "inspect_revolute_control", lambda *args: source)
    graph = _graph("twist", 32)
    result = owner.resolve_control_program_declarations(
        graph,
        SimpleNamespace(articulations=({"uid": "control"},)),
        robot_uid="native_robot",
        hands=(_hand(),),
        table_uid="native_table",
        twist_chunk_angles={"operation": math.radians(20.0)},
    )
    assert result.operations[0].chunk_count == 32
    assert result.action_options["gen_sim.twist.operation"][
        "max_twist_angle"
    ] == math.radians(20.0)
    bindings = result.operation_groups["operation"]
    assert len(bindings) == 34 and all(len(binding.post) == 1 for binding in bindings)
    phases = [
        result.conditions[binding.post[0]["preset"]].phase for binding in bindings
    ]
    assert phases == ["ready"] + ["chunk"] * 31 + ["released", "released"]
    assert result.sensors[0]["filter_need_both_actor"] is False
    assert result.operations[0].call.entity_id == "control::knob"
    assert all(value.post[0]["entity"] == "control" for value in bindings)
    assert (
        bindings[-2].post[0]["preset"]
        == bindings[-1].post[0]["preset"]
        == "gen_sim.twist.operation.turned"
    )
    assert result.post_policy_presets[bindings[-2].node_id] == (
        "gen_sim.twist.operation.turned",
    )
    assert result.post_policy_presets[bindings[-1].node_id] == (
        "gen_sim.twist.operation.terminal",
    )

    graph["nodes"].pop(-2)
    graph["task_groups"][0]["node_ids"].pop(-2)
    with pytest.raises((ValueError, KeyError)):
        owner.resolve_control_program_declarations(
            graph,
            SimpleNamespace(articulations=({"uid": "control"},)),
            robot_uid="native_robot",
            hands=(_hand(),),
            table_uid="native_table",
        )


def test_historical_hand_service_addition_preserves_physical_and_control_inputs():
    from pathlib import Path
    from copy import deepcopy
    import hashlib
    import yaml
    from embodichain.gen_sim.task_engine.embodiment_declarations import (
        resolve_source_hand_services,
    )
    from embodichain.toolkits.graspkit.pg_grasp.model_geometry import (
        get_parallel_jaw_model_asset_files,
    )
    from embodichain.utils.config_paths import resolve_config_path

    component = Path(
        resolve_config_path(
            "extensions/embodichain_tasks/configs/components/embodiments/dual_franka_robotiq_arg2f_140.yaml"
        )
    )
    historical = yaml.safe_load(component.read_text())
    expected = historical["skill_profile"]["runtime_services"].pop("hand_geometries")
    before = deepcopy(historical)
    resolved, hands, inputs = resolve_source_hand_services(historical)
    assert historical == before
    assert resolved["simulation"] == before["simulation"]
    assert resolved["sensor"] == before["sensor"]
    assert (
        resolved["skill_profile"]["command_presets"]
        == before["skill_profile"]["command_presets"]
    )
    assert resolved["skill_profile"]["runtime_services"]["hand_geometries"] == expected
    model_inputs = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in get_parallel_jaw_model_asset_files("robotiq_arg2f_140")
    }
    assert len(hands) == 2
    assert inputs == {
        str(component.resolve()): hashlib.sha256(component.read_bytes()).hexdigest(),
        **model_inputs,
    }
    second, _, additional = resolve_source_hand_services(resolved)
    assert second == resolved and additional == model_inputs


@pytest.mark.parametrize(
    "identity",
    [
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
    ],
)
def test_source_native_grip_scope_and_experimental_driver_identity(identity):
    from pathlib import Path
    import yaml
    from embodichain.gen_sim.task_engine.embodiment_declarations import (
        resolve_source_hand_services,
        declare_source_grip_evidence,
    )

    path = (
        Path(__file__).parents[3]
        / "extensions/embodichain_tasks/configs/components/embodiments"
        / (identity + ".yaml")
    )
    embodiment = yaml.safe_load(path.read_text())
    owned, hands, _ = resolve_source_hand_services(embodiment)
    source = owned["skill_profile"]["runtime_services"]["control_part_evidence"]
    binding = {
        "rigid_objects": [
            {"entity_id": "table", "simulation_uid": "table_native", "affordances": []},
            {
                "entity_id": "payload",
                "simulation_uid": "payload_native",
                "affordances": [{"kind": "antipodal_grasp"}],
            },
        ]
    }
    declaration = declare_source_grip_evidence(
        owned, hands, binding, experimental_open_release=True
    )
    assert declaration["object_bindings"] == [["payload", "payload_native"]]
    assert declaration["algorithm_revision"] == 2
    for opening, hand in zip(declaration["opening_bindings"], hands):
        assert opening["open_qpos"] == source["open_qpos"]
        assert opening["tolerance"] == source["minimum_displacement"]
        (index,) = opening["driver_indices"]
        assert opening["joint_names"][index] == hand.side + "_finger_joint"
    assert "control_part_evidence" not in owned["skill_profile"]["runtime_services"]
